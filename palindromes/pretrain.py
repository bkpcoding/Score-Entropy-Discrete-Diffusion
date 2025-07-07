import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf, open_dict
import logging
import os
from pathlib import Path
from itertools import chain

# Import existing modules  
from model.transformer import SEDD
from model.ema import ExponentialMovingAverage
import graph_lib
import noise_lib
import losses
import utils
from palindromes.byte_data import get_byte_wikipedia_dataloaders, ByteProcessor
from logger_utils import setup_logger


def setup_byte_model(config):
    """Setup model with byte-level vocabulary"""
    # Update config for byte-level processing
    config.tokens = 259  # 256 bytes + PAD + BOS + EOS
    
    model = SEDD(config)
    return model


def save_checkpoint_single_gpu(ckpt_dir, state):
    """Save checkpoint handling both DDP and non-DDP models"""
    model = state['model']
    ema = state['ema']
    
    # Handle both DDP and non-DDP models
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    # Always save EMA weights as a separate model state dict
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    
    if hasattr(model, 'module'):
        ema_model_state_dict = model.module.state_dict()
    else:
        ema_model_state_dict = model.state_dict()
    
    # Restore original weights
    ema.restore(model.parameters())
    
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': model_state_dict,  # Raw model weights
        'model_ema': ema_model_state_dict,  # EMA model weights
        'ema': state['ema'].state_dict(),  # EMA state for continuation
        'step': state['step'],
        'config': state.get('config', None)
    }
    
    torch.save(saved_state, ckpt_dir)


def restore_checkpoint_single_gpu(ckpt_dir, state, device, load_ema_as_model=False):
    """Restore checkpoint handling both DDP and non-DDP models"""
    if not os.path.exists(ckpt_dir):
        utils.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        
        # Handle both DDP and non-DDP models
        model = state['model']
        
        # Choose which model weights to load
        if load_ema_as_model and 'model_ema' in loaded_state:
            # Load EMA weights as the main model
            model_weights = loaded_state['model_ema']
        else:
            # Load regular model weights
            model_weights = loaded_state['model']
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_weights, strict=False)
        else:
            model.load_state_dict(model_weights, strict=False)
            
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def generate_sample_text(model, graph, noise, step, device, logger, processor):
    """Generate sample text during pre-training"""
    try:
        import sampling
        
        model.eval()
        with torch.no_grad():
            # Create simple sampling function
            sampling_shape = (3, 64)  # 3 samples, length 64
            sampling_fn = sampling.get_pc_sampler(
                graph=graph,
                noise=noise,
                batch_dims=sampling_shape,
                predictor='analytic',
                steps=32,
                device=device
            )
            
            samples = sampling_fn(model)
            
            logger.info(f"Step {step} - Sample texts:")
            for i, sample in enumerate(samples):
                text = processor.decode_text(sample.cpu().numpy())
                logger.info(f"  {i+1}: '{text[:100]}...'")
        
        model.train()
        
    except Exception as e:
        logger.warning(f"Could not generate samples: {e}")


def setup_distributed(rank, world_size, port):
    """Setup distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    """Run training with multiprocessing"""
    try:
        setup_distributed(rank, world_size, port)
        _run_pretraining(rank, world_size, cfg)
    finally:
        cleanup_distributed()


def _run_pretraining(rank, world_size, cfg):
    """Main pre-training function"""
    if world_size > 1:
        torch.cuda.set_device(rank)
    
    # Initialize distributed backend for Muon optimizer even in single-GPU mode
    if cfg.optim.use_muon and world_size == 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        dist.init_process_group("nccl", rank=0, world_size=1)
    
    work_dir = cfg.work_dir
    
    # Create directories
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))
    
    # Setup logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
        training_logger = setup_logger(cfg, work_dir, rank)
    else:
        logger = None
        training_logger = None
    
    def mprint(msg):
        if rank == 0:
            logger.info(msg)
    
    mprint(f"Working directory: {work_dir}")
    mprint(f"Pre-training byte-level model on Wikipedia")
    mprint(f"Distributed training: Rank {rank}/{world_size}")
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    mprint(f"Using device: {device}")
    
    if world_size > 1:
        mprint(f"Multi-GPU training enabled with {world_size} GPUs")
    else:
        mprint(f"Single GPU training")
    
    # Build graph and noise
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        noise = DDP(noise, device_ids=[rank], static_graph=True)
    
    # Build model
    model = setup_byte_model(cfg).to(device)
    
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank], static_graph=True, find_unused_parameters=True)
    
    num_parameters = sum(p.numel() for p in model.parameters())
    mprint(f"Number of parameters: {num_parameters}")
    
    # Setup EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    
    # Watch model for wandb
    if training_logger:
        training_logger.watch_model(model)
    
    # Setup optimizer
    if cfg.optim.use_muon:
        # Use Muon optimizer for model parameters, standard optimizer for noise
        model_optimizer = losses.get_muon_optimizer(cfg, model)
        noise_optimizer = losses.get_optimizer(cfg, noise.parameters())
        
        # Create a combined optimizer wrapper
        class CombinedOptimizer:
            def __init__(self, model_opt, noise_opt):
                self.model_opt = model_opt
                self.noise_opt = noise_opt
                self.param_groups = model_opt.param_groups + noise_opt.param_groups
            
            def zero_grad(self):
                self.model_opt.zero_grad()
                self.noise_opt.zero_grad()
            
            def step(self):
                self.model_opt.step()
                self.noise_opt.step()
            
            def state_dict(self):
                return {
                    'model_optimizer': self.model_opt.state_dict(),
                    'noise_optimizer': self.noise_opt.state_dict()
                }
            
            def load_state_dict(self, state_dict):
                self.model_opt.load_state_dict(state_dict['model_optimizer'])
                self.noise_opt.load_state_dict(state_dict['noise_optimizer'])
        
        optimizer = CombinedOptimizer(model_optimizer, noise_optimizer)
        mprint(f"Using Muon optimizer for model parameters")
    else:
        optimizer = losses.get_optimizer(cfg, chain(model.parameters(), noise.parameters()))
        mprint(f"Using standard {cfg.optim.optimizer} optimizer")
    
    scaler = torch.cuda.amp.GradScaler()
    
    # State dictionary
    state = dict(
        optimizer=optimizer, 
        scaler=scaler, 
        model=model, 
        noise=noise, 
        ema=ema, 
        step=0,
        config=cfg
    )
    
    # Load checkpoint if exists and not disabled
    if not getattr(cfg.training, 'disable_checkpoint_loading', False):
        state = restore_checkpoint_single_gpu(checkpoint_meta_dir, state, device)
        mprint(f"Checkpoint loading enabled. Restored from step: {state['step']}")
    else:
        mprint("Checkpoint loading disabled. Starting from scratch.")
    
    initial_step = int(state['step'])
    
    # Get data loaders - now memory efficient with streaming
    mprint("Loading Wikipedia dataset (this may take a few minutes)...")
    train_ds, eval_ds = get_byte_wikipedia_dataloaders(cfg, distributed=(world_size > 1), max_samples=cfg.training.get('max_samples', 500000))
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)
    mprint(f"Dataset loaded successfully")
    
    # Build step functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)
    
    # Create processor for text generation
    processor = ByteProcessor(max_length=cfg.model.length)
    
    num_train_steps = cfg.training.n_iters
    mprint(f"Starting pre-training loop at step {initial_step}")
    
    # Training loop
    while state['step'] < num_train_steps + 1:
        step = state['step']
        
        try:
            # Get training batch
            batch = next(train_iter)['input_ids'].to(device)
            loss, grad_norm = train_step_fn(state, batch)
            
            # Check if step was incremented (full batch computed)
            if step != state['step']:
                if step % cfg.training.log_freq == 0:
                    if world_size > 1:
                        dist.all_reduce(loss)
                        loss /= world_size
                    
                    # Log to console
                    mprint(f"step: {step}, training_loss: {loss.item():.5e}, grad_norm: {grad_norm:.5e}")
                    
                    # Log comprehensive metrics
                    if training_logger:
                        train_metrics = training_logger.compute_training_metrics(model, optimizer, loss, cfg, grad_norm=grad_norm)
                        training_logger.log_metrics(train_metrics, step)
                
                # Save checkpoint for preemption
                if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                    save_checkpoint_single_gpu(checkpoint_meta_dir, state)
                
                # Evaluation
                if step % cfg.training.eval_freq == 0:
                    eval_batch = next(eval_iter)['input_ids'].to(device)
                    eval_loss, _ = eval_step_fn(state, eval_batch)  # Ignore grad_norm for eval
                    
                    if world_size > 1:
                        dist.all_reduce(eval_loss)
                        eval_loss /= world_size
                    
                    # Log to console
                    mprint(f"step: {step}, evaluation_loss: {eval_loss.item():.5e}")
                    
                    # Log evaluation metrics
                    if training_logger:
                        eval_metrics = training_logger.compute_eval_metrics(eval_loss)
                        training_logger.log_metrics(eval_metrics, step)
                
                # Save checkpoints and generate samples
                if (step > 0 and step % cfg.training.snapshot_freq == 0) or step == num_train_steps:
                    save_step = step // cfg.training.snapshot_freq
                    if rank == 0:
                        save_checkpoint_single_gpu(
                            os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), 
                            state
                        )
                    
                    # Generate samples
                    if cfg.training.snapshot_sampling and rank == 0:
                        mprint(f"Generating text samples at step: {step}")
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        generate_sample_text(model, graph, noise, step, device, logger, processor)
                        ema.restore(model.parameters())
                    
                    if world_size > 1:
                        dist.barrier()
        
        except Exception as e:
            mprint(f"Error at step {step}: {e}")
            # Save emergency checkpoint
            if rank == 0:
                save_checkpoint_single_gpu(
                    os.path.join(checkpoint_dir, f'emergency_checkpoint_{step}.pth'), 
                    state
                )
            break
    
    # Cleanup logging
    if training_logger:
        training_logger.finish()
    
    # Cleanup distributed if initialized for Muon
    if cfg.optim.use_muon and world_size == 1:
        cleanup_distributed()
    
    mprint("Pre-training completed!")


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain_byte")
def main(config: DictConfig) -> None:
    """Main pre-training function"""
    logging.basicConfig(level=logging.INFO)
    
    # Setup work directory
    work_dir = os.getcwd()
    utils.makedirs(work_dir)
    
    # Generate unique wandb name based on config settings
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = getattr(config.logging, 'wandb_name', 'pretrain_byte')
    
    # Create informative run name
    run_name = f"{base_name}_{timestamp}"
    if hasattr(config.model, 'hidden_size'):
        run_name += f"_h{config.model.hidden_size}"
    if hasattr(config.training, 'batch_size'):
        run_name += f"_bs{config.training.batch_size}"
    if hasattr(config.optim, 'lr'):
        run_name += f"_lr{config.optim.lr}"
    if hasattr(config.optim, 'use_muon') and config.optim.use_muon:
        run_name += "_muon"
    
    with open_dict(config):
        config.work_dir = work_dir
        config.wandb_name = run_name
    # Setup multiprocessing for multi-GPU training
    if config.ngpus > 1:
        import torch.multiprocessing as mp
        import numpy as np
        
        port = int(np.random.randint(10000, 20000))
        try:
            mp.set_start_method("forkserver")
            mp.spawn(run_multiprocess, args=(config.ngpus, config, port), nprocs=config.ngpus, join=True)
        except Exception as e:
            logging.critical(e, exc_info=True)
    else:
        # Single GPU training
        _run_pretraining(0, 1, config)


if __name__ == "__main__":
    main()