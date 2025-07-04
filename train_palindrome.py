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
from palindrome_json_loader import get_palindrome_dataloaders
from logger_utils import setup_logger


def save_checkpoint_single_gpu(ckpt_dir, state):
    """Save checkpoint handling both DDP and non-DDP models"""
    model = state['model']
    # Handle both DDP and non-DDP models
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': model_state_dict,
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def restore_checkpoint_single_gpu(ckpt_dir, state, device):
    """Restore checkpoint handling both DDP and non-DDP models with graceful optimizer loading"""
    if not os.path.exists(ckpt_dir):
        utils.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        
        # Try to load optimizer state, but gracefully handle incompatibilities
        try:
            state['optimizer'].load_state_dict(loaded_state['optimizer'])
            logging.info("Successfully loaded optimizer state from checkpoint")
        except Exception as e:
            logging.warning(f"Failed to load optimizer state from checkpoint: {e}")
            logging.info("Continuing with fresh optimizer state (common for fine-tuning)")
        
        # Handle both DDP and non-DDP models
        model = state['model']
        if hasattr(model, 'module'):
            model.module.load_state_dict(loaded_state['model'], strict=False)
        else:
            model.load_state_dict(loaded_state['model'], strict=False)
        logging.info("Successfully loaded model weights from checkpoint")
            
        # Try to load EMA state
        try:
            state['ema'].load_state_dict(loaded_state['ema'])
            logging.info("Successfully loaded EMA state from checkpoint")
        except Exception as e:
            logging.warning(f"Failed to load EMA state from checkpoint: {e}")
            logging.info("Continuing with fresh EMA state")
            
        # Load step count
        state['step'] = loaded_state.get('step', 0)
        logging.info(f"Loaded checkpoint from step: {state['step']}")
        
        return state


def setup_palindrome_model(config):
    """Setup model with byte-level vocabulary"""
    # Update config for byte-level processing
    config.tokens = 259  # 256 bytes + PAD + BOS + EOS
    
    model = SEDD(config)
    return model


def setup_distributed(rank, world_size, port):
    """Setup distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def generate_sample_palindromes(model, graph, noise, step, device, logger, training_logger=None):
    """Generate sample palindromes during training"""
    try:
        from palindrome_sampling import sample_palindromes
        
        model.eval()
        with torch.no_grad():
            results = sample_palindromes(
                model, graph, noise, 
                num_samples=10,  # More samples for better metrics
                max_length=64, 
                steps=32, 
                device=device
            )
        
        if logger:
            logger.info(f"Step {step} - Sample palindromes:")
            for i, result in enumerate(results[:5]):  # Show first 5
                palindrome_status = "✓" if result['is_palindrome'] else "✗"
                logger.info(f"  {i+1}: {palindrome_status} '{result['text']}'")
        
        # Log palindrome metrics to wandb
        if training_logger:
            training_logger.log_palindrome_metrics(results, step)
        
        model.train()
        return results
        
    except Exception as e:
        if logger:
            logger.warning(f"Could not generate samples: {e}")
        return []


def run_multiprocess(rank, world_size, cfg, port):
    """Run training with multiprocessing"""
    try:
        setup_distributed(rank, world_size, port) 
        _run_training(rank, world_size, cfg)
    finally:
        cleanup_distributed()


def _run_training(rank, world_size, cfg):
    """Main training function"""
    torch.cuda.set_device(rank)
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
    mprint(f"Palindrome training: Rank {rank}/{world_size}")
    
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
    model = setup_palindrome_model(cfg).to(device)
    
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank], static_graph=True, find_unused_parameters=True)
    
    num_parameters = sum(p.numel() for p in model.parameters())
    mprint(f"Number of parameters: {num_parameters}")
    
    # Setup EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    mprint(f"EMA: {ema}")
    
    # Watch model for wandb
    if training_logger:
        training_logger.watch_model(model)
    
    # Setup optimizer
    optimizer = losses.get_optimizer(cfg, chain(model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    
    scaler = torch.cuda.amp.GradScaler()
    
    # State dictionary
    state = dict(
        optimizer=optimizer, 
        scaler=scaler, 
        model=model, 
        noise=noise, 
        ema=ema, 
        step=0
    )
    
    # Load checkpoint if exists and not disabled
    if not getattr(cfg.training, 'disable_checkpoint_loading', False):
        state = restore_checkpoint_single_gpu(checkpoint_meta_dir, state, device)
        mprint(f"Checkpoint loading enabled. Restored from step: {state['step']}")
    else:
        mprint("Checkpoint loading disabled. Starting from scratch.")
    
    # Load pre-training checkpoint if specified
    if hasattr(cfg.training, 'pretrain_checkpoint') and cfg.training.pretrain_checkpoint:
        mprint(f"Loading pre-training checkpoint: {cfg.training.pretrain_checkpoint}")
        try:
            pretrain_state = torch.load(cfg.training.pretrain_checkpoint, map_location=device)
            # Load only the model weights, not optimizer/step
            model_to_load = state['model']
            if hasattr(model_to_load, 'module'):
                model_to_load.module.load_state_dict(pretrain_state['model'], strict=False)
            else:
                model_to_load.load_state_dict(pretrain_state['model'], strict=False)
            mprint("Pre-training checkpoint loaded successfully")
        except Exception as e:
            mprint(f"Failed to load pre-training checkpoint: {e}")
    
    initial_step = int(state['step'])
    
    # Get data loaders
    train_ds, eval_ds = get_palindrome_dataloaders(cfg, distributed=(world_size > 1))
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)
    
    # Build step functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)
    
    # Setup sampling if needed
    if cfg.training.snapshot_sampling:
        import sampling
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, 1e-5, device)
    
    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}")
    
    # Training loop
    while state['step'] < num_train_steps + 1:
        step = state['step']
        
        # Get training batch
        batch = next(train_iter)['input_ids'].to(device)
        loss = train_step_fn(state, batch)
        
        # Check if step was incremented (full batch computed)
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                if world_size > 1:
                    dist.all_reduce(loss)
                    loss /= world_size
                
                # Log to console
                mprint(f"step: {step}, training_loss: {loss.item():.5e}")
                
                # Log comprehensive metrics
                if training_logger:
                    train_metrics = training_logger.compute_training_metrics(model, optimizer, loss, cfg)
                    training_logger.log_metrics(train_metrics, step)
            
            # Save checkpoint for preemption
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                save_checkpoint_single_gpu(checkpoint_meta_dir, state)
            
            # Evaluation
            if step % cfg.training.eval_freq == 0:
                eval_batch = next(eval_iter)['input_ids'].to(device)
                eval_loss = eval_step_fn(state, eval_batch)
                
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
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating palindromes at step: {step}")
                    
                    this_sample_dir = os.path.join(sample_dir, f"iter_{step}")
                    utils.makedirs(this_sample_dir)
                    
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    
                    # Generate palindromes
                    generate_sample_palindromes(
                        model, graph, noise, step, device, 
                        logger if rank == 0 else None,
                        training_logger if rank == 0 else None
                    )
                    
                    ema.restore(model.parameters())
                
                if world_size > 1:
                    dist.barrier()


@hydra.main(version_base=None, config_path="configs", config_name="palindrome_byte")
def main(config: DictConfig) -> None:
    """Main training function for palindrome model - Single GPU only"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Setup work directory
    work_dir = os.getcwd()
    utils.makedirs(work_dir)
    
    with open_dict(config):
        config.work_dir = work_dir
        config.wandb_name = os.path.basename(os.path.normpath(work_dir))
        config.ngpus = 1  # Force single GPU
    
    # Single GPU training only
    _run_training(0, 1, config)


if __name__ == "__main__":
    main()