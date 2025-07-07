import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def get_muon_optimizer(config, model):
    """
    Create Muon optimizer with proper parameter grouping.
    Muon is used for hidden weights (2D parameters), AdamW for everything else.
    """
    try:
        from muon import MuonWithAuxAdam
    except ImportError:
        raise ImportError("Muon optimizer not found. Please install it with: pip install muon")
    
    # Categorize parameters based on model structure
    hidden_weights = []
    hidden_gains_biases = []
    embed_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'vocab_embed' in name:
            embed_params.append(param)
        elif 'output_layer' in name:
            head_params.append(param)
        elif 'blocks' in name and param.ndim >= 2:
            hidden_weights.append(param)
        else:
            hidden_gains_biases.append(param)
    
    # Create parameter groups according to Muon requirements
    param_groups = [
        dict(params=hidden_weights, 
             lr=config.optim.muon_lr, 
             momentum=config.optim.muon_momentum,
             weight_decay=config.optim.weight_decay,
             use_muon=True),
        dict(params=hidden_gains_biases + embed_params + head_params, 
             lr=config.optim.adamw_lr, 
             betas=(config.optim.beta1, config.optim.beta2),
             eps=config.optim.eps,
             weight_decay=config.optim.weight_decay,
             use_muon=False),
    ]
    
    optimizer = MuonWithAuxAdam(param_groups)
    return optimizer


def get_lr_schedule(step, config):
    """
    Learning rate schedule with warmup, constant phase, and cosine decay.
    
    Args:
        step: Current training step
        config: Configuration object containing training parameters
        
    Returns:
        Learning rate multiplier
    """
    total_steps = config.training.n_iters
    warmup_steps = config.optim.warmup
    
    # Calculate phase boundaries
    decay_start_step = int(total_steps * 0.8)  # Last 20% for cosine decay
    
    if step <= warmup_steps:
        # Linear warmup phase
        return step / warmup_steps
    elif step <= decay_start_step:
        # Constant learning rate phase
        return 1.0
    else:
        # Cosine decay phase (last 20% of training)
        decay_steps = total_steps - decay_start_step
        current_decay_step = step - decay_start_step
        
        # Cosine decay from 1.0 to 0.1 (one order of magnitude)
        decay_factor = 0.5 * (1 + np.cos(np.pi * current_decay_step / decay_steps))
        return 0.1 + 0.9 * decay_factor


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with advanced learning rate schedule and gradient clipping."""
        scaler.unscale_(optimizer)

        # Apply advanced learning rate schedule
        lr_multiplier = get_lr_schedule(step, config)
        
        if config.optim.use_muon:
            # Handle Muon optimizer with advanced schedule
            for g in optimizer.param_groups:
                if g.get('use_muon', False):
                    g['lr'] = config.optim.muon_lr * lr_multiplier
                else:
                    g['lr'] = config.optim.adamw_lr * lr_multiplier
        else:
            # Standard optimizer with advanced schedule
            for g in optimizer.param_groups:
                g['lr'] = lr * lr_multiplier
        
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def compute_grad_norm(model):
    """Compute gradient norm for model parameters"""
    total_norm = 0.0
    param_count = 0
    
    # Handle both DDP and non-DDP models
    if hasattr(model, 'module'):
        parameters = model.module.parameters()
    else:
        parameters = model.parameters()
    
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
    
    return total_norm


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']
        grad_norm = 0.0

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                # Compute gradient norm BEFORE optimizer step
                grad_norm = compute_grad_norm(model)
                
                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss, grad_norm

    return step_fn