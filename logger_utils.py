import os
import torch
import logging
from typing import Dict, Any, Optional
import json
from pathlib import Path


class TrainingLogger:
    """Unified logger for training metrics with optional wandb support"""
    
    def __init__(self, config, work_dir: str, rank: int = 0):
        self.config = config
        self.work_dir = work_dir
        self.rank = rank
        self.is_main_process = rank == 0
        
        # Initialize wandb if enabled
        self.use_wandb = getattr(config.logging, 'use_wandb', False)
        self.wandb_run = None
        
        if self.use_wandb and self.is_main_process:
            self._setup_wandb()
        
        # Setup local logging
        self.metrics_file = None
        if self.is_main_process:
            self._setup_local_logging()
    
    def _setup_wandb(self):
        """Initialize wandb logging"""
        try:
            import wandb
            
            # Initialize wandb
            self.wandb_run = wandb.init(
                project=getattr(self.config.logging, 'wandb_project', 'diffusion-training'),
                entity=getattr(self.config.logging, 'wandb_entity', 'sagar8'),
                name=f"{self.config.get('experiment_name', 'experiment')}_{os.path.basename(self.work_dir)}",
                dir=self.work_dir,
                config=self._flatten_config(self.config),
                reinit=True
            )
            
            # Watch model if provided
            logging.info("Wandb logging initialized")
            
        except ImportError:
            logging.warning("wandb not installed. Install with: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def _setup_local_logging(self):
        """Setup local metrics logging"""
        metrics_dir = Path(self.work_dir) / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        self.metrics_file = metrics_dir / "training_metrics.jsonl"
    
    def _flatten_config(self, config, parent_key='', sep='.'):
        """Flatten nested config for wandb"""
        items = []
        if hasattr(config, '_content'):
            config_dict = config._content
        else:
            config_dict = dict(config)
            
        for k, v in config_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if hasattr(v, '_content') or (hasattr(v, 'items') and not isinstance(v, str)):
                try:
                    items.extend(self._flatten_config(v, new_key, sep=sep).items())
                except:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def watch_model(self, model):
        """Watch model parameters for wandb"""
        if self.use_wandb and self.wandb_run and self.is_main_process:
            try:
                import wandb
                wandb.watch(model, log="all", log_freq=100)
            except Exception as e:
                logging.warning(f"Failed to watch model: {e}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to both wandb and local file"""
        if not self.is_main_process:
            return
        
        # Add step to metrics
        metrics_with_step = {"step": step, **metrics}
        
        # Log to wandb
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.log(metrics_with_step, step=step)
            except Exception as e:
                logging.warning(f"Failed to log to wandb: {e}")
        
        # Log to local file
        if self.metrics_file:
            try:
                with open(self.metrics_file, 'a') as f:
                    f.write(json.dumps(metrics_with_step) + '\n')
            except Exception as e:
                logging.warning(f"Failed to log to local file: {e}")
    
    def compute_training_metrics(self, model, optimizer, loss, config):
        """Compute comprehensive training metrics"""
        metrics = {
            "train/loss": float(loss.item()) if torch.is_tensor(loss) else float(loss),
        }
        
        # Learning rate
        if hasattr(config.logging, 'log_learning_rate') and config.logging.log_learning_rate:
            if hasattr(optimizer, 'param_groups'):
                current_lr = optimizer.param_groups[0]['lr']
                metrics["train/learning_rate"] = current_lr
        
        # Gradient norms
        if hasattr(config.logging, 'log_grad_norm') and config.logging.log_grad_norm:
            grad_norm = self._compute_grad_norm(model)
            if grad_norm is not None:
                metrics["train/grad_norm"] = grad_norm
        
        # Parameter norms
        if hasattr(config.logging, 'log_param_norm') and config.logging.log_param_norm:
            param_norm = self._compute_param_norm(model)
            if param_norm is not None:
                metrics["train/param_norm"] = param_norm
        
        return metrics
    
    def compute_eval_metrics(self, loss, additional_metrics=None):
        """Compute evaluation metrics"""
        metrics = {
            "eval/loss": float(loss.item()) if torch.is_tensor(loss) else float(loss),
        }
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                metrics[f"eval/{key}"] = value
        
        return metrics
    
    def _compute_grad_norm(self, model):
        """Compute gradient norm"""
        try:
            total_norm = 0.0
            param_count = 0
            
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                return total_norm
        except Exception:
            pass
        return None
    
    def _compute_param_norm(self, model):
        """Compute parameter norm"""
        try:
            total_norm = 0.0
            param_count = 0
            
            for p in model.parameters():
                param_norm = p.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                return total_norm
        except Exception:
            pass
        return None
    
    def log_palindrome_metrics(self, palindrome_results, step):
        """Log palindrome-specific metrics"""
        if not palindrome_results:
            return
        
        metrics = {}
        
        # Count valid palindromes
        valid_palindromes = [r for r in palindrome_results if r.get('is_palindrome', False) and len(r.get('text', '').strip()) > 2]
        total_samples = len(palindrome_results)
        non_empty = len([r for r in palindrome_results if len(r.get('text', '').strip()) > 0])
        
        metrics["palindrome/success_rate"] = len(valid_palindromes) / max(total_samples, 1)
        metrics["palindrome/non_empty_rate"] = non_empty / max(total_samples, 1)
        metrics["palindrome/valid_count"] = len(valid_palindromes)
        metrics["palindrome/total_count"] = total_samples
        
        # Average length of valid palindromes
        if valid_palindromes:
            avg_length = sum(len(r['text']) for r in valid_palindromes) / len(valid_palindromes)
            metrics["palindrome/avg_length"] = avg_length
        
        self.log_metrics(metrics, step)
    
    def finish(self):
        """Cleanup logging"""
        if self.use_wandb and self.wandb_run and self.is_main_process:
            try:
                self.wandb_run.finish()
            except Exception as e:
                logging.warning(f"Failed to finish wandb run: {e}")


def setup_logger(config, work_dir: str, rank: int = 0) -> TrainingLogger:
    """Setup training logger with config"""
    return TrainingLogger(config, work_dir, rank)


if __name__ == "__main__":
    # Test the logger
    from omegaconf import OmegaConf
    
    # Create test config
    config = OmegaConf.create({
        'logging': {
            'use_wandb': False,
            'wandb_project': 'test-project',
            'log_grad_norm': True,
            'log_param_norm': True,
            'log_learning_rate': True
        }
    })
    
    # Test logger
    logger = TrainingLogger(config, "/tmp/test_logs", rank=0)
    
    # Test metrics logging
    test_metrics = {
        "train/loss": 1.5,
        "train/accuracy": 0.85
    }
    
    logger.log_metrics(test_metrics, step=100)
    print("Logger test completed successfully!")