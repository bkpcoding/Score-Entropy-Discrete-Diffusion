import torch
import numpy as np
from sampling import get_pc_sampler, Denoiser
from model import utils as mutils
import re

vocab_size = 256 + 3  # 256 bytes + PAD + BOS + EOS
PAD = 256
BOS = 257  # Beginning of sequence
EOS = 258  # End of sequence


def is_palindrome(text):
    """Check if text is a palindrome (ignoring spaces, punctuation, case)"""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
    return cleaned == cleaned[::-1] and len(cleaned) > 0

def decode_palindrome(byte_seq):
    """Decode byte sequence back to text"""
    # Remove padding and special tokens for text reconstruction
    clean_seq = []
    for b in byte_seq:
        if b == EOS:
            break
        elif b != PAD and b != BOS and b < 256:
            clean_seq.append(b)
    
    return bytes_to_text(clean_seq)

def bytes_to_text(byte_seq):
    """Convert byte sequence back to text"""
    try:
        # Filter out special tokens
        clean_bytes = [b for b in byte_seq if b < 256]
        return bytes(clean_bytes).decode('utf-8', errors='ignore')
    except:
        return ""


class PalindromeConstrainedSampler:
    """Sampling with palindrome constraints at byte level"""
    
    def __init__(self, graph, noise, processor, constraint_strength=0.8):
        self.graph = graph
        self.noise = noise
        self.processor = processor
        self.constraint_strength = constraint_strength
        self.denoiser = Denoiser(graph, noise)
    
    def enforce_palindrome_constraint(self, x, strength=None):
        """Enforce palindrome constraint by averaging positions"""
        if strength is None:
            strength = self.constraint_strength
            
        batch_size, seq_len = x.shape[:2]
        
        # Create palindromic version by averaging symmetric positions
        palindromic_x = x.clone()
        for i in range(seq_len // 2):
            j = seq_len - 1 - i
            # Average the distributions at symmetric positions
            avg_dist = (x[:, i] + x[:, j]) / 2
            palindromic_x[:, i] = avg_dist
            palindromic_x[:, j] = avg_dist
        # Interpolate between original and palindromic version
        return strength * palindromic_x + (1 - strength) * x
    
    def palindrome_projection(self, x):
        """Project samples to satisfy palindrome constraints"""
        batch_size, seq_len = x.shape
        
        # For discrete samples, enforce hard palindrome constraint
        palindromic_x = x.clone()
        
        for i in range(seq_len // 2):
            j = seq_len - 1 - i
            # For the left half, copy to right half
            palindromic_x[:, j] = palindromic_x[:, i]
        
        return palindromic_x


def get_palindrome_pc_sampler(graph, noise, batch_dims, predictor, steps, 
                             denoise=True, eps=1e-5, device=torch.device('cpu'),
                             constraint_strength=0.8, apply_constraints=True, proj_fun=lambda x: x):
    """Get palindrome-constrained PC sampler"""
    
    from sampling import get_predictor
    predictor = get_predictor(predictor)(graph, noise)
    processor = BytePalindromeProcessor(max_length=batch_dims[1])
    
    if apply_constraints:
        palindrome_sampler = PalindromeConstrainedSampler(
            graph, noise, processor, constraint_strength
        )
    else:
        palindrome_sampler = None

    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        projector = proj_fun
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            print(f"step {i} x.dtype: {x[0][0].dtype}")
            x = predictor.update_fn(sampling_score_fn, x, t, dt)
            
            # Apply palindrome constraints during sampling (soft constraints only)
            # if apply_constraints and palindrome_sampler and i % 5 == 0:  # Apply every 5 steps
            #     # Apply soft constraint directly to x
            #     x = palindrome_sampler.enforce_palindrome_constraint(x)
        if denoise:
            # Final denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        return x
    
    return pc_sampler




def sample_palindromes(model, graph, noise, num_samples=5, max_length=128, 
                      steps=64, device=torch.device('cuda')):
    """Sample palindromes and decode them to text"""
    
    # processor = BytePalindromeProcessor(max_length=max_length)
    
    # Create sampler with soft constraints only
    sampling_fn = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(num_samples, max_length),
        predictor='analytic',
        steps=steps,
        denoise=True,
        device=device,
        palindrome_sampling=True
    )
    
    # Generate samples
    samples = sampling_fn(model)
    
    # Decode to text
    results = []
    for i in range(num_samples):
        sample = samples[i].cpu().numpy()
        text = decode_palindrome(sample)
        is_pal = is_palindrome(text)
        results.append({
            'text': text,
            'is_palindrome': is_pal,
            'sample': sample
        })
    
    return results

