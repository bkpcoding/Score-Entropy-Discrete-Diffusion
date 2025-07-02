import torch
import numpy as np
from sampling import get_pc_sampler, Denoiser
from byte_palindrome_data import BytePalindromeProcessor, is_palindrome
from model import utils as mutils


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
                             constraint_strength=0.8, apply_constraints=True):
    """Get palindrome-constrained PC sampler"""
    
    from sampling import get_predictor
    predictor = get_predictor(predictor)(graph, noise)
    processor = BytePalindromeProcessor(max_length=batch_dims[1])
    
    if apply_constraints:
        palindrome_sampler = PalindromeConstrainedSampler(
            graph, noise, processor, constraint_strength
        )
        projector = palindrome_sampler.palindrome_projection
    else:
        projector = lambda x: x

    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)
            
            # Apply palindrome constraints during sampling
            if apply_constraints and i % 5 == 0:  # Apply every 5 steps
                # Get score and apply constraint
                score = sampling_score_fn(x, timesteps[i] * torch.ones(x.shape[0], 1, device=device))
                constrained_score = palindrome_sampler.enforce_palindrome_constraint(score)
                # Update x based on constrained score (simplified)
                x = projector(x)

        if denoise:
            # Final denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            x = projector(x)  # Final constraint application
            
        return x
    
    return pc_sampler


def sample_palindromes(model, graph, noise, num_samples=5, max_length=128, 
                      steps=64, device=torch.device('cuda'), constraint_strength=0.9):
    """Sample palindromes and decode them to text"""
    
    processor = BytePalindromeProcessor(max_length=max_length)
    
    # Create sampler
    sampling_fn = get_palindrome_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(num_samples, max_length),
        predictor='analytic',
        steps=steps,
        denoise=True,
        device=device,
        constraint_strength=constraint_strength,
        apply_constraints=True
    )
    
    # Generate samples
    samples = sampling_fn(model)
    
    # Decode to text
    results = []
    for i in range(num_samples):
        sample = samples[i].cpu().numpy()
        text = processor.decode_palindrome(sample)
        is_pal = is_palindrome(text)
        results.append({
            'text': text,
            'is_palindrome': is_pal,
            'sample': sample
        })
    
    return results


def evaluate_palindrome_quality(model, graph, noise, num_samples=100, device=torch.device('cuda')):
    """Evaluate the quality of generated palindromes"""
    
    results = sample_palindromes(model, graph, noise, num_samples, device=device)
    
    # Count valid palindromes
    valid_palindromes = [r for r in results if r['is_palindrome'] and len(r['text'].strip()) > 2]
    palindrome_rate = len(valid_palindromes) / len(results)
    
    # Calculate average length
    avg_length = np.mean([len(r['text']) for r in results if len(r['text']) > 0])
    
    print(f"Generated {len(results)} samples")
    print(f"Valid palindromes: {len(valid_palindromes)} ({palindrome_rate:.2%})")
    print(f"Average length: {avg_length:.1f} characters")
    
    # Show some examples
    print("\nExample palindromes:")
    for i, result in enumerate(valid_palindromes[:5]):
        print(f"{i+1}: '{result['text']}'")
    
    print("\nExample non-palindromes:")
    non_palindromes = [r for r in results if not r['is_palindrome']]
    for i, result in enumerate(non_palindromes[:3]):
        print(f"{i+1}: '{result['text']}'")
    
    return {
        'palindrome_rate': palindrome_rate,
        'avg_length': avg_length,
        'valid_palindromes': valid_palindromes,
        'all_results': results
    }


if __name__ == "__main__":
    # Test palindrome constraint functions
    processor = BytePalindromeProcessor(max_length=16)
    sampler = PalindromeConstrainedSampler(None, None, processor)
    
    # Test constraint enforcement
    batch_size, seq_len, vocab_size = 2, 8, 259
    x = torch.randn(batch_size, seq_len, vocab_size)
    
    print("Testing palindrome constraints:")
    print(f"Input shape: {x.shape}")
    
    constrained_x = sampler.enforce_palindrome_constraint(x)
    print(f"Constrained shape: {constrained_x.shape}")
    
    # Test discrete projection
    discrete_x = torch.randint(0, vocab_size, (batch_size, seq_len))
    projected_x = sampler.palindrome_projection(discrete_x)
    
    print(f"Discrete input: {discrete_x[0]}")
    print(f"Projected: {projected_x[0]}")
    print(f"Is symmetric: {torch.equal(projected_x[0], projected_x[0].flip(0))}")