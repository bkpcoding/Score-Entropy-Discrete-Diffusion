import torch
import argparse
from pathlib import Path

from load_model import load_model
from palindromes.palindrome_sampling import sample_palindromes

def main():
    parser = argparse.ArgumentParser(description="Generate palindrome samples")
    parser.add_argument("--model_path", default="checkpoints/best_model.pth", type=str,
                       help="Path to trained palindrome model")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of palindromes to generate")
    parser.add_argument("--max_length", type=int, default=12,
                       help="Maximum length of generated palindromes")
    parser.add_argument("--steps", type=int, default=100,
                       help="Number of diffusion steps")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for generation")
    parser.add_argument("--generate_text", action="store_true",
                       help="Generate non-palindromic text samples")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    try:
        if Path(args.model_path).exists():
            # Load local checkpoint
            checkpoint = torch.load(args.model_path, map_location=device)
            
            # Extract model from checkpoint if needed
            if 'model' in checkpoint and 'config' in checkpoint:
                config = checkpoint['config']
                model_state_dict = checkpoint['model']
                
                # Create model instance from config
                from model.transformer import SEDD
                model = SEDD(config).to(device)
                
                # Load the state dict into the model
                model.load_state_dict(model_state_dict, strict=False)
                model.eval()
            else:
                raise ValueError("Invalid checkpoint format - missing 'model' or 'config' keys")
                
            # Load graph and noise from config
            import graph_lib
            import noise_lib
            graph = graph_lib.get_graph(config, device)
            noise = noise_lib.get_noise(config).to(device)
            
        else:
            # Try loading as huggingface model
            model, graph, noise = load_model(args.model_path, device)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure to train the palindrome model first using train_palindrome.py")
        return
    
    
    print(f"\nGenerating {args.num_samples} palindromes...")
    print(f"Max length: {args.max_length}, Steps: {args.steps}")
    print("=" * 60)
        
    # results = sample_palindromes_with_intermediates(
    #     model=model,
    #     graph=graph,
    #     noise=noise,
    #     num_samples=1,
    #     max_length=args.max_length,
    #     steps=args.steps,
    #     device=device,
    #     constraint_strength=args.constraint_strength,
    #     save_every=args.save_every
    # )
    results = sample_palindromes(
        model=model,
        graph=graph,
        noise=noise,
        num_samples=args.num_samples,
        max_length=args.max_length,
        steps=args.steps,
        device=device,
    )

            
    # Generate additional samples normally if more than 1 requested
    if args.num_samples > 1:
        print(f"\nGenerating {args.num_samples - 1} additional palindromes...")
        additional_results = sample_palindromes(
            model=model,
            graph=graph,
            noise=noise,
            num_samples=args.num_samples - 1,
            max_length=args.max_length,
            steps=args.steps,
            device=device,
        )
        results.extend(additional_results)
        
    # Display results
    valid_palindromes = []
    invalid_palindromes = []
    
    for i, result in enumerate(results):
        text = result['text'].strip()
        is_pal = result['is_palindrome']
        
        if len(text) > 0:
            status = "✓ PALINDROME" if is_pal else "✗ Not palindrome"
            print(f"{i+1:2d}: {status:15s} | '{text}'")
            
            if is_pal and len(text) > 2:
                valid_palindromes.append(text)
            else:
                invalid_palindromes.append(text)
        else:
            print(f"{i+1:2d}: ✗ Empty | ''")
        
    # Summary statistics
    print("=" * 60)
    total_non_empty = len([r for r in results if len(r['text'].strip()) > 0])
    palindrome_rate = len(valid_palindromes) / max(total_non_empty, 1)
        
    print(f"Generated samples: {args.num_samples}")
    print(f"Non-empty: {total_non_empty}")
    print(f"Valid palindromes: {len(valid_palindromes)}")
    print(f"Palindrome rate: {palindrome_rate:.1%}")
    
    if valid_palindromes:
        avg_length = sum(len(p) for p in valid_palindromes) / len(valid_palindromes)
        print(f"Average palindrome length: {avg_length:.1f} characters")
    
    
    # Show some interesting examples
    if not args.generate_text and valid_palindromes:
        print(f"\nBest palindromes generated:")
        # Sort by length and show diverse examples
        sorted_palindromes = sorted(set(valid_palindromes), key=len, reverse=True)
        for i, palindrome in enumerate(sorted_palindromes[:5]):
            print(f"  {i+1}: '{palindrome}' (length: {len(palindrome)})")



if __name__ == "__main__":
    import sys
    
    main()