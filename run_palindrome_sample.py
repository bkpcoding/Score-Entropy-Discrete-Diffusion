import torch
import argparse
from pathlib import Path

from load_model import load_model
from palindrome_sampling import sample_palindromes, evaluate_palindrome_quality
from byte_palindrome_data import BytePalindromeProcessor, is_palindrome


def main():
    parser = argparse.ArgumentParser(description="Generate palindrome samples")
    parser.add_argument("--model_path", default="checkpoints/best_model.pth", type=str,
                       help="Path to trained palindrome model")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of palindromes to generate")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum length of generated palindromes")
    parser.add_argument("--steps", type=int, default=64,
                       help="Number of diffusion steps")
    parser.add_argument("--constraint_strength", type=float, default=0.8,
                       help="Strength of palindrome constraints (0.0 to 1.0)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run comprehensive evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for generation")
    
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
    
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} palindromes...")
    print(f"Max length: {args.max_length}, Steps: {args.steps}")
    print(f"Constraint strength: {args.constraint_strength}")
    print("=" * 60)
    
    results = sample_palindromes(
        model=model,
        graph=graph,
        noise=noise,
        num_samples=args.num_samples,
        max_length=args.max_length,
        steps=args.steps,
        device=device,
        constraint_strength=args.constraint_strength
    )
    
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
            print(f"{i+1:2d}: ✗ Empty          | ''")
    
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
    
    # Comprehensive evaluation
    if args.evaluate:
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        eval_results = evaluate_palindrome_quality(
            model, graph, noise, 
            num_samples=100, 
            device=device
        )
    
    # Show some interesting examples
    if valid_palindromes:
        print(f"\nBest palindromes generated:")
        # Sort by length and show diverse examples
        sorted_palindromes = sorted(set(valid_palindromes), key=len, reverse=True)
        for i, palindrome in enumerate(sorted_palindromes[:5]):
            print(f"  {i+1}: '{palindrome}' (length: {len(palindrome)})")


def test_without_model():
    """Test palindrome processing without a trained model"""
    print("Testing palindrome processing (no model required)...")
    
    processor = BytePalindromeProcessor(max_length=64)
    
    test_cases = [
        "racecar",
        "A man a plan a canal Panama", 
        "Was it a car or a cat I saw",
        "hello world",
        "level",
        "Madam Im Adam"
    ]
    
    print("\nTest cases:")
    for i, text in enumerate(test_cases):
        encoded = processor.encode_palindrome(text)
        decoded = processor.decode_palindrome(encoded)
        is_pal = is_palindrome(text)
        
        print(f"{i+1}: '{text}' -> palindrome: {is_pal}")
        print(f"   Encoded length: {len([x for x in encoded if x != processor.PAD])}")
        print(f"   Decoded: '{decoded}'")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("No arguments provided. Running test mode...")
        test_without_model()
    else:
        main()