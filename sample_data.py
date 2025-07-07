#!/usr/bin/env python3
"""
Sample and print examples from processed byte data files.
Shows both the raw byte tokens and decoded text to verify data quality.
"""

import numpy as np
import argparse
import os

# Byte-level vocabulary (should match prepare_byte_data.py)
PAD = 256
BOS = 257  # Beginning of sequence
EOS = 258  # End of sequence
vocab_size = 259

def decode_bytes(byte_sequence):
    """Decode byte sequence back to text, handling special tokens"""
    decoded_parts = []
    
    for token in byte_sequence:
        if token == PAD:
            decoded_parts.append('<PAD>')
        elif token == BOS:
            decoded_parts.append('<BOS>')
        elif token == EOS:
            decoded_parts.append('<EOS>')
        elif token < 256:
            try:
                # Convert single byte to character
                decoded_parts.append(chr(token))
            except ValueError:
                decoded_parts.append(f'<BYTE:{token}>')
        else:
            decoded_parts.append(f'<UNK:{token}>')
    
    return ''.join(decoded_parts)

def sample_data(filename, num_samples=5, block_size=32):
    """Sample and display data from binary file"""
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        return
    
    print(f"\nSampling from {filename}")
    print("=" * 60)
    
    # Load data
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    total_blocks = total_tokens // block_size
    
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total blocks: {total_blocks:,}")
    print(f"Block size: {block_size}")
    print()
    
    # Sample random blocks
    sample_indices = np.random.choice(total_blocks, size=min(num_samples, total_blocks), replace=False)
    
    for i, block_idx in enumerate(sample_indices):
        start_idx = block_idx * block_size
        end_idx = start_idx + block_size
        
        block_data = data[start_idx:end_idx]
        decoded_text = decode_bytes(block_data)
        
        print(f"Sample {i+1} (Block {block_idx}):")
        print(f"Raw tokens: {list(block_data)}")
        print(f"Decoded: {repr(decoded_text)}")
        print(f"Text: {decoded_text}")
        print("-" * 40)

def analyze_data(filename, block_size=32):
    """Analyze data statistics"""
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        return
    
    print(f"\nAnalyzing {filename}")
    print("=" * 60)
    
    # Load data
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    
    # Token statistics
    unique_tokens, counts = np.unique(data, return_counts=True)
    
    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique tokens: {len(unique_tokens)}")
    print(f"Token range: {unique_tokens.min()} - {unique_tokens.max()}")
    print()
    
    # Special token counts
    special_tokens = {
        'PAD': PAD,
        'BOS': BOS,
        'EOS': EOS
    }
    
    print("Special token counts:")
    for name, token_id in special_tokens.items():
        count = counts[unique_tokens == token_id].sum() if token_id in unique_tokens else 0
        percentage = (count / total_tokens) * 100
        print(f"  {name} ({token_id}): {count:,} ({percentage:.2f}%)")
    
    # Byte token statistics (0-255)
    byte_mask = unique_tokens < 256
    byte_tokens = unique_tokens[byte_mask]
    byte_counts = counts[byte_mask]
    
    print(f"\nByte tokens (0-255): {len(byte_tokens)}")
    print(f"Most common bytes:")
    
    # Sort by count and show top 10
    sorted_indices = np.argsort(byte_counts)[::-1]
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        token = byte_tokens[idx]
        count = byte_counts[idx]
        percentage = (count / total_tokens) * 100
        char = chr(token) if token < 128 else f'<{token}>'
        print(f"  {token:3d} ('{char}'): {count:,} ({percentage:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Sample and analyze byte data files")
    parser.add_argument("--train_file", type=str, default="train_filter.bin", 
                       help="Training data file")
    parser.add_argument("--val_file", type=str, default="val_filter.bin", 
                       help="Validation data file")
    parser.add_argument("--num_samples", type=int, default=5, 
                       help="Number of samples to show")
    parser.add_argument("--block_size", type=int, default=32, 
                       help="Block size used in data preparation")
    parser.add_argument("--analyze", action="store_true", 
                       help="Show detailed statistics")
    
    args = parser.parse_args()
    
    print("BYTE DATA SAMPLING")
    print("=" * 60)
    
    # Sample from training data
    if os.path.exists(args.train_file):
        sample_data(args.train_file, args.num_samples, args.block_size)
        if args.analyze:
            analyze_data(args.train_file, args.block_size)
    
    # Sample from validation data
    if os.path.exists(args.val_file):
        sample_data(args.val_file, args.num_samples, args.block_size)
        if args.analyze:
            analyze_data(args.val_file, args.block_size)

if __name__ == '__main__':
    main()
