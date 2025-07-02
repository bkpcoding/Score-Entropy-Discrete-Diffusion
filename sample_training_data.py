#!/usr/bin/env python3
"""
Script to sample and visualize training data batches from the byte-level Wikipedia dataset.
This helps understand what the model is training on and learning from.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse
import os
from byte_data import get_byte_wikipedia_dataset, ByteProcessor
import hydra
from omegaconf import DictConfig, OmegaConf


def analyze_batch_statistics(batch, processor):
    """Analyze statistics of a batch of data"""
    stats = {
        'batch_size': len(batch),
        'sequence_length': len(batch[0]) if len(batch) > 0 else 0,
        'vocab_distribution': Counter(),
        'sequence_lengths': [],
        'special_token_counts': {'PAD': 0, 'BOS': 0, 'EOS': 0},
        'byte_value_stats': {'min': 256, 'max': 0, 'mean': 0}
    }
    
    all_tokens = []
    for sequence in batch:
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        
        # Count tokens
        for token in sequence:
            stats['vocab_distribution'][token] += 1
            all_tokens.append(token)
            
            # Count special tokens
            if token == processor.PAD:
                stats['special_token_counts']['PAD'] += 1
            elif token == processor.BOS:
                stats['special_token_counts']['BOS'] += 1
            elif token == processor.EOS:
                stats['special_token_counts']['EOS'] += 1
        
        # Calculate effective sequence length (without padding)
        effective_len = len([t for t in sequence if t != processor.PAD])
        stats['sequence_lengths'].append(effective_len)
    
    # Calculate byte value statistics (exclude special tokens)
    byte_values = [token for token in all_tokens if token < 256]
    if byte_values:
        stats['byte_value_stats'] = {
            'min': min(byte_values),
            'max': max(byte_values),
            'mean': np.mean(byte_values)
        }
    
    return stats


def visualize_batch_data(batch, processor, save_dir=None, batch_idx=0):
    """Create visualizations of the batch data"""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Convert to list if tensor
    if isinstance(batch, torch.Tensor):
        batch = batch.tolist()
    
    # Analyze statistics
    stats = analyze_batch_statistics(batch, processor)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Batch Analysis (Batch {batch_idx})', fontsize=16)
    
    # 1. Vocabulary distribution
    vocab_counts = dict(stats['vocab_distribution'])
    # Separate special tokens and byte values
    special_tokens = {k: v for k, v in vocab_counts.items() if k >= 256}
    byte_tokens = {k: v for k, v in vocab_counts.items() if k < 256}
    
    # Plot byte value distribution
    if byte_tokens:
        byte_vals, byte_counts = zip(*sorted(byte_tokens.items()))
        axes[0, 0].bar(byte_vals, byte_counts, alpha=0.7, width=1.0)
        axes[0, 0].set_title('Byte Value Distribution (0-255)')
        axes[0, 0].set_xlabel('Byte Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_xlim(0, 256)
    
    # 2. Special token counts
    special_names = ['PAD', 'BOS', 'EOS']
    special_counts = [stats['special_token_counts'][name] for name in special_names]
    axes[0, 1].bar(special_names, special_counts, color=['red', 'green', 'blue'], alpha=0.7)
    axes[0, 1].set_title('Special Token Counts')
    axes[0, 1].set_ylabel('Count')
    
    # 3. Sequence length distribution
    axes[0, 2].hist(stats['sequence_lengths'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Effective Sequence Lengths')
    axes[0, 2].set_xlabel('Length (excluding padding)')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. Character frequency in decoded text
    char_counter = Counter()
    decoded_texts = []
    for i, sequence in enumerate(batch[:5]):  # Sample first 5 sequences
        decoded = processor.decode_text(sequence)
        decoded_texts.append(decoded)
        char_counter.update(decoded.lower())
    
    # Plot most common characters
    if char_counter:
        common_chars = char_counter.most_common(20)
        chars, counts = zip(*common_chars)
        # Replace special characters for display
        display_chars = [repr(c) if ord(c) < 32 or ord(c) > 126 else c for c in chars]
        axes[1, 0].bar(range(len(chars)), counts, alpha=0.7)
        axes[1, 0].set_title('Most Common Characters (Top 20)')
        axes[1, 0].set_xlabel('Character')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_xticks(range(len(chars)))
        axes[1, 0].set_xticklabels(display_chars, rotation=45)
    
    # 5. Byte value heatmap (first few sequences)
    sample_sequences = batch[:min(10, len(batch))]
    if sample_sequences:
        # Convert to numpy array for heatmap
        seq_array = np.array(sample_sequences)
        im = axes[1, 1].imshow(seq_array, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Byte Values Heatmap (First 10 Sequences)')
        axes[1, 1].set_xlabel('Position in Sequence')
        axes[1, 1].set_ylabel('Sequence Index')
        plt.colorbar(im, ax=axes[1, 1])
    
    # 6. Text statistics
    axes[1, 2].text(0.1, 0.9, f"Batch Size: {stats['batch_size']}", transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.8, f"Sequence Length: {stats['sequence_length']}", transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, f"Unique Tokens: {len(stats['vocab_distribution'])}", transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, f"Avg Seq Length: {np.mean(stats['sequence_lengths']):.1f}", transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f"Byte Range: {stats['byte_value_stats']['min']}-{stats['byte_value_stats']['max']}", transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, f"Avg Byte Value: {stats['byte_value_stats']['mean']:.1f}", transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Batch Statistics')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'batch_analysis_{batch_idx}.png'), dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {os.path.join(save_dir, f'batch_analysis_{batch_idx}.png')}")
    
    plt.show()
    
    return decoded_texts, stats


def sample_and_analyze_data(config_path=None, num_batches=3, max_samples=1000):
    """Sample training data and analyze what the model sees"""
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        # Default configuration
        config = OmegaConf.create({
            'model': {'length': 128},
            'data': {'cache_dir': 'data'},
            'training': {'batch_size': 32},
            'eval': {'batch_size': 16},
            'ngpus': 1
        })
    
    print("=" * 60)
    print("TRAINING DATA ANALYSIS")
    print("=" * 60)
    
    print(f"Loading dataset with max_samples={max_samples}...")
    
    # Create processor
    processor = ByteProcessor(max_length=config.model.length)
    
    # Load dataset
    try:
        dataset = get_byte_wikipedia_dataset(
            mode="train", 
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            max_samples=max_samples
        )
        print(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This might be due to internet connectivity or missing cached data.")
        return
    
    # Create data loader
    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size // config.ngpus,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    print(f"\nAnalyzing {num_batches} batches...")
    print("-" * 60)
    
    # Sample and analyze batches
    all_stats = []
    all_texts = []
    
    for batch_idx, batch_data in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
            
        batch = batch_data['input_ids']
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Shape: {batch.shape}")
        
        # Decode and show sample texts
        print("  Sample texts from this batch:")
        sample_texts = []
        for i in range(min(3, len(batch))):
            text = processor.decode_text(batch[i].tolist())
            sample_texts.append(text)
            print(f"    {i+1}: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Visualize this batch
        decoded_texts, stats = visualize_batch_data(
            batch, processor, save_dir="training_data_analysis", batch_idx=batch_idx
        )
        
        all_stats.append(stats)
        all_texts.extend(decoded_texts)
        
        print(f"  Statistics:")
        print(f"    Unique tokens: {len(stats['vocab_distribution'])}")
        print(f"    Avg effective length: {np.mean(stats['sequence_lengths']):.1f}")
        print(f"    Padding ratio: {stats['special_token_counts']['PAD'] / (stats['batch_size'] * stats['sequence_length']):.2%}")
    
    # Overall analysis
    print("\n" + "=" * 60)
    print("OVERALL ANALYSIS")
    print("=" * 60)
    
    # Combine statistics
    total_vocab = Counter()
    all_lengths = []
    for stats in all_stats:
        total_vocab.update(stats['vocab_distribution'])
        all_lengths.extend(stats['sequence_lengths'])
    
    print(f"Total unique tokens across all batches: {len(total_vocab)}")
    print(f"Most common tokens: {total_vocab.most_common(10)}")
    print(f"Average sequence length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print(f"Sequence length range: {min(all_lengths)} - {max(all_lengths)}")
    
    # Character analysis
    all_text = " ".join(all_texts)
    char_freq = Counter(all_text.lower())
    print(f"\nText analysis:")
    print(f"  Total characters: {len(all_text)}")
    print(f"  Unique characters: {len(char_freq)}")
    print(f"  Most common characters: {char_freq.most_common(10)}")
    
    print(f"\nAnalysis complete! Visualizations saved in 'training_data_analysis/' directory.")


def main():
    parser = argparse.ArgumentParser(description="Sample and analyze training data batches")
    parser.add_argument("--config", type=str, default="configs/pretrain_byte.yaml", 
                       help="Path to config file")
    parser.add_argument("--num_batches", type=int, default=3, 
                       help="Number of batches to analyze")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of samples to load from dataset")
    
    args = parser.parse_args()
    
    sample_and_analyze_data(
        config_path=args.config,
        num_batches=args.num_batches,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()