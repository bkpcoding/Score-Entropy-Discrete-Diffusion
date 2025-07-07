#!/usr/bin/env python3
"""
Preprocesses Wikipedia dataset for byte-level training and saves to binary files.
This is much faster than processing on-the-fly during training.

Usage:
    python prepare_byte_data.py --dataset_percent 10.0  # Use 10% of dataset
    python prepare_byte_data.py --dataset_percent 1.0   # Use 1% for quick testing
    python prepare_byte_data.py --dataset_percent 100.0 # Use full dataset
    
    # Custom configuration
    python prepare_byte_data.py --dataset_percent 5.0 --block_size 256 --num_proc 16

Arguments:
    --dataset_percent: Percentage of dataset to use (default: 10.0)
    --val_split_ratio: Validation split ratio (default: 0.001)
    --block_size: Sequence block size (default: 128)
    --num_proc: Number of processes (default: 8)
    --cache_dir: Cache directory (default: data)
    --batch_size: Chunking batch size (default: 5000)

Output:
    train.bin - Training data (byte sequences)
    val.bin - Validation data (byte sequences)
"""

import os
import re
import time
import argparse
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import psutil
from multiprocessing import Pool

# Byte-level vocabulary
PAD = 256
BOS = 257  # Beginning of sequence
EOS = 258  # End of sequence
vocab_size = 259  # 256 bytes + PAD + BOS + EOS


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Preprocess Wikipedia dataset for byte-level training")
    
    parser.add_argument("--dataset_percent", type=float, default=20.0,
                       help="Percentage of dataset to use (default: 15.0)")
    parser.add_argument("--val_split_ratio", type=float, default=0.001,
                       help="Validation split ratio (default: 0.001)")
    parser.add_argument("--block_size", type=int, default=32,
                       help="Sequence block size (default: 32)")
    parser.add_argument("--num_proc", type=int, default=8,
                       help="Number of processes for parallel processing (default: 8)")
    parser.add_argument("--cache_dir", type=str, default="data",
                       help="Cache directory for dataset (default: data)")
    parser.add_argument("--batch_size", type=int, default=5000,
                       help="Batch size for chunking (default: 5000)")
    
    return parser.parse_args()


def clean_text(text):
    """Clean Wikipedia text - keep spaces between words and remove non-English characters"""
    if not text or len(text) < 50:
        return None
    
    # Remove wiki markup and excessive whitespace
    text = re.sub(r'\[\[.*?\]\]', '', text)  # Remove wiki links
    text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove templates
    text = re.sub(r'<.*?>', '', text)        # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)         # Normalize whitespace to single spaces
    
    # Convert to lowercase
    text = text.lower()
    
    # Keep only English letters (a-z) and spaces - remove all punctuation, numbers, and non-English characters
    text = re.sub(r'[^a-z ]', '', text)
    
    # Remove extra spaces that might have been created
    text = re.sub(r'\s+', ' ', text)
    
    # Split into words and filter out very short words
    words = text.split()
    filtered_words = []
    
    for word in words:
        # Keep words that are at least 2 characters long
        if len(word) >= 2:
            filtered_words.append(word)
    
    # Join words with spaces to maintain word boundaries
    text = ' '.join(filtered_words)
    
    text = text.strip()
    
    # Filter out very short or very long texts
    if len(text) < 100 or len(text) > 5000:
        return None
    
    return text


def text_to_bytes(text):
    """Convert text to byte sequence"""
    return list(text.encode('utf-8'))


def process_example(example):
    """Process a single example - encode to bytes with BOS/EOS"""
    text = example['text']
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        # Return empty list but ensure consistent type
        return {'ids': [], 'len': 0}
    
    # Convert to bytes
    byte_seq = text_to_bytes(cleaned_text)
    
    # Add BOS and EOS tokens
    ids = [BOS] + byte_seq + [EOS]
    
    # Ensure ids is always a list of integers
    ids = [int(x) for x in ids]
    
    return {'ids': ids, 'len': len(ids)}


def chunk_single_sequence(args):
    """Chunk a single sequence - for multiprocessing, respecting word boundaries"""
    ids, block_size = args
    if len(ids) == 0:
        return []
    
    chunks = []
    i = 0
    space_byte = ord(' ')  # Space character as byte
    
    while i < len(ids):
        # Start with a chunk of block_size
        chunk_end = min(i + block_size, len(ids))
        
        # If we're at the end of the sequence, take what's left
        if chunk_end == len(ids):
            chunk = ids[i:chunk_end]
        else:
            # Look for the last space within the block to avoid cutting words
            last_space_pos = None
            for j in range(chunk_end - 1, i, -1):
                if ids[j] == space_byte:
                    last_space_pos = j
                    break
            
            # If we found a space and it's not too close to the beginning
            # (keep at least 20 tokens to avoid very short chunks)
            if last_space_pos is not None and (last_space_pos - i) >= 20:
                chunk = ids[i:last_space_pos + 1]  # Include the space
            else:
                # No good space found, take the full block (may cut word)
                chunk = ids[i:chunk_end]
        
        # Pad if necessary
        if len(chunk) < block_size:
            chunk = chunk + [PAD] * (block_size - len(chunk))
        
        chunks.append(chunk)
        i += len(chunk) - chunk.count(PAD)  # Move past the actual content (not padding)
    
    return chunks


def chunk_sequences_streaming(dataset, block_size, num_proc=8, batch_size=5000):
    """Chunk long sequences into fixed-size blocks using streaming with multiprocessing"""
    print(f"Chunking sequences into fixed-size blocks using {num_proc} processes (streaming)...")
    
    chunk_start = time.time()
    all_chunks = []
    total_processed = 0
    
    # Use a single pool for efficiency
    with Pool(num_proc) as pool:
        # Process in batches to avoid memory issues
        batch = []
        for example in tqdm(dataset, desc="Processing sequences"):
            if len(example['ids']) > 0:
                batch.append((example['ids'], block_size))
            
            # Process batch when it reaches batch_size
            if len(batch) >= batch_size:
                batch_results = pool.map(chunk_single_sequence, batch)
                
                # Flatten and add to results
                for chunks in batch_results:
                    all_chunks.extend(chunks)
                
                total_processed += len(batch)
                batch = []
                
                # Print progress
                if total_processed % 10000 == 0:
                    print(f"  Processed {total_processed} sequences, generated {len(all_chunks)} chunks")
        
        # Process remaining batch
        if batch:
            batch_results = pool.map(chunk_single_sequence, batch)
            
            for chunks in batch_results:
                all_chunks.extend(chunks)
            total_processed += len(batch)
    
    chunk_time = time.time() - chunk_start
    print(f"✓ Streaming chunking completed in {chunk_time:.2f}s")
    print(f"✓ Processed {total_processed} sequences into {len(all_chunks)} chunks")
    
    return all_chunks



def main():
    # Parse command line arguments
    args = parse_args()
    
    print("=" * 60)
    print("PREPROCESSING WIKIPEDIA DATASET FOR BYTE-LEVEL TRAINING")
    print("=" * 60)
    print(f"Dataset percentage: {args.dataset_percent}%")
    print(f"Validation split: {args.val_split_ratio:.1%}")
    print(f"Block size: {args.block_size}")
    print(f"Processes: {args.num_proc}")
    print(f"Cache dir: {args.cache_dir}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Memory monitoring
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.1f}MB")
    
    # Load dataset
    print(f"\\nLoading Wikipedia dataset (cache_dir: {args.cache_dir})...")
    load_start = time.time()
    
    try:
        full_dataset = load_dataset("wikipedia", "20220301.en", split="train", 
                                   cache_dir=args.cache_dir, num_proc=args.num_proc)
        load_time = time.time() - load_start
        print(f"✓ Full dataset loaded in {load_time:.2f}s: {len(full_dataset)} articles")
        
        # Select subset based on percentage
        if args.dataset_percent < 100.0:
            subset_size = int(len(full_dataset) * args.dataset_percent / 100.0)
            print(f"\\nSelecting {args.dataset_percent}% of dataset ({subset_size:,} articles)...")
            select_start = time.time()
            
            # Create indices for subset selection
            indices = list(range(subset_size))
            dataset = full_dataset.select(indices)
            
            select_time = time.time() - select_start
            print(f"✓ Subset selected in {select_time:.2f}s: {len(dataset)} articles")
        else:
            dataset = full_dataset
            print(f"Using full dataset: {len(dataset)} articles")
            
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Create train/val split
    print(f"\\nCreating train/val split ({args.val_split_ratio:.1%} for validation)...")
    split_start = time.time()
    
    split_dataset = dataset.train_test_split(
        test_size=args.val_split_ratio, 
        seed=42, 
        shuffle=True
    )
    split_dataset['val'] = split_dataset.pop('test')  # rename test to val
    
    split_time = time.time() - split_start
    print(f"✓ Split created in {split_time:.2f}s")
    print(f"  Train: {len(split_dataset['train'])} articles")
    print(f"  Val: {len(split_dataset['val'])} articles")
    
    # Process the dataset with HuggingFace's built-in parallel processing
    print(f"\\nProcessing dataset using HuggingFace multiprocessing with {args.num_proc} processes...")
    process_start = time.time()
    
    # Process train and validation splits separately to avoid schema conflicts
    tokenized = {}
    for split_name in ['train', 'val']:
        print(f"Processing {split_name} split...")
        tokenized[split_name] = split_dataset[split_name].map(
            process_example,
            remove_columns=split_dataset[split_name].column_names,
            desc=f"Processing {split_name} articles",
            num_proc=args.num_proc,  # Use multi-processing for faster processing
        )
    
    process_time = time.time() - process_start
    print(f"✓ Processing completed in {process_time:.2f}s")
    
    # Memory usage after processing
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage after processing: {current_memory:.1f}MB (+{current_memory-initial_memory:.1f}MB)")
    
    # Save to binary files
    print(f"\\nSaving to binary files (block_size: {args.block_size})...")
    
    for split_name, dset in tokenized.items():
        print(f"\\nProcessing {split_name} split...")
        
        # Filter out empty sequences
        dset = dset.filter(lambda x: x['len'] > 0)
        print(f"  Filtered dataset: {len(dset)} examples")
        
        # Chunk sequences with streaming parallel processing (avoids slow list conversion)
        all_chunks = chunk_sequences_streaming(dset, args.block_size, num_proc=args.num_proc, batch_size=args.batch_size)
        
        print(f"  ✓ Generated {len(all_chunks)} chunks of size {args.block_size}")
        
        # Save to binary file with optimized batch writing
        save_start = time.time()
        filename = f'{split_name}_filter.bin'
        
        print(f"  Saving to {filename}...")
        
        # Use uint16 since our vocab_size (259) < 2^16
        dtype = np.uint16
        total_tokens = len(all_chunks) * args.block_size
        
        # Create memory-mapped file
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_tokens,))
        
        # Batch writing for better performance
        write_batch_size = 10000  # Write 10k chunks at a time
        for batch_start in tqdm(range(0, len(all_chunks), write_batch_size), desc=f"Writing {filename}"):
            batch_end = min(batch_start + write_batch_size, len(all_chunks))
            batch_chunks = all_chunks[batch_start:batch_end]
            
            # Convert batch to numpy array
            batch_array = np.array(batch_chunks, dtype=dtype)
            
            # Write batch to file
            start_idx = batch_start * args.block_size
            end_idx = start_idx + len(batch_chunks) * args.block_size
            arr[start_idx:end_idx] = batch_array.flatten()
        
        arr.flush()
        save_time = time.time() - save_start
        
        # File size
        file_size = os.path.getsize(filename) / 1024 / 1024  # MB
        
        print(f"  ✓ Saved to {filename} in {save_time:.2f}s")
        print(f"  ✓ File size: {file_size:.1f}MB")
        print(f"  ✓ Total tokens: {total_tokens:,}")
        print(f"  ✓ Chunks: {len(all_chunks):,}")
    
    # Final statistics
    total_time = time.time() - start_time
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print("\\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Dataset percentage used: {args.dataset_percent}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Peak memory: {final_memory:.1f}MB")
    print(f"Block size: {args.block_size}")
    print(f"Vocab size: {vocab_size}")
    print()
    print("Files created:")
    for filename in ['train_filter.bin', 'val_filter.bin']:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024 / 1024  # MB
            print(f"  {filename}: {size:.1f}MB")
    
    print("\\nTo read the binary files later:")
    print("  train_data = np.memmap('train_filter.bin', dtype=np.uint16, mode='r')")
    print("  val_data = np.memmap('val_filter.bin', dtype=np.uint16, mode='r')")


if __name__ == '__main__':
    main()
