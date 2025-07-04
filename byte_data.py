import torch
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, DistributedSampler
from data import cycle_loader
import re
from multiprocessing import Pool
import time
import psutil
import os


class ByteProcessor:
    """Byte-level processor for general text"""
    
    def __init__(self, max_length=128):
        self.max_length = max_length
        # Byte vocabulary: 0-255 for all possible bytes, plus special tokens
        self.vocab_size = 256 + 3  # 256 bytes + PAD + BOS + EOS
        self.PAD = 256
        self.BOS = 257  # Beginning of sequence
        self.EOS = 258  # End of sequence
        
    def text_to_bytes(self, text):
        """Convert text to byte sequence"""
        return list(text.encode('utf-8'))
    
    def bytes_to_text(self, byte_seq):
        """Convert byte sequence back to text"""
        try:
            # Filter out special tokens
            clean_bytes = [b for b in byte_seq if b < 256]
            return bytes(clean_bytes).decode('utf-8', errors='ignore')
        except:
            return ""
    
    def encode_text(self, text):
        """Encode text to byte sequence with special tokens"""
        # Clean and truncate text
        text = text.strip()
        if len(text) == 0:
            text = "Hello world"  # Fallback for empty text
            
        byte_seq = self.text_to_bytes(text)
        
        # Add BOS and EOS tokens
        encoded = [self.BOS] + byte_seq + [self.EOS]
        
        # Pad or truncate to max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length-1] + [self.EOS]
        else:
            encoded = encoded + [self.PAD] * (self.max_length - len(encoded))
            
        return encoded
    
    def decode_text(self, byte_seq):
        """Decode byte sequence back to text"""
        # Remove padding and special tokens for text reconstruction
        clean_seq = []
        for b in byte_seq:
            if b == self.EOS:
                break
            elif b != self.PAD and b != self.BOS and b < 256:
                clean_seq.append(b)
        
        return self.bytes_to_text(clean_seq)


def clean_text(text):
    """Clean Wikipedia text"""
    # Remove wiki markup and excessive whitespace
    text = re.sub(r'\[\[.*?\]\]', '', text)  # Remove wiki links
    text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove templates
    text = re.sub(r'<.*?>', '', text)        # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)         # Normalize whitespace
    text = text.strip()
    
    # Filter out very short texts
    if len(text) < 100:
        print(f"Skipping text of length {len(text)}")
        return None
    
    return text


def process_single_example(args):
    """Process a single text example and yield chunks - global function for multiprocessing"""
    text, block_size = args
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return []
        
    # Split long texts into chunks - work directly with bytes
    chunk_size = block_size - 2  # Leave room for special tokens
    processor = ByteProcessor(max_length=block_size)
    text_bytes = processor.text_to_bytes(cleaned_text)
    
    chunks = []
    # Create multiple samples from long texts - avoid redundant conversions
    for i in range(0, len(text_bytes), chunk_size):
        chunk = text_bytes[i:i + chunk_size]
        if len(chunk) >= 10:  # Minimum chunk size
            # Direct encoding without intermediate text conversion
            encoded = [processor.BOS] + chunk + [processor.EOS]
            if len(encoded) > block_size:
                encoded = encoded[:block_size-1] + [processor.EOS]
            else:
                encoded = encoded + [processor.PAD] * (block_size - len(encoded))
            chunks.append({"input_ids": encoded})
    
    return chunks


def ensure_wikipedia_download(cache_dir=None):
    """Ensure Wikipedia dataset is fully downloaded and cached"""
    print("\n=== ENSURING WIKIPEDIA DATASET IS COMPLETE ===")
    
    if not cache_dir:
        cache_dir = "data"
    
    # Force download to complete
    try:
        print("Forcing complete download of Wikipedia dataset...")
        dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=cache_dir)
        print(f"✓ Wikipedia dataset ready: {len(dataset)} samples")
        return True
    except Exception as e:
        print(f"✗ Failed to download Wikipedia dataset: {e}")
        return False


def get_byte_wikipedia_dataset(mode="train", cache_dir=None, block_size=128, num_proc=8, max_samples=50000):
    """Get byte-level Wikipedia dataset using cached data for memory efficiency"""
    
    start_time = time.time()
    print(f"\n=== BENCHMARK: Starting dataset creation ===")
    print(f"Mode: {mode}, Block size: {block_size}, Num processes: {num_proc}, Max samples: {max_samples}")
    
    # Load cached Wikipedia dataset without streaming first to get length info
    load_start = time.time()
    try:
        # Check if cache exists and is complete
        import os
        if cache_dir and os.path.exists(cache_dir):
            cache_path = os.path.join(cache_dir, "wikipedia", "20220301.en", "2.0.0")
            if os.path.exists(cache_path):
                print(f"Found cache directory: {cache_path}")
                # Check for incomplete lock files
                incomplete_files = [f for f in os.listdir(cache_path) if f.endswith('.incomplete_info.lock')]
                if incomplete_files:
                    print(f"⚠️  Found incomplete cache files: {incomplete_files}")
                    print("Cache is incomplete, will need to download...")
        full_dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=cache_dir)
        total_samples = len(full_dataset)
        load_time = time.time() - load_start
        print(f"✓ Dataset loading: {load_time:.2f}s - Found cached dataset with {total_samples} samples")
    except Exception as e:
        # Fallback to streaming if cached version fails
        load_time = time.time() - load_start
        print(f"✗ Dataset loading: {load_time:.2f}s - Cache failed ({e}), using streaming dataset (slower)")
        full_dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=cache_dir, streaming=True)
        total_samples = None
    
    # Determine sample range
    select_start = time.time()
    if mode == "train":
        start_idx = 0
        end_idx = min(max_samples, total_samples) if total_samples else None
    else:
        # Use different samples for validation
        start_idx = max_samples if total_samples else max_samples
        end_idx = min(start_idx + max_samples, total_samples) if total_samples else None
    
    # Use cached dataset with selection if we have total_samples
    if total_samples:
        if end_idx:
            dataset = full_dataset.select(range(start_idx, end_idx))
        else:
            dataset = full_dataset.select(range(start_idx, min(start_idx + max_samples * 2, total_samples)))
    else:
        # Use streaming dataset
        dataset = full_dataset
    
    select_time = time.time() - select_start
    print(f"✓ Dataset selection: {select_time:.2f}s - Selected {len(dataset) if hasattr(dataset, '__len__') else 'streaming'} samples")
    
    
    # Process the dataset using multiprocessing for better performance
    processed_data = []
    
    # Process dataset (either selected subset or streaming)
    if total_samples:
        # Process selected subset with multiprocessing
        extraction_start = time.time()
        print(f"Processing {len(dataset)} articles from cached dataset using {num_proc} processes")
        # Sagar - each example is a article
        texts = [example['text'] for example in dataset]
        extraction_time = time.time() - extraction_start
        print(f"✓ Text extraction: {extraction_time:.2f}s")
        
        # Memory usage before processing
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Use multiprocessing for faster processing
        processing_start = time.time()
        # Prepare arguments for multiprocessing
        # Sagar- each text is a article
        args_list = [(text, block_size) for text in texts]
        with Pool(num_proc) as pool:
            # Sagar - The chunk results seems to have a lot 256 tokens (PAD)?
            chunk_results = pool.map(process_single_example, args_list)
        processing_time = time.time() - processing_start
        print(f"✓ Text Processing: {processing_time:.2f}s ({processing_time/len(texts)*1000:.1f}ms per article)")
        # Flatten results and limit to max_samples
        flatten_start = time.time()
        for chunks in chunk_results:
            processed_data.extend(chunks)
            if len(processed_data) >= max_samples:
                processed_data = processed_data[:max_samples]
                break
        flatten_time = time.time() - flatten_start
        
        # Memory usage after processing
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        print(f"✓ Result flattening: {flatten_time:.2f}s")
        print(f"✓ Memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB (+{mem_after-mem_before:.1f}MB)")
        print(f"✓ Processed {len(texts)} articles, got {len(processed_data)} samples")
    else:
        # Process streaming dataset (fallback to sequential for streaming)
        print("Processing streaming dataset")
        samples_processed = 0
        for example in dataset:
            chunks = process_single_example((example['text'], block_size))
            processed_data.extend(chunks)
            samples_processed += 1
            
            # Stop when we have enough samples
            if len(processed_data) >= max_samples:
                processed_data = processed_data[:max_samples]
                break
                
            # Print progress occasionally
            if samples_processed % 1000 == 0:
                print(f"Processed {samples_processed} articles, got {len(processed_data)} samples")
    
    # Create dataset
    dataset_creation_start = time.time()
    final_dataset = Dataset.from_list(processed_data)
    final_dataset = final_dataset.with_format('torch')
    dataset_creation_time = time.time() - dataset_creation_start
    
    total_time = time.time() - start_time
    print(f"✓ Dataset creation: {dataset_creation_time:.2f}s")
    print(f"=== BENCHMARK COMPLETE: {total_time:.2f}s total ===")
    print(f"Final dataset size: {len(final_dataset)} samples")
    print(f"Processing rate: {len(final_dataset)/total_time:.1f} samples/s\n")
    
    return final_dataset


def get_byte_wikipedia_dataloaders(config, distributed=True, max_samples=50000):
    """Get byte-level Wikipedia dataloaders - uses binary files if available, otherwise processes on-the-fly"""
    
    # Check if preprocessed binary files exist
    if os.path.exists('train.bin') and os.path.exists('val.bin'):
        print("✓ Found preprocessed binary files - using fast binary loader")
        try:
            from binary_data_loader import get_binary_dataloaders
            return get_binary_dataloaders(config, distributed)
        except ImportError:
            print("⚠️  binary_data_loader not found, falling back to on-the-fly processing")
    else:
        print("⚠️  Binary files not found - using slower on-the-fly processing")
        print("    Run 'python prepare_byte_data.py' to create fast binary files")
    
    # Fallback to original processing
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    train_set = get_byte_wikipedia_dataset("train", cache_dir=config.data.cache_dir, block_size=config.model.length, max_samples=max_samples)
    valid_set = get_byte_wikipedia_dataset("validation", cache_dir=config.data.cache_dir, block_size=config.model.length, max_samples=max_samples//10)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader


if __name__ == "__main__":
    # Test the byte-level text processing
    processor = ByteProcessor(max_length=64)
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Wikipedia is a free online encyclopedia.",
        "Machine learning models can learn from data."
    ]
    
    print("Testing byte-level text processing:")
    for text in test_texts:
        encoded = processor.encode_text(text)
        decoded = processor.decode_text(encoded)
        
        print(f"Original: '{text}'")
        print(f"Encoded length: {len([x for x in encoded if x != processor.PAD])}")
        print(f"Decoded: '{decoded}'")
        print(f"Match: {text.lower() == decoded.lower()}")
        print("-" * 50)
    
    # Test dataset creation (small sample)
    print("\nTesting dataset creation (this may take a moment)...")
    try:
        dataset = get_byte_wikipedia_dataset(max_samples=10)
        print(f"Created dataset with {len(dataset)} samples")
        
        # Show first few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]['input_ids']
            decoded = processor.decode_text(sample.tolist())
            print(f"Sample {i+1}: '{decoded[:100]}...'")
    except Exception as e:
        print(f"Dataset creation failed: {e}")
        print("This is normal if you don't have internet or the Wikipedia dataset isn't cached.")