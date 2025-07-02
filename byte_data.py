import torch
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, DistributedSampler
from data import cycle_loader
import re


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
    
    # Filter out very short or very long texts
    if len(text) < 10 or len(text) > 500:
        return None
    
    return text


def get_byte_wikipedia_dataset(mode="train", cache_dir=None, block_size=128, num_proc=8, max_samples=50000):
    """Get byte-level Wikipedia dataset"""
    
    # Load Wikipedia dataset
    if mode == "train":
        dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=cache_dir, streaming=False)
        # Take subset for faster processing
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    else:
        # Use a smaller validation set
        dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=cache_dir, streaming=False)
        start_idx = max_samples
        end_idx = min(start_idx + max_samples // 10, len(dataset))
        dataset = dataset.select(range(start_idx, end_idx))
    
    processor = ByteProcessor(max_length=block_size)
    
    def process_examples(examples):
        encoded_data = []
        for text in examples['text']:
            cleaned_text = clean_text(text)
            if cleaned_text:
                # Split long texts into chunks
                chunk_size = block_size - 20  # Leave room for special tokens
                text_bytes = processor.text_to_bytes(cleaned_text)
                
                # Create multiple samples from long texts
                for i in range(0, len(text_bytes), chunk_size):
                    chunk = text_bytes[i:i + chunk_size]
                    if len(chunk) >= 10:  # Minimum chunk size
                        chunk_text = processor.bytes_to_text(chunk)
                        byte_seq = processor.encode_text(chunk_text)
                        encoded_data.append({"input_ids": byte_seq})
        
        return {"processed": encoded_data}
    
    # Process the dataset
    processed_data = []
    for i in range(0, len(dataset), 1000):  # Process in batches
        batch = dataset[i:i+1000]
        result = process_examples(batch)
        processed_data.extend(result["processed"])
        
        if len(processed_data) >= max_samples:
            break
    
    # Limit to max_samples
    processed_data = processed_data[:max_samples]
    
    # Create dataset
    final_dataset = Dataset.from_list(processed_data)
    final_dataset = final_dataset.with_format('torch')
    
    return final_dataset


def get_byte_wikipedia_dataloaders(config, distributed=True, max_samples=50000):
    """Get byte-level Wikipedia dataloaders"""
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