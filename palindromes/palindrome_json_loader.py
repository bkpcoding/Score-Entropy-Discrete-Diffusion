#!/usr/bin/env python3
"""
Simple dataloader for palindromes.json dataset.
Loads the JSON file and creates a PyTorch dataset for fine-tuning.
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

# Byte-level vocabulary constants (matching prepare_byte_data.py)
PAD = 256
BOS = 257  # Beginning of sequence
EOS = 258  # End of sequence
vocab_size = 259  # 256 bytes + PAD + BOS + EOS


def text_to_bytes(text):
    """Convert text to byte sequence"""
    return list(text.encode('utf-8'))


def process_palindrome(palindrome_text, max_length=128):
    """Process a single palindrome into byte sequence with BOS/EOS"""
    # Convert to bytes
    byte_seq = text_to_bytes(palindrome_text)
    
    # Add BOS and EOS tokens
    ids = [BOS] + byte_seq + [EOS]
    
    # Truncate or pad to max_length
    if len(ids) > max_length:
        ids = ids[:max_length]
    elif len(ids) < max_length:
        ids = ids + [PAD] * (max_length - len(ids))
    
    return ids


class PalindromeDataset(Dataset):
    """Dataset for palindromes from JSON file"""
    
    def __init__(self, json_path, max_length=128, augment=True):
        self.max_length = max_length
        self.augment = augment
        
        # Load palindromes from JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            self.palindromes = json.load(f)
        
        print(f"Loaded {len(self.palindromes)} palindromes from {json_path}")
        
        # Pre-process all palindromes
        self.processed_data = []
        for palindrome in self.palindromes:
            processed = process_palindrome(palindrome, max_length)
            self.processed_data.append(processed)
        
        # If augment is True, add case variations and duplicates for training
        if augment:
            original_size = len(self.processed_data)
            augmented_data = []
            
            for palindrome in self.palindromes:
                # Add lowercase version
                if palindrome.lower() != palindrome:
                    processed = process_palindrome(palindrome.lower(), max_length)
                    augmented_data.append(processed)
                
                # Add uppercase version
                if palindrome.upper() != palindrome:
                    processed = process_palindrome(palindrome.upper(), max_length)
                    augmented_data.append(processed)
                
                # Add title case version
                if palindrome.title() != palindrome:
                    processed = process_palindrome(palindrome.title(), max_length)
                    augmented_data.append(processed)
            
            self.processed_data.extend(augmented_data)
            
            # Duplicate dataset multiple times for more training data
            original_data = self.processed_data.copy()
            for _ in range(4):  # 5x total data
                self.processed_data.extend(original_data)
            
            print(f"Augmented dataset: {original_size} -> {len(self.processed_data)} samples")
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.processed_data[idx], dtype=torch.long)
        }


def create_palindrome_dataloaders(json_path, max_length=128, batch_size=32, 
                                 val_split=0.1, num_workers=4):
    """Create train and validation dataloaders for palindromes"""
    
    # Load full dataset with augmentation
    full_dataset = PalindromeDataset(json_path, max_length=max_length, augment=True)
    
    # Split into train/val
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split: {train_size} train, {val_size} val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_palindrome_dataloaders(cfg, distributed=False):
    """Get palindrome dataloaders compatible with the training script"""
    json_path = "palindromes.json"
    
    # Use config values if available, otherwise defaults
    max_length = getattr(cfg.model, 'length', 128)
    batch_size = getattr(cfg.training, 'batch_size', 32)
    
    # Adjust batch size for distributed training
    if distributed:
        batch_size = batch_size // cfg.ngpus
    
    train_loader, val_loader = create_palindrome_dataloaders(
        json_path=json_path,
        max_length=max_length,
        batch_size=batch_size,
        val_split=0.1,
        num_workers=4
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataloader
    print("Testing palindrome dataloader...")
    
    train_loader, val_loader = create_palindrome_dataloaders(
        "palindromes.json", 
        max_length=64,
        batch_size=8
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Sample sequence: {batch['input_ids'][0][:20].tolist()}")
    
    # Decode first few tokens
    sample_ids = batch['input_ids'][0][:20].tolist()
    print(f"Decoded: {[chr(x) if x < 256 else f'<{x}>' for x in sample_ids]}")