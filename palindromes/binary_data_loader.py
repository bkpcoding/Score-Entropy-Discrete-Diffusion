#!/usr/bin/env python3
"""
Efficient binary data loader for preprocessed Wikipedia dataset.
This is much faster than processing on-the-fly during training.

Usage:
    from binary_data_loader import get_binary_dataloaders
    train_loader, val_loader = get_binary_dataloaders(config)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from data import cycle_loader


class BinaryDataset(Dataset):
    """Dataset for loading preprocessed binary data"""
    
    def __init__(self, data_path, block_size=128):
        self.data_path = data_path
        self.block_size = block_size
        
        # Load the binary data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Binary data file not found: {data_path}")
        
        # Memory-map the file for efficient access
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        # Calculate number of samples
        self.num_samples = len(self.data) // block_size
        
        print(f"Loaded {data_path}: {len(self.data):,} tokens, {self.num_samples:,} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get the sequence starting at idx * block_size
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        
        # Extract the sequence
        sequence = self.data[start_idx:end_idx]
        
        # Convert to tensor
        input_ids = torch.from_numpy(sequence.astype(np.int64))
        
        return {'input_ids': input_ids}


def get_binary_dataloaders(config, distributed=True):
    """Get binary dataloaders for training and validation"""
    
    # Check batch size divisibility
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    
    # Create datasets
    train_dataset = BinaryDataset('train_filter.bin', block_size=config.model.length)
    #  ********* for validation overfit
    # train_dataset = BinaryDataset('val.bin', block_size=config.model.length)
    val_dataset = BinaryDataset('val_filter.bin', block_size=config.model.length)
    
    # Create samplers
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = cycle_loader(DataLoader(
        train_dataset,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    
    val_loader = cycle_loader(DataLoader(
        val_dataset,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(val_sampler is None),
    ))
    
    return train_loader, val_loader


def test_binary_loader():
    """Test the binary data loader"""
    print("Testing binary data loader...")
    
    # Check if binary files exist
    if not os.path.exists('train.bin'):
        print("Error: train.bin not found. Run prepare_byte_data.py first.")
        return
    
    if not os.path.exists('val.bin'):
        print("Error: val.bin not found. Run prepare_byte_data.py first.")
        return
    
    # Test datasets
    train_dataset = BinaryDataset('train.bin', block_size=128)
    val_dataset = BinaryDataset('val.bin', block_size=128)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Test a few samples
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        input_ids = sample['input_ids']
        print(f"Sample {i}: shape={input_ids.shape}, dtype={input_ids.dtype}")
        print(f"  First 10 tokens: {input_ids[:10].tolist()}")
        print(f"  Last 10 tokens: {input_ids[-10:].tolist()}")
        print()
    
    # Test data loader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    
    print("Testing data loader...")
    for i, batch in enumerate(train_loader):
        if i >= 2:  # Only test first 2 batches
            break
        
        input_ids = batch['input_ids']
        print(f"Batch {i}: shape={input_ids.shape}, dtype={input_ids.dtype}")
        print(f"  Min token: {input_ids.min().item()}")
        print(f"  Max token: {input_ids.max().item()}")
        print()
    
    print("Binary data loader test completed successfully!")


if __name__ == '__main__':
    test_binary_loader()