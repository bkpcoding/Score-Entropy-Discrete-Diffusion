#!/usr/bin/env python3
"""Test script to verify multi-GPU setup"""

import torch
import torch.distributed as dist
import os
import sys


def test_gpu_visibility():
    """Test GPU visibility and setup"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def test_distributed_setup():
    """Test distributed training setup"""
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    
    if dist.is_available():
        print("PyTorch distributed is available")
        if dist.is_initialized():
            print(f"Distributed initialized - Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
        else:
            print("Distributed not initialized")
    else:
        print("PyTorch distributed not available")


def test_model_on_gpus():
    """Test model creation on multiple GPUs"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        print(f"Testing GPU {i}:")
        try:
            # Create a simple model
            model = torch.nn.Linear(10, 1).to(device)
            x = torch.randn(5, 10).to(device)
            y = model(x)
            print(f"  ✓ Successfully created and ran model on GPU {i}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.1f} MB")
        except Exception as e:
            print(f"  ✗ Error on GPU {i}: {e}")


def main():
    print("=== Multi-GPU Test ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    print("1. GPU Visibility Test:")
    test_gpu_visibility()
    print()
    
    print("2. Distributed Setup Test:")
    test_distributed_setup()
    print()
    
    print("3. Model on GPUs Test:")
    test_model_on_gpus()
    print()
    
    # Test argument parsing
    if len(sys.argv) > 1:
        print(f"Arguments received: {sys.argv[1:]}")
        if "ngpus" in " ".join(sys.argv):
            print("✓ ngpus argument detected")


if __name__ == "__main__":
    main()