#!/usr/bin/env python3
"""
Benchmark script for byte_data.py functions
"""

import time
import sys
import os
from byte_data import get_byte_wikipedia_dataset, ByteProcessor, clean_text

def benchmark_byte_processor():
    """Benchmark individual ByteProcessor functions"""
    print("=== BENCHMARKING ByteProcessor ===")
    
    processor = ByteProcessor(max_length=128)
    
    # Test texts of different lengths
    test_texts = [
        "Short text",
        "Medium length text with some more words to process and encode properly",
        "Very long text " * 50,  # ~700 characters
        "Unicode test: 你好世界 🌍 émojis and special chars ñáéíóú",
        ""  # Empty text
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {len(text)} chars")
        
        # Time text_to_bytes
        start = time.time()
        for _ in range(1000):
            bytes_result = processor.text_to_bytes(text)
        text_to_bytes_time = time.time() - start
        
        # Time encode_text
        start = time.time()
        for _ in range(1000):
            encoded = processor.encode_text(text)
        encode_time = time.time() - start
        
        # Time decode_text
        start = time.time()
        for _ in range(1000):
            decoded = processor.decode_text(encoded)
        decode_time = time.time() - start
        
        print(f"  text_to_bytes: {text_to_bytes_time*1000:.2f}ms (1000 calls)")
        print(f"  encode_text: {encode_time*1000:.2f}ms (1000 calls)")
        print(f"  decode_text: {decode_time*1000:.2f}ms (1000 calls)")

def benchmark_clean_text():
    """Benchmark clean_text function"""
    print("\n=== BENCHMARKING clean_text ===")
    
    # Test texts with different amounts of markup
    test_texts = [
        "Simple text without markup",
        "Text with [[wiki links]] and {{templates}}",
        "Text with <html>tags</html> and   multiple   spaces",
        "Complex [[wiki|link]] with {{template|param=value}} and <div>html</div>   extra   spaces",
        "A" * 1000,  # Very long text
        "Short",  # Very short text
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {len(text)} chars")
        
        start = time.time()
        for _ in range(1000):
            result = clean_text(text)
        clean_time = time.time() - start
        
        print(f"  clean_text: {clean_time*1000:.2f}ms (1000 calls)")
        print(f"  Result: {result[:50] + '...' if result and len(result) > 50 else result}")

def benchmark_dataset_creation():
    """Benchmark dataset creation with different parameters"""
    print("\n=== BENCHMARKING Dataset Creation ===")
    
    # Test different configurations
    configs = [
        {"max_samples": 10000, "num_proc": 1, "block_size": 64, "cache_dir": "data"},
        {"max_samples": 10000, "num_proc": 2, "block_size": 128, "cache_dir": "data"},
        {"max_samples": 10000, "num_proc": 4, "block_size": 256, "cache_dir": "data"},
        {"max_samples": 10000, "num_proc": 8, "block_size": 128, "cache_dir": "data"},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        
        try:
            dataset = get_byte_wikipedia_dataset(**config)
            print(f"SUCCESS: Created dataset with {len(dataset)} samples")
        except Exception as e:
            print(f"ERROR: {e}")

def profile_memory_usage():
    """Profile memory usage during dataset creation"""
    print("\n=== MEMORY PROFILING ===")
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        
        print(f"Initial memory: {process.memory_info().rss / 1024 / 1024:.1f}MB")
        
        # Create dataset and monitor memory
        dataset = get_byte_wikipedia_dataset(max_samples=10000, num_proc=4, cache_dir="data")
        
        print(f"Final memory: {process.memory_info().rss / 1024 / 1024:.1f}MB")
        print(f"Dataset size: {len(dataset)} samples")
        
    except ImportError:
        print("psutil not available for memory profiling")

if __name__ == "__main__":
    print("Starting comprehensive benchmark of byte_data.py")
    print("=" * 60)
    
    # Run all benchmarks
    # benchmark_byte_processor()
    # benchmark_clean_text()
    benchmark_dataset_creation()
    # profile_memory_usage()
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")