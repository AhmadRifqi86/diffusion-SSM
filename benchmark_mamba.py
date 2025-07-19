#!/usr/bin/env python3
"""
Benchmark script to compare custom vs official Mamba implementation performance
"""

import torch
import time
import numpy as np
from model_v2 import SelectiveSSM

def benchmark_mamba_implementation():
    """Benchmark both implementations"""
    
    # Test parameters
    batch_size = 4
    seq_len = 256
    d_model = 256
    d_state = 16
    d_conv = 4
    expand = 2
    num_runs = 100
    
    print(f"=== Mamba Implementation Benchmark ===")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print(f"State dimension: {d_state}")
    print(f"Number of runs: {num_runs}")
    print()
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model).cuda()
    
    # Test custom implementation (if available)
    try:
        # Force custom implementation by temporarily setting MAMBA_AVAILABLE to False
        import model_v2
        original_mamba_available = model_v2.MAMBA_AVAILABLE
        model_v2.MAMBA_AVAILABLE = False
        
        ssm_custom = SelectiveSSM(d_model, d_state, d_conv, expand).cuda()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = ssm_custom(x)
        
        torch.cuda.synchronize()
        
        # Benchmark custom implementation
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = ssm_custom(x)
        torch.cuda.synchronize()
        custom_time = time.time() - start_time
        
        print(f"Custom implementation:")
        print(f"  Time: {custom_time:.4f}s")
        print(f"  Time per run: {custom_time/num_runs*1000:.2f}ms")
        print(f"  Throughput: {num_runs/custom_time:.1f} runs/s")
        
        # Restore original setting
        model_v2.MAMBA_AVAILABLE = original_mamba_available
        
    except Exception as e:
        print(f"Custom implementation test failed: {e}")
        custom_time = None
    
    print()
    
    # Test official implementation (if available)
    try:
        ssm_official = SelectiveSSM(d_model, d_state, d_conv, expand).cuda()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = ssm_official(x)
        
        torch.cuda.synchronize()
        
        # Benchmark official implementation
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = ssm_official(x)
        torch.cuda.synchronize()
        official_time = time.time() - start_time
        
        print(f"Official implementation:")
        print(f"  Time: {official_time:.4f}s")
        print(f"  Time per run: {official_time/num_runs*1000:.2f}ms")
        print(f"  Throughput: {num_runs/official_time:.1f} runs/s")
        
    except Exception as e:
        print(f"Official implementation test failed: {e}")
        official_time = None
    
    print()
    
    # Compare results
    if custom_time is not None and official_time is not None:
        speedup = custom_time / official_time
        print(f"=== Performance Comparison ===")
        print(f"Speedup: {speedup:.2f}x faster with official implementation")
        
        if speedup > 1.5:
            print("🎉 Significant performance improvement!")
        elif speedup > 1.1:
            print("✅ Moderate performance improvement")
        else:
            print("📊 Minimal performance difference")
    
    # Memory usage comparison
    print(f"\n=== Memory Usage ===")
    torch.cuda.empty_cache()
    
    if custom_time is not None:
        ssm_custom = SelectiveSSM(d_model, d_state, d_conv, expand).cuda()
        torch.cuda.synchronize()
        custom_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Custom implementation: {custom_memory:.1f} MB")
        del ssm_custom
    
    if official_time is not None:
        ssm_official = SelectiveSSM(d_model, d_state, d_conv, expand).cuda()
        torch.cuda.synchronize()
        official_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Official implementation: {official_memory:.1f} MB")
        del ssm_official
    
    torch.cuda.empty_cache()

def test_mixed_precision():
    """Test mixed precision performance"""
    print(f"\n=== Mixed Precision Test ===")
    
    batch_size, seq_len, d_model = 4, 256, 256
    x = torch.randn(batch_size, seq_len, d_model).cuda()
    
    try:
        ssm = SelectiveSSM(d_model, 16, 4, 2).cuda()
        
        # Test with autocast
        scaler = torch.cuda.amp.GradScaler()
        
        # Warmup
        for _ in range(10):
            with torch.cuda.amp.autocast():
                _ = ssm(x)
        
        torch.cuda.synchronize()
        
        # Benchmark with mixed precision
        start_time = time.time()
        for _ in range(50):
            with torch.cuda.amp.autocast():
                _ = ssm(x)
        torch.cuda.synchronize()
        mixed_time = time.time() - start_time
        
        print(f"Mixed precision (autocast): {mixed_time:.4f}s")
        print(f"Time per run: {mixed_time/50*1000:.2f}ms")
        
    except Exception as e:
        print(f"Mixed precision test failed: {e}")

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. Please run on a GPU.")
        exit(1)
    
    print(f"Using device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print()
    
    benchmark_mamba_implementation()
    test_mixed_precision() 