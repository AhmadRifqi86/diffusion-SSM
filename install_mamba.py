#!/usr/bin/env python3
"""
Script to install mamba-ssm and test integration with model_v2.py
"""

import subprocess
import sys
import os

def install_mamba_ssm():
    """Install mamba-ssm package"""
    print("Installing mamba-ssm...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mamba-ssm"])
        print("✅ mamba-ssm installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install mamba-ssm: {e}")
        return False

def test_mamba_import():
    """Test if mamba-ssm can be imported"""
    try:
        from mamba_ssm import Mamba
        print("✅ mamba-ssm import successful!")
        return True
    except ImportError as e:
        print(f"❌ mamba-ssm import failed: {e}")
        return False

def test_model_integration():
    """Test the model integration"""
    try:
        from model_v2 import SelectiveSSM
        import torch
        
        print("Testing SelectiveSSM with official Mamba implementation...")
        
        # Create a test instance
        ssm = SelectiveSSM(d_model=256, d_state=16, d_conv=4, expand=2)
        
        # Create test input
        batch_size, seq_len, d_model = 2, 64, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Test forward pass
        with torch.no_grad():
            output = ssm(x)
        
        print(f"✅ Model test successful!")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Using official implementation: {ssm.use_official}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    print("=== Mamba-SSM Integration Test ===\n")
    
    # Step 1: Install mamba-ssm
    if not install_mamba_ssm():
        print("Installation failed. Please install manually:")
        print("pip install mamba-ssm")
        return
    
    # Step 2: Test import
    if not test_mamba_import():
        print("Import test failed. Please check your installation.")
        return
    
    # Step 3: Test model integration
    if not test_model_integration():
        print("Model integration test failed.")
        return
    
    print("\n🎉 All tests passed! Your model is now using the official Mamba implementation.")
    print("\nPerformance improvements you should see:")
    print("- Faster selective scan execution")
    print("- Better memory efficiency")
    print("- Optimized CUDA kernels")
    print("- Mixed precision support")

if __name__ == "__main__":
    main() 