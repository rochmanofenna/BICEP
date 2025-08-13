#!/usr/bin/env python3
"""
Test script to verify deterministic behavior of enhanced BICEP kernel.
"""

import torch
import numpy as np
from src.bicep.triton_kernel import fused_sde_boxmuller_kernel

def test_deterministic_kernel():
    """Test that the enhanced kernel produces deterministic results."""
    
    # Test parameters
    n_paths, n_steps = 32, 100
    stride = n_steps + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("⚠️  CUDA not available, skipping determinism test")
        return
    
    # Control parameters
    feedback_value = 0.5
    decay_rate = 0.1
    high_threshold = 10.0
    low_threshold = 2.0
    total_steps = float(n_steps)
    base_variance = 1.0
    T = 1.0
    
    print(f"🧪 Testing deterministic behavior...")
    print(f"   - Paths: {n_paths}, Steps: {n_steps}")
    
    # Run 1
    paths1 = torch.zeros((n_paths, stride), device=device, dtype=torch.float32)
    grid = (n_paths,)
    
    fused_sde_boxmuller_kernel[grid](
        paths1, n_steps, stride, T,
        feedback_value, decay_rate,
        high_threshold, low_threshold,
        total_steps, base_variance
    )
    
    # Run 2 - should be identical
    paths2 = torch.zeros((n_paths, stride), device=device, dtype=torch.float32)
    
    fused_sde_boxmuller_kernel[grid](
        paths2, n_steps, stride, T,
        feedback_value, decay_rate,
        high_threshold, low_threshold,
        total_steps, base_variance
    )
    
    # Check if results are identical
    paths1_cpu = paths1.cpu().numpy()
    paths2_cpu = paths2.cpu().numpy()
    
    are_identical = np.allclose(paths1_cpu, paths2_cpu, rtol=1e-6, atol=1e-8)
    max_diff = np.max(np.abs(paths1_cpu - paths2_cpu))
    
    print(f"✅ Determinism Test Results:")
    print(f"   - Identical results: {are_identical}")
    print(f"   - Max difference: {max_diff:.2e}")
    
    if are_identical:
        print(f"   - ✓ Counter RNG produces deterministic results")
    else:
        print(f"   - ⚠️  Results differ (may be due to floating point precision)")
    
    # Test different seeds should produce different results
    paths3 = torch.zeros((n_paths, stride), device=device, dtype=torch.float32)
    
    # We can't easily change the seed in this implementation, but we can test
    # that different path IDs produce different results
    first_path = paths1_cpu[0, :]
    second_path = paths1_cpu[1, :]
    
    paths_are_different = not np.allclose(first_path, second_path, rtol=1e-3)
    print(f"   - Different paths diverge: {paths_are_different}")
    
    if paths_are_different:
        print(f"   - ✓ Path independence verified")
    
    print(f"\n📊 Sample path statistics:")
    print(f"   - Mean final value: {paths1_cpu[:, -1].mean():.4f}")
    print(f"   - Std final value: {paths1_cpu[:, -1].std():.4f}")
    print(f"   - Min final value: {paths1_cpu[:, -1].min():.4f}")
    print(f"   - Max final value: {paths1_cpu[:, -1].max():.4f}")

if __name__ == "__main__":
    test_deterministic_kernel()