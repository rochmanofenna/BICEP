import triton
import triton.language as tl
import torch
import time

# Custom Philox-like counter RNG for better determinism
@triton.jit
def philox_like_rng(counter: tl.tensor, key: tl.tensor) -> tl.tensor:
    """
    Simple counter-based RNG inspired by Philox.
    Better determinism than tl.rand for parallel streams.
    """
    # Simple mixing function (not cryptographically secure, but sufficient for Monte Carlo)
    x = counter
    x = x ^ (x >> 16)
    x = x * 0x45d9f3b
    x = x ^ (x >> 16) 
    x = x * 0x45d9f3b
    x = x ^ (x >> 16)
    x = x + key
    
    # Convert to [0, 1) uniform
    return (x & 0x7fffffff).to(tl.float32) / 2147483648.0

# Enhanced SDE kernel with autotune and counter RNG
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=4),
    ],
    key=['n_steps', 'stride']
)
@triton.jit
def fused_sde_boxmuller_kernel(path_ptr, n_steps, stride, T,
                               feedback_value, decay_rate,
                               high_threshold, low_threshold,
                               total_steps, base_variance, 
                               BLOCK_SIZE: tl.constexpr):
    pid    = tl.program_id(0)
    path   = path_ptr + pid * stride
    dt     = T / n_steps
    acc    = tl.load(path)   # initial

    for i in range(n_steps):
        # Counter-based RNG for better determinism
        counter1 = pid * n_steps * 2 + i * 2
        counter2 = counter1 + 1
        key = 12345  # Fixed key for reproducibility
        
        # Two uniform draws using counter RNG
        u1 = philox_like_rng(counter1, key)
        u2 = philox_like_rng(counter2, key)
        
        # Box–Muller transformation → standard normal
        # Add small epsilon to prevent log(0)
        u1 = tl.maximum(u1, 1e-7)
        z0 = tl.sqrt(-2.0 * tl.log(u1)) * tl.cos(2 * 3.141592653589793 * u2)
        rnd = z0

        # your existing control math…
        norm    = 1.0 / total_steps
        factor1 = tl.where(norm < low_threshold,
                           1.5,
                           tl.where(norm > high_threshold, 0.5, 1.0))
        t       = i * dt
        vf      = base_variance * factor1 * tl.exp(-decay_rate * t)
        scale2  = tl.clamp(0.5 + feedback_value * 0.5, 0.2, 1.0)

        # Compute increment with better numerical stability
        vol_term = tl.sqrt(dt * vf)
        inc = rnd * vol_term * scale2
        acc += inc
        
        # Coalesced store
        tl.store(path + i + 1, acc)

def benchmark_bicep_kernel():
    """Benchmark the enhanced BICEP kernel."""
    # Test parameters
    n_paths, n_steps = 1024, 1000
    stride = n_steps + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("⚠️  CUDA not available, skipping GPU benchmark")
        return
    
    paths = torch.zeros((n_paths, stride), device=device, dtype=torch.float32)
    grid = (n_paths,)

    # Control parameters
    feedback_value = 0.5
    decay_rate = 0.1
    high_threshold = 10.0
    low_threshold = 2.0
    total_steps = float(n_steps)
    base_variance = 1.0

    print(f"🔥 Benchmarking enhanced BICEP kernel...")
    print(f"   - Paths: {n_paths}, Steps: {n_steps}")
    print(f"   - Device: {device}")
    
    # Warm-up and autotune
    print("   - Running autotune...")
    for _ in range(3):
        fused_sde_boxmuller_kernel[grid](
            paths, n_steps, stride, 1.0,
            feedback_value, decay_rate,
            high_threshold, low_threshold,
            total_steps, base_variance
        )
    
    # Synchronize for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    num_runs = 100
    print(f"   - Running {num_runs} iterations...")
    
    start_time = time.time()
    for _ in range(num_runs):
        fused_sde_boxmuller_kernel[grid](
            paths, n_steps, stride, 1.0,
            feedback_value, decay_rate,
            high_threshold, low_threshold,
            total_steps, base_variance
        )
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    avg_time = (time.time() - start_time) / num_runs
    paths_per_sec = n_paths / avg_time
    
    print(f"✅ Benchmark Results:")
    print(f"   - Average kernel time: {avg_time*1000:.2f} ms")
    print(f"   - Paths per second: {paths_per_sec:,.0f}")
    print(f"   - Enhanced features: Counter RNG ✓, Autotune ✓")

# Run benchmark if executed directly
if __name__ == "__main__":
    benchmark_bicep_kernel()
