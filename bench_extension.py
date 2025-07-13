import time, torch
from torch.utils.cpp_extension import load

# 1) compile
sde_ext = load(
    name="sde_ext",
    sources=[
      "/src/randomness/sde_int/curand_kernel.cu",
      "/src/randomness/sde_int/binding.cu",
    ],
    extra_cuda_cflags=["-O3", "-I/usr/local/cuda/include"],
    extra_ldflags=["-lcurand"],
    verbose=True,
)

# 2) benchmark
n_paths, n_steps = 1024, 1000
stride = n_steps + 1
paths  = torch.zeros((n_paths, stride), device="cuda", dtype=torch.float32)
# warm-up
for _ in range(10):
    sde_ext.sde_curand(paths, n_steps, stride,
                       1.0, 0.5, 0.1, 10.0, 2.0,
                       float(n_steps), 1.0)

starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)
starter.record()
for _ in range(100):
    sde_ext.sde_curand(paths, n_steps, stride,
                       1.0, 0.5, 0.1, 10.0, 2.0,
                       float(n_steps), 1.0)
ender.record(); torch.cuda.synchronize()

ms = starter.elapsed_time(ender) / 100
print(f"CURAND avg time: {ms:.4f}ms")
