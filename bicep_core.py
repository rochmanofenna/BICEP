#!/usr/bin/env python3
"""
BICEP: Brownian Compute Engine for Paths
High-performance stochastic path generation using GPU acceleration

This module provides optimized implementations for generating Brownian motion paths
with support for multiple backends including PyTorch, Metal Performance Shaders (MPS),
and CUDA. The implementation focuses on minimal latency and maximum throughput
for financial modeling, scientific computing, and research applications.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import math
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class BICEPConfig:
    """Configuration parameters for BICEP engine"""
    device: str = 'mps'
    max_paths: int = 100000
    max_steps: int = 4000
    tile_size: int = 32
    use_half_precision: bool = True
    use_memory_pool: bool = True
    warmup_iterations: int = 5


class BICEPCore:
    """
    Core BICEP implementation with optimized memory management and computation
    
    This class provides the fundamental path generation capabilities with
    support for various optimization levels and hardware backends.
    """
    
    def __init__(self, config: Optional[BICEPConfig] = None):
        self.config = config or BICEPConfig()
        self.device = self.config.device
        
        # Memory management
        self._initialize_memory_pools()
        
        # Pre-computed constants
        self._initialize_constants()
        
        # Warm up kernels
        if self.config.warmup_iterations > 0:
            self._warmup()
    
    def _initialize_memory_pools(self):
        """Initialize pre-allocated memory pools for optimal performance"""
        if not self.config.use_memory_pool:
            return
            
        # Align to tile boundaries for optimal memory access
        aligned_paths = ((self.config.max_paths + self.config.tile_size - 1) // 
                        self.config.tile_size * self.config.tile_size)
        
        # Primary computation pools
        self._path_pool = torch.zeros(
            aligned_paths, self.config.max_steps + 1,
            device=self.device, dtype=torch.float32
        )
        
        if self.config.use_half_precision:
            self._increment_pool = torch.zeros(
                aligned_paths, self.config.max_steps,
                device=self.device, dtype=torch.float16
            )
            self._uniform_pool = torch.zeros(
                aligned_paths, self.config.max_steps, 2,
                device=self.device, dtype=torch.float16
            )
        else:
            self._increment_pool = torch.zeros(
                aligned_paths, self.config.max_steps,
                device=self.device, dtype=torch.float32
            )
            self._uniform_pool = torch.zeros(
                aligned_paths, self.config.max_steps, 2,
                device=self.device, dtype=torch.float32
            )
    
    def _initialize_constants(self):
        """Pre-compute mathematical constants"""
        dtype = torch.float16 if self.config.use_half_precision else torch.float32
        self._pi = torch.tensor(math.pi, device=self.device, dtype=dtype)
        self._two_pi = torch.tensor(2.0 * math.pi, device=self.device, dtype=dtype)
        self._neg_two = torch.tensor(-2.0, device=self.device, dtype=dtype)
        self._sqrt_2 = torch.tensor(math.sqrt(2.0), device=self.device, dtype=dtype)
    
    def _warmup(self):
        """Warm up GPU kernels to ensure optimal performance"""
        for _ in range(self.config.warmup_iterations):
            self.generate_paths(32, 100)
    
    def _box_muller_transform(self, u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        """
        Optimized Box-Muller transformation for normal distribution generation
        
        Args:
            u1: Uniform random values in (0, 1)
            u2: Uniform random values in (0, 1)
            
        Returns:
            Normal distributed random values
        """
        # Use appropriate precision
        if self.config.use_half_precision and u1.dtype == torch.float32:
            u1 = u1.half()
            u2 = u2.half()
        
        # Box-Muller transformation with fused operations
        log_u1 = torch.log(u1)
        cos_term = torch.cos(self._two_pi * u2)
        z0 = torch.sqrt(self._neg_two * log_u1) * cos_term
        
        return z0.float() if self.config.use_half_precision else z0
    
    def _apply_sde_controls(self, increments: torch.Tensor, n_steps: int, 
                           dt: float, feedback_value: float, decay_rate: float) -> torch.Tensor:
        """
        Apply stochastic differential equation controls to path increments
        
        Args:
            increments: Random increments
            n_steps: Number of time steps
            dt: Time step size
            feedback_value: Feedback control parameter [0, 1]
            decay_rate: Exponential decay rate
            
        Returns:
            Controlled increments
        """
        if decay_rate != 0:
            # Vectorized time decay
            t_grid = torch.arange(n_steps, device=self.device, dtype=increments.dtype) * dt
            decay_factors = torch.exp(-decay_rate * t_grid).unsqueeze(0)
            increments *= decay_factors
        
        if feedback_value != 0.5:
            # Feedback scaling with bounds
            scale = torch.clamp(torch.tensor(0.5 + feedback_value * 0.5), 0.2, 1.0)
            increments *= scale
        
        return increments
    
    def generate_paths(self, n_paths: int, n_steps: int, T: float = 1.0,
                      feedback_value: float = 0.5, decay_rate: float = 0.1) -> torch.Tensor:
        """
        Generate Brownian motion paths with optional SDE controls
        
        Args:
            n_paths: Number of paths to generate
            n_steps: Number of time steps per path
            T: Total time horizon
            feedback_value: Feedback control parameter [0, 1]
            decay_rate: Exponential decay rate for time-dependent volatility
            
        Returns:
            Generated paths of shape [n_paths, n_steps + 1]
        """
        assert n_paths <= self.config.max_paths, f"n_paths exceeds maximum {self.config.max_paths}"
        assert n_steps <= self.config.max_steps, f"n_steps exceeds maximum {self.config.max_steps}"
        
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        
        # Use memory pools if available
        if self.config.use_memory_pool:
            # Align to tile boundaries
            padded_paths = ((n_paths + self.config.tile_size - 1) // 
                           self.config.tile_size) * self.config.tile_size
            
            uniform_view = self._uniform_pool[:padded_paths, :n_steps]
            increment_view = self._increment_pool[:padded_paths, :n_steps]
            path_view = self._path_pool[:padded_paths, :n_steps + 1]
            
            # Generate uniform random numbers
            torch.rand(padded_paths, n_steps, 2, out=uniform_view)
            
            # Box-Muller transformation
            u1 = uniform_view[:, :, 0]
            u2 = uniform_view[:, :, 1]
            z0 = self._box_muller_transform(u1, u2)
            
            # Apply SDE controls
            z0_controlled = self._apply_sde_controls(z0, n_steps, dt, feedback_value, decay_rate)
            
            # Scale by sqrt(dt)
            increments = z0_controlled * sqrt_dt
            
            # Store increments
            if self.config.use_half_precision:
                increment_view[:, :] = increments.half()
            else:
                increment_view[:, :] = increments
            
            # Compute cumulative sum for paths
            path_view[:, 0] = 0  # Initial condition
            torch.cumsum(increment_view[:, :n_steps].float(), dim=1, out=path_view[:, 1:])
            
            # Return only requested paths
            return path_view[:n_paths, :].clone()
        else:
            # Direct generation without memory pools
            increments = torch.randn(n_paths, n_steps, device=self.device) * sqrt_dt
            increments = self._apply_sde_controls(increments, n_steps, dt, feedback_value, decay_rate)
            
            paths = torch.zeros(n_paths, n_steps + 1, device=self.device)
            paths[:, 1:] = torch.cumsum(increments, dim=1)
            
            return paths


class StreamingBICEP:
    """
    Streaming interface for continuous path generation with optimal memory usage
    
    This class enables generation of large numbers of paths by processing them
    in chunks, allowing for efficient memory usage and overlapped computation.
    """
    
    def __init__(self, config: Optional[BICEPConfig] = None, buffer_size: int = 20000):
        self.config = config or BICEPConfig()
        self.core = BICEPCore(self.config)
        self.buffer_size = buffer_size
        
        # Triple buffering for maximum overlap
        self.buffers = [
            torch.zeros(buffer_size, 1001, device=self.config.device, dtype=torch.float32)
            for _ in range(3)
        ]
        self.current_buffer = 0
    
    def stream_generate(self, total_paths: int, n_steps: int = 1000, **kwargs):
        """
        Generate paths in streaming fashion
        
        Args:
            total_paths: Total number of paths to generate
            n_steps: Number of steps per path
            **kwargs: Additional arguments passed to generate_paths
            
        Yields:
            Batches of generated paths
        """
        for start in range(0, total_paths, self.buffer_size):
            chunk_size = min(self.buffer_size, total_paths - start)
            
            # Generate into current buffer
            result = self.core.generate_paths(chunk_size, n_steps, **kwargs)
            
            # Copy to buffer
            current_buf = self.buffers[self.current_buffer]
            current_buf[:chunk_size, :n_steps + 1] = result
            
            yield current_buf[:chunk_size, :n_steps + 1]
            
            # Rotate buffer
            self.current_buffer = (self.current_buffer + 1) % 3


class NeuralBICEPLayer(nn.Module):
    """
    Neural network layer integrating BICEP stochastic dynamics
    
    This module enables integration of BICEP path generation into
    deep learning architectures for hybrid deterministic-stochastic models.
    """
    
    def __init__(self, input_size: int, output_size: int, n_steps: int = 100, 
                 config: Optional[BICEPConfig] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_steps = n_steps
        self.config = config or BICEPConfig()
        
        # BICEP engine
        self.bicep_core = BICEPCore(self.config)
        
        # Learnable parameters
        self.feedback_transform = nn.Linear(input_size, 1)
        self.path_processor = nn.Sequential(
            nn.Linear(n_steps + 1, output_size * 2),
            nn.ReLU(),
            nn.Linear(output_size * 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BICEP stochastic layer
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Convert input to feedback parameters
        feedback_raw = self.feedback_transform(x)
        feedback_values = torch.sigmoid(feedback_raw).squeeze(-1)
        
        # Generate paths for each sample
        outputs = []
        for i in range(batch_size):
            feedback = feedback_values[i].item()
            
            # Generate single path with feedback control
            path = self.bicep_core.generate_paths(1, self.n_steps, feedback_value=feedback)
            
            # Process path through neural network
            processed = self.path_processor(path.squeeze(0))
            outputs.append(processed)
        
        return torch.stack(outputs, dim=0)


def benchmark_performance():
    """Comprehensive performance benchmark of BICEP implementation"""
    print("BICEP Performance Benchmark")
    print("-" * 60)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test configurations
    configs = [
        BICEPConfig(device=device, use_half_precision=True, use_memory_pool=True),
        BICEPConfig(device=device, use_half_precision=False, use_memory_pool=True),
        BICEPConfig(device=device, use_half_precision=True, use_memory_pool=False),
    ]
    
    config_names = [
        "Optimized (FP16 + Memory Pool)",
        "Standard (FP32 + Memory Pool)",
        "Direct (FP16, No Pool)",
    ]
    
    # Test cases
    test_cases = [
        (1, 1000, "Single path"),
        (100, 1000, "Small batch"),
        (1000, 1000, "Medium batch"),
        (10000, 1000, "Large batch"),
    ]
    
    for config, name in zip(configs, config_names):
        print(f"\nConfiguration: {name}")
        bicep = BICEPCore(config)
        
        for n_paths, n_steps, desc in test_cases:
            times = []
            iterations = 100 if n_paths <= 100 else 20
            
            for _ in range(iterations):
                start = time.perf_counter()
                _ = bicep.generate_paths(n_paths, n_steps)
                if device == 'mps':
                    torch.mps.synchronize()
                times.append(time.perf_counter() - start)
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            per_path = avg_time / n_paths
            throughput = n_paths / (avg_time / 1000)
            
            print(f"  {desc:12s}: {per_path:.6f}ms/path, {throughput:10.0f} paths/sec")


if __name__ == "__main__":
    benchmark_performance()