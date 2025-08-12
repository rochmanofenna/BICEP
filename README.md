# BICEP: Brownian Compute Engine for Paths

A high-performance stochastic path generation library optimized for GPU acceleration, designed for financial modeling, scientific computing, and research applications.

## Overview

BICEP provides state-of-the-art implementations for generating Brownian motion paths with minimal latency and maximum throughput. The library supports multiple hardware backends including NVIDIA CUDA, Apple Metal Performance Shaders (MPS), and CPU, with automatic optimization for each platform.

### Key Features

- **Sub-millisecond latency**: Single path generation in under 0.4ms on modern GPUs
- **High throughput**: Over 2.5 million paths per second on consumer hardware
- **Multiple backends**: Optimized implementations for CUDA, Metal, and CPU
- **Memory efficient**: Advanced memory pooling and streaming capabilities
- **Neural network integration**: Seamless integration with PyTorch models
- **Precision options**: Support for both FP16 and FP32 computation

## Performance Benchmarks

Performance metrics on Apple M3 (Metal Performance Shaders):

| Batch Size | Time per Path | Throughput |
|------------|---------------|------------|
| 1          | 0.390 ms      | 2,564 paths/sec |
| 100        | 0.009 ms      | 111,111 paths/sec |
| 1,000      | 0.004 ms      | 250,000 paths/sec |
| 10,000     | 0.0004 ms     | 2,500,000 paths/sec |

Projected performance on NVIDIA A100: ~0.036 ms per path (10.8x faster)
Projected performance on NVIDIA H100: ~0.020 ms per path (20x faster)

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- CUDA toolkit (for NVIDIA GPU support)
- Metal Performance Shaders (for Apple Silicon)

### Install from source

```bash
git clone https://github.com/rochmanofenna/BICEP.git
cd BICEP
pip install -e .
```

### Dependencies

```bash
pip install torch numpy
```

For CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage

```python
from bicep_core import BICEPCore, BICEPConfig

# Initialize with default configuration
config = BICEPConfig(device='mps')  # or 'cuda' for NVIDIA GPUs
bicep = BICEPCore(config)

# Generate 1000 paths with 1000 time steps
paths = bicep.generate_paths(n_paths=1000, n_steps=1000)
print(f"Generated paths shape: {paths.shape}")  # [1000, 1001]
```

### Advanced Configuration

```python
# Custom configuration for optimal performance
config = BICEPConfig(
    device='cuda',
    max_paths=100000,
    max_steps=4000,
    use_half_precision=True,  # Use FP16 for faster computation
    use_memory_pool=True,     # Enable memory pooling
    tile_size=32              # Optimize for GPU warp size
)

bicep = BICEPCore(config)

# Generate paths with SDE controls
paths = bicep.generate_paths(
    n_paths=10000,
    n_steps=1000,
    T=1.0,                    # Time horizon
    feedback_value=0.7,       # Feedback control [0, 1]
    decay_rate=0.1           # Exponential decay rate
)
```

### Streaming Generation

For generating large numbers of paths with limited memory:

```python
from bicep_core import StreamingBICEP

streaming = StreamingBICEP(config, buffer_size=20000)

# Generate 1 million paths in chunks
total_generated = 0
for chunk in streaming.stream_generate(total_paths=1000000, n_steps=1000):
    # Process each chunk
    process_paths(chunk)
    total_generated += chunk.shape[0]
```

### Neural Network Integration

Integrate BICEP into PyTorch models:

```python
import torch
import torch.nn as nn
from bicep_core import NeuralBICEPLayer

class StochasticNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.bicep_layer = NeuralBICEPLayer(hidden_size, hidden_size, n_steps=100)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h = torch.relu(self.encoder(x))
        h_stochastic = self.bicep_layer(h)
        return self.decoder(h + h_stochastic)

# Use in training
model = StochasticNeuralNetwork(64, 128, 10)
optimizer = torch.optim.Adam(model.parameters())
```

## API Reference

### BICEPConfig

Configuration dataclass for BICEP engine:

- `device`: Computing device ('cuda', 'mps', 'cpu')
- `max_paths`: Maximum number of paths for memory pre-allocation
- `max_steps`: Maximum time steps per path
- `tile_size`: Memory alignment size for optimal GPU access
- `use_half_precision`: Enable FP16 computation
- `use_memory_pool`: Enable memory pooling
- `warmup_iterations`: Number of warmup iterations

### BICEPCore

Core path generation class:

#### Methods

- `generate_paths(n_paths, n_steps, T=1.0, feedback_value=0.5, decay_rate=0.1)`: Generate Brownian motion paths
  - `n_paths`: Number of paths to generate
  - `n_steps`: Number of time steps
  - `T`: Time horizon
  - `feedback_value`: Feedback control parameter [0, 1]
  - `decay_rate`: Exponential decay for time-dependent volatility

### StreamingBICEP

Streaming interface for large-scale generation:

#### Methods

- `stream_generate(total_paths, n_steps, **kwargs)`: Generate paths in streaming fashion
  - Yields chunks of paths for memory-efficient processing

### NeuralBICEPLayer

PyTorch module for neural network integration:

#### Parameters

- `input_size`: Input dimension
- `output_size`: Output dimension  
- `n_steps`: Number of time steps for path generation
- `config`: Optional BICEPConfig instance

## Architecture

BICEP employs several optimization techniques:

1. **Memory Pooling**: Pre-allocated memory pools eliminate allocation overhead
2. **Precision Optimization**: Optional FP16 computation for improved throughput
3. **Vectorized Operations**: Fully vectorized Box-Muller transformation
4. **Tiled Memory Access**: Aligned memory access patterns for GPU efficiency
5. **Kernel Fusion**: Combined operations to minimize memory bandwidth usage

## Benchmarking

Run the included benchmark suite:

```bash
python bicep_core.py
```

This will output detailed performance metrics for various configurations and batch sizes.

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/rochmanofenna/BICEP.git
cd BICEP

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## Citation

If you use BICEP in your research, please cite:

```bibtex
@software{bicep2025,
  title = {BICEP: Brownian Compute Engine for Paths},
  author = {Rochman, Ryan},
  year = {2025},
  url = {https://github.com/rochmanofenna/BICEP}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

- Author: Ryan Rochman
- Email: [Contact via GitHub]
- GitHub: [@rochmanofenna](https://github.com/rochmanofenna)

## Acknowledgments

This work leverages optimizations for modern GPU architectures including NVIDIA CUDA and Apple Metal Performance Shaders. Special thanks to the PyTorch team for their excellent GPU abstraction layer.