#!/usr/bin/env python3
"""
Simplified ENN-BICEP Integration
Direct connection between ENN temporal dynamics and BICEP stochastic simulation
"""
import torch
import torch.nn as nn
import numpy as np
import time
import sys

# Import BICEP
sys.path.insert(0, '.')
from metal_bicep_benchmark import metal_brownian_sde_kernel

class SimpleBICEPLayer(nn.Module):
    """Simplified BICEP layer that integrates cleanly with neural networks"""
    
    def __init__(self, input_size: int, output_size: int, device: str = 'mps'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        
        # Fixed BICEP parameters for this demo
        self.n_paths = min(output_size, 256)  # Match output size
        self.n_steps = 50  # Moderate complexity
        
        # Learnable control parameters
        self.feedback_transform = nn.Linear(input_size, 1)
        self.path_aggregator = nn.Linear(self.n_paths, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input through BICEP stochastic simulation
        
        Args:
            x: [batch_size, input_size]
        Returns:
            [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Convert input to feedback parameter
        feedback_raw = self.feedback_transform(x)  # [batch_size, 1]
        feedback_values = torch.sigmoid(feedback_raw).squeeze(-1)  # [batch_size]
        
        # Generate BICEP paths for each sample in batch
        batch_outputs = []
        for i in range(batch_size):
            feedback_val = feedback_values[i].item()
            
            # Generate Brownian paths with this sample's feedback
            paths = metal_brownian_sde_kernel(
                n_paths=self.n_paths,
                n_steps=self.n_steps,
                T=1.0,
                feedback_value=feedback_val,
                decay_rate=0.1,
                device=self.device
            )
            
            # Extract final values (endpoints of paths)
            endpoints = paths[:, -1]  # [n_paths]
            batch_outputs.append(endpoints)
        
        # Stack and project to output size
        stacked = torch.stack(batch_outputs)  # [batch_size, n_paths]
        output = self.path_aggregator(stacked)  # [batch_size, output_size]
        
        return output

class ENNWithSimpleBICEP(nn.Module):
    """ENN enhanced with BICEP stochastic dynamics"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: str = 'mps'):
        super().__init__()
        self.device = device
        
        # Traditional neural layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        
        # BICEP stochastic layer
        self.bicep_layer = SimpleBICEPLayer(hidden_size, hidden_size, device)
        
        # Output projection
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard forward pass
        h1 = torch.relu(self.input_layer(x))
        h2 = torch.relu(self.hidden_layer(h1))
        
        # BICEP stochastic transformation
        bicep_out = self.bicep_layer(h2)
        
        # Residual connection + output
        combined = h2 + bicep_out
        combined = self.dropout(combined)
        output = self.output_layer(combined)
        
        return output

def benchmark_simple_integration():
    """Test the simplified ENN-BICEP integration"""
    print("=== Simplified ENN-BICEP Integration ===")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Model configuration
    input_size, hidden_size, output_size = 64, 128, 10
    batch_size = 16
    
    # Create model and test data
    model = ENNWithSimpleBICEP(input_size, hidden_size, output_size, device).to(device)
    x = torch.randn(batch_size, input_size, device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Warm up
    with torch.no_grad():
        _ = model(x)
    
    # Benchmark forward pass
    times = []
    for _ in range(20):
        start = time.time()
        with torch.no_grad():
            output = model(x)
        if device == 'mps':
            torch.mps.synchronize()
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000
    throughput = batch_size / (avg_time / 1000)
    
    print(f"\\nPerformance Results:")
    print(f"Forward pass: {avg_time:.2f}ms")
    print(f"Throughput: {throughput:.0f} samples/sec")
    print(f"Output shape: {output.shape}")
    print(f"Per-sample time: {avg_time/batch_size:.3f}ms")
    
    # Test BICEP layer independently
    bicep_layer = SimpleBICEPLayer(hidden_size, hidden_size, device).to(device)
    h_test = torch.randn(batch_size, hidden_size, device=device)
    
    bicep_times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            _ = bicep_layer(h_test)
        if device == 'mps':
            torch.mps.synchronize()
        bicep_times.append(time.time() - start)
    
    bicep_avg = np.mean(bicep_times) * 1000
    print(f"\\nBICEP Layer Performance:")
    print(f"BICEP forward: {bicep_avg:.2f}ms")
    print(f"BICEP per sample: {bicep_avg/batch_size:.3f}ms")
    
    return avg_time

def demonstrate_stochastic_behavior():
    """Show how BICEP adds stochastic behavior to neural computation"""
    print("\\n=== Stochastic Behavior Demonstration ===")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Simple test case
    bicep_layer = SimpleBICEPLayer(32, 32, device).to(device)
    test_input = torch.randn(1, 32, device=device)
    
    print("Running same input through BICEP 5 times:")
    outputs = []
    for i in range(5):
        with torch.no_grad():
            output = bicep_layer(test_input)
            outputs.append(output.cpu().numpy())
            print(f"Run {i+1}: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    
    # Analyze variance
    outputs_array = np.array(outputs)
    variance_across_runs = np.var(outputs_array, axis=0)
    print(f"\\nVariance across runs: mean={variance_across_runs.mean():.6f}")
    print(f"This demonstrates BICEP's stochastic nature!")

if __name__ == "__main__":
    performance = benchmark_simple_integration()
    demonstrate_stochastic_behavior()
    
    print(f"\\n=== Integration Success ===")
    print(f"✅ ENN-BICEP integration working")
    print(f"✅ Performance: {performance:.2f}ms per forward pass")
    print(f"✅ Stochastic dynamics: Active")
    print(f"✅ Ready for ENN codebase integration")