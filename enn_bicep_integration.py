#!/usr/bin/env python3
"""
ENN-BICEP Integration Architecture
Connects Entangled Neural Networks with Brownian Compute Engine
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
import sys
import os

# Import our Metal BICEP
sys.path.insert(0, '.')
from metal_bicep_benchmark import metal_brownian_sde_kernel

class BICEPLayer(nn.Module):
    """
    BICEP Layer for Neural Networks
    Integrates stochastic Brownian motion simulation into neural computation
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 n_paths: int = 1000,
                 n_steps: int = 100,
                 T: float = 1.0,
                 learnable_params: bool = True,
                 device: str = 'mps'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.T = T
        self.device = device
        
        # Learnable stochastic control parameters
        if learnable_params:
            self.feedback_scale = nn.Parameter(torch.tensor(0.5))
            self.decay_rate = nn.Parameter(torch.tensor(0.1))
            self.variance_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('feedback_scale', torch.tensor(0.5))
            self.register_buffer('decay_rate', torch.tensor(0.1))
            self.register_buffer('variance_scale', torch.tensor(1.0))
        
        # Projection layers to/from BICEP space
        self.input_projection = nn.Linear(input_size, n_paths)
        self.output_projection = nn.Linear(n_paths, output_size)
        
        # Optional temporal aggregation
        self.temporal_aggregator = nn.Conv1d(n_steps + 1, 1, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BICEP layer
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Output tensor [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Project input to BICEP parameter space
        projected_input = self.input_projection(x)  # [batch_size, n_paths]
        
        # Generate BICEP paths for each batch element
        bicep_outputs = []
        for i in range(batch_size):
            # Use projected input as feedback values for each path
            feedback_values = torch.sigmoid(projected_input[i])  # [n_paths]
            
            # Generate paths with varying feedback per path
            batch_paths = []
            for j in range(min(10, self.n_paths)):  # Limit for efficiency
                feedback_val = feedback_values[j].item()
                paths = metal_brownian_sde_kernel(
                    n_paths=self.n_paths // 10,  # Distribute across feedback values
                    n_steps=self.n_steps,
                    T=self.T,
                    feedback_value=feedback_val,
                    decay_rate=self.decay_rate.item(),
                    device=self.device
                )
                batch_paths.append(paths)
            
            # Concatenate and aggregate
            all_paths = torch.cat(batch_paths, dim=0)  # [n_paths, n_steps+1]
            
            # Temporal aggregation: reduce time dimension
            aggregated = self.temporal_aggregator(all_paths.unsqueeze(0))  # [1, 1, n_paths]
            aggregated = aggregated.squeeze()  # [n_paths]
            
            bicep_outputs.append(aggregated)
        
        # Stack batch outputs
        bicep_tensor = torch.stack(bicep_outputs)  # [batch_size, n_paths]
        
        # Project to output space
        output = self.output_projection(bicep_tensor)  # [batch_size, output_size]
        
        return output

class ENNWithBICEP(nn.Module):
    """
    Enhanced ENN with integrated BICEP layers
    Combines neuron entanglement with stochastic Brownian dynamics
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_neurons: int = 64,
                 num_states: int = 4,
                 bicep_integration: str = 'parallel',  # 'parallel', 'sequential', 'residual'
                 device: str = 'mps'):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_neurons = num_neurons
        self.num_states = num_states
        self.bicep_integration = bicep_integration
        self.device = device
        
        # Core ENN components (simplified)
        self.neuron_embedding = nn.Linear(input_size, num_neurons * num_states)
        self.state_transition = nn.LSTM(num_states, num_states, batch_first=True)
        self.neuron_interaction = nn.MultiheadAttention(num_states, num_heads=4, batch_first=True)
        
        # BICEP integration layers
        if bicep_integration == 'parallel':
            self.bicep_layer = BICEPLayer(
                input_size=input_size,
                output_size=hidden_size,
                n_paths=num_neurons,
                n_steps=num_states * 10,
                device=device
            )
            self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)
            
        elif bicep_integration == 'sequential':
            self.bicep_layer = BICEPLayer(
                input_size=hidden_size,
                output_size=hidden_size,
                n_paths=num_neurons,
                n_steps=num_states * 10,
                device=device
            )
            
        elif bicep_integration == 'residual':
            self.bicep_layer = BICEPLayer(
                input_size=hidden_size,
                output_size=hidden_size,
                n_paths=num_neurons // 2,
                n_steps=num_states * 5,
                device=device
            )
        
        # Output layers
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with ENN-BICEP integration
        
        Returns:
            Dictionary with 'output', 'enn_features', 'bicep_features'
        """
        batch_size = x.size(0)
        
        # ENN processing
        neuron_states = self.neuron_embedding(x)  # [batch_size, num_neurons * num_states]
        neuron_states = neuron_states.view(batch_size, self.num_neurons, self.num_states)
        
        # LSTM state transitions
        lstm_out, _ = self.state_transition(neuron_states)
        
        # Multi-head attention for neuron interactions
        attended_states, attention_weights = self.neuron_interaction(
            lstm_out, lstm_out, lstm_out
        )
        
        # Aggregate neuron states
        enn_features = attended_states.mean(dim=1)  # [batch_size, num_states]
        enn_features = nn.functional.relu(enn_features)
        
        # Expand to hidden size for consistency
        if enn_features.size(-1) != self.hidden_size:
            enn_features = nn.functional.linear(enn_features, 
                                               torch.randn(self.hidden_size, enn_features.size(-1)).to(x.device))
        
        # BICEP integration
        if self.bicep_integration == 'parallel':
            bicep_features = self.bicep_layer(x)
            combined = torch.cat([enn_features, bicep_features], dim=-1)
            fused_features = self.fusion_layer(combined)
            
        elif self.bicep_integration == 'sequential':
            bicep_features = self.bicep_layer(enn_features)
            fused_features = bicep_features
            
        elif self.bicep_integration == 'residual':
            bicep_features = self.bicep_layer(enn_features)
            fused_features = enn_features + bicep_features
        
        # Output projection
        fused_features = self.dropout(fused_features)
        output = self.output_projection(fused_features)
        
        return {
            'output': output,
            'enn_features': enn_features,
            'bicep_features': bicep_features if 'bicep_features' in locals() else None,
            'attention_weights': attention_weights
        }

def benchmark_enn_bicep_integration():
    """Benchmark ENN-BICEP integration performance"""
    print("=== ENN-BICEP Integration Benchmark ===")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Test different integration modes
    modes = ['parallel', 'sequential', 'residual']
    input_size, hidden_size, output_size = 128, 256, 10
    batch_size = 32
    
    x = torch.randn(batch_size, input_size, device=device)
    
    for mode in modes:
        print(f"\n--- {mode.upper()} Integration ---")
        
        model = ENNWithBICEP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            bicep_integration=mode,
            device=device
        ).to(device)
        
        # Warm up
        _ = model(x)
        
        # Benchmark
        import time
        times = []
        for _ in range(10):
            start = time.time()
            result = model(x)
            if device == 'mps':
                torch.mps.synchronize()
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        throughput = batch_size / (avg_time / 1000)
        
        print(f"Forward pass: {avg_time:.2f}ms")
        print(f"Throughput: {throughput:.0f} samples/sec") 
        print(f"Output shape: {result['output'].shape}")
        
        # Parameter count
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

if __name__ == "__main__":
    benchmark_enn_bicep_integration()