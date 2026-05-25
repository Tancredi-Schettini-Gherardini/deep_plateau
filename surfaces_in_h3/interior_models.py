import torch
import torch.nn as nn
import math

# ------------------------------
class MLP(nn.Module):
    def __init__(
            self,
            output_dim,
            hidden_dim = 64,
            activation = nn.Tanh(),
            depth = 4):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(activation)
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Combine layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

# --------------------------------
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class SIREN(nn.Module):
    def __init__(
            self,
            output_dim,
            input_dim=2,
            hidden_dim=256,
            depth=3,
            w0=30.0
        ):
        super().__init__()
        self.w0 = w0

        layers = []

        # First layer with special initialization
        first_linear = nn.Linear(input_dim, hidden_dim)
        self._init_first_layer(first_linear)
        layers.append(first_linear)
        layers.append(Sine(w0=self.w0))

        # Hidden layers
        for _ in range(depth - 1):
            linear = nn.Linear(hidden_dim, hidden_dim)
            self._init_hidden_layer(linear)
            layers.append(linear)
            layers.append(Sine(w0=self.w0))

        # Output layer (linear, typically no activation)
        final_linear = nn.Linear(hidden_dim, output_dim)
        nn.init.uniform_(final_linear.weight, -math.sqrt(6 / hidden_dim) / self.w0, 
                                          math.sqrt(6 / hidden_dim) / self.w0)
        layers.append(final_linear)

        self.model = nn.Sequential(*layers)

    def _init_first_layer(self, layer):
        # From SIREN paper: U[-1/input_dim, 1/input_dim]
        nn.init.uniform_(layer.weight, -1 / layer.in_features, 1 / layer.in_features)

    def _init_hidden_layer(self, layer):
        # From SIREN paper: U[-sqrt(6/input_dim)/w0, sqrt(6/input_dim)/w0]
        nn.init.uniform_(layer.weight, -math.sqrt(6 / layer.in_features) / self.w0,
                                          math.sqrt(6 / layer.in_features) / self.w0)

    def forward(self, x):
        return self.model(x)