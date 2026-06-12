import torch.nn as nn

class MLP(nn.Module):
    """
    The MLP backbone.
    Defaults to 4 hidden layers, 64 units each, with Tanh activations.
    """
    def __init__(
        self,
        in_dim=2,
        out_dim=4,
        hidden_dim=64,
        num_hidden_layers=4):
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)