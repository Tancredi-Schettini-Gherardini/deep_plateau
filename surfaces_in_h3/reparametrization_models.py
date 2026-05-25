import torch
import torch.nn as nn

class DiskPlaneBijection(nn.Module):
    """
    Implements a fixed diffeomorphism between the open unit disk D^2 and R^2.
    Using the algebraic map:
    Phi(x) = x / sqrt(1 - |x|^2)
    Phi^-1(y) = y / sqrt(1 + |y|^2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Maps D^2 -> R^2
        # x shape: (N, 2)
        norm_sq = torch.sum(x**2, dim=1, keepdim=True)
        # Numerical guard: strictly less than 1.0
        norm_sq = torch.clamp(norm_sq, max=0.999999) 
        return x / torch.sqrt(1.0 - norm_sq)

    def inverse(self, y):
        # Maps R^2 -> D^2
        # y shape: (N, 2)
        norm_sq = torch.sum(y**2, dim=1, keepdim=True)
        return y / torch.sqrt(1.0 + norm_sq)

class R2Diffeomorphism(nn.Module):
    """
    A neural network representing a perturbation of the identity in R^2.
    f(y) = y + decay(|y|) * Net(y)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Simple MLP for the displacement field
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(), # Smooth activation (Swish)
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )
        # Learnable scalar to control magnitude of perturbation
        self.scale = nn.Parameter(torch.tensor(0.1))

    def radial_decay(self, y):
        """
        Smooth radial function vanishing at infinity.
        Using Gaussian decay: e^(-|y|^2)
        This ensures the perturbation is zero at the boundary of the disk (infinity in R^2).
        """
        r_sq = torch.sum(y**2, dim=1, keepdim=True)
        return torch.exp(-r_sq)

    def forward(self, y):
        displacement = self.net(y)
        decay = self.radial_decay(y)
        
        # The map in R^2: y -> y + scale * decay * displacement
        return y + self.scale * decay * displacement

class DiskDiffeomorphismModel(nn.Module):
    """
    The full composite model:
    x -> Phi(x) -> R2_Diffeo -> Phi^-1(y) -> x_new
    """
    def __init__(self):
        super().__init__()
        self.phi = DiskPlaneBijection()
        self.r2_diffeo = R2Diffeomorphism()

    def forward(self, x):
        # 1. Map to R^2
        y = self.phi(x)
        
        # 2. Apply diffeomorphism in R^2 (with decay at infinity)
        y_new = self.r2_diffeo(y)
        
        # 3. Map back to D^2
        x_new = self.phi.inverse(y_new)
        
        return x_new