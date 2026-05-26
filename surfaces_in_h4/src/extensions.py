import math
import torch

class StereobiharmonicEvaluator(torch.nn.Module):
    """
    Pure PyTorch module that holds the Fourier coefficients and evaluates 
    the stereobiharmonic extension.
    """
    def __init__(self, A0, An, Bn, N):
        super().__init__()
        self.N = N
        
        # Registering these as buffers means PyTorch handles device/dtype 
        # casting automatically, and Dynamo captures them cleanly.
        self.register_buffer("A0", A0)
        self.register_buffer("An", An)
        self.register_buffer("Bn", Bn)

    def forward(self, xy):
        # xy shape: (B, 2)
        x = xy[..., 0:1] 
        y = xy[..., 1:2] 
        
        r2 = x**2 + y**2
        multiplier = 2.0 / (1.0 + r2)  # Shape: (B, 1)
        
        # Base result
        base_res = self.A0 * (1.0 + r2) / 2.0
        
        if self.N > 0:
            U, V = x, y 
            
            for n in range(self.N):
                k = n + 1
                
                # The stereographic-biharmonic weight
                weight = (k + 1) / 2.0 + (1 - k) / 2.0 * r2
                
                # Broadcast and add
                base_res = base_res + weight * ((U * self.An[n]) + (V * self.Bn[n]))
                
                if n < self.N - 1:
                    U_next = U * x - V * y
                    V_next = V * x + U * y
                    U, V = U_next, V_next
                    
        return multiplier * base_res


class Extension:
    def __init__(self, evaluator_module):
        """
        Wraps an evaluator module mapping xy of shape (N, 2) to (N, 3).
        """
        self.evaluator_module = evaluator_module
        self.parametrized = True

    def __call__(self, xy):
        return self.evaluator_module(xy)

    def x(self, xy):
        return self.evaluator_module(xy)[..., 0]

    def y(self, xy):
        return self.evaluator_module(xy)[..., 1]

    def z(self, xy):
        return self.evaluator_module(xy)[..., 2]


def get_stereobiharmonic_extension(knot, N=15, num_samples=20000, device='cpu', dtype=torch.float64):
    """
    Computes the biharmonic extension of the knot.
    Returns an Extension object.
    """
    # --- 1. PRECOMPUTATION (Fourier Coefficients) ---
    step = 2 * math.pi / num_samples
    
    # We use .view(-1, 1) so it natively matches the (N, 1) shape 
    # expected by our compiled knot evaluators!
    th = (torch.arange(num_samples, device=device, dtype=dtype) * step).view(-1, 1)
    
    # Evaluate the knot
    pts = knot(th)  # Shape: (num_samples, 3)
    
    A0 = pts.mean(dim=0)  
    An = torch.zeros((N, 3), device=device, dtype=dtype)
    Bn = torch.zeros((N, 3), device=device, dtype=dtype)
    
    for n in range(1, N + 1):
        cos_n = torch.cos(n * th)  # Shape: (num_samples, 1)
        sin_n = torch.sin(n * th)  # Shape: (num_samples, 1)
        
        An[n - 1] = (2.0 / num_samples) * (pts * cos_n).sum(dim=0)
        Bn[n - 1] = (2.0 / num_samples) * (pts * sin_n).sum(dim=0)

    # --- 2. CREATE EVALUATOR ---
    evaluator = StereobiharmonicEvaluator(A0, An, Bn, N)
    
    # --- 3. RETURN UNIFIED EXTENSION OBJECT ---
    ext = Extension(evaluator)
    ext.name = 'stereobiharmonic'
    ext.config = {
        'kind': 'stereobiharmonic',
        'kwargs': {'N': N, 'num_samples': num_samples}
    }
    
    return ext