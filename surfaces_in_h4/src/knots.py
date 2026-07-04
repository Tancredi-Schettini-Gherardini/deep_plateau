import torch


# ------------------------
# Knot generator fucntion
# ------------------------
def build_knot(
    knot_type,
    knot_kwargs,
    knot_perturbation_matrix = None
):
    match knot_type:
        case 'unknot':
            knot = get_unknot(**knot_kwargs)

        case 'torus':
            knot = get_torus_knot(**knot_kwargs)

        case 'conformal_torus':
            knot = get_conformal_torus_knot(**knot_kwargs)

        case 'figure8':
            knot = get_figure8(**knot_kwargs)

        case 'figure8_torus':
            knot = get_figure8_torus(**knot_kwargs)

        case 'stevedore':
            knot = get_stevedore_knot(**knot_kwargs)

        case 'square':
            knot = get_square_knot(**knot_kwargs)
        
        case 'three_twist':
            knot = get_three_twist_knot(**knot_kwargs)
        
        case 'lissajous':
            knot = get_lissajous_knot(**knot_kwargs)

        case _:
            raise NotImplementedError(f'Knot type "{knot_type}" not implemented.')
    
    if knot_perturbation_matrix is not None:
        knot.perturb(knot_perturbation_matrix)

    return knot


# ------------------------
# helper functions
# ------------------------
def generate_knot_perturbation_matrix(
        N=4,
        scale=0.1,
        seed=None,
        dtype=torch.float64,
        device='cpu'
):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(6, N, device=device, dtype=dtype) * scale


# ------------------------
# KnotEvaluator Class
# ------------------------
class KnotEvaluator(torch.nn.Module):
    """
    A pure PyTorch module representing the mathematical state of the knot.
    This is highly optimized for torch.compile().
    """
    def __init__(self, base_x, base_y, base_z, perturbation_matrix=None):
        super().__init__()
        self.base_x = base_x
        self.base_y = base_y
        self.base_z = base_z
        
        if perturbation_matrix is not None:
            # Registering as buffers means PyTorch automatically handles moving 
            # these to the correct device/dtype when you call .to(device)
            self.register_buffer("P", perturbation_matrix)
            # Frequencies: 1, 2, ..., N_freq
            self.register_buffer("ns", torch.arange(1, perturbation_matrix.shape[1] + 1))
        else:
            self.P = None

    def forward(self, t):
        # t is expected to be shape (N, 1)
        
        # 1. Evaluate base parametrizations. These should return shape (N, 1)
        x = self.base_x(t)
        y = self.base_y(t)
        z = self.base_z(t)
        
        # 2. Add perturbations if they exist
        if self.P is not None:
            # Broadcasting: t (N, 1) * ns (N_freq,) -> nt (N, N_freq)
            nt = t * self.ns 
            cost = torch.cos(nt)
            sint = torch.sin(nt)
            
            # self.P shape is (6, N_freq). self.P[0] is Ax, self.P[3] is Bx, etc.
            # cost * P[0] -> (N, N_freq). Summing over dim=-1 with keepdim gives (N, 1)
            px = (cost * self.P[0] + sint * self.P[3]).sum(dim=-1, keepdim=True)
            py = (cost * self.P[1] + sint * self.P[4]).sum(dim=-1, keepdim=True)
            pz = (cost * self.P[2] + sint * self.P[5]).sum(dim=-1, keepdim=True)
            
            x = x + px
            y = y + py
            z = z + pz
            
        # 3. Concatenate along the last dimension
        # x, y, z are all (N, 1). Concatenating them gives (N, 3)
        return torch.cat([x, y, z], dim=-1)


# ------------------------
# Knot Class
# ------------------------
class Knot():
    def __init__(self):
        self.name = "Knot"
        self.perturbation_matrix = None
        self.base_x = None
        self.base_y = None
        self.base_z = None

    def make_parametrization(self, x, y, z):
        self.parametrized = True
        # Store the BASE functions here.
        self.base_x = x
        self.base_y = y
        self.base_z = z

    def perturb(self, perturbation_matrix):
        self.perturbation_matrix = perturbation_matrix

    def get_evaluator(self):
        """
        Returns a KnotEvaluator for this knot's parametrization.
        """
        # 1. Create the evaluator module
        evaluator = KnotEvaluator(
            self.base_x, 
            self.base_y, 
            self.base_z, 
            self.perturbation_matrix
        )
        
        # 2. Return the evaluator
        return evaluator
    

# ------------------------
# Implemented knots
# ------------------------
def get_unknot(R=1.0):
    """
    Factory function to generate an Unknot ready for PyTorch compilation.
    """
    # 1. Define base functions using 'def'. 
    # TorchDynamo perfectly captures the 'R' variable here without graph breaks.
    def base_x(th): 
        return R * torch.cos(th)
    
    def base_y(th): 
        return R * torch.sin(th)
    
    def base_z(th): 
        return R * torch.zeros_like(th)

    # 2. Build the Knot configuration
    unknot = Knot()
    unknot.make_parametrization(base_x, base_y, base_z)
    
    # 3. Keep your metadata
    unknot.config = {'kind': 'unknot', 'kwargs': {'R': R}}
    
    return unknot


def get_torus_knot(p, q, R=2.0, r=1.0):
    """
    Factory function to generate a Torus Knot ready for PyTorch compilation.
    """
    # 1. Define base functions using 'def' for optimal Dynamo capture
    def base_x(theta): 
        return (R + r * torch.cos(theta * q)) * torch.cos(theta * p)
    
    def base_y(theta): 
        return (R + r * torch.cos(theta * q)) * torch.sin(theta * p)
    
    def base_z(theta): 
        return r * torch.sin(theta * q)

    # 2. Build the Knot configuration
    knot = Knot()
    knot.make_parametrization(base_x, base_y, base_z)
    
    # 3. Maintain your metadata
    knot.config = {
        'kind': 'torus',
        'kwargs': {'p': p, 'q': q, 'R': R, 'r': r}
    }
    
    return knot


def get_conformal_torus_knot(p, q):
    """
    Generates a (p,q) torus knot via stereographic projection of the 
    Clifford torus from S3 to R3. These lie on the exact same torus 
    as before (R=1, r=1/sqrt(2)), but their parameterization is 
    conformally symmetric. Ready for PyTorch compilation.
    """
    
    # 1. Define the shared denominator as a local 'def'
    def denom(theta):
        # 2**0.5 is fine, but torch.sqrt(torch.tensor(2.0)) or writing it 
        # as a constant is slightly safer for strict graph compilation.
        # However, Python floats like 2**0.5 are usually handled well by Inductor.
        return 2.0 - (2.0 ** 0.5) * torch.sin(theta * q)

    # 2. Define base functions using 'def'
    def base_x(theta): 
        return torch.cos(theta * p) / denom(theta)
    
    def base_y(theta): 
        return torch.sin(theta * p) / denom(theta)
    
    def base_z(theta): 
        return torch.cos(theta * q) / denom(theta)

    # 3. Build the Knot configuration
    knot = Knot()
    knot.make_parametrization(base_x, base_y, base_z)
    
    # 4. Maintain your metadata
    knot.config = {
        'kind': 'conformal_torus',
        'kwargs': {'p': p, 'q': q}
    }
    
    return knot


def get_figure8(R=2.0, r=1.0):
    """
    Factory function to generate a Figure-8 Knot ready for PyTorch compilation.
    """
    # 1. Define base functions using 'def' and explicit floats for constants
    def base_x(t): 
        return R * torch.cos(3.0 * t) + r * torch.cos(t)
    
    def base_y(t): 
        return R * torch.sin(3.0 * t) + r * torch.sin(t)
    
    def base_z(t): 
        return R * torch.sin(4.0 * t)

    # 2. Build the Knot configuration
    knot = Knot()
    knot.make_parametrization(base_x, base_y, base_z)
    
    # 3. Maintain your metadata
    knot.config = {
        'kind': 'figure8', 
        'kwargs': {'R': R, 'r': r}
    }
    
    return knot


def get_figure8_torus(R=2.0, A=1.0, mirror=False):
    """
    Returns a figure-eight knot via a modulated-torus parametrization.
    """
    # 1. Define base functions using 'def' and explicit float multipliers
    def base_x(th):
        if mirror:
            return -(R + A * torch.cos(2.0 * th)) * torch.cos(3.0 * th)
        else:
            return (R + A * torch.cos(2.0 * th)) * torch.cos(3.0 * th)

    def base_y(th):
        if mirror:
            return -(R + A * torch.cos(2.0 * th)) * torch.sin(3.0 * th)
        else:
            return (R + A * torch.cos(2.0 * th)) * torch.sin(3.0 * th)
    
    def base_z(th):
        if mirror:
            return -A * torch.sin(4.0 * th)
        else:
            return A * torch.sin(4.0 * th)

    # 2. Build the Knot configuration
    figure_eight = Knot()
    figure_eight.make_parametrization(base_x, base_y, base_z)
    
    # 3. Maintain your metadata
    figure_eight.config = {
        'kind': 'figure8_torus', 
        'kwargs': {'R': R, 'A': A, 'mirror': mirror}
    }
    
    return figure_eight


def get_stevedore_knot(R=1.0, mirror = False):
    """
    The 6_1 knot (Stevedore knot) parameterized as a Lissajous knot.
    Frequencies (nx, ny, nz) = (3, 2, 5).
    As a slice knot, this is a prime candidate for finding
    embedded (rather than just immersed) minimal discs in H4.
    Ready for PyTorch compilation.
    """
    
    # Phase shifts required to strictly embed the 6_1 topology.
    # Without these exact shifts, the curve will self-intersect.
    phi_x = 1.5
    phi_y = 0.2
    phi_z = 0.0

    # 1. Define base functions using 'def' and explicit float multipliers
    def base_x(t):
        if mirror:
            return R * torch.cos(3.0 * t + phi_x)
        else:
            return -R * torch.cos(3.0 * t + phi_x)
    
    def base_y(t):
        if mirror:
            return R * torch.cos(2.0 * t + phi_y)
        else:
            return -R * torch.cos(2.0 * t + phi_y)
    
    def base_z(t):
        if mirror:
            return R * torch.cos(5 * t + phi_z)
        else:
            return - R * torch.cos(5 * t + phi_z)

    # 2. Build the Knot configuration
    knot = Knot()
    knot.make_parametrization(base_x, base_y, base_z)
    
    # 3. Maintain your metadata
    knot.config = {
        'kind': 'stevedore', 
        'kwargs': {'R': R, 'mirror': mirror}
    }
    
    return knot


def get_square_knot(R=1.0):
    """Returns the square knot, parameterized as a Lissajous curve."""
    n_x = 3
    n_y = 5
    n_z = 7
    phi_x = 0.7
    phi_y = 1.0
    phi_z = 0.0

    # 1. Define base functions using 'def' and explicit float multipliers
    def base_x(t): 
        return R * torch.cos(n_x * t + phi_x)
    
    def base_y(t): 
        return R * torch.cos(n_y * t + phi_y)
    
    def base_z(t):
        return R * torch.cos(n_z * t + phi_z)

    # 2. Build the Knot configuration
    knot = Knot()
    knot.make_parametrization(base_x, base_y, base_z)
    
    # 3. Maintain your metadata
    knot.config = {
        'kind': 'square', 
        'kwargs': {'R': R}
    }
    
    return knot


def get_three_twist_knot(R=1.0, mirror = False):
    """Returns the three-twist knot, parameterized as a Lissajous curve."""
    n_x = 3
    n_y = 2
    n_z = 7
    phi_x = 0.7
    phi_y = 0.2
    phi_z = 0.0

    # 1. Define base functions using 'def' and explicit float multipliers
    def base_x(t):
        if mirror:
            return R * torch.cos(n_x * t + phi_x)
        else:
            return -R * torch.cos(n_x * t + phi_x)
    
    def base_y(t):
        if mirror:
            return R * torch.cos(n_y * t + phi_y)
        else:
            return -R * torch.cos(n_y * t + phi_y)
    
    def base_z(t):
        if mirror:
            return R * torch.cos(n_z * t + phi_z)
        else:
            return -R * torch.cos(n_z * t + phi_z)

    # 2. Build the Knot configuration
    knot = Knot()
    knot.make_parametrization(base_x, base_y, base_z)
    
    # 3. Maintain your metadata
    knot.config = {
        'kind': 'three_twist', 
        'kwargs': {'R': R, 'mirror': mirror}
    }
    
    return knot


def get_lissajous_knot(                       
    n_x,
    n_y,
    n_z,
    phi_x,
    phi_y,
    phi_z,
    R=1.0):
    """Returns a Lissajous knot with the given frequencies and phase shifts."""

    # 1. Define base functions using 'def' and explicit float multipliers
    def base_x(t): 
        return R * torch.cos(n_x * t + phi_x)
    
    def base_y(t): 
        return R * torch.cos(n_y * t + phi_y)
    
    def base_z(t): 
        return R * torch.cos(n_z * t + phi_z)

    # 2. Build the Knot configuration
    knot = Knot()
    knot.make_parametrization(base_x, base_y, base_z)
    
    # 3. Maintain your metadata
    knot.config = {
        'kind': 'lissajous', 
        'kwargs': {
            'R': R,
            'n_x': n_x,
            'n_y': n_y,
            'n_z': n_z,
            'phi_x': phi_x,
            'phi_y': phi_y,
            'phi_z': phi_z
        }
    }
    
    return knot