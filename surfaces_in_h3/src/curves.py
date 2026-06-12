import torch

# ------------------------
# Curve generator function
# ------------------------
def build_curve(
    curve_type,
    curve_kwargs,
    curve_perturbation_matrix=None
):
    match curve_type:
        case 'circle':
            curve = get_circle(**curve_kwargs)
        case 'ellipse':
            curve = get_ellipse(**curve_kwargs)
        case 'peanut':
            curve = get_peanut(**curve_kwargs)
        case 'star':
            curve = get_star(**curve_kwargs)
        case 'limacon':
            curve = get_limacon(**curve_kwargs)
        case 'spherical_oscillator':
            curve = get_spherical_oscillator(**curve_kwargs)
        case 'multi_winding':
            curve = get_multi_winding_curve(**curve_kwargs)
        case 'lissajous':
            curve = get_lissajous(**curve_kwargs)
        case 'superellipse':
            curve = get_superellipse(**curve_kwargs)
        case 'cassini_oval':
            curve = get_cassini_oval(**curve_kwargs)
        case 'gear':
            curve = get_gear(**curve_kwargs)
        case 'asymmetric_airfoil':
            curve = get_asymmetric_airfoil(**curve_kwargs)
        case 'rounded_triangle':
            curve = get_rounded_triangle(**curve_kwargs)
        case 'pear':
            curve = get_pear(**curve_kwargs)
        case 'epitrochoid_square':
            curve = get_epitrochoid_square(**curve_kwargs)
        case 'amoeba':
            curve = get_amoeba(**curve_kwargs)
        case 'bicuspid':
            curve = get_bicuspid(**curve_kwargs)
        case 'buzzsaw':
            curve = get_buzzsaw(**curve_kwargs)
        case 'gielis_leaf':
            curve = get_gielis_leaf(**curve_kwargs)
        case 'ruffled_ellipse':
            curve = get_ruffled_ellipse(**curve_kwargs)
        case 'embryo':
            curve = get_embryo(**curve_kwargs)
        case 'spiral_galaxy':
            curve = get_spiral_galaxy(**curve_kwargs)
        case 's_scroll':
            curve = get_s_scroll(**curve_kwargs)
        case 'conch':
            curve = get_conch(**curve_kwargs)
        case 'hurricane':
            curve = get_hurricane(**curve_kwargs)
        case 'splash':
            curve = get_splash(**curve_kwargs)
        case 'fm_ripple':
            curve = get_fm_ripple(**curve_kwargs)
        case _:
            raise NotImplementedError(f'Curve type "{curve_type}" not implemented.')
    
    if curve_perturbation_matrix is not None:
        curve.perturb(curve_perturbation_matrix)

    return curve


# ------------------------
# Helper functions
# ------------------------
def generate_curve_perturbation_matrix(
        N=4,
        scale=0.1,
        seed=None,
        dtype=torch.float64,
        device='cpu'
):
    """
    Generates a 4xN perturbation matrix.
    Rows: [Ax, Ay, Bx, By]
    """
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(4, N, device=device, dtype=dtype) * scale


# ------------------------
# CurveEvaluator Class
# ------------------------
class CurveEvaluator(torch.nn.Module):
    """
    A pure PyTorch module representing the mathematical state of the 2D curve.
    Optimized for torch.compile().
    """
    def __init__(self, base_x, base_y, perturbation_matrix=None):
        super().__init__()
        self.base_x = base_x
        self.base_y = base_y
        
        if perturbation_matrix is not None:
            self.register_buffer("P", perturbation_matrix)
            self.register_buffer("ns", torch.arange(1, perturbation_matrix.shape[1] + 1))
        else:
            self.P = None

    def forward(self, t):
        # 1. Evaluate base parametrizations
        x = self.base_x(t)
        y = self.base_y(t)
        
        # 2. Add perturbations if they exist
        if self.P is not None:
            nt = t * self.ns 
            cost = torch.cos(nt)
            sint = torch.sin(nt)
            
            # self.P shape is (4, N_freq). P[0]=Ax, P[1]=Ay, P[2]=Bx, P[3]=By
            px = (cost * self.P[0] + sint * self.P[2]).sum(dim=-1, keepdim=True)
            py = (cost * self.P[1] + sint * self.P[3]).sum(dim=-1, keepdim=True)
            
            x = x + px
            y = y + py
            
        # 3. Concatenate along the last dimension to get (N, 2)
        return torch.cat([x, y], dim=-1)


# ------------------------
# Curve2D Class
# ------------------------
class Curve2D():
    def __init__(self):
        self.name = "Curve2D"
        self.perturbation_matrix = None
        self.base_x = None
        self.base_y = None

    def make_parametrization(self, x, y):
        self.parametrized = True
        self.base_x = x
        self.base_y = y

    def perturb(self, perturbation_matrix):
        self.perturbation_matrix = perturbation_matrix

    def get_evaluator(self):
        return CurveEvaluator(
            self.base_x, 
            self.base_y, 
            self.perturbation_matrix
        )
    

# ------------------------
# Implemented curves
# ------------------------

def get_circle(R=1.0):
    def base_x(th): return R * torch.cos(th)
    def base_y(th): return R * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'circle', 'kwargs': {'R': R}}
    return curve

def get_ellipse(a=2.0, b=1.0):
    def base_x(th): return a * torch.cos(th)
    def base_y(th): return b * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'ellipse', 'kwargs': {'a': a, 'b': b}}
    return curve

def get_peanut(R=2.0, r=0.8):
    def radius(th): return R + r * torch.cos(2.0 * th)
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'peanut', 'kwargs': {'R': R, 'r': r}}
    return curve

def get_star(R=2.0, r=0.3, petals=5.0):
    def radius(th): return R + r * torch.cos(petals * th)
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'star', 'kwargs': {'R': R, 'r': r, 'petals': petals}}
    return curve

def get_limacon(a=2.0, b=1.0):
    def radius(th): return a + b * torch.cos(th)
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'limacon', 'kwargs': {'a': a, 'b': b}}
    return curve

def get_spherical_oscillator(freq=6.0, amp=0.8):
    def radius(th): return 1.0 + amp * torch.sin(freq * th)
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'spherical_oscillator', 'kwargs': {'freq': freq, 'amp': amp}}
    return curve

def get_multi_winding_curve(p=3.0, q=2.0, R=1.5, r=0.5):
    def radius(th): return R + r * torch.cos(q * th)
    def base_x(th): return radius(th) * torch.cos(p * th)
    def base_y(th): return radius(th) * torch.sin(p * th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'multi_winding', 'kwargs': {'p': p, 'q': q, 'R': R, 'r': r}}
    return curve

def get_lissajous(a=3.0, b=2.0, scale_x=1.5, scale_y=1.5, phase=torch.pi/2.0):
    def base_x(th): return scale_x * torch.sin(a * th + phase)
    def base_y(th): return scale_y * torch.sin(b * th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'lissajous', 'kwargs': {'a': a, 'b': b, 'scale_x': scale_x, 'scale_y': scale_y, 'phase': phase}}
    return curve

def get_superellipse(n=4.0, a=2.0, b=2.0):
    def base_x(th): return a * torch.sign(torch.cos(th)) * torch.pow(torch.abs(torch.cos(th)), 2.0 / n)
    def base_y(th): return b * torch.sign(torch.sin(th)) * torch.pow(torch.abs(torch.sin(th)), 2.0 / n)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'superellipse', 'kwargs': {'n': n, 'a': a, 'b': b}}
    return curve

def get_cassini_oval(a=1.0, b=1.05):
    def radius(th):
        term1 = a**2 * torch.cos(2.0 * th)
        term2 = torch.sqrt(b**4 - a**4 * torch.sin(2.0 * th)**2)
        return torch.sqrt(term1 + term2)
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'cassini_oval', 'kwargs': {'a': a, 'b': b}}
    return curve

def get_gear(R=2.0, amp=0.5, teeth=6.0, sharpness=3.0):
    def radius(th): return R + amp * torch.tanh(sharpness * torch.sin(teeth * th))
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'gear', 'kwargs': {'R': R, 'amp': amp, 'teeth': teeth, 'sharpness': sharpness}}
    return curve

def get_asymmetric_airfoil(cx=-0.1, cy=0.1, R=1.1):
    def z_real(th): return cx + R * torch.cos(th)
    def z_imag(th): return cy + R * torch.sin(th)
    
    def base_x(th):
        zr, zi = z_real(th), z_imag(th)
        r2 = zr**2 + zi**2
        return zr + zr / r2
        
    def base_y(th):
        zr, zi = z_real(th), z_imag(th)
        r2 = zr**2 + zi**2
        return zi - zi / r2
        
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'asymmetric_airfoil', 'kwargs': {'cx': cx, 'cy': cy, 'R': R}}
    return curve

def get_rounded_triangle(a=2.0, b=0.4):
    def radius(th): return a + b * torch.cos(3.0 * th)
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'rounded_triangle', 'kwargs': {'a': a, 'b': b}}
    return curve

def get_pear(a=2.0, b=0.8):
    def radius(th): return a + b * torch.sin(th) * torch.cos(th)**2
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'pear', 'kwargs': {'a': a, 'b': b}}
    return curve

def get_epitrochoid_square(R=1.0, a=0.15):
    def base_x(th): return R * torch.cos(th) + a * torch.cos(3.0 * th)
    def base_y(th): return R * torch.sin(th) - a * torch.sin(3.0 * th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'epitrochoid_square', 'kwargs': {'R': R, 'a': a}}
    return curve

def get_amoeba():
    def radius(th): return 2.0 + 0.3 * torch.cos(th) + 0.5 * torch.sin(2.0 * th) - 0.2 * torch.cos(3.0 * th)
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'amoeba', 'kwargs': {}}
    return curve

def get_bicuspid(a=2.0, b=1.0):
    def base_x(th): return a * torch.cos(th)
    def base_y(th): return b * torch.sin(th) * (1.5 + torch.cos(2.0 * th))
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'bicuspid', 'kwargs': {'a': a, 'b': b}}
    return curve

def get_buzzsaw(R=2.0, amp=0.4, teeth=7.0, skew=0.12):
    def radius(th): return R + amp * torch.cos(teeth * th)
    def phase(th): return th + skew * torch.sin(teeth * th)
    
    def base_x(th): return radius(th) * torch.cos(phase(th))
    def base_y(th): return radius(th) * torch.sin(phase(th))
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'buzzsaw', 'kwargs': {'R': R, 'amp': amp, 'teeth': teeth, 'skew': skew}}
    return curve

def get_gielis_leaf(m=12.0, n1=0.5, n2=1.5, n3=4.0):
    def radius(th):
        t = (m / 4.0) * th
        term1 = torch.pow(torch.abs(torch.cos(t)), n2)
        term2 = torch.pow(torch.abs(torch.sin(t)), n3)
        return torch.pow(term1 + term2, -1.0 / n1)
        
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'gielis_leaf', 'kwargs': {'m': m, 'n1': n1, 'n2': n2, 'n3': n3}}
    return curve

def get_ruffled_ellipse(a=3.0, b=1.0, ripples=14.0, amp=0.15):
    def base_x(th): return a * torch.cos(th) + amp * torch.cos(ripples * th) * torch.cos(th)
    def base_y(th): return b * torch.sin(th) + amp * torch.cos(ripples * th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'ruffled_ellipse', 'kwargs': {'a': a, 'b': b, 'ripples': ripples, 'amp': amp}}
    return curve

def get_embryo():
    def base_x(th): return 2.0 * torch.cos(th) - 0.4 * torch.cos(2.0 * th) + 0.1 * torch.cos(3.0 * th)
    def base_y(th): return 2.0 * torch.sin(th) + 0.4 * torch.sin(2.0 * th) + 0.1 * torch.sin(3.0 * th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'embryo', 'kwargs': {}}
    return curve

def get_spiral_galaxy(R=2.0, amp=0.8, arms=3.0, twist=1.5):
    def r_base(th): return R + amp * torch.cos(arms * th)
    def th_twist(th): return th + twist * r_base(th)
    
    def base_x(th): return r_base(th) * torch.cos(th_twist(th))
    def base_y(th): return r_base(th) * torch.sin(th_twist(th))
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'spiral_galaxy', 'kwargs': {'R': R, 'amp': amp, 'arms': arms, 'twist': twist}}
    return curve

def get_s_scroll(R=2.0, r=1.2, twist=2.0):
    def r_base(th): return R + r * torch.cos(2.0 * th)
    def th_twist(th): return th + twist * r_base(th)
    
    def base_x(th): return r_base(th) * torch.cos(th_twist(th))
    def base_y(th): return r_base(th) * torch.sin(th_twist(th))
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 's_scroll', 'kwargs': {'R': R, 'r': r, 'twist': twist}}
    return curve

def get_conch(a=2.0, b=1.0, twist=2.5):
    def r_base(th): return a + b * torch.sin(th) * torch.cos(th)**2
    def th_twist(th): return th + twist * r_base(th)
    
    def base_x(th): return r_base(th) * torch.cos(th_twist(th))
    def base_y(th): return r_base(th) * torch.sin(th_twist(th))
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'conch', 'kwargs': {'a': a, 'b': b, 'twist': twist}}
    return curve

def get_hurricane(R=2.0, amp=1.5, arms=4.0, twist_power=3.0, twist_scale=0.15):
    def r_base(th): return R + amp * torch.cos(arms * th)
    def th_twist(th): return th + twist_scale * torch.pow(r_base(th), twist_power)
    
    def base_x(th): return r_base(th) * torch.cos(th_twist(th))
    def base_y(th): return r_base(th) * torch.sin(th_twist(th))
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'hurricane', 'kwargs': {'R': R, 'amp': amp, 'arms': arms, 'twist_power': twist_power, 'twist_scale': twist_scale}}
    return curve

def get_splash(R=3.0, amp=2.5, spikes=6.0):
    def radius(th):
        base_exp = torch.exp(torch.cos(spikes * th))
        return R + amp * (base_exp - 0.367) / 2.35 
        
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'splash', 'kwargs': {'R': R, 'amp': amp, 'spikes': spikes}}
    return curve

def get_fm_ripple(R=4.0, amp=0.8, carrier_freq=15.0, mod_freq=3.0, mod_index=2.5):
    def radius(th):
        phase = carrier_freq * th + mod_index * torch.sin(mod_freq * th)
        return R + amp * torch.cos(phase)
        
    def base_x(th): return radius(th) * torch.cos(th)
    def base_y(th): return radius(th) * torch.sin(th)
    
    curve = Curve2D()
    curve.make_parametrization(base_x, base_y)
    curve.config = {'kind': 'fm_ripple', 'kwargs': {'R': R, 'amp': amp, 'carrier_freq': carrier_freq, 'mod_freq': mod_freq, 'mod_index': mod_index}}
    return curve