import torch
import copy

class Curve2D():
    def __init__(self):
        self.parametrized = False

    def make_parametrization(self, x, y):
        self.parametrized = True

        self.x = x
        self.y = y

    def perturb(self, perturbation_matrix=None, N=4, scale=0.1, seed=None, dtype=torch.float32, device='cpu'):
        if perturbation_matrix is None:
            if seed is not None:
                torch.manual_seed(seed)

            # Random Fourier coefficients for x and y coordinates
            # Shape: (N,) for cos and (N,) for sin
            Ax = torch.randn(N, device=device, dtype=dtype) * scale
            Ay = torch.randn(N, device=device, dtype=dtype) * scale

            Bx = torch.randn(N, device=device, dtype=dtype) * scale
            By = torch.randn(N, device=device, dtype=dtype) * scale
        else:
            Ax, Ay, Bx, By = perturbation_matrix
            N = perturbation_matrix.shape[1]

        ns = torch.arange(1, N+1, device=device, dtype=dtype)  # frequencies

        def px(t):
            nt = t[..., None] * ns          # (..., N)
            return (torch.cos(nt) * Ax + torch.sin(nt) * Bx).sum(dim=-1)

        def py(t):
            nt = t[..., None] * ns
            return (torch.cos(nt) * Ay + torch.sin(nt) * By).sum(dim=-1)
        
        x = copy.deepcopy(self.x)
        y = copy.deepcopy(self.y)

        self.x = lambda theta: (x(theta) + px(theta)).to(device)
        self.y = lambda theta: (y(theta) + py(theta)).to(device)

        self.perturbation_matrix = torch.stack([Ax, Ay, Bx, By])

    def __call__(self, th):
        return torch.stack([self.x(th), self.y(th)], dim=-1)
    
# ------------------
# 1. Standard Circle (The 2D equivalent of the unknot)
circle = Curve2D()
circle.make_parametrization(
    x = lambda th: torch.cos(th),
    y = lambda th: torch.sin(th)
)

# ------------------
# 2. Ellipse
def get_ellipse(a=2.0, b=1.0):
    """
    A standard ellipse. 'a' and 'b' are the semi-major and semi-minor axes.
    Always a simple closed curve as long as a > 0 and b > 0.
    """
    ellipse = Curve2D()
    ellipse.make_parametrization(
        x = lambda th: a * torch.cos(th),
        y = lambda th: b * torch.sin(th)
    )
    return ellipse

# ------------------
# 3. Peanut Curve / Dumbbell shape
def get_peanut(R=2.0, r=0.8):
    """
    A polar curve shaped like a peanut or bowtie.
    Condition for simple closed curve without origin intersection: R > r.
    Condition for no self-intersection loops: R > 3*r (roughly, to keep it strictly convex-ish or softly pinched).
    """
    peanut = Curve2D()
    
    # Radius function r(theta) = R + r * cos(2 * theta)
    def radius(th):
        return R + r * torch.cos(2 * th)
        
    peanut.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return peanut

# ------------------
# 4. Star / Squircle Shape
def get_star(R=2.0, r=0.3, petals=5):
    """
    A wobbly, star-like shape. 
    To ensure it remains a simple closed curve without self-intersecting loops,
    the perturbation amplitude 'r' must be sufficiently small relative to 'R' and 'petals'.
    """
    star = Curve2D()
    
    def radius(th):
        return R + r * torch.cos(petals * th)
        
    star.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return star

# ------------------
# 5. Off-Center Limaçon (Simple variant)
def get_limacon(a=2.0, b=1.0):
    """
    A limaçon without an inner loop. 
    To remain a simple closed curve, we must enforce a >= 2b.
    If a < b, it will self-intersect and form an inner loop.
    """
    limacon = Curve2D()
    
    def radius(th):
        return a + b * torch.cos(th)
        
    limacon.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return limacon


# ------------------
# 6. Spherical Oscillator 
def get_spherical_oscillator(freq=6, amp=0.8):
    """
    A simple closed curve that oscillates between the Northern and Southern 
    hemispheres of S^2. 
    In R^2, it crosses the unit circle (the equator) 2*freq times.
    Keep amp < 1.0 to ensure it does not cross the origin (South Pole).
    """
    osc = Curve2D()
    
    # Base radius 1.0 corresponds to the equator on S^2.
    def radius(th):
        return 1.0 + amp * torch.sin(freq * th)
        
    osc.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return osc

# ------------------
# 7. Multi-Winding Curve (Self-Intersecting)
def get_multi_winding_curve(p=3, q=2, R=1.5, r=0.5):
    """
    Winds 'p' times around the origin while oscillating 'q' times.
    This creates a beautiful rosette that self-intersects.
    Because p=3, the winding number around the Z-axis of S^2 is 3.
    """
    winding = Curve2D()
    
    # Notice we scale theta by 'p' inside the trigonometric functions for x and y,
    # but we need the parameter domain to remain [0, 2*pi] to close the loop perfectly.
    # So we don't multiply the overall angle by p, we just trace it faster.
    def radius(th):
        return R + r * torch.cos(q * th)
        
    winding.make_parametrization(
        x = lambda th: radius(th) * torch.cos(p * th),
        y = lambda th: radius(th) * torch.sin(p * th)
    )
    return winding

# ------------------
# 8. Lissajous Curve
def get_lissajous(a=3, b=2, scale_x=1.5, scale_y=1.5, phase=torch.pi/2):
    """
    A Lissajous knot mapped to 2D. 
    Creates overlapping figure-eight style lobes.
    """
    lissajous = Curve2D()
    
    lissajous.make_parametrization(
        x = lambda th: scale_x * torch.sin(a * th + phase),
        y = lambda th: scale_y * torch.sin(b * th)
    )
    return lissajous

def get_superellipse(n=4.0, a=2.0, b=2.0):
    """
    A superellipse. As n increases past 2, the shape approaches a rectangle.
    Allows testing of nearly-sharp corners on the boundary.
    """
    super_e = Curve2D()
    
    # We use torch.sign and absolute values to handle fractional powers of negative numbers
    def x_param(th):
        return a * torch.sign(torch.cos(th)) * torch.pow(torch.abs(torch.cos(th)), 2.0 / n)
        
    def y_param(th):
        return b * torch.sign(torch.sin(th)) * torch.pow(torch.abs(torch.sin(th)), 2.0 / n)
        
    super_e.make_parametrization(x=x_param, y=y_param)
    return super_e


def get_cassini_oval(a=1.0, b=1.05):
    """
    Cassini oval. 
    Keep b > a for a simple closed curve. 
    As b approaches a, the curve pinches infinitely close to the origin.
    """
    cassini = Curve2D()
    
    def radius(th):
        # Polar equation: r^2 = a^2 cos(2t) + sqrt(b^4 - a^4 sin^2(2t))
        term1 = a**2 * torch.cos(2 * th)
        term2 = torch.sqrt(b**4 - a**4 * torch.sin(2 * th)**2)
        return torch.sqrt(term1 + term2)
        
    cassini.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return cassini

def get_gear(R=2.0, amp=0.5, teeth=6, sharpness=3.0):
    """
    A gear-like curve with flattened crests and troughs.
    'sharpness' controls how close the teeth are to square waves.
    """
    gear = Curve2D()
    
    def radius(th):
        return R + amp * torch.tanh(sharpness * torch.sin(teeth * th))
        
    gear.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return gear


def get_asymmetric_airfoil(cx=-0.1, cy=0.1, R=1.1):
    """
    Applies a complex mapping z -> z + 1/z to a displaced circle 
    to create an asymmetric, airfoil-like boundary.
    """
    airfoil = Curve2D()
    
    # Base circle parameterized in the complex plane
    def z_real(th):
        return cx + R * torch.cos(th)
    def z_imag(th):
        return cy + R * torch.sin(th)
        
    def x_param(th):
        zr = z_real(th)
        zi = z_imag(th)
        r2 = zr**2 + zi**2
        # Real part of z + 1/z
        return zr + zr / r2
        
    def y_param(th):
        zr = z_real(th)
        zi = z_imag(th)
        r2 = zr**2 + zi**2
        # Imaginary part of z + 1/z
        return zi - zi / r2
        
    airfoil.make_parametrization(x=x_param, y=y_param)
    return airfoil


# 1. The Rounded Triangle (Trefoil-ish boundary)
def get_rounded_triangle(a=2.0, b=0.4):
    """
    Creates a smooth, 3-fold symmetric shape resembling a Reuleaux triangle.
    To ensure it remains a simple closed curve without cusps or loops, 
    keep 'a' significantly larger than 'b' (a > 9b is perfectly safe, 
    but a > 3b usually prevents inner loops in the normal vector).
    """
    tri = Curve2D()
    
    def radius(th):
        return a + b * torch.cos(3 * th)
        
    tri.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return tri

# ------------------
# 2. The Pear / Teardrop
def get_pear(a=2.0, b=0.8):
    """
    An asymmetric, smooth teardrop or pear shape.
    It uses a mixed frequency perturbation in the radial direction.
    As long as a > b, r(theta) remains positive and the curve is simple.
    """
    pear = Curve2D()
    
    def radius(th):
        # Adds an asymmetric bulge
        return a + b * torch.sin(th) * torch.cos(th)**2
        
    pear.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return pear

# ------------------
# 3. Squircular Epitrochoid (Rounded Square)
def get_epitrochoid_square(R=1.0, a=0.15):
    """
    A parametric rounded square generated via epitrochoid kinematics.
    Unlike the superellipse, this is constructed from pure trigonometric 
    polynomials, which can have nicer analytic properties for minimal surfaces.
    To prevent self-intersection (loops), we MUST have a < R/3.
    """
    sq = Curve2D()
    
    sq.make_parametrization(
        x = lambda th: R * torch.cos(th) + a * torch.cos(3 * th),
        y = lambda th: R * torch.sin(th) - a * torch.sin(3 * th)
    )
    return sq

# ------------------
# 4. The Amoeba / Asymmetric Blob
def get_amoeba():
    """
    A delightfully wobbly, highly asymmetric simple closed curve.
    By summing multiple low-frequency Fourier terms but keeping their 
    total amplitude smaller than the base radius, it guarantees 
    no self-intersections.
    """
    amoeba = Curve2D()
    
    def radius(th):
        return 2.0 + 0.3 * torch.cos(th) + 0.5 * torch.sin(2 * th) - 0.2 * torch.cos(3 * th)
        
    amoeba.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return amoeba

# ------------------
# 5. The Bicuspid / Bowtie (Non-intersecting)
def get_bicuspid(a=2.0, b=1.0):
    """
    A smooth, gently pinched shape similar to a peanut but defined 
    via parametric coordinate scaling rather than purely radial.
    Creates a nice 'soft' pinching effect.
    """
    bowtie = Curve2D()
    
    bowtie.make_parametrization(
        x = lambda th: a * torch.cos(th),
        # Scales the y-coordinate based on the x-coordinate's phase
        y = lambda th: b * torch.sin(th) * (1.5 + torch.cos(2 * th))
    )
    return bowtie

# ------------------
# 1. The Buzzsaw (Phase-Modulated Star)
def get_buzzsaw(R=2.0, amp=0.4, teeth=7, skew=0.12):
    """
    Unlike a normal star where the points radiate straight out from the origin,
    this curve modulates the angular phase, causing the teeth to "lean" or whip 
    forward like a circular saw blade. 
    To guarantee no self-intersections, we must ensure: 1 - skew * teeth > 0.
    """
    saw = Curve2D()
    
    def radius(th):
        return R + amp * torch.cos(teeth * th)
        
    def phase(th):
        # Shifts the angle to lean the geometry
        return th + skew * torch.sin(teeth * th)
        
    saw.make_parametrization(
        x = lambda th: radius(th) * torch.cos(phase(th)),
        y = lambda th: radius(th) * torch.sin(phase(th))
    )
    return saw

# ------------------
# 2. The Gielis Leaf (Superformula)
def get_gielis_leaf(m=12, n1=0.5, n2=1.5, n3=4.0):
    """
    Based on Johan Gielis's superformula, which unifies many natural and 
    biological shapes. By using vastly different n2 and n3 exponents, 
    we create a highly organic, asymmetrical lobe structure.
    To ensure it closes perfectly at 2*pi, 'm' must be a multiple of 4.
    """
    leaf = Curve2D()
    
    def radius(th):
        # The m/4 factor is why m must be a multiple of 4 to close at 2*pi
        t = (m / 4.0) * th
        term1 = torch.pow(torch.abs(torch.cos(t)), n2)
        term2 = torch.pow(torch.abs(torch.sin(t)), n3)
        return torch.pow(term1 + term2, -1.0 / n1)
        
    leaf.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return leaf

# ------------------
# 3. The Ruffled Ellipse (Two-Scale Boundary)
def get_ruffled_ellipse(a=3.0, b=1.0, ripples=14, amp=0.15):
    """
    A highly eccentric ellipse with a high-frequency, low-amplitude ripple.
    This is mathematically fascinating for minimal surfaces because high-frequency 
    boundary oscillations decay exponentially fast as the surface dips into the bulk.
    """
    ruffle = Curve2D()
    
    def x_base(th): return a * torch.cos(th)
    def y_base(th): return b * torch.sin(th)
    
    # We add an orthogonal high-frequency ripple
    ruffle.make_parametrization(
        x = lambda th: x_base(th) + amp * torch.cos(ripples * th) * torch.cos(th),
        y = lambda th: y_base(th) + amp * torch.cos(ripples * th) * torch.sin(th)
    )
    return ruffle

# ------------------
# 4. The Embryo (Asymmetric Conformal Approximation)
def get_embryo():
    """
    Simulates a complex conformal map $f(z) = z + c_1 z^2 + c_2 z^3$ applied 
    to the unit circle. This creates a deeply concave, kidney/embryo-like shape 
    that is incredibly smooth but globally asymmetric.
    """
    embryo = Curve2D()
    
    embryo.make_parametrization(
        # The specific coefficients guarantee univalence (no crossing)
        x = lambda th: 2.0 * torch.cos(th) - 0.4 * torch.cos(2 * th) + 0.1 * torch.cos(3 * th),
        y = lambda th: 2.0 * torch.sin(th) + 0.4 * torch.sin(2 * th) + 0.1 * torch.sin(3 * th)
    )
    return embryo

# 1. The Spiral Galaxy (Twisted Star)
def get_spiral_galaxy(R=2.0, amp=0.8, arms=3, twist=1.5):
    """
    Takes a standard multi-lobed star and applies a radial twist. 
    The lobes get swept out into beautiful, non-intersecting spiral arms.
    Increase 'twist' to wind the arms tighter.
    """
    galaxy = Curve2D()
    
    # Base radius of a star
    def r_base(th): 
        return R + amp * torch.cos(arms * th)
        
    # The twist diffeomorphism
    def th_twist(th): 
        return th + twist * r_base(th)
        
    galaxy.make_parametrization(
        x = lambda th: r_base(th) * torch.cos(th_twist(th)),
        y = lambda th: r_base(th) * torch.sin(th_twist(th))
    )
    return galaxy

# ------------------
# 2. The S-Scroll / Yin-Yang Boundary
def get_s_scroll(R=2.0, r=1.2, twist=2.0):
    """
    Takes a 2-lobed peanut/dumbbell and twists it into a sweeping 'S' shape.
    This creates deep, spiraling channels that will force the minimal 
    surface in H^3 to form a stunning, twisting saddle.
    """
    scroll = Curve2D()
    
    # Base radius of a peanut
    def r_base(th): 
        return R + r * torch.cos(2 * th)
        
    def th_twist(th): 
        return th + twist * r_base(th)
        
    scroll.make_parametrization(
        x = lambda th: r_base(th) * torch.cos(th_twist(th)),
        y = lambda th: r_base(th) * torch.sin(th_twist(th))
    )
    return scroll

# ------------------
# 3. The Conch Shell (Asymmetric Spiral)
def get_conch(a=2.0, b=1.0, twist=2.5):
    """
    Takes an asymmetrical teardrop/pear and twists it heavily. 
    Because the base shape is unbalanced, the resulting spiral resembles 
    the cross-section of a nautilus or conch shell, coiling tightly on 
    one side and sweeping wide on the other.
    """
    conch = Curve2D()
    
    # Asymmetric base radius
    def r_base(th): 
        return a + b * torch.sin(th) * torch.cos(th)**2
        
    def th_twist(th): 
        return th + twist * r_base(th)
        
    conch.make_parametrization(
        x = lambda th: r_base(th) * torch.cos(th_twist(th)),
        y = lambda th: r_base(th) * torch.sin(th_twist(th))
    )
    return conch


def get_hurricane(R=2.0, amp=1.5, arms=4, twist_power=3.0, twist_scale=0.15):
    """
    A wildly coiling spiral. The outer edges get wrapped exponentially tighter 
    than the inner valleys.
    Condition for no self-intersection: R > amp (so radius stays > 0).
    """
    hurricane = Curve2D()
    
    def r_base(th):
        return R + amp * torch.cos(arms * th)
        
    def th_twist(th):
        # The phase shift grows non-linearly with the radius
        return th + twist_scale * torch.pow(r_base(th), twist_power)
        
    hurricane.make_parametrization(
        x = lambda th: r_base(th) * torch.cos(th_twist(th)),
        y = lambda th: r_base(th) * torch.sin(th_twist(th))
    )
    return hurricane


def get_splash(R=3.0, amp=2.5, spikes=6):
    """
    Creates sharp, isolated outward spikes with wide, flat valleys in between,
    mimicking a milk droplet splash. 
    Because it remains a simple polar function r(theta) > 0, it will never intersect.
    """
    splash = Curve2D()
    
    def radius(th):
        # torch.exp(cos(th)) ranges from 1/e to e. 
        # We shift it so the valleys rest near R.
        base_exp = torch.exp(torch.cos(spikes * th))
        return R + amp * (base_exp - 0.367) / 2.35  # Normalized scaling
        
    splash.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return splash

def get_fm_ripple(R=4.0, amp=0.8, carrier_freq=15, mod_freq=3, mod_index=2.5):
    """
    An FM-modulated boundary. The ripples bunch up tightly and then stretch out.
    carrier_freq = total number of base ripples.
    mod_freq = how many times they bunch up/stretch out around the circle.
    mod_index = how severe the bunching is.
    """
    fm = Curve2D()
    
    def radius(th):
        # The phase itself is modulated by a lower-frequency sine wave
        phase = carrier_freq * th + mod_index * torch.sin(mod_freq * th)
        return R + amp * torch.cos(phase)
        
    fm.make_parametrization(
        x = lambda th: radius(th) * torch.cos(th),
        y = lambda th: radius(th) * torch.sin(th)
    )
    return fm