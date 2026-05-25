import torch
import numpy as np
import matplotlib.pyplot as plt

from samplers import MixSampler

from geometry import PDE
from geometry_alt import minimal_in_H3_PDE

sample_unit_disc = MixSampler()

def plot_curve_and_projection(curve_obj, n_points=2000, r2_lim=(-5, 5)):
    if not curve_obj.parametrized:
        raise NameError("Not parametrized")
    
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=True)
    
    # Evaluate 2D curve
    curve_2d = curve_obj(torch.tensor(theta, dtype=torch.float32)).cpu().numpy()
    x, y = curve_2d[:,0], curve_2d[:,1]
    
    # Map R^2 -> S^2 (Boundary isometry: Upper Half-Space -> Poincaré Ball)
    r2 = x**2 + y**2
    denom = r2 + 1
    u = 2 * x / denom
    v = 2 * y / denom
    w = (r2 - 1) / denom
    
    # Set up figure with two subplots
    fig = plt.figure(figsize=(6, 12))
    
    # --- Subplot 1: 2D Curve ---
    ax1 = fig.add_subplot(211)
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.set_aspect('equal')
    ax1.set_xlim(r2_lim)
    ax1.set_ylim(r2_lim)
    ax1.set_title("Curve in R^2 (Half-Space Boundary)")
    ax1.grid(True)
    
    # --- Subplot 2: 3D Curve on S^2 ---
    ax2 = fig.add_subplot(212, projection='3d')
    
    # Plot the sphere for visual context
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u_sphere), np.sin(v_sphere))
    ys = np.outer(np.sin(u_sphere), np.sin(v_sphere))
    zs = np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    
    # Plot sphere surface with low opacity
    ax2.plot_surface(xs, ys, zs, color='lightgray', alpha=0.2, edgecolor='none')
    
    # Plot the projected curve
    ax2.plot(u, v, w, 'r-', linewidth=2.5)
    
    # Keep aspect ratio geometric so the sphere isn't distorted
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim((-1.1, 1.1))
    ax2.set_ylim((-1.1, 1.1))
    ax2.set_zlim((-1.1, 1.1))
    ax2.set_title("Curve mapped to S^2 (Poincaré Ball Boundary)")
    
    plt.tight_layout()
    plt.show()

# Example usage (assuming you defined the 'get_star' curve from earlier):
# star = get_star(R=2.0, r=0.5, petals=4)
# plot_curve_and_projection(star)

# original versions
def plot_model(
        model,
        vmin,
        vmax,
        grid_size = 500,
        ):
    def get_component(xy_array, model, component):
        xy_tensor = torch.from_numpy(xy_array.astype(np.float32))  # shape (M,2)
        with torch.no_grad():  # avoid gradient computation
            out = model(xy_tensor)[:, component]  # shape (M,)
        return out.numpy()  # return as NumPy array

    # Create a dense grid
    grid_size = grid_size
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)

    # Mask points outside the disc
    mask = grid_x**2 + grid_y**2 <= 1

    # Prepare points inside the disc
    points = np.column_stack([grid_x[mask], grid_y[mask]])

    # Prepare the plot
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    extent = (-1, 1, -1, 1)
    vmin=vmin
    vmax=vmax

    # Prepare the components
    components = []
    grid = {}
    for i in range(4):
        grid[i] = np.full_like(grid_x, np.nan, dtype=float)
        grid[i][mask] = get_component(points, model, i)
        components.append(grid[i])

    # Plot them
    for ax, comp in zip(axes.flat, components):
        im = ax.imshow(
            comp.T,
            extent=extent,
            origin='lower',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_aspect('equal')

    # A single colorbar shared among all panels:
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label='value')

    fig.suptitle('Gradient Maps of the components on the Disc')
    plt.show()


def plot_error(
        model,
        vmin = None,
        vmax = None,
        loss = lambda t: (t**2).sum(dim=-1),
        grid_size = 500
):
    
    def err(xy_array, model):
        xy_tensor = torch.from_numpy(xy_array.astype(np.float32))  # shape (M,2)
        xy_tensor.requires_grad_()
        u = model(xy_tensor)  # shape (M,)
        t = PDE(u, xy_tensor) # shape (N, 4)
        loss = (t**2).sum(dim=-1)
        return loss.detach().numpy()  # return as NumPy array

    # Create a dense grid
    grid_size = grid_size
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)

    # Mask points outside the disc
    mask = grid_x**2 + grid_y**2 <= 1

    # Prepare points inside the disc
    points = np.column_stack([grid_x[mask], grid_y[mask]])

    # Evaluate rho on all points at once
    grid_err = np.full_like(grid_x, np.nan, dtype=float)
    grid_err[mask] = err(points, model)

    # Plot
    plt.figure(figsize=(4,4))
    plt.imshow(grid_err.T, extent=(-1,1,-1,1), origin='lower', cmap='viridis', vmin = vmin, vmax = vmax)
    plt.colorbar(label='value')
    plt.gca().set_aspect('equal')
    plt.title('Error')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def montecarlo_error(
        model,
        loss_fcn,
        num_samples = 100,
        size_samples = 2000,
        vmax = np.inf
        ):
    errors = []
    k = 1
    for i in range(num_samples):
        if i == int(k/10 * num_samples)-1:
            print(f'{k*10}% done...')
            k += 1
        xy = sample_unit_disc(size_samples)
        xy.requires_grad_()

        u = model(xy)
        t = PDE(u, xy) # shape (N, 4)
        loss = loss_fcn(t)
        if loss < vmax:
            errors.append(loss.detach().numpy())
    
    plt.figure()
    plt.hist(errors, bins=30)
    plt.title(f'Distribution of MSEs for uniform samples of size = {size_samples}')
    plt.show()

    mean_err = np.mean(errors)
    std_err = np.std(errors)

    print(f"Mean error: {mean_err:.6e}")
    print(f"Standard deviation of error: {std_err:.6e}")


def plot_H3_surfaces(
    model_A, model_B,
    grid_size_A=500,
    min_r_A = 0,
    max_r_A = 1,
    min_theta_A = 0,
    max_theta_A = 2*np.pi,
    alpha_A = 1,
    grid_size_B=500,
    min_r_B = 0,
    max_r_B = 1,
    min_theta_B = 0,
    max_theta_B = 2*np.pi,
    alpha_B = 1
):
    def get_eval(xy_array, model):
        xy_tensor = torch.from_numpy(xy_array.astype(np.float32))
        with torch.no_grad():
            out = model(xy_tensor)
        return out.cpu().numpy()

    # ----- Grid A -----
    nrA = grid_size_A
    ntA = grid_size_A
    rA = np.linspace(min_r_A, max_r_A, nrA)
    tA = np.linspace(min_theta_A, max_theta_A, ntA)
    RA, TA = np.meshgrid(rA, tA)
    XA = RA * np.cos(TA)
    YA = RA * np.sin(TA)
    pointsA = np.column_stack([XA.ravel(), YA.ravel()])

    # ----- Grid B -----
    nrB = grid_size_B
    ntB = grid_size_B
    rB = np.linspace(min_r_B, max_r_B, nrB)
    tB = np.linspace(min_theta_B, max_theta_B, ntB)
    RB, TB = np.meshgrid(rB, tB)
    XB = RB * np.cos(TB)
    YB = RB * np.sin(TB)
    pointsB = np.column_stack([XB.ravel(), YB.ravel()])

    # Evaluate both models (Output indices: 0 is rho, 1 is x, 2 is y)
    outA = get_eval(pointsA, model_A)
    outB = get_eval(pointsB, model_B)

    # Reshape coordinates for Model A
    rho_A = outA[:, 0].reshape(ntA, nrA)
    x_A = outA[:, 1].reshape(ntA, nrA)
    y_A = outA[:, 2].reshape(ntA, nrA)

    # Reshape coordinates for Model B
    rho_B = outB[:, 0].reshape(ntB, nrB)
    x_B = outB[:, 1].reshape(ntB, nrB)
    y_B = outB[:, 2].reshape(ntB, nrB)

    # Poincaré Ball Mapping helper
    def to_poincare(x, y, rho):
        denom = x**2 + y**2 + (rho + 1)**2
        u = 2 * x / denom
        v = 2 * y / denom
        w = (x**2 + y**2 + rho**2 - 1) / denom
        return u, v, w

    u_A, v_A, w_A = to_poincare(x_A, y_A, rho_A)
    u_B, v_B, w_B = to_poincare(x_B, y_B, rho_B)

    # ----- Subplots -----
    fig = plt.figure(figsize=(8, 14))

    # --- 1. Half-Space Model ---
    ax1 = fig.add_subplot(2, 1, 1, projection="3d")
    ax1.set_title("Half-Space Model (x, y, ρ)")
    
    # Model A Surface
    ax1.plot_surface(x_A, y_A, rho_A, cmap='coolwarm', edgecolor='none', alpha=alpha_A)
    # Model B Surface
    ax1.plot_surface(x_B, y_B, rho_B, cmap='viridis', edgecolor='none', alpha=alpha_B)
    
    # Boundary Curve (Soft dark gray, depth-sorted naturally)
    ax1.plot(x_A[:, -1], y_A[:, -1], rho_A[:, -1], color='#2F4F4F', linewidth=2.5)
    
    # Calculate physical dimensions to enforce an equal aspect ratio dynamically
    dx = max(np.ptp(x_A), np.ptp(x_B))
    dy = max(np.ptp(y_A), np.ptp(y_B))
    dz = max(np.ptp(rho_A), np.ptp(rho_B))
    # Safeguard against 0-height at initialization
    ax1.set_box_aspect((max(dx, 1e-5), max(dy, 1e-5), max(dz, 1e-5))) 
    
    ax1.set_zlim(bottom=0)
    ax1.view_init(elev=20, azim=-45)

    # --- 2. Poincaré Ball Model ---
    ax2 = fig.add_subplot(2, 1, 2, projection="3d")
    ax2.set_title("Poincaré Ball Model (u, v, w)")

    # Plot faint bounding sphere (S^2 boundary)
    u_sph = np.linspace(0, 2 * np.pi, 100)
    v_sph = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u_sph), np.sin(v_sph))
    ys = np.outer(np.sin(u_sph), np.sin(v_sph))
    zs = np.outer(np.ones(np.size(u_sph)), np.cos(v_sph))
    ax2.plot_surface(xs, ys, zs, color='lightgray', alpha=0.1, edgecolor='none')

    # Model A Surface
    ax2.plot_surface(u_A, v_A, w_A, cmap='coolwarm', edgecolor='none', alpha=alpha_A)
    # Model B Surface
    ax2.plot_surface(u_B, v_B, w_B, cmap='viridis', edgecolor='none', alpha=alpha_B)

    # Boundary Curve mapped to S^2
    ax2.plot(u_A[:, -1], v_A[:, -1], w_A[:, -1], color='#2F4F4F', linewidth=2.5)

    ax2.set_box_aspect((1, 1, 1))
    ax2.set_xlim((-1.1, 1.1))
    ax2.set_ylim((-1.1, 1.1))
    ax2.set_zlim((-1.1, 1.1))
    ax2.view_init(elev=20, azim=-45)

    plt.tight_layout()
    plt.show()


# batched versions
def plot_model_batched(model, vmin, vmax, grid_size=500, batch_size=5000):
    """
    Plot model components by processing grid points in batches to avoid memory issues.
    """
    def get_component_batched(xy_array, model, component, batch_size):
        """Compute model output in batches"""
        n_points = len(xy_array)
        results = []
        
        with torch.no_grad():
            for i in range(0, n_points, batch_size):
                batch = xy_array[i:i+batch_size]
                xy_tensor = torch.from_numpy(batch.astype(np.float32))
                out = model(xy_tensor)[:, component]
                results.append(out.numpy())
        
        return np.concatenate(results)
    
    # Create a dense grid
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Mask points outside the disc
    mask = grid_x**2 + grid_y**2 <= 1
    
    # Prepare points inside the disc
    points = np.column_stack([grid_x[mask], grid_y[mask]])
    print(f"Processing {len(points)} points in batches of {batch_size}...")
    
    # Prepare the plot
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    extent = (-1, 1, -1, 1)
    
    # Prepare and plot the components
    for idx in range(4):
        grid_comp = np.full_like(grid_x, np.nan, dtype=float)
        grid_comp[mask] = get_component_batched(points, model, idx, batch_size)
        
        ax = axes.flat[idx]
        im = ax.imshow(
            grid_comp.T,
            extent=extent,
            origin='lower',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_aspect('equal')
    
    # A single colorbar shared among all panels
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label='value')
    fig.suptitle('Gradient Maps of the components on the Disc')
    plt.show()


def plot_error_batched(model, vmin=None, vmax=None, grid_size=500, batch_size=2000):
    """
    Plot PDE error by processing grid points in batches to avoid memory issues.
    
    Note: batch_size is smaller here because PDE computation requires gradients
    and second-order derivatives, which consume more memory.
    """
    def compute_error_batched(xy_array, model, batch_size):
        """Compute PDE error in batches"""
        n_points = len(xy_array)
        results = []
        
        for i in range(0, n_points, batch_size):
            batch = xy_array[i:i+batch_size]
            xy_tensor = torch.from_numpy(batch.astype(np.float32))
            xy_tensor.requires_grad_(True)
            
            # Compute PDE residual
            u = model(xy_tensor)
            t = PDE(u, xy_tensor)
            loss = (t**2).sum(dim=-1)
            
            results.append(loss.detach().numpy())
            
            # Clear computation graph for this batch
            del xy_tensor, u, t, loss
        
        return np.concatenate(results)
    
    # Create a dense grid
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Mask points outside the disc
    mask = grid_x**2 + grid_y**2 <= 1
    
    # Prepare points inside the disc
    points = np.column_stack([grid_x[mask], grid_y[mask]])
    print(f"Processing {len(points)} points in batches of {batch_size}...")
    print("(This may take a while due to PDE computation with second derivatives)")
    
    # Compute error
    grid_err = np.full_like(grid_x, np.nan, dtype=float)
    grid_err[mask] = compute_error_batched(points, model, batch_size)
    
    # Plot
    plt.figure(figsize=(4, 4))
    plt.imshow(grid_err.T, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='value')
    plt.gca().set_aspect('equal')
    plt.title('Error')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_four_poincare_models(
    models_dict, 
    colormap='viridis', 
    grid_size=300,
    show_axes=False,
    show_grid=False,
    show_ticks=False,
    wspace=0.0,  
    hspace=0.0,  
    zoom=1.35     
):
    """
    Plots four models in a 2x2 grid, maximizing the 3D plots and eliminating whitespace and titles.
    """
    def get_eval(xy_array, model):
        xy_tensor = torch.from_numpy(xy_array.astype(np.float32))
        with torch.no_grad():
            out = model(xy_tensor)
        return out.cpu().numpy()

    def to_poincare(x, y, rho):
        denom = x**2 + y**2 + (rho + 1)**2
        u = 2 * x / denom
        v = 2 * y / denom
        w = (x**2 + y**2 + rho**2 - 1) / denom
        return u, v, w

    r = np.linspace(0, 1, grid_size)
    t = np.linspace(0, 2*np.pi, grid_size)
    R, T = np.meshgrid(r, t)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    points = np.column_stack([X.ravel(), Y.ravel()])

    u_sph = np.linspace(0, 2 * np.pi, 100)
    v_sph = np.linspace(0, np.pi, 100)
    x_sph = np.outer(np.cos(u_sph), np.sin(v_sph))
    y_sph = np.outer(np.sin(u_sph), np.sin(v_sph))
    z_sph = np.outer(np.ones(np.size(u_sph)), np.cos(v_sph))

    fig = plt.figure(figsize=(10, 10))
    
    # We still unpack 'title' from the dict items, but we just won't use it
    for idx, (title, model) in enumerate(models_dict.items(), 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")

        out = get_eval(points, model)
        rho = out[:, 0].reshape(grid_size, grid_size)
        x_m = out[:, 1].reshape(grid_size, grid_size)
        y_m = out[:, 2].reshape(grid_size, grid_size)

        u, v, w = to_poincare(x_m, y_m, rho)

        ax.plot_surface(x_sph, y_sph, z_sph, color='lightgray', alpha=0.1, edgecolor='none')
        ax.plot_surface(u, v, w, cmap=colormap, edgecolor='none', alpha=1.0)
        ax.plot(u[:, -1], v[:, -1], w[:, -1], color='#2F4F4F', linewidth=2.5)

        try:
            ax.set_box_aspect((1, 1, 1), zoom=zoom)
        except TypeError:
            ax.set_box_aspect((1, 1, 1))
            
        ax.set_xlim((-1.1, 1.1))
        ax.set_ylim((-1.1, 1.1))
        ax.set_zlim((-1.1, 1.1))
        ax.view_init(elev=20, azim=-45)

        if not show_axes:
            ax.set_axis_off()
        else:
            if not show_grid:
                ax.grid(False)
            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

    plt.subplots_adjust(
        left=0.0, 
        right=1.0, 
        bottom=0.0, 
        top=1.0, 
        wspace=wspace, 
        hspace=hspace
    )
    plt.show()


def plot_single_poincare_model(
    model, 
    title=None,
    colormap='viridis', 
    grid_size=300,
    show_axes=False,
    show_grid=False,
    show_ticks=False,
    zoom=1.35     
):
    """
    Plots a single model in a 3D Poincaré ball, maximizing the plot space.
    """
    def get_eval(xy_array, model):
        xy_tensor = torch.from_numpy(xy_array.astype(np.float32))
        with torch.no_grad():
            out = model(xy_tensor)
        return out.cpu().numpy()

    def to_poincare(x, y, rho):
        denom = x**2 + y**2 + (rho + 1)**2
        u = 2 * x / denom
        v = 2 * y / denom
        w = (x**2 + y**2 + rho**2 - 1) / denom
        return u, v, w

    # Generate grid
    r = np.linspace(0, 1, grid_size)
    t = np.linspace(0, 2*np.pi, grid_size)
    R, T = np.meshgrid(r, t)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    points = np.column_stack([X.ravel(), Y.ravel()])

    # Generate background sphere boundary
    u_sph = np.linspace(0, 2 * np.pi, 100)
    v_sph = np.linspace(0, np.pi, 100)
    x_sph = np.outer(np.cos(u_sph), np.sin(v_sph))
    y_sph = np.outer(np.sin(u_sph), np.sin(v_sph))
    z_sph = np.outer(np.ones(np.size(u_sph)), np.cos(v_sph))

    # Single plot setup
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Evaluate model
    out = get_eval(points, model)
    rho = out[:, 0].reshape(grid_size, grid_size)
    x_m = out[:, 1].reshape(grid_size, grid_size)
    y_m = out[:, 2].reshape(grid_size, grid_size)

    u, v, w = to_poincare(x_m, y_m, rho)

    # Plot surfaces
    ax.plot_surface(x_sph, y_sph, z_sph, color='lightgray', alpha=0.1, edgecolor='none')
    ax.plot_surface(u, v, w, cmap=colormap, edgecolor='none', alpha=1.0)
    ax.plot(u[:, -1], v[:, -1], w[:, -1], color='#2F4F4F', linewidth=2.5)

    # Formatting
    try:
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
        
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_zlim((-1.1, 1.1))
    ax.view_init(elev=20, azim=-45)

    if title:
        ax.set_title(title)

    if not show_axes:
        ax.set_axis_off()
    else:
        if not show_grid:
            ax.grid(False)
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

    # Tight boundaries to eliminate whitespace
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    plt.show()