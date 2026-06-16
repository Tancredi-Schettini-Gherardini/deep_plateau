import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.font_manager import font_scalings
from src.samplers import MixSampler


# --- GENERAL PLOTTING FUNCTIONS
def plot_error(
        compiled_pde,
        device='cpu',
        dtype=torch.float64,
        vmin=None,
        vmax=None,
        loss_fn=lambda t: (t**2).sum(dim=-1),
        grid_size=500,
        batch_size=10000,
        block=True,
        # --- Graphical Parameters ---
        figsize=(7, 6),
        cmap_name='viridis',
        nan_color='white',
        boundary_color='black',
        boundary_linewidth=1.5,
        colorbar_label='PDE Error',
        aspect='equal',
        xlim=(-1.05, 1.05),
        ylim=(-1.05, 1.05),
        axis_style='off',
        title='PDE Error Distribution'
):
    # 1. Create the grid natively in PyTorch
    x = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    y = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

    # 2. Mask points outside the disc
    mask = (grid_x**2 + grid_y**2) <= 1

    # 3. Prepare points inside the disc -> Shape: (M, 2)
    points = torch.stack([grid_x[mask], grid_y[mask]], dim=-1)

    # 4. Evaluate in batches
    errors = []
    
    for i in range(0, points.shape[0], batch_size):
        # No requires_grad needed: the PDE evaluator differentiates analytically.
        batch_pts = points[i:i+batch_size]
        
        # Evaluate PDE directly (the model evaluation is baked into this call)
        t = compiled_pde(batch_pts)
        
        # Calculate loss
        batch_loss = loss_fn(t)
        
        errors.append(batch_loss.detach().cpu())

    # 5. Reassemble the batch errors
    errors = torch.cat(errors, dim=0)

    # 6. Map the errors back onto the 2D grid structure
    grid_err = torch.full_like(grid_x.cpu(), float('nan'))
    grid_err[mask.cpu()] = errors

    # 7. Plot
    cmap = plt.colormaps[cmap_name].with_extremes(bad=nan_color)
    
    plt.figure(figsize=figsize)
    im = plt.imshow(
        grid_err.numpy(),
        extent=(-1, 1, -1, 1),
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # Clip the imshow to the disc so the boundary is smooth (analytical
    # circle), not stair-cased pixels. Draw the boundary itself separately.
    ax = plt.gca()
    im.set_clip_path(plt.Circle((0, 0), 1.0, transform=ax.transData))
    
    # Draw boundary if a linewidth is provided
    if boundary_linewidth > 0:
        ax.add_patch(plt.Circle(
            (0, 0), 1.0, 
            fill=False, 
            color=boundary_color, 
            linewidth=boundary_linewidth
        ))

    # Add a colorbar only when a label is provided.
    if colorbar_label is not None:
        plt.colorbar(im, label=colorbar_label)

    # Apply axis settings
    ax.set_aspect(aspect)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis(axis_style)
    
    # Apply title
    if title:
        plt.title(title)
        
    plt.tight_layout()

    if block:
        plt.show()
    else:
        plt.show(block=False)

def montecarlo_error(
        compiled_pde,
        device='cpu',
        dtype=torch.float64,
        num_samples=100,
        size_samples=2000,
        vmax=np.inf,
        mix=1,
        bias=0.5,
        # --- Upgraded Graphical Parameters ---
        figsize=(8, 5),              # Slightly wider looks better for distributions
        dpi=150,                     # Crisper resolution
        bins=30,
        hist_color='#4C72B0',        # A pleasant, muted deep blue
        hist_alpha=0.85,             # Slight transparency
        edgecolor='white',           # Separates the bins cleanly
        title=None,
        xlabel='Mean Squared Error',
        ylabel='Frequency',
        show_stats=True,             
        stats_pos='upper right',     
        block=True
):
    errors = []
    k = 1
    # Build the disc sampler for the Monte Carlo draws.
    sampler = MixSampler(mix=mix, bias=bias)
    for i in range(num_samples):
        if i == int(k/10 * num_samples)-1:
            print(f'{k*10}% done...')
            k += 1
        xy = sampler(size_samples).to(device=device, dtype=dtype)
        xy.requires_grad_()

        t = compiled_pde(xy) 
        loss = (t ** 2).sum(dim=-1).mean()
        if loss < vmax:
            errors.append(loss.detach().cpu().numpy())
            
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    
    # Apply DPI
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Add a soft grid BEFORE plotting the histogram so it sits behind the bars
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    ax.hist(
        errors, 
        bins=bins, 
        color=hist_color, 
        alpha=hist_alpha, 
        edgecolor=edgecolor,
        zorder=3  # Puts the bars in front of the grid
    )
    
    # --- Clean up the "Spines" (the bounding box) ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
        
    if title:
        ax.set_title(title, pad=15, fontweight='bold')
        
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=10)

    if show_stats:
        stats_text = f"Mean: {mean_err:.6e}\nStd:  {std_err:.6e}"
        
        if isinstance(stats_pos, str):
            pos_map = {
                'upper right': (0.95, 0.95, 'right', 'top'),
                'upper left':  (0.05, 0.95, 'left', 'top'),
                'lower right': (0.95, 0.05, 'right', 'bottom'),
                'lower left':  (0.05, 0.05, 'left', 'bottom'),
                'center':      (0.50, 0.50, 'center', 'center')
            }
            x, y, ha, va = pos_map.get(stats_pos.lower(), (0.95, 0.95, 'right', 'top'))
        elif isinstance(stats_pos, (list, tuple)) and len(stats_pos) == 2:
            x, y = stats_pos
            ha, va = 'left', 'bottom'
        else:
            x, y, ha, va = 0.95, 0.95, 'right', 'top'

        # Softer stats box
        bbox_props = dict(boxstyle='round,pad=0.6', facecolor='#FAFAFA', alpha=0.9, edgecolor='#DDDDDD')
        ax.text(x, y, stats_text, transform=ax.transAxes, 
                fontsize=10, family='monospace', verticalalignment=va, horizontalalignment=ha, bbox=bbox_props)

    plt.tight_layout()

    if block:
        plt.show()
    else:
        plt.show(block=False)

    print(f"Mean error: {mean_err:.6e}")
    print(f"Standard deviation of error: {std_err:.6e}")

# --- SPECIFIC H4 PLOTTING FUNCTIONS
def plot_knot(
        evaluator,           # Now takes the compiled module
        n_points=2000,
        device='cpu',        # Added so you can match the evaluator's device
        dtype=torch.float64,
        xlim=(-2, 2),
        ylim=(-2, 2),
        zlim=(-2, 2),
        block=True,):
    
    # Sample theta on [0, 2*pi] as an (N, 1) column for the evaluator.
    theta = torch.linspace(0, 2*np.pi, n_points, device=device, dtype=dtype).view(-1, 1)
    
    # Evaluate the knot without tracking gradients.
    with torch.no_grad():
        # Move to CPU and convert to NumPy for plotting.
        curve = evaluator(theta).cpu().numpy()
        
    x, y, z = curve[:, 0], curve[:, 1], curve[:, 2]

    # Plot the curve in 3D.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    if block:
        plt.show()
    else:
        plt.show(block=False)

def plot_mu_heatmap_log(
    u_callable, epsilon,
    candidates=None, refined_pairs=None, jacobians=None,
    ax=None,  # optional Axes to draw into (e.g. for subplots)
    # --- Titles & Text Labels ---
    title='default', 
    colorbar_label=r'$\log_{10}(\mu_{\varepsilon}(p))$',
    legend_label_candidates='Candidate pairs',
    legend_label_refined_pos=r'Refined ($\det J > 0$)',
    legend_label_refined_neg=r'Refined ($\det J < 0$)',
    legend_label_refined_neutral='Refined pairs',
    # --- Palette / Core Style ---
    cmap='viridis_r', 
    cmap_bad_color='white',
    candidate_color='#3a3a3a',
    refined_pos_color='black',
    refined_neg_color='white',
    refined_neutral_color='black',
    scatter_facecolor='none',
    show_colorbar=True,
    alpha=0.4,
    # --- Compute Settings ---
    grid_resolution=200, 
    batch_size=2000, 
    dtype=torch.float64, 
    device='cpu',
    domain_radius=1.0,
    # --- Figure / Axes Settings ---
    figsize=(7, 6),
    axes_limit=1.05,
    axes_aspect='equal',
    axes_adjustable='box',
    axes_axis_state='off',
    imshow_origin='lower',
    boundary_color='black', 
    boundary_linewidth=3.5,
    boundary_fill=False,
    # --- Candidate Pair Styling ---
    candidate_scatter_s=20, 
    candidate_scatter_lw=0.5, 
    candidate_scatter_zorder=4,
    candidate_line_ls='--', 
    candidate_line_lw=0.6, 
    candidate_line_zorder=3,
    # --- Refined Pair Styling ---
    refined_scatter_s=45, 
    refined_scatter_lw=1.0, 
    refined_scatter_zorder=6,
    refined_line_ls='--', 
    refined_line_lw=1.0, 
    refined_line_zorder=5,
    # --- Legend Settings ---
    legend_title=None, 
    legend_loc='upper center', 
    legend_bbox_to_anchor=(0.5, -0.02),
    legend_ncols=1,
    legend_numpoints=2, 
    legend_frameon=False,
    legend_marker='o',
    legend_candidate_ms=7.2,
    legend_candidate_alpha=0.7,
    legend_refined_ms=9.6
):
    """
    Plots the log-scale heatmap of mu_epsilon(p) over the unit disc D^2.
    Accepts an optional 'ax' parameter to plot within a subplot.
    """
    # Ensure device is properly instantiated if passed as string
    if isinstance(device, str):
        device = torch.device(device)

    # 1. Create a 2D grid bounding the disc
    x = torch.linspace(-domain_radius, domain_radius, grid_resolution, dtype=dtype, device=device)
    y = torch.linspace(-domain_radius, domain_radius, grid_resolution, dtype=dtype, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # 2. Isolate points inside the defined domain radius
    norms = torch.norm(points, dim=1)
    disc_mask = norms <= domain_radius
    p = points[disc_mask]
    M = p.shape[0]

    # 3. Evaluate u(p)
    with torch.no_grad():
        u_p = u_callable(p)

    # 4. Compute mu_epsilon batched
    mu_vals = torch.empty(M, dtype=dtype, device=device)

    for i in range(0, M, batch_size):
        p_batch = p[i : i + batch_size]
        u_batch = u_p[i : i + batch_size]

        dist_p = torch.cdist(p_batch, p)
        dist_u = torch.cdist(u_batch, u_p)

        dist_u[dist_p <= epsilon] = float('inf')

        batch_min_vals, _ = torch.min(dist_u, dim=1)
        mu_vals[i : i + batch_size] = batch_min_vals

    mu_vals[torch.isinf(mu_vals)] = float('nan')

    # 5. Reconstruct 2D grid
    heatmap = torch.full((grid_resolution * grid_resolution,), float('nan'), dtype=dtype, device=device)
    heatmap[disc_mask] = mu_vals
    heatmap = heatmap.view(grid_resolution, grid_resolution).cpu().numpy()
    heatmap_log = np.log10(heatmap)

    # 6. Plotting Setup (Handle subplots)
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        fig = ax.figure # Grab the figure associated with the provided ax

    cmap_obj = plt.colormaps[cmap].with_extremes(bad=cmap_bad_color)

    im = ax.imshow(
        heatmap_log,
        origin=imshow_origin,
        extent=[-domain_radius, domain_radius, -domain_radius, domain_radius],
        cmap=cmap_obj,
    )

    # Clip the imshow to the disc so the boundary is smooth
    im.set_clip_path(plt.Circle((0, 0), domain_radius, transform=ax.transData))
    ax.add_patch(plt.Circle((0, 0), domain_radius, fill=boundary_fill, color=boundary_color, linewidth=boundary_linewidth))

    # Attach colorbar specifically to this axis
    if show_colorbar:
        fig.colorbar(im, ax=ax, label=colorbar_label)

    # --- Overlay candidate pairs ---
    if candidates is not None:
        for p_tensor, q_tensor in candidates:
            p_np = p_tensor.cpu().numpy()
            q_np = q_tensor.cpu().numpy()
            px, py = p_np[0], p_np[1]
            qx, qy = q_np[0], q_np[1]

            ax.scatter([px, qx], [py, qy],
                       facecolors=scatter_facecolor,
                       edgecolors=candidate_color,
                       s=candidate_scatter_s,
                       linewidths=candidate_scatter_lw,
                       alpha=alpha,
                       zorder=candidate_scatter_zorder)
            ax.plot([px, qx], [py, qy], 
                    color=candidate_color, 
                    linestyle=candidate_line_ls, 
                    linewidth=candidate_line_lw, 
                    alpha=alpha, 
                    zorder=candidate_line_zorder)

    # --- Overlay refined pairs ---
    use_signs = (refined_pairs is not None
                 and jacobians is not None
                 and len(jacobians) == len(refined_pairs))
    
    if refined_pairs is not None:
        for i, (p_tensor, q_tensor) in enumerate(refined_pairs):
            p_np = p_tensor.cpu().numpy()
            q_np = q_tensor.cpu().numpy()
            px, py = p_np[0], p_np[1]
            qx, qy = q_np[0], q_np[1]

            if use_signs:
                color = refined_pos_color if jacobians[i] > 0 else refined_neg_color
                ax.scatter([px, qx], [py, qy],
                           facecolor=scatter_facecolor,
                           edgecolor=color,
                           s=refined_scatter_s,
                           linewidth=refined_scatter_lw,
                           zorder=refined_scatter_zorder)
                ax.plot([px, qx], [py, qy], 
                        color=color, 
                        linestyle=refined_line_ls, 
                        linewidth=refined_line_lw, 
                        zorder=refined_line_zorder)
            else:
                ax.scatter([px, qx], [py, qy], 
                           facecolor=scatter_facecolor, 
                           edgecolor=refined_neutral_color, 
                           s=refined_scatter_s, 
                           linewidth=refined_scatter_lw, 
                           zorder=refined_scatter_zorder)
                ax.plot([px, qx], [py, qy], 
                        color=refined_neutral_color, 
                        linestyle=refined_line_ls, 
                        linewidth=refined_line_lw, 
                        zorder=refined_line_zorder)

    # --- Legend ---
    legend_handles = []
    if candidates is not None:
        legend_handles.append(Line2D(
            [0], [0],
            marker=legend_marker, markerfacecolor=scatter_facecolor, markeredgecolor=candidate_color,
            markeredgewidth=candidate_scatter_lw * 1.2, markersize=legend_candidate_ms,
            color=candidate_color, linestyle=candidate_line_ls, linewidth=candidate_line_lw * 1.2, alpha=legend_candidate_alpha,
            label=legend_label_candidates,
        ))
    if refined_pairs is not None:
        if use_signs:
            has_pos = any(j > 0 for j in jacobians)
            has_neg = any(j < 0 for j in jacobians)
            
            if has_pos:
                legend_handles.append(Line2D(
                    [0], [0],
                    marker=legend_marker,
                    markerfacecolor=scatter_facecolor,
                    markeredgecolor=refined_pos_color,
                    markeredgewidth=refined_scatter_lw * 1.2, markersize=legend_refined_ms,
                    color=refined_pos_color, linestyle=refined_line_ls, linewidth=refined_line_lw * 1.2,
                    label=legend_label_refined_pos,
                ))
            if has_neg:
                legend_handles.append(Line2D(
                    [0], [0],
                    marker=legend_marker,
                    markerfacecolor=scatter_facecolor,
                    markeredgecolor=refined_neg_color,
                    markeredgewidth=refined_scatter_lw * 1.2, markersize=legend_refined_ms,
                    color=refined_neg_color, linestyle=refined_line_ls, linewidth=refined_line_lw * 1.2,
                    label=legend_label_refined_neg,
                ))
        else:
            legend_handles.append(Line2D(
                [0], [0],
                marker=legend_marker,
                markerfacecolor=scatter_facecolor,
                markeredgecolor=refined_neutral_color,
                markeredgewidth=refined_scatter_lw * 1.2, markersize=legend_refined_ms,
                color=refined_neutral_color, linestyle=refined_line_ls, linewidth=refined_line_lw * 1.2,
                label=legend_label_refined_neutral,
            ))
            
    if legend_handles:
        # 1.2x the standard title font size for both the legend entries and its title.
        _ts = plt.rcParams['axes.titlesize']
        if isinstance(_ts, str):
            _ts = font_scalings.get(_ts, 1.0) * plt.rcParams['font.size']
        _legend_fs = 1.2 * _ts
        ax.legend(
            handles=legend_handles,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            ncols=legend_ncols,
            numpoints=legend_numpoints,
            frameon=legend_frameon,
            fontsize=_legend_fs,
            title=legend_title,
            title_fontsize=_legend_fs
        )

    # Handle title logic properly
    if title == 'default':
        ax.set_title(fr'Log Heatmap of $\mu_{{\varepsilon}}(p)$ for $\varepsilon = {epsilon}$')
    elif title is not None:
        ax.set_title(title)
    
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.set_aspect(axes_aspect, adjustable=axes_adjustable)
    ax.axis(axes_axis_state)
    
    if show_plot:
        plt.tight_layout()
        plt.show()
        
    return ax

# --- SPECIFIC H3 PLOTTING FUNCTIONS
def plot_curve_and_projection(curve_obj, n_points=2000, r2_lim=(-5, 5)):
    # Check if the curve has been parametrized
    if not getattr(curve_obj, 'parametrized', False):
        raise NameError("Not parametrized")
    
    # Create theta as (n_points, 1) to ensure proper broadcasting for perturbations
    # and to guarantee torch.cat([x, y], dim=-1) produces an (N, 2) tensor.
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=True).reshape(-1, 1)
    t_tensor = torch.tensor(theta, dtype=torch.float32)
    
    # Extract the callable nn.Module evaluator
    evaluator = curve_obj.get_evaluator()
    
    # Evaluate 2D curve and safely detach to numpy
    with torch.no_grad():
        curve_2d = evaluator(t_tensor).cpu().numpy()
        
    x, y = curve_2d[:, 0], curve_2d[:, 1]
    
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
    
    # Fetch the curve name from config if available
    curve_name = getattr(curve_obj, 'config', {}).get('kind', 'Custom Curve')
    ax1.set_title(f"Curve in R^2 ({curve_name})")
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

def plot_H3_surfaces(
    model_A_trained, model_B_untrained,
    # --- Grid A Parameters ---
    grid_size_A=500,
    min_r_A=0,
    max_r_A=1,
    min_theta_A=0,
    max_theta_A=2*np.pi,
    alpha_A=1, # Surface A transparency
    # --- Grid B Parameters ---
    grid_size_B=500,
    min_r_B=0,
    max_r_B=1,
    min_theta_B=0,
    max_theta_B=2*np.pi,
    alpha_B=1, # Surface B transparency
    # --- Figure & Layout Parameters ---
    figsize=(8, 14),
    show_grid=True,
    show_axis=True,
    show_legend=True,
    legend_loc='upper right',
    elev=20,
    azim=-45,
    # --- Titles & Labels ---
    title_ax1="Half-Space Model (x, y, z)",
    title_ax2="Poincaré Ball Model (u, v, w)",
    xlabel="",
    ylabel="",
    zlabel="",
    # --- Model A Aesthetics ---
    cmap_A='coolwarm',
    edgecolor_A='none',
    label_A='Model A (Trained)',
    # --- Model B Aesthetics ---
    cmap_B='viridis',
    edgecolor_B='none',
    label_B='Model B (Untrained)',
    # --- Boundary Curve Aesthetics ---
    bound_color='#2F4F4F',
    bound_linewidth=2.5,
    bound_alpha=1.0,         # <--- NEW: Boundary curve transparency
    label_bound='Boundary Curve',
    # --- Bounding Sphere Aesthetics ---
    sphere_color='lightgray',
    sphere_alpha=0.1,
    label_sphere='Bounding Sphere',
    # --- Axis Limits ---
    ax1_zmin=0,
    ax2_xlim=(-1.1, 1.1),
    ax2_ylim=(-1.1, 1.1),
    ax2_zlim=(-1.1, 1.1)
):
    def get_eval(xy_array, model):
        # Dynamically map the input tensor to the model's device
        device = next(model.parameters()).device
        xy_tensor = torch.from_numpy(xy_array.astype(np.float32)).to(device)
        
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

    # Evaluate both models
    outA = get_eval(pointsA, model_A_trained)
    outB = get_eval(pointsB, model_B_untrained)

    # Reshape coordinates for Model A
    z_A = outA[:, 0].reshape(ntA, nrA)
    x_A = outA[:, 1].reshape(ntA, nrA)
    y_A = outA[:, 2].reshape(ntA, nrA)

    # Reshape coordinates for Model B
    z_B = outB[:, 0].reshape(ntB, nrB)
    x_B = outB[:, 1].reshape(ntB, nrB)
    y_B = outB[:, 2].reshape(ntB, nrB)

    # Poincaré Ball Mapping helper
    def to_poincare(x, y, z):
        denom = x**2 + y**2 + (z + 1)**2
        u = 2 * x / denom
        v = 2 * y / denom
        w = (x**2 + y**2 + z**2 - 1) / denom
        return u, v, w

    u_A, v_A, w_A = to_poincare(x_A, y_A, z_A)
    u_B, v_B, w_B = to_poincare(x_B, y_B, z_B)

    # ----- Subplots -----
    fig = plt.figure(figsize=figsize)

    # --- Create Proxy Artists for Legends ---
    patch_A = mpatches.Patch(color=plt.cm.get_cmap(cmap_A)(0.8), label=label_A, alpha=alpha_A)
    patch_B = mpatches.Patch(color=plt.cm.get_cmap(cmap_B)(0.5), label=label_B, alpha=alpha_B)
    # <--- ADDED: bound_alpha to the legend Line2D object
    line_bound = Line2D([0], [0], color=bound_color, linewidth=bound_linewidth, alpha=bound_alpha, label=label_bound) 
    
    base_handles = [patch_A, patch_B, line_bound]

    # --- Helper function to apply common axis styling ---
    def apply_axis_styling(ax, title):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.grid(show_grid)
        if not show_axis:
            ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)

    # --- 1. Half-Space Model ---
    ax1 = fig.add_subplot(2, 1, 1, projection="3d")
    apply_axis_styling(ax1, title_ax1)
    
    # Model A Surface
    ax1.plot_surface(x_A, y_A, z_A, cmap=cmap_A, edgecolor=edgecolor_A, alpha=alpha_A)
    # Model B Surface
    ax1.plot_surface(x_B, y_B, z_B, cmap=cmap_B, edgecolor=edgecolor_B, alpha=alpha_B)
    
    # Boundary Curve
    # <--- ADDED: alpha=bound_alpha
    ax1.plot(x_A[:, -1], y_A[:, -1], z_A[:, -1], color=bound_color, linewidth=bound_linewidth, alpha=bound_alpha)
    
    # Calculate physical dimensions to enforce an equal aspect ratio dynamically
    dx = max(np.ptp(x_A), np.ptp(x_B))
    dy = max(np.ptp(y_A), np.ptp(y_B))
    dz = max(np.ptp(z_A), np.ptp(z_B))
    ax1.set_box_aspect((max(dx, 1e-5), max(dy, 1e-5), max(dz, 1e-5))) 
    ax1.set_zlim(bottom=ax1_zmin)
    
    if show_legend:
        ax1.legend(handles=base_handles, loc=legend_loc)

    # --- 2. Poincaré Ball Model ---
    ax2 = fig.add_subplot(2, 1, 2, projection="3d")
    apply_axis_styling(ax2, title_ax2)

    # Plot faint bounding sphere (S^2 boundary)
    u_sph = np.linspace(0, 2 * np.pi, 100)
    v_sph = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u_sph), np.sin(v_sph))
    ys = np.outer(np.sin(u_sph), np.sin(v_sph))
    zs = np.outer(np.ones(np.size(u_sph)), np.cos(v_sph))
    ax2.plot_surface(xs, ys, zs, color=sphere_color, alpha=sphere_alpha, edgecolor='none')

    # Model A Surface
    ax2.plot_surface(u_A, v_A, w_A, cmap=cmap_A, edgecolor=edgecolor_A, alpha=alpha_A)
    # Model B Surface
    ax2.plot_surface(u_B, v_B, w_B, cmap=cmap_B, edgecolor=edgecolor_B, alpha=alpha_B)

    # Boundary Curve mapped to S^2
    # <--- ADDED: alpha=bound_alpha
    ax2.plot(u_A[:, -1], v_A[:, -1], w_A[:, -1], color=bound_color, linewidth=bound_linewidth, alpha=bound_alpha)

    ax2.set_box_aspect((1, 1, 1))
    ax2.set_xlim(ax2_xlim)
    ax2.set_ylim(ax2_ylim)
    ax2.set_zlim(ax2_zlim)

    if show_legend:
        patch_sphere = mpatches.Patch(color=sphere_color, label=label_sphere, alpha=0.3)
        ax2.legend(handles=base_handles + [patch_sphere], loc=legend_loc)

    plt.tight_layout()
    plt.show()