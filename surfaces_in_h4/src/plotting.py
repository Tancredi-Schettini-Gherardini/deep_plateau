import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import font_scalings
from src.samplers import MixSampler

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