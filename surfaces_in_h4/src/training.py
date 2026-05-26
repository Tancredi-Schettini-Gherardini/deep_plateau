import torch
from src.geometry import minimal_in_H4_PDE_flat_new

import torch

def train_PINN_Adam(
    model,
    xy_grid,              
    epochs=10000,
    batch_size=2**10,       
    lr=1e-3,
    lr_min=1e-5,
    scheduler_type='plateau', # Choose 'plateau' or 'cosine'
    scheduler_patience=200,   
    scheduler_factor=0.8,     
    scheduler_threshold=1e-3, 
    verbose=True
):
    device = torch.device("cpu")
    dtype = model.kwargs['dtype']
    model.to(device=device, dtype=dtype)
    
    # Pre-load the entire grid into RAM to bypass DataLoader bottlenecks
    xy_grid = xy_grid.to(device=device, dtype=dtype)

    # 1. Build the compiled PDE-residual evaluator
    fast_pde = minimal_in_H4_PDE_flat_new(
        model, 
        use_compile=True, 
        compile_kwargs={"mode": "default", "fullgraph": True, "dynamic": False}
    )

    num_points = xy_grid.shape[0]
    batch_size = min(batch_size, num_points)
    num_batches = num_points // batch_size

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 2. Inject the parameterized tuning knobs into the chosen scheduler
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            threshold_mode='rel',    
            cooldown=50,             
            min_lr=lr_min,           
            eps=1e-08
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,  # Decays smoothly over the total epochs
            eta_min=lr_min # Bottoms out at your specified min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler_type: '{scheduler_type}'. Choose 'plateau' or 'cosine'.")

    history = []
    best_loss = torch.inf
    best_model_state = None

    for ep in range(epochs):
        # Direct memory slicing for batching
        indices = torch.randperm(num_points, device=device)
        epoch_loss_sum = 0.0

        for i in range(num_batches):
            # Slice the batch manually from the pre-loaded grid
            batch_xy = xy_grid[indices[i*batch_size : (i+1)*batch_size]]
            
            optimizer.zero_grad()
            
            # Evaluate the PDE residual on the batch
            residual = fast_pde(batch_xy)
            
            # Mean Squared Error
            loss = (residual ** 2).mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss_sum += loss.item()

        avg_epoch_loss = epoch_loss_sum / num_batches
        
        # 3. Step the scheduler correctly based on its type
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_epoch_loss)
        else:
            scheduler.step() # Cosine does not take a metric
            
        history.append(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            # Snapshot the best weights seen so far
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and ep % 10 == 0:
            # Pull the current LR directly from the optimizer
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {ep:5d}/{epochs} | Avg Loss: {avg_epoch_loss:.3e} | LR: {current_lr:.3e}")

    # Restore the weights of the best epoch before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_loss, history


def refine_PINN_lbfgs(
        model,
        xy_grid,              
        lr=1.0,
        max_iter=300,
        log_every=10,
        tolerance_grad=1e-12,     # L-BFGS gradient-norm stopping tolerance
        tolerance_change=1e-14    # L-BFGS loss-change stopping tolerance
    ):
    device = torch.device("cpu")
    dtype = model.kwargs['dtype']
    model.to(device=device, dtype=dtype)
    
    # The input grid needs no requires_grad: the flat PDE module computes all
    # spatial derivatives analytically (see mlp_uJH in geometry.py).
    xy_grid = xy_grid.to(device=device, dtype=dtype)

    # L-BFGS needs a static objective, so the full grid is used unchanged for
    # the entire refinement phase (no per-step reshuffling or resampling).
    fast_pde = minimal_in_H4_PDE_flat_new(
        model, 
        use_compile=True, 
        compile_kwargs={"mode": "default", "fullgraph": True, "dynamic": False}
    )

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,       # Applied strict gradient tolerance
        tolerance_change=tolerance_change,   # Applied strict loss change tolerance
        line_search_fn="strong_wolfe", 
        history_size=50,  
        max_eval=int(max_iter * 1.25)        # Caps the number of line search evaluations
    )

    history = []
    closure_calls = 0

    def closure():
        nonlocal closure_calls
        optimizer.zero_grad()
        
        residual = fast_pde(xy_grid)
        loss = (residual ** 2).mean()
        
        loss.backward()

        loss_val = loss.item()
        history.append(loss_val)
        closure_calls += 1

        if closure_calls % log_every == 0:
            print(f"L-BFGS Step {closure_calls:4d} | Loss: {loss_val:.3e}")

        return loss

    optimizer.step(closure)

    return history