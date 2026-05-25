import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math

from geometry_alt import *
from losses import L2_squared

def smooth_radius(x: torch.Tensor, x0=0.1) -> torch.Tensor:
    assert x0 < 1.0, "x0 must be less than 1"
    
    diff = x - x0
    mask = diff > 0
    safe_diff = torch.where(mask, diff, torch.ones_like(diff))
    norm_const = 1.0 / (1.0 - x0)
    exponent = norm_const - (1.0 / safe_diff)
    output = torch.exp(exponent)
    final_output = torch.where(mask, output, torch.zeros_like(output))
    
    return final_output

# ------------------------------------------------------------------------------------
class MainModel(nn.Module):
    def __init__(
            self,
            curve,
            interior_model,
            sampler,
            reparametrization_model = None,
            extension_method = 'stereographic',
            smooth_extension = False,
            treshold = None
            ):
        super().__init__()
        self.curve = curve
        self.interior_model = interior_model
        self.reparametrization_model = reparametrization_model
        self.sampler = sampler
        self.extension_method = extension_method
        self.smooth_extension = smooth_extension
        self.treshold = treshold

        r = lambda xy: torch.sqrt((xy ** 2).sum(dim=1))
        
        if extension_method == 'disc':
            self.R = r
            self.phi0 = lambda xy: 0.5 * torch.log(1 - (xy ** 2).sum(axis=-1))
        elif extension_method == 'stereographic':
            self.R = lambda xy: 2*r(xy) / (1 + r(xy)**2)
            self.phi0 = lambda xy: torch.log(2 /(1 + (xy ** 2).sum(axis=-1)) - 1)

        def fatten(xy):
            th = torch.atan2(xy[...,1], xy[...,0])
            curve_point = self.curve(th)
            if smooth_extension:
                return smooth_radius(self.R(xy), self.treshold).unsqueeze(-1) * curve_point
            else:
                return self.R(xy).unsqueeze(-1) * curve_point
            
        self.fatten = fatten   
    
    def forward(self, xy):
        if self.reparametrization_model is not None:
            XY = self.reparametrization_model(xy)
        else:
            XY = xy

        M = self.interior_model(XY) # shape (N, 3) for H3
        phi = M[:,0]
        v = M[:,1:]

        ans_X = torch.exp(self.phi0(XY) + phi) # shape (N, )
        ans_Y = self.fatten(XY) + torch.exp(self.phi0(XY)).unsqueeze(-1) * v # shape (N, 2)

        return torch.cat([ans_X.unsqueeze(-1), ans_Y], dim=-1) # shape (N, 3)
    
    def train_unsupervised(
            self,
            norm_fcn = L2_squared,
            epochs = 1000,
            batch_size = 2**10,
            lr = 1e-3,
            lr_min = None,  # Minimum learning rate for scheduler (defaults to lr*0.01)
            resample_step = 1,  # Resample collocation points every N epochs (1 = every epoch)
            verbose = True,):
        device = next(self.parameters()).device
        
        if lr_min is None:
            lr_min = lr * 0.01  # Default to 1% of initial learning rate

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Cosine annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
        losses = []

        # training loop
        best_loss = torch.inf
        best_model = None
        
        for ep in range(epochs+1):
            if ep % resample_step == 0:
                # Select sampling scheme
                xy = self.sampler(batch_size).to(device)
                xy.requires_grad_(True)

            optimizer.zero_grad()

            # u = self(xy)
            # t = PDE(u, xy) # shape (N, 3)
            t = minimal_in_H3_PDE(self)(xy)
            
            loss = norm_fcn(t)
            
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            loss_val = loss.item()
            losses.append(loss_val)
            if loss_val < best_loss:
                best_loss = loss_val
                # Only deepcopy when we have a new best (not every iteration)
                best_model = {k: v.cpu().clone() for k, v in self.state_dict().items()}

            if verbose and ep % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {ep:5d}/{epochs}: Loss = {losses[-1]:.3e}, LR = {current_lr:.3e}")
                
            if math.isnan(float(loss_val)):
                self.load_state_dict(best_model)

        self.load_state_dict(best_model)
        return best_model, best_loss, losses
    
    def train_supervised(
            self,
            model_supervised,
            norm_fcn = L2_squared,
            epochs = 1000,
            batch_size = 2**10,
            lr = 1e-3,
            lr_min = None,  # Minimum learning rate for scheduler (defaults to lr*0.01)
            resample_step = 1,  # Resample collocation points every N epochs (1 = every epoch)
            verbose = True,):
        device = next(self.parameters()).device
        
        if lr_min is None:
            lr_min = lr * 0.01  # Default to 1% of initial learning rate

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Cosine annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
        losses = []

        # training loop
        best_loss = torch.inf
        best_model = None
        
        for ep in range(epochs+1):
            if ep % resample_step == 0:
                # Select sampling scheme
                xy = self.sampler(batch_size).to(device)
                xy.requires_grad_(True)

            optimizer.zero_grad()

            u = self(xy)
            v = model_supervised(xy)
            loss = norm_fcn(u-v)
            
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            loss_val = loss.item()
            losses.append(loss_val)
            if loss_val < best_loss:
                best_loss = loss_val
                # Only deepcopy when we have a new best (not every iteration)
                best_model = {k: v.cpu().clone() for k, v in self.state_dict().items()}

            if verbose and ep % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {ep:5d}/{epochs}: Loss = {losses[-1]:.3e}, LR = {current_lr:.3e}")

        self.load_state_dict(best_model)
        return best_model, best_loss, losses
    
    def train_unsupervised_sgd(
            self,
            norm_fcn=L2_squared,
            epochs=100,          # Number of passes through the full dataset
            dataset_size=2**14,  # Size of the "very big sample"
            batch_size=2**10,    # Size of the mini-batch
            lr=1e-3,
            lr_min=None,
            verbose=True,
            resample_step = None,
    ):
        device = next(self.parameters()).device
        
        if lr_min is None:
            lr_min = lr * 0.01
        
        if resample_step is None:
            resample_step = epochs+1

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Scheduler steps per epoch usually, but can be per batch. 
        # Standard convention is per epoch for CosineAnnealing.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
        
        history = [] # storing average epoch losses
        best_loss = torch.inf
        best_model = None

        for ep in range(epochs):
            epoch_loss_sum = 0.0
            num_batches = 0

            if ep % resample_step == 0:
                # 1. Generate the "Very Big Sample" once
                # (You could also move this inside the loop if you wanted to resample the big pool every N epochs)
                full_xy = self.sampler(dataset_size).to(device)
                
                # 2. Create DataLoader
                # We don't need requires_grad here yet; we set it inside the loop
                dataset = TensorDataset(full_xy)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # --- Mini-Batch Loop (SGD) ---
            for batch_idx, (xy_batch,) in enumerate(dataloader):
                xy_batch = xy_batch.to(device)
                
                # CRITICAL for PINNs: Enable gradient tracking on the input batch
                # so we can compute PDE derivatives (d_u/d_x, etc.)
                xy_batch.requires_grad_(True)

                optimizer.zero_grad()

                # Calculate PDE residuals
                t = minimal_in_H3_PDE(self)(xy_batch)
                
                loss = norm_fcn(t)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss_sum += loss.item()
                num_batches += 1

            # --- End of Epoch Operations ---
            scheduler.step() # Update LR once per epoch
            
            # Calculate average loss for this epoch
            avg_epoch_loss = epoch_loss_sum / num_batches
            history.append(avg_epoch_loss)

            # Check for best model based on average epoch loss (more stable than batch loss)
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model = {k: v.cpu().clone() for k, v in self.state_dict().items()}

            if verbose and ep % 10 == 0: # Print less frequently as epochs are now "heavier"
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {ep:5d}/{epochs}: Avg Loss = {avg_epoch_loss:.3e}, LR = {current_lr:.3e}")

            # Safety check for NaN
            if math.isnan(avg_epoch_loss):
                print("Loss is NaN, reverting to best model and stopping.")
                if best_model is not None:
                    self.load_state_dict(best_model)
                break

        if best_model is not None:
            self.load_state_dict(best_model)
            
        return best_model, best_loss, history

    def train_supervised_sgd(
            self,
            model_supervised,
            norm_fcn=L2_squared,
            epochs=100,
            dataset_size=2**14,
            batch_size=2**10,
            lr=1e-3,
            lr_min=None,
            verbose=True,
    ):
        device = next(self.parameters()).device
        
        if lr_min is None:
            lr_min = lr * 0.01

        # 1. Generate the "Very Big Sample"
        full_xy = self.sampler(dataset_size).to(device)
        
        # 2. Create DataLoader
        dataset = TensorDataset(full_xy)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
        
        history = []
        best_loss = torch.inf
        best_model = None

        for ep in range(epochs):
            epoch_loss_sum = 0.0
            num_batches = 0
            
            for batch_idx, (xy_batch,) in enumerate(dataloader):
                xy_batch = xy_batch.to(device)
                # Usually supervised learning doesn't require input grads unless
                # the loss function involves derivatives of the input. 
                # If standard MSE on values: not strictly needed, but safe to keep for consistency.
                xy_batch.requires_grad_(True) 

                optimizer.zero_grad()

                u = self(xy_batch)
                v = model_supervised(xy_batch)
                loss = norm_fcn(u - v)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss_sum += loss.item()
                num_batches += 1

            scheduler.step()

            avg_epoch_loss = epoch_loss_sum / num_batches
            history.append(avg_epoch_loss)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model = {k: v.cpu().clone() for k, v in self.state_dict().items()}

            if verbose and ep % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {ep:5d}/{epochs}: Avg Loss = {avg_epoch_loss:.3e}, LR = {current_lr:.3e}")
                
            if math.isnan(avg_epoch_loss):
                if best_model is not None:
                    self.load_state_dict(best_model)
                break

        if best_model is not None:
            self.load_state_dict(best_model)
            
        return best_model, best_loss, history