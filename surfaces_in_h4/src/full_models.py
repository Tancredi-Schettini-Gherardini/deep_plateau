import torch
import torch.nn as nn
import os

from src.interior_models import MLP
from src.knots import build_knot
from src.extensions import get_stereobiharmonic_extension

class HyperbolicMinimalSurfacePINN(nn.Module):
    """
    The full physics-informed neural network combining the MLP with the 
    geometric ansatz: u(x, y) = (rho * exp(NN^X), ext(gamma) + rho^k * NN^Y)
    """
    def __init__(
            self,
            knot_type,
            knot_kwargs,
            knot_perturbation_matrix = None,
            interior_model_type = 'mlp',
            interior_model_kwargs = None,
            bdf_type = 'stereographic',
            ext_type = 'stereobiharmonic',
            ext_kwargs = None,
            decay_exponent = 2,
            dtype = torch.float64):
        super().__init__()

        # Build fresh mutable defaults inside the body (avoids the shared
        # mutable-default-argument pitfall).
        if interior_model_kwargs is None:
            interior_model_kwargs = {
                'in_dim': 2,
                'out_dim': 4,
                'hidden_dim': 64,
                'num_hidden_layers': 4
            }
        if ext_kwargs is None:
            ext_kwargs = {'N': 15, 'num_samples': 10000}

        self.kwargs = {
            'knot_type': knot_type,
            'knot_kwargs': knot_kwargs,
            'knot_perturbation_matrix': knot_perturbation_matrix,
            'interior_model_type': interior_model_type,
            'interior_model_kwargs': interior_model_kwargs,
            'bdf_type': bdf_type,
            'ext_type': ext_type,
            'ext_kwargs': ext_kwargs,
            'decay_exponent': decay_exponent,
            'dtype': dtype,
        }

        # interior model
        if interior_model_type == 'mlp':
            self.NN = MLP(**interior_model_kwargs).to(dtype=dtype)
        else:
            raise NotImplementedError(f'Interior model "{interior_model_type}" not implemented.')
        
        # bdf definition
        if bdf_type == 'stereographic':
            def rho_fn(xy):
                r_sq = (xy ** 2).sum(dim=-1, keepdim=True)
                return (1 - r_sq) / (1 + r_sq)
            
            self.bdf = rho_fn
        else:
            raise NotImplementedError(f'Boundary defining function "{bdf_type}" not implemented.')
        
        # knot generation
        knot = build_knot(
            knot_type = knot_type,
            knot_kwargs = knot_kwargs,
            knot_perturbation_matrix = knot_perturbation_matrix.to(dtype=dtype) if knot_perturbation_matrix is not None else None
        )
        knot_param = knot.get_evaluator()

        # ext generation
        if ext_type == 'stereobiharmonic':
            ext = get_stereobiharmonic_extension(
                knot_param,
                dtype=dtype,
                **ext_kwargs
            )
        else:
            raise NotImplementedError(f'Extension method "{ext_type}" not implemented.')
        
        self.ext = ext
        self.decay_exponent = decay_exponent
   
    def forward(self, xy):
        """
        Args:
            xy: Tensor of shape (N, 2) representing (x, y) coordinates in the unit disc D^2.
        Returns:
            Tensor of shape (N, 4) representing (X, Y1, Y2, Y3) in the half-space model.
        """
        xy = xy.to(dtype=self.kwargs['dtype'])
        
        # Get the value of the BDF
        rho = self.bdf(xy)
        
        # Get the network outputs
        # Shape: (N, 4)
        nn_out = self.NN(xy)
        
        # Split outputs into X component and Y components
        nn_X = nn_out[..., 0:1]      # Shape: (N, 1)
        nn_Y = nn_out[..., 1:4]      # Shape: (N, 3)
        
        # Get the extension operator output for the knot
        # Shape: (N, 3)
        ext_Y = self.ext(xy)
        
        # Construct the final map u_theta
        # X component: rho * exp(NN^X)
        X = rho * torch.exp(nn_X)
        
        # Y components: ext(gamma) + rho^k * NN^Y
        Y = ext_Y + (rho ** self.decay_exponent) * nn_Y
        
        # Concatenate to return (X, Y1, Y2, Y3)
        return torch.cat([X, Y], dim=-1)
    
    @property
    def name(self):
        """
        Generates a standardized name for the model based on its knot parameters.
        Example: KNOT_trefoil_KNOT_PAR_R1_p2_PERTURBED
        """
        knot_type = self.kwargs.get('knot_type', 'unknown')
        knot_kwargs = self.kwargs.get('knot_kwargs', {})
        perturb = self.kwargs.get('knot_perturbation_matrix', None)
        
        name = f"KNOT_{knot_type}"
        
        if knot_kwargs:
            # Sorting the items ensures the parameters are always in the same 
            # order, preventing accidental duplicates like p2_R1 vs R1_p2
            par_str = "_".join([f"{k}{v}" for k, v in sorted(knot_kwargs.items())])
            name += f"_KNOT_PAR_{par_str}"
        
        if perturb is not None:
            name += '_PERTURBED'
            
        return name
    
    def save(self, filepath=None, directory="."):
        """
        Saves the model's initialization kwargs and learnable parameters.
        If a file with the same name exists, appends _2, _3, etc. to prevent overwriting.
        """
        # 1. Determine the initial target filepath
        if filepath is None:
            filepath = os.path.join(directory, f"{self.name}.pt")
            
        # 2. Split the path into directory, base name, and extension (.pt)
        base_dir, file_name = os.path.split(filepath)
        name_only, extension = os.path.splitext(file_name)
        
        # 3. Check for collisions and increment a suffix until the path is clear
        counter = 2
        while os.path.exists(filepath):
            new_file_name = f"{name_only}_{counter}{extension}"
            filepath = os.path.join(base_dir, new_file_name)
            counter += 1
            
        # 4. Save the model
        checkpoint = {
            'kwargs': self.kwargs,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, device='cpu'):
        """
        Loads the model from a checkpoint, initializing it with the saved kwargs
        and then loading the state dictionary.
        """
        # map_location ensures we can load a GPU-trained model on a CPU machine if needed
        checkpoint = torch.load(filepath, map_location=device)
        
        # 1. Initialize the model using the saved kwargs
        model = cls(**checkpoint['kwargs'])
        
        # 2. Load the learnable parameters
        model.load_state_dict(checkpoint['state_dict'])
        
        # 3. Move model to the desired device
        model = model.to(device)
        
        return model
    
    def get_knot(self):
        knot_type = self.kwargs['knot_type']
        knot_kwargs = self.kwargs['knot_kwargs']
        knot_perturbation_matrix = self.kwargs['knot_perturbation_matrix']

        knot = build_knot(
            knot_type = knot_type,
            knot_kwargs = knot_kwargs,
            knot_perturbation_matrix = knot_perturbation_matrix
            )
        
        return knot