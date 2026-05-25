import torch
import torch.nn as nn
import numpy as np

# ------------------------------------------------------------------------------------
class MixSampler(nn.Module):
    def __init__(self, bias = 0.5, mix=1, target=(0,0), sigma=0.2, dtype=torch.float64, device='cpu'):
        super().__init__()
        self.bias = bias
        self.mix = mix
        self.target = target
        self.sigma = sigma
        self.mix = mix
        self.dtype = dtype
        self.device = device

        # Descriptor for portable save/load. dtype/device are intentionally
        # excluded — `MainModel.__init__` always overrides them based on
        # `use_float64`.
        self.config = {
            'kind': 'MixSampler',
            'kwargs': {
                'bias': bias, 'mix': mix,
                'target': tuple(target), 'sigma': sigma,
            },
        }
    
    def background_sample(self, N, bias=0.5):
        # Use power law for radial distribution: r = u^b where u ~ Uniform(0,1)
        r = torch.rand(N, dtype=self.dtype, device=self.device) ** bias
        theta = 2 * torch.pi * torch.rand(N, dtype=self.dtype, device=self.device)
        
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        return torch.stack([x, y], dim=1)

    def forward(self, N):
        target = torch.tensor(self.target, dtype=self.dtype, device=self.device)

        n_background = int(N * self.mix)
        n_gauss = N - n_background

        # Truncated Gaussian around p
        if n_gauss > 0:
            gauss_samples = []
            while len(gauss_samples) < n_gauss:
                batch = target + self.sigma * torch.randn(n_gauss, 2, dtype=self.dtype, device=self.device)
                mask = (batch[:,0]**2 + batch[:,1]**2) <= 1.0
                gauss_samples.append(batch[mask])
                if sum(s.shape[0] for s in gauss_samples) >= n_gauss:
                    break
            gauss_samples = torch.cat(gauss_samples, dim=0)[:n_gauss]
        else:
            gauss_samples = torch.empty(0, 2, dtype=self.dtype, device=self.device)

        # background samples
        background_samples = self.background_sample(n_background, bias = self.bias) if n_background > 0 else torch.empty(0, 2, dtype=self.dtype, device=self.device)

        # Combine
        samples = torch.cat([gauss_samples, background_samples], dim=0)
        return samples