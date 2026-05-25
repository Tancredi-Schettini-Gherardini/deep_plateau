import torch
from torch.func import vmap, jacrev, jacfwd

def minimal_in_H3_PDE(u_callable, eps=0):
    """
    Optimized drop-in replacement for H3.
    Strictly preserves the math of the original, including the (1/rho) scaling 
    at the end and the (N, 3) vs (3,) output behavior.
    """
    def u_single(x):
        return u_callable(x.unsqueeze(0)).squeeze(0)

    # Pre-compile derivatives
    # hess_fn: (N, D, 2, 2), jac_fn: (N, D, 2)
    hess_fn = vmap(jacfwd(jacrev(u_single)))
    jac_fn = vmap(jacrev(u_single))

    def PDE_callable(xy):
        single = False
        if xy.dim() == 1:
            xy = xy.unsqueeze(0)
            single = True
        
        # 1. Forward pass
        u = u_callable(xy)      # (N, D) - For H3, D=3
        J = jac_fn(xy)          # (N, D, 2)
        H = hess_fn(xy)         # (N, D, 2, 2)
        
        N, D = u.shape
        rho = u[:, 0]
        
        # 2. Metric Construction: g = rho^-2 * J^T J
        inv_rho2 = (rho.pow(-2)).view(N, 1, 1)
        JtJ = torch.einsum('ndi,ndj->nij', J, J) 
        g = JtJ * inv_rho2
        
        # 3. Analytic Inverse (2x2) & Determinant
        # g = [[a, b], [c, d]]
        a, b = g[:, 0, 0], g[:, 0, 1]
        c, d = g[:, 1, 0], g[:, 1, 1]
        
        det = a*d - b*c
        safe_det = det.clamp(min=eps)
        
        # Vectorized inverse construction
        inv_det = 1.0 / safe_det
        row0 = torch.stack([d, -b], dim=1)
        row1 = torch.stack([-c, a], dim=1)
        g_inv = torch.stack([row0, row1], dim=1) * inv_det.view(N, 1, 1)

        # 4. Derivatives of Metric (Algebraic)
        # d(J^T J)_kij = sum_d (H_dki * J_dj + J_di * H_dkj)
        # term: (N, k, i, j)
        term = torch.einsum('ndki,ndj->nkij', H, J)
        dJtJ = term + term.transpose(-1, -2)
        
        # d(rho^-2)
        drho = J[:, 0, :] # (N, 2)
        d_inv_rho2 = -2.0 * (rho.pow(-3)).view(N, 1) * drho # (N, k)
        
        # dg[k] = d(inv_rho2)[k] * JtJ + inv_rho2 * d(JtJ)[k]
        dg = d_inv_rho2.view(N, 2, 1, 1) * JtJ.view(N, 1, 2, 2) + \
             inv_rho2.view(N, 1, 1, 1) * dJtJ

        # 5. Derivatives of Inverse & log_det
        # dg_inv = - g_inv @ dg @ g_inv
        dg_inv = -torch.matmul(torch.matmul(g_inv.unsqueeze(1), dg), g_inv.unsqueeze(1))
        
        # grad_log_det = 0.5 * tr(g_inv @ dg)
        trace_term = (g_inv.unsqueeze(1) * dg).sum(dim=(-1, -2)) # (N, k)
        
        # 6. Laplace-Beltrami Components
        # Delta u = g^ij H_ij + (div g^row) . J + (grad log_det) . J
        
        # Term 1: g^ij * H_ij
        lap_1 = torch.einsum('nij,ndij->nd', g_inv, H)
        
        # Term 2: (d_i g^ij) * J_j
        div_g_inv = torch.einsum('niij->nj', dg_inv) 
        lap_2 = torch.einsum('nj,ndj->nd', div_g_inv, J)
        
        # Term 3: g^ij (d_i log sqrt_g) * d_j u
        grad_log_det = 0.5 * trace_term
        vec_field = torch.einsum('nij,ni->nj', g_inv, grad_log_det)
        lap_3 = torch.einsum('nj,ndj->nd', vec_field, J)
        
        laplace_op = lap_1 + lap_2 + lap_3 # (N, D)

        # 7. Final PDE Algebra (Ernst Eq)
        dv = J[:, 1:, :] # (N, D-1, 2)
        
        # Squared norms/inner prods in metric g
        norm_drho_sq = torch.einsum('ni,nij,nj->n', drho, g_inv, drho)
        norm_dv_sq = torch.einsum('nki,nij,nkj->n', dv, g_inv, dv)
        inner_rho_v = torch.einsum('ni,nij,nkj->nk', drho, g_inv, dv)
        
        # Construct Residuals
        # Eq 1: lap(rho) + (1/rho)(|dv|^2 - |drho|^2)
        res_rho = laplace_op[:, 0] + (1.0/rho) * (norm_dv_sq - norm_drho_sq)
        
        # Eq 2: lap(v) - (2/rho)<drho, dv>
        res_v = laplace_op[:, 1:] - (2.0/rho).view(N, 1) * inner_rho_v
        
        # Combine
        out_unscaled = torch.cat([res_rho.view(N, 1), res_v], dim=1)
        
        # FINAL SCALING (Correction): Multiply by 1/rho to match original code
        out = out_unscaled * (rho.pow(-1)).view(N, 1)

        return out[0] if single else out

    return PDE_callable