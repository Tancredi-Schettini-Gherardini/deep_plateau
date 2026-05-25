import sys
import warnings
import torch

# ===========================================================================
# Compile-compatibility shim
# ===========================================================================
_DEFAULT_COMPILE_KW = dict(mode='default', dynamic=False, fullgraph=True)
_COMPILE_SUPPORTED = sys.version_info < (3, 14)
_COMPILE_WARNED = False

def _maybe_compile(fn, use_compile: bool, compile_kwargs=None):
    if not use_compile or not _COMPILE_SUPPORTED:
        return fn
    kw = dict(_DEFAULT_COMPILE_KW)
    if compile_kwargs:
        kw.update(compile_kwargs)
    return torch.compile(fn, **kw)


# ===========================================================================
# Math & Algebra Engine (Exact math from original, eps removed)
# ===========================================================================
def _v4_algebra(u, J, H):
    N = u.shape[0]
    rho = u[:, 0]
    drho = J[:, 0, :]
    dv = J[:, 1:, :]

    inv_rho2 = rho.pow(-2)
    JtJ = torch.einsum('ndi,ndj->nij', J, J)
    g = JtJ * inv_rho2.view(N, 1, 1)

    g00, g01 = g[:, 0, 0], g[:, 0, 1]
    g10, g11 = g[:, 1, 0], g[:, 1, 1]
    det = g00 * g11 - g01 * g10
    inv_det = det.reciprocal()

    g_inv = torch.empty_like(g)
    g_inv[:, 0, 0] = g11 * inv_det
    g_inv[:, 0, 1] = -g01 * inv_det
    g_inv[:, 1, 0] = -g10 * inv_det
    g_inv[:, 1, 1] = g00 * inv_det

    g_inv_drho = torch.einsum('nij,nj->ni', g_inv, drho)
    g_inv_dv = torch.einsum('nij,nkj->nki', g_inv, dv)

    term = torch.einsum('ndki,ndj->nkij', H, J)
    dJtJ = term + term.transpose(-1, -2)
    d_inv_rho2 = (-2.0 * rho.pow(-3)).view(N, 1) * drho
    dg = (d_inv_rho2.view(N, 2, 1, 1) * JtJ.view(N, 1, 2, 2)
          + inv_rho2.view(N, 1, 1, 1) * dJtJ)

    g_inv_u = g_inv.unsqueeze(1)
    P = torch.matmul(g_inv_u, dg)

    trace_term = P.diagonal(dim1=-2, dim2=-1).sum(-1)
    S = P[:, 0, 0, :] + P[:, 1, 1, :]
    div_g_inv = -torch.einsum('nij,ni->nj', g_inv, S)

    lap_1 = torch.einsum('nij,ndij->nd', g_inv, H)
    lap_2 = torch.einsum('nj,ndj->nd', div_g_inv, J)
    grad_log_det = 0.5 * trace_term
    vec_field = torch.einsum('nij,ni->nj', g_inv, grad_log_det)
    lap_3 = torch.einsum('nj,ndj->nd', vec_field, J)

    laplace_op = lap_1 + lap_2 + lap_3

    inv_rho = rho.reciprocal()
    norm_drho_sq = (drho * g_inv_drho).sum(dim=-1)
    norm_dv_sq = (dv * g_inv_dv).sum(dim=(-1, -2))
    inner_rho_v = torch.einsum('ni,nki->nk', g_inv_drho, dv)

    res_rho = laplace_op[:, 0] + inv_rho * (norm_dv_sq - norm_drho_sq)
    res_v = laplace_op[:, 1:] - (2.0 * inv_rho).view(N, 1) * inner_rho_v
    return torch.cat([res_rho.view(N, 1), res_v], dim=1) * inv_rho.view(N, 1)


def _combine_product_rule(mlp_u, mlp_J, mlp_H,
                          rho, rho_J, rho_H,
                          ext_u, ext_J, ext_H,
                          p: int):
    N = mlp_u.shape[0]
    nn_0 = mlp_u[:, 0]                                     
    nn_k = mlp_u[:, 1:]                                    
    nnJ_0 = mlp_J[:, 0, :]                                 
    nnJ_k = mlp_J[:, 1:, :]                                
    nnH_0 = mlp_H[:, 0, :, :]                              
    nnH_k = mlp_H[:, 1:, :, :]                             

    E = torch.exp(nn_0)                                     

    # ----- u -----
    u0 = rho * E                                            
    rho_view = rho.view(N, 1)
    rho_p_vec = rho_view.pow(p)                             
    u_rest = ext_u + rho_p_vec * nn_k                       
    u = torch.cat([u0.view(N, 1), u_rest], dim=1)           

    # ----- J -----
    G = rho_J + rho_view * nnJ_0                            
    J0 = E.view(N, 1) * G                                   

    rho_pm1_vec = rho_view.pow(p - 1)                       
    rho_p_bc = rho_p_vec.view(N, 1, 1)                      
    rho_pm1_bc = rho_pm1_vec.view(N, 1, 1)                  

    term_A = p * rho_pm1_bc * nn_k.view(N, 3, 1) * rho_J.view(N, 1, 2)
    term_B = rho_p_bc * nnJ_k
    J_rest = ext_J + term_A + term_B                        
    J = torch.cat([J0.view(N, 1, 2), J_rest], dim=1)        

    # ----- H -----
    nnJ_0_l = nnJ_0.view(N, 1, 2)                           
    G_j = G.view(N, 2, 1)                                   
    rhoJ_l = rho_J.view(N, 1, 2)
    nnJ_0_j = nnJ_0.view(N, 2, 1)
    rho_bc = rho.view(N, 1, 1)
    M = (nnJ_0_l * G_j
         + rho_H
         + rhoJ_l * nnJ_0_j
         + rho_bc * nnH_0)                                  
    H0 = E.view(N, 1, 1) * M                                

    rho_pm2_bc = rho_view.pow(p - 2).view(N, 1, 1, 1)       
    rho_pm1_bc4 = rho_pm1_vec.view(N, 1, 1, 1)
    rho_p_bc4 = rho_p_vec.view(N, 1, 1, 1)

    outer_rhoJ = rho_J.view(N, 2, 1) * rho_J.view(N, 1, 2)  
    outer_rhoJ_bc = outer_rhoJ.view(N, 1, 2, 2)             
    nn_k_bc = nn_k.view(N, 3, 1, 1)
    rho_H_bc = rho_H.view(N, 1, 2, 2)

    A = (nnJ_k.view(N, 3, 1, 2) * rho_J.view(N, 1, 2, 1)
         + nnJ_k.view(N, 3, 2, 1) * rho_J.view(N, 1, 1, 2))  

    H_rest = (ext_H
              + p * (p - 1) * rho_pm2_bc * outer_rhoJ_bc * nn_k_bc
              + p * rho_pm1_bc4 * rho_H_bc * nn_k_bc
              + p * rho_pm1_bc4 * A
              + rho_p_bc4 * nnH_k)                          

    H = torch.cat([H0.view(N, 1, 2, 2), H_rest], dim=1)     

    return u, J, H


def _stereographic_bdf_uJH(xy):
    x = xy[:, 0]; y = xy[:, 1]
    r2 = x * x + y * y
    m = 1.0 + r2
    inv_m = 1.0 / m
    inv_m2 = inv_m * inv_m
    inv_m3 = inv_m2 * inv_m

    u = (1.0 - r2) * inv_m
    Jx = -4.0 * x * inv_m2
    Jy = -4.0 * y * inv_m2
    Hxx = 4.0 * (3.0 * x * x - y * y - 1.0) * inv_m3
    Hyy = 4.0 * (3.0 * y * y - x * x - 1.0) * inv_m3
    Hxy = 16.0 * x * y * inv_m3

    u = u.unsqueeze(-1)
    J = torch.stack([Jx, Jy], dim=-1).unsqueeze(1)
    H = torch.stack([
        torch.stack([Hxx, Hxy], dim=-1),
        torch.stack([Hxy, Hyy], dim=-1),
    ], dim=-2).unsqueeze(1)
    return u, J, H


def _tanh_deriv(z):
    s = torch.tanh(z)
    s_p = 1.0 - s * s
    s_pp = -2.0 * s * s_p
    return s, s_p, s_pp


def _silu_deriv(z):
    phi = torch.sigmoid(z)
    s = z * phi
    one_minus_phi = 1.0 - phi
    s_p = phi + z * phi * one_minus_phi
    s_pp = phi * one_minus_phi * (2.0 + z * (1.0 - 2.0 * phi))
    return s, s_p, s_pp


def mlp_uJH(mlp_seq, x):
    import torch.nn as nn 
    N, in_dim = x.shape
    a = x
    Ja = torch.eye(in_dim, dtype=x.dtype, device=x.device)\
              .unsqueeze(0).expand(N, in_dim, in_dim).contiguous()
    Ha = torch.zeros(N, in_dim, in_dim, in_dim,
                     dtype=x.dtype, device=x.device)

    for layer in mlp_seq:
        if isinstance(layer, nn.Linear):
            W = layer.weight                                
            b = layer.bias                                  
            a  = a @ W.T + b                                
            Ja = torch.einsum('hi,nij->nhj', W, Ja)         
            Ha = torch.einsum('hi,nijk->nhjk', W, Ha)       
        elif isinstance(layer, (nn.Tanh, nn.SiLU)):
            sigma, sigma_p, sigma_pp = (
                _tanh_deriv(a) if isinstance(layer, nn.Tanh) else _silu_deriv(a)
            )
            outer = Ja.unsqueeze(-1) * Ja.unsqueeze(-2)     
            new_Ja = sigma_p.unsqueeze(-1) * Ja
            new_Ha = (sigma_pp.unsqueeze(-1).unsqueeze(-1) * outer
                      + sigma_p.unsqueeze(-1).unsqueeze(-1) * Ha)
            a, Ja, Ha = sigma, new_Ja, new_Ha
        else:
            raise ValueError(
                f"Hand-rolled MLP J/H supports nn.Linear + nn.Tanh / nn.SiLU only; "
                f"got {type(layer).__name__}."
            )
    return a, Ja, Ha


def _stereobiharmonic_ext_uJH_with_coeffs(xy, A0, An, Bn):
    N_terms = An.shape[0]
    x = xy[:, 0]; y = xy[:, 1]
    B = x.shape[0]
    _A0 = A0.to(xy); _An = An.to(xy); _Bn = Bn.to(xy)

    r2 = x * x + y * y
    m  = 1.0 + r2
    inv_m  = 1.0 / m
    inv_m2 = inv_m * inv_m
    inv_m3 = inv_m2 * inv_m
    mu     = 2.0 * inv_m
    mu_x   = -4.0 * x * inv_m2
    mu_y   = -4.0 * y * inv_m2
    mu_xx  = 4.0 * (4.0 * x * x - m) * inv_m3
    mu_yy  = 4.0 * (4.0 * y * y - m) * inv_m3
    mu_xy  = 16.0 * x * y * inv_m3

    U_list = [torch.ones_like(x), x]
    V_list = [torch.zeros_like(x), y]
    for n in range(1, N_terms):
        U_next = U_list[-1] * x - V_list[-1] * y
        V_next = V_list[-1] * x + U_list[-1] * y
        U_list.append(U_next); V_list.append(V_next)
    U_stack = torch.stack(U_list, dim=-1)              
    V_stack = torch.stack(V_list, dim=-1)

    zeros_col = torch.zeros_like(x).unsqueeze(-1)      
    Un_full   = U_stack[:, 1:N_terms + 1]              
    Vn_full   = V_stack[:, 1:N_terms + 1]
    Un_1_full = U_stack[:, 0:N_terms]                  
    Vn_1_full = V_stack[:, 0:N_terms]
    Un_2_full = torch.cat([zeros_col, U_stack[:, 0:N_terms - 1]], dim=-1)  
    Vn_2_full = torch.cat([zeros_col, V_stack[:, 0:N_terms - 1]], dim=-1)

    n_vec = torch.arange(1, N_terms + 1, dtype=xy.dtype, device=xy.device)  
    half = 0.5
    n_plus_1_half     = (n_vec + 1.0) * half           
    one_minus_n_half  = (1.0 - n_vec) * half           
    Wp                = one_minus_n_half               
    nn1_vec           = n_vec * (n_vec - 1)            

    W_arr = n_plus_1_half.view(1, -1) + one_minus_n_half.view(1, -1) * r2.unsqueeze(-1)

    Q    = Un_full.unsqueeze(-1)   * _An + Vn_full.unsqueeze(-1)   * _Bn
    n_b  = n_vec.view(1, -1, 1)                                              
    Q_x  = n_b * (Un_1_full.unsqueeze(-1)  * _An + Vn_1_full.unsqueeze(-1)  * _Bn)
    Q_y  = n_b * (-Vn_1_full.unsqueeze(-1) * _An + Un_1_full.unsqueeze(-1)  * _Bn)
    nn1_b = nn1_vec.view(1, -1, 1)
    Q_xx = nn1_b * (Un_2_full.unsqueeze(-1)  * _An + Vn_2_full.unsqueeze(-1) * _Bn)
    Q_yy = -Q_xx
    Q_xy = nn1_b * (-Vn_2_full.unsqueeze(-1) * _An + Un_2_full.unsqueeze(-1) * _Bn)

    W_col       = W_arr.unsqueeze(-1)                                       
    alpha_x     = (2.0 * x.unsqueeze(-1)) * Wp.view(1, -1)                  
    alpha_y     = (2.0 * y.unsqueeze(-1)) * Wp.view(1, -1)
    beta        = (2.0 * Wp).view(1, -1)                                    
    beta_x      = (4.0 * x.unsqueeze(-1)) * Wp.view(1, -1)                  
    beta_y      = (4.0 * y.unsqueeze(-1)) * Wp.view(1, -1)

    S    = (W_col * Q).sum(dim=1)                                            
    S_x  = (alpha_x.unsqueeze(-1) * Q + W_col * Q_x).sum(dim=1)
    S_y  = (alpha_y.unsqueeze(-1) * Q + W_col * Q_y).sum(dim=1)
    S_xx = (beta.unsqueeze(-1) * Q
            + beta_x.unsqueeze(-1) * Q_x
            + W_col * Q_xx).sum(dim=1)
    S_yy = (beta.unsqueeze(-1) * Q
            + beta_y.unsqueeze(-1) * Q_y
            + W_col * Q_yy).sum(dim=1)
    S_xy = (alpha_x.unsqueeze(-1) * Q_y
            + alpha_y.unsqueeze(-1) * Q_x
            + W_col * Q_xy).sum(dim=1)

    mu_c   = mu.unsqueeze(-1)
    mux_c  = mu_x.unsqueeze(-1)
    muy_c  = mu_y.unsqueeze(-1)
    muxx_c = mu_xx.unsqueeze(-1)
    muyy_c = mu_yy.unsqueeze(-1)
    muxy_c = mu_xy.unsqueeze(-1)

    u = _A0 + mu_c * S
    Jx = mux_c * S + mu_c * S_x
    Jy = muy_c * S + mu_c * S_y
    J  = torch.stack([Jx, Jy], dim=-1)

    Hxx = muxx_c * S + 2.0 * mux_c * S_x + mu_c * S_xx
    Hyy = muyy_c * S + 2.0 * muy_c * S_y + mu_c * S_yy
    Hxy = muxy_c * S + mux_c * S_y + muy_c * S_x + mu_c * S_xy
    H = torch.stack([
        torch.stack([Hxx, Hxy], dim=-1),
        torch.stack([Hxy, Hyy], dim=-1),
    ], dim=-2)
    return u, J, H


# ===========================================================================
# The Flat Module (Wired to your HyperbolicMinimalSurfacePINN)
# ===========================================================================
class FlatStereoPDEModel(torch.nn.Module):
    def __init__(self, host_model):
        super().__init__()
        
        # Pull parameters directly from your HyperbolicMinimalSurfacePINN structure
        self.mlp = host_model.NN.net
        self.p = int(host_model.decay_exponent)

        # Pull precomputed extension matrices
        evaluator = host_model.ext.evaluator_module
        self.register_buffer('A0', evaluator.A0.clone())
        self.register_buffer('An', evaluator.An.clone())
        self.register_buffer('Bn', evaluator.Bn.clone())

    def forward(self, xy):
        single = False
        if xy.dim() == 1:
            xy = xy.unsqueeze(0)
            single = True

        # 1. Hand-rolled MLP (u, J, H)
        mlp_u, mlp_J, mlp_H = mlp_uJH(self.mlp, xy)
        
        # 2. Analytical stereographic bdf
        bdf_u, bdf_J, bdf_H = _stereographic_bdf_uJH(xy)
        rho, rho_J, rho_H = bdf_u[:, 0], bdf_J[:, 0, :], bdf_H[:, 0, :, :]

        # 3. Analytical stereobiharmonic ext
        ext_u, ext_J, ext_H = _stereobiharmonic_ext_uJH_with_coeffs(xy, self.A0, self.An, self.Bn)

        # 4. Product-rule combine
        u, J, H = _combine_product_rule(mlp_u, mlp_J, mlp_H, rho, rho_J, rho_H, ext_u, ext_J, ext_H, self.p)

        # 5. Raw algebra execution (eps removed)
        out = _v4_algebra(u, J, H)
        return out[0] if single else out


def minimal_in_H4_PDE_flat_new(u_callable, use_compile=True, compile_kwargs=None):
    flat = FlatStereoPDEModel(u_callable)
    return _maybe_compile(flat, use_compile, compile_kwargs)