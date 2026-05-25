import torch

def D(f, xy):
    N = f.shape[0] # number of points
    out_shape = f.shape[1:] # shape of target space

    f_flat = f.reshape(N, -1) # shape (N, D)
    D = f_flat.shape[-1]
    
    # for each flattened output component compute gradient wrt xy:
    grads = []
    for j in range(D):
        g = torch.autograd.grad(
            outputs=f_flat[:, j].sum(),
            inputs=xy,
            create_graph=True,
            retain_graph=True,
        )[0]  # shape (N, 2)
        grads.append(g)

    # Stack over output dimension → shape (N, D, 2)
    J = torch.stack(grads, dim=1)

    # Unflatten the output dimension:
    return J.reshape(N, *out_shape, 2)


def pb_metric(f, xy):
    # only works when f has shape (N, K+1)
    N = f.shape[0]

    rho = f[:, 0] # first component
    J = D(f, xy) # shape (N, K+1, 2)
    g = torch.matmul(J.transpose(-1,-2), J) # shape (N, 2, 2)
    g *= (rho**-2).view(N, 1, 1)

    return g

def laplace_beltrami(f, g, g_inv, g_det, xy):
    N = f.shape[0] # number of points
    out_shape = f.shape[1:] # shape of target space

    f_flat = f.reshape(N, -1) # shape (N, D)
    
    Jf_flat = D(f_flat, xy) # shape (N, D, 2)
    sqrt_det_g = torch.sqrt(g_det)
    
    ans = sqrt_det_g.view(N, 1, 1) * torch.einsum('nij,ndj->nid', g_inv, Jf_flat) # shape (N, 2, D)
    ans = D(ans, xy) # shape (N, 2, D, 2)
    ans = torch.einsum('niki->nk', ans) # shape (N, D)
    ans = ans / sqrt_det_g.view(N, 1) # shape (N, D)
     
    return ans.reshape(N, *out_shape)

def inner_prod_1forms(a, b, g_inv):
    # assumes a has shape (N, ... , 2)
    # b has shape (N, ... , 2)
    # and g_inv has shape (N, 2, 2)
    # internal shapes of a and b might be different
    N = a.shape[0]
    
    a_flat = a.reshape(N, -1, 2)   # (N, A, 2)
    b_flat = b.reshape(N, -1, 2)   # (N, B, 2)
    result_flat = torch.einsum('nki,nij,nlj->nkl', a_flat, g_inv, b_flat)
    # reshape back to (N, *a_internal, *b_internal)
    result = result_flat.reshape(N, *a.shape[1:-1], *b.shape[1:-1])
    return result

def PDE(u, xy):
    # u has shape (N, K+1)
    # xy has shape (N, 2)
    N = u.shape[0]

    rho = u[:, 0] # shape (N,)
    v = u[:, 1:] # shape (N, K)

    # OPTIMIZATION 1: Compute Jacobian once instead of 3 times
    # Old: D(rho), D(v), and D(u) inside pb_metric
    # New: D(u) once, then slice
    J_u = D(u, xy) # shape (N, K+1, 2)
    drho = J_u[:, 0, :] # shape (N, 2)
    dv = J_u[:, 1:, :] # shape (N, K, 2)

    # OPTIMIZATION 2: Compute metric directly without calling pb_metric
    g = torch.matmul(J_u.transpose(-1,-2), J_u) # shape (N, 2, 2)
    g *= (rho**-2).view(N, 1, 1)

    # OPTIMIZATION 3: Explicit 2x2 determinant and inverse (faster than torch.inverse/det)
    a = g[:, 0, 0]
    b = g[:, 0, 1]
    c = g[:, 1, 0]
    d = g[:, 1, 1]

    # determinant
    g_det = a * d - b * c  # shape (N,)

    # inverse
    g_inv = torch.empty_like(g)
    g_inv[:, 0, 0] =  d / g_det
    g_inv[:, 0, 1] = -b / g_det
    g_inv[:, 1, 0] = -c / g_det
    g_inv[:, 1, 1] =  a / g_det

    norm_drho_sq = inner_prod_1forms(drho, drho, g_inv)
    inner_dv_dv = inner_prod_1forms(dv, dv, g_inv) # shape (N, K, K)
    norm_dv_sq = torch.einsum('nii->n', inner_dv_dv) # shape (N,)
    inner_drho_dv = inner_prod_1forms(drho, dv, g_inv) # shape (N, K)
    
    t_XdX = laplace_beltrami(rho, g, g_inv, g_det, xy) # shape (N,)
    t_XdX+= rho**-1 * (-norm_drho_sq + norm_dv_sq)
    t_XdX*= rho**-1

    t_XdY = laplace_beltrami(v, g, g_inv, g_det, xy) # shape (N, K)
    t_XdY+= (rho**-1).view(N, 1) * (-2*inner_drho_dv)
    t_XdY*= (rho**-1).view(N, 1)

    return torch.column_stack([t_XdX, t_XdY])