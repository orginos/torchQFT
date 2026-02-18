import torch

def so3_generators(device=None, dtype=torch.float64):
    lam = torch.zeros((3, 3, 3), device=device, dtype=dtype)
    lam[0, 1, 2] = -1.0; lam[0, 2, 1] =  1.0
    lam[1, 2, 0] = -1.0; lam[1, 0, 2] =  1.0
    lam[2, 0, 1] = -1.0; lam[2, 1, 0] =  1.0
    return lam

def normalize_spins(n, eps=1e-12):
    norm = torch.sqrt((n * n).sum(dim=1, keepdim=True).clamp_min(eps))
    return n / norm

def dot3(a, b):
    return (a * b).sum(dim=1)

def matvec3(M, v):
    return v @ M.T

def action_o3(n, beta):
    # periodic BC, count each undirected bond once using forward rolls
    nxp = torch.roll(n, shifts=-1, dims=2)
    nyp = torch.roll(n, shifts=-1, dims=3)
    bond_x = (n * nxp).sum(dim=1)
    bond_y = (n * nyp).sum(dim=1)
    S0 = -(bond_x + bond_y).sum(dim=(1, 2))
    return beta * S0

def neighbors_2d(ix, iy, Lx, Ly):
    return [
        ((ix + 1) % Lx, iy),
        ((ix - 1) % Lx, iy),
        (ix, (iy + 1) % Ly),
        (ix, (iy - 1) % Ly),
    ]

def make_O_fixed_sites(yx, yy, zx, zy):
    def O(n):
        ny = n[:, :, yx, yy]
        nz = n[:, :, zx, zy]
        return dot3(ny, nz)
    return O

def make_U0_fixed_sites(yx, yy, zx, zy):
    def U0(n):
        ny = n[:, :, yx, yy]
        nz = n[:, :, zx, zy]
        return -0.25 * dot3(ny, nz)
    return U0

def make_U1_fixed_sites(yx, yy, zx, zy):
    def U1(n):
        B, _, Lx, Ly = n.shape
        ny = n[:, :, yx, yy]
        nz = n[:, :, zx, zy]
        Cyz = dot3(ny, nz)

        s_y = torch.zeros((B,), device=n.device, dtype=n.dtype)
        for (wx, wy) in neighbors_2d(yx, yy, Lx, Ly):
            nw = n[:, :, wx, wy]
            f1 = dot3(nw, nz)
            f2 = dot3(ny, nw) * Cyz
            s_y = s_y + (-1.0 / 20.0) * f1 + (1.0 / 40.0) * f2

        s_z = torch.zeros((B,), device=n.device, dtype=n.dtype)
        for (wx, wy) in neighbors_2d(zx, zy, Lx, Ly):
            nw = n[:, :, wx, wy]
            f1 = dot3(nw, ny)
            f2 = dot3(nz, nw) * Cyz
            s_z = s_z + (-1.0 / 20.0) * f1 + (1.0 / 40.0) * f2

        return s_y + s_z
    return U1

def make_U0_plus_betaU1(yx, yy, zx, zy, beta):
    U0 = make_U0_fixed_sites(yx, yy, zx, zy)
    U1 = make_U1_fixed_sites(yx, yy, zx, zy)
    def U(n):
        return U0(n) + beta * U1(n)
    return U

def compute_FU(n, beta, U_fn, lam=None):
    """
    F_U = ∂^2 U - (∂U · ∂S) using Lie-directional derivatives.
    Robust: forces grad enabled even if caller uses torch.no_grad().
    """
    if lam is None:
        lam = so3_generators(device=n.device, dtype=n.dtype)

    with torch.enable_grad():
        n = n.clone().detach().requires_grad_(True)

        S = action_o3(n, beta)  # [B]
        gradS = torch.autograd.grad(S.sum(), n, create_graph=True)[0]  # [B,3,Lx,Ly]

        U = U_fn(n)
        if U.ndim == 0:
            U = U.expand(n.shape[0])
        gradU = torch.autograd.grad(U.sum(), n, create_graph=True)[0]

        B, _, Lx, Ly = n.shape
        partial2U = torch.zeros((B,), device=n.device, dtype=n.dtype)
        partialU_dot_partialS = torch.zeros((B,), device=n.device, dtype=n.dtype)

        for a in range(3):
            for ix in range(Lx):
                for iy in range(Ly):
                    n_site = n[:, :, ix, iy]
                    dn_site = matvec3(lam[a], n_site)

                    dU = (gradU[:, :, ix, iy] * dn_site).sum(dim=1)
                    dS = (gradS[:, :, ix, iy] * dn_site).sum(dim=1)

                    partialU_dot_partialS = partialU_dot_partialS + dU * dS

                    grad_dU = torch.autograd.grad(dU.sum(), n, create_graph=True)[0]
                    d2U = (grad_dU[:, :, ix, iy] * dn_site).sum(dim=1)
                    partial2U = partial2U + d2U

        F_U = partial2U - partialU_dot_partialS
        return F_U.detach()

def check_first_order_PDE(n, yx, yy, zx, zy, betas=(0.05, 0.1, 0.2)):
    lam = so3_generators(device=n.device, dtype=n.dtype)
    O_fn = make_O_fixed_sites(yx, yy, zx, zy)

    print("Checking residual R(beta) = [∂^2U - (∂U·∂S)] - O for U=U0+beta U1")
    print("Expect: ||R|| ~ O(beta^2) for |y-z|>1\n")

    prev_mean = None
    prev_beta = None

    for beta in betas:
        U_fn = make_U0_plus_betaU1(yx, yy, zx, zy, beta)
        F = compute_FU(n, beta=beta, U_fn=U_fn, lam=lam)
        O = O_fn(n).detach()
        R = F - O

        mean_abs = R.abs().mean().item()
        mean_abs_over_b2 = (R.abs() / (beta * beta)).mean().item()

        print(f"beta={beta: .3g}  mean|R|={mean_abs: .3e}   mean|R|/beta^2={mean_abs_over_b2: .3e}")

        if prev_mean is not None:
            ratio = mean_abs / prev_mean
            expected = (beta / prev_beta) ** 2
            print(f"         ratio={ratio: .3f}  expected~{expected: .3f}")

        prev_mean, prev_beta = mean_abs, beta

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    B, Lx, Ly = 2, 6, 6
    n = normalize_spins(torch.randn((B, 3, Lx, Ly)))

    # Pick two sites with |y-z|>1 (e.g. (0,0) and (0,3) on 6x6)
    yx, yy = 0, 0
    zx, zy = 0, 3

    check_first_order_PDE(n, yx, yy, zx, zy, betas=(0.05, 0.1, 0.2,0.4,0.8,2.0))
    
