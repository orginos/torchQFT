import torch

# ---------- so(3) generators ----------
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

# ---------- Action with periodic BC (each undirected bond counted once) ----------
def action_o3(n, beta):
    nxp = torch.roll(n, shifts=-1, dims=2)
    nyp = torch.roll(n, shifts=-1, dims=3)
    bond_x = (n * nxp).sum(dim=1)
    bond_y = (n * nyp).sum(dim=1)
    S0 = -(bond_x + bond_y).sum(dim=(1, 2))
    return beta * S0  # [B]

# ---------- Neighbors ----------
def neighbors_2d(ix, iy, Lx, Ly):
    return [
        ((ix + 1) % Lx, iy),
        ((ix - 1) % Lx, iy),
        (ix, (iy + 1) % Ly),
        (ix, (iy - 1) % Ly),
    ]

# ---------- O, U0, U1 for fixed sites (y,z) with |y-z|>1 ----------
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

# ---------- F_U via Lie derivatives (forces grad-enabled) ----------
def compute_FU(n, beta, U_fn, lam=None):
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

        return (partial2U - partialU_dot_partialS).detach()

# ---------- Reweighting check: <F_U>_beta = 0 ----------
@torch.no_grad()
def check_mean_FU_by_reweighting(
    Lx=6, Ly=6, n_samples=200, batch=10, betas=(0.1, 0.3, 0.6),
    yx=0, yy=0, zx=0, zy=3, device="cpu", dtype=torch.float64
):
    """
    Samples configs from beta=0 (uniform spins), then reweights to beta.
    Checks <F_U>_beta ~ 0.
    """
    assert n_samples % batch == 0
    lam = so3_generators(device=device, dtype=dtype)

    print("Reweighting check of <F_U>_beta = 0 (importance sampling from beta=0)")
    print(f"Lattice {Lx}x{Ly}, samples={n_samples}, batch={batch}")
    print(f"Sites: y=({yx},{yy}), z=({zx},{zy})  (ensure |y-z|>1)\n")

    for beta in betas:
        U_fn = make_U0_plus_betaU1(yx, yy, zx, zy, beta)

        num = 0.0
        den = 0.0
        unweighted_mean = 0.0

        for _ in range(n_samples // batch):
            n = torch.randn((batch, 3, Lx, Ly), device=device, dtype=dtype)
            n = normalize_spins(n)

            S = action_o3(n, beta)              # [B]
            w = torch.exp(-S)                   # reweight from beta=0 to beta
            F = compute_FU(n, beta, U_fn, lam)  # [B]

            num += (w * F).sum().item()
            den += w.sum().item()
            unweighted_mean += F.mean().item()

        weighted_mean = num / den
        unweighted_mean /= (n_samples // batch)

        print(f"beta={beta: .3g}  <F_U>_beta (reweighted) = {weighted_mean: .3e}   "
              f"(unweighted mean over beta=0 samples = {unweighted_mean: .3e})")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    check_mean_FU_by_reweighting(
        Lx=6, Ly=6, n_samples=200, batch=10,
        betas=(0.1, 0.3, 0.6),
        yx=0, yy=0, zx=0, zy=3,
        device="cpu"
    )
    
