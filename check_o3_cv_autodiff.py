import torch

# ============================================================
# Basic helpers
# ============================================================
def normalize_spins(n, eps=1e-12):
    # n: [B,3,Lx,Ly]
    norm = torch.sqrt((n * n).sum(dim=1, keepdim=True).clamp_min(eps))
    return n / norm

def dot3(a, b):
    # a,b: [B,3] -> [B]
    return (a * b).sum(dim=1)

def neighbors_2d(ix, iy, Lx, Ly):
    # periodic 2D NN
    return [
        ((ix + 1) % Lx, iy),
        ((ix - 1) % Lx, iy),
        (ix, (iy + 1) % Ly),
        (ix, (iy - 1) % Ly),
    ]

# ============================================================
# Action S = beta S0 with periodic BC, each bond counted once
# ============================================================
def action_o3(n, beta):
    nxp = torch.roll(n, shifts=-1, dims=2)
    nyp = torch.roll(n, shifts=-1, dims=3)
    bond_x = (n * nxp).sum(dim=1)
    bond_y = (n * nyp).sum(dim=1)
    S0 = -(bond_x + bond_y).sum(dim=(1, 2))
    return beta * S0  # [B]

# ============================================================
# so(3) generators (for autograd Lie-derivative implementation)
# Convention: (lambda^a)_{bc} = eps_{abc}
# ============================================================
def so3_generators(device=None, dtype=torch.float64):
    lam = torch.zeros((3, 3, 3), device=device, dtype=dtype)
    lam[0, 1, 2] = -1.0; lam[0, 2, 1] =  1.0
    lam[1, 2, 0] = -1.0; lam[1, 0, 2] =  1.0
    lam[2, 0, 1] = -1.0; lam[2, 1, 0] =  1.0
    return lam

def matvec3(M, v):
    # M: [3,3], v: [B,3] -> [B,3]
    return v @ M.T

# ============================================================
# U0 and U1 (fixed sites (yx,yy) and (zx,zy), assume |y-z|>1)
# ============================================================
def U0_fixed(n, yx, yy, zx, zy):
    ny = n[:, :, yx, yy]
    nz = n[:, :, zx, zy]
    return -0.25 * dot3(ny, nz)  # [B]

def U1_fixed(n, yx, yy, zx, zy):
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

    return s_y + s_z  # [B]

def O_fixed(n, yx, yy, zx, zy):
    ny = n[:, :, yx, yy]
    nz = n[:, :, zx, zy]
    return dot3(ny, nz)  # [B]

# ============================================================
# Analytic Lie-contraction using the projector identity:
# If ∂^a f = (lambda^a n)·Gf, ∂^a g = (lambda^a n)·Gg then
#   sum_a ∂^a f ∂^a g = Gf·Gg - (n·Gf)(n·Gg)
# We implement it site-summed.
# ============================================================
def lie_contract_sum(n, Gf, Gg):
    """
    n:  [B,3,Lx,Ly]
    Gf: [B,3,Lx,Ly] (ambient gradients)
    Gg: [B,3,Lx,Ly]
    returns: [B]  sum_{sites} [Gf·Gg - (n·Gf)(n·Gg)]
    """
    dot_fg = (Gf * Gg).sum(dim=1)           # [B,Lx,Ly]
    dot_nf = (n  * Gf).sum(dim=1)           # [B,Lx,Ly]
    dot_ng = (n  * Gg).sum(dim=1)           # [B,Lx,Ly]
    val = dot_fg - dot_nf * dot_ng          # [B,Lx,Ly]
    return val.sum(dim=(1, 2))              # [B]

# ============================================================
# Ambient gradients G_{S0}, G_{U0}, G_{U1}
# ============================================================
def G_S0(n):
    """
    S0 = -sum_<yw> n_y·n_w
    => G_{S0}(y) = - sum_{w in nn(y)} n_w
    """
    # n: [B,3,Lx,Ly]
    # periodic nn sum
    n_xp = torch.roll(n, shifts=-1, dims=2)
    n_xm = torch.roll(n, shifts=+1, dims=2)
    n_yp = torch.roll(n, shifts=-1, dims=3)
    n_ym = torch.roll(n, shifts=+1, dims=3)
    return -(n_xp + n_xm + n_yp + n_ym)  # [B,3,Lx,Ly]

def G_U0(n, yx, yy, zx, zy):
    B, _, Lx, Ly = n.shape
    G = torch.zeros_like(n)  # [B,3,Lx,Ly]
    ny = n[:, :, yx, yy]
    nz = n[:, :, zx, zy]
    G[:, :, yx, yy] += (-0.25) * nz
    G[:, :, zx, zy] += (-0.25) * ny
    return G

def G_U1(n, yx, yy, zx, zy):
    """
    Builds ambient gradient field of U1 (nonzero only on y,z and their nn sites).
    """
    B, _, Lx, Ly = n.shape
    G = torch.zeros_like(n)

    ny = n[:, :, yx, yy]
    nz = n[:, :, zx, zy]
    Cyz = dot3(ny, nz)  # [B]

    # --- terms around y ---
    for (wx, wy) in neighbors_2d(yx, yy, Lx, Ly):
        nw = n[:, :, wx, wy]
        Ayw = dot3(ny, nw)      # (n_y·n_w)
        # term t1 = -(1/20) (n_w·n_z)
        G[:, :, wx, wy] += (-1.0 / 20.0) * nz
        G[:, :, zx, zy] += (-1.0 / 20.0) * nw
        # term t2 = +(1/40) (n_y·n_w)(n_y·n_z) = (1/40) Ayw * Cyz
        # gradients:
        # d/d n_y: (1/40)[Cyz*nw + Ayw*nz]
        G[:, :, yx, yy] += (1.0 / 40.0) * (Cyz[:, None] * nw + Ayw[:, None] * nz)
        # d/d n_w: (1/40)[Cyz*n_y]
        G[:, :, wx, wy] += (1.0 / 40.0) * (Cyz[:, None] * ny)
        # d/d n_z: (1/40)[Ayw*n_y]
        G[:, :, zx, zy] += (1.0 / 40.0) * (Ayw[:, None] * ny)

    # --- terms around z ---
    for (wx, wy) in neighbors_2d(zx, zy, Lx, Ly):
        nw = n[:, :, wx, wy]
        Azw = dot3(nz, nw)      # (n_z·n_w)
        # term t1 = -(1/20) (n_w·n_y)
        G[:, :, wx, wy] += (-1.0 / 20.0) * ny
        G[:, :, yx, yy] += (-1.0 / 20.0) * nw
        # term t2 = +(1/40) (n_z·n_w)(n_y·n_z) = (1/40) Azw * Cyz
        # gradients:
        # d/d n_z: (1/40)[Cyz*nw + Azw*n_y]  (since dCyz/dn_z = n_y)
        G[:, :, zx, zy] += (1.0 / 40.0) * (Cyz[:, None] * nw + Azw[:, None] * ny)
        # d/d n_w: (1/40)[Cyz*n_z]
        G[:, :, wx, wy] += (1.0 / 40.0) * (Cyz[:, None] * nz)
        # d/d n_y: (1/40)[Azw*n_z]  (since dCyz/dn_y = n_z)
        G[:, :, yx, yy] += (1.0 / 40.0) * (Azw[:, None] * nz)

    return G

# ============================================================
# Analytic ∂^2 U for U0 and U1
# - ∂^2 U0 is exactly O
# - ∂^2 U1 can be computed two ways:
#   (i) via monomial Laplacian identities (closure), or
#   (ii) via BU0 = (∂U0·∂S0) using lie_contract_sum
# We'll compute (i) explicitly to "show we have all we need".
# ============================================================
def partial2_U0_analytic(n, yx, yy, zx, zy):
    return O_fixed(n, yx, yy, zx, zy)  # because ∂^2(n_y·n_z) = -4(n_y·n_z)

def partial2_U1_analytic_monomials(n, yx, yy, zx, zy):
    """
    Uses:
      ∂^2 (n_w·n_z) = -4 (n_w·n_z)
      ∂^2 [(n_y·n_w)(n_y·n_z)] = 2(n_w·n_z) - 10[(n_y·n_w)(n_y·n_z)]
    (and symmetric y<->z)
    """
    B, _, Lx, Ly = n.shape
    ny = n[:, :, yx, yy]
    nz = n[:, :, zx, zy]
    Cyz = dot3(ny, nz)

    out = torch.zeros((B,), device=n.device, dtype=n.dtype)

    # y-side
    for (wx, wy) in neighbors_2d(yx, yy, Lx, Ly):
        nw = n[:, :, wx, wy]
        f1 = dot3(nw, nz)            # n_w·n_z
        f2 = dot3(ny, nw) * Cyz      # (n_y·n_w)(n_y·n_z)

        # U1 has: (-1/20) f1 + (1/40) f2
        # Apply ∂^2:
        # ∂^2 f1 = -4 f1
        # ∂^2 f2 = 2 f1 - 10 f2
        out += (-1.0/20.0) * (-4.0 * f1) + (1.0/40.0) * (2.0 * f1 - 10.0 * f2)

    # z-side
    for (wx, wy) in neighbors_2d(zx, zy, Lx, Ly):
        nw = n[:, :, wx, wy]
        f1 = dot3(nw, ny)            # n_w·n_y
        f2 = dot3(nz, nw) * Cyz      # (n_z·n_w)(n_y·n_z)

        out += (-1.0/20.0) * (-4.0 * f1) + (1.0/40.0) * (2.0 * f1 - 10.0 * f2)

    return out

# ============================================================
# Analytic F_U for U = U0 + beta U1 with S = beta S0:
#   F = ∂^2(U0+beta U1) - beta * (∂(U0+beta U1) · ∂S0)
#     = ∂^2 U0 + beta ∂^2 U1 - beta (∂U0·∂S0) - beta^2 (∂U1·∂S0)
# For our U1, beta terms cancel pointwise, so:
#   F = O - beta^2 (∂U1·∂S0)
# We'll compute both ways and compare.
# ============================================================
def FU_analytic_full(n, beta, yx, yy, zx, zy, use_partial2U1_monomials=True):
    GS0 = G_S0(n)
    GU0 = G_U0(n, yx, yy, zx, zy)
    GU1 = G_U1(n, yx, yy, zx, zy)

    # ∂U0·∂S0 and ∂U1·∂S0 via projector identity
    dU0dS0 = lie_contract_sum(n, GU0, GS0)  # [B]
    dU1dS0 = lie_contract_sum(n, GU1, GS0)  # [B]

    p2U0 = partial2_U0_analytic(n, yx, yy, zx, zy)

    if use_partial2U1_monomials:
        p2U1 = partial2_U1_analytic_monomials(n, yx, yy, zx, zy)
    else:
        # For the constructed U1 (|y-z|>1), ∂^2 U1 == (∂U0·∂S0) pointwise
        p2U1 = dU0dS0

    F = p2U0 + beta * p2U1 - beta * dU0dS0 - (beta ** 2) * dU1dS0
    return F, {"p2U0": p2U0, "p2U1": p2U1, "dU0dS0": dU0dS0, "dU1dS0": dU1dS0}

def FU_analytic_simplified(n, beta, yx, yy, zx, zy):
    # F = O - beta^2 (∂U1·∂S0)
    GS0 = G_S0(n)
    GU1 = G_U1(n, yx, yy, zx, zy)
    dU1dS0 = lie_contract_sum(n, GU1, GS0)
    return O_fixed(n, yx, yy, zx, zy) - (beta ** 2) * dU1dS0

# ============================================================
# Autograd F_U (your definition)
# ============================================================
def compute_FU_autograd(n, beta, yx, yy, zx, zy, lam=None):
    if lam is None:
        lam = so3_generators(device=n.device, dtype=n.dtype)

    def U_fn(n_in):
        return U0_fixed(n_in, yx, yy, zx, zy) + beta * U1_fixed(n_in, yx, yy, zx, zy)

    with torch.enable_grad():
        n_req = n.clone().detach().requires_grad_(True)
        S = action_o3(n_req, beta)  # [B]
        gradS = torch.autograd.grad(S.sum(), n_req, create_graph=True)[0]

        U = U_fn(n_req)
        gradU = torch.autograd.grad(U.sum(), n_req, create_graph=True)[0]

        B, _, Lx, Ly = n_req.shape
        partial2U = torch.zeros((B,), device=n.device, dtype=n.dtype)
        partialU_dot_partialS = torch.zeros((B,), device=n.device, dtype=n.dtype)

        for a in range(3):
            for ix in range(Lx):
                for iy in range(Ly):
                    n_site = n_req[:, :, ix, iy]
                    dn_site = matvec3(lam[a], n_site)

                    dU = (gradU[:, :, ix, iy] * dn_site).sum(dim=1)
                    dS = (gradS[:, :, ix, iy] * dn_site).sum(dim=1)
                    partialU_dot_partialS = partialU_dot_partialS + dU * dS

                    grad_dU = torch.autograd.grad(dU.sum(), n_req, create_graph=True)[0]
                    d2U = (grad_dU[:, :, ix, iy] * dn_site).sum(dim=1)
                    partial2U = partial2U + d2U

        return (partial2U - partialU_dot_partialS).detach()

# ============================================================
# Checks
# ============================================================
@torch.no_grad()
def run_checks():
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    B, Lx, Ly = 4, 6, 6
    n = normalize_spins(torch.randn((B, 3, Lx, Ly), device=device))

    # choose sites with |y-z|>1
    yx, yy = 0, 0
    zx, zy = 0, 3

    lam = so3_generators(device=device, dtype=n.dtype)

    for beta in [0.05, 0.1, 0.2, 0.4]:
        F_an, parts = FU_analytic_full(n, beta, yx, yy, zx, zy, use_partial2U1_monomials=True)
        F_an_s = FU_analytic_simplified(n, beta, yx, yy, zx, zy)
        F_ag = compute_FU_autograd(n, beta, yx, yy, zx, zy, lam=lam)

        err_full = (F_ag - F_an).abs()
        err_simp = (F_ag - F_an_s).abs()

        # also verify the key cancellation: ∂^2 U1 == (∂U0·∂S0)
        cancel_err = (parts["p2U1"] - parts["dU0dS0"]).abs()

        print(f"\nbeta={beta}")
        print(f"  |F_autograd - F_analytic(full)|  mean={err_full.mean().item():.3e}  max={err_full.max().item():.3e}")
        print(f"  |F_autograd - F_analytic(simpl)| mean={err_simp.mean().item():.3e}  max={err_simp.max().item():.3e}")
        print(f"  |(∂^2U1)_analytic - (∂U0·∂S0)_analytic| mean={cancel_err.mean().item():.3e}  max={cancel_err.max().item():.3e}")

        # show the expected residual scaling explicitly:
        R = (F_ag - O_fixed(n, yx, yy, zx, zy)).abs().mean().item()
        print(f"  mean|F - O| = {R:.3e}  (should scale ~ beta^2)")

if __name__ == "__main__":
    run_checks()
    
