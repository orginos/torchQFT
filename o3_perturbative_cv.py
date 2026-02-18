import argparse
import time
import numpy as np
import torch as tr
from tqdm import tqdm

import O3 as qft
import integrators as integ
import update as upd


def shift2d(t, sx, sy):
    return tr.roll(t, shifts=(sx, sy), dims=(2, 3))


def zero_momentum_c2pt_sum(n, tau):
    # C(r) = sum_y n_y · n_{y+r}, with r = (tau, 0)
    n_z = shift2d(n, -tau, 0)
    return (n * n_z).sum(dim=1).sum(dim=(1, 2))


def G_S0(n):
    # S0 = -sum_<yw> n_y·n_w  =>  G_S0(y) = -sum_nn n_w
    return -(
        shift2d(n, -1, 0) + shift2d(n, 1, 0) +
        shift2d(n, 0, -1) + shift2d(n, 0, 1)
    )


def G_U1_zero_momentum(n, tau):
    # U1 summed over y with z = y + r, r = (tau, 0)
    rx = int(tau) % n.shape[2]
    ry = 0
    n_z = shift2d(n, -rx, -ry)
    Cyz = (n * n_z).sum(dim=1)

    G = tr.zeros_like(n)
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in deltas:
        n_y_delta = shift2d(n, -dx, -dy)
        n_z_delta = shift2d(n_z, -dx, -dy)

        Ayw = (n * n_y_delta).sum(dim=1)
        Azw = (n_z * n_z_delta).sum(dim=1)

        # A: -(1/20) (n_{y+delta} · n_z)
        G += (-0.05) * shift2d(n_z, dx, dy)
        G += (-0.05) * shift2d(n_y_delta, rx, ry)

        # B: +(1/40) (n_y·n_{y+delta})(n_y·n_z)
        G += 0.025 * (Cyz[:, None, :, :] * n_y_delta + Ayw[:, None, :, :] * n_z)
        G += 0.025 * shift2d(Cyz[:, None, :, :] * n, dx, dy)
        G += 0.025 * shift2d(Ayw[:, None, :, :] * n, rx, ry)

        # C: -(1/20) (n_{z+delta} · n_y)
        G += (-0.05) * shift2d(n, rx + dx, ry + dy)
        G += (-0.05) * n_z_delta

        # D: +(1/40) (n_z·n_{z+delta})(n_y·n_z)
        G += 0.025 * (Azw[:, None, :, :] * n_z)
        vec_z = Cyz[:, None, :, :] * n_z_delta + Azw[:, None, :, :] * n
        G += 0.025 * shift2d(vec_z, rx, ry)
        G += 0.025 * shift2d(Cyz[:, None, :, :] * n_z, rx + dx, ry + dy)

    return G


def lie_contract_sum(n, Gf, Gg):
    # sum_sites [Gf·Gg - (n·Gf)(n·Gg)]
    dot_fg = (Gf * Gg).sum(dim=1)
    dot_nf = (n * Gf).sum(dim=1)
    dot_ng = (n * Gg).sum(dim=1)
    return (dot_fg - dot_nf * dot_ng).sum(dim=(1, 2))


def dU1dS0_zero_momentum(n, tau):
    GU1 = G_U1_zero_momentum(n, tau)
    GS0 = G_S0(n)
    return lie_contract_sum(n, GU1, GS0)


parser = argparse.ArgumentParser()
parser.add_argument('-tau'   , type=int,   default=2)
parser.add_argument('-Nwarm' , type=int,   default=1000)
parser.add_argument('-Nskip' , type=int,   default=5)
parser.add_argument('-Nmeas' , type=int,   default=50, help="number of measurements")
parser.add_argument('-beta'  , type=float, default=0.5)
parser.add_argument('-hmc_bs', type=int,   default=128)
parser.add_argument('-L'     , type=int,   default=8, help="Lattice size")
parser.add_argument('-Nmd'   , type=int,   default=2, help="md steps")

args = parser.parse_args()

if tr.backends.mps.is_available():
    device = tr.device("mps")
else:
    print("MPS device not found.")
    device = "cpu"
device = tr.device("cpu")  # keep things simple

L = args.L
lat = [L, L]
beta = args.beta
Nwarm = args.Nwarm
Nskip = args.Nskip
Nmeas = args.Nmeas
hmc_batch_size = args.hmc_bs

Vol = np.prod(lat)

if L < 3 or (args.tau % L) in (0, 1, L - 1):
    print("Warning: the first-order formula assumes |r|>1. Choose tau with |tau|>1 mod L.")

sg = qft.O3(lat, beta, batch_size=hmc_batch_size, device=device)

phi = sg.hotStart()
mn2 = integ.minnorm2(sg.force, sg.evolveQ, args.Nmd, 1.0)

print("HMC Initial field characteristics: ", phi.shape, Vol, tr.mean(phi), tr.std(phi))

hmc = upd.hmc(T=sg, I=mn2, verbose=False)

tic = time.perf_counter()
phi = hmc.evolve(phi, Nwarm)
toc = time.perf_counter()
print(f"time {(toc - tic)*1.0e3/Nwarm:0.4f} ms per HMC trajecrory")
print("Acceptance: ", hmc.calc_Acceptance())

phi = hmc.evolve(phi, Nskip)

def jackknife_gain(o_blocks, oimp_blocks):
    # blocks: [Nmeas, B] tensors
    n_blocks = o_blocks.shape[0]
    gains = []
    for i in range(n_blocks):
        o_cat = tr.cat([o_blocks[:i], o_blocks[i + 1:]], dim=0).flatten()
        oi_cat = tr.cat([oimp_blocks[:i], oimp_blocks[i + 1:]], dim=0).flatten()
        v_o = o_cat.var(unbiased=False).item()
        v_i = oi_cat.var(unbiased=False).item()
        gains.append(v_o / v_i if v_i > 0 else float("inf"))
    gains = np.asarray(gains, dtype=np.float64)
    g_mean = gains.mean()
    g_err = np.sqrt((n_blocks - 1) * np.mean((gains - g_mean) ** 2))
    return g_mean, g_err

def block_mean_err(blocks):
    # blocks: [Nmeas, B]
    bmeans = blocks.mean(dim=1).cpu().numpy()
    mean = bmeans.mean()
    err = bmeans.std(ddof=1) / np.sqrt(len(bmeans)) if len(bmeans) > 1 else 0.0
    return mean, err

def fmt_two_sig(x):
    try:
        return f"{x:.2g}"
    except Exception:
        return str(x)

def fmt_mean_with_err(mean, err):
    try:
        if err == 0 or not np.isfinite(err):
            return f"{mean}"
        err_sig = float(f"{err:.2g}")
        if err_sig == 0:
            return f"{mean}"
        decimals = max(int(-np.floor(np.log10(abs(err_sig))) + 1), 0)
        return f"{mean:.{decimals}f}"
    except Exception:
        return f"{mean}"
with tr.no_grad():
    o_blocks = []
    oimp_blocks = []
    f_blocks = []
    for _ in tqdm(range(Nmeas), desc="meas", leave=False):
        O = zero_momentum_c2pt_sum(phi, args.tau)
        dU1dS0 = dU1dS0_zero_momentum(phi, args.tau)
        F_U = O - (beta ** 2) * dU1dS0
        O_imp = O - F_U  # improved estimator

        o_blocks.append(O.detach().cpu())
        oimp_blocks.append(O_imp.detach().cpu())
        f_blocks.append(F_U.detach().cpu())

        phi = hmc.evolve(phi, Nskip)

    o_blocks = tr.stack(o_blocks, dim=0)      # [Nmeas,B]
    oimp_blocks = tr.stack(oimp_blocks, dim=0)
    f_blocks = tr.stack(f_blocks, dim=0)

    o_all = o_blocks.flatten()
    oimp_all = oimp_blocks.flatten()

    var_O = o_all.var(unbiased=False).item()
    var_imp = oimp_all.var(unbiased=False).item()
    gain = (var_O / var_imp) if var_imp > 0 else float("inf")

    g_jk, g_err = jackknife_gain(o_blocks, oimp_blocks)

    o_mean, o_err = block_mean_err(o_blocks)
    oi_mean, oi_err = block_mean_err(oimp_blocks)
    f_mean, f_err = block_mean_err(f_blocks)

    print("Zero-momentum C(tau) mean: ", fmt_mean_with_err(o_mean, o_err), "+/-", fmt_two_sig(o_err), "(sum over sites)")
    print("Improved C(tau) mean      : ", fmt_mean_with_err(oi_mean, oi_err), "+/-", fmt_two_sig(oi_err), "(sum over sites)")
    print("Control variate mean      : ", fmt_mean_with_err(f_mean, f_err), "+/-", fmt_two_sig(f_err))
    print("Zero-momentum C(tau) var : ", var_O)
    print("Improved C(tau) var      : ", var_imp)
    print("Variance improvement    : ", gain)
    print("Jackknife gain          : ", g_jk, "+/-", fmt_two_sig(g_err))
