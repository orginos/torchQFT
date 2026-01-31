import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as tr
from tqdm import tqdm

import O3 as qft
import integrators as integ
import update as upd


def shift2d(t, sx, sy):
    return tr.roll(t, shifts=(sx, sy), dims=(2, 3))


def zero_momentum_c2pt_sum(n, tau):
    n_z = shift2d(n, -tau, 0)
    return (n * n_z).sum(dim=1).sum(dim=(1, 2))


def G_S0(n):
    return -(
        shift2d(n, -1, 0) + shift2d(n, 1, 0) +
        shift2d(n, 0, -1) + shift2d(n, 0, 1)
    )


def G_U1_zero_momentum(n, tau):
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

        G += (-0.05) * shift2d(n_z, dx, dy)
        G += (-0.05) * shift2d(n_y_delta, rx, ry)

        G += 0.025 * (Cyz[:, None, :, :] * n_y_delta + Ayw[:, None, :, :] * n_z)
        G += 0.025 * shift2d(Cyz[:, None, :, :] * n, dx, dy)
        G += 0.025 * shift2d(Ayw[:, None, :, :] * n, rx, ry)

        G += (-0.05) * shift2d(n, rx + dx, ry + dy)
        G += (-0.05) * n_z_delta

        G += 0.025 * (Azw[:, None, :, :] * n_z)
        vec_z = Cyz[:, None, :, :] * n_z_delta + Azw[:, None, :, :] * n
        G += 0.025 * shift2d(vec_z, rx, ry)
        G += 0.025 * shift2d(Cyz[:, None, :, :] * n_z, rx + dx, ry + dy)

    return G


def lie_contract_sum(n, Gf, Gg):
    dot_fg = (Gf * Gg).sum(dim=1)
    dot_nf = (n * Gf).sum(dim=1)
    dot_ng = (n * Gg).sum(dim=1)
    return (dot_fg - dot_nf * dot_ng).sum(dim=(1, 2))


def dU1dS0_zero_momentum(n, tau):
    GU1 = G_U1_zero_momentum(n, tau)
    GS0 = G_S0(n)
    return lie_contract_sum(n, GU1, GS0)


def block_mean_err(block_means):
    valid = np.isfinite(block_means)
    if not np.any(valid):
        return None, None
    vals = block_means[valid]
    mean = vals.mean()
    err = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, err


def jackknife_gain_from_sums(sum_o, sumsq_o, sum_oi, sumsq_oi, block_sums_o, block_sumsq_o,
                             block_sums_oi, block_sumsq_oi, block_count):
    n_blocks = len(block_sums_o)
    gains = []
    n_total = block_count * n_blocks
    for i in range(n_blocks):
        n = n_total - block_count
        so = sum_o - block_sums_o[i]
        soi = sum_oi - block_sums_oi[i]
        sso = sumsq_o - block_sumsq_o[i]
        ssoi = sumsq_oi - block_sumsq_oi[i]
        var_o = sso / n - (so / n) ** 2
        var_oi = ssoi / n - (soi / n) ** 2
        gains.append(var_o / var_oi if var_oi > 0 else float("inf"))
    gains = np.asarray(gains, dtype=np.float64)
    g_mean = gains.mean()
    g_err = np.sqrt((n_blocks - 1) * np.mean((gains - g_mean) ** 2)) if n_blocks > 1 else 0.0
    return g_mean, g_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-Nwarm', type=int, default=1000)
    parser.add_argument('-Nskip', type=int, default=5)
    parser.add_argument('-Nmeas', type=int, default=50)
    parser.add_argument('-beta', type=float, default=0.5)
    parser.add_argument('-hmc_bs', type=int, default=128)
    parser.add_argument('-L', type=int, default=8, help="Lattice size")
    parser.add_argument('-Nmd', type=int, default=2, help="md steps")
    parser.add_argument('--out_dir', required=True, help="directory to save db")
    parser.add_argument('--tag', default="", help="tag for output filename")
    args = parser.parse_args()

    if tr.backends.mps.is_available():
        device = tr.device("mps")
    else:
        print("MPS device not found.")
        device = "cpu"
    device = tr.device("cpu")

    L = args.L
    lat = [L, L]
    beta = args.beta
    Nwarm = args.Nwarm
    Nskip = args.Nskip
    Nmeas = args.Nmeas
    hmc_batch_size = args.hmc_bs

    taus = list(range(0, L))
    if len(taus) == 0:
        raise ValueError("Need L>=1.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"o3_c2pt_cv_L{L}_b{beta}{tag}.json"

    sg = qft.O3(lat, beta, batch_size=hmc_batch_size, device=device)

    phi = sg.hotStart()
    mn2 = integ.minnorm2(sg.force, sg.evolveQ, args.Nmd, 1.0)
    hmc = upd.hmc(T=sg, I=mn2, verbose=False)

    tic = time.perf_counter()
    phi = hmc.evolve(phi, Nwarm)
    toc = time.perf_counter()
    print(f"time {(toc - tic)*1.0e3/Nwarm:0.4f} ms per HMC trajecrory")
    print("Acceptance: ", hmc.calc_Acceptance())

    phi = hmc.evolve(phi, Nskip)

    n_taus = len(taus)
    invalid = np.zeros(n_taus, dtype=bool)
    for i, tau in enumerate(taus):
        if tau < 2 or tau == (L - 1):
            invalid[i] = True
    block_mean_O = np.zeros((Nmeas, n_taus), dtype=np.float64)
    block_mean_Oi = np.full((Nmeas, n_taus), np.nan, dtype=np.float64)
    block_mean_F = np.full((Nmeas, n_taus), np.nan, dtype=np.float64)

    block_sum_O = np.zeros((Nmeas, n_taus), dtype=np.float64)
    block_sumsq_O = np.zeros((Nmeas, n_taus), dtype=np.float64)
    block_sum_Oi = np.full((Nmeas, n_taus), np.nan, dtype=np.float64)
    block_sumsq_Oi = np.full((Nmeas, n_taus), np.nan, dtype=np.float64)
    block_sum_OOi = np.full((Nmeas, n_taus), np.nan, dtype=np.float64)

    with tr.no_grad():
        for m in tqdm(range(Nmeas), desc="meas"):
            for t_idx, tau in enumerate(taus):
                O = zero_momentum_c2pt_sum(phi, tau)
                if invalid[t_idx]:
                    o = O.detach().cpu().numpy()
                    block_mean_O[m, t_idx] = o.mean()
                    block_sum_O[m, t_idx] = o.sum()
                    block_sumsq_O[m, t_idx] = (o * o).sum()
                else:
                    dU1dS0 = dU1dS0_zero_momentum(phi, tau)
                    F_U = O - (beta ** 2) * dU1dS0
                    O_imp = O - F_U

                    o = O.detach().cpu().numpy()
                    oi = O_imp.detach().cpu().numpy()
                    f = F_U.detach().cpu().numpy()

                    block_mean_O[m, t_idx] = o.mean()
                    block_mean_Oi[m, t_idx] = oi.mean()
                    block_mean_F[m, t_idx] = f.mean()

                    block_sum_O[m, t_idx] = o.sum()
                    block_sumsq_O[m, t_idx] = (o * o).sum()
                    block_sum_Oi[m, t_idx] = oi.sum()
                    block_sumsq_Oi[m, t_idx] = (oi * oi).sum()
                    block_sum_OOi[m, t_idx] = (o * oi).sum()

            phi = hmc.evolve(phi, Nskip)

    means_O = []
    errs_O = []
    means_Oi = []
    errs_Oi = []
    means_F = []
    errs_F = []
    gains = []
    gains_jk = []
    gains_jk_err = []
    covs = []
    covs_jk = []
    covs_jk_err = []

    count = hmc_batch_size
    for t_idx, _ in enumerate(taus):
        m_o, e_o = block_mean_err(block_mean_O[:, t_idx])
        m_oi, e_oi = block_mean_err(block_mean_Oi[:, t_idx])
        m_f, e_f = block_mean_err(block_mean_F[:, t_idx])

        means_O.append(m_o)
        errs_O.append(e_o)
        means_Oi.append(m_oi)
        errs_Oi.append(e_oi)
        means_F.append(m_f)
        errs_F.append(e_f)

        if invalid[t_idx]:
            gains.append(None)
            gains_jk.append(None)
            gains_jk_err.append(None)
            covs.append(None)
            covs_jk.append(None)
            covs_jk_err.append(None)
            continue

        sum_o = np.nansum(block_sum_O[:, t_idx])
        sumsq_o = np.nansum(block_sumsq_O[:, t_idx])
        sum_oi = np.nansum(block_sum_Oi[:, t_idx])
        sumsq_oi = np.nansum(block_sumsq_Oi[:, t_idx])
        sum_ooi = np.nansum(block_sum_OOi[:, t_idx])

        n_total = count * Nmeas
        var_o = sumsq_o / n_total - (sum_o / n_total) ** 2
        var_oi = sumsq_oi / n_total - (sum_oi / n_total) ** 2
        gain = var_o / var_oi if var_oi > 0 else float("inf")

        g_jk, g_err = jackknife_gain_from_sums(
            sum_o, sumsq_o, sum_oi, sumsq_oi,
            block_sum_O[:, t_idx], block_sumsq_O[:, t_idx],
            block_sum_Oi[:, t_idx], block_sumsq_Oi[:, t_idx],
            count,
        )

        cov = sum_ooi / n_total - (sum_o / n_total) * (sum_oi / n_total)
        if var_o > 0 and var_oi > 0:
            cov_norm = cov / np.sqrt(var_o * var_oi)
        else:
            cov_norm = None

        cov_jk = []
        for i in range(Nmeas):
            n = n_total - count
            so = sum_o - block_sum_O[i, t_idx]
            soi = sum_oi - block_sum_Oi[i, t_idx]
            soo = sum_ooi - block_sum_OOi[i, t_idx]
            sso = sumsq_o - block_sumsq_O[i, t_idx]
            ssoi = sumsq_oi - block_sumsq_Oi[i, t_idx]
            var_o_i = sso / n - (so / n) ** 2
            var_oi_i = ssoi / n - (soi / n) ** 2
            cov_i = soo / n - (so / n) * (soi / n)
            if var_o_i > 0 and var_oi_i > 0:
                cov_jk.append(cov_i / np.sqrt(var_o_i * var_oi_i))
        cov_jk = np.asarray(cov_jk, dtype=np.float64)
        if len(cov_jk) > 0:
            cov_jk_mean = cov_jk.mean()
            cov_jk_err = np.sqrt((len(cov_jk) - 1) * np.mean((cov_jk - cov_jk_mean) ** 2)) if len(cov_jk) > 1 else 0.0
        else:
            cov_jk_mean = None
            cov_jk_err = None

        gains.append(gain)
        gains_jk.append(g_jk)
        gains_jk_err.append(g_err)
        covs.append(cov_norm)
        covs_jk.append(cov_jk_mean)
        covs_jk_err.append(cov_jk_err)

    payload = {
        "meta": {
            "L": L,
            "beta": beta,
            "Nwarm": Nwarm,
            "Nskip": Nskip,
            "Nmeas": Nmeas,
            "hmc_bs": hmc_batch_size,
            "Nmd": args.Nmd,
        },
        "taus": taus,
        "c2pt": {"mean": means_O, "err": errs_O},
        "c2pt_improved": {"mean": means_Oi, "err": errs_Oi},
        "control_variate": {"mean": means_F, "err": errs_F},
        "gain": {"mean": gains, "jk_mean": gains_jk, "jk_err": gains_jk_err},
        "covariance": {"mean": covs, "jk_mean": covs_jk, "jk_err": covs_jk_err},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
