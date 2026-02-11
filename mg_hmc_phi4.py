#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MG-HMC driver for phi^4 in 2d.

One "MG update" = n fine HMC steps + m coarse HMC steps.
Coarse steps use fixed fluctuations and autodiff force via phi4_sokal_mg.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import numpy as np
import torch as tr

import integrators as i
import update as u
import phi4 as s
from phi4_sokal_mg import (
    sokal_coarse_hmc_update,
    sokal_coarse_hmc_update_mapped,
    IdentityMap,
    LocalAdditiveCoupling2x2,
)


def correlation_length(L: int, ChiM: tr.Tensor, C2p: tr.Tensor) -> tr.Tensor:
    ratio = ChiM / C2p
    valid = (C2p > 0) & (ratio > 1.0)
    xi = tr.full_like(ratio, float("nan"))
    xi[valid] = 1 / (2 * np.sin(np.pi / L)) * tr.sqrt(ratio[valid] - 1.0)
    return xi


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.shape[0]
    x = x - np.mean(x)
    if n < 2:
        return np.ones(1, dtype=np.float64)
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conjugate(f))[:n]
    acov /= n
    if acov[0] == 0:
        return np.ones(1, dtype=np.float64)
    return acov / acov[0]


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> Tuple[float, int]:
    rho = autocorr_fft(x)
    if rho.size == 1:
        return float("nan"), 0
    tau = 0.5
    m = 0
    for t in range(1, len(rho)):
        tau += rho[t]
        m = t
        if t >= c * tau:
            break
    return float(tau), int(m)


def summarize_observable(name: str, data: np.ndarray) -> Dict[str, float]:
    # data shape: [Nmeas, B]
    n_meas, n_streams = data.shape
    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        return {"mean": float("nan"), "err": float("nan"), "tau_int": float("nan"), "tau_int_std": float("nan"), "n_eff": float("nan")}

    mean_all = float(np.nanmean(data))
    var_all = float(np.nanvar(data, ddof=1))

    taus = []
    for b in range(n_streams):
        tau_b, _ = integrated_autocorr_time(data[:, b])
        if np.isfinite(tau_b):
            taus.append(tau_b)
    tau_mean = float(np.mean(taus)) if taus else float("nan")
    tau_std = float(np.std(taus)) if taus else float("nan")

    n_eff = (n_meas * n_streams) / (2.0 * tau_mean) if tau_mean > 0 else float("nan")
    err = math.sqrt(2.0 * tau_mean * var_all / (n_meas * n_streams)) if tau_mean > 0 else float("nan")

    return {
        "mean": mean_all,
        "err": err,
        "tau_int": tau_mean,
        "tau_int_std": tau_std,
        "n_eff": n_eff,
    }


def mg_update(
    phi: tr.Tensor,
    fine_theory: s.phi4,
    fine_hmc: u.hmc,
    n_fine: int,
    m_coarse: int,
    block: int,
    coarse_Nmd: int,
    coarse_tau: float,
    coarse_integrator: str,
    vmap: tr.nn.Module,
) -> Tuple[tr.Tensor, int, int]:
    coarse_accepts = 0
    coarse_trials = 0
    if n_fine > 0:
        phi = fine_hmc.evolve(phi, n_fine)
    if m_coarse > 0:
        if isinstance(vmap, IdentityMap):
            phi, _, _, acc, trials = sokal_coarse_hmc_update(
                phi,
                fine_theory,
                Nmd=coarse_Nmd,
                tau=coarse_tau,
                Nhmc=m_coarse,
                block=block,
                integrator=coarse_integrator,
                verbose=False,
                return_acceptance=True,
            )
        else:
            phi, _, _, acc, trials = sokal_coarse_hmc_update_mapped(
                phi,
                fine_theory,
                vmap,
                Nmd=coarse_Nmd,
                tau=coarse_tau,
                Nhmc=m_coarse,
                block=block,
                integrator=coarse_integrator,
                verbose=False,
                return_acceptance=True,
            )
        coarse_trials += trials
        coarse_accepts += int(round(acc * trials))
    return phi, coarse_accepts, coarse_trials


def main():
    parser = argparse.ArgumentParser(description="MG-HMC for phi^4 in 2d")
    parser.add_argument("--L", type=int, default=256, help="Lattice size L (square LxL)")
    parser.add_argument("--lam", type=float, default=0.5, help="Phi^4 coupling lambda")
    parser.add_argument("--mass", type=float, default=-0.205, help="Bare mass m")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (parallel chains)")

    parser.add_argument("--Nwarm", type=int, default=1000, help="Warmup updates")
    parser.add_argument("--Nmeas", type=int, default=1000, help="Number of measurements")
    parser.add_argument("--Nskip", type=int, default=10, help="Updates between measurements")

    parser.add_argument("--n-fine", type=int, default=1, help="Fine HMC steps per update")
    parser.add_argument("--m-coarse", type=int, default=1, help="Coarse HMC steps per update")
    parser.add_argument("--block", type=int, default=2, help="Coarsening block size")

    parser.add_argument("--fine-Nmd", type=int, default=7, help="Fine HMC MD steps")
    parser.add_argument("--fine-tau", type=float, default=1.0, help="Fine HMC trajectory length")
    parser.add_argument("--fine-integrator", choices=["minnorm2", "leapfrog"], default="minnorm2")

    parser.add_argument("--coarse-Nmd", type=int, default=5, help="Coarse HMC MD steps")
    parser.add_argument("--coarse-tau", type=float, default=1.0, help="Coarse HMC trajectory length")
    parser.add_argument("--coarse-integrator", choices=["minnorm2", "leapfrog"], default="minnorm2")
    parser.add_argument("--map", choices=["identity", "additive"], default="identity", help="Pre-map before coarsening")
    parser.add_argument("--map-steps", type=int, default=1, help="Number of coupling steps in map")
    parser.add_argument("--map-hidden", type=int, default=32, help="Hidden width for map coupling net")
    parser.add_argument("--map-no-shift", action="store_true", help="Disable block shift in map")
    parser.add_argument("--map-trainable", action="store_true", help="Allow map parameters to be trainable")
    parser.add_argument("--map-checkpoint", type=str, default=None, help="Load map checkpoint")
    parser.add_argument(
        "--mode",
        choices=["mg", "baseline", "compare"],
        default="mg",
        help="Run MG-HMC, baseline HMC, or compare both",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (torch/numpy)")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tr.manual_seed(args.seed)

    device = tr.device("cpu")
    lat = [args.L, args.L]
    Vol = np.prod(lat)

    fine = s.phi4(lat, args.lam, args.mass, batch_size=args.batch_size, device=device)
    phi = fine.hotStart()

    if args.map == "identity":
        vmap = IdentityMap()
    else:
        vmap = LocalAdditiveCoupling2x2(
            n_steps=args.map_steps,
            hidden=args.map_hidden,
            use_shift=not args.map_no_shift,
        )
        if args.map_checkpoint is not None:
            ckpt = tr.load(args.map_checkpoint, map_location="cpu")
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            vmap.load_state_dict(state)
    if not args.map_trainable:
        for p in vmap.parameters():
            p.requires_grad_(False)

    if args.fine_integrator == "minnorm2":
        fine_I = i.minnorm2(fine.force, fine.evolveQ, args.fine_Nmd, args.fine_tau)
    else:
        fine_I = i.leapfrog(fine.force, fine.evolveQ, args.fine_Nmd, args.fine_tau)

    fine_hmc = u.hmc(T=fine, I=fine_I, verbose=False)

    def run_chain(run_mode: str):
        phi_local = phi.clone()
        fine_hmc.reset_Acceptance()
        coarse_accepts = 0
        coarse_trials = 0

        nf = args.n_fine if run_mode != "baseline" else max(args.n_fine, 1)
        mc = args.m_coarse if run_mode == "mg" else 0

        print(f"Warming up ({run_mode})...")
        for _ in range(args.Nwarm):
            phi_local, ca, ct = mg_update(
                phi_local,
                fine,
                fine_hmc,
                n_fine=nf,
                m_coarse=mc,
                block=args.block,
                coarse_Nmd=args.coarse_Nmd,
                coarse_tau=args.coarse_tau,
                coarse_integrator=args.coarse_integrator,
                vmap=vmap,
            )
            coarse_accepts += ca
            coarse_trials += ct

        fine_acc = fine_hmc.calc_Acceptance()
        coarse_acc = (coarse_accepts / coarse_trials) if coarse_trials > 0 else float("nan")
        print(f"Acceptance after warmup ({run_mode}): fine={fine_acc} coarse={coarse_acc}")

        fine_hmc.reset_Acceptance()
        coarse_accepts = 0
        coarse_trials = 0

        phase = tr.tensor(np.exp(1j * np.indices(tuple(lat))[0] * 2 * np.pi / lat[0]))
        lE = []
        lav_phi = []
        lchi_m = []
        lC2p = []
        lxi = []

        print(f"Measuring ({run_mode})...")
        for k in range(args.Nmeas):
            ttE = fine.action(phi_local) / Vol
            av_sigma = tr.mean(phi_local.view(fine.Bs, Vol), axis=1)
            chi_m = av_sigma * av_sigma * Vol
            p1_av_sig = tr.mean(phi_local.view(fine.Bs, Vol) * phase.view(1, Vol), axis=1)
            C2p = tr.real(tr.conj(p1_av_sig) * p1_av_sig) * Vol
            xi = correlation_length(lat[0], chi_m, C2p)

            lE.append(ttE.detach().cpu().numpy())
            lav_phi.append(av_sigma.detach().cpu().numpy())
            lchi_m.append(chi_m.detach().cpu().numpy())
            lC2p.append(C2p.detach().cpu().numpy())
            lxi.append(xi.detach().cpu().numpy())

            if k % 10 == 0:
                print(
                    "k=",
                    k,
                    "(av_phi,chi_m,c2p,E) ",
                    av_sigma.mean().item(),
                    chi_m.mean().item(),
                    C2p.mean().item(),
                    ttE.mean().item(),
                )

            for _ in range(args.Nskip):
                phi_local, ca, ct = mg_update(
                    phi_local,
                    fine,
                    fine_hmc,
                    n_fine=nf,
                    m_coarse=mc,
                    block=args.block,
                    coarse_Nmd=args.coarse_Nmd,
                    coarse_tau=args.coarse_tau,
                    coarse_integrator=args.coarse_integrator,
                    vmap=vmap,
                )
                coarse_accepts += ca
                coarse_trials += ct

        fine_acc = fine_hmc.calc_Acceptance()
        coarse_acc = (coarse_accepts / coarse_trials) if coarse_trials > 0 else float("nan")
        print(f"Acceptance after measurements ({run_mode}): fine={fine_acc} coarse={coarse_acc}")

        E = np.stack(lE, axis=0)
        av_phi = np.stack(lav_phi, axis=0)
        chi_m = np.stack(lchi_m, axis=0)
        C2p = np.stack(lC2p, axis=0)
        xi = np.stack(lxi, axis=0)

        results = {}
        print("\nResults with integrated autocorrelation time (Sokal windowing):")
        for name, arr in [
            ("E", E),
            ("av_phi", av_phi),
            ("chi_m", chi_m),
            ("C2p", C2p),
            ("xi", xi),
        ]:
            stats = summarize_observable(name, arr)
            results[name] = stats
            print(
                f"{name}: mean={stats['mean']}  err={stats['err']}  "
                f"tau_int={stats['tau_int']} (std={stats['tau_int_std']})  "
                f"n_eff={stats['n_eff']}"
            )
        return results

    if args.mode == "mg":
        run_chain("mg")
    elif args.mode == "baseline":
        run_chain("baseline")
    else:
        base_stats = run_chain("baseline")
        mg_stats = run_chain("mg")
        print("\nMG - baseline differences (mean ± combined err):")
        for name in ["E", "av_phi", "chi_m", "C2p", "xi"]:
            if name not in base_stats or name not in mg_stats:
                continue
            m0 = base_stats[name]["mean"]
            e0 = base_stats[name]["err"]
            m1 = mg_stats[name]["mean"]
            e1 = mg_stats[name]["err"]
            if not (np.isfinite(m0) and np.isfinite(m1) and np.isfinite(e0) and np.isfinite(e1)):
                diff = float("nan")
                err = float("nan")
            else:
                diff = m1 - m0
                err = math.sqrt(e0 * e0 + e1 * e1)
            print(f"{name}: {diff} ± {err}")


if __name__ == "__main__":
    main()
