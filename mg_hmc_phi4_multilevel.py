#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilevel MG-HMC driver for phi^4 using JSON config.
Does not modify the single-level driver.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch as tr

import integrators as i
import update as u
import phi4 as s
from phi4_sokal_mg import IdentityMap, LocalAdditiveCoupling2x2
from multilevel_mg_phi4 import multilevel_vcycle_hmc_update


def default_config() -> Dict[str, Any]:
    return {
        "lattice": {"Lx": 32, "Ly": 32, "min_size": 4, "levels": None},
        "physics": {"lam": 0.5, "mass": -0.205},
        "run": {"Nwarm": 1000, "Nmeas": 1000, "Nskip": 10, "batch_size": 32, "seed": None},
        "fine_hmc": {"Nmd": 7, "tau": 1.0, "integrator": "minnorm2", "n_fine": 1},
        "coarse_hmc": {"Nmd": 5, "tau": 1.0, "integrator": "minnorm2", "m_coarse": 1},
        "mg": {"block": 2},
        "map": {
            "type": "identity",
            "steps": 1,
            "hidden": 32,
            "use_shift": True,
            "checkpoint": None,
            "checkpoints": None,
        },
    }


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


def summarize_observable(data: np.ndarray) -> Dict[str, float]:
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


def compute_levels(Lx: int, Ly: int, min_size: int, levels: int | None) -> int:
    min_dim = min(Lx, Ly)
    max_levels = int(math.floor(math.log2(min_dim / min_size))) if min_dim >= min_size else 0
    if levels is None:
        return max_levels
    if levels > max_levels:
        raise ValueError(f"Requested levels={levels} exceeds max_levels={max_levels} for min_size={min_size}")
    return levels


def build_vmaps(cfg: Dict[str, Any], levels: int) -> List[tr.nn.Module]:
    vmaps: List[tr.nn.Module] = []
    map_cfg = cfg["map"]
    ckpt_list = map_cfg.get("checkpoints")
    if ckpt_list is not None and len(ckpt_list) != levels:
        raise ValueError("map.checkpoints length must match levels")
    for _ in range(levels):
        if map_cfg["type"] == "identity":
            vmaps.append(IdentityMap())
        else:
            vmap = LocalAdditiveCoupling2x2(
                n_steps=int(map_cfg["steps"]),
                hidden=int(map_cfg["hidden"]),
                use_shift=bool(map_cfg["use_shift"]),
            )
            ckpt_path = None
            if ckpt_list is not None:
                ckpt_path = ckpt_list[len(vmaps)]
            elif map_cfg.get("checkpoint"):
                ckpt_path = map_cfg["checkpoint"]
            if ckpt_path:
                ckpt = tr.load(ckpt_path, map_location="cpu")
                state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                vmap.load_state_dict(state)
            for p in vmap.parameters():
                p.requires_grad_(False)
            vmaps.append(vmap)
    return vmaps


def main():
    parser = argparse.ArgumentParser(description="Multilevel MG-HMC for phi^4 (JSON config)")
    parser.add_argument("--config", type=str, help="Config JSON path")
    parser.add_argument("--write-default-config", type=str, help="Write default JSON config and exit")
    parser.add_argument(
        "--mode",
        choices=["mg", "baseline", "compare"],
        default="mg",
        help="Run MG-HMC, baseline HMC, or compare both",
    )
    args = parser.parse_args()

    if args.write_default_config:
        with open(args.write_default_config, "w") as f:
            json.dump(default_config(), f, indent=2)
        print("Wrote default config to", args.write_default_config)
        return

    if not args.config:
        raise SystemExit("Provide --config or --write-default-config")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    Lx = int(cfg["lattice"]["Lx"])
    Ly = int(cfg["lattice"]["Ly"])
    min_size = int(cfg["lattice"].get("min_size", 4))
    levels = compute_levels(Lx, Ly, min_size, cfg["lattice"].get("levels"))
    if levels <= 0:
        raise SystemExit("Computed zero levels; increase lattice size or lower min_size.")

    seed = cfg["run"].get("seed", None)
    if seed is not None:
        np.random.seed(seed)
        tr.manual_seed(seed)

    device = tr.device("cpu")
    lat = [Lx, Ly]
    Vol = np.prod(lat)

        fine = s.phi4(lat, cfg["physics"]["lam"], cfg["physics"]["mass"],
                      batch_size=cfg["run"]["batch_size"], device=device)
    phi = fine.hotStart()

    fine_hmc_cfg = cfg["fine_hmc"]
    fine_Nmd = fine_hmc_cfg["Nmd"][0] if isinstance(fine_hmc_cfg["Nmd"], list) else fine_hmc_cfg["Nmd"]
    fine_tau = fine_hmc_cfg["tau"][0] if isinstance(fine_hmc_cfg["tau"], list) else fine_hmc_cfg["tau"]
    if fine_hmc_cfg["integrator"] == "minnorm2":
        fine_I = i.minnorm2(fine.force, fine.evolveQ, fine_Nmd, fine_tau)
    else:
        fine_I = i.leapfrog(fine.force, fine.evolveQ, fine_Nmd, fine_tau)
    fine_hmc = u.hmc(T=fine, I=fine_I, verbose=False)

    vmaps = build_vmaps(cfg, levels)
    block = int(cfg["mg"]["block"])

    print(f"Using levels={levels}, coarsest size ~ {Lx // (2**levels)}x{Ly // (2**levels)}")

    def run_chain(run_mode: str):
        phi_local = phi.clone()
        n_fine = fine_hmc_cfg["n_fine"]
        m_coarse = cfg["coarse_hmc"]["m_coarse"] if run_mode == "mg" else 0
        fine_hmc.reset_Acceptance()
        acc_down_sum = [0.0] * levels
        acc_up_sum = [0.0] * levels
        trials_down_sum = [0] * levels
        trials_up_sum = [0] * levels

        print(f"Warming up ({run_mode})...")
        for _ in range(cfg["run"]["Nwarm"]):
            phi_local = fine_hmc.evolve(phi_local, n_fine)
            if m_coarse > 0:
                phi_local, acc_d, tr_d, acc_u, tr_u = multilevel_vcycle_hmc_update(
                    phi_local,
                    fine,
                    vmaps,
                    block=block,
                    Nmd=cfg["coarse_hmc"]["Nmd"],
                    tau=cfg["coarse_hmc"]["tau"],
                    Nhmc=m_coarse,
                    integrator=cfg["coarse_hmc"]["integrator"],
                    verbose=False,
                    return_acceptance=True,
                )
                for l in range(levels):
                    acc_down_sum[l] += acc_d[l] * tr_d[l]
                    trials_down_sum[l] += tr_d[l]
                    acc_up_sum[l] += acc_u[l] * tr_u[l]
                    trials_up_sum[l] += tr_u[l]

        fine_acc = fine_hmc.calc_Acceptance()
        print(f"Acceptance after warmup ({run_mode}): fine={fine_acc}")
        for l in range(levels):
            level_num = l + 1
            size = (Lx // (2 ** level_num), Ly // (2 ** level_num))
            ad = (acc_down_sum[l] / trials_down_sum[l]) if trials_down_sum[l] > 0 else float("nan")
            print(f"  level {level_num} ({size[0]}x{size[1]}) down={ad}")
        for l in range(1, levels):
            level_num = l
            size = (Lx // (2 ** level_num), Ly // (2 ** level_num))
            au = (acc_up_sum[l] / trials_up_sum[l]) if trials_up_sum[l] > 0 else float("nan")
            print(f"  level {level_num} ({size[0]}x{size[1]}) up={au}")

        fine_hmc.reset_Acceptance()
        acc_down_sum = [0.0] * levels
        acc_up_sum = [0.0] * levels
        trials_down_sum = [0] * levels
        trials_up_sum = [0] * levels

        phase = tr.tensor(np.exp(1j * np.indices(tuple(lat))[0] * 2 * np.pi / lat[0]))
        lE = []
        lav_phi = []
        lchi_m = []
        lC2p = []
        lxi = []

        print(f"Measuring ({run_mode})...")
        for k in range(cfg["run"]["Nmeas"]):
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

            for _ in range(cfg["run"]["Nskip"]):
                phi_local = fine_hmc.evolve(phi_local, n_fine)
                if m_coarse > 0:
                    phi_local, acc_d, tr_d, acc_u, tr_u = multilevel_vcycle_hmc_update(
                        phi_local,
                        fine,
                        vmaps,
                        block=block,
                        Nmd=cfg["coarse_hmc"]["Nmd"],
                        tau=cfg["coarse_hmc"]["tau"],
                        Nhmc=m_coarse,
                        integrator=cfg["coarse_hmc"]["integrator"],
                        verbose=False,
                        return_acceptance=True,
                    )
                    for l in range(levels):
                        acc_down_sum[l] += acc_d[l] * tr_d[l]
                        trials_down_sum[l] += tr_d[l]
                        acc_up_sum[l] += acc_u[l] * tr_u[l]
                        trials_up_sum[l] += tr_u[l]

        fine_acc = fine_hmc.calc_Acceptance()
        print(f"Acceptance after measurements ({run_mode}): fine={fine_acc}")
        for l in range(levels):
            level_num = l + 1
            size = (Lx // (2 ** level_num), Ly // (2 ** level_num))
            ad = (acc_down_sum[l] / trials_down_sum[l]) if trials_down_sum[l] > 0 else float("nan")
            print(f"  level {level_num} ({size[0]}x{size[1]}) down={ad}")
        for l in range(1, levels):
            level_num = l
            size = (Lx // (2 ** level_num), Ly // (2 ** level_num))
            au = (acc_up_sum[l] / trials_up_sum[l]) if trials_up_sum[l] > 0 else float("nan")
            print(f"  level {level_num} ({size[0]}x{size[1]}) up={au}")

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
            stats = summarize_observable(arr)
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

    print("Done.")


if __name__ == "__main__":
    main()
