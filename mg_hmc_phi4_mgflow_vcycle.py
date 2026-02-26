#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V-cycle MGMC using trained MGflow (phi4_mg.py) with non-trivial Jacobian.
HMC operates on coarse latent variables at each level with fixed residuals.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Dict, List

import numpy as np
import torch as tr

import integrators as i
import update as u
import phi4 as s
import phi4_mg as pmg


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


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0):
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


def default_config() -> Dict[str, Any]:
    return {
        "lattice": {"Lx": 32, "Ly": 32},
        "physics": {"lam": 0.5, "mass": -0.205},
        "run": {"Nwarm": 200, "Nmeas": 200, "Nskip": 10, "batch_size": 16, "seed": None},
        "fine_hmc": {"integrator": "minnorm2", "Nmd": 5, "tau": 1.0, "Nhmc": 1},
        "mgflow": {
            "checkpoint": "mg-model.dict",
            "rg_type": "average",
            "bijector": "FlowBijector",
            "Nlayers": 3,
            "width": 256,
            "Nconvs": 1,
            "fbj": False,
            "sbj": False,
            "stacked_index": None
        },
        "vcycle": {
            "integrator": "minnorm2",
            "Nmd": [5, 3, 1, 1, 1],
            "tau": [1.0, 0.7, 0.5, 0.3, 0.2],
            "Nhmc": 1
        }
    }


def build_mgflow(cfg: Dict[str, Any], device):
    Lx = cfg["lattice"]["Lx"]
    Ly = cfg["lattice"]["Ly"]
    mgcfg = cfg["mgflow"]
    state = tr.load(mgcfg["checkpoint"], map_location="cpu")
    ckpt_arch = state.get("arch") if isinstance(state, dict) else None

    if ckpt_arch:
        if (ckpt_arch.get("Lx") != Lx) or (ckpt_arch.get("Ly") != Ly):
            print(f"Warning: checkpoint Lx/Ly {ckpt_arch.get('Lx')}/{ckpt_arch.get('Ly')} != config {Lx}/{Ly}")
        # auto-configure from checkpoint
        mgcfg = {
            "rg_type": ckpt_arch.get("rg_type", mgcfg.get("rg_type", "average")),
            "bijector": ckpt_arch.get("bijector", mgcfg.get("bijector", "FlowBijector")),
            "Nlayers": ckpt_arch.get("Nlayers", mgcfg.get("Nlayers", 3)),
            "width": ckpt_arch.get("width", mgcfg.get("width", 256)),
            "Nconvs": ckpt_arch.get("Nconvs", mgcfg.get("Nconvs", 1)),
            "fbj": ckpt_arch.get("fbj", mgcfg.get("fbj", False)),
            "sbj": ckpt_arch.get("sbj", mgcfg.get("sbj", False)),
        }
        print("Using MGflow arch from checkpoint:", mgcfg)

    rg = pmg.RGlayer(mgcfg["rg_type"])
    prior = tr.distributions.Normal(0.0, 1.0)
    bij_name = mgcfg["bijector"]
    Nlayers = mgcfg.get("Nlayers", 3)
    width = mgcfg.get("width", 256)
    def base_bij():
        if bij_name == "FlowBijector":
            return pmg.FlowBijector(Nlayers=Nlayers, width=width)
        if bij_name == "FlowBijectorParity":
            return pmg.FlowBijectorParity(Nlayers=Nlayers, width=width)
        return pmg.FlowBijector_3layers(Nlayers=Nlayers, width=width)

    bij = base_bij
    if mgcfg.get("sbj", False):
        bij = lambda: pmg.FlowBijectorParity(Nlayers=Nlayers, width=width)
    if mgcfg.get("fbj", False):
        bij_list = [bij() for _ in range(2 * mgcfg["Nconvs"])]
        bij = pmg.BijectorFactory(bij_list).bij
    flow = pmg.MGflow([Lx, Ly], bij, rg, prior, mgcfg["Nconvs"])
    if isinstance(state, dict) and "model_state" in state:
        # New single-model checkpoint format
        missing, unexpected = flow.load_state_dict(state["model_state"], strict=False)
        if missing or unexpected:
            print("MGflow load_state_dict (single) missing:", missing)
            print("MGflow load_state_dict (single) unexpected:", unexpected)
    else:
        # Legacy raw state_dict
        missing, unexpected = flow.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("MGflow load_state_dict missing:", missing)
            print("MGflow load_state_dict unexpected:", unexpected)
    flow.to(device)
    return flow, rg


def build_z_pyramid(z: tr.Tensor, rg: pmg.RGlayer, levels: int):
    z_levels = [z]
    residuals = []
    cur = z
    for _ in range(levels):
        c, r = rg.coarsen(cur)
        residuals.append(r)
        z_levels.append(c)
        cur = c
    return z_levels, residuals


def refine_from_level(z_level: tr.Tensor, residuals: List[tr.Tensor], rg: pmg.RGlayer) -> tr.Tensor:
    cur = z_level
    for l in reversed(range(len(residuals))):
        cur = rg.refine(cur, residuals[l])
    return cur


class LatentLevelTheory:
    def __init__(
        self,
        fine_theory: s.phi4,
        flow: pmg.MGflow,
        rg: pmg.RGlayer,
        residuals: List[tr.Tensor],
        z_level: tr.Tensor,
    ):
        self.fine = fine_theory
        self.flow = flow
        self.rg = rg
        self.residuals = [r.detach() for r in residuals]
        self.Bs = z_level.shape[0]
        self.V = z_level.shape[1:]
        self.device = fine_theory.device
        self.dtype = fine_theory.dtype

    def action(self, z_level: tr.Tensor) -> tr.Tensor:
        z_full = refine_from_level(z_level, self.residuals, self.rg)
        x = self.flow.forward(z_full)
        _, logdet = self.flow.backward(x)
        return self.fine.action(x) + logdet

    def force(self, z_level: tr.Tensor) -> tr.Tensor:
        z_req = z_level.detach().requires_grad_(True)
        A = self.action(z_req)
        grad = tr.autograd.grad(A.sum(), z_req, create_graph=False)[0]
        return -grad

    def refreshP(self) -> tr.Tensor:
        return tr.normal(0.0, 1.0, [self.Bs, self.V[0], self.V[1]], dtype=self.dtype, device=self.device)

    def evolveQ(self, dt: float, P: tr.Tensor, Q: tr.Tensor) -> tr.Tensor:
        return Q + dt * P

    def kinetic(self, P: tr.Tensor) -> tr.Tensor:
        return tr.einsum("bxy,bxy->b", P, P) / 2.0


def vcycle_update(
    phi: tr.Tensor,
    fine: s.phi4,
    flow: pmg.MGflow,
    rg: pmg.RGlayer,
    vcfg: Dict[str, Any],
    debug: bool = False,
):
    z, _ = flow.backward(phi)
    levels = flow.depth - 1
    z_levels, residuals = build_z_pyramid(z, rg, levels)

    Nmd = vcfg["Nmd"]
    tau = vcfg["tau"]
    integrator = vcfg["integrator"]
    Nhmc = vcfg["Nhmc"]

    acc_down = [0.0] * (levels + 1)
    trials_down = [0] * (levels + 1)
    acc_up = [0.0] * (levels + 1)
    trials_up = [0] * (levels + 1)

    # down sweep: update levels 0..levels
    for l in range(0, levels + 1):
        if debug:
            print(f"[debug] HMC down: level {l} size={list(z_levels[l].shape[1:])}")
        theory = LatentLevelTheory(fine, flow, rg, residuals[:l], z_levels[l])
        Nmd_l = Nmd[l] if isinstance(Nmd, list) else Nmd
        tau_l = tau[l] if isinstance(tau, list) else tau
        if integrator == "minnorm2":
            I = i.minnorm2(theory.force, theory.evolveQ, Nmd_l, tau_l)
        else:
            I = i.leapfrog(theory.force, theory.evolveQ, Nmd_l, tau_l)
        hmc = u.hmc(T=theory, I=I, verbose=False)
        t0 = time.perf_counter()
        z_levels[l] = hmc.evolve(z_levels[l], Nhmc)
        t1 = time.perf_counter()
        acc_down[l] = hmc.calc_Acceptance()
        trials_down[l] = len(hmc.AcceptReject)
        if debug:
            print(
                f"[debug] done down level {l}: Nmd={Nmd_l} tau={tau_l} "
                f"acc={acc_down[l]:.4g} steps={trials_down[l]} time={t1 - t0:.4f}s"
            )

        # update residuals for next level (down)
        if l < levels:
            c, r = rg.coarsen(z_levels[l])
            z_levels[l + 1] = c
            residuals[l] = r

    # up sweep: update levels levels-1..0
    for l in reversed(range(0, levels)):
        # coarse correction: refine one level using fixed residuals[l]
        z_levels[l] = rg.refine(z_levels[l + 1], residuals[l])

        if debug:
            print(f"[debug] HMC up: level {l} size={list(z_levels[l].shape[1:])}")
        theory = LatentLevelTheory(fine, flow, rg, residuals[:l], z_levels[l])
        Nmd_l = Nmd[l] if isinstance(Nmd, list) else Nmd
        tau_l = tau[l] if isinstance(tau, list) else tau
        if integrator == "minnorm2":
            I = i.minnorm2(theory.force, theory.evolveQ, Nmd_l, tau_l)
        else:
            I = i.leapfrog(theory.force, theory.evolveQ, Nmd_l, tau_l)
        hmc = u.hmc(T=theory, I=I, verbose=False)
        t0 = time.perf_counter()
        z_levels[l] = hmc.evolve(z_levels[l], Nhmc)
        t1 = time.perf_counter()
        acc_up[l] = hmc.calc_Acceptance()
        trials_up[l] = len(hmc.AcceptReject)
        if debug:
            print(
                f"[debug] done up level {l}: Nmd={Nmd_l} tau={tau_l} "
                f"acc={acc_up[l]:.4g} steps={trials_up[l]} time={t1 - t0:.4f}s"
            )

        # residuals are kept fixed on the way up; do not refresh here

    z_new = z_levels[0]
    phi_new = flow.forward(z_new)
    return phi_new, acc_down, trials_down, acc_up, trials_up


def main():
    parser = argparse.ArgumentParser(description="MGMC with trained MGflow (V-cycle)")
    parser.add_argument("--config", type=str, help="Config JSON path")
    parser.add_argument("--write-default-config", type=str, help="Write default JSON config and exit")
    parser.add_argument("--debug", action="store_true", help="Verbose per-level HMC debug output")
    parser.add_argument(
        "--mode",
        choices=["fine", "direct", "vcycle", "compare_direct", "compare_vcycle", "compare_all"],
        default="vcycle",
        help="Run mode: fine (phi), direct (latent z), vcycle, or comparisons",
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

    seed = cfg["run"].get("seed", None)
    if seed is not None:
        np.random.seed(seed)
        tr.manual_seed(seed)

    device = tr.device("cpu")
    Lx = cfg["lattice"]["Lx"]
    Ly = cfg["lattice"]["Ly"]
    fine = s.phi4([Lx, Ly], cfg["physics"]["lam"], cfg["physics"]["mass"],
                  batch_size=cfg["run"]["batch_size"], device=device)

    flow, rg = build_mgflow(cfg, device)
    levels = flow.depth - 1
    level_sizes = [(Lx // (2 ** l), Ly // (2 ** l)) for l in range(levels + 1)]

    def run_mode(mode: str) -> Dict[str, Dict[str, float]]:
        phi = fine.hotStart()
        print(f"Warming up ({mode})...")
        if mode == "vcycle":
            acc_down_sum = [0.0] * (levels + 1)
            acc_up_sum = [0.0] * (levels + 1)
            trials_down_sum = [0] * (levels + 1)
            trials_up_sum = [0] * (levels + 1)
            for _ in range(cfg["run"]["Nwarm"]):
                phi, acc_d, tr_d, acc_u, tr_u = vcycle_update(phi, fine, flow, rg, cfg["vcycle"], debug=args.debug)
                for l in range(levels + 1):
                    acc_down_sum[l] += acc_d[l] * tr_d[l]
                    trials_down_sum[l] += tr_d[l]
                    acc_up_sum[l] += acc_u[l] * tr_u[l]
                    trials_up_sum[l] += tr_u[l]
            print("Acceptance after warmup (mgflow vcycle):")
            for l in range(levels + 1):
                acc_d = acc_down_sum[l] / max(trials_down_sum[l], 1)
                tag = f"level {l} ({level_sizes[l][0]}x{level_sizes[l][1]}) down={acc_d:.4g}"
                if l < levels:
                    acc_u = acc_up_sum[l] / max(trials_up_sum[l], 1)
                    tag += f" up={acc_u:.4g}"
                print(" ", tag)
        elif mode == "direct":
            z, _ = flow.backward(phi)
            theory = LatentLevelTheory(fine, flow, rg, residuals=[], z_level=z)
            direct_cfg = cfg.get("direct_hmc", {})
            Nmd = direct_cfg.get("Nmd", cfg["vcycle"]["Nmd"][-1] if isinstance(cfg["vcycle"]["Nmd"], list) else cfg["vcycle"]["Nmd"])
            tau = direct_cfg.get("tau", cfg["vcycle"]["tau"][-1] if isinstance(cfg["vcycle"]["tau"], list) else cfg["vcycle"]["tau"])
            integrator = direct_cfg.get("integrator", cfg["vcycle"]["integrator"])
            if integrator == "minnorm2":
                I = i.minnorm2(theory.force, theory.evolveQ, Nmd, tau)
            else:
                I = i.leapfrog(theory.force, theory.evolveQ, Nmd, tau)
            hmc = u.hmc(T=theory, I=I, verbose=False)
            for _ in range(cfg["run"]["Nwarm"]):
                z = hmc.evolve(z, direct_cfg.get("Nhmc", cfg["vcycle"]["Nhmc"]))
            with tr.no_grad():
                phi = flow.forward(z)
            print(f"Acceptance after warmup (mgflow direct): {hmc.calc_Acceptance():.4g}")
        else:
            fine_cfg = cfg.get("fine_hmc", {})
            Nmd = fine_cfg.get("Nmd", cfg["vcycle"]["Nmd"][0] if isinstance(cfg["vcycle"]["Nmd"], list) else cfg["vcycle"]["Nmd"])
            tau = fine_cfg.get("tau", cfg["vcycle"]["tau"][0] if isinstance(cfg["vcycle"]["tau"], list) else cfg["vcycle"]["tau"])
            integrator = fine_cfg.get("integrator", cfg["vcycle"]["integrator"])
            Nhmc = fine_cfg.get("Nhmc", cfg["vcycle"]["Nhmc"])
            if integrator == "minnorm2":
                I = i.minnorm2(fine.force, fine.evolveQ, Nmd, tau)
            else:
                I = i.leapfrog(fine.force, fine.evolveQ, Nmd, tau)
            hmc = u.hmc(T=fine, I=I, verbose=False)
            for _ in range(cfg["run"]["Nwarm"]):
                phi = hmc.evolve(phi, Nhmc)
            print(f"Acceptance after warmup (fine HMC): {hmc.calc_Acceptance():.4g}")

        phase = tr.tensor(np.exp(1j * np.indices((Lx, Ly))[0] * 2 * np.pi / Lx))
        Vol = Lx * Ly
        lE = []
        lav_phi = []
        lchi_m = []
        lC2p = []
        lxi = []

        print(f"Measuring ({mode})...")
        acc_down_sum = [0.0] * (levels + 1)
        acc_up_sum = [0.0] * (levels + 1)
        trials_down_sum = [0] * (levels + 1)
        trials_up_sum = [0] * (levels + 1)
        acc_direct = []
        if mode == "direct":
            # reuse z + theory + integrator across the chain
            if "z" not in locals():
                z, _ = flow.backward(phi)
            theory = LatentLevelTheory(fine, flow, rg, residuals=[], z_level=z)
            direct_cfg = cfg.get("direct_hmc", {})
            Nmd = direct_cfg.get("Nmd", cfg["vcycle"]["Nmd"][-1] if isinstance(cfg["vcycle"]["Nmd"], list) else cfg["vcycle"]["Nmd"])
            tau = direct_cfg.get("tau", cfg["vcycle"]["tau"][-1] if isinstance(cfg["vcycle"]["tau"], list) else cfg["vcycle"]["tau"])
            integrator = direct_cfg.get("integrator", cfg["vcycle"]["integrator"])
            if integrator == "minnorm2":
                I = i.minnorm2(theory.force, theory.evolveQ, Nmd, tau)
            else:
                I = i.leapfrog(theory.force, theory.evolveQ, Nmd, tau)
            hmc = u.hmc(T=theory, I=I, verbose=False)
        t_meas0 = time.perf_counter()
        t_update = 0.0
        t_measure = 0.0
        for k in range(cfg["run"]["Nmeas"]):
            t0 = time.perf_counter()
            if mode == "direct":
                with tr.no_grad():
                    phi = flow.forward(z)
            ttE = fine.action(phi) / Vol
            av_sigma = tr.mean(phi.view(fine.Bs, Vol), axis=1)
            chi_m = av_sigma * av_sigma * Vol
            p1_av_sig = tr.mean(phi.view(fine.Bs, Vol) * phase.view(1, Vol), axis=1)
            C2p = tr.real(tr.conj(p1_av_sig) * p1_av_sig) * Vol
            xi = correlation_length(Lx, chi_m, C2p)
            t1 = time.perf_counter()
            t_measure += t1 - t0

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
            t0 = time.perf_counter()
            if mode == "vcycle":
                for _ in range(cfg["run"]["Nskip"]):
                    phi, acc_d, tr_d, acc_u, tr_u = vcycle_update(phi, fine, flow, rg, cfg["vcycle"], debug=args.debug)
                    for l in range(levels + 1):
                        acc_down_sum[l] += acc_d[l] * tr_d[l]
                        trials_down_sum[l] += tr_d[l]
                        acc_up_sum[l] += acc_u[l] * tr_u[l]
                        trials_up_sum[l] += tr_u[l]
            elif mode == "direct":
                for _ in range(cfg["run"]["Nskip"]):
                    z = hmc.evolve(z, direct_cfg.get("Nhmc", cfg["vcycle"]["Nhmc"]))
                    acc_direct.append(hmc.calc_Acceptance())
            else:
                fine_cfg = cfg.get("fine_hmc", {})
                Nmd = fine_cfg.get("Nmd", cfg["vcycle"]["Nmd"][0] if isinstance(cfg["vcycle"]["Nmd"], list) else cfg["vcycle"]["Nmd"])
                tau = fine_cfg.get("tau", cfg["vcycle"]["tau"][0] if isinstance(cfg["vcycle"]["tau"], list) else cfg["vcycle"]["tau"])
                integrator = fine_cfg.get("integrator", cfg["vcycle"]["integrator"])
                Nhmc = fine_cfg.get("Nhmc", cfg["vcycle"]["Nhmc"])
                if integrator == "minnorm2":
                    I = i.minnorm2(fine.force, fine.evolveQ, Nmd, tau)
                else:
                    I = i.leapfrog(fine.force, fine.evolveQ, Nmd, tau)
                hmc = u.hmc(T=fine, I=I, verbose=False)
                for _ in range(cfg["run"]["Nskip"]):
                    phi = hmc.evolve(phi, Nhmc)
                    acc_direct.append(hmc.calc_Acceptance())
            t1 = time.perf_counter()
            t_update += t1 - t0
        t_meas1 = time.perf_counter()
        nmeas = cfg["run"]["Nmeas"]
        total_time = t_meas1 - t_meas0
        time_per_cfg = total_time / max(nmeas, 1)
        print(f"Timing: total measurement time {total_time:.3f}s, per config {time_per_cfg:.6f}s")
        print(f"Timing breakdown: measure {t_measure:.3f}s, update {t_update:.3f}s")
        if mode == "vcycle":
            print("Acceptance after measurements (mgflow vcycle):")
            for l in range(levels + 1):
                acc_d = acc_down_sum[l] / max(trials_down_sum[l], 1)
                tag = f"level {l} ({level_sizes[l][0]}x{level_sizes[l][1]}) down={acc_d:.4g}"
                if l < levels:
                    acc_u = acc_up_sum[l] / max(trials_up_sum[l], 1)
                    tag += f" up={acc_u:.4g}"
                print(" ", tag)
        elif mode == "direct":
            if acc_direct:
                print(f"Acceptance after measurements (mgflow direct): {float(np.mean(acc_direct)):.4g}")
        else:
            if acc_direct:
                print(f"Acceptance after measurements (fine HMC): {float(np.mean(acc_direct)):.4g}")

        E = np.stack(lE, axis=0)
        av_phi = np.stack(lav_phi, axis=0)
        chi_m = np.stack(lchi_m, axis=0)
        C2p = np.stack(lC2p, axis=0)
        xi = np.stack(lxi, axis=0)

        results = {"_timing": {"total": total_time, "per_config": time_per_cfg}}
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
        print("\nEfficiency report (time_per_config * 2*tau_int):")
        for name in ["E", "av_phi", "chi_m", "C2p", "xi"]:
            tau_int = results[name]["tau_int"]
            eff = time_per_cfg * 2.0 * tau_int if np.isfinite(tau_int) else float("nan")
            print(f"{name}: {eff}")
        av = results["av_phi"]
        if np.isfinite(av["tau_int"]):
            eff_av = time_per_cfg * 2.0 * av["tau_int"]
            print(f"\nSummary: av_phi tau_int={av['tau_int']}  time_per_config={time_per_cfg}  eff={eff_av}")
        return results

    if args.mode == "compare_direct":
        fine_stats = run_mode("fine")
        direct_stats = run_mode("direct")
        print("\nDirect - fine differences (mean ± combined err):")
        for name in ["E", "av_phi", "chi_m", "C2p", "xi"]:
            m0 = fine_stats[name]["mean"]
            e0 = fine_stats[name]["err"]
            m1 = direct_stats[name]["mean"]
            e1 = direct_stats[name]["err"]
            dm = m1 - m0
            de = math.sqrt(e0 * e0 + e1 * e1) if np.isfinite(e0) and np.isfinite(e1) else float("nan")
            print(f"{name}: {dm} ± {de}")
    elif args.mode == "compare_vcycle":
        fine_stats = run_mode("fine")
        vcycle_stats = run_mode("vcycle")
        print("\nV-cycle - fine differences (mean ± combined err):")
        for name in ["E", "av_phi", "chi_m", "C2p", "xi"]:
            m0 = fine_stats[name]["mean"]
            e0 = fine_stats[name]["err"]
            m1 = vcycle_stats[name]["mean"]
            e1 = vcycle_stats[name]["err"]
            dm = m1 - m0
            de = math.sqrt(e0 * e0 + e1 * e1) if np.isfinite(e0) and np.isfinite(e1) else float("nan")
            print(f"{name}: {dm} ± {de}")
    elif args.mode == "compare_all":
        fine_stats = run_mode("fine")
        direct_stats = run_mode("direct")
        vcycle_stats = run_mode("vcycle")
        print("\nDirect - fine differences (mean ± combined err):")
        for name in ["E", "av_phi", "chi_m", "C2p", "xi"]:
            m0 = fine_stats[name]["mean"]
            e0 = fine_stats[name]["err"]
            m1 = direct_stats[name]["mean"]
            e1 = direct_stats[name]["err"]
            dm = m1 - m0
            de = math.sqrt(e0 * e0 + e1 * e1) if np.isfinite(e0) and np.isfinite(e1) else float("nan")
            print(f"{name}: {dm} ± {de}")
        print("\nV-cycle - fine differences (mean ± combined err):")
        for name in ["E", "av_phi", "chi_m", "C2p", "xi"]:
            m0 = fine_stats[name]["mean"]
            e0 = fine_stats[name]["err"]
            m1 = vcycle_stats[name]["mean"]
            e1 = vcycle_stats[name]["err"]
            dm = m1 - m0
            de = math.sqrt(e0 * e0 + e1 * e1) if np.isfinite(e0) and np.isfinite(e1) else float("nan")
            print(f"{name}: {dm} ± {de}")
        print("\nV-cycle - direct differences (mean ± combined err):")
        for name in ["E", "av_phi", "chi_m", "C2p", "xi"]:
            m0 = direct_stats[name]["mean"]
            e0 = direct_stats[name]["err"]
            m1 = vcycle_stats[name]["mean"]
            e1 = vcycle_stats[name]["err"]
            dm = m1 - m0
            de = math.sqrt(e0 * e0 + e1 * e1) if np.isfinite(e0) and np.isfinite(e1) else float("nan")
            print(f"{name}: {dm} ± {de}")
    else:
        run_mode(args.mode)


if __name__ == "__main__":
    main()
