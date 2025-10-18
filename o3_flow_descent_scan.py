#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Integrate the O(3) flow with  \dot s = - grad S_flow  from t=0 to t=1.

Uses flow actions and integrators from an external O3flow.py.

For each (L, beta) on user-specified grids, draws a batch of uniformly
random spins, integrates with a chosen integrator (using *_with_logdet),
computes
    ΔS = S_target(s_T) - log|det J|
and reports:
  - std(ΔS) with jackknife error bars,
  - ESS from reweighting: w_i = exp(-ΔS_i), ESS = (Σ w)^2 / Σ w^2
  - (optional) an "acceptance" ESS curve based on p_i = min(1, exp(-ΔS_i)),
    plotted if --plot-accept-ess is given.

Outputs per-run directory:
  results/<FLOW>__<INTEGRATOR>__<STEPLABEL>/
    - summary.csv
    - std_vs_L_betaXX.pdf, ess_vs_L_betaXX.pdf
    - std_vs_beta_LYY.pdf, ess_vs_beta_LYY.pdf
    - replot.py  (to regenerate the plots)

Notes:
- We present a DESCENT flow by wrapping your flow object:
    .grad -> negative of base.grad (or base.mgrad if available)
    .mlapl -> negative of base.mlapl
  This aligns with logdet accumulation in ascent-coded integrators so that
  the divergence for descent is integrated correctly.
- We set torch.set_default_dtype BEFORE importing O3flow.py to respect
  any module-level dtype bindings.
"""

import argparse
import csv
import math
import os
import importlib.util
from typing import Tuple

import numpy as np
import torch as tr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------ dynamic import ------------------

def import_by_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# ------------------ helpers ------------------

def random_o3_spins(B, Lx, Ly, device=None, dtype=tr.float64):
    z = tr.randn(B, 3, Lx, Ly, device=device, dtype=dtype)
    n = tr.clamp(tr.norm(z, dim=1, keepdim=True), min=1e-12)
    return z / n

def ensure_tensor1d(x, B: int, device, dtype) -> tr.Tensor:
    """
    Ensure x is shape [B] (per-sample scalar). If x is already [B], return.
    If x is [B,1,1,1] or [B,3,L,L], reduce by summing.
    """
    if isinstance(x, tr.Tensor):
        if x.ndim == 1 and x.numel() == B:
            return x.to(device=device, dtype=dtype)
        if x.shape[0] == B:
            return x.reshape(B, -1).sum(dim=1).to(device=device, dtype=dtype)
    return tr.as_tensor(x, dtype=dtype, device=device).reshape(-1)

def std_jackknife(x: np.ndarray) -> Tuple[float, float]:
    """
    Jackknife estimate of std(x) with delete-1 blocks.
    Returns (std_hat, jackknife_SE).
    """
    N = x.shape[0]
    mean = x.mean()
    s2 = ((x - mean)**2).sum() / (N - 1) if N > 1 else 0.0
    std_hat = math.sqrt(max(s2, 0.0))
    if N <= 2:
        return std_hat, float("nan")
    thetas = np.empty(N, dtype=np.float64)
    for i in range(N):
        xi = np.delete(x, i)
        mi = xi.mean()
        s2i = ((xi - mi)**2).sum() / (N - 2) if N > 2 else 0.0
        thetas[i] = math.sqrt(max(s2i, 0.0))
    theta_bar = thetas.mean()
    var_jk = (N - 1) * ((thetas - theta_bar)**2).mean()
    se = math.sqrt(max(var_jk, 0.0))
    return std_hat, se

def ess_from_logw(logw: np.ndarray) -> float:
    """
    ESS for importance weights (Kish): (Σ w)^2 / Σ w^2, with log-weight input.
    """
    m = logw.max()
    w = np.exp(logw - m)
    sw = w.sum()
    sw2 = (w**2).sum()
    return float((sw*sw) / max(sw2, 1e-300))

def ess_jackknife_from_deltas(deltas: np.ndarray) -> Tuple[float, float]:
    """
    Reweighting ESS from ΔS via w = exp(-ΔS), with jackknife SE.
    """
    N = deltas.shape[0]
    logw = -deltas
    ess_hat = ess_from_logw(logw)
    if N <= 2:
        return ess_hat, float("nan")
    thetas = np.empty(N, dtype=np.float64)
    for i in range(N):
        li = np.delete(logw, i)
        thetas[i] = ess_from_logw(li)
    theta_bar = thetas.mean()
    var_jk = (N - 1) * ((thetas - theta_bar)**2).mean()
    se = math.sqrt(max(var_jk, 0.0))
    return ess_hat, se

def ess_accept_from_deltas(deltas: np.ndarray) -> float:
    """
    "Acceptance" ESS proxy using acceptance probabilities p = min(1, e^{-ΔS}):
    treat p as soft weights and compute Kish ESS: (Σ p)^2 / Σ p^2.
    (Provides an upper bound on expected accepted independent count.)
    """
    p = np.minimum(1.0, np.exp(-deltas))
    sp = p.sum()
    sp2 = (p**2).sum()
    return float((sp*sp) / max(sp2, 1e-300))

def ess_accept_jackknife(deltas: np.ndarray) -> Tuple[float, float]:
    N = deltas.shape[0]
    val = ess_accept_from_deltas(deltas)
    if N <= 2:
        return val, float("nan")
    thetas = np.empty(N, dtype=np.float64)
    for i in range(N):
        d = np.delete(deltas, i)
        thetas[i] = ess_accept_from_deltas(d)
    theta_bar = thetas.mean()
    var_jk = (N - 1) * ((thetas - theta_bar)**2).mean()
    se = math.sqrt(max(var_jk, 0.0))
    return val, se

def make_outdir(base_dir: str, flow_name: str, integrator: str, step_label: str) -> str:
    d = os.path.join(base_dir, f"{flow_name}__{integrator}__{step_label}")
    os.makedirs(d, exist_ok=True)
    return d

# -------------- descent wrapper --------------

class DescentFlowWrapper:
    """
    Present a flow object as DESCENT:
      .grad(s,t)  -> - base.grad(s,t)  (or base.mgrad if available)
      .mlapl(s,t) -> - base.mlapl(s,t) (so ascent-coded logdet accumulators yield +div for descent)
      __call__(s,t) forwards to base for S_flow(s,t) if needed.
    """
    def __init__(self, base):
        self.base = base
        for k, v in getattr(base, "__dict__", {}).items():
            setattr(self, k, v)

    def __call__(self, s, t=None):
        try:
            return self.base(s, t)
        except TypeError:
            return self.base(s)

    def grad(self, s, t=None):
        if hasattr(self.base, "mgrad"):
            try:
                return  - self.base.mgrad(s, t)
            except TypeError:
                return - self.base.mgrad(s)
        try:
            return  self.base.grad(s, t)
        except TypeError:
            return  self.base.grad(s)

    def mlapl(self, s, t=None):
        try:
            val = self.base.mlapl(s, t)
        except TypeError:
            val = self.base.mlapl(s)
        return  val

# -------------- plotting helpers --------------

def errbar(ax, x, y, yerr, xlabel, ylabel, title=None, logy=False):
    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if logy: ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.4)

# -------------- main --------------

def main():
    parser = argparse.ArgumentParser(description="Integrate descent O(3) flow and compute ΔS statistics.")
    parser.add_argument("--o3flow", type=str, default="O3flow.py", help="Path to O3flow.py")
    parser.add_argument("--flow", type=str, choices=["SflowO0","SflowO1","SflowO2"], default="SflowO2")
    parser.add_argument("--integrator", type=str, default="rkmk4",
                        help="Suffix name for integrator in O3flow.py (e.g., rkmk2, rkmk3, rkmk4, rkmkdp5, lgim, rkmk_gl4).")
    parser.add_argument("--nsteps", type=int, default=200, help="Number of steps for fixed-step integrators")
    parser.add_argument("--rtol", type=float, default=1e-5, help="RTOL for adaptive DP5 (if used)")
    parser.add_argument("--atol", type=float, default=1e-7, help="ATOL for adaptive DP5 (if used)")
    parser.add_argument("--t1", type=float, default=1.0, help="Final flow time")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--betamin", type=float, default=0.1)
    parser.add_argument("--betamax", type=float, default=1.2)
    parser.add_argument("--betastep", type=float, default=0.1)
    parser.add_argument("--Lsmin", type=int, default=4)
    parser.add_argument("--Lsmax", type=int, default=48)
    parser.add_argument("--Lsstep", type=int, default=4)
    parser.add_argument("--outdir", type=str, default="results", help="Base output directory")
    parser.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto")
    parser.add_argument("--dtype", choices=["float32","float64"], default="float64")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--plot-accept-ess", action="store_true",
                        help="Overlay acceptance-based ESS curve using p=min(1,exp(-ΔS)).")
    args = parser.parse_args()

    # Device/dtype and import
    if args.device == "auto":
        if tr.cuda.is_available():
            device = "cuda"
        elif hasattr(tr.backends, "mps") and tr.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    desired_dtype = tr.float32 if args.dtype == "float32" else tr.float64
    if device == "mps" and desired_dtype == tr.float64:
        print("[note] MPS has no float64; switching to float32.")
        desired_dtype = tr.float32
    tr.set_default_dtype(desired_dtype)

    mod = import_by_path("O3flow_module", args.o3flow)

    # Seed
    tr.manual_seed(args.seed)
    if device == "cuda":
        tr.cuda.manual_seed_all(args.seed)

    # L and beta grids
    if args.Lsstep <= 0:
        raise ValueError("--Lsstep must be positive")
    L_list = list(range(args.Lsmin, args.Lsmax + 1, args.Lsstep))
    n_steps_beta = int(round((args.betamax - args.betamin) / args.betastep)) + 1
    betas = [round(args.betamin + i*args.betastep, 10) for i in range(n_steps_beta)]

    # Flow and target action constructors
    flow_ctor = getattr(mod, args.flow)
    target_ctor = getattr(mod, "Starget")

    def build_flow_and_target(beta: float, L: int):
        try:
            flow_base = flow_ctor(beta=beta)
        except TypeError:
            flow_base = flow_ctor(beta)
        flow = DescentFlowWrapper(flow_base)
        try:
            St = target_ctor(beta=beta)
        except TypeError:
            St = target_ctor(beta)
        return flow, St

    # Pick integrator function with logdet
    integ_name = args.integrator
    name_map = {
        "rkmk2": "integrate_rkmk2_with_logdet",
        "rkmk3": "integrate_rkmk3_with_logdet",
        "rkmk4": "integrate_rkmk4_with_logdet",
        "rkmkdp5": "integrate_rkmk_dp5_with_logdet",
        "lgim": "integrate_lgim_with_logdet",
        "rkmk_gl4": "integrate_rkmk_gl4_with_logdet",
        "gl4": "integrate_rkmk_gl4_with_logdet",
    }
    integ_func_name = name_map.get(integ_name, f"integrate_{integ_name}_with_logdet")
    if not hasattr(mod, integ_func_name):
        raise AttributeError(f"Integrator '{integ_func_name}' not found in {args.o3flow}")
    stepper = getattr(mod, integ_func_name)

    # Step label for directory
    if integ_name == "rkmkdp5":
        step_label = f"{integ_name}_rtol{args.rtol:g}_atol{args.atol:g}_t1{args.t1:g}"
    else:
        h = args.t1 / float(args.nsteps)
        step_label = f"{integ_name}_N{args.nsteps}_h{h:.4g}_t1{args.t1:g}"

    out_dir = make_outdir(args.outdir, args.flow, integ_name, step_label)

    # Optional: ensure lattice operator matches dtype/device
    if hasattr(mod, "reset_L"):
        try:
            mod.L = mod.reset_L().to(dtype=desired_dtype, device=device)
        except Exception:
            pass

    # CSV setup
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["flow","integrator","L","beta","batch","dtype","device","nsteps","t1","rtol","atol",
                         "std_dS","std_dS_jk_se",
                         "ESS_rw","ESS_rw_jk_se",
                         "ESS_acc","ESS_acc_jk_se"])

    # Main sweep
    for L in L_list:
        s0 = random_o3_spins(args.batch, L, L, device=device, dtype=desired_dtype)
        for beta in betas:
            flow, S_tgt = build_flow_and_target(beta, L)

            # integrate
            with tr.no_grad():
                if integ_name == "rkmkdp5":
                    sT, logdet, info = stepper(flow, s0, t0=0.0, t1=args.t1, rtol=args.rtol, atol=args.atol)
                elif integ_name in ("lgim","rkmk_gl4","gl4"):
                    sT, logdet = stepper(flow, s0, t0=0.0, t1=args.t1, n_steps=args.nsteps,
                                         tol=1e-12, max_iters=100, damping=1.0)
                else:
                    sT, logdet = stepper(flow, s0, t0=0.0, t1=args.t1, n_steps=args.nsteps)

            # evaluate target action per-sample
            with tr.enable_grad():
                try:
                    St_vals = S_tgt(sT)
                except TypeError:
                    St_vals = S_tgt(sT, args.t1)

            B = s0.size(0)
            St_B = ensure_tensor1d(St_vals, B, device=s0.device, dtype=desired_dtype)
            logdet_B = ensure_tensor1d(logdet, B, device=s0.device, dtype=desired_dtype)

            deltaS = (St_B - logdet_B).detach().cpu().numpy()

            # stats
            std_hat, std_se = std_jackknife(deltaS)
            ess_rw, ess_rw_se = ess_jackknife_from_deltas(deltaS)
            ess_acc, ess_acc_se = ess_accept_jackknife(deltaS)

            # append CSV row
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([args.flow, integ_name, L, beta, args.batch, args.dtype, device,
                                 (args.nsteps if integ_name!='rkmkdp5' else ''), args.t1,
                                 (args.rtol if integ_name=='rkmkdp5' else ''),
                                 (args.atol if integ_name=='rkmkdp5' else ''),
                                 f"{std_hat:.10e}", f"{std_se:.10e}",
                                 f"{ess_rw:.10e}", f"{ess_rw_se:.10e}",
                                 f"{ess_acc:.10e}", f"{ess_acc_se:.10e}"])

    # --- plotting ---
    import pandas as pd
    df = pd.read_csv(csv_path)

    # std(ΔS) vs L at fixed beta
    for beta in betas:
        sub = df[df["beta"] == beta].sort_values("L")
        if sub.empty: continue
        fig, ax = plt.subplots(figsize=(6,4))
        errbar(ax, sub["L"].values, sub["std_dS"].astype(float).values, sub["std_dS_jk_se"].astype(float).values,
               xlabel="L", ylabel="std(ΔS)", title=f"std(ΔS) vs L  (β={beta:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"std_vs_L_beta{beta:.2f}.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6,4))
        y = sub["ESS_rw"].astype(float).values
        ye = sub["ESS_rw_jk_se"].astype(float).values
        errbar(ax, sub["L"].values, y, ye, xlabel="L", ylabel="ESS", title=f"ESS (reweight) vs L (β={beta:.2f})", logy=True)
        if args.plot_accept_ess:
            y2 = sub["ESS_acc"].astype(float).values
            ye2 = sub["ESS_acc_jk_se"].astype(float).values
            ax.errorbar(sub["L"].values, y2, yerr=ye2, fmt='s--', capsize=3, label="accept-ESS")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"ess_vs_L_beta{beta:.2f}.pdf"))
        plt.close(fig)

    # std(ΔS) vs beta at fixed L
    for L in L_list:
        sub = df[df["L"] == L].sort_values("beta")
        if sub.empty: continue
        fig, ax = plt.subplots(figsize=(6,4))
        errbar(ax, sub["beta"].values, sub["std_dS"].astype(float).values, sub["std_dS_jk_se"].astype(float).values,
               xlabel="β", ylabel="std(ΔS)", title=f"std(ΔS) vs β (L={L})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"std_vs_beta_L{L}.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6,4))
        y = sub["ESS_rw"].astype(float).values
        ye = sub["ESS_rw_jk_se"].astype(float).values
        errbar(ax, sub["beta"].values, y, ye, xlabel="β", ylabel="ESS", title=f"ESS (reweight) vs β (L={L})", logy=True)
        if args.plot_accept_ess:
            y2 = sub["ESS_acc"].astype(float).values
            ye2 = sub["ESS_acc_jk_se"].astype(float).values
            ax.errorbar(sub["beta"].values, y2, yerr=ye2, fmt='s--', capsize=3, label="accept-ESS")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"ess_vs_beta_L{L}.pdf"))
        plt.close(fig)

    # --- dump a tiny replot script ---
    replot_code = """#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def errbar(ax, x, y, yerr, xlabel, ylabel, title=None, logy=False):
    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if logy: ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.4)

def main():
    df = pd.read_csv('summary.csv')
    betas = sorted(df['beta'].unique())
    Ls = sorted(df['L'].unique())
    for beta in betas:
        sub = df[df['beta']==beta].sort_values('L')
        if sub.empty: continue
        fig, ax = plt.subplots(figsize=(6,4))
        errbar(ax, sub['L'].values, sub['std_dS'].values, sub['std_dS_jk_se'].values, 'L', 'std(ΔS)', f'std(ΔS) vs L (β={beta:.2f})')
        fig.tight_layout(); fig.savefig(f'std_vs_L_beta{beta:.2f}.pdf'); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6,4))
        errbar(ax, sub['L'].values, sub['ESS_rw'].values, sub['ESS_rw_jk_se'].values, 'L', 'ESS', f'ESS (reweight) vs L (β={beta:.2f})', logy=True)
        if 'ESS_acc' in sub.columns:
            ax.errorbar(sub['L'].values, sub['ESS_acc'].values, yerr=sub['ESS_acc_jk_se'].values, fmt='s--', capsize=3, label='accept-ESS')
            ax.legend()
        fig.tight_layout(); fig.savefig(f'ess_vs_L_beta{beta:.2f}.pdf'); plt.close(fig)
    for L in Ls:
        sub = df[df['L']==L].sort_values('beta')
        if sub.empty: continue
        fig, ax = plt.subplots(figsize=(6,4))
        errbar(ax, sub['beta'].values, sub['std_dS'].values, sub['std_dS_jk_se'].values, 'β', 'std(ΔS)', f'std(ΔS) vs β (L={L})')
        fig.tight_layout(); fig.savefig(f'std_vs_beta_L{L}.pdf'); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6,4))
        errbar(ax, sub['beta'].values, sub['ESS_rw'].values, sub['ESS_rw_jk_se'].values, 'β', 'ESS', f'ESS (reweight) vs β (L={L})', logy=True)
        if 'ESS_acc' in sub.columns:
            ax.errorbar(sub['beta'].values, sub['ESS_acc'].values, yerr=sub['ESS_acc_jk_se'].values, fmt='s--', capsize=3, label='accept-ESS')
            ax.legend()
        fig.tight_layout(); fig.savefig(f'ess_vs_beta_L{L}.pdf'); plt.close(fig)

if __name__ == '__main__':
    main()
"""
    with open(os.path.join(out_dir, "replot.py"), "w") as g:
        g.write(replot_code)

    print(f"[done] Results saved under: {out_dir}")
    print(f" - data: {csv_path}")
    print(f" - plots: std_vs_L_beta*.pdf, ess_vs_L_beta*.pdf, std_vs_beta_L*.pdf, ess_vs_beta_L*.pdf")
    print(f" - replot script: {os.path.join(out_dir, 'replot.py')}")

if __name__ == "__main__":
    main()
