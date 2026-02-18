import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fit_cosh(taus, means, L, m_min=1e-3, m_max=2.0, n_grid=400):
    # simple grid search for m, then least-squares A
    taus = np.asarray(taus, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    x = taus - 0.5 * L
    best = (np.inf, 0.0, 0.0)
    for m in np.linspace(m_min, m_max, n_grid):
        c = np.cosh(m * x)
        denom = np.dot(c, c)
        if denom == 0:
            continue
        A = np.dot(means, c) / denom
        resid = means - A * c
        sse = np.dot(resid, resid)
        if sse < best[0]:
            best = (sse, A, m)
    return best[1], best[2]

def fit_multi_cosh(taus, means, L, n_cosh=1, m_min=1e-3, m_max=2.0, n_grid=200):
    # Greedy residual fit: fit one cosh at a time.
    taus = np.asarray(taus, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    x = taus - 0.5 * L
    residual = means.copy()
    params = []
    for _ in range(n_cosh):
        best = (np.inf, 0.0, 0.0)
        for m in np.linspace(m_min, m_max, n_grid):
            c = np.cosh(m * x)
            denom = np.dot(c, c)
            if denom == 0:
                continue
            A = np.dot(residual, c) / denom
            resid = residual - A * c
            sse = np.dot(resid, resid)
            if sse < best[0]:
                best = (sse, A, m)
        _, A_best, m_best = best
        params.append((A_best, m_best))
        residual = residual - A_best * np.cosh(m_best * x)
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="path to json db")
    parser.add_argument("--out_dir", default=None, help="directory to save plots")
    parser.add_argument("--tag", default="", help="tag for output filenames")
    parser.add_argument("--ncosh", type=int, default=1, help="number of cosh terms in fit")
    parser.add_argument("--tau_min", type=int, default=2, help="minimum tau used in fit")
    args = parser.parse_args()

    db_path = Path(args.db)
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    taus = np.asarray(data["taus"], dtype=np.int64)
    c2 = data["c2pt"]
    c2i = data["c2pt_improved"]
    cv = data["control_variate"]
    gain = data["gain"]
    cov = data["covariance"]
    L = data["meta"]["L"]

    out_dir = Path(args.out_dir) if args.out_dir else db_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    # Two-point functions
    plt.figure(figsize=(6, 4))
    plt.errorbar(taus, c2["mean"], yerr=c2["err"], fmt="o", label="C(tau)")
    c2i_mean = np.asarray(c2i["mean"], dtype=float)
    c2i_err = np.asarray(c2i["err"], dtype=float)
    valid_i = np.isfinite(c2i_mean) & np.isfinite(c2i_err)
    plt.errorbar(taus[valid_i], c2i_mean[valid_i], yerr=c2i_err[valid_i], fmt="o", label="Improved C(tau)")
    fit_params = None
    try:
        fit_mask = taus >= args.tau_min
        fit_taus = taus[fit_mask]
        fit_means = np.asarray(c2["mean"], dtype=float)[fit_mask]
        if args.ncosh <= 1:
            A, m = fit_cosh(fit_taus, fit_means, L)
            fit_params = [(A, m)]
        else:
            fit_params = fit_multi_cosh(fit_taus, fit_means, L, n_cosh=args.ncosh)
        t_fit = np.linspace(taus.min(), taus.max(), 200)
        fit_curve = np.zeros_like(t_fit, dtype=np.float64)
        for A, m in fit_params:
            fit_curve += A * np.cosh(m * (t_fit - 0.5 * L))
        plt.plot(t_fit, fit_curve, "-", alpha=0.6, label="Cosh fit")
    except Exception:
        pass
    plt.xlabel("tau")
    plt.ylabel("C(tau) (sum over sites)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"c2pt_vs_tau{tag}.png", dpi=150)
    plt.savefig(out_dir / f"c2pt_vs_tau{tag}.pdf")
    plt.close()

    # Two-point functions (semilog)
    plt.figure(figsize=(6, 4))
    plt.errorbar(taus, c2["mean"], yerr=c2["err"], fmt="o", label="C(tau)")
    plt.errorbar(taus[valid_i], c2i_mean[valid_i], yerr=c2i_err[valid_i], fmt="o", label="Improved C(tau)")
    if fit_params is not None:
        t_fit = np.linspace(taus.min(), taus.max(), 200)
        fit_curve = np.zeros_like(t_fit, dtype=np.float64)
        for A, m in fit_params:
            fit_curve += A * np.cosh(m * (t_fit - 0.5 * L))
        plt.plot(t_fit, fit_curve, "-", alpha=0.6, label="Cosh fit")
    plt.yscale("log")
    plt.xlabel("tau")
    plt.ylabel("C(tau) (sum over sites)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"c2pt_vs_tau_semilog{tag}.png", dpi=150)
    plt.savefig(out_dir / f"c2pt_vs_tau_semilog{tag}.pdf")
    plt.close()

    # Ratio to fit
    if fit_params is not None:
        fit_vals = np.zeros_like(taus, dtype=np.float64)
        for A, m in fit_params:
            fit_vals += A * np.cosh(m * (taus - 0.5 * L))
        plt.figure(figsize=(6, 4))
        plt.errorbar(taus, np.asarray(c2["mean"]) / fit_vals, yerr=np.asarray(c2["err"]) / fit_vals, fmt="o", label="C/fit")
        plt.errorbar(taus[valid_i], c2i_mean[valid_i] / fit_vals[valid_i],
                     yerr=c2i_err[valid_i] / fit_vals[valid_i], fmt="o", label="Cimp/fit")
        plt.axhline(1.0, color="#888888", lw=1.2, alpha=0.9)
        plt.xlabel("tau")
        plt.ylabel("C(tau) / fit")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"c2pt_ratio_to_fit{tag}.png", dpi=150)
        plt.savefig(out_dir / f"c2pt_ratio_to_fit{tag}.pdf")
        plt.close()

    # Control variate
    plt.figure(figsize=(6, 4))
    cv_mean = np.asarray(cv["mean"], dtype=float)
    cv_err = np.asarray(cv["err"], dtype=float)
    valid_cv = np.isfinite(cv_mean) & np.isfinite(cv_err)
    plt.errorbar(taus[valid_cv], cv_mean[valid_cv], yerr=cv_err[valid_cv], fmt="o", label="Control variate")
    plt.axhline(0.0, color="#888888", lw=1.2, alpha=0.9)
    plt.xlabel("tau")
    plt.ylabel("F_U (sum over sites)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"control_variate_vs_tau{tag}.png", dpi=150)
    plt.savefig(out_dir / f"control_variate_vs_tau{tag}.pdf")
    plt.close()

    # Gain
    plt.figure(figsize=(6, 4))
    gain_mean = np.asarray(gain["jk_mean"], dtype=float)
    gain_err = np.asarray(gain["jk_err"], dtype=float)
    valid_g = np.isfinite(gain_mean) & np.isfinite(gain_err)
    plt.errorbar(taus[valid_g], gain_mean[valid_g], yerr=gain_err[valid_g], fmt="o", label="Gain")
    plt.xlabel("tau")
    plt.ylabel("Variance gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"gain_vs_tau{tag}.png", dpi=150)
    plt.savefig(out_dir / f"gain_vs_tau{tag}.pdf")
    plt.close()

    # Covariance
    plt.figure(figsize=(6, 4))
    cov_mean = np.asarray(cov["jk_mean"], dtype=float)
    cov_err = np.asarray(cov["jk_err"], dtype=float)
    valid_c = np.isfinite(cov_mean) & np.isfinite(cov_err)
    plt.errorbar(taus[valid_c], cov_mean[valid_c], yerr=cov_err[valid_c], fmt="o", label="Norm. Cov(C, Cimp)")
    plt.xlabel("tau")
    plt.ylabel("Normalized covariance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"covariance_vs_tau{tag}.png", dpi=150)
    plt.savefig(out_dir / f"covariance_vs_tau{tag}.pdf")
    plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
