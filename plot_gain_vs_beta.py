import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_db(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbs", nargs="+", required=True, help="list of json db files or glob patterns")
    parser.add_argument("--tau", type=int, required=True, help="tau to extract gain for")
    parser.add_argument("--out", default="gain_vs_beta", help="output filename prefix")
    parser.add_argument("--title", default=None, help="plot title")
    parser.add_argument("--ylog", action="store_true", help="log-scale gain axis")
    parser.add_argument("--deg", type=int, default=0, help="polynomial degree for poly fit")
    parser.add_argument("--fit", choices=["poly", "expb", "expb2", "none"], default="expb",
                        help="fit model: poly (1+A/[b^4(1+bP)]), expb (1+(A/b^4)exp(-c b)), expb2 (1+(A/b^4)exp(-m b^2))")
    parser.add_argument("--offset", action="store_true", help="include the +1 offset for exp fits")
    parser.add_argument("--beta_max", type=float, default=None, help="maximum beta used in fit")
    args = parser.parse_args()

    paths = []
    for item in args.dbs:
        p = Path(item)
        if "*" in item or "?" in item or ("[" in item and "]" in item):
            paths.extend(sorted(Path().glob(item)))
        elif p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        else:
            paths.append(p)

    betas = []
    gains = []
    errs = []
    skipped = 0

    for path in paths:
        if not path.is_file():
            continue
        data = load_db(path)
        taus = data.get("taus", [])
        if args.tau not in taus:
            skipped += 1
            continue
        idx = taus.index(args.tau)
        g = data["gain"]["jk_mean"][idx]
        e = data["gain"]["jk_err"][idx]
        if g is None or e is None:
            skipped += 1
            continue
        betas.append(data["meta"]["beta"])
        gains.append(float(g))
        errs.append(float(e))

    if len(betas) == 0:
        raise SystemExit("No valid entries found for requested tau.")

    betas = np.asarray(betas, dtype=float)
    gains = np.asarray(gains, dtype=float)
    errs = np.asarray(errs, dtype=float)

    order = np.argsort(betas)
    betas = betas[order]
    gains = gains[order]
    errs = errs[order]

    chi2 = None
    n_fit = 0
    plt.figure(figsize=(6, 4))
    plt.errorbar(betas, gains, yerr=errs, fmt="o", label=f"tau={args.tau}")
    if args.fit == "poly":
        # fit g = 1 + A / [beta^4 (1 + beta P(beta))]
        # => 1/(g-1) = c0*beta^4 + sum_{k=0..deg} c_{k+1} * beta^{k+5}
        fit_mask = (betas > 0) & (gains > 1)
        if args.beta_max is not None:
            fit_mask = fit_mask & (betas <= args.beta_max)
        if np.any(fit_mask):
            b_fit_data = betas[fit_mask]
            g_fit_data = gains[fit_mask]
            y = 1.0 / (g_fit_data - 1.0)
            cols = [b_fit_data ** 4]
            for k in range(args.deg + 1):
                cols.append(b_fit_data ** (k + 5))
            X = np.vstack(cols).T
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            b_fit = np.linspace(betas.min(), betas.max(), 200)
            denom = coeffs[0] * (b_fit ** 4)
            for k in range(args.deg + 1):
                denom += coeffs[k + 1] * (b_fit ** (k + 5))
            valid_d = denom > 0
            if np.any(valid_d):
                g_fit = 1.0 + 1.0 / denom
                plt.plot(b_fit[valid_d], g_fit[valid_d], "-", alpha=0.6, label=f"fit: poly deg={args.deg}")
                fit_vals = 1.0 + 1.0 / (coeffs[0] * (b_fit_data ** 4) + sum(
                    coeffs[k + 1] * (b_fit_data ** (k + 5)) for k in range(args.deg + 1)
                ))
                n_fit = len(b_fit_data)
                chi2 = np.sum(((g_fit_data - fit_vals) / errs[fit_mask]) ** 2)
        else:
            print("No valid points for poly fit (need beta>0 and gain>1).")
    elif args.fit == "expb":
        # fit g = 1 + (A/beta^4) * exp(-c beta)  (or without +1 if default)
        if args.offset:
            fit_mask = (betas > 0) & (gains > 1)
        else:
            fit_mask = (betas > 0) & (gains > 0)
        if args.beta_max is not None:
            fit_mask = fit_mask & (betas <= args.beta_max)
        if np.any(fit_mask):
            b_fit_data = betas[fit_mask]
            g_fit_data = gains[fit_mask]
            if args.offset:
                y = np.log((g_fit_data - 1.0) * (b_fit_data ** 4))
            else:
                y = np.log(g_fit_data * (b_fit_data ** 4))
            X = np.vstack([np.ones_like(b_fit_data), -b_fit_data]).T
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            logA, c = coeffs
            A = np.exp(logA)
            b_fit = np.linspace(betas.min(), betas.max(), 200)
            g_fit = (A / (b_fit ** 4)) * np.exp(-c * b_fit)
            if args.offset:
                g_fit = 1.0 + g_fit
            plt.plot(b_fit, g_fit, "-", alpha=0.6,
                     label="fit: 1+(A/b^4)exp(-c b)" if args.offset else "fit: (A/b^4)exp(-c b)")
            fit_vals = (A / (b_fit_data ** 4)) * np.exp(-c * b_fit_data)
            if args.offset:
                fit_vals = 1.0 + fit_vals
            n_fit = len(b_fit_data)
            chi2 = np.sum(((g_fit_data - fit_vals) / errs[fit_mask]) ** 2)
    elif args.fit == "expb2":
        # fit g = 1 + (A/beta^4) * exp(-m beta^2) (or without +1 if default)
        if args.offset:
            fit_mask = (betas > 0) & (gains > 1)
        else:
            fit_mask = (betas > 0) & (gains > 0)
        if args.beta_max is not None:
            fit_mask = fit_mask & (betas <= args.beta_max)
        if np.any(fit_mask):
            b_fit_data = betas[fit_mask]
            g_fit_data = gains[fit_mask]
            if args.offset:
                y = np.log((g_fit_data - 1.0) * (b_fit_data ** 4))
            else:
                y = np.log(g_fit_data * (b_fit_data ** 4))
            X = np.vstack([np.ones_like(b_fit_data), -(b_fit_data ** 2)]).T
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            logA, m = coeffs
            A = np.exp(logA)
            b_fit = np.linspace(betas.min(), betas.max(), 200)
            g_fit = (A / (b_fit ** 4)) * np.exp(-m * (b_fit ** 2))
            if args.offset:
                g_fit = 1.0 + g_fit
            plt.plot(b_fit, g_fit, "-", alpha=0.6,
                     label="fit: 1+(A/b^4)exp(-m b^2)" if args.offset else "fit: (A/b^4)exp(-m b^2)")
            fit_vals = (A / (b_fit_data ** 4)) * np.exp(-m * (b_fit_data ** 2))
            if args.offset:
                fit_vals = 1.0 + fit_vals
            n_fit = len(b_fit_data)
            chi2 = np.sum(((g_fit_data - fit_vals) / errs[fit_mask]) ** 2)
    plt.xlabel("beta")
    plt.ylabel("Variance gain")
    if args.title:
        plt.title(args.title)
    if args.ylog:
        plt.yscale("log")
    plt.tight_layout()

    tag_parts = [args.fit]
    if args.fit in ("expb", "expb2"):
        tag_parts.append("off" if args.offset else "nooff")
    if args.fit == "poly":
        tag_parts.append(f"deg{args.deg}")
    if args.beta_max is not None:
        tag_parts.append(f"bmax{args.beta_max}")
    fit_tag = "_".join(tag_parts) if args.fit != "none" else "nofit"

    out_base = Path(f"{args.out}_{fit_tag}")
    plt.savefig(out_base.with_suffix(".png"), dpi=150)
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.close()

    print("beta  gain  err")
    for b, g, e in zip(betas, gains, errs):
        print(f"{b}  {g}  {e}")
    if chi2 is not None and n_fit > 0:
        dof = max(n_fit - 2, 1)
        print(f"Fit chi^2 = {chi2:.3f} (dof={dof}, chi^2/dof={chi2/dof:.3f})")

    if skipped:
        print(f"Skipped {skipped} db(s) without valid gain at tau={args.tau}.")
    print(f"Saved: {out_base.with_suffix('.png')} and {out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
