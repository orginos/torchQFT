#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train stacked multigrid (MG) flows with parity-aware S/T networks and architecture-free resume.
Includes optional SWA fine-tuning after normal training.
"""

from __future__ import annotations
import argparse, os, time, json, datetime
import numpy as np
import torch as tr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import phi4 as p
from updated_phi4_mg import MGflow, FlowBijector, RGlayer, upscale_mgflow
from updated_stacked_model import StackedModel, trainSM, trainSMaveraged

# ---------------------------- helpers ----------------------------

def pick_device(pref: str):
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return tr.device("cpu")
    if pref == "cuda" and tr.cuda.is_available():
        return tr.device("cuda")
    if pref == "mps":
        try:
            if getattr(tr.backends, "mps", None) and tr.backends.mps.is_available():
                return tr.device("mps")
        except Exception:
            pass
        return tr.device("cpu")
    # auto
    if tr.cuda.is_available():
        return tr.device("cuda")
    try:
        if getattr(tr.backends, "mps", None) and tr.backends.mps.is_available():
            return tr.device("mps")
    except Exception:
        pass
    return tr.device("cpu")


def build_stage(L, width, layers, steps, rg_type, fixed_bijector, device, max_logscale, parity):
    bij_factory = FlowBijector(n_layers=layers, width=width,
                               log_scale_clip=max_logscale, parity=parity)
    rg = RGlayer(rg_type)
    return MGflow(size=(L, L), bijector_factory=bij_factory, rg=rg,
                  Nconvs=steps, fixed_bijector=fixed_bijector).to(device)


def _collect_finite_deltaS(model: StackedModel, action_fn, target_count: int, start_bs: int, max_tries: int = 8):
    device = next(model.parameters()).device
    collected, tried, bs = [], 0, start_bs
    with tr.no_grad():
        while sum(len(t) for t in collected) < target_count and tried < max_tries:
            z = model.prior_sample(bs, device=device)
            x = model.g(z)
            deltaS = (model.log_prob(x) + action_fn(x))  # [B]
            finite = tr.isfinite(deltaS)
            if finite.any():
                collected.append(deltaS[finite].detach().cpu())
            bs = max(64, bs // 2); tried += 1
    return tr.cat(collected)[:target_count] if collected else tr.tensor([])


def validate_and_plots(model: StackedModel, action_fn, batch_val: int, pdf_path: str, history: list):
    deltaS_cpu = _collect_finite_deltaS(model, action_fn, target_count=batch_val,
                                        start_bs=batch_val, max_tries=10)
    n_finite = int(deltaS_cpu.numel())

    with PdfPages(pdf_path) as pdf:
        fig0 = plt.figure(); plt.axis("off")
        txt = "Validation summary\n\n" + f"Requested batch: {batch_val}\nFinite ΔS collected: {n_finite}\n"
        plt.text(0.02, 0.98, txt, va="top", ha="left"); pdf.savefig(fig0); plt.close(fig0)

        fig1 = plt.figure(); plt.plot(history)
        plt.xlabel("Epoch"); plt.ylabel("Loss (ΔS mean)"); plt.title("Training history")
        pdf.savefig(fig1); plt.close(fig1)

        if n_finite == 0:
            return dict(ESS=0.0, mean_deltaS=float("nan"), std_deltaS=float("nan"),
                        mean_log10w=float("nan"), std_log10w=float("nan"), finite_frac=0.0)

        deltaS_np = deltaS_cpu.numpy()
        logw = (-deltaS_cpu.double())
        m = float(logw.max().item())
        wtilde = tr.exp(logw - m)
        ess = float((wtilde.sum()**2 / (wtilde.pow(2).sum())).item())
        log10w_np = (logw / np.log(10.0)).numpy()

        fig2 = plt.figure(); plt.hist(deltaS_np, bins=50)
        plt.xlabel("ΔS"); plt.ylabel("Count"); plt.title("ΔS distribution (finite subset)")
        pdf.savefig(fig2); plt.close(fig2)

        fig3 = plt.figure(); plt.hist(log10w_np, bins=50)
        plt.xlabel("log10 w (w = exp(-ΔS))"); plt.ylabel("Count"); plt.title("Reweighting (finite subset)")
        pdf.savefig(fig3); plt.close(fig3)

    return dict(ESS=ess, mean_deltaS=float(deltaS_np.mean()), std_deltaS=float(deltaS_np.std()),
                mean_log10w=float(log10w_np.mean()), std_log10w=float(log10w_np.std()),
                finite_frac=n_finite/float(batch_val))


# -------------------- arch save/load helpers --------------------

def arch_from_args(args):
    return {
        "model_type": "stacked_mgflow",
        "L": args.L,
        "stages": args.stages,
        "width": args.width,
        "layers": args.layers,
        "steps": args.steps,
        "rg": args.rg,
        "fixed_bijector": bool(args.fixed_bijector),
        "parity": args.parity,
        "max_logscale": args.max_logscale,
    }


def build_model_from_arch(arch: dict, device: tr.device) -> StackedModel:
    L = int(arch["L"])
    stages = int(arch["stages"])
    width = int(arch["width"])
    layers = int(arch["layers"])
    steps = int(arch["steps"])
    rg = arch["rg"]
    fixed_bijector = bool(arch["fixed_bijector"])
    parity = arch["parity"]
    max_logscale = float(arch["max_logscale"])

    stage_list = [build_stage(L=L, width=width, layers=layers, steps=steps,
                              rg_type=rg, fixed_bijector=fixed_bijector, device=device,
                              max_logscale=max_logscale, parity=parity)
                  for _ in range(stages)]
    M = StackedModel(stage_list, action_fn=lambda x: x.sum()*0.0).to(device)
    return M


# ------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser()
    # arch-ish
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--rg", type=str, default="average", choices=["average","select"])
    ap.add_argument("--fixed-bijector", action="store_true")
    ap.add_argument("--parity", type=str, default="none", choices=["none","sym","x2"])
    ap.add_argument("--max-logscale", type=float, default=5.0)

    # training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--superbatch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--joint", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=1.0)

    # SWA fine-tune
    ap.add_argument("--swa-epochs", type=int, default=0, help=">0 to run SWA after normal training")
    ap.add_argument("--swa-start-frac", type=float, default=0.5, help="fraction of SWA epochs after which averaging starts")
    ap.add_argument("--swa-lr-mult", type=float, default=0.1, help="SWA LR = lr * swa_lr_mult")

    # physics
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--mass", type=float, default=-0.2)

    # io
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","mps","cpu"])
    ap.add_argument("--save-dir", type=str, default="checkpoints")
    ap.add_argument("--tag", type=str, default="mgflow")
    ap.add_argument("--validate-batch", type=int, default=4096)

    # resume/upscale
    ap.add_argument("--resume", type=str, default="", help="checkpoint with config['arch']")
    ap.add_argument("--resume-opt", action="store_true", help="also resume optimizer/scaler")
    ap.add_argument("--upscale-from", type=str, default="", help="checkpoint to upscale from smaller L")

    args = ap.parse_args()
    device = pick_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Build model either from checkpoint arch (resume) or from args (fresh) ----
    history = []
    if args.resume:
        ckpt = tr.load(args.resume, map_location=device)
        cfg = ckpt.get("config", {})
        arch = cfg.get("arch", None) or cfg  # backward compat
        train_cfg = cfg.get("train", {})

        SuperM = build_model_from_arch(arch, device)
        SuperM.load_state_dict(ckpt["model"], strict=True)
        print(f"[resume] rebuilt StackedModel from checkpoint arch: "
              f"L={arch['L']} stages={arch['stages']} width={arch['width']} "
              f"layers={arch['layers']} steps={arch['steps']} parity={arch['parity']} rg={arch['rg']}")

        lam = float(train_cfg.get("lam", args.lam))
        mass = float(train_cfg.get("mass", args.mass))
        batch = int(train_cfg.get("batch", args.batch))
        superbatch = int(train_cfg.get("superbatch", args.superbatch))
        lr = float(train_cfg.get("lr", args.lr))

        o = p.phi4([arch["L"], arch["L"]], lam, mass, batch_size=batch)
        SuperM.action_fn = lambda x: o.action(x)
    else:
        arch = arch_from_args(args)
        SuperM = build_model_from_arch(arch, device)

        lam, mass = args.lam, args.mass
        batch, superbatch, lr = args.batch, args.superbatch, args.lr

        if args.upscale_from:
            old = tr.load(args.upscale_from, map_location=device)
            cfg_old = old.get("config", {})
            arch_old = cfg_old.get("arch", None) or cfg_old
            OldM = build_model_from_arch(arch_old, device)
            OldM.load_state_dict(old["model"], strict=True)
            print(f"[upscale] loaded old model (L={arch_old['L']}) -> new L={arch['L']}")
            for s in range(min(int(arch_old["stages"]), int(arch["stages"]))):
                up = upscale_mgflow(
                    OldM.stages[s],
                    new_size=(arch["L"], arch["L"]),
                    bijector_factory=FlowBijector(n_layers=arch["layers"], width=arch["width"],
                                                  log_scale_clip=arch["max_logscale"], parity=arch["parity"]),
                    rg=RGlayer(arch["rg"]),
                    Nconvs=arch["steps"],
                    fixed_bijector=bool(arch["fixed_bijector"]),
                )
                SuperM.stages[s] = up

        o = p.phi4([arch["L"], arch["L"]], lam, mass, batch_size=batch)
        SuperM.action_fn = lambda x: o.action(x)

    # ---- Sanity ----
    with tr.no_grad():
        z = SuperM.prior_sample(8, device=device)
        x = SuperM.g(z); z2, _ = SuperM.f(x)
        print(f"[Sanity] round-trip max|z2-z| = {(z2 - z).abs().max().item():.3e}")

    # ---- Base optimizer/scaler ----
    opt = tr.optim.AdamW(SuperM.parameters(), lr=lr, betas=(0.9, 0.99))
    scaler = tr.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    if args.resume and args.resume_opt:
        try:
            opt.load_state_dict(ckpt["optimizer"]); print("[resume] optimizer state loaded.")
        except Exception:
            print("[resume] WARN: optimizer mismatch; starting fresh.")
        try:
            if scaler.is_enabled() and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"]); print("[resume] AMP scaler loaded.")
        except Exception:
            print("[resume] WARN: scaler mismatch; starting fresh.")
        history = list(ckpt.get("history", []))

    # ---- Train: warmup per stage, then joint ----
    tic = time.perf_counter()
    hist = trainSM(
        SuperM, epochs=args.epochs, batch_size=batch, super_batch_size=superbatch,
        lr=lr, warmup_per_stage=args.warmup, joint_per_stage=args.joint,
        use_amp=(device.type == 'cuda'), compile_model=False, grad_clip=args.clip_grad,
        optimizer=opt, scaler=scaler, tqdm_desc="Training (Stacked MG)"
    )
    history.extend(hist)
    toc = time.perf_counter()
    print(f"[base] finished in {toc-tic:.2f}s. Last loss: {history[-1]:.6f}")

    # ---- Save base checkpoint ----
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_ckpt = os.path.join(args.save_dir, f"{args.tag}_L{arch['L']}_S{arch['stages']}_{stamp}.pth")
    base_config = {
        "arch": arch,
        "train": {
            "lr": lr, "batch": batch, "superbatch": superbatch,
            "lam": lam, "mass": mass, "warmup": args.warmup, "joint": args.joint,
        },
        "code": {"script": "updated_train_stacked.py", "version": 3},
    }
    tr.save({
        "model": SuperM.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
        "history": history,
        "config": base_config,
    }, base_ckpt)
    print(f"[base] saved: {base_ckpt}")

    # ---- Optional SWA fine-tune ----
    if args.swa_epochs > 0:
        print(f"[SWA] running SWA for {args.swa_epochs} epochs "
              f"(start_frac={args.swa_start_frac}, lr_mult={args.swa_lr_mult})")
        swa_hist, swa_state = trainSMaveraged(
            SuperM,
            epochs=args.swa_epochs,
            batch_size=batch,
            super_batch_size=superbatch,
            lr=lr,
            swa_start_frac=args.swa_start_frac,
            swa_lr_mult=args.swa_lr_mult,
            use_amp=(device.type == 'cuda'),
            grad_clip=args.clip_grad,
            optimizer=tr.optim.AdamW(SuperM.parameters(), lr=lr, betas=(0.9, 0.99)),
            scaler=tr.amp.GradScaler('cuda', enabled=(device.type == 'cuda')),
            tqdm_desc="Training(SWA)"
        )
        # Load averaged weights into the active model
        SuperM.load_state_dict(swa_state, strict=True)
        history.extend(swa_hist)

        # Save SWA checkpoint and validate
        swa_ckpt = os.path.join(args.save_dir, f"{args.tag}_L{arch['L']}_S{arch['stages']}_{stamp}_SWA.pth")
        swa_config = {
            "arch": arch,
            "train": {
                "lr": lr, "batch": batch, "superbatch": superbatch,
                "lam": lam, "mass": mass,
                "warmup": args.warmup, "joint": args.joint,
                "swa_epochs": args.swa_epochs,
                "swa_start_frac": args.swa_start_frac,
                "swa_lr_mult": args.swa_lr_mult,
            },
            "code": {"script": "updated_train_stacked.py", "version": 3, "swa": True},
        }
        tr.save({
            "model": SuperM.state_dict(),
            "history": history,
            "config": swa_config,
        }, swa_ckpt)
        print(f"[SWA] saved: {swa_ckpt}")

        # Validation for SWA weights
        pdf_path = os.path.join(args.save_dir, f"{args.tag}_L{arch['L']}_S{arch['stages']}_validation_{stamp}_SWA.pdf")
        metrics = validate_and_plots(SuperM, action_fn=lambda x: o.action(x),
                                     batch_val=args.validate_batch, pdf_path=pdf_path, history=history)
        metrics_path = os.path.join(args.save_dir, f"{args.tag}_L{arch['L']}_S{arch['stages']}_metrics_{stamp}_SWA.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SWA] Validation PDF: {pdf_path}")
        print(f"[SWA] Metrics JSON:   {metrics_path}")
        print(f"[SWA] Finite fraction: {metrics['finite_frac']:.3f} | ESS: {metrics['ESS']:.1f}")
    else:
        # Base validation only
        pdf_path = os.path.join(args.save_dir, f"{args.tag}_L{arch['L']}_S{arch['stages']}_validation_{stamp}.pdf")
        metrics = validate_and_plots(SuperM, action_fn=lambda x: o.action(x),
                                     batch_val=args.validate_batch, pdf_path=pdf_path, history=history)
        metrics_path = os.path.join(args.save_dir, f"{args.tag}_L{arch['L']}_S{arch['stages']}_metrics_{stamp}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Validation PDF: {pdf_path}")
        print(f"Metrics JSON:   {metrics_path}")
        print(f"Finite fraction: {metrics['finite_frac']:.3f} | ESS: {metrics['ESS']:.1f}")

if __name__ == "__main__":
    main()
