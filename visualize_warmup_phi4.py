#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warm up with baseline HMC, save images for each configuration in the batch.
Then warm up with MG-HMC using a trained map, save images again.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch as tr
import matplotlib.pyplot as plt

import integrators as i
import update as u
import phi4 as s
from phi4_sokal_mg import (
    sokal_coarse_hmc_update_mapped,
    LocalAdditiveCoupling2x2,
)


def save_batch_images(phi: tr.Tensor, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    phi_np = phi.detach().cpu().numpy()
    for b in range(phi_np.shape[0]):
        plt.figure()
        plt.imshow(phi_np[b, :, :], cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.title(f"{prefix} batch={b}")
        path = os.path.join(out_dir, f"{prefix}_b{b}.png")
        plt.savefig(path, dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize HMC vs MG warmup configurations")
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=-0.205)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--Nwarm-hmc", type=int, default=200)
    parser.add_argument("--Nwarm-mg", type=int, default=200)
    parser.add_argument("--fine-Nmd", type=int, default=7)
    parser.add_argument("--fine-tau", type=float, default=1.0)
    parser.add_argument("--coarse-Nmd", type=int, default=5)
    parser.add_argument("--coarse-tau", type=float, default=1.0)
    parser.add_argument("--m-coarse", type=int, default=1)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--map-steps", type=int, default=1)
    parser.add_argument("--map-hidden", type=int, default=32)
    parser.add_argument("--map-no-shift", action="store_true")
    parser.add_argument("--map-checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="warmup_images")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tr.manual_seed(args.seed)

    device = tr.device("cpu")
    lat = [args.L, args.L]

    fine = s.phi4(lat, args.lam, args.mass, batch_size=args.batch_size, device=device)
    phi = fine.hotStart()

    fine_I = i.minnorm2(fine.force, fine.evolveQ, args.fine_Nmd, args.fine_tau)
    fine_hmc = u.hmc(T=fine, I=fine_I, verbose=False)

    print("Warming up baseline HMC...")
    for _ in range(args.Nwarm_hmc):
        phi = fine_hmc.evolve(phi, 1)

    save_batch_images(phi, args.out_dir, "hmc")

    vmap = LocalAdditiveCoupling2x2(
        n_steps=args.map_steps,
        hidden=args.map_hidden,
        use_shift=not args.map_no_shift,
    )
    ckpt = tr.load(args.map_checkpoint, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    vmap.load_state_dict(state)
    for p in vmap.parameters():
        p.requires_grad_(False)

    print("Warming up MG-HMC (mapped coarse updates)...")
    for _ in range(args.Nwarm_mg):
        phi = fine_hmc.evolve(phi, 1)
        if args.m_coarse > 0:
            phi, _, _ = sokal_coarse_hmc_update_mapped(
                phi,
                fine,
                vmap,
                Nmd=args.coarse_Nmd,
                tau=args.coarse_tau,
                Nhmc=args.m_coarse,
                block=args.block,
                integrator="minnorm2",
                verbose=False,
                return_acceptance=False,
            )

    save_batch_images(phi, args.out_dir, "mg")
    print("Saved images to", args.out_dir)


if __name__ == "__main__":
    main()
