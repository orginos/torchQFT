#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy per-level training of local maps for multilevel MG.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import torch as tr

import integrators as i
import update as u
import phi4 as s
from phi4_sokal_mg import SokalRG2x2, LocalAdditiveCoupling2x2
from mg_hmc_phi4_multilevel import compute_levels


def pack2x2(x: tr.Tensor) -> tr.Tensor:
    x4 = x.unsqueeze(1)
    c00 = x4[:, :, 0::2, 0::2]
    c01 = x4[:, :, 0::2, 1::2]
    c10 = x4[:, :, 1::2, 0::2]
    c11 = x4[:, :, 1::2, 1::2]
    return tr.cat([c00, c01, c10, c11], dim=1)


def cov_loss(Phi: tr.Tensor, pi: tr.Tensor) -> tr.Tensor:
    Phi_c = Phi - Phi.mean()
    pi4 = pack2x2(pi)
    loss = 0.0
    for c in range(pi4.shape[1]):
        pc = pi4[:, c] - pi4[:, c].mean()
        cov = (Phi_c * pc).mean()
        loss = loss + cov * cov
    return loss / pi4.shape[1]


def main():
    parser = argparse.ArgumentParser(description="Greedy multilevel map training for phi^4")
    parser.add_argument("--Lx", type=int, default=32)
    parser.add_argument("--Ly", type=int, default=32)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--levels", type=int, default=None)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=-0.205)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--Nwarm", type=int, default=200)
    parser.add_argument("--Ntrain", type=int, default=500)
    parser.add_argument("--Nskip", type=int, default=5)
    parser.add_argument("--fine-Nmd", type=int, default=7)
    parser.add_argument("--fine-tau", type=float, default=1.0)
    parser.add_argument("--fine-integrator", choices=["minnorm2", "leapfrog"], default="minnorm2")
    parser.add_argument("--map-steps", type=int, default=1)
    parser.add_argument("--map-hidden", type=int, default=32)
    parser.add_argument("--map-no-shift", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="maps_multilevel")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tr.manual_seed(args.seed)

    levels = compute_levels(args.Lx, args.Ly, args.min_size, args.levels)
    if levels <= 0:
        raise SystemExit("Computed zero levels; increase lattice size or lower min_size.")

    device = tr.device("cpu")
    lat = [args.Lx, args.Ly]

    fine = s.phi4(lat, args.lam, args.mass, batch_size=args.batch_size, device=device)
    phi = fine.hotStart()

    if args.fine_integrator == "minnorm2":
        fine_I = i.minnorm2(fine.force, fine.evolveQ, args.fine_Nmd, args.fine_tau)
    else:
        fine_I = i.leapfrog(fine.force, fine.evolveQ, args.fine_Nmd, args.fine_tau)
    fine_hmc = u.hmc(T=fine, I=fine_I, verbose=False)

    rg = SokalRG2x2(block=2)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Warming up HMC...")
    for _ in range(args.Nwarm):
        phi = fine_hmc.evolve(phi, 1)

    vmaps: List[LocalAdditiveCoupling2x2] = []
    ckpts: List[str] = []

    for level in range(levels):
        print(f"Training level {level} map...")
        vmap = LocalAdditiveCoupling2x2(
            n_steps=args.map_steps,
            hidden=args.map_hidden,
            use_shift=not args.map_no_shift,
        )
        opt = tr.optim.Adam(vmap.parameters(), lr=args.lr)

        for it in range(args.Ntrain):
            for _ in range(args.Nskip):
                phi = fine_hmc.evolve(phi, 1)
            with tr.no_grad():
                cur = phi.detach()
                for j in range(level):
                    psi = vmaps[j].g(cur)
                    Phi, _ = rg.coarsen(psi)
                    cur = Phi

            opt.zero_grad()
            psi = vmap.g(cur)
            Phi, pi = rg.coarsen(psi)
            loss = cov_loss(Phi, pi)
            loss.backward()
            opt.step()

            if it % 50 == 0:
                print(f"level {level} iter {it} loss {loss.item()}")

        ckpt_path = os.path.join(args.out_dir, f"map_level_{level}.pt")
        tr.save(
            {
                "state_dict": vmap.state_dict(),
                "config": {
                    "map": "additive",
                    "map_steps": args.map_steps,
                    "map_hidden": args.map_hidden,
                    "map_no_shift": bool(args.map_no_shift),
                    "level": level,
                },
            },
            ckpt_path,
        )
        print("Saved", ckpt_path)
        vmaps.append(vmap)
        ckpts.append(ckpt_path)

    meta = {
        "levels": levels,
        "checkpoints": ckpts,
        "map_steps": args.map_steps,
        "map_hidden": args.map_hidden,
        "map_no_shift": bool(args.map_no_shift),
    }
    with open(os.path.join(args.out_dir, "maps_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta:", os.path.join(args.out_dir, "maps_meta.json"))


if __name__ == "__main__":
    main()
