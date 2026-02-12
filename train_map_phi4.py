#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a volume-preserving local map F for phi^4 MG.
Objective: minimize covariance between coarse field Phi and local residuals pi.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

import numpy as np
import torch as tr

import integrators as i
import update as u
import phi4 as s
from phi4_sokal_mg import SokalRG2x2, LocalAdditiveCoupling2x2


def pack2x2(x: tr.Tensor) -> tr.Tensor:
    # x: [B, H, W] -> [B, 4, H/2, W/2]
    x4 = x.unsqueeze(1)
    c00 = x4[:, :, 0::2, 0::2]
    c01 = x4[:, :, 0::2, 1::2]
    c10 = x4[:, :, 1::2, 0::2]
    c11 = x4[:, :, 1::2, 1::2]
    return tr.cat([c00, c01, c10, c11], dim=1)


def cov_loss(Phi: tr.Tensor, pi: tr.Tensor, return_corr: bool = False):
    # Feature covariances between A=[Phi,Phi^2,Phi^3] and B=[pi,pi^2,pi^3]
    pi4 = pack2x2(pi)  # [B,4,H/2,W/2]
    A = [Phi, Phi**2, Phi**3]
    B = [pi4, pi4**2, pi4**3]

    loss = 0.0
    count = 0
    for a in A:
        ac = a - a.mean()
        astd = ac.std() + 1e-8
        ac = ac / astd
        for b in B:
            # b has 4 channels
            for c in range(b.shape[1]):
                bc = b[:, c] - b[:, c].mean()
                bstd = bc.std() + 1e-8
                bc = bc / bstd
                cov = (ac * bc).mean()
                loss = loss + cov * cov
                count += 1
    loss = loss / max(count, 1)
    if return_corr:
        mean_abs_corr = tr.sqrt(loss)
        return loss, mean_abs_corr
    return loss


def main():
    parser = argparse.ArgumentParser(description="Train local map for MG phi^4")
    parser.add_argument("--L", type=int, default=32, help="Lattice size L (square LxL)")
    parser.add_argument("--lam", type=float, default=0.5, help="Phi^4 coupling lambda")
    parser.add_argument("--mass", type=float, default=-0.205, help="Bare mass m")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (parallel chains)")
    parser.add_argument("--Nwarm", type=int, default=200, help="Warmup HMC steps")
    parser.add_argument("--Ntrain", type=int, default=1000, help="Training steps")
    parser.add_argument("--Nskip", type=int, default=5, help="HMC steps between training batches")

    parser.add_argument("--fine-Nmd", type=int, default=7, help="Fine HMC MD steps")
    parser.add_argument("--fine-tau", type=float, default=1.0, help="Fine HMC trajectory length")
    parser.add_argument("--fine-integrator", choices=["minnorm2", "leapfrog"], default="minnorm2")

    parser.add_argument("--map-steps", type=int, default=1, help="Number of coupling steps in map")
    parser.add_argument("--map-hidden", type=int, default=32, help="Hidden width for map coupling net")
    parser.add_argument("--map-no-shift", action="store_true", help="Disable block shift in map")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (torch/numpy)")
    parser.add_argument("--save", type=str, default="map_phi4.pt", help="Checkpoint path")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tr.manual_seed(args.seed)

    device = tr.device("cpu")
    lat = [args.L, args.L]

    fine = s.phi4(lat, args.lam, args.mass, batch_size=args.batch_size, device=device)
    phi = fine.hotStart()

    if args.fine_integrator == "minnorm2":
        fine_I = i.minnorm2(fine.force, fine.evolveQ, args.fine_Nmd, args.fine_tau)
    else:
        fine_I = i.leapfrog(fine.force, fine.evolveQ, args.fine_Nmd, args.fine_tau)
    fine_hmc = u.hmc(T=fine, I=fine_I, verbose=False)

    vmap = LocalAdditiveCoupling2x2(
        n_steps=args.map_steps,
        hidden=args.map_hidden,
        use_shift=not args.map_no_shift,
    )
    vmap.train()
    opt = tr.optim.Adam(vmap.parameters(), lr=args.lr)
    rg = SokalRG2x2(block=2)

    print("Warming up HMC...")
    for _ in range(args.Nwarm):
        phi = fine_hmc.evolve(phi, 1)

    print("Training map...")
    for it in range(args.Ntrain):
        for _ in range(args.Nskip):
            phi = fine_hmc.evolve(phi, 1)
        with tr.no_grad():
            phi_det = phi.detach()

        opt.zero_grad()
        psi = vmap.g(phi_det)
        Phi, pi = rg.coarsen(psi)
        loss, mean_abs_corr = cov_loss(Phi, pi, return_corr=True)
        loss.backward()
        opt.step()

        if it % 50 == 0:
            print(f"iter {it} loss {loss.item()} mean|corr| {mean_abs_corr.item()}")

    ckpt: Dict[str, object] = {
        "state_dict": vmap.state_dict(),
        "config": {
            "map": "additive",
            "map_steps": args.map_steps,
            "map_hidden": args.map_hidden,
            "map_no_shift": bool(args.map_no_shift),
        },
    }
    tr.save(ckpt, args.save)
    print("Saved map to", args.save)


if __name__ == "__main__":
    main()
