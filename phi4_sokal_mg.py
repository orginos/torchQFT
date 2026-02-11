#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sokal-style 2x2 blocking HMC for scalar phi^4 in 2d.

We keep fine fluctuations (pi) fixed and update coarse fields (Phi)
using HMC on the coarse action Sc(Phi) = S(F(pi, Phi)).
The force for the coarse HMC is computed with autodiff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch as tr
import torch.nn as nn

import integrators as i
import update as u
import phi4 as s


@dataclass
class SokalRG2x2:
    """2x2 blocking with average coarse field and residual fluctuations."""

    block: int = 2

    def _check(self, phi: tr.Tensor) -> Tuple[int, int]:
        if phi.dim() != 3:
            raise ValueError("Expected phi with shape [B, Lx, Ly].")
        lx, ly = phi.shape[1], phi.shape[2]
        if lx % self.block != 0 or ly % self.block != 0:
            raise ValueError("Lattice size must be divisible by block size.")
        return lx, ly

    def coarsen(self, phi: tr.Tensor) -> Tuple[tr.Tensor, tr.Tensor]:
        """Return (Phi, pi) where Phi is the 2x2 block average and pi the residual."""
        lx, ly = self._check(phi)
        b = self.block
        # [B, Lx, Ly] -> [B, Lx/2, 2, Ly/2, 2]
        reshaped = phi.view(phi.shape[0], lx // b, b, ly // b, b)
        Phi = reshaped.mean(dim=(2, 4))
        fine_from_Phi = self.refine(Phi, tr.zeros_like(phi))
        pi = phi - fine_from_Phi
        return Phi, pi

    def refine(self, Phi: tr.Tensor, pi: tr.Tensor) -> tr.Tensor:
        """Map (Phi, pi) to fine field phi."""
        if Phi.dim() != 3 or pi.dim() != 3:
            raise ValueError("Expected Phi and pi with shape [B, Lx, Ly].")
        b = self.block
        fine = Phi.repeat_interleave(b, dim=1).repeat_interleave(b, dim=2)
        if fine.shape != pi.shape:
            raise ValueError("Phi and pi shapes are incompatible.")
        return fine + pi


class IdentityMap(nn.Module):
    def g(self, x: tr.Tensor) -> tr.Tensor:
        return x

    def f(self, x: tr.Tensor) -> tr.Tensor:
        return x


class _BlockCouplingNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        return self.net(x)


class LocalAdditiveCoupling2x2(nn.Module):
    """
    Volume-preserving local 2x2 additive coupling with optional block shifts.
    Jacobian determinant is 1.
    """
    def __init__(self, n_steps: int = 1, hidden: int = 32, use_shift: bool = True):
        super().__init__()
        self.n_steps = int(n_steps)
        self.use_shift = bool(use_shift)
        self.nets = nn.ModuleList([_BlockCouplingNet(hidden) for _ in range(2 * self.n_steps)])

    @staticmethod
    def _pack2x2(z4: tr.Tensor) -> tr.Tensor:
        c00 = z4[:, :, 0::2, 0::2]
        c01 = z4[:, :, 0::2, 1::2]
        c10 = z4[:, :, 1::2, 0::2]
        c11 = z4[:, :, 1::2, 1::2]
        return tr.cat([c00, c01, c10, c11], dim=1)

    @staticmethod
    def _unpack2x2(y: tr.Tensor) -> tr.Tensor:
        B, C, H2, W2 = y.shape
        out = y.new_zeros((B, 1, H2 * 2, W2 * 2))
        out[:, :, 0::2, 0::2] = y[:, 0:1]
        out[:, :, 0::2, 1::2] = y[:, 1:2]
        out[:, :, 1::2, 0::2] = y[:, 2:3]
        out[:, :, 1::2, 1::2] = y[:, 3:4]
        return out

    def _apply_g(self, z4: tr.Tensor, net: nn.Module) -> tr.Tensor:
        y = self._pack2x2(z4)
        B, C, H2, W2 = y.shape
        flat = y.permute(0, 2, 3, 1).contiguous().view(B * H2 * W2, C)
        A, Bv = tr.chunk(flat, 2, dim=1)
        t = net(A)
        Bv = Bv + t
        out = tr.cat([A, Bv], dim=1)
        y = out.view(B, H2, W2, C).permute(0, 3, 1, 2).contiguous()
        return self._unpack2x2(y)

    def _apply_f(self, x4: tr.Tensor, net: nn.Module) -> tr.Tensor:
        y = self._pack2x2(x4)
        B, C, H2, W2 = y.shape
        flat = y.permute(0, 2, 3, 1).contiguous().view(B * H2 * W2, C)
        A, Bv = tr.chunk(flat, 2, dim=1)
        t = net(A)
        Bv = Bv - t
        out = tr.cat([A, Bv], dim=1)
        y = out.view(B, H2, W2, C).permute(0, 3, 1, 2).contiguous()
        return self._unpack2x2(y)

    def g(self, z: tr.Tensor) -> tr.Tensor:
        z4 = z.view(z.shape[0], 1, z.shape[1], z.shape[2])
        for k in range(self.n_steps):
            net0 = self.nets[2 * k]
            net1 = self.nets[2 * k + 1]
            z4 = self._apply_g(z4, net0)
            if self.use_shift:
                z4 = tr.roll(z4, shifts=(-1, -1), dims=(2, 3))
            z4 = self._apply_g(z4, net1)
            if self.use_shift:
                z4 = tr.roll(z4, shifts=(+1, +1), dims=(2, 3))
        return z4.squeeze(1)

    def f(self, x: tr.Tensor) -> tr.Tensor:
        z4 = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        for k in reversed(range(self.n_steps)):
            net0 = self.nets[2 * k]
            net1 = self.nets[2 * k + 1]
            if self.use_shift:
                z4 = tr.roll(z4, shifts=(-1, -1), dims=(2, 3))
            z4 = self._apply_f(z4, net1)
            if self.use_shift:
                z4 = tr.roll(z4, shifts=(+1, +1), dims=(2, 3))
            z4 = self._apply_f(z4, net0)
        return z4.squeeze(1)


class SokalCoarsePhi4:
    """Coarse theory that uses fine phi4 action via F(pi, Phi)."""

    def __init__(self, fine_theory: s.phi4, pi: tr.Tensor, rg: SokalRG2x2):
        self.fine = fine_theory
        self.pi = pi.detach()
        self.rg = rg
        self.Bs = pi.shape[0]
        self.V = (pi.shape[1] // rg.block, pi.shape[2] // rg.block)
        self.device = pi.device
        self.dtype = pi.dtype

    def action(self, Phi: tr.Tensor) -> tr.Tensor:
        phi = self.rg.refine(Phi, self.pi)
        return self.fine.action(phi)

    def force(self, Phi: tr.Tensor) -> tr.Tensor:
        Phi_req = Phi.detach().requires_grad_(True)
        A = self.action(Phi_req)
        grad = tr.autograd.grad(A.sum(), Phi_req, create_graph=False)[0]
        return -grad

    def refreshP(self) -> tr.Tensor:
        return tr.normal(0.0, 1.0, [self.Bs, self.V[0], self.V[1]],
                         dtype=self.dtype, device=self.device)

    def evolveQ(self, dt: float, P: tr.Tensor, Q: tr.Tensor) -> tr.Tensor:
        return Q + dt * P

    def kinetic(self, P: tr.Tensor) -> tr.Tensor:
        return tr.einsum("bxy,bxy->b", P, P) / 2.0


class MappedSokalCoarsePhi4:
    def __init__(self, fine_theory: s.phi4, pi: tr.Tensor, rg: SokalRG2x2, vmap: nn.Module):
        self.fine = fine_theory
        self.pi = pi.detach()
        self.rg = rg
        self.vmap = vmap
        self.Bs = pi.shape[0]
        self.V = (pi.shape[1] // rg.block, pi.shape[2] // rg.block)
        self.device = pi.device
        self.dtype = pi.dtype

    def action(self, Phi: tr.Tensor) -> tr.Tensor:
        psi = self.rg.refine(Phi, self.pi)
        phi = self.vmap.f(psi)
        return self.fine.action(phi)

    def force(self, Phi: tr.Tensor) -> tr.Tensor:
        Phi_req = Phi.detach().requires_grad_(True)
        A = self.action(Phi_req)
        grad = tr.autograd.grad(A.sum(), Phi_req, create_graph=False)[0]
        return -grad

    def refreshP(self) -> tr.Tensor:
        return tr.normal(0.0, 1.0, [self.Bs, self.V[0], self.V[1]],
                         dtype=self.dtype, device=self.device)

    def evolveQ(self, dt: float, P: tr.Tensor, Q: tr.Tensor) -> tr.Tensor:
        return Q + dt * P

    def kinetic(self, P: tr.Tensor) -> tr.Tensor:
        return tr.einsum("bxy,bxy->b", P, P) / 2.0


def sokal_coarse_hmc_update(
    phi: tr.Tensor,
    fine_theory: s.phi4,
    Nmd: int = 5,
    tau: float = 1.0,
    Nhmc: int = 1,
    block: int = 2,
    integrator: str = "minnorm2",
    verbose: bool = False,
    return_acceptance: bool = False,
):
    """One or more coarse HMC updates with fixed fine fluctuations."""
    rg = SokalRG2x2(block=block)
    Phi, pi = rg.coarsen(phi)
    coarse = SokalCoarsePhi4(fine_theory, pi, rg)

    if integrator == "minnorm2":
        I = i.minnorm2(coarse.force, coarse.evolveQ, Nmd, tau)
    elif integrator == "leapfrog":
        I = i.leapfrog(coarse.force, coarse.evolveQ, Nmd, tau)
    else:
        raise ValueError("Unknown integrator: " + str(integrator))

    hmc = u.hmc(T=coarse, I=I, verbose=verbose)
    Phi_new = hmc.evolve(Phi, Nhmc)
    phi_new = rg.refine(Phi_new, pi)
    if return_acceptance:
        acc = hmc.calc_Acceptance()
        trials = len(hmc.AcceptReject)
        return phi_new, Phi_new, pi, acc, trials
    return phi_new, Phi_new, pi


def sokal_coarse_hmc_update_mapped(
    phi: tr.Tensor,
    fine_theory: s.phi4,
    vmap: nn.Module,
    Nmd: int = 5,
    tau: float = 1.0,
    Nhmc: int = 1,
    block: int = 2,
    integrator: str = "minnorm2",
    verbose: bool = False,
    return_acceptance: bool = False,
):
    rg = SokalRG2x2(block=block)
    psi = vmap.g(phi)
    Phi, pi = rg.coarsen(psi)
    coarse = MappedSokalCoarsePhi4(fine_theory, pi, rg, vmap)

    if integrator == "minnorm2":
        I = i.minnorm2(coarse.force, coarse.evolveQ, Nmd, tau)
    elif integrator == "leapfrog":
        I = i.leapfrog(coarse.force, coarse.evolveQ, Nmd, tau)
    else:
        raise ValueError("Unknown integrator: " + str(integrator))

    hmc = u.hmc(T=coarse, I=I, verbose=verbose)
    Phi_new = hmc.evolve(Phi, Nhmc)
    psi_new = rg.refine(Phi_new, pi)
    phi_new = vmap.f(psi_new)
    if return_acceptance:
        acc = hmc.calc_Acceptance()
        trials = len(hmc.AcceptReject)
        return phi_new, Phi_new, pi, acc, trials
    return phi_new, Phi_new, pi


def _build_fine_theory(L: int, lam: float, mas: float, batch_size: int, device):
    lat = [L, L]
    return s.phi4(lat, lam, mas, batch_size=batch_size, device=device)


def test_basic(
    L: int,
    lam: float,
    mas: float,
    batch_size: int,
    verbose: bool = True,
):
    device = tr.device("cpu")

    fine = _build_fine_theory(L, lam, mas, batch_size, device)
    phi = fine.hotStart()

    rg = SokalRG2x2(block=2)
    Phi, pi = rg.coarsen(phi)
    phi_rec = rg.refine(Phi, pi)

    recon_err = (phi_rec - phi).abs().max().item()
    if verbose:
        print("[basic] Test: coarsen/refine reconstruction")
        print("[basic] max|phi_rec - phi| =", recon_err)
    assert tr.allclose(phi_rec, phi, atol=1e-6, rtol=1e-6)

    coarse = SokalCoarsePhi4(fine, pi, rg)
    A = coarse.action(Phi)
    F = coarse.force(Phi)
    if verbose:
        print("[basic] Test: coarse action/force shapes and finiteness")
        print("[basic] action shape:", tuple(A.shape))
        print("[basic] force shape :", tuple(F.shape))
        print("[basic] force finite:", bool(tr.isfinite(F).all().item()))
    assert A.shape == tr.Size([batch_size])
    assert F.shape == Phi.shape
    assert tr.isfinite(F).all()

    phi_new, Phi_new, pi_new = sokal_coarse_hmc_update(
        phi, fine, Nmd=2, tau=0.5, Nhmc=1, block=2, integrator="minnorm2"
    )
    if verbose:
        print("[basic] Test: coarse HMC preserves fixed fluctuations")
        print("[basic] max|pi_new - pi| =", (pi_new - pi).abs().max().item())
    assert tr.allclose(pi_new, pi, atol=1e-6, rtol=1e-6)

    if verbose:
        print("[basic] OK: reconstruction error <= 1e-6, shapes correct, force finite, pi preserved")


def epsilon2_test(
    L: int,
    lam: float,
    mas: float,
    batch_size: int,
    verbose: bool = True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    device = tr.device("cpu")
    fine = _build_fine_theory(L, lam, mas, batch_size, device)
    phi = fine.hotStart()
    rg = SokalRG2x2(block=2)
    Phi, pi = rg.coarsen(phi)
    coarse = SokalCoarsePhi4(fine, pi, rg)

    U = Phi
    P = coarse.refreshP()
    K = coarse.kinetic(P)
    V = coarse.action(U)
    Hi = K + V
    if verbose:
        print("The total initial energy is: ", Hi.detach().cpu().numpy())

    x = []
    y = []
    y2 = []
    for rk in np.logspace(1, 3, 50):
        k = int(rk)
        dt = 1.0 / k
        l = i.leapfrog(coarse.force, coarse.evolveQ, k, 1.0)
        l2 = i.minnorm2(coarse.force, coarse.evolveQ, k, 1.0)
        PP, QQ = l.integrate(P, U)
        PP2, QQ2 = l2.integrate(P, U)
        Hf = coarse.kinetic(PP) + coarse.action(QQ)
        Hf2 = coarse.kinetic(PP2) + coarse.action(QQ2)
        if verbose:
            print("Using dt= ", dt, "The total final energy is: ", Hf.detach().cpu().numpy())
        DH = tr.abs(Hf - Hi)
        DH2 = tr.abs(Hf2 - Hi)
        x.append(dt**2)
        y.append(DH)
        y2.append(DH2)

    plt.plot(x, y, x, y2)
    plt.xlabel('$\\epsilon^2$')
    plt.ylabel('$\\Delta H$')
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sokal coarse HMC tests for phi^4")
    parser.add_argument(
        "--test",
        choices=["basic", "epsilon2", "all"],
        default="basic",
        help="Which test to run",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce test verbosity")
    parser.add_argument("--L", type=int, default=8, help="Lattice size L (square LxL)")
    parser.add_argument("--lam", type=float, default=0.5, help="Phi^4 coupling lambda")
    parser.add_argument("--mass", type=float, default=-0.2, help="Bare mass m")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    args = parser.parse_args()

    verbose = not args.quiet
    args = parser.parse_args()

    verbose = not args.quiet

    if args.test in ("basic", "all"):
        test_basic(
            L=args.L,
            lam=args.lam,
            mas=args.mass,
            batch_size=args.batch_size,
            verbose=verbose,
        )
    if args.test in ("epsilon2", "all"):
        epsilon2_test(
            L=args.L,
            lam=args.lam,
            mas=args.mass,
            batch_size=args.batch_size,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
