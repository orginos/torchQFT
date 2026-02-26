#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilevel MG coarse update utilities for phi^4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch as tr
import torch.nn as nn

import integrators as i
import update as u
import phi4 as s
from phi4_sokal_mg import SokalRG2x2, IdentityMap


@dataclass
class Pyramid:
    phis: List[tr.Tensor]
    psis: List[tr.Tensor]
    Phis: List[tr.Tensor]
    pis: List[tr.Tensor]


def build_pyramid(phi: tr.Tensor, vmaps: List[nn.Module], block: int) -> Pyramid:
    rg = SokalRG2x2(block=block)
    phis = [phi]
    psis = []
    Phis = []
    pis = []
    cur = phi
    for vmap in vmaps:
        psi = vmap.g(cur)
        Phi, pi = rg.coarsen(psi)
        psis.append(psi)
        Phis.append(Phi)
        pis.append(pi)
        cur = Phi
        phis.append(cur)
    return Pyramid(phis=phis, psis=psis, Phis=Phis, pis=pis)


def reconstruct_from_coarse(Phi_coarse: tr.Tensor, pis: List[tr.Tensor], vmaps: List[nn.Module], block: int) -> tr.Tensor:
    rg = SokalRG2x2(block=block)
    cur = Phi_coarse
    for level in reversed(range(len(vmaps))):
        psi = rg.refine(cur, pis[level])
        cur = vmaps[level].f(psi)
    return cur


def reconstruct_from_level(Phi_level: tr.Tensor, pis: List[tr.Tensor], vmaps: List[nn.Module], block: int) -> tr.Tensor:
    rg = SokalRG2x2(block=block)
    cur = Phi_level
    for level in reversed(range(len(vmaps))):
        psi = rg.refine(cur, pis[level])
        cur = vmaps[level].f(psi)
    return cur


class MultiLevelCoarsePhi4:
    def __init__(self, fine_theory: s.phi4, pis: List[tr.Tensor], vmaps: List[nn.Module], block: int):
        self.fine = fine_theory
        self.pis = [p.detach() for p in pis]
        self.vmaps = vmaps
        self.block = block
        self.Bs = pis[-1].shape[0]
        self.V = (pis[-1].shape[1] // block, pis[-1].shape[2] // block)
        self.device = pis[-1].device
        self.dtype = pis[-1].dtype

    def action(self, Phi_coarse: tr.Tensor) -> tr.Tensor:
        phi = reconstruct_from_coarse(Phi_coarse, self.pis, self.vmaps, self.block)
        return self.fine.action(phi)

    def force(self, Phi_coarse: tr.Tensor) -> tr.Tensor:
        Phi_req = Phi_coarse.detach().requires_grad_(True)
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


class LevelCoarsePhi4:
    def __init__(self, fine_theory: s.phi4, pis: List[tr.Tensor], vmaps: List[nn.Module], block: int):
        self.fine = fine_theory
        self.pis = [p.detach() for p in pis]
        self.vmaps = vmaps
        self.block = block
        self.Bs = pis[-1].shape[0]
        self.V = (pis[-1].shape[1] // block, pis[-1].shape[2] // block)
        self.device = pis[-1].device
        self.dtype = pis[-1].dtype

    def action(self, Phi_level: tr.Tensor) -> tr.Tensor:
        phi = reconstruct_from_level(Phi_level, self.pis, self.vmaps, self.block)
        return self.fine.action(phi)

    def force(self, Phi_level: tr.Tensor) -> tr.Tensor:
        Phi_req = Phi_level.detach().requires_grad_(True)
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


def multilevel_coarse_hmc_update(
    phi: tr.Tensor,
    fine_theory: s.phi4,
    vmaps: List[nn.Module],
    block: int,
    Nmd: int,
    tau: float,
    Nhmc: int,
    integrator: str = "minnorm2",
    verbose: bool = False,
    return_acceptance: bool = False,
):
    pyr = build_pyramid(phi, vmaps, block)
    Phi_coarse = pyr.Phis[-1]
    coarse = MultiLevelCoarsePhi4(fine_theory, pyr.pis, vmaps, block)

    if integrator == "minnorm2":
        I = i.minnorm2(coarse.force, coarse.evolveQ, Nmd, tau)
    elif integrator == "leapfrog":
        I = i.leapfrog(coarse.force, coarse.evolveQ, Nmd, tau)
    else:
        raise ValueError("Unknown integrator: " + str(integrator))

    hmc = u.hmc(T=coarse, I=I, verbose=verbose)
    Phi_new = hmc.evolve(Phi_coarse, Nhmc)
    phi_new = reconstruct_from_coarse(Phi_new, pyr.pis, vmaps, block)

    if return_acceptance:
        acc = hmc.calc_Acceptance()
        trials = len(hmc.AcceptReject)
        return phi_new, acc, trials
    return phi_new


def multilevel_vcycle_hmc_update(
    phi: tr.Tensor,
    fine_theory: s.phi4,
    vmaps: List[nn.Module],
    block: int,
    Nmd,
    tau,
    Nhmc: int,
    integrator: str = "minnorm2",
    verbose: bool = False,
    return_acceptance: bool = False,
):
    rg = SokalRG2x2(block=block)
    levels = len(vmaps)
    pis: List[tr.Tensor] = []
    Phis: List[tr.Tensor] = [phi]
    psi_list: List[tr.Tensor] = []
    acc_down = [0.0] * levels
    trials_down = [0] * levels
    acc_up = [0.0] * levels
    trials_up = [0] * levels

    # Down sweep: update Phi_{l+1} at each level l
    cur = phi
    for l in range(levels):
        psi = vmaps[l].g(cur)
        psi_list.append(psi)
        Phi_next, pi = rg.coarsen(psi)
        pis.append(pi)

        # update Phi_next using maps[0..l] and pis[0..l]
        level_theory = LevelCoarsePhi4(fine_theory, pis[: l + 1], vmaps[: l + 1], block)
        Nmd_l = Nmd[l] if isinstance(Nmd, list) else Nmd
        tau_l = tau[l] if isinstance(tau, list) else tau
        if integrator == "minnorm2":
            I = i.minnorm2(level_theory.force, level_theory.evolveQ, Nmd_l, tau_l)
        elif integrator == "leapfrog":
            I = i.leapfrog(level_theory.force, level_theory.evolveQ, Nmd_l, tau_l)
        else:
            raise ValueError("Unknown integrator: " + str(integrator))
        hmc = u.hmc(T=level_theory, I=I, verbose=verbose)
        Phi_next = hmc.evolve(Phi_next, Nhmc)
        acc_down[l] = hmc.calc_Acceptance()
        trials_down[l] = len(hmc.AcceptReject)

        Phis.append(Phi_next)
        cur = Phi_next

    # Up sweep: refine and update levels from l=levels-1 down to 1
    for l in reversed(range(1, levels)):
        psi = rg.refine(Phis[l + 1], pis[l])
        Phis[l] = vmaps[l].f(psi)

        level_theory = LevelCoarsePhi4(fine_theory, pis[:l], vmaps[:l], block)
        Nmd_l = Nmd[l-1] if isinstance(Nmd, list) else Nmd
        tau_l = tau[l-1] if isinstance(tau, list) else tau
        if integrator == "minnorm2":
            I = i.minnorm2(level_theory.force, level_theory.evolveQ, Nmd_l, tau_l)
        elif integrator == "leapfrog":
            I = i.leapfrog(level_theory.force, level_theory.evolveQ, Nmd_l, tau_l)
        else:
            raise ValueError("Unknown integrator: " + str(integrator))
        hmc = u.hmc(T=level_theory, I=I, verbose=verbose)
        Phis[l] = hmc.evolve(Phis[l], Nhmc)
        acc_up[l] = hmc.calc_Acceptance()
        trials_up[l] = len(hmc.AcceptReject)

    # Refine to fine field
    psi0 = rg.refine(Phis[1], pis[0])
    phi_new = vmaps[0].f(psi0)

    if return_acceptance:
        return phi_new, acc_down, trials_down, acc_up, trials_up
    return phi_new
