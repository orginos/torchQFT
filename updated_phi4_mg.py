#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multigrid normalizing flow for scalar phi^4, with parity-aware S/T networks.

Features:
- RealNVP has no parity logic; it asks an STProvider for s,t.
- STProvider supports parity modes: 'none', 'sym' (explicit symmetrization), 'x2' (single-pass).
- Correct log-det: sum only over transformed dims ((1-mask) * s).
- 2x2 block bijector (no unfold/fold).
- Invertible RG (average or select).
- Optional *fixed bijector* across scales/layers.
- Upscaling utility to a larger lattice.
"""

from __future__ import annotations
import math
import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional

def _std_normal_log_prob(z: tr.Tensor, sum_dims) -> tr.Tensor:
    return (-0.5 * (z*z + math.log(2*math.pi))).sum(dim=sum_dims)

# ----------------------- Parity-aware S/T provider -----------------------

class STProvider(nn.Module):
    """
    Provides s(x_A), t(x_A) with desired Z2 parity for each RealNVP block.

    Modes:
      - 'none':    standard: s,t = MLP(x_A)
      - 'sym':     explicit symmetrization: s_even = (f(x_A)+f(-x_A))/2,
                                          t_odd  = (g(x_A)-g(-x_A))/2
      - 'x2':      single-pass: h(x_A^2) -> [s_even, t_even]; t_hat = t_even * phi(x_A),
                   where phi(x_A) = sum(x_A) / sqrt(1 + sum(x_A^2)) is odd scalar.
    """
    def __init__(self, D: int, n_blocks: int, width: int, mode: str = "none"):
        super().__init__()
        assert mode in ("none", "sym", "x2")
        self.mode = mode
        self.D = int(D)
        self.n_blocks = int(n_blocks)

        if mode in ("none", "sym", "x2"):
            # one MLP per block, outputs 2D (concat of s and t)
            self.st_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(D, width), nn.SiLU(),
                    nn.Linear(width, width), nn.SiLU(),
                    nn.Linear(width, 2*D),
                ) for _ in range(n_blocks)
            ])
            # initializes to the identity map. The resulting RealNVP is the identity
            for net in self.st_nets:
                nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)

    def st(self, i: int, xA: tr.Tensor) -> Tuple[tr.Tensor, tr.Tensor]:
        """
        Return s,t tensors of shape [N, D] given the frozen-part input xA (masked input).
        """
        if self.mode == "none":
            st = self.st_nets[i](xA)
            s, t = tr.chunk(st, 2, dim=-1)
            return s, t

        if self.mode == "sym":
            net = self.st_nets[i]
            st_pos = net(xA)
            st_neg = net(-xA)
            D = xA.shape[-1]
            s = 0.5*(st_pos[..., :D] + st_neg[..., :D])     # even in xA
            t = 0.5*(st_pos[..., D:] - st_neg[..., D:])     # odd in xA
            return s, t

        # mode == "x2": single pass on xA^2 -> [s_even, t_even]; t_hat = t_even * phi(xA)
        net = self.st_nets[i]
        xA2 = xA * xA
        st = net(xA2)
        s_even, t_even = tr.chunk(st, 2, dim=-1)
        sum_xA = xA.sum(dim=1, keepdim=True)                       # [N,1]
        norm_xA = tr.sqrt(1.0 + (xA*xA).sum(dim=1, keepdim=True)) # [N,1]
        phi = sum_xA / norm_xA                                    # odd scalar
        t_hat = t_even * phi                                      # odd vector
        return s_even, t_hat

# ----------------------------- RealNVP -----------------------------

class RealNVP(nn.Module):
    """
    Affine coupling with binary masks. s,t come from an STProvider (parity-agnostic).
    """
    def __init__(self, st_provider: STProvider, masks: tr.Tensor,
                 data_dims=(1,), log_scale_clip: float = 5.0):
        super().__init__()
        assert masks.ndim == 2 and masks.shape[1] > 0, "masks [n_blocks, D]"
        self.st_provider = st_provider
        self.register_buffer("masks", masks.float())
        self.register_buffer("masks_inv", 1.0 - masks.float())
        self.data_dims = (1,)  # feature dim for (N, D)
        self.log_scale_clip = float(log_scale_clip)

    def g(self, z: tr.Tensor) -> tr.Tensor:
        x = z
        for i in range(self.masks.shape[0]):
            m  = self.masks[i]
            mi = self.masks_inv[i]
            xA = x * m
            s, t = self.st_provider.st(i, xA)
            s = tr.tanh(s) * self.log_scale_clip
            # apply only on transformed coords
            s = s * mi
            t = t * mi
            x = xA + mi * (x * tr.exp(s) + t)
        return x

    def f(self, x: tr.Tensor):
        log_det_J = x.new_zeros(x.shape[0])
        z = x
        for i in reversed(range(self.masks.shape[0])):
            m  = self.masks[i]
            mi = self.masks_inv[i]
            zA = z * m
            s, t = self.st_provider.st(i, zA)
            s = tr.tanh(s) * self.log_scale_clip
            s = s * mi
            t = t * mi
            z = mi * (z - t) * tr.exp(-s) + zA
            # sum ONLY over transformed dims
            log_det_J -= (mi * s).sum(dim=self.data_dims)
        return z, log_det_J

    def forward(self, z):  # for completeness
        return self.g(z)

# ------------------------ 2x2 conv-flow layer ------------------------

class ConvFlowLayer(nn.Module):
    def __init__(self, size: Tuple[int,int], bijector_factory: Callable[[], nn.Module],
                 Nsteps: int = 1, fixed_bijector: bool = False, shared_bijector: Optional[nn.Module] = None):
        super().__init__()
        self.Nsteps = int(Nsteps)
        self.size = size
        self.fixed_bijector = fixed_bijector
        if fixed_bijector:
            self.shared_bj = bijector_factory() if shared_bijector is None else shared_bijector
            self.bj = None
        else:
            self.bj = nn.ModuleList([bijector_factory() for _ in range(2*self.Nsteps)])
            self.shared_bj = None

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
        out = y.new_zeros((B, 1, H2*2, W2*2))
        out[:, :, 0::2, 0::2] = y[:, 0:1]
        out[:, :, 0::2, 1::2] = y[:, 1:2]
        out[:, :, 1::2, 0::2] = y[:, 2:3]
        out[:, :, 1::2, 1::2] = y[:, 3:4]
        return out

    def _apply_g(self, z4: tr.Tensor, bij: nn.Module) -> tr.Tensor:
        y = self._pack2x2(z4)
        B, C, H2, W2 = y.shape
        flat = y.permute(0,2,3,1).contiguous().view(B*H2*W2, C)
        flat = bij.g(flat)
        y = flat.view(B, H2, W2, C).permute(0,3,1,2).contiguous()
        return self._unpack2x2(y)

    def _apply_f(self, x4: tr.Tensor, bij: nn.Module):
        y = self._pack2x2(x4)
        B, C, H2, W2 = y.shape
        flat = y.permute(0,2,3,1).contiguous().view(B*H2*W2, C)
        zf, ld = bij.f(flat)
        y = zf.view(B, H2, W2, C).permute(0,3,1,2).contiguous()
        out = self._unpack2x2(y)
        ld_per_sample = ld.view(B, H2, W2).sum(dim=(1,2))
        return out, ld_per_sample

    def g(self, z: tr.Tensor) -> tr.Tensor:
        z4 = z.view(z.shape[0], 1, z.shape[1], z.shape[2])
        for k in range(self.Nsteps):
            bj0 = self.shared_bj if self.fixed_bijector else self.bj[2*k]
            bj1 = self.shared_bj if self.fixed_bijector else self.bj[2*k+1]
            z4 = self._apply_g(z4, bj0)
            z4 = tr.roll(z4, shifts=(-1, -1), dims=(2, 3))
            z4 = self._apply_g(z4, bj1)
            z4 = tr.roll(z4, shifts=(+1, +1), dims=(2, 3))
        return z4.squeeze(1)

    def f(self, x: tr.Tensor):
        log_det_J = x.new_zeros(x.shape[0])
        z4 = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        for k in reversed(range(self.Nsteps)):
            bj0 = self.shared_bj if self.fixed_bijector else self.bj[2*k]
            bj1 = self.shared_bj if self.fixed_bijector else self.bj[2*k+1]
            z4 = tr.roll(z4, shifts=(-1, -1), dims=(2, 3))
            z4, ld = self._apply_f(z4, bj1)
            log_det_J += ld
            z4 = tr.roll(z4, shifts=(+1, +1), dims=(2, 3))
            z4, ld = self._apply_f(z4, bj0)
            log_det_J += ld
        return z4.squeeze(1), log_det_J

# ----------------------------- RG layer -----------------------------

class RGlayer(nn.Module):
    def __init__(self, transformation_type: str = "select"):
        super().__init__()
        self.type = transformation_type

    def coarsen(self, f: tr.Tensor):
        ff = f.unsqueeze(1)
        if self.type == "average":
            c = F.avg_pool2d(ff, kernel_size=2, stride=2)
            r = ff - F.interpolate(c, scale_factor=2, mode="nearest")
        else:  # "select"
            c = ff[..., ::2, ::2]
            r = ff - F.interpolate(c, scale_factor=2, mode="nearest")
        return c.squeeze(1), r.squeeze(1)

    def refine(self, c: tr.Tensor, r: tr.Tensor) -> tr.Tensor:
        up = F.interpolate(c.unsqueeze(1), scale_factor=2, mode="nearest").squeeze(1)
        return up + r

# ----------------------------- MG flow -----------------------------

class MGflow(nn.Module):
    def __init__(self, size: Tuple[int,int], bijector_factory: Callable[[], nn.Module],
                 rg: RGlayer, Nconvs: int = 1, fixed_bijector: bool = False):
        super().__init__()
        H, W = size
        assert H == W and (H & (H-1)) == 0, "size must be square, power of 2"
        self.size = size
        self.rg = rg
        self.depth = int(math.log2(H))
        self.fixed_bijector = fixed_bijector
        shared = bijector_factory() if fixed_bijector else None
        sizes = [(H // (2**k), W // (2**k)) for k in range(self.depth)]
        self.cflow = nn.ModuleList([
            ConvFlowLayer(s, bijector_factory, Nconvs, fixed_bijector=fixed_bijector, shared_bijector=shared)
            for s in sizes
        ])

    def prior_sample(self, batch_size: int, device=None, dtype=None) -> tr.Tensor:
        p = next(self.parameters(), None)
        if device is None: device = (p.device if p is not None else tr.device("cpu"))
        if dtype is None: dtype = (p.dtype if p is not None else tr.float32)
        H, W = self.size
        return tr.randn((batch_size, H, W), device=device, dtype=dtype)

    def prior_log_prob(self, z: tr.Tensor) -> tr.Tensor:
        return _std_normal_log_prob(z, sum_dims=(1,2))

    def g(self, z: tr.Tensor) -> tr.Tensor:
        x = z
        fines = []
        for k in range(self.depth-1):
            fx = self.cflow[k].g(x)
            cx, ff = self.rg.coarsen(fx)
            fines.append(ff)
            x = cx
        fx = self.cflow[self.depth-1].g(x)
        for k in range(self.depth-1, 0, -1):
            fx = self.rg.refine(fx, fines[k-1])
        return fx

    def f(self, x: tr.Tensor):
        """
        Inverse of g:
          forward g:  x_k --(flow_k.g)--> y_k --(coarsen)--> (c_{k+1}, f_k),  set x_{k+1}=c_{k+1}
          last: x_{K} --(flow_K.g)--> y_K, then refine back with {f_k}
        inverse f must: coarsen only down to x_K, invert flow_K, then for k=K-1..0:
          y_k = refine(z_{k+1}, f_k);  z_k, ld = flow_k.f(y_k)
        """
        log_det_J = x.new_zeros(x.shape[0])

        # --- down: coarsen only, stash fines
        fines = []
        z = x
        for _k in range(self.depth - 1):
            c, f = self.rg.coarsen(z)
            fines.append(f)
            z = c

        # --- coarsest inverse
        z, Jk = self.cflow[self.depth - 1].f(z)
        log_det_J += Jk

        # --- up: refine + invert flows in reverse order
        for k in range(self.depth - 2, -1, -1):
            y = self.rg.refine(z, fines[k])
            y, Jk = self.cflow[k].f(y)
            log_det_J += Jk
            z = y

        return z, log_det_J

    def log_prob(self, x: tr.Tensor) -> tr.Tensor:
        z, ldj = self.f(x)
        return self.prior_log_prob(z) + ldj

    def sample(self, batch_size: int) -> tr.Tensor:
        return self.g(self.prior_sample(batch_size))

# ---------------------- Bijector factory (2x2) ----------------------

def FlowBijector(n_layers: int = 3, width: int = 256, log_scale_clip: float = 5.0,
                 parity: str = "none") -> Callable[[], nn.Module]:
    """
    Factory that builds a 4D RealNVP with checkerboard masks and an STProvider
    using the requested parity mode.
    """
    mm = np.array([1,0,0,1], dtype=np.float32)
    D = int(mm.size)
    masks = tr.from_numpy(np.array([mm, 1-mm] * n_layers).astype(np.float32))  # [2n, D]

    def make_bij():
        stp = STProvider(D=D, n_blocks=masks.shape[0], width=width, mode=parity)
        return RealNVP(st_provider=stp, masks=masks, data_dims=(1,), log_scale_clip=log_scale_clip)

    return make_bij

# ----------------------- Upscaling utility -----------------------

def upscale_mgflow(old: MGflow, new_size: Tuple[int,int], bijector_factory: Callable[[], nn.Module], rg: RGlayer,
                   Nconvs: int = 1, fixed_bijector: bool = False) -> MGflow:
    device = next(old.parameters()).device
    new = MGflow(new_size, bijector_factory, rg, Nconvs=Nconvs, fixed_bijector=fixed_bijector).to(device)
    old_depth = old.depth
    new_depth = new.depth
    if new_depth < old_depth:
        raise ValueError("new lattice must not be smaller than old lattice")
    shift = new_depth - old_depth
    with tr.no_grad():
        for k in range(old_depth):
            dst = new.cflow[k+shift]
            src = old.cflow[k]
            if fixed_bijector:
                dst.shared_bj.load_state_dict(src.shared_bj.state_dict())
            else:
                for i in range(len(src.bj)):
                    dst.bj[i].load_state_dict(src.bj[i].state_dict())
    return new
