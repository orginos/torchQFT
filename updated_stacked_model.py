#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacked multigrid flow with stage-wise training, gradient accumulation,
CUDA-only AMP, optional torch.compile, tqdm progress, NaN-guard, and diagnostics.
"""

from __future__ import annotations
from contextlib import nullcontext
from typing import List, Optional, Callable
import torch as tr
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm.auto import tqdm

from updated_phi4_mg import MGflow

def set_requires_grad(mod: nn.Module, flag: bool):
    for p in mod.parameters():
        p.requires_grad = flag

def device_of(mod: nn.Module):
    p = next(mod.parameters(), None)
    return p.device if p is not None else tr.device("cpu")

def autocast_ctx(use_amp: bool, device: tr.device):
    if not use_amp or device.type != 'cuda':
        return nullcontext()
    return tr.amp.autocast(device_type='cuda')

def make_scaler(use_amp: bool, device: tr.device):
    return tr.amp.GradScaler('cuda', enabled=(use_amp and device.type == 'cuda'))

class StackedModel(nn.Module):
    def __init__(self, stages: List[MGflow], action_fn: Callable[[tr.Tensor], tr.Tensor]):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.action_fn = action_fn
        self.size = stages[0].size

    def prior_sample(self, batch_size: int, device=None, dtype=None):
        return self.stages[0].prior_sample(batch_size, device=device, dtype=dtype)

    def prior_log_prob(self, z: tr.Tensor):
        return self.stages[0].prior_log_prob(z)

    def g(self, z: tr.Tensor):
        x = z
        for st in self.stages:
            x = st.g(x)
        return x

    def f(self, x: tr.Tensor):
        z = x
        ldj = x.new_zeros(x.shape[0])
        for st in reversed(self.stages):
            z, j = st.f(z)
            ldj += j
        return z, ldj

    def log_prob(self, x: tr.Tensor):
        z, j = self.f(x)
        return self.prior_log_prob(z) + j

    def prefix_g(self, z: tr.Tensor, upto: int) -> tr.Tensor:
        x = z
        for k in range(upto):
            x = self.stages[k].g(x)
        return x

    def diff(self, x: tr.Tensor) -> tr.Tensor:
        return self.log_prob(x) + self.action_fn(x)

    def diff_full_from(self, x: tr.Tensor, from_stage: int) -> tr.Tensor:
        xt = x
        for k in range(from_stage, len(self.stages)):
            xt = self.stages[k].g(xt)
        return self.diff(xt)

def _sample_prefix(model: StackedModel, big: int, upto: int, device) -> tr.Tensor:
    with tr.no_grad():
        z = model.prior_sample(big, device=device)
        return z if upto == 0 else model.prefix_g(z, upto=upto)

def trainSM(
    model,
    epochs: int,
    batch_size: int,
    super_batch_size: int = 1,
    lr: float = 1e-4,
    warmup_per_stage: int = 50,
    joint_per_stage: int = 50,
    use_amp: bool = False,
    compile_model: bool = False,  # kept for API compatibility; unused here
    grad_clip: float | None = None,
    optimizer: "tr.optim.Optimizer" | None = None,
    scaler: "tr.cuda.amp.GradScaler | tr.amp.GradScaler | None" = None,
    tqdm_desc: str = "Training (Stacked MG)",
):
    """
    Train a StackedModel with per-stage warmup then joint tuning.

    - Warmup: for each stage s, only that stage's params require grad; others are frozen.
    - Joint: enable all params and train together.
    - Optimizer/scaler: if not provided, create AdamW + (optional) CUDA AMP scaler.

    Returns:
      history: list[float] of mean-ΔS losses for each outer iteration
               (length = stages * warmup_per_stage + joint_per_stage)
    """
    device = next(model.parameters()).device
    # Lazy optimizer/scaler
    if optimizer is None:
        optimizer = tr.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    if scaler is None:
        # torch>=2.1 style; safe on older too. AMP only useful on CUDA.
        scaler = tr.amp.GradScaler('cuda', enabled=(use_amp and device.type == 'cuda'))

    def set_all_requires_grad(flag: bool):
        for p in model.parameters():
            p.requires_grad = flag

    def set_only_stage_trainable(stage_idx: int):
        set_all_requires_grad(False)
        for p in model.stages[stage_idx].parameters():
            p.requires_grad = True

    def train_outer(iters: int, phase_name: str):
        hist = []
        pbar = tqdm(range(iters), desc=f"{tqdm_desc}: {phase_name}")
        for _ in pbar:
            # one optimizer step made from `super_batch_size` microbatches
            optimizer.zero_grad(set_to_none=True)
            loss_meter = 0.0
            for __ in range(max(1, super_batch_size)):
                z = model.prior_sample(batch_size, device=device)
                ctx = tr.amp.autocast(device_type='cuda') if (use_amp and device.type == 'cuda') else contextlib.nullcontext()
                with ctx:
                    x = model.g(z)
                    # ΔS = log_prob(x) + S(x); model.action_fn was set by the caller
                    deltaS = model.log_prob(x) + model.action_fn(x)
                    loss = deltaS.mean()
                    micro = loss / max(1, super_batch_size)
                if not tr.isfinite(micro):
                    continue
                if scaler.is_enabled():
                    scaler.scale(micro).backward()
                else:
                    micro.backward()
                loss_meter += float(loss.detach())
            # step
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                if grad_clip is not None:
                    tr.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                if grad_clip is not None:
                    tr.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            hist.append(loss_meter / max(1, super_batch_size))
            pbar.set_postfix(loss=f"{hist[-1]:.4f}")
        return hist

    import contextlib  # used above
    history = []

    # -------- Warmup per stage --------
    S = len(model.stages)
    for s in range(S):
        set_only_stage_trainable(s)
        history += train_outer(warmup_per_stage, phase_name=f"warmup s={s}")

    # -------- Joint tuning --------
    set_all_requires_grad(True)
    history += train_outer(joint_per_stage, phase_name="joint")

    return history



from torch.optim.swa_utils import AveragedModel, SWALR
import contextlib as _ctx

def trainSMaveraged(
    SuperM,
    epochs: int = 100,
    batch_size: int = 16,
    super_batch_size: int = 1,
    lr: float = 1e-4,
    swa_start_frac: float = 0.7,
    swa_lr_mult: float = 0.1,
    use_amp: bool = True,
    grad_clip: float | None = 1.0,
    optimizer: "tr.optim.Optimizer" | None = None,
    scaler: "tr.amp.GradScaler | None" = None,
    tqdm_desc: str = "Training(SWA)",
):
    """
    Stochastic Weight Averaging (SWA) on a StackedModel.

    - Optimizes the same ΔS objective as trainSM:  ΔS = log_prob(x) + action(x)
    - Gradient accumulation via super_batch_size
    - Optional AMP on CUDA
    - SWA starts at epoch floor(swa_start_frac * epochs); before then, trains normally
    - Returns: (history, swa_state_dict) so caller can load averaged weights

    Notes:
      * We don't use BatchNorm in these flows, so no BN update pass is needed.
    """
    device = next(SuperM.parameters()).device
    use_amp = bool(use_amp and device.type == 'cuda')

    if optimizer is None:
        optimizer = tr.optim.AdamW(SuperM.parameters(), lr=lr, betas=(0.9, 0.99))
    if scaler is None:
        scaler = tr.amp.GradScaler('cuda', enabled=use_amp)

    swa_model = AveragedModel(SuperM)
    swa_start = int(max(0, round(swa_start_frac * epochs)))
    swa_sched = SWALR(optimizer, swa_lr=max(1e-12, lr * float(swa_lr_mult)))

    history: list[float] = []
    pbar = tqdm(range(epochs), desc=tqdm_desc)
    for ep in pbar:
        SuperM.train()
        accum = max(1, int(super_batch_size))
        optimizer.zero_grad(set_to_none=True)
        loss_meter = 0.0

        for _ in range(accum):
            z = SuperM.prior_sample(batch_size, device=device)
            ctx = tr.amp.autocast(device_type='cuda') if use_amp else _ctx.nullcontext()
            with ctx:
                x = SuperM.g(z)
                deltaS = SuperM.log_prob(x) + SuperM.action_fn(x)  # ΔS objective
                loss = deltaS.mean()
                micro = loss / accum
            if not tr.isfinite(micro):
                continue
            if scaler.is_enabled():
                scaler.scale(micro).backward()
            else:
                micro.backward()
            loss_meter += float(loss.detach())

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if grad_clip is not None:
            tr.nn.utils.clip_grad_norm_(SuperM.parameters(), grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # SWA bookkeeping after optimizer step
        if ep >= swa_start:
            swa_model.update_parameters(SuperM)
            swa_sched.step()

        history.append(loss_meter / accum)
        pbar.set_postfix(loss=f"{history[-1]:.4f}")

    # Return the averaged weights so the caller can load them into SuperM
    return history, swa_model.module.state_dict()


