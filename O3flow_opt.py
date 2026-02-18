#!/usr/bin/env python3
"""
Second-generation optimized flow actions for O(3).

Key ideas:
  - Cache rolls/dot products of the spin field so Psi kernels reuse them.
  - Compute all Psi actions/grads/lapls/grad_lapls once per call, then form
    SflowO1/SflowO2 directly as linear/polynomial combinations of those values.
  - Keep formulas identical to O3flow.py for correctness; avoid duplicate work.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable
from contextlib import nullcontext

import torch as tr

import O3flow as base
from O3flow import (
    Psi0,
    Psi11,
    Psi11_l,
    Psi111,
    Psi111c,
    Psi111l,
    Psi12d,
    Psi12l,
    Psi1l1,
    Psi2,
    Psi21,
    Psi3,
    SflowO0 as BaseO0,
    SflowO1 as BaseO1,
    SflowO2 as BaseO2,
)
try:
    import o3psi_native as native_mod
    from o3psi_native import (
        HAS_NATIVE,
        Psi0Native,
        Psi2Native,
        Psi11Native,
        Psi11lNative,
    Psi3Native,
    Psi21Native,
        Psi12dNative,
        Psi12lNative,
        Psi111Native,
        Psi111cNative,
        Psi1l1Native,
    Psi111lNative,
    sflowo2_grad,
    sflowo2_action,
    sflowo2_lapl,
    sflowo2_grad_lapl,
    sflowo1_action,
    sflowo1_grad,
    sflowo1_lapl,
    sflowo1_grad_lapl,
)
except Exception:
    HAS_NATIVE = False
    native_mod = None
    Psi0Native = None
    Psi2Native = None
    Psi11Native = None
    Psi11lNative = None
    Psi3Native = None
    Psi21Native = None
    Psi12dNative = None
    Psi12lNative = None
    Psi111Native = None
    Psi111cNative = None
    Psi1l1Native = None
    Psi111lNative = None
    sflowo2_action = None
    sflowo2_grad = None
    sflowo2_lapl = None
    sflowo2_grad_lapl = None
    sflowo1_action = None
    sflowo1_grad = None
    sflowo1_lapl = None
    sflowo1_grad_lapl = None

USE_NATIVE = HAS_NATIVE and os.environ.get("O3FLOW_OPT2_FORCE_PY", "").lower() not in {
    "1",
    "true",
    "yes",
    "on",
}


class _NeighborCache:
    """Cache roll(s,dx,dy) and dot3(s, roll(s,dx,dy)) for a given input s."""

    def __init__(self, s: tr.Tensor):
        self.s = s
        self.roll_cache: Dict[tuple[int, int], tr.Tensor] = {}
        self.shift_by_ptr: Dict[int, tuple[int, int]] = {}
        self.dot_cache: Dict[tuple[str, tuple[int, int]], tr.Tensor] = {}
        self.orig_roll = base.roll
        self.orig_dot3 = base.dot3

    def roll(self, t: tr.Tensor, dx: int, dy: int) -> tr.Tensor:
        if t.data_ptr() == self.s.data_ptr():
            key = (dx, dy)
            if key not in self.roll_cache:
                r = self.orig_roll(t, dx, dy)
                self.roll_cache[key] = r
                self.shift_by_ptr[r.data_ptr()] = key
            return self.roll_cache[key]
        return self.orig_roll(t, dx, dy)

    def dot3(self, a: tr.Tensor, b: tr.Tensor) -> tr.Tensor:
        key = None
        if a.data_ptr() == self.s.data_ptr():
            shift = self.shift_by_ptr.get(b.data_ptr())
            if shift is not None:
                key = ("s", shift)
        elif b.data_ptr() == self.s.data_ptr():
            shift = self.shift_by_ptr.get(a.data_ptr())
            if shift is not None:
                key = ("s", shift)

        if key is not None:
            if key not in self.dot_cache:
                self.dot_cache[key] = self.orig_dot3(a, b)
            return self.dot_cache[key]

        return self.orig_dot3(a, b)

    def __enter__(self):
        self.prev_roll = base.roll
        self.prev_dot3 = base.dot3
        base.roll = self.roll
        base.dot3 = self.dot3
        return self

    def __exit__(self, exc_type, exc, tb):
        base.roll = self.prev_roll
        base.dot3 = self.prev_dot3
        return False


def _compute(
    psis: Dict[str, base.Functional],
    s: tr.Tensor,
    need_a: bool = False,
    need_g: bool = False,
    need_lapl: bool = False,
    need_gl: bool = False,
):
    """Compute only requested quantities for given psis."""
    vals = {}
    for name, p in psis.items():
        entry = {}
        if need_a:
            entry["a"] = p.action(s)
        if need_g:
            entry["g"] = p.grad(s)
        if need_lapl:
            entry["lapl"] = p.lapl(s)
        if need_gl:
            entry["gl"] = p.grad_lapl(s)
        vals[name] = entry
    return vals


def _psi_factory(name: str, use_native: bool | None = None):
    native_map = {
        "psi0": Psi0Native,
        "psi2": Psi2Native,
        "psi11": Psi11Native,
        "psi11l": Psi11lNative,
        "psi3": Psi3Native,
        "psi21": Psi21Native,
        "psi12d": Psi12dNative,
        "psi12l": Psi12lNative,
        "psi111": Psi111Native,
        "psi111c": Psi111cNative,
        "psi1l1": Psi1l1Native,
        "psi111l": Psi111lNative,
    }
    flag = USE_NATIVE if use_native is None else (HAS_NATIVE and use_native)
    if flag:
        cls = native_map.get(name)
        if cls is not None:
            return cls()
    cls_map = {
        "psi0": Psi0,
        "psi2": Psi2,
        "psi11": Psi11,
        "psi11l": Psi11_l,
        "psi3": Psi3,
        "psi21": Psi21,
        "psi12d": Psi12d,
        "psi111": Psi111,
        "psi111c": Psi111c,
        "psi12l": Psi12l,
        "psi1l1": Psi1l1,
        "psi111l": Psi111l,
    }
    return cls_map[name]()


@dataclass
class SflowO0:
    beta: float

    def __post_init__(self):
        self.c0 = self.beta / 8.0
        self.psi0 = Psi0()

    def __call__(self, s: tr.Tensor, t: float) -> tr.Tensor:
        return self.c0 * self.psi0.action(s)

    def grad(self, s: tr.Tensor, t: float) -> tr.Tensor:
        return self.c0 * self.psi0.grad(s)

    def mgrad(self, s: tr.Tensor, t: float) -> tr.Tensor:
        return -self.grad(s, t)

    def mlapl(self, s: tr.Tensor, t: float) -> tr.Tensor:
        return -self.c0 * self.psi0.lapl(s)

    def grad_lapl(self, s: tr.Tensor, t: float) -> tr.Tensor:
        return self.c0 * self.psi0.grad_lapl(s)


@dataclass
class SflowO1:
    beta: float
    use_native: bool | None = None
    use_compile: bool = False
    compile_backend: str | None = "inductor"

    def __post_init__(self):
        coef = self.beta ** 2 / 8.0
        self.c = {
            "psi0": self.beta / 8.0,
            "psi2": 1.0 / 3.0 * coef,
            "psi11": -1.0 / 5.0 * coef,
            "psi11l": -1.0 / 6.0 * coef,
        }
        self.psi = {name: _psi_factory(name, self.use_native) for name in self.c.keys()}
        self._setup_compile()

    def _setup_compile(self):
        if not (self.use_compile and hasattr(tr, "compile")):
            self._call_impl = self._call_impl_nojit
            self._grad_impl = self._grad_impl_nojit
            self._mlapl_impl = self._mlapl_impl_nojit
            self._grad_lapl_impl = self._grad_lapl_impl_nojit
            return
        backend = self.compile_backend
        self._call_impl = tr.compile(self._call_impl_nojit, backend=backend)
        self._grad_impl = tr.compile(self._grad_impl_nojit, backend=backend)
        self._mlapl_impl = tr.compile(self._mlapl_impl_nojit, backend=backend)
        self._grad_lapl_impl = tr.compile(self._grad_lapl_impl_nojit, backend=backend)

    def _combine(self, vals, t: float, key: str):
        r = tr.zeros_like(vals["psi0"][key])
        for name, c in self.c.items():
            weight = c if name == "psi0" else t * c
            r = r + weight * vals[name][key]
        return r

    def _call_impl_nojit(self, s: tr.Tensor, t: float) -> tr.Tensor:
        ctx = nullcontext() if self.use_native else _NeighborCache(s)
        with ctx:
            vals = _compute(self.psi, s, need_a=True)
            return self._combine(vals, t, "a")

    def __call__(self, s: tr.Tensor, t: float) -> tr.Tensor:
        if self.use_native and HAS_NATIVE and sflowo1_action is not None:
            return sflowo1_action(s, float(self.beta), float(t))
        return self._call_impl(s, t)

    def _grad_impl_nojit(self, s: tr.Tensor, t: float) -> tr.Tensor:
        ctx = nullcontext() if self.use_native else _NeighborCache(s)
        with ctx:
            vals = _compute(self.psi, s, need_g=True)
            return self._combine(vals, t, "g")

    def grad(self, s: tr.Tensor, t: float) -> tr.Tensor:
        # Use fused native grad if available and enabled
        if self.use_native and HAS_NATIVE and sflowo1_grad is not None:
            return sflowo1_grad(s, float(self.beta), float(t))
        return self._grad_impl(s, t)

    def mgrad(self, s: tr.Tensor, t: float) -> tr.Tensor:
        return -self.grad(s, t)

    def _mlapl_impl_nojit(self, s: tr.Tensor, t: float) -> tr.Tensor:
        ctx = nullcontext() if self.use_native else _NeighborCache(s)
        with ctx:
            vals = _compute(self.psi, s, need_lapl=True)
            r = self._combine(vals, t, "lapl")
            return -r

    def mlapl(self, s: tr.Tensor, t: float) -> tr.Tensor:
        if self.use_native and HAS_NATIVE and sflowo1_lapl is not None:
            return sflowo1_lapl(s, float(self.beta), float(t))
        return self._mlapl_impl(s, t)

    def _grad_lapl_impl_nojit(self, s: tr.Tensor, t: float) -> tr.Tensor:
        ctx = nullcontext() if self.use_native else _NeighborCache(s)
        with ctx:
            vals = _compute(self.psi, s, need_gl=True)
            return self._combine(vals, t, "gl")

    def grad_lapl(self, s: tr.Tensor, t: float) -> tr.Tensor:
        if self.use_native and HAS_NATIVE and sflowo1_grad_lapl is not None:
            return sflowo1_grad_lapl(s, float(self.beta), float(t))
        return self._grad_lapl_impl(s, t)


@dataclass
class SflowO2:
    beta: float
    use_native: bool | None = None
    use_compile: bool = False
    compile_backend: str | None = "inductor"

    def __post_init__(self):
        coef1 = self.beta / 8.0
        coef2 = self.beta ** 2 / 8.0
        coef3 = self.beta ** 3 / 8.0
        # Total weights for each Psi when expanded: S0 + t S1 + t^2 S2
        self.c0 = {"psi0": coef1}
        self.c1 = {
            "psi2": 1.0 / 3.0 * coef2,
            "psi11": -1.0 / 5.0 * coef2,
            "psi11l": -1.0 / 6.0 * coef2,
        }
        self.c2 = {
            "psi0": -1489.0 / 3000.0 * coef3,
            "psi3": +29.0 / 200.0 * coef3,
            "psi21": -11.0 / 100.0 * coef3,
            "psi12d": -1.0 / 30.0 * coef3,
            "psi111": +2.0 / 90.0 * coef3,
            "psi111c": +1.0 / 40.0 * coef3,
            "psi12l": +41.0 / 1500.0 * coef3,
            "psi1l1": -7.0 / 300.0 * coef3,
            "psi111l": +7.0 / 1800.0 * coef3,
        }
        self.psi = {
            name: _psi_factory(name, self.use_native) for name in {**self.c0, **self.c1, **self.c2}.keys()
        }
        self._setup_compile()

    def _setup_compile(self):
        if not (self.use_compile and hasattr(tr, "compile")):
            self._call_impl = self._call_impl_nojit
            self._grad_impl = self._grad_impl_nojit
            self._mlapl_impl = self._mlapl_impl_nojit
            self._grad_lapl_impl = self._grad_lapl_impl_nojit
            return
        backend = self.compile_backend
        self._call_impl = tr.compile(self._call_impl_nojit, backend=backend)
        self._grad_impl = tr.compile(self._grad_impl_nojit, backend=backend)
        self._mlapl_impl = tr.compile(self._mlapl_impl_nojit, backend=backend)
        self._grad_lapl_impl = tr.compile(self._grad_lapl_impl_nojit, backend=backend)

    def _combine(self, vals, t: float, key: str):
        r = tr.zeros_like(vals["psi0"][key])
        # t^0 part
        for name, c in self.c0.items():
            r = r + c * vals[name][key]
        if t == 0:
            return r
        # t^1 part
        for name, c in self.c1.items():
            r = r + (t * c) * vals[name][key]
        # t^2 part
        t2 = t * t
        for name, c in self.c2.items():
            r = r + (t2 * c) * vals[name][key]
        return r

    def _call_impl_nojit(self, s: tr.Tensor, t: float) -> tr.Tensor:
        ctx = nullcontext() if self.use_native else _NeighborCache(s)
        with ctx:
            vals = _compute(self.psi, s, need_a=True)
            return self._combine(vals, t, "a")

    def __call__(self, s: tr.Tensor, t: float) -> tr.Tensor:
        if self.use_native and HAS_NATIVE and sflowo2_action is not None:
            return sflowo2_action(s, float(self.beta), float(t))
        return self._call_impl(s, t)

    def _grad_impl_nojit(self, s: tr.Tensor, t: float) -> tr.Tensor:
        ctx = nullcontext() if self.use_native else _NeighborCache(s)
        with ctx:
            vals = _compute(self.psi, s, need_g=True)
            return self._combine(vals, t, "g")

    def grad(self, s: tr.Tensor, t: float) -> tr.Tensor:
        return self._grad_impl(s, t)

    def mgrad(self, s: tr.Tensor, t: float) -> tr.Tensor:
        return -self.grad(s, t)

    def _mlapl_impl_nojit(self, s: tr.Tensor, t: float) -> tr.Tensor:
        ctx = nullcontext() if self.use_native else _NeighborCache(s)
        with ctx:
            vals = _compute(self.psi, s, need_lapl=True)
            r = self._combine(vals, t, "lapl")
            return -r

    def mlapl(self, s: tr.Tensor, t: float) -> tr.Tensor:
        return self._mlapl_impl(s, t)

    def _grad_lapl_impl_nojit(self, s: tr.Tensor, t: float) -> tr.Tensor:
        ctx = nullcontext() if self.use_native else _NeighborCache(s)
        with ctx:
            vals = _compute(self.psi, s, need_gl=True)
            return self._combine(vals, t, "gl")

    def grad_lapl(self, s: tr.Tensor, t: float) -> tr.Tensor:
        return self._grad_lapl_impl(s, t)


def _assert_close(name: str, a: tr.Tensor, b: tr.Tensor, atol=3e-6, rtol=3e-6):
    if not tr.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{name} mismatch: max diff {diff}")


def _time_fn(fn, repeats: int) -> float:
    for _ in range(3):
        fn()
    tic = time.perf_counter()
    for _ in range(repeats):
        fn()
    toc = time.perf_counter()
    return toc - tic


def benchmark(
    beta: float = 1.2345,
    lat: Iterable[int] = (8, 8),
    batch: int = 16,
    t: float = 0.37,
    repeats: int = 50,
    use_compile: bool = False,
    compile_backend: str | None = "inductor",
    use_native: bool | None = None,
) -> None:
    s = base.uniform_spin(batch, list(lat)).to(dtype=base.dtype)
    pairs = [
        ("O0", BaseO0(beta), SflowO0(beta)),
        ("O1", BaseO1(beta), SflowO1(beta, use_native=use_native)),
        ("O2", BaseO2(beta), SflowO2(beta, use_native=use_native)),
    ]

    def maybe_compile(name, sflow):
        if not use_compile or not hasattr(tr, "compile"):
            return sflow
        try:
            compiled = tr.compile(sflow, backend=compile_backend)
            needed = ["grad", "mgrad", "mlapl", "grad_lapl"]
            if all(hasattr(compiled, m) for m in needed):
                return compiled
            # If compiled object lacks needed methods, skip compile.
            print(f"Compile for {name} skipped: missing attrs on compiled object")
            return sflow
        except Exception as exc:
            print(f"Compile for {name} skipped: {exc}")
            return sflow

    compiled = [(n, o, maybe_compile(n, n_new)) for n, o, n_new in pairs]

    def bench_pair(name, old, new):
        with tr.no_grad():
            _assert_close(f"{name} action", old(s, t), new(s, t))
            _assert_close(f"{name} grad", old.grad(s, t), new.grad(s, t))
            _assert_close(f"{name} mlapl", old.mlapl(s, t), new.mlapl(s, t))

        act_old = _time_fn(lambda: old(s, t), repeats)
        act_new = _time_fn(lambda: new(s, t), repeats)
        grad_old = _time_fn(lambda: old.grad(s, t), repeats)
        grad_new = _time_fn(lambda: new.grad(s, t), repeats)
        mlapl_old = _time_fn(lambda: old.mlapl(s, t), repeats)
        mlapl_new = _time_fn(lambda: new.mlapl(s, t), repeats)
        gradlap_old = _time_fn(lambda: old.grad_lapl(s, t), repeats)
        gradlap_new = _time_fn(lambda: new.grad_lapl(s, t), repeats)

        print(
            f"{name}: action {act_old:.4f}s -> {act_new:.4f}s (x{act_old/act_new:.2f}), "
            f"grad {grad_old:.4f}s -> {grad_new:.4f}s (x{grad_old/grad_new:.2f}), "
            f"mlapl {mlapl_old:.4f}s -> {mlapl_new:.4f}s (x{mlapl_old/mlapl_new:.2f}), "
            f"grad_lapl {gradlap_old:.4f}s -> {gradlap_new:.4f}s (x{gradlap_old/gradlap_new:.2f})"
        )

    print(f"Benchmark opt2 vs base on batch={batch}, lat={tuple(lat)}, repeats={repeats}")
    if use_compile:
        print(f"torch.compile enabled (backend={compile_backend})")
    for name, old, new in compiled:
        bench_pair(name, old, new)


if __name__ == "__main__":
    benchmark()
