"""
Native (C++ extension) kernels for selected Psi functions.

Currently implements Psi0 action/grad/lapl/grad_lapl in C++.
Falls back to Python implementations if compilation/import fails.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch as tr
from torch.utils.cpp_extension import load, CppExtension, BuildExtension
import setuptools

HAS_NATIVE = False
EXT = None
_BUILD_ERROR = None


def _build_ext():
    src = Path(__file__).parent / "native" / "o3psi_native.cpp"
    name = "o3psi_native_ext"
    extra = ["-std=c++17"]
    build_directory = Path(__file__).parent / ".build"
    build_directory.mkdir(exist_ok=True)
    try:
        return load(
            name=name,
            sources=[str(src)],
            extra_cflags=extra,
            build_directory=str(build_directory),
            verbose=False,
        )
    except Exception as exc:
        # fallback: try setuptools without ninja
        setuptools.setup(
            name=name,
            ext_modules=[
                CppExtension(
                    name=name,
                    sources=[str(src)],
                    extra_compile_args=extra,
                )
            ],
            cmdclass={"build_ext": BuildExtension},
            script_args=["build_ext", "--inplace"],
        )
        return load(
            name=name,
            sources=[str(src)],
            extra_cflags=extra,
            build_directory=str(build_directory),
            verbose=False,
        )


try:
    EXT = _build_ext()
    HAS_NATIVE = True
except Exception as exc:
    HAS_NATIVE = False
    EXT = None
    _BUILD_ERROR = exc
    # Uncomment for immediate feedback:
    # print(f"o3psi_native: build failed ({exc}); using Python kernels")


class Psi0Native:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi0_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi0_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi0_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi0_grad_lapl(s)


class Psi2Native:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi2_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi2_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi2_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi2_grad_lapl(s)


class Psi11Native:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi11_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi11_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi11_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi11_grad_lapl(s)


class Psi11lNative:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi11l_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi11l_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi11l_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi11l_grad_lapl(s)


class Psi3Native:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi3_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi3_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi3_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi3_grad_lapl(s)


class Psi21Native:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi21_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi21_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi21_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi21_grad_lapl(s)


class Psi12dNative:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi12d_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi12d_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi12d_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi12d_grad_lapl(s)


class Psi12lNative:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi12l_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi12l_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi12l_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi12l_grad_lapl(s)


class Psi111Native:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111_grad_lapl(s)


class Psi111cNative:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111c_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111c_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111c_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111c_grad_lapl(s)


class Psi1l1Native:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi1l1_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi1l1_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi1l1_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi1l1_grad_lapl(s)


class Psi111lNative:
    def action(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111l_action(s)

    def grad(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111l_grad(s)

    def lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111l_lapl(s)

    def grad_lapl(self, s: tr.Tensor) -> tr.Tensor:
        return EXT.psi111l_grad_lapl(s)

# Fused SflowO2 ops
def sflowo2_action(s: tr.Tensor, beta: float, t: float) -> tr.Tensor:
    return EXT.sflowo2_action(s, beta, t)

def sflowo2_grad(s: tr.Tensor, beta: float, t: float) -> tr.Tensor:
    return EXT.sflowo2_grad(s, beta, t)

def sflowo2_lapl(s: tr.Tensor, beta: float, t: float) -> tr.Tensor:
    return EXT.sflowo2_lapl(s, beta, t)

def sflowo2_grad_lapl(s: tr.Tensor, beta: float, t: float) -> tr.Tensor:
    return EXT.sflowo2_grad_lapl(s, beta, t)

# Fused SflowO1 ops
def sflowo1_action(s: tr.Tensor, beta: float, t: float) -> tr.Tensor:
    return EXT.sflowo1_action(s, beta, t)

def sflowo1_grad(s: tr.Tensor, beta: float, t: float) -> tr.Tensor:
    return EXT.sflowo1_grad(s, beta, t)

def sflowo1_lapl(s: tr.Tensor, beta: float, t: float) -> tr.Tensor:
    return EXT.sflowo1_lapl(s, beta, t)

def sflowo1_grad_lapl(s: tr.Tensor, beta: float, t: float) -> tr.Tensor:
    return EXT.sflowo1_grad_lapl(s, beta, t)


__all__ = [
    "HAS_NATIVE",
    "_BUILD_ERROR",
    "Psi0Native",
    "Psi2Native",
    "Psi11Native",
    "Psi11lNative",
    "Psi3Native",
    "Psi21Native",
    "Psi12dNative",
    "Psi12lNative",
    "Psi111Native",
    "Psi111cNative",
    "Psi1l1Native",
    "Psi111lNative",
    "sflowo2_action",
    "sflowo2_grad",
    "sflowo2_lapl",
    "sflowo2_grad_lapl",
    "sflowo1_action",
    "sflowo1_grad",
    "sflowo1_lapl",
    "sflowo1_grad_lapl",
]
