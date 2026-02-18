#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autocorrelation utilities for MCMC diagnostics.

Implements a normalized autocorrelation function (ACF), integrated
autocorrelation time with Madras–Sokal windowing, and effective sample size.
"""

from __future__ import annotations

import numpy as np


def autocorr(x: np.ndarray) -> np.ndarray:
    """Return normalized autocorrelation function of a 1D series."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1)
    n = x.size
    if n < 2:
        return np.array([1.0], dtype=float)
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")[n - 1 :]
    if corr[0] == 0.0:
        return np.ones(1, dtype=float)
    return corr / corr[0]


def tau_int_madras_sokal(x: np.ndarray, c: float = 5.0, maxlag: int | None = None) -> float:
    """
    Integrated autocorrelation time using the Madras–Sokal window.

    Window condition: M >= c * tau_int(M), where tau_int(M) = 0.5 + sum_{t=1}^M rho(t).
    """
    rho = autocorr(x)
    n = len(rho)
    if n <= 1:
        return 0.5
    if maxlag is None:
        maxlag = n - 1
    maxlag = max(1, min(maxlag, n - 1))

    tau = 0.5
    for M in range(1, maxlag + 1):
        tau = 0.5 + np.sum(rho[1 : M + 1])
        if M >= c * tau:
            break
    return float(tau)


def effective_sample_size(x: np.ndarray, tau_int: float | None = None, c: float = 5.0) -> float:
    """Effective sample size using tau_int (computed if not provided)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    if tau_int is None:
        tau_int = tau_int_madras_sokal(x, c=c)
    if tau_int <= 0:
        return float(n)
    return float(n) / (2.0 * tau_int)


def summary(x: np.ndarray, c: float = 5.0, maxlag: int | None = None) -> dict:
    """Return a dict with tau_int and ESS for convenience."""
    tau = tau_int_madras_sokal(x, c=c, maxlag=maxlag)
    ess = effective_sample_size(x, tau_int=tau, c=c)
    return {"tau_int": tau, "ess": ess}


if __name__ == "__main__":
    # Small self-test
    rng = np.random.default_rng(123)
    x = rng.normal(size=1000)
    s = summary(x)
    print(f"tau_int={s['tau_int']:.3f}  ess={s['ess']:.1f}")
