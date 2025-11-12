import numpy as np
import torch
from typing import Callable, Union, Sequence, Dict, Any


def jackknife(*args, F):
    """
    Unified jackknife estimator with bias correction and standard error.
    Works with any number of input arrays (list, np.ndarray, or torch.Tensor).

    Parameters
    ----------
    *args : list of array-like
        Input samples of equal length.
    F : callable
        Function F(mean(x1), mean(x2), ..., mean(xn)) -> scalar or tensor.

    Returns
    -------
    C : float or torch.Tensor
        Original estimator F(means).
    C_jk_mean : float or torch.Tensor
        Mean of jackknife replicates.
    C_err : float or torch.Tensor
        Jackknife standard error.
    bias : float or torch.Tensor
        Jackknife bias estimate.
    C_corr : float or torch.Tensor
        Bias-corrected estimator.
    """

    # Check if we're in torch or numpy mode
    is_torch = any(torch.is_tensor(a) for a in args)

    # --- PyTorch backend ---
    if is_torch:
        # Convert everything to torch tensors (same dtype/device as first tensor)
        base = next(a for a in args if torch.is_tensor(a))
        device, dtype = base.device, base.dtype
        arrays = [torch.as_tensor(a, dtype=dtype, device=device).float() for a in args]
        N = arrays[0].numel()
        assert all(x.numel() == N for x in arrays), "All inputs must have same length"

        # Compute means and original estimate
        means = [x.mean() for x in arrays]
        C = F(*means)

        # Leave-one-out means
        sums = [x.sum() for x in arrays]
        means_jk = [(s - x) / (N - 1) for s, x in zip(sums, arrays)]

        # Evaluate jackknife replicates (vectorized)
        C_jk = F(*means_jk)

        # Jackknife stats
        C_jk_mean = C_jk.mean()
        C_err = torch.sqrt((N - 1) * ((C_jk - C_jk_mean) ** 2).mean())
        bias = (N - 1) * (C_jk_mean - C)
        C_corr = C - bias

        return C, C_jk_mean, C_err, bias, C_corr

    # --- NumPy backend ---
    else:
        arrays = [np.asarray(a, dtype=float) for a in args]
        N = len(arrays[0])
        assert all(len(x) == N for x in arrays), "All inputs must have same length"

        means = [np.mean(x) for x in arrays]
        C = F(*means)

        sums = [np.sum(x) for x in arrays]
        means_jk = [(s - x) / (N - 1) for s, x in zip(sums, arrays)]

        C_jk = F(*means_jk)

        C_jk_mean = np.mean(C_jk)
        C_err = np.sqrt((N - 1) * np.mean((C_jk - C_jk_mean)**2))
        bias = (N - 1) * (C_jk_mean - C)
        C_corr = C - bias

        return C, C_jk_mean, C_err, bias, C_corr






###### THIS JACKKNIFE WORKS WITH TORCH
import torch as tr

Array = Union[np.ndarray, tr.Tensor]
ScalarLike = Union[float, int, np.number, tr.Tensor]

def _backend(arrs: Sequence[Array]) -> str:
    """
    Decide whether to run in torch or numpy space.
    Requires all arrays to be the same backend.
    """
    kinds = {("torch" if isinstance(a, tr.Tensor) else "numpy") for a in arrs}
    if len(kinds) != 1:
        raise TypeError("All inputs must be either all torch tensors or all numpy arrays.")
    return kinds.pop()

def _shape_check(arrs: Sequence[Array]) -> tuple[int, int]:
    M, B = None, None
    for a in arrs:
        if a.ndim != 2:
            raise ValueError("Each input must have shape [Nmeas, B].")
        if M is None and B is None:
            M, B = a.shape
        elif a.shape != (M, B):
            raise ValueError("All inputs must share the same [Nmeas, B] shape.")
    if B is None or B < 2:
        raise ValueError("Need B >= 2 independent streams for stream-jackknife.")
    return M, B

def _as_float(x: ScalarLike) -> float:
    if isinstance(x, tr.Tensor):
        return float(x.item())
    return float(x)

def jackknife_over_streams(
    F: Callable[..., ScalarLike],
    *histories: Array,
) -> Dict[str, Any]:
    """
    Stream jackknife for a general scalar functional F(A, B, C, ...).

    Parameters
    ----------
    F : callable
        A function that accepts the full set of histories (each [Nmeas, B])
        and returns a scalar (float/numpy scalar / torch 0-d tensor).
        Inside F, you may compute anything (means, variances, nonlinear combos, etc.).
    histories : arrays
        One or more arrays, each with shape [Nmeas, B]. Each column is an
        independent HMC stream (jackknife 'block').

    Returns
    -------
    dict with keys:
      - estimate: plug-in estimate using all streams
      - se: jackknife standard error (for the plug-in; same SE is used for bias-corrected)
      - jk_bias: jackknife bias estimate
      - estimate_bias_corrected: bias-corrected estimator = B*estimate - (B-1)*jk_mean
      - jk_values: array/list of the B leave-one-out estimates
      - B, Nmeas: diagnostics
    """
    if len(histories) == 0:
        raise ValueError("Provide at least one history array.")
    backend = _backend(histories)
    M, B = _shape_check(histories)

    # Full-sample (plug-in) estimate
    theta = _as_float(F(*histories))

    # Leave-one-stream-out replicates
    jk_vals = []
    for b in range(B):
        if backend == "torch":
            keep = tr.cat((tr.arange(0, b), tr.arange(b + 1, B)))
            loo_args = [h[:, keep] for h in histories]
        else:
            loo_args = [np.delete(h, b, axis=1) for h in histories]
        val_b = _as_float(F(*loo_args))
        jk_vals.append(val_b)

    jk_vals = np.asarray(jk_vals, dtype=float)
    jk_mean = float(jk_vals.mean())

    # Jackknife variance of the plug-in estimator (standard formula)
    se = float(np.sqrt((B - 1) / B * np.sum((jk_vals - jk_mean) ** 2)))

    # Bias and bias-corrected estimator
    jk_bias = float((B - 1) * (jk_mean - theta))
    theta_bc = float(theta - jk_bias)  # == B*theta - (B-1)*jk_mean

    return dict(
        estimate=float(theta),
        se=se,
        jk_bias=jk_bias,
        estimate_bias_corrected=theta_bc,
        jk_values=jk_vals,
        B=B,
        Nmeas=M,
    )
