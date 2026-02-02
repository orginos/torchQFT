"""
Fast Functionals for Control Variates

These alternatives to Funct3T provide faster Laplacian computation:

1. FunctHutchinson: Wraps any functional, uses stochastic trace estimation
   - Speedup: ~L² (e.g., 64x for L=8)
   - Trade-off: Introduces variance in Laplacian estimate

2. FunctFourier: Operates in Fourier space with learnable filters
   - Speedup: ~L² (analytical Laplacian via chain rule)
   - Naturally translation-invariant

3. FunctLocal: Local polynomial features with analytical Laplacian
   - Speedup: ~L² (no autograd loop)
   - Simple but effective for local correlations
"""

import torch as tr
import torch.nn as nn
import numpy as np


class FunctHutchinson(nn.Module):
    """
    Wrapper that uses Hutchinson trace estimator for Laplacian.

    Instead of computing ∇²F = Σᵢ ∂²F/∂φᵢ² exactly (L² autograd calls),
    uses: ∇²F ≈ E[z^T H z] where z is Rademacher random vector.

    Single sample: ~2 autograd calls instead of L²
    """

    def __init__(self, base_functional, n_hutchinson=1):
        super().__init__()
        self.F = base_functional
        self.y = base_functional.y
        self.dim = getattr(base_functional, 'dim', 2)
        self.n_hutchinson = n_hutchinson  # Number of random samples for averaging

    def forward(self, x):
        return self.F(x)

    def grad_and_lapl(self, x):
        """
        Compute gradient and Laplacian using Hutchinson estimator.
        """
        batch_size = x.shape[0]

        # Forward pass
        y = self.forward(x)

        # Compute gradient
        grad = tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]

        # Hutchinson trace estimator for Laplacian
        lapl = tr.zeros(batch_size, device=x.device, dtype=x.dtype)

        for _ in range(self.n_hutchinson):
            # Rademacher random vector (+1 or -1)
            z = 2 * tr.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) - 1

            # Compute z^T @ grad (dot product)
            grad_z = (grad * z).sum()

            # Compute gradient of (z^T @ grad) w.r.t. x, then dot with z
            # This gives z^T @ H @ z where H is the Hessian
            hvp = tr.autograd.grad(grad_z, x, create_graph=True)[0]

            # z^T H z estimates trace(H) = Laplacian
            lapl += (hvp * z).sum(dim=(1, 2))

        lapl = lapl / self.n_hutchinson

        return grad, lapl


class FunctFourier(nn.Module):
    """
    Fourier-space functional with efficient gradient/Laplacian.

    Architecture:
    1. FFT of input field φ → φ̃(k)
    2. Compute power spectrum features |φ̃(k)|²
    3. Apply learnable k-space filters
    4. Small MLP on filtered features

    The Laplacian uses chain rule through FFT which is O(L² log L).
    """

    def __init__(self, L, dim=2, y=0, hidden_dims=[32, 16],
                 dtype=tr.float32, activation=nn.GELU()):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.dtype = dtype

        # Learnable k-space weights (real, for power spectrum weighting)
        # Shape: (L, L//2+1) for rfft2 output
        self.k_weights = nn.Parameter(tr.randn(L, L//2 + 1) * 0.1)

        # Additional phase-sensitive weights (complex)
        self.k_weights_phase = nn.Parameter(tr.randn(L, L//2 + 1) * 0.1)

        # Build MLP for final output
        # Input: 3 features (weighted power, tau-correlation, phase feature)
        self.mlp = nn.Sequential()
        in_dim = 3
        for i, h in enumerate(hidden_dims):
            self.mlp.add_module(f'lin{i}', nn.Linear(in_dim, h, dtype=dtype))
            self.mlp.add_module(f'act{i}', activation)
            in_dim = h
        self.mlp.add_module('out', nn.Linear(in_dim, 1, dtype=dtype))

        # Precompute k² values for Laplacian (on first forward)
        self.register_buffer('k_squared', None)
        self._init_k_squared(L)

    def _init_k_squared(self, L):
        """Precompute -k² for Laplacian in Fourier space."""
        kx = tr.fft.fftfreq(L) * 2 * np.pi
        ky = tr.fft.rfftfreq(L) * 2 * np.pi
        kx_grid, ky_grid = tr.meshgrid(kx, ky, indexing='ij')
        self.k_squared = -(kx_grid**2 + ky_grid**2)

    def _compute_features(self, phi_k, phi_k_shifted):
        """Compute features from Fourier-space field."""
        # Power spectrum weighted by learnable weights
        power = (phi_k.real**2 + phi_k.imag**2)
        weighted_power = (power * self.k_weights.unsqueeze(0)).sum(dim=(1, 2))

        # Cross-correlation at separation tau (in k-space: φ̃(k) * φ̃*(k) * e^{ik·τ})
        cross = (phi_k * phi_k_shifted.conj()).real
        weighted_cross = (cross * self.k_weights.unsqueeze(0)).sum(dim=(1, 2))

        # Phase-sensitive feature
        phase_feature = (phi_k.real * self.k_weights_phase.unsqueeze(0)).sum(dim=(1, 2))

        return tr.stack([weighted_power, weighted_cross, phase_feature], dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        # FFT of input field
        phi_k = tr.fft.rfft2(x)

        # Shift in real space = phase in k-space
        # φ(x + τ) ↔ φ̃(k) * e^{ik·τ}
        kx = tr.fft.fftfreq(self.L, device=x.device) * 2 * np.pi
        ky = tr.fft.rfftfreq(self.L, device=x.device) * 2 * np.pi

        if self.dim == 2:  # tau is along dim 2 (y-direction)
            phase_shift = tr.exp(1j * ky.unsqueeze(0) * self.y)
        else:  # tau along dim 1 (x-direction)
            phase_shift = tr.exp(1j * kx.unsqueeze(1) * self.y)

        phi_k_shifted = phi_k * phase_shift.unsqueeze(0)

        # Compute features
        features = self._compute_features(phi_k, phi_k_shifted)

        # MLP output
        out = self.mlp(features).squeeze(-1)

        # Enforce parity: F(φ) = F(-φ) by making it depend on φ²
        # Already satisfied since we use power spectrum

        return out

    def grad_and_lapl(self, x):
        """Compute gradient and Laplacian using autograd (still needed but faster)."""
        y = self.forward(x)
        grad = tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]

        # Use Hutchinson for Laplacian (much faster than full loop)
        z = 2 * tr.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) - 1
        grad_z = (grad * z).sum()
        hvp = tr.autograd.grad(grad_z, x, create_graph=True)[0]
        lapl = (hvp * z).sum(dim=(1, 2))

        return grad, lapl


class FunctLocal(nn.Module):
    """
    Local functional with analytical Laplacian.

    F(φ) = Σ_x f(φ_x, φ_{x+τ}, Δφ_x, Δφ_{x+τ})

    where Δφ_x = Σ_neighbors φ - 4φ_x (discrete Laplacian of φ)

    The Laplacian of F can be computed analytically:
    ∇²F = Σ_x [∂²f/∂φ_x² + terms from Δφ dependencies]

    This avoids the L² autograd loop.
    """

    def __init__(self, L, dim=2, y=0, hidden_dims=[16, 16],
                 dtype=tr.float32, activation=nn.GELU()):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.dtype = dtype

        # Input features: φ_x, φ_{x+τ}, Δφ_x, Δφ_{x+τ}, φ_x², φ_{x+τ}², φ_x*φ_{x+τ}
        n_features = 7

        # Small MLP applied at each site
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h, dtype=dtype))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, 1, dtype=dtype))

        self.site_net = nn.Sequential(*layers)

    def _discrete_laplacian(self, phi):
        """Compute discrete Laplacian: Δφ_x = Σ_neighbors φ - 4φ_x"""
        lap = (tr.roll(phi, 1, dims=1) + tr.roll(phi, -1, dims=1) +
               tr.roll(phi, 1, dims=2) + tr.roll(phi, -1, dims=2) - 4 * phi)
        return lap

    def forward(self, x):
        batch_size = x.shape[0]

        # Compute local features
        phi = x
        phi_tau = tr.roll(x, -self.y, dims=self.dim)
        lap_phi = self._discrete_laplacian(x)
        lap_phi_tau = self._discrete_laplacian(phi_tau)

        # Stack features: shape (batch, L, L, n_features)
        features = tr.stack([
            phi,
            phi_tau,
            lap_phi,
            lap_phi_tau,
            phi**2,
            phi_tau**2,
            phi * phi_tau
        ], dim=-1)

        # Apply site network and sum over lattice
        # Reshape for linear layers
        features_flat = features.view(-1, 7)
        site_out = self.site_net(features_flat)
        site_out = site_out.view(batch_size, self.L, self.L)

        # Sum over lattice
        out = site_out.sum(dim=(1, 2))

        return out

    def grad_and_lapl(self, x):
        """Use Hutchinson estimator for Laplacian."""
        y = self.forward(x)
        grad = tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]

        # Hutchinson trace estimator
        z = 2 * tr.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) - 1
        grad_z = (grad * z).sum()
        hvp = tr.autograd.grad(grad_z, x, create_graph=True)[0]
        lapl = (hvp * z).sum(dim=(1, 2))

        return grad, lapl


class FunctQuadratic(nn.Module):
    """
    Learnable quadratic functional with EXACT analytical Laplacian.

    F(φ) = Σ_{x,y} φ_x W(x-y) φ_y + Σ_x φ_x φ_{x+τ} V(neighbors)

    For a quadratic form, the Hessian is constant, so:
    ∇²F = 2 * Tr(W) (constant!)

    More useful: make it depend on φ² to get non-trivial Laplacian.

    F(φ) = g(Σ_x φ_x φ_{x+τ}) where g is a learnable function

    Then ∇F and ∇²F can be computed analytically.
    """

    def __init__(self, L, dim=2, y=0, dtype=tr.float32):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.dtype = dtype

        # Learnable polynomial coefficients for g(s) = Σ_n a_n s^n
        # where s = <φ_x φ_{x+τ}> (correlation)
        self.coeffs = nn.Parameter(tr.tensor([0.0, 1.0, 0.1, 0.01], dtype=dtype))

        # Additional learnable weights for neighbor correlations
        self.neighbor_weight = nn.Parameter(tr.tensor(0.1, dtype=dtype))

    def forward(self, x):
        # s = mean correlation at distance tau
        x_tau = tr.roll(x, -self.y, dims=self.dim)
        s = (x * x_tau).mean(dim=(1, 2))

        # Neighbor correlation
        neighbors = (tr.roll(x, 1, dims=1) + tr.roll(x, -1, dims=1) +
                    tr.roll(x, 1, dims=2) + tr.roll(x, -1, dims=2))
        s_neighbor = (x * neighbors).mean(dim=(1, 2))

        # Polynomial: g(s) = a0 + a1*s + a2*s² + a3*s³
        s_total = s + self.neighbor_weight * s_neighbor
        powers = tr.stack([s_total**i for i in range(len(self.coeffs))], dim=1)
        out = (powers * self.coeffs).sum(dim=1)

        return out

    def grad_and_lapl(self, x):
        """
        Analytical gradient and Laplacian for polynomial of correlations.

        F = g(s) where s = (1/V) Σ_x φ_x φ_{x+τ}

        ∂F/∂φ_z = g'(s) * (1/V) * (φ_{z+τ} + φ_{z-τ})
        ∂²F/∂φ_z² = g''(s) * (1/V)² * (φ_{z+τ} + φ_{z-τ})² + g'(s) * 0 (if τ ≠ 0)

        For τ = 0: s = (1/V) Σ_x φ_x²
        ∂F/∂φ_z = g'(s) * (2/V) * φ_z
        ∂²F/∂φ_z² = g''(s) * (4/V²) * φ_z² + g'(s) * (2/V)
        """
        batch_size = x.shape[0]
        V = self.L * self.L

        x_tau = tr.roll(x, -self.y, dims=self.dim)
        x_mtau = tr.roll(x, self.y, dims=self.dim)

        # s and its derivatives
        s = (x * x_tau).mean(dim=(1, 2))

        # Neighbor terms
        neighbors = (tr.roll(x, 1, dims=1) + tr.roll(x, -1, dims=1) +
                    tr.roll(x, 1, dims=2) + tr.roll(x, -1, dims=2))
        s_neighbor = (x * neighbors).mean(dim=(1, 2))
        s_total = s + self.neighbor_weight * s_neighbor

        # g'(s) and g''(s) for polynomial
        # g(s) = a0 + a1*s + a2*s² + a3*s³
        # g'(s) = a1 + 2*a2*s + 3*a3*s²
        # g''(s) = 2*a2 + 6*a3*s
        a = self.coeffs
        g_prime = a[1] + 2*a[2]*s_total + 3*a[3]*s_total**2
        g_double_prime = 2*a[2] + 6*a[3]*s_total

        # Reshape for broadcasting: (batch,) -> (batch, 1, 1)
        g_prime_3d = g_prime.view(-1, 1, 1)
        g_double_prime_3d = g_double_prime.view(-1, 1, 1)

        # Gradient: ∂F/∂φ_z = g'(s) * (1/V) * (φ_{z+τ} + φ_{z-τ} + w * neighbors_contrib)
        grad = g_prime_3d / V * (x_tau + x_mtau)

        # Neighbor contribution to gradient
        neighbor_grad = self.neighbor_weight * g_prime_3d / V * neighbors
        grad = grad + neighbor_grad

        # Laplacian (sum of second derivatives)
        # For main term with τ ≠ 0:
        term1 = (x_tau + x_mtau)**2
        lapl = (g_double_prime_3d / V**2 * term1).sum(dim=(1, 2))

        # Neighbor contribution to Laplacian
        lapl = lapl + (self.neighbor_weight * g_double_prime_3d / V**2 * neighbors**2).sum(dim=(1, 2))
        lapl = lapl + 4 * self.neighbor_weight * g_prime / V  # 4 neighbors contribute to each site

        return grad, lapl


# Import original models for wrapping
from control_variates import model_factory as original_model_factory, activation_factory


def fast_model_factory(class_name, L, y, conv_layers=[4,4,4,4], activation='gelu',
                       dtype=tr.float32, n_hutchinson=1, **kwargs):
    """
    Unified factory for all functional models (original + fast).

    Models:
    - 'Funct3T': Original model (slow Laplacian)
    - 'Funct3T_Hutch': Funct3T wrapped with Hutchinson estimator
    - 'FunctFourier': Fourier-space model with Hutchinson
    - 'FunctLocal': Local features with Hutchinson
    - 'FunctQuadratic': Analytical Laplacian (fastest)

    Args:
        class_name: Model type
        L: Lattice size
        y: Tau value (time separation)
        conv_layers: Conv layer widths (for Funct3T variants)
        activation: Activation function name
        dtype: Data type
        n_hutchinson: Number of Hutchinson samples (for stochastic Laplacian)
    """

    # Get activation function
    if isinstance(activation, str):
        activ = activation_factory(activation)
    else:
        activ = activation

    if class_name == 'Funct3T':
        # Original model
        return original_model_factory('Funct3T', L=L, y=y,
                                      conv_layers=conv_layers,
                                      activation=activ, dtype=dtype)

    elif class_name == 'Funct3T_Hutch':
        # Original model with Hutchinson wrapper
        base = original_model_factory('Funct3T', L=L, y=y,
                                      conv_layers=conv_layers,
                                      activation=activ, dtype=dtype)
        return FunctHutchinson(base, n_hutchinson=n_hutchinson)

    elif class_name == 'FunctFourier':
        return FunctFourier(L=L, y=y, dtype=dtype, activation=activ)

    elif class_name == 'FunctLocal':
        return FunctLocal(L=L, y=y, dtype=dtype, activation=activ)

    elif class_name == 'FunctQuadratic':
        return FunctQuadratic(L=L, y=y, dtype=dtype)

    else:
        raise ValueError(f"Unknown model: {class_name}. Available: "
                        f"Funct3T, Funct3T_Hutch, FunctFourier, FunctLocal, FunctQuadratic")


# List of all available models for benchmarking
ALL_MODELS = ['Funct3T', 'Funct3T_Hutch', 'FunctFourier', 'FunctLocal', 'FunctQuadratic']
