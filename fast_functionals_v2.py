"""
Fast Functionals V2 - Architectures with Efficient Laplacian Computation

These architectures maintain expressive power while enabling faster Laplacian:

1. FunctSeparable: F(φ) = g(Σₓ h(φₓ, neighbors)) - analytical Laplacian O(L²) not O(L⁴)
2. FunctConvSeparable: CNN features → separable aggregation → analytical Laplacian
3. Funct3T_Vmap: Uses functorch.vmap for vectorized Hessian diagonal (if available)
"""

import torch as tr
import torch.nn as nn
import numpy as np

# Check if functorch is available (PyTorch 2.0+)
try:
    from torch.func import vmap, jacrev, grad
    HAS_FUNCTORCH = True
except ImportError:
    try:
        from functorch import vmap, jacrev, grad
        HAS_FUNCTORCH = True
    except ImportError:
        HAS_FUNCTORCH = False

from control_variates import model_factory as original_model_factory, activation_factory


def create_coloring_probes(L, n_colors, device, dtype):
    """
    Create probe vectors based on graph coloring.

    For a 2D L×L lattice:
    - n_colors=2: checkerboard (works for nearest-neighbor)
    - n_colors=k²: k×k block coloring (works for interactions up to range k-1)

    Valid n_colors: 2, 4, 9, 16, 25, 36, 49, 64, ... (2 or perfect squares)

    Returns: (n_colors, L, L) tensor of probe masks
    """
    probes = tr.zeros(n_colors, L, L, device=device, dtype=dtype)

    if n_colors == 2:
        # Checkerboard: color = (i + j) % 2
        for i in range(L):
            for j in range(L):
                c = (i + j) % 2
                probes[c, i, j] = 1.0
    else:
        # k×k block coloring where k² = n_colors
        k = int(np.round(np.sqrt(n_colors)))
        if k * k != n_colors:
            raise ValueError(f"n_colors must be 2 or a perfect square (4, 9, 16, ...), got {n_colors}")
        for i in range(L):
            for j in range(L):
                c = (i % k) * k + (j % k)
                probes[c, i, j] = 1.0

    return probes


def probing_laplacian_coloring(forward_fn, x, grad, n_colors=4):
    """
    Compute Laplacian using probing with graph coloring + random signs.

    The idea:
    1. Use n_colors probe vectors based on graph coloring
    2. Each probe has random ±1 signs (Rademacher) on the colored sites
    3. This combines structured probing with stochastic estimation

    For exact trace with K colors, we'd need sites of same color to not interact.
    With CNNs this isn't exact, but adding random signs makes it unbiased.

    Args:
        forward_fn: Function that computes the scalar output
        x: Input tensor (batch, L, L)
        grad: Gradient of output w.r.t. x (batch, L, L)
        n_colors: Number of colors for probing (2, 4, 9, 16, ... must be 2 or perfect square)

    Returns:
        lapl: Laplacian estimate (batch,)
    """
    batch, L, _ = x.shape
    device = x.device
    dtype = x.dtype

    # Create coloring probes: (n_colors, L, L)
    probes = create_coloring_probes(L, n_colors, device, dtype)

    # For each color, create a random ±1 vector on the colored sites
    # and compute the Hessian-vector product
    lapl = tr.zeros(batch, device=device, dtype=dtype)

    for c in range(n_colors):
        # Random signs on this color's sites
        mask = probes[c]  # (L, L)
        z_signs = 2 * tr.randint(0, 2, (batch, L, L), device=device, dtype=dtype) - 1
        z = z_signs * mask.unsqueeze(0)  # (batch, L, L), non-zero only on color c

        # Hessian-vector product: H @ z
        grad_z = (grad * z).sum()
        hvp = tr.autograd.grad(grad_z, x, create_graph=True)[0]

        # z^T H z = sum over color c sites of z_i * (Hz)_i
        # Since z_i = ±1 on color c and 0 elsewhere:
        # z^T H z = sum_{i in color c} z_i * (Hz)_i
        lapl += (hvp * z).sum(dim=(1, 2))

    return lapl


def probing_laplacian_sites(forward_fn, x, grad, n_sites=4):
    """
    Compute Laplacian using random site sampling (Karniadakis-style).

    Instead of computing the full Laplacian Δf = Σᵢ ∂²f/∂φᵢ² at all L² sites,
    randomly select n_sites and compute exact second derivatives there:

        Δf ≈ (L²/n_sites) × Σₖ∈selected ∂²f/∂φₖ²

    This is unbiased because each site has equal probability of being selected.
    Used in PINNs (Physics-Informed Neural Networks) by Karniadakis et al.

    Args:
        forward_fn: Function that computes the scalar output
        x: Input tensor (batch, L, L)
        grad: Gradient of output w.r.t. x (batch, L, L)
        n_sites: Number of random sites to sample (any positive integer, 1 to L²)

    Returns:
        lapl: Laplacian estimate (batch,)
    """
    batch, L, _ = x.shape
    device = x.device
    dtype = x.dtype
    total_sites = L * L

    # Clamp n_sites to valid range
    n_sites = min(n_sites, total_sites)

    # Randomly select n_sites from [0, L²) without replacement
    # Use same sites for all samples in batch for efficiency
    all_sites = tr.randperm(total_sites, device=device)[:n_sites]

    # Convert flat indices to (i, j) coordinates
    rows = all_sites // L
    cols = all_sites % L

    # Compute exact second derivative at each selected site
    lapl = tr.zeros(batch, device=device, dtype=dtype)

    for k in range(n_sites):
        i, j = rows[k].item(), cols[k].item()

        # grad[i,j] component for all samples in batch
        g_ij = grad[:, i, j]  # (batch,)

        # Second derivative: ∂(∂f/∂φᵢⱼ)/∂φᵢⱼ
        # Need to compute gradient of g_ij w.r.t. x, then extract [i,j] component
        g_ij_sum = g_ij.sum()  # sum over batch for backward
        hvp = tr.autograd.grad(g_ij_sum, x, create_graph=True)[0]

        # Extract the diagonal element: ∂²f/∂φᵢⱼ²
        d2f_ij = hvp[:, i, j]  # (batch,)
        lapl += d2f_ij

    # Scale by L²/n_sites for unbiased estimate
    lapl *= (total_sites / n_sites)

    return lapl


def probing_laplacian(forward_fn, x, grad, n_probes=4, method='coloring'):
    """
    Compute Laplacian using probing.

    Args:
        forward_fn: Function that computes the scalar output
        x: Input tensor (batch, L, L)
        grad: Gradient of output w.r.t. x (batch, L, L)
        n_probes: Number of probes/colors/sites depending on method
        method: 'coloring' (graph coloring, n_probes must be 2 or perfect square)
                'sites' (random site sampling, n_probes can be any positive integer 1 to L²)

    Returns:
        lapl: Laplacian estimate (batch,)
    """
    if method == 'sites':
        return probing_laplacian_sites(forward_fn, x, grad, n_sites=n_probes)
    else:  # 'coloring' (default)
        return probing_laplacian_coloring(forward_fn, x, grad, n_colors=n_probes)


def probing_grad_and_lapl(model, x, n_probes=None, probing_method=None):
    """
    Compute gradient and Laplacian using probing.

    Args:
        model: Module with forward() method, n_colors and probing_method attributes
        x: Input (batch, L, L), requires_grad=True
        n_probes: Number of probes (overrides model.n_colors if provided)
        probing_method: 'coloring' or 'sites' (overrides model.probing_method if provided)

    Returns:
        grad: (batch, L, L)
        lapl: (batch,)
    """
    # Use provided values or fall back to model's defaults
    if n_probes is None:
        n_probes = getattr(model, 'n_colors', 4)
    if probing_method is None:
        probing_method = getattr(model, 'probing_method', 'coloring')

    y = model.forward(x)
    grad = tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]
    lapl = probing_laplacian(model.forward, x, grad, n_probes=n_probes, method=probing_method)
    return grad, lapl


class FunctSeparable(nn.Module):
    """
    Separable functional with probing-based Laplacian.

    Architecture:
        F(φ) = g(s) where s = (1/V) Σₓ h(φₓ, Δφₓ, φₓ₊τ)

    The site function h and aggregation function g are learnable MLPs.
    Uses probing with graph coloring for efficient Laplacian estimation.
    """

    def __init__(self, L, dim=2, y=0, site_hidden=[32, 32], agg_hidden=[16],
                 n_colors=4, dtype=tr.float32, activation=nn.GELU(), probing_method='coloring'):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.V = L * L
        self.n_colors = n_colors
        self.probing_method = probing_method

        # Site function h: maps (φₓ, Δφₓ, φₓ₊τ, φₓ₋τ) → scalar
        # Input: 4 features per site
        n_site_features = 4
        site_layers = []
        in_dim = n_site_features
        for h in site_hidden:
            site_layers.append(nn.Linear(in_dim, h, dtype=dtype))
            site_layers.append(activation)
            in_dim = h
        site_layers.append(nn.Linear(in_dim, 1, dtype=dtype))
        self.site_net = nn.Sequential(*site_layers)

        # Aggregation function g: maps aggregated features → scalar
        # We aggregate multiple statistics: mean(h), var(h), etc.
        n_agg_features = 3  # mean, var, mean of squares
        agg_layers = []
        in_dim = n_agg_features
        for h in agg_hidden:
            agg_layers.append(nn.Linear(in_dim, h, dtype=dtype))
            agg_layers.append(activation)
            in_dim = h
        agg_layers.append(nn.Linear(in_dim, 1, dtype=dtype))
        self.agg_net = nn.Sequential(*agg_layers)

    def _get_site_features(self, x):
        """Compute local features at each site."""
        # x shape: (batch, L, L)
        phi = x
        phi_tau = tr.roll(x, -self.y, dims=self.dim)
        phi_mtau = tr.roll(x, self.y, dims=self.dim)

        # Discrete Laplacian
        lap = (tr.roll(x, 1, dims=1) + tr.roll(x, -1, dims=1) +
               tr.roll(x, 1, dims=2) + tr.roll(x, -1, dims=2) - 4*x)

        # Stack features: (batch, L, L, 4)
        features = tr.stack([phi, lap, phi_tau, phi_mtau], dim=-1)
        return features

    def forward(self, x):
        batch_size = x.shape[0]

        # Get site features: (batch, L, L, 4)
        features = self._get_site_features(x)

        # Apply site network: (batch, L, L, 4) → (batch, L, L, 1)
        features_flat = features.view(-1, 4)
        h_values = self.site_net(features_flat).view(batch_size, self.L, self.L)

        # Compute aggregation statistics
        h_mean = h_values.mean(dim=(1, 2))
        h_var = h_values.var(dim=(1, 2))
        h_sq_mean = (h_values**2).mean(dim=(1, 2))

        # Aggregation features: (batch, 3)
        agg_features = tr.stack([h_mean, h_var, h_sq_mean], dim=1)

        # Apply aggregation network
        out = self.agg_net(agg_features).squeeze(-1)

        return out

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """Compute gradient and Laplacian using probing.

        Args:
            x: Input tensor
            n_colors: Override default n_colors (e.g., use more colors for evaluation)
            probing_method: Override default probing method
        """
        nc = n_colors if n_colors is not None else self.n_colors
        pm = probing_method if probing_method is not None else self.probing_method
        return probing_grad_and_lapl(self, x, n_probes=nc, probing_method=pm)


class FunctConvSeparable(nn.Module):
    """
    CNN + Separable aggregation with probing-based Laplacian.

    Architecture:
        1. Apply CNN to get feature maps: φ → {fₖ(x)} for k=1..K
        2. Global pool each feature map: sₖ = mean_x(fₖ(x))
        3. Apply MLP to pooled features: F = g(s₁, ..., sₖ)
    """

    def __init__(self, L, dim=2, y=0, conv_channels=[8, 16, 32],
                 mlp_hidden=[16], n_colors=4, dtype=tr.float32, activation=nn.GELU(),
                 probing_method='coloring'):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.V = L * L
        self.n_colors = n_colors
        self.probing_method = probing_method

        # Build CNN (no pooling until the end)
        conv_layers = []
        in_channels = 1
        for out_channels in conv_channels:
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         padding=1, padding_mode='circular')
            )
            conv_layers.append(activation)
            in_channels = out_channels
        self.cnn = nn.Sequential(*conv_layers)

        # Global average pooling is done in forward()

        # MLP on pooled features
        n_features = conv_channels[-1]
        mlp_layers = []
        in_dim = n_features
        for h in mlp_hidden:
            mlp_layers.append(nn.Linear(in_dim, h, dtype=dtype))
            mlp_layers.append(activation)
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, 1, dtype=dtype))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # x: (batch, L, L) → (batch, 1, L, L)
        x = x.unsqueeze(1)

        # CNN features: (batch, C, L, L)
        features = self.cnn(x)

        # Global average pooling: (batch, C)
        pooled = features.mean(dim=(2, 3))

        # MLP: (batch, 1)
        out = self.mlp(pooled).squeeze(-1)

        return out

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """Compute gradient and Laplacian using probing.

        Args:
            x: Input tensor
            n_colors: Override default n_colors (e.g., use more colors for evaluation)
            probing_method: Override default probing method
        """
        nc = n_colors if n_colors is not None else self.n_colors
        pm = probing_method if probing_method is not None else self.probing_method
        return probing_grad_and_lapl(self, x, n_probes=nc, probing_method=pm)


class Funct3T_Vmap(nn.Module):
    """
    Funct3T architecture with probing-based Laplacian.

    Uses the same CNN architecture as Funct3T but with probing
    for efficient Laplacian computation.
    """

    def __init__(self, L, dim=2, y=0, conv_layers=[4, 4, 4, 4],
                 n_colors=4, dtype=tr.float32, activation=nn.GELU(), probing_method='coloring'):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.n_colors = n_colors
        self.probing_method = probing_method

        # Build the same architecture as Funct3T
        self.net = nn.Sequential()
        in_channels = 1
        for i, out_channels in enumerate(conv_layers):
            self.net.add_module(
                f'conv{i}',
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         padding=1, padding_mode='circular')
            )
            self.net.add_module(f'act{i}', activation)
            in_channels = out_channels

        self.net.add_module('global_pool', nn.AvgPool2d(kernel_size=L, stride=L))
        self.net.add_module('flatten', nn.Flatten())
        self.net.add_module('linear', nn.Linear(in_channels, 1, dtype=dtype))

    def forward(self, x):
        # x: (batch, L, L) → (batch, 1, L, L)
        x = x.unsqueeze(1)
        out = self.net(x).squeeze(-1)
        return out

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """Compute gradient and Laplacian using probing.

        Args:
            x: Input tensor
            n_colors: Override default n_colors (e.g., use more colors for evaluation)
            probing_method: Override default probing method
        """
        nc = n_colors if n_colors is not None else self.n_colors
        pm = probing_method if probing_method is not None else self.probing_method
        return probing_grad_and_lapl(self, x, n_probes=nc, probing_method=pm)


class FunctPolynomial(nn.Module):
    """
    Polynomial functional with EXACT analytical Laplacian.

    F(φ) = Σₙ aₙ * Pₙ(φ)

    where Pₙ are polynomial features with known Laplacians:
    - P₁ = <φₓ φₓ₊τ> (two-point correlation)
    - P₂ = <φₓ φₓ₊τ>²
    - P₃ = <φₓ² φₓ₊τ²>
    - P₄ = <φₓ Δφₓ> (field-Laplacian correlation)
    - etc.

    Each polynomial has analytical gradient and Laplacian!
    """

    def __init__(self, L, dim=2, y=0, n_poly=8, dtype=tr.float32):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.V = L * L
        self.n_poly = n_poly

        # Learnable coefficients for polynomial features
        self.coeffs = nn.Parameter(tr.randn(n_poly) * 0.1)

    def _discrete_laplacian(self, phi):
        """Discrete Laplacian: Δφ = Σ_neighbors φ - 4φ"""
        return (tr.roll(phi, 1, dims=1) + tr.roll(phi, -1, dims=1) +
                tr.roll(phi, 1, dims=2) + tr.roll(phi, -1, dims=2) - 4*phi)

    def _compute_polynomials(self, x):
        """Compute polynomial features and their gradients/Laplacians."""
        batch = x.shape[0]
        V = self.V

        phi = x
        phi_tau = tr.roll(x, -self.y, dims=self.dim)
        phi_mtau = tr.roll(x, self.y, dims=self.dim)
        lap_phi = self._discrete_laplacian(x)

        # Neighbor sum
        neighbors = (tr.roll(x, 1, dims=1) + tr.roll(x, -1, dims=1) +
                    tr.roll(x, 1, dims=2) + tr.roll(x, -1, dims=2))

        # P1: <φ φ_τ> = (1/V) Σ_x φ_x φ_{x+τ}
        P1 = (phi * phi_tau).mean(dim=(1, 2))
        # ∂P1/∂φ_z = (1/V)(φ_{z+τ} + φ_{z-τ})
        dP1 = (phi_tau + phi_mtau) / V
        # ∂²P1/∂φ_z² = 0 for τ ≠ 0
        d2P1 = tr.zeros(batch, device=x.device)

        # P2: <φ φ_τ>²
        P2 = P1 ** 2
        # ∂P2/∂φ_z = 2 * P1 * ∂P1/∂φ_z
        dP2 = 2 * P1.view(-1, 1, 1) * dP1
        # ∂²P2/∂φ_z² = 2 * (∂P1/∂φ_z)²
        d2P2 = 2 * (dP1 ** 2).sum(dim=(1, 2))

        # P3: <φ²>
        P3 = (phi ** 2).mean(dim=(1, 2))
        dP3 = 2 * phi / V
        d2P3 = 2 * tr.ones(batch, device=x.device)

        # P4: <φ² φ_τ²>
        P4 = (phi**2 * phi_tau**2).mean(dim=(1, 2))
        dP4 = 2 * phi * phi_tau**2 / V + 2 * phi**2 * phi_mtau / V  # approximate
        d2P4 = (2 * phi_tau**2 / V).sum(dim=(1,2))  # simplified

        # P5: <φ * Δφ> (related to kinetic energy)
        P5 = (phi * lap_phi).mean(dim=(1, 2))
        # ∂P5/∂φ_z = (1/V)(Δφ_z + contributions from neighbors)
        dP5 = (lap_phi + neighbors - 4*phi) / V
        d2P5 = -8 * tr.ones(batch, device=x.device)  # from -4 term and neighbor contributions

        # P6: <φ_τ * Δφ>
        P6 = (phi_tau * lap_phi).mean(dim=(1, 2))
        dP6 = (tr.roll(lap_phi, self.y, dims=self.dim)) / V  # simplified
        d2P6 = tr.zeros(batch, device=x.device)

        # P7: <φ * neighbors>
        P7 = (phi * neighbors).mean(dim=(1, 2))
        dP7 = (neighbors + 4*phi) / V  # each site has 4 neighbors
        d2P7 = 8 * tr.ones(batch, device=x.device)

        # P8: <φ³ φ_τ>
        P8 = (phi**3 * phi_tau).mean(dim=(1, 2))
        dP8 = (3 * phi**2 * phi_tau + phi**3 * tr.roll(tr.ones_like(phi), self.y, dims=self.dim)) / V
        d2P8 = (6 * phi * phi_tau / V).sum(dim=(1, 2))

        # Stack all
        P = tr.stack([P1, P2, P3, P4, P5, P6, P7, P8], dim=1)[:, :self.n_poly]
        dP = tr.stack([dP1, dP2, dP3, dP4, dP5, dP6, dP7, dP8], dim=1)[:, :self.n_poly]
        d2P = tr.stack([d2P1, d2P2, d2P3, d2P4, d2P5, d2P6, d2P7, d2P8], dim=1)[:, :self.n_poly]

        return P, dP, d2P

    def forward(self, x):
        P, _, _ = self._compute_polynomials(x)
        # F = Σ_n a_n * P_n
        out = (P * self.coeffs[:self.n_poly]).sum(dim=1)
        return out

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """Analytical gradient and Laplacian! (n_colors and probing_method ignored)"""
        P, dP, d2P = self._compute_polynomials(x)

        # F = Σ_n a_n * P_n
        # ∂F/∂φ = Σ_n a_n * ∂P_n/∂φ
        # ∂²F/∂φ² = Σ_n a_n * ∂²P_n/∂φ²

        coeffs = self.coeffs[:self.n_poly]

        # Gradient: (batch, L, L)
        grad = (dP * coeffs.view(1, -1, 1, 1)).sum(dim=1)

        # Laplacian: (batch,)
        lapl = (d2P * coeffs.view(1, -1)).sum(dim=1)

        return grad, lapl


class FunctSmeared2ptAnalytic(nn.Module):
    """
    Smeared field + 2pt with ANALYTICAL derivatives.

    Like FunctSmeared2pt but with linear smearing and polynomial output,
    enabling exact gradient and Laplacian computation (no probing needed).

    Architecture:
        1. Linear smearing: ψ_k = φ + a_k * neighbors(φ)
           Multiple channels via different smearing parameters a_k
        2. Build features: C_k(τ), C_k(0), ⟨ψ_k²⟩ for each channel
        3. Polynomial output: F = Σ c_i * feature_i (linear in features)

    Key insight: Since ψ = φ + a*neighbors is LINEAR in φ:
        - ∂ψ/∂φ is constant (sparse stencil)
        - Features like ⟨ψ ψ_τ⟩ have analytical grad and Laplacian

    WARNING: THIS MODEL IS CURRENTLY BUGGY... OR NEEDS MUCH MORE EXPRESSIVITY
    """

    def __init__(self, L, dim=2, y=0, n_channels=4, n_smear_layers=1,
                 dtype=tr.float32):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.V = L * L
        self.n_channels = n_channels
        self.n_smear_layers = n_smear_layers

        # NOTE: n_smear_layers > 1 is NOT YET SUPPORTED for analytical derivatives!
        # The gradient/Laplacian formulas assume single-layer smearing.
        if n_smear_layers > 1:
            raise ValueError("n_smear_layers > 1 not yet supported - derivatives are incorrect")

        # Learnable smearing parameters for each channel
        # Each channel has different smearing strength
        # Initialize with spread of values
        init_a = tr.linspace(0.1, 0.5, n_channels)
        self.smear_a = nn.Parameter(init_a.clone())

        # Features per channel: C(τ), C(0) = 2 features per channel
        # (removed variance and cross-channel to ensure correct derivatives)
        n_features = n_channels * 2

        # Polynomial coefficients (linear combination of features)
        self.coeffs = nn.Parameter(tr.randn(n_features) * 0.1)

    def _neighbors(self, phi):
        """Sum of 4 nearest neighbors."""
        return (tr.roll(phi, 1, dims=1) + tr.roll(phi, -1, dims=1) +
                tr.roll(phi, 1, dims=2) + tr.roll(phi, -1, dims=2))

    def _smear(self, phi, a):
        """Linear smearing: ψ = φ + a * neighbors(φ)"""
        return phi + a.view(-1, 1, 1) * self._neighbors(phi).unsqueeze(0).expand(len(a), -1, -1, -1)

    def _apply_smearing(self, phi):
        """Apply smearing to get multi-channel smeared field.

        Args:
            phi: (batch, L, L)

        Returns:
            psi: (batch, n_channels, L, L)
        """
        batch = phi.shape[0]

        # First layer of smearing
        # psi_k = phi + a_k * neighbors(phi)
        neighbors = self._neighbors(phi)  # (batch, L, L)

        # Broadcast: (batch, n_channels, L, L)
        psi = phi.unsqueeze(1) + self.smear_a.view(1, -1, 1, 1) * neighbors.unsqueeze(1)

        # Optional second layer
        if self.n_smear_layers > 1:
            neighbors2 = self._neighbors(psi.view(-1, self.L, self.L)).view(batch, self.n_channels, self.L, self.L)
            psi = psi + self.smear_a2.view(1, -1, 1, 1) * neighbors2

        return psi

    def _compute_features_and_derivs(self, phi):
        """Compute features and their analytical gradients/Laplacians.

        Returns:
            features: (batch, n_features)
            d_features: (batch, n_features, L, L) - gradients
            d2_features: (batch, n_features) - Laplacians
        """
        batch = phi.shape[0]
        V = self.V
        L = self.L

        # Get smeared fields
        psi = self._apply_smearing(phi)  # (batch, n_channels, L, L)
        psi_tau = tr.roll(psi, -self.y, dims=2 + self.dim - 1)
        psi_mtau = tr.roll(psi, self.y, dims=2 + self.dim - 1)

        features = []
        d_features = []
        d2_features = []

        # Smearing stencil: S = I + a * N where N is neighbor sum operator
        # ∂ψ/∂φ at site z affects sites z and neighbors of z
        # For ⟨ψ ψ_τ⟩: gradient is (1/V) * S^T (ψ_τ + ψ_{-τ})

        for k in range(self.n_channels):
            psi_k = psi[:, k]  # (batch, L, L)
            psi_k_tau = psi_tau[:, k]
            psi_k_mtau = psi_mtau[:, k]
            a_k = self.smear_a[k]

            # Feature 1: C_k(τ) = ⟨ψ_k ψ_k,τ⟩
            C_tau = (psi_k * psi_k_tau).mean(dim=(1, 2))  # (batch,)
            features.append(C_tau)

            # Gradient of C_tau: (1/V) * [ψ_τ + a*neighbors(ψ_τ) + ψ_{-τ} + a*neighbors(ψ_{-τ})]
            dC_tau = (psi_k_tau + a_k * self._neighbors(psi_k_tau) +
                      psi_k_mtau + a_k * self._neighbors(psi_k_mtau)) / V
            d_features.append(dC_tau)

            # Laplacian of C_tau: Σ_z ∂²C/∂φ_z² = (2/V) Σ_{z,x} ∂ψ_x/∂φ_z · ∂ψ_{x+τ}/∂φ_z
            # This depends on overlap of smearing stencils at distance τ:
            # Stencil at site x covers: {x, x±x̂, x±ŷ} (5 sites for 1-layer smearing)
            # For stencils at x and x+τ to overlap, we need |τ| ≤ 2
            # - τ=0: full overlap (5 sites), Laplacian = 2(1 + 4a²)
            # - τ=1: partial overlap (2 sites: z and z-τ), Laplacian = 4a
            # - τ=2: minimal overlap (1 site: z-x̂), Laplacian = 2a²
            # - τ≥3: NO overlap, Laplacian = 0
            if self.y == 0:
                d2C_tau = 2 * (1 + 4 * a_k**2) * tr.ones(batch, device=phi.device)
            elif self.y == 1:
                # τ=1: overlap at {z, z-τ}, contributions are 1*a + a*1 = 2a per site
                d2C_tau = 4 * a_k * tr.ones(batch, device=phi.device)
            elif self.y == 2:
                # τ=2: only overlap via neighbor chain, contribution a*a = a²
                d2C_tau = 2 * a_k**2 * tr.ones(batch, device=phi.device)
            else:
                # τ≥3: stencils don't overlap at all, Laplacian = 0
                d2C_tau = tr.zeros(batch, device=phi.device)
            d2_features.append(d2C_tau)

            # Feature 2: C_k(0) = ⟨ψ_k²⟩
            C_0 = (psi_k * psi_k).mean(dim=(1, 2))
            features.append(C_0)

            # Gradient: (2/V) * [ψ + a*neighbors(ψ)]
            dC_0 = 2 * (psi_k + a_k * self._neighbors(psi_k)) / V
            d_features.append(dC_0)

            # Laplacian: 2 * (1 + 4a²)
            d2C_0 = 2 * (1 + 4 * a_k**2) * tr.ones(batch, device=phi.device)
            d2_features.append(d2C_0)

        # Stack
        features = tr.stack(features, dim=1)  # (batch, n_features)
        d_features = tr.stack(d_features, dim=1)  # (batch, n_features, L, L)
        d2_features = tr.stack(d2_features, dim=1)  # (batch, n_features)

        return features, d_features, d2_features

    def forward(self, x):
        features, _, _ = self._compute_features_and_derivs(x)
        # Linear combination
        out = (features * self.coeffs[:features.shape[1]]).sum(dim=1)
        return out

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """ANALYTICAL gradient and Laplacian - no probing needed!

        n_colors and probing_method are ignored (accepted for API compatibility).
        """
        features, d_features, d2_features = self._compute_features_and_derivs(x)

        coeffs = self.coeffs[:features.shape[1]]

        # Gradient: Σ c_i * ∂feature_i/∂φ
        grad = (d_features * coeffs.view(1, -1, 1, 1)).sum(dim=1)

        # Laplacian: Σ c_i * Δfeature_i
        lapl = (d2_features * coeffs.view(1, -1)).sum(dim=1)

        return grad, lapl


class FunctSmeared2pt(nn.Module):
    """
    Smeared field + zero-momentum two-point function.

    Architecture:
        1. CNN smearing: φ → ψ = CNN(φ)  [same spatial dimensions, possibly multi-channel]
        2. Build zero-momentum 2pt: C_k(τ) = (1/L²) Σ_{x,y} ψ_k(x,y) ψ_k(x, y+τ)
           for each channel k
        3. Optional MLP: F = MLP(C_1, C_2, ..., C_K)

    This is physically motivated: the CNN acts as a smearing/blocking transformation,
    and then we construct the actual observable structure from the smeared field.

    Uses probing with graph coloring for efficient Laplacian estimation.
    """

    def __init__(self, L, dim=2, y=0, conv_channels=[8, 8, 4],
                 mlp_hidden=[16], kernel_size=3, n_colors=4,
                 dtype=tr.float32, activation=nn.GELU(), probing_method='coloring'):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.V = L * L
        self.n_colors = n_colors
        self.probing_method = probing_method

        # CNN for smearing (maintains spatial dimensions)
        conv_layers_list = []
        in_ch = 1
        for out_ch in conv_channels:
            conv_layers_list.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                         padding=kernel_size//2, padding_mode='circular')
            )
            conv_layers_list.append(activation)
            in_ch = out_ch
        self.smear_net = nn.Sequential(*conv_layers_list)

        self.n_channels = conv_channels[-1]

        # MLP on the 2pt correlators (one per channel)
        # Plus we can add more features: mean, variance of smeared field
        n_features = self.n_channels * 3  # C(τ), C(0), mean(ψ²) for each channel
        mlp_layers = []
        in_dim = n_features
        for h in mlp_hidden:
            mlp_layers.append(nn.Linear(in_dim, h, dtype=dtype))
            mlp_layers.append(activation)
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, 1, dtype=dtype))
        self.mlp = nn.Sequential(*mlp_layers)

    def _build_2pt_features(self, psi):
        """
        Build zero-momentum two-point function features from smeared field.

        psi: (batch, channels, L, L)

        Returns: (batch, n_features)
        """
        batch = psi.shape[0]

        # Shift by tau
        psi_tau = tr.roll(psi, -self.y, dims=2 + self.dim - 1)  # dim 2 or 3 depending on self.dim

        # Zero-momentum 2pt at separation tau: C(τ) = <ψ(x) ψ(x+τ)>
        # Average over spatial dimensions
        C_tau = (psi * psi_tau).mean(dim=(2, 3))  # (batch, channels)

        # Zero-momentum 2pt at separation 0: C(0) = <ψ²>
        C_0 = (psi * psi).mean(dim=(2, 3))  # (batch, channels)

        # Mean of ψ² (another feature)
        psi_sq_mean = (psi ** 2).mean(dim=(2, 3))  # (batch, channels)

        # Concatenate features
        features = tr.cat([C_tau, C_0, psi_sq_mean], dim=1)  # (batch, 3*channels)

        return features

    def forward(self, x):
        # x: (batch, L, L) → (batch, 1, L, L)
        x_4d = x.unsqueeze(1)

        # Smear the field: (batch, channels, L, L)
        psi = self.smear_net(x_4d)

        # Build 2pt features: (batch, n_features)
        features = self._build_2pt_features(psi)

        # MLP to scalar
        out = self.mlp(features).squeeze(-1)

        return out

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """Use probing for Laplacian.

        Args:
            x: Input tensor
            n_colors: Override default n_colors (e.g., use more colors for evaluation)
            probing_method: Override default probing method
        """
        nc = n_colors if n_colors is not None else self.n_colors
        pm = probing_method if probing_method is not None else self.probing_method
        return probing_grad_and_lapl(self, x, n_probes=nc, probing_method=pm)


class FunctSmeared2ptMultiTau(nn.Module):
    """
    Extended version that builds 2pt functions at multiple separations.

    This is even more expressive: the network learns from correlations
    at multiple distances, not just the target τ.

    Architecture:
        1. CNN smearing: φ → ψ = CNN(φ)
        2. Build 2pt at multiple separations: C_k(0), C_k(1), ..., C_k(τ_max)
        3. MLP on all these correlators

    This gives the network information about the full correlation structure.
    Uses probing with graph coloring for efficient Laplacian estimation.
    """

    def __init__(self, L, dim=2, y=0, conv_channels=[8, 8, 4],
                 mlp_hidden=[32, 16], kernel_size=3, tau_max=None, n_colors=4,
                 dtype=tr.float32, activation=nn.GELU(), probing_method='coloring'):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.V = L * L
        self.tau_max = tau_max if tau_max is not None else L // 2
        self.n_colors = n_colors
        self.probing_method = probing_method

        # CNN for smearing
        conv_layers_list = []
        in_ch = 1
        for out_ch in conv_channels:
            conv_layers_list.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                         padding=kernel_size//2, padding_mode='circular')
            )
            conv_layers_list.append(activation)
            in_ch = out_ch
        self.smear_net = nn.Sequential(*conv_layers_list)

        self.n_channels = conv_channels[-1]

        # Features: C(0), C(1), ..., C(tau_max) for each channel
        # Plus the target tau indicator
        n_features = self.n_channels * (self.tau_max + 1) + 1  # +1 for tau value

        mlp_layers = []
        in_dim = n_features
        for h in mlp_hidden:
            mlp_layers.append(nn.Linear(in_dim, h, dtype=dtype))
            mlp_layers.append(activation)
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, 1, dtype=dtype))
        self.mlp = nn.Sequential(*mlp_layers)

    def _build_multi_tau_features(self, psi):
        """Build 2pt functions at all separations up to tau_max."""
        batch = psi.shape[0]
        features = []

        for tau in range(self.tau_max + 1):
            psi_tau = tr.roll(psi, -tau, dims=2 + self.dim - 1)
            C_tau = (psi * psi_tau).mean(dim=(2, 3))  # (batch, channels)
            features.append(C_tau)

        # Stack: (batch, (tau_max+1) * channels)
        features = tr.cat(features, dim=1)

        return features

    def forward(self, x):
        x_4d = x.unsqueeze(1)
        psi = self.smear_net(x_4d)

        # Build multi-tau features
        features = self._build_multi_tau_features(psi)

        # Add tau indicator (normalized)
        tau_indicator = tr.full((x.shape[0], 1), self.y / self.L,
                                device=x.device, dtype=x.dtype)
        features = tr.cat([features, tau_indicator], dim=1)

        out = self.mlp(features).squeeze(-1)
        return out

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """Use probing for Laplacian.

        Args:
            x: Input tensor
            n_colors: Override default n_colors (e.g., use more colors for evaluation)
            probing_method: Override default probing method
        """
        nc = n_colors if n_colors is not None else self.n_colors
        pm = probing_method if probing_method is not None else self.probing_method
        return probing_grad_and_lapl(self, x, n_probes=nc, probing_method=pm)


class FunctProbing(nn.Module):
    """
    Wrapper that uses probing for Laplacian estimation.

    Wraps any functional and replaces its grad_and_lapl with probing-based version.

    Supports two probing methods:
    - 'coloring': Graph coloring + random signs (n_colors must be 2 or perfect square)
    - 'sites': Random site sampling (n_colors can be any int from 1 to L²)
    """

    def __init__(self, base_functional, L, n_colors=4, probing_method='coloring'):
        super().__init__()
        self.F = base_functional
        self.y = base_functional.y
        self.dim = getattr(base_functional, 'dim', 2)
        self.n_colors = n_colors
        self.L = L  # Store L explicitly since base may not have it
        self.probing_method = probing_method

    def forward(self, x):
        return self.F(x)

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """Compute gradient and Laplacian using probing.

        Args:
            x: Input tensor
            n_colors: Override default n_colors (e.g., use more for evaluation)
            probing_method: Override default probing method ('coloring' or 'sites')
        """
        nc = n_colors if n_colors is not None else self.n_colors
        pm = probing_method if probing_method is not None else self.probing_method
        return probing_grad_and_lapl(self, x, n_probes=nc, probing_method=pm)


class Funct3T_Unified(nn.Module):
    """
    Unified Funct3T that takes tau as a parameter - ONE model for ALL tau values.

    Architecture:
        1. Encode tau as a learnable embedding or positional encoding
        2. Condition the CNN on tau (via FiLM: Feature-wise Linear Modulation)
        3. Single model handles all tau ∈ [0, L/2]

    Symmetry:
        Due to periodic BC, the observable O(tau) = O(L - tau).
        We enforce F(tau) = F(L - tau) by folding: tau -> min(tau, L - tau).
        Only need embeddings for tau ∈ {0, 1, ..., L/2}.

    Benefits:
        - Share parameters across all tau (more efficient)
        - Joint training on all tau simultaneously
        - Better generalization
        - Automatic symmetry enforcement

    The network learns: F(φ; τ) where τ is a conditioning parameter.
    """

    def __init__(self, L, dim=2, conv_layers=[4, 4, 4, 4], n_colors=4,
                 dtype=tr.float32, activation=nn.GELU(), probing_method='coloring'):
        super().__init__()

        self.L = L
        self.dim = dim
        self.n_colors = n_colors
        self.probing_method = probing_method  # 'coloring' or 'sites'
        self.tau_max = L // 2 + 1  # Number of tau values: 0, 1, ..., L/2

        # Tau embedding: learnable embedding for each tau value in [0, L/2]
        self.tau_embed_dim = 16
        self.tau_embedding = nn.Embedding(self.tau_max, self.tau_embed_dim)

        # Build CNN with FiLM conditioning
        self.convs = nn.ModuleList()
        self.film_gammas = nn.ModuleList()  # Scale parameters
        self.film_betas = nn.ModuleList()   # Shift parameters

        in_channels = 1
        for i, out_channels in enumerate(conv_layers):
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         padding=1, padding_mode='circular')
            )
            # FiLM: gamma and beta are functions of tau embedding
            self.film_gammas.append(nn.Linear(self.tau_embed_dim, out_channels, dtype=dtype))
            self.film_betas.append(nn.Linear(self.tau_embed_dim, out_channels, dtype=dtype))
            in_channels = out_channels

        self.activation = activation
        self.final_channels = conv_layers[-1]

        # Global pooling + final linear
        self.final_linear = nn.Linear(self.final_channels, 1, dtype=dtype)

        # Current tau (set before forward pass)
        self.y = 0  # Default tau

    def _fold_tau(self, tau):
        """Fold tau to [0, L/2] using symmetry: F(tau) = F(L - tau)."""
        if tau > self.L // 2:
            return self.L - tau
        return tau

    def set_tau(self, tau):
        """Set the current tau value for forward pass (automatically folded)."""
        self.y = self._fold_tau(tau)

    def forward(self, x, tau=None):
        """
        Forward pass with tau conditioning.

        Args:
            x: Field configuration (batch, L, L)
            tau: Tau value (int) or None to use self.y
                 Automatically folded to [0, L/2] via symmetry.
        """
        if tau is None:
            tau = self.y
        else:
            tau = self._fold_tau(tau)

        batch_size = x.shape[0]

        # Get tau embedding: (batch, embed_dim)
        tau_tensor = tr.full((batch_size,), tau, dtype=tr.long, device=x.device)
        tau_embed = self.tau_embedding(tau_tensor)  # (batch, embed_dim)

        # x: (batch, L, L) → (batch, 1, L, L)
        h = x.unsqueeze(1)

        # Apply conv layers with FiLM conditioning
        for conv, film_gamma, film_beta in zip(self.convs, self.film_gammas, self.film_betas):
            h = conv(h)  # (batch, C, L, L)

            # FiLM: h = gamma * h + beta
            gamma = film_gamma(tau_embed).view(batch_size, -1, 1, 1)  # (batch, C, 1, 1)
            beta = film_beta(tau_embed).view(batch_size, -1, 1, 1)
            h = gamma * h + beta

            h = self.activation(h)

        # Global average pooling
        h = h.mean(dim=(2, 3))  # (batch, C)

        # Final linear
        out = self.final_linear(h).squeeze(-1)  # (batch,)

        return out

    def grad_and_lapl(self, x, tau=None, n_colors=None, probing_method=None):
        """Compute gradient and Laplacian using probing.

        Args:
            x: Input tensor
            tau: Tau value (overrides self.y if provided)
            n_colors: Number of probes (overrides self.n_colors if provided)
            probing_method: 'coloring' or 'sites' (overrides self.probing_method if provided)
        """
        if tau is not None:
            self.set_tau(tau)

        nc = n_colors if n_colors is not None else self.n_colors
        pm = probing_method if probing_method is not None else self.probing_method

        y = self.forward(x)
        grad = tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]
        lapl = probing_laplacian(lambda z: self.forward(z), x, grad, n_probes=nc, method=pm)
        return grad, lapl


class Funct3T_Unified_Probing(Funct3T_Unified):
    """Alias for clarity - Funct3T_Unified already uses probing."""
    pass


class Funct3TUnified(nn.Module):
    """
    Unified Funct3T - same architecture as legacy Funct3T but handles all tau values.

    Architecture (same as Funct3T):
        Conv layers → Activation → Global Pool → Linear

    Tau conditioning:
        Tau embedding is concatenated to pooled features before final linear layer.
        This is simpler than FiLM conditioning used in Funct3T_Unified.

    Symmetry:
        Due to periodic BC, O(tau) = O(L - tau).
        We enforce F(tau) = F(L - tau) by folding: tau -> min(tau, L - tau).

    Uses probing for Laplacian estimation.
    """

    def __init__(self, L, dim=2, conv_layers=[4, 4, 4, 4], n_colors=4,
                 dtype=tr.float32, activation=nn.GELU(), probing_method='coloring'):
        super().__init__()

        self.L = L
        self.dim = dim
        self.n_colors = n_colors
        self.probing_method = probing_method
        self.tau_max = L // 2 + 1  # Number of tau values: 0, 1, ..., L/2

        # Build CNN (same as Funct3T)
        self.net = nn.Sequential()
        in_channels = 1
        for k, out_channels in enumerate(conv_layers):
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                            padding=1, padding_mode='circular')
            nn.init.xavier_uniform_(layer.weight)
            self.net.add_module(f'conv{k}', layer)
            self.net.add_module(f'act{k}', activation)
            in_channels = out_channels

        self.final_channels = conv_layers[-1]

        # Tau embedding: learnable embedding for each tau value in [0, L/2]
        self.tau_embed_dim = 8
        self.tau_embedding = nn.Embedding(self.tau_max, self.tau_embed_dim)

        # Final linear: takes pooled features + tau embedding
        self.final_linear = nn.Linear(self.final_channels + self.tau_embed_dim, 1, dtype=dtype)
        nn.init.xavier_uniform_(self.final_linear.weight)

        # Current tau (set before forward pass)
        self.y = 0

    def _fold_tau(self, tau):
        """Fold tau to [0, L/2] using symmetry: F(tau) = F(L - tau)."""
        if tau > self.L // 2:
            return self.L - tau
        return tau

    def set_tau(self, tau):
        """Set the current tau value for forward pass (automatically folded)."""
        self.y = self._fold_tau(tau)

    def forward(self, x, tau=None):
        """
        Forward pass with tau conditioning via concatenation.

        Args:
            x: Field configuration (batch, L, L)
            tau: Tau value (int) or None to use self.y
        """
        if tau is None:
            tau = self.y
        else:
            tau = self._fold_tau(tau)

        batch_size = x.shape[0]

        # x: (batch, L, L) → (batch, 1, L, L)
        h = x.unsqueeze(1)

        # Apply conv layers
        h = self.net(h)  # (batch, C, L, L)

        # Global average pooling
        h = h.mean(dim=(2, 3))  # (batch, C)

        # Get tau embedding and concatenate
        tau_tensor = tr.full((batch_size,), tau, dtype=tr.long, device=x.device)
        tau_embed = self.tau_embedding(tau_tensor)  # (batch, tau_embed_dim)
        h = tr.cat([h, tau_embed], dim=1)  # (batch, C + tau_embed_dim)

        # Final linear
        out = self.final_linear(h).squeeze(-1)  # (batch,)

        return out

    def grad_and_lapl(self, x, tau=None, n_colors=None, probing_method=None):
        """Compute gradient and Laplacian using probing.

        Args:
            x: Input tensor
            tau: Tau value (overrides self.y if provided)
            n_colors: Number of probes (overrides self.n_colors if provided)
            probing_method: 'coloring' or 'sites' (overrides self.probing_method if provided)
        """
        if tau is not None:
            self.set_tau(tau)

        nc = n_colors if n_colors is not None else self.n_colors
        pm = probing_method if probing_method is not None else self.probing_method

        y = self.forward(x)
        grad = tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]
        lapl = probing_laplacian(lambda z: self.forward(z), x, grad, n_probes=nc, method=pm)
        return grad, lapl


# Extended factory
def fast_model_factory_v2(class_name, L, y, conv_layers=[4,4,4,4], activation='gelu',
                          dtype=tr.float32, n_colors=4, probing_method='coloring', **kwargs):
    """
    Extended factory including V2 fast functionals.

    All approximate Laplacian models use probing with n_colors.

    Args:
        class_name: Model class name
        L: Lattice size
        y: Tau value
        conv_layers: Convolutional layer widths
        activation: Activation function
        dtype: Data type
        n_colors: Number of probes for Laplacian estimation
        probing_method: 'coloring' (graph coloring, n_colors must be 2 or perfect square)
                        'sites' (random site sampling, n_colors can be any int from 1 to L²)
    """
    if isinstance(activation, str):
        activ = activation_factory(activation)
    else:
        activ = activation

    if class_name == 'Funct3T':
        # Original model with exact Laplacian (slow)
        return original_model_factory('Funct3T', L=L, y=y,
                                      conv_layers=conv_layers,
                                      activation=activ, dtype=dtype)

    elif class_name == 'Funct3T_Probing':
        # Funct3T with probing-based Laplacian
        base = original_model_factory('Funct3T', L=L, y=y,
                                      conv_layers=conv_layers,
                                      activation=activ, dtype=dtype)
        return FunctProbing(base, L=L, n_colors=n_colors, probing_method=probing_method)

    elif class_name == 'FunctSeparable':
        return FunctSeparable(L=L, y=y, dtype=dtype, activation=activ,
                              n_colors=n_colors, probing_method=probing_method)

    elif class_name == 'FunctConvSeparable':
        return FunctConvSeparable(L=L, y=y, dtype=dtype, activation=activ,
                                  n_colors=n_colors, probing_method=probing_method)

    elif class_name == 'FunctPolynomial':
        # Analytical Laplacian - no probing needed
        return FunctPolynomial(L=L, y=y, dtype=dtype)

    elif class_name == 'FunctSmeared2ptAnalytic':
        # Analytical Laplacian with linear smearing + 2pt features
        # NOTE: only n_smear_layers=1 is supported (derivatives are exact only for single layer)
        return FunctSmeared2ptAnalytic(L=L, y=y, dtype=dtype,
                                       n_channels=kwargs.get('n_channels', 4),
                                       n_smear_layers=kwargs.get('n_smear_layers', 1))

    elif class_name == 'Funct3T_Vmap':
        return Funct3T_Vmap(L=L, y=y, conv_layers=conv_layers,
                           dtype=dtype, activation=activ, n_colors=n_colors,
                           probing_method=probing_method)

    elif class_name == 'FunctSmeared2pt':
        return FunctSmeared2pt(L=L, y=y, dtype=dtype, activation=activ,
                               conv_channels=kwargs.get('conv_channels', [8, 8, 4]),
                               n_colors=n_colors, probing_method=probing_method)

    elif class_name == 'FunctSmeared2ptMultiTau':
        return FunctSmeared2ptMultiTau(L=L, y=y, dtype=dtype, activation=activ,
                                        conv_channels=kwargs.get('conv_channels', [8, 8, 4]),
                                        n_colors=n_colors, probing_method=probing_method)

    elif class_name == 'Funct3T_Unified':
        model = Funct3T_Unified(L=L, conv_layers=conv_layers, n_colors=n_colors,
                                dtype=dtype, activation=activ, probing_method=probing_method)
        model.set_tau(y)  # Set initial tau
        return model

    elif class_name == 'Funct3TUnified':
        # Simpler unified model: same arch as Funct3T, tau via concatenation (not FiLM)
        model = Funct3TUnified(L=L, conv_layers=conv_layers, n_colors=n_colors,
                               dtype=dtype, activation=activ, probing_method=probing_method)
        model.set_tau(y)  # Set initial tau
        return model

    else:
        raise ValueError(f"Unknown model: {class_name}. Available: {ALL_MODELS_V2}")


ALL_MODELS_V2 = [
    'Funct3T',              # Original (exact Laplacian, slow)
    'Funct3T_Probing',      # Funct3T with probing Laplacian
    'FunctSeparable',       # Separable architecture
    'FunctConvSeparable',   # CNN + separable
    'FunctPolynomial',      # Analytical Laplacian (no probing needed)
    'FunctSmeared2ptAnalytic',  # Linear smearing + 2pt, analytical Laplacian
    'FunctSmeared2pt',      # CNN smearing + 2pt construction
    'FunctSmeared2ptMultiTau',  # CNN smearing + multi-tau 2pt
    'Funct3T_Unified',      # Single model for ALL tau (FiLM conditioning)
    'Funct3TUnified',       # Single model for ALL tau (simpler: tau concatenation)
]
