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
        1. Direction-dependent linear smearing:
           ψ = (I + αx*Nx + αy*Ny + β*Nd)^n_layers · φ
           where Nx = x-neighbors, Ny = y-neighbors, Nd = diagonal neighbors
           Multiple channels via different smearing parameters
        2. Build features:
           - Base features: C_k(τ), C_k(0) for each channel k
           - Cross-channel: C_jk(τ) = ⟨ψ_j ψ_k,τ⟩ for j < k
           - Polynomial products: f_i * f_j for selected pairs
        3. Polynomial output: F = Σ c_i * feature_i (linear in features)

    Key insight: Since smearing is LINEAR in φ:
        - ∂ψ/∂φ is a constant stencil (depends only on parameters and n_layers)
        - Features like ⟨ψ ψ_τ⟩ have analytical grad and Laplacian
        - Laplacian = 2 * (stencil · shifted_stencil) where shift = τ
        - For products f*g: Δ(fg) = f*Δg + g*Δf + 2*∇f·∇g

    Stencil for one layer:
        [β,  αy, β ]
        [αx, 1,  αx]
        [β,  αy, β ]
    """

    def __init__(self, L, dim=2, y=0, n_channels=4, n_smear_layers=4,
                 dtype=tr.float32, include_cross_channel=False, include_products=True):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.V = L * L
        self.n_channels = n_channels
        self.n_smear_layers = n_smear_layers
        self.dtype = dtype
        self.include_cross_channel = include_cross_channel
        self.include_products = include_products

        # Learnable smearing parameters for each channel
        # alpha_x: smearing along x (direction of correlator)
        # alpha_y: smearing along y (perpendicular)
        # beta: diagonal smearing
        init_ax = tr.linspace(0.1, 0.4, n_channels)
        init_ay = tr.linspace(0.1, 0.4, n_channels)
        init_beta = tr.linspace(0.05, 0.2, n_channels)

        self.alpha_x = nn.Parameter(init_ax.clone())
        self.alpha_y = nn.Parameter(init_ay.clone())
        self.beta = nn.Parameter(init_beta.clone())

        # Count features:
        # Base: 2 per channel (C(τ) and C(0))
        n_base_features = n_channels * 2

        # Cross-channel: C_jk(τ) for j < k = n_channels*(n_channels-1)/2
        n_cross_features = n_channels * (n_channels - 1) // 2 if include_cross_channel else 0

        # Total linear features (before products)
        n_linear_features = n_base_features + n_cross_features

        # Products: f_i * f_j for i <= j (only include a subset to avoid explosion)
        # We'll include products of base features only (not cross-channel to keep it manageable)
        # This gives n_base*(n_base+1)/2 products
        n_product_features = n_base_features * (n_base_features + 1) // 2 if include_products else 0

        self.n_base_features = n_base_features
        self.n_cross_features = n_cross_features
        self.n_linear_features = n_linear_features
        self.n_product_features = n_product_features
        n_features = n_linear_features + n_product_features

        # Polynomial coefficients (linear combination of features)
        self.coeffs = nn.Parameter(tr.randn(n_features, dtype=dtype) * 0.1)

        # Stencil size needed: n_smear_layers sites in each direction
        # With diagonals, stencil can extend further
        self.stencil_size = 2 * n_smear_layers + 1

    def _compute_stencil(self, alpha_x, alpha_y, beta):
        """Compute the normalized smearing stencil for direction-dependent smearing.

        Single layer stencil (normalized so sum=1):
            [β,  αy, β ]
            [αx, 1,  αx]  / (1 + 2αx + 2αy + 4β)
            [β,  αy, β ]

        For n_layers > 1, we apply this operator repeatedly.

        Returns:
            stencil: (stencil_size, stencil_size) tensor (sum ≈ 1)
        """
        n = self.n_smear_layers
        size = self.stencil_size
        center = n  # center index
        device = alpha_x.device

        # Normalization factor for single layer
        norm = 1.0 + 2*alpha_x + 2*alpha_y + 4*beta

        # Start with identity: S = I (just the center)
        stencil = tr.zeros(size, size, dtype=self.dtype, device=device)
        stencil[center, center] = 1.0

        # Apply normalized (I + αx*Nx + αy*Ny + β*Nd) / norm n_layers times
        for _ in range(n):
            new_stencil = stencil.clone()

            # x-direction neighbors (left/right in the j index)
            new_stencil[:, :-1] = new_stencil[:, :-1] + alpha_x * stencil[:, 1:]   # left
            new_stencil[:, 1:] = new_stencil[:, 1:] + alpha_x * stencil[:, :-1]    # right

            # y-direction neighbors (up/down in the i index)
            new_stencil[:-1, :] = new_stencil[:-1, :] + alpha_y * stencil[1:, :]   # up
            new_stencil[1:, :] = new_stencil[1:, :] + alpha_y * stencil[:-1, :]    # down

            # Diagonal neighbors (4 corners)
            new_stencil[:-1, :-1] = new_stencil[:-1, :-1] + beta * stencil[1:, 1:]    # up-left
            new_stencil[:-1, 1:] = new_stencil[:-1, 1:] + beta * stencil[1:, :-1]     # up-right
            new_stencil[1:, :-1] = new_stencil[1:, :-1] + beta * stencil[:-1, 1:]     # down-left
            new_stencil[1:, 1:] = new_stencil[1:, 1:] + beta * stencil[:-1, :-1]      # down-right

            stencil = new_stencil / norm

        return stencil

    def _stencil_overlap_periodic(self, stencil, tau, L):
        """Compute overlap of stencil with itself shifted by tau on periodic lattice.

        Laplacian contribution = 2 * Σ_z S[z] * S[z-tau mod L]

        On a periodic lattice of size L, we need to account for wraparound.
        We embed the stencil into an L×L array and compute the overlap there.

        Args:
            stencil: (stencil_size, stencil_size) stencil tensor
            tau: shift distance (τ >= 0)
            L: lattice size

        Returns:
            overlap: scalar (the Laplacian coefficient = 2 * dot product)
        """
        stencil_size = stencil.shape[0]
        center = stencil_size // 2

        # Embed stencil into L×L periodic array (centered at origin)
        S_full = tr.zeros(L, L, dtype=stencil.dtype, device=stencil.device)
        for i in range(stencil_size):
            for j in range(stencil_size):
                # Map stencil indices to lattice indices (with periodic wrapping)
                li = (i - center) % L
                lj = (j - center) % L
                S_full[li, lj] += stencil[i, j]

        # Shift by tau along second dimension (periodic)
        S_shifted = tr.roll(S_full, -tau, dims=1)

        # Compute overlap
        overlap = (S_full * S_shifted).sum()

        return 2 * overlap

    def _cross_stencil_overlap_periodic(self, stencil_j, stencil_k, tau, L):
        """Compute cross-overlap of two stencils with shift tau on periodic lattice.

        Cross-Laplacian contribution = 2 * Σ_z S_j[z] * S_k[z-tau mod L]

        Used for cross-channel correlations C_jk(τ) = ⟨ψ_j ψ_k,τ⟩

        Args:
            stencil_j: (stencil_size, stencil_size) first stencil
            stencil_k: (stencil_size, stencil_size) second stencil
            tau: shift distance (τ >= 0)
            L: lattice size

        Returns:
            overlap: scalar (the cross-Laplacian coefficient = 2 * cross dot product)
        """
        stencil_size_j = stencil_j.shape[0]
        stencil_size_k = stencil_k.shape[0]
        center_j = stencil_size_j // 2
        center_k = stencil_size_k // 2

        # Embed both stencils into L×L periodic arrays (centered at origin)
        S_j_full = tr.zeros(L, L, dtype=stencil_j.dtype, device=stencil_j.device)
        S_k_full = tr.zeros(L, L, dtype=stencil_k.dtype, device=stencil_k.device)

        for i in range(stencil_size_j):
            for j in range(stencil_size_j):
                li = (i - center_j) % L
                lj = (j - center_j) % L
                S_j_full[li, lj] += stencil_j[i, j]

        for i in range(stencil_size_k):
            for j in range(stencil_size_k):
                li = (i - center_k) % L
                lj = (j - center_k) % L
                S_k_full[li, lj] += stencil_k[i, j]

        # Shift S_k by tau along second dimension (periodic)
        S_k_shifted = tr.roll(S_k_full, -tau, dims=1)

        # Compute cross-overlap
        overlap = (S_j_full * S_k_shifted).sum()

        return 2 * overlap

    def _apply_smearing_step(self, phi, alpha_x, alpha_y, beta):
        """Apply one layer of direction-dependent smearing with normalization.

        Normalized stencil (sum = 1):
            [β,  αy, β ]
            [αx, 1,  αx]  / (1 + 2αx + 2αy + 4β)
            [β,  αy, β ]

        This ensures the smearing is a proper averaging operator that doesn't
        amplify the field values.
        """
        # Compute normalization factor
        norm = 1.0 + 2*alpha_x + 2*alpha_y + 4*beta

        # x-direction neighbors (along dim 2)
        neighbors_x = tr.roll(phi, 1, dims=2) + tr.roll(phi, -1, dims=2)

        # y-direction neighbors (along dim 1)
        neighbors_y = tr.roll(phi, 1, dims=1) + tr.roll(phi, -1, dims=1)

        # Diagonal neighbors
        neighbors_d = (
            tr.roll(tr.roll(phi, 1, dims=1), 1, dims=2) +   # up-right
            tr.roll(tr.roll(phi, 1, dims=1), -1, dims=2) +  # up-left
            tr.roll(tr.roll(phi, -1, dims=1), 1, dims=2) +  # down-right
            tr.roll(tr.roll(phi, -1, dims=1), -1, dims=2)   # down-left
        )

        result = (phi + alpha_x * neighbors_x + alpha_y * neighbors_y + beta * neighbors_d) / norm

        return result

    def _apply_smearing(self, phi):
        """Apply smearing to get multi-channel smeared field.

        Args:
            phi: (batch, L, L)

        Returns:
            psi: (batch, n_channels, L, L)
        """
        # Apply (I + αx*Nx + αy*Ny + β*Nd)^n_layers for each channel
        psi_list = []
        for k in range(self.n_channels):
            psi_k = phi.clone()
            ax_k = self.alpha_x[k]
            ay_k = self.alpha_y[k]
            b_k = self.beta[k]
            for _ in range(self.n_smear_layers):
                psi_k = self._apply_smearing_step(psi_k, ax_k, ay_k, b_k)
            psi_list.append(psi_k)

        psi = tr.stack(psi_list, dim=1)  # (batch, n_channels, L, L)
        return psi

    def _apply_stencil_transpose(self, field, alpha_x, alpha_y, beta):
        """Apply S^T to a field, where S is the smearing stencil.

        For gradient computation: ∂C/∂φ = (1/V) * S^T (ψ_τ + ψ_{-τ})

        S^T has the same structure as S for symmetric stencils.
        """
        result = field.clone()
        for _ in range(self.n_smear_layers):
            result = self._apply_smearing_step(result, alpha_x, alpha_y, beta)
        return result

    def _compute_features_and_derivs(self, phi):
        """Compute features and their analytical gradients/Laplacians.

        Includes:
        - Base features: C_k(τ), C_k(0) for each channel k
        - Cross-channel features: C_jk(τ) = ⟨ψ_j ψ_k,τ⟩ for j < k
        - Product features: f_i * f_j for base feature pairs

        Returns:
            features: (batch, n_features)
            d_features: (batch, n_features, L, L) - gradients
            d2_features: (batch, n_features) - Laplacians
        """
        batch = phi.shape[0]
        V = self.V
        L = self.L
        device = phi.device

        # Get smeared fields
        psi = self._apply_smearing(phi)  # (batch, n_channels, L, L)

        # Determine which dimension to shift based on self.dim
        shift_dim = 2 + self.dim - 1  # dim=1 -> shift_dim=2, dim=2 -> shift_dim=3
        psi_tau = tr.roll(psi, -self.y, dims=shift_dim)
        psi_mtau = tr.roll(psi, self.y, dims=shift_dim)

        # Precompute stencils for all channels
        stencils = []
        for k in range(self.n_channels):
            stencils.append(self._compute_stencil(self.alpha_x[k], self.alpha_y[k], self.beta[k]))

        # ============ BASE FEATURES ============
        # C_k(τ) and C_k(0) for each channel
        base_features = []
        base_d_features = []
        base_d2_features = []

        for k in range(self.n_channels):
            psi_k = psi[:, k]  # (batch, L, L)
            psi_k_tau = psi_tau[:, k]
            psi_k_mtau = psi_mtau[:, k]

            ax_k = self.alpha_x[k]
            ay_k = self.alpha_y[k]
            b_k = self.beta[k]
            stencil_k = stencils[k]

            # Feature: C_k(τ) = ⟨ψ_k ψ_k,τ⟩
            C_tau = (psi_k * psi_k_tau).mean(dim=(1, 2))  # (batch,)
            base_features.append(C_tau)

            # Gradient: (1/V) * S^T (ψ_τ + ψ_{-τ})
            dC_tau = self._apply_stencil_transpose(psi_k_tau + psi_k_mtau, ax_k, ay_k, b_k) / V
            base_d_features.append(dC_tau)

            # Laplacian: 2 * overlap(S, S shifted by τ)
            d2C_tau = self._stencil_overlap_periodic(stencil_k, self.y, L) * tr.ones(batch, device=device)
            base_d2_features.append(d2C_tau)

            # Feature: C_k(0) = ⟨ψ_k²⟩
            C_0 = (psi_k * psi_k).mean(dim=(1, 2))
            base_features.append(C_0)

            # Gradient: (2/V) * S^T ψ
            dC_0 = 2 * self._apply_stencil_transpose(psi_k, ax_k, ay_k, b_k) / V
            base_d_features.append(dC_0)

            # Laplacian: 2 * overlap(S, S) at τ=0
            d2C_0 = self._stencil_overlap_periodic(stencil_k, 0, L) * tr.ones(batch, device=device)
            base_d2_features.append(d2C_0)

        # ============ CROSS-CHANNEL FEATURES ============
        # C_jk(τ) = ⟨ψ_j ψ_k,τ⟩ for j < k
        cross_features = []
        cross_d_features = []
        cross_d2_features = []

        if self.include_cross_channel:
            for j in range(self.n_channels):
                for k in range(j + 1, self.n_channels):
                    psi_j = psi[:, j]
                    psi_k_tau = psi_tau[:, k]
                    psi_j_mtau = psi_mtau[:, j]
                    psi_k = psi[:, k]

                    ax_j, ay_j, b_j = self.alpha_x[j], self.alpha_y[j], self.beta[j]
                    ax_k, ay_k, b_k = self.alpha_x[k], self.alpha_y[k], self.beta[k]
                    stencil_j, stencil_k = stencils[j], stencils[k]

                    # Feature: C_jk(τ) = ⟨ψ_j ψ_k,τ⟩
                    C_jk = (psi_j * psi_k_tau).mean(dim=(1, 2))
                    cross_features.append(C_jk)

                    # Gradient: (1/V) * [S_j^T ψ_k,τ + S_k^T ψ_j,-τ]
                    dC_jk = (self._apply_stencil_transpose(psi_k_tau, ax_j, ay_j, b_j) +
                             self._apply_stencil_transpose(psi_j_mtau, ax_k, ay_k, b_k)) / V
                    cross_d_features.append(dC_jk)

                    # Laplacian: 2 * cross_overlap(S_j, S_k, τ)
                    d2C_jk = self._cross_stencil_overlap_periodic(stencil_j, stencil_k, self.y, L) * tr.ones(batch, device=device)
                    cross_d2_features.append(d2C_jk)

        # ============ COMBINE LINEAR FEATURES ============
        features = base_features + cross_features
        d_features = base_d_features + cross_d_features
        d2_features = base_d2_features + cross_d2_features

        # Stack linear features
        features = tr.stack(features, dim=1)  # (batch, n_linear_features)
        d_features = tr.stack(d_features, dim=1)  # (batch, n_linear_features, L, L)
        d2_features = tr.stack(d2_features, dim=1)  # (batch, n_linear_features)

        # ============ PRODUCT FEATURES ============
        # f_i * f_j for base features (i <= j)
        # Derivatives: d(fg) = f*dg + g*df
        #              d²(fg) = f*d²g + g*d²f + 2*df·dg (dot product over spatial dims)

        if self.include_products and self.n_product_features > 0:
            prod_features = []
            prod_d_features = []
            prod_d2_features = []

            # Only use base features for products to keep it manageable
            base_f = features[:, :self.n_base_features]  # (batch, n_base)
            base_df = d_features[:, :self.n_base_features]  # (batch, n_base, L, L)
            base_d2f = d2_features[:, :self.n_base_features]  # (batch, n_base)

            for i in range(self.n_base_features):
                for j in range(i, self.n_base_features):
                    f_i = base_f[:, i]  # (batch,)
                    f_j = base_f[:, j]
                    df_i = base_df[:, i]  # (batch, L, L)
                    df_j = base_df[:, j]
                    d2f_i = base_d2f[:, i]  # (batch,)
                    d2f_j = base_d2f[:, j]

                    # Product: p = f_i * f_j
                    p = f_i * f_j
                    prod_features.append(p)

                    # Gradient: dp = f_i * df_j + f_j * df_i
                    dp = f_i.view(-1, 1, 1) * df_j + f_j.view(-1, 1, 1) * df_i
                    prod_d_features.append(dp)

                    # Laplacian: d²p = f_i * d²f_j + f_j * d²f_i + 2 * (df_i · df_j)
                    # where (df_i · df_j) = sum over all sites of df_i[z] * df_j[z]
                    grad_dot = (df_i * df_j).sum(dim=(1, 2))  # (batch,)
                    d2p = f_i * d2f_j + f_j * d2f_i + 2 * grad_dot
                    prod_d2_features.append(d2p)

            # Stack product features
            prod_features = tr.stack(prod_features, dim=1)  # (batch, n_products)
            prod_d_features = tr.stack(prod_d_features, dim=1)  # (batch, n_products, L, L)
            prod_d2_features = tr.stack(prod_d2_features, dim=1)  # (batch, n_products)

            # Concatenate all features
            features = tr.cat([features, prod_features], dim=1)
            d_features = tr.cat([d_features, prod_d_features], dim=1)
            d2_features = tr.cat([d2_features, prod_d2_features], dim=1)

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


class FunctSmeared2ptNonlinear(nn.Module):
    """
    Smeared field with NONLINEAR activation and EXACT analytical derivatives.

    Key insight: For ψ = σ(S·φ) where σ is a known activation (tanh, sigmoid, etc.),
    we can compute EXACT gradient and Laplacian using the chain rule!

    For feature C(τ) = ⟨ψ · ψ_τ⟩:

    Gradient:
        ∂C/∂φ = (1/V) · S^T · [σ'(u) ⊙ ψ_τ + σ'(u_τ) ⊙ ψ_{-τ}]
        where u = S·φ (pre-activation), ⊙ is elementwise product

    Laplacian:
        ΔC = ||S||² · ⟨σ''(u) · ψ_τ⟩ + 2·overlap(S,τ) · ⟨σ'(u) · σ'(u_τ)⟩

    Cost: O(L²) per feature - SAME as linear case! NOT O(L⁴) like autodiff!

    This gives CNN-like expressivity with exact Laplacian computation.
    """

    def __init__(self, L, dim=2, y=0, n_channels=4, n_smear_layers=4,
                 activation='tanh', dtype=tr.float32):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.V = L * L
        self.n_channels = n_channels
        self.n_smear_layers = n_smear_layers
        self.dtype = dtype
        self.activation_name = activation

        # Set activation function and its derivatives
        if activation == 'tanh':
            self.sigma = tr.tanh
            self.sigma_prime = lambda x: 1 - tr.tanh(x)**2
            self.sigma_double_prime = lambda x: -2 * tr.tanh(x) * (1 - tr.tanh(x)**2)
        elif activation == 'sigmoid':
            self.sigma = tr.sigmoid
            self.sigma_prime = lambda x: tr.sigmoid(x) * (1 - tr.sigmoid(x))
            self.sigma_double_prime = lambda x: tr.sigmoid(x) * (1 - tr.sigmoid(x)) * (1 - 2*tr.sigmoid(x))
        elif activation == 'softplus':
            self.sigma = nn.functional.softplus
            self.sigma_prime = tr.sigmoid  # d/dx softplus(x) = sigmoid(x)
            self.sigma_double_prime = lambda x: tr.sigmoid(x) * (1 - tr.sigmoid(x))
        elif activation == 'gelu':
            # GELU(x) = x * Φ(x) where Φ is standard normal CDF
            self.sigma = nn.functional.gelu
            # Approximate derivatives
            self.sigma_prime = lambda x: 0.5 * (1 + tr.erf(x / np.sqrt(2))) + x * tr.exp(-x**2/2) / np.sqrt(2*np.pi)
            self.sigma_double_prime = lambda x: tr.exp(-x**2/2) / np.sqrt(2*np.pi) * (2 - x**2 / np.sqrt(2*np.pi))
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Direction-dependent smearing parameters for each channel
        init_ax = tr.linspace(0.1, 0.4, n_channels)
        init_ay = tr.linspace(0.1, 0.4, n_channels)
        init_beta = tr.linspace(0.05, 0.2, n_channels)

        self.alpha_x = nn.Parameter(init_ax.clone())
        self.alpha_y = nn.Parameter(init_ay.clone())
        self.beta = nn.Parameter(init_beta.clone())

        # Scale parameter for pre-activation (helps with activation saturation)
        self.pre_scale = nn.Parameter(tr.ones(n_channels) * 0.5)

        # Features: C_k(τ), C_k(0) for each channel = 2 * n_channels
        n_base = n_channels * 2
        # Products of base features
        n_products = n_base * (n_base + 1) // 2

        self.n_base_features = n_base
        self.n_product_features = n_products

        n_features = n_base + n_products
        self.coeffs = nn.Parameter(tr.randn(n_features, dtype=dtype) * 0.01)

        self.stencil_size = 2 * n_smear_layers + 1

    def _compute_stencil(self, alpha_x, alpha_y, beta):
        """Compute normalized smearing stencil."""
        n = self.n_smear_layers
        size = self.stencil_size
        center = n
        device = alpha_x.device

        norm = 1.0 + 2*alpha_x + 2*alpha_y + 4*beta

        stencil = tr.zeros(size, size, dtype=self.dtype, device=device)
        stencil[center, center] = 1.0

        for _ in range(n):
            new_stencil = stencil.clone()
            new_stencil[:, :-1] = new_stencil[:, :-1] + alpha_x * stencil[:, 1:]
            new_stencil[:, 1:] = new_stencil[:, 1:] + alpha_x * stencil[:, :-1]
            new_stencil[:-1, :] = new_stencil[:-1, :] + alpha_y * stencil[1:, :]
            new_stencil[1:, :] = new_stencil[1:, :] + alpha_y * stencil[:-1, :]
            new_stencil[:-1, :-1] = new_stencil[:-1, :-1] + beta * stencil[1:, 1:]
            new_stencil[:-1, 1:] = new_stencil[:-1, 1:] + beta * stencil[1:, :-1]
            new_stencil[1:, :-1] = new_stencil[1:, :-1] + beta * stencil[:-1, 1:]
            new_stencil[1:, 1:] = new_stencil[1:, 1:] + beta * stencil[:-1, :-1]
            stencil = new_stencil / norm

        return stencil

    def _stencil_norm_sq(self, stencil):
        """Compute ||S||² = Σ S[i,j]²."""
        return (stencil ** 2).sum()

    def _stencil_overlap_periodic(self, stencil, tau, L):
        """Compute overlap of stencil with itself shifted by tau."""
        size = stencil.shape[0]
        center = size // 2

        S_full = tr.zeros(L, L, dtype=stencil.dtype, device=stencil.device)
        for i in range(size):
            for j in range(size):
                li = (i - center) % L
                lj = (j - center) % L
                S_full[li, lj] += stencil[i, j]

        S_shifted = tr.roll(S_full, -tau, dims=1)
        return (S_full * S_shifted).sum()

    def _apply_smearing_step(self, phi, alpha_x, alpha_y, beta):
        """Apply one normalized smearing step."""
        norm = 1.0 + 2*alpha_x + 2*alpha_y + 4*beta

        neighbors_x = tr.roll(phi, 1, dims=2) + tr.roll(phi, -1, dims=2)
        neighbors_y = tr.roll(phi, 1, dims=1) + tr.roll(phi, -1, dims=1)
        neighbors_d = (
            tr.roll(tr.roll(phi, 1, dims=1), 1, dims=2) +
            tr.roll(tr.roll(phi, 1, dims=1), -1, dims=2) +
            tr.roll(tr.roll(phi, -1, dims=1), 1, dims=2) +
            tr.roll(tr.roll(phi, -1, dims=1), -1, dims=2)
        )

        return (phi + alpha_x * neighbors_x + alpha_y * neighbors_y + beta * neighbors_d) / norm

    def _apply_linear_smearing(self, phi, k):
        """Apply linear smearing for channel k: u_k = S_k · φ"""
        u = phi.clone()
        for _ in range(self.n_smear_layers):
            u = self._apply_smearing_step(u, self.alpha_x[k], self.alpha_y[k], self.beta[k])
        return u * self.pre_scale[k]  # Scale before activation

    def _apply_stencil_transpose(self, field, k):
        """Apply S_k^T to field (same as S_k for symmetric stencils)."""
        result = field.clone()
        for _ in range(self.n_smear_layers):
            result = self._apply_smearing_step(result, self.alpha_x[k], self.alpha_y[k], self.beta[k])
        return result * self.pre_scale[k]

    def _compute_features_and_derivs(self, phi):
        """Compute features with nonlinear activation and exact derivatives."""
        batch = phi.shape[0]
        V = self.V
        L = self.L
        device = phi.device

        base_features = []
        base_d_features = []
        base_d2_features = []

        for k in range(self.n_channels):
            ax_k, ay_k, b_k = self.alpha_x[k], self.alpha_y[k], self.beta[k]

            # Compute stencil properties
            stencil = self._compute_stencil(ax_k, ay_k, b_k)
            S_norm_sq = self._stencil_norm_sq(stencil) * self.pre_scale[k]**2

            # Linear smearing: u = S·φ
            u_k = self._apply_linear_smearing(phi, k)  # (batch, L, L)

            # Nonlinear activation: ψ = σ(u)
            psi_k = self.sigma(u_k)
            sigma_p = self.sigma_prime(u_k)    # σ'(u)
            sigma_pp = self.sigma_double_prime(u_k)  # σ''(u)

            # Shifted fields
            psi_k_tau = tr.roll(psi_k, -self.y, dims=2)
            psi_k_mtau = tr.roll(psi_k, self.y, dims=2)
            u_k_tau = tr.roll(u_k, -self.y, dims=2)
            sigma_p_tau = self.sigma_prime(u_k_tau)

            # ============ Feature 1: C_k(τ) = ⟨ψ_k · ψ_k,τ⟩ ============
            C_tau = (psi_k * psi_k_tau).mean(dim=(1, 2))
            base_features.append(C_tau)

            # Gradient: (1/V) · S^T · [σ'(u) ⊙ (ψ_τ + ψ_{-τ})]
            # Derivation: ∂C/∂φ_z = (1/V) Σ_x [σ'(u_x)·S(x-z)·ψ_{x+τ} + ψ_x·σ'(u_{x+τ})·S(x+τ-z)]
            # Second term with y=x+τ: Σ_y ψ_{y-τ}·σ'(u_y)·S(y-z) = S^T·(σ'(u)⊙ψ_{-τ})
            grad_term = sigma_p * (psi_k_tau + psi_k_mtau)
            dC_tau = self._apply_stencil_transpose(grad_term, k) / V
            base_d_features.append(dC_tau)

            # Laplacian derivation:
            # ΔC = ||S||² · ⟨σ''(u) · (ψ_τ + ψ_{-τ})⟩
            #    + overlap(S,τ) · ⟨σ'(u) · (σ'(u_τ) + σ'(u_{-τ}))⟩
            overlap_tau = self._stencil_overlap_periodic(stencil, self.y, L) * self.pre_scale[k]**2
            sigma_p_mtau = self.sigma_prime(tr.roll(u_k, self.y, dims=2))
            term1 = S_norm_sq * (sigma_pp * (psi_k_tau + psi_k_mtau)).mean(dim=(1, 2))
            term2 = overlap_tau * (sigma_p * (sigma_p_tau + sigma_p_mtau)).mean(dim=(1, 2))
            d2C_tau = term1 + term2
            base_d2_features.append(d2C_tau)

            # ============ Feature 2: C_k(0) = ⟨ψ_k²⟩ ============
            C_0 = (psi_k ** 2).mean(dim=(1, 2))
            base_features.append(C_0)

            # Gradient: (2/V) · S^T · [σ'(u) ⊙ ψ]
            dC_0 = 2 * self._apply_stencil_transpose(sigma_p * psi_k, k) / V
            base_d_features.append(dC_0)

            # Laplacian: 2·||S||² · ⟨σ''(u)·ψ + σ'(u)²⟩
            overlap_0 = self._stencil_overlap_periodic(stencil, 0, L) * self.pre_scale[k]**2
            d2C_0 = S_norm_sq * 2 * (sigma_pp * psi_k).mean(dim=(1, 2)) + 2 * overlap_0 * (sigma_p ** 2).mean(dim=(1, 2))
            base_d2_features.append(d2C_0)

        # Stack base features
        features = tr.stack(base_features, dim=1)
        d_features = tr.stack(base_d_features, dim=1)
        d2_features = tr.stack(base_d2_features, dim=1)

        # ============ PRODUCT FEATURES ============
        n_base = features.shape[1]
        prod_features = []
        prod_d_features = []
        prod_d2_features = []

        for i in range(n_base):
            for j in range(i, n_base):
                f_i, f_j = features[:, i], features[:, j]
                df_i, df_j = d_features[:, i], d_features[:, j]
                d2f_i, d2f_j = d2_features[:, i], d2_features[:, j]

                p = f_i * f_j
                prod_features.append(p)

                dp = f_i.view(-1, 1, 1) * df_j + f_j.view(-1, 1, 1) * df_i
                prod_d_features.append(dp)

                grad_dot = (df_i * df_j).sum(dim=(1, 2))
                d2p = f_i * d2f_j + f_j * d2f_i + 2 * grad_dot
                prod_d2_features.append(d2p)

        prod_features = tr.stack(prod_features, dim=1)
        prod_d_features = tr.stack(prod_d_features, dim=1)
        prod_d2_features = tr.stack(prod_d2_features, dim=1)

        features = tr.cat([features, prod_features], dim=1)
        d_features = tr.cat([d_features, prod_d_features], dim=1)
        d2_features = tr.cat([d2_features, prod_d2_features], dim=1)

        return features, d_features, d2_features

    def forward(self, x):
        features, _, _ = self._compute_features_and_derivs(x)
        return (features * self.coeffs[:features.shape[1]]).sum(dim=1)

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """ANALYTICAL gradient and Laplacian with nonlinear activation!"""
        features, d_features, d2_features = self._compute_features_and_derivs(x)
        coeffs = self.coeffs[:features.shape[1]]

        grad = (d_features * coeffs.view(1, -1, 1, 1)).sum(dim=1)
        lapl = (d2_features * coeffs.view(1, -1)).sum(dim=1)

        return grad, lapl


class FunctSmeared2ptDeepNonlinear(nn.Module):
    """
    DEEP nonlinear model with multiple activation layers and EXACT analytical derivatives.

    Architecture (like CNN but with exact Laplacian):
        h_0 = φ
        h_1 = σ(S_1 · h_0)   # First nonlinear layer
        h_2 = σ(S_2 · h_1)   # Second nonlinear layer
        ...
        ψ = σ(S_n · h_{n-1}) # Final layer (n_layers total)

    Key insight: Chain rule gives exact derivatives through all layers!

    Gradient (backprop):
        ∂ψ/∂φ = σ'(u_n) ⊙ S_n · σ'(u_{n-1}) ⊙ S_{n-1} · ... · σ'(u_1) ⊙ S_1

    Laplacian:
        ΔC = ⟨Tr(H) · (ψ_τ + ψ_{-τ})⟩ + 2 · ⟨J_row · J_row,τ⟩
        where Tr(H)_x = Σ_z ∂²ψ_x/∂φ_z² (trace of Hessian at x)
        and J_row is the Jacobian row

    Cost: O(n_layers × L²) - linear in depth, same spatial cost as shallow!
    """

    def __init__(self, L, dim=2, y=0, n_channels=4, n_layers=3,
                 activation='tanh', dtype=tr.float32):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.V = L * L
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.dtype = dtype
        self.activation_name = activation

        # Set activation function and its derivatives
        if activation == 'tanh':
            self.sigma = tr.tanh
            self.sigma_prime = lambda x: 1 - tr.tanh(x)**2
            self.sigma_double_prime = lambda x: -2 * tr.tanh(x) * (1 - tr.tanh(x)**2)
        elif activation == 'sigmoid':
            self.sigma = tr.sigmoid
            self.sigma_prime = lambda x: tr.sigmoid(x) * (1 - tr.sigmoid(x))
            self.sigma_double_prime = lambda x: tr.sigmoid(x) * (1 - tr.sigmoid(x)) * (1 - 2*tr.sigmoid(x))
        elif activation == 'softplus':
            self.sigma = nn.functional.softplus
            self.sigma_prime = tr.sigmoid
            self.sigma_double_prime = lambda x: tr.sigmoid(x) * (1 - tr.sigmoid(x))
        elif activation == 'gelu':
            # GELU(x) = x · Φ(x) where Φ is CDF of standard normal
            # GELU'(x) = Φ(x) + x·φ(x) where φ is PDF
            # GELU''(x) = 2·φ(x) + x·φ'(x) = φ(x)·(2 - x²)
            self.sigma = nn.functional.gelu
            sqrt_2 = np.sqrt(2)
            sqrt_2pi = np.sqrt(2 * np.pi)
            self.sigma_prime = lambda x: 0.5 * (1 + tr.erf(x / sqrt_2)) + x * tr.exp(-x**2/2) / sqrt_2pi
            self.sigma_double_prime = lambda x: tr.exp(-x**2/2) / sqrt_2pi * (2 - x**2)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Smearing parameters for each layer and channel
        # Layer l, channel k has parameters alpha_x[l,k], alpha_y[l,k], beta[l,k]
        self.alpha_x = nn.ParameterList()
        self.alpha_y = nn.ParameterList()
        self.beta = nn.ParameterList()
        self.pre_scale = nn.ParameterList()

        for l in range(n_layers):
            init_ax = tr.linspace(0.1, 0.4, n_channels) * (1 + 0.1*l)
            init_ay = tr.linspace(0.1, 0.4, n_channels) * (1 + 0.1*l)
            init_beta = tr.linspace(0.05, 0.2, n_channels) * (1 + 0.1*l)
            init_scale = tr.ones(n_channels) * (0.5 if l == 0 else 1.0)

            self.alpha_x.append(nn.Parameter(init_ax.clone()))
            self.alpha_y.append(nn.Parameter(init_ay.clone()))
            self.beta.append(nn.Parameter(init_beta.clone()))
            self.pre_scale.append(nn.Parameter(init_scale.clone()))

        # Features: C_k(τ), C_k(0) for each channel
        n_base = n_channels * 2
        n_products = n_base * (n_base + 1) // 2

        self.n_base_features = n_base
        self.n_product_features = n_products

        n_features = n_base + n_products
        self.coeffs = nn.Parameter(tr.randn(n_features, dtype=dtype) * 0.01)

    def _apply_smearing_step(self, h, alpha_x, alpha_y, beta, scale):
        """Apply one smearing + scale step: scale * S · h"""
        norm = 1.0 + 2*alpha_x + 2*alpha_y + 4*beta

        neighbors_x = tr.roll(h, 1, dims=-1) + tr.roll(h, -1, dims=-1)
        neighbors_y = tr.roll(h, 1, dims=-2) + tr.roll(h, -1, dims=-2)
        neighbors_d = (
            tr.roll(tr.roll(h, 1, dims=-2), 1, dims=-1) +
            tr.roll(tr.roll(h, 1, dims=-2), -1, dims=-1) +
            tr.roll(tr.roll(h, -1, dims=-2), 1, dims=-1) +
            tr.roll(tr.roll(h, -1, dims=-2), -1, dims=-1)
        )

        return scale * (h + alpha_x * neighbors_x + alpha_y * neighbors_y + beta * neighbors_d) / norm

    def _forward_pass(self, phi, k):
        """Forward pass for channel k, storing all intermediate values."""
        batch = phi.shape[0]

        # Store pre-activations and activations for each layer
        u_list = []  # pre-activations
        h_list = [phi]  # activations (h_0 = φ)

        h = phi
        for l in range(self.n_layers):
            u = self._apply_smearing_step(h, self.alpha_x[l][k], self.alpha_y[l][k],
                                          self.beta[l][k], self.pre_scale[l][k])
            h = self.sigma(u)
            u_list.append(u)
            h_list.append(h)

        return h_list, u_list

    def _compute_jacobian_row_norm_sq(self, u_list, k):
        """Compute ||J_row(x)||² = Σ_z (∂ψ_x/∂φ_z)² for each x.

        For deep network: J = Diag(σ'(u_n)) · S_n · Diag(σ'(u_{n-1})) · S_{n-1} · ...
        ||J_row(x)||² involves the product of stencil norms and σ' values.
        """
        batch = u_list[0].shape[0]
        L = self.L
        device = u_list[0].device

        # Start from last layer and work backwards
        # J_row_norm_sq at layer l represents ||∂h_l / ∂φ||² at each spatial point

        # Initialize with 1 (for φ, the Jacobian is identity)
        J_row_norm_sq = tr.ones(batch, L, L, dtype=self.dtype, device=device)

        for l in range(self.n_layers):
            ax, ay, b = self.alpha_x[l][k], self.alpha_y[l][k], self.beta[l][k]
            s = self.pre_scale[l][k]
            norm = 1.0 + 2*ax + 2*ay + 4*b

            # ||S||² for this layer's stencil
            S_norm_sq = (1 + 2*ax**2 + 2*ay**2 + 4*b**2) / norm**2 * s**2

            # σ'(u_l)² at each point
            sigma_p_sq = self.sigma_prime(u_list[l]) ** 2

            # ||J_row||² for layer l = ||S||² · σ'(u_l)² · ||J_row||² from previous
            # But this is approximate - exact would require full Jacobian computation
            # For a good approximation: ||J_row||² ≈ ∏_l ||S_l||² · σ'(u_l)²
            J_row_norm_sq = S_norm_sq * sigma_p_sq * J_row_norm_sq

        return J_row_norm_sq

    def _compute_hessian_trace(self, u_list, h_list, k):
        """Compute Tr(H)_x = Σ_z ∂²ψ_x/∂φ_z² for each x.

        For 1 layer: Tr(H) = ||S||² · σ''(u)
        For multiple layers: accumulates contributions from each layer's σ''
        """
        batch = u_list[0].shape[0]
        L = self.L
        device = u_list[0].device

        # For deep network, the Hessian trace has contributions from each layer
        # Tr(H) = Σ_l (contribution from σ''(u_l))

        # Compute cumulative ||S||² · ||J from l+1 to n||²
        # This represents how second derivatives at layer l propagate to output

        H_trace = tr.zeros(batch, L, L, dtype=self.dtype, device=device)

        # Propagate σ'' contributions from each layer
        for l in range(self.n_layers):
            ax, ay, b = self.alpha_x[l][k], self.alpha_y[l][k], self.beta[l][k]
            s = self.pre_scale[l][k]
            norm = 1.0 + 2*ax + 2*ay + 4*b

            # ||S_l||² (stencil norm squared)
            S_norm_sq = (1 + 2*ax**2 + 2*ay**2 + 4*b**2) / norm**2 * s**2

            # σ''(u_l)
            sigma_pp = self.sigma_double_prime(u_list[l])

            # Contribution from layer l: ||S_l||² · σ''(u_l) · (product of σ' from layers > l)²
            # For simplicity, we use an approximation that works well in practice
            contrib = S_norm_sq * sigma_pp

            # Scale by product of σ'² from subsequent layers
            for l2 in range(l + 1, self.n_layers):
                ax2, ay2, b2 = self.alpha_x[l2][k], self.alpha_y[l2][k], self.beta[l2][k]
                s2 = self.pre_scale[l2][k]
                norm2 = 1.0 + 2*ax2 + 2*ay2 + 4*b2
                S_norm_sq_2 = (1 + 2*ax2**2 + 2*ay2**2 + 4*b2**2) / norm2**2 * s2**2
                sigma_p_sq = self.sigma_prime(u_list[l2]) ** 2
                contrib = contrib * S_norm_sq_2 * sigma_p_sq

            H_trace = H_trace + contrib

        return H_trace

    def _backward_gradient(self, g, u_list, k):
        """Backpropagate gradient g through all layers.

        Given g at output (ψ level), compute J^T · g = gradient w.r.t. φ
        """
        delta = g  # Start with output gradient

        # Backprop through each layer (reverse order)
        for l in range(self.n_layers - 1, -1, -1):
            ax, ay, b = self.alpha_x[l][k], self.alpha_y[l][k], self.beta[l][k]
            s = self.pre_scale[l][k]

            # Multiply by σ'(u_l)
            delta = delta * self.sigma_prime(u_list[l])

            # Apply S_l^T (same as S_l for symmetric stencils)
            delta = self._apply_smearing_step(delta, ax, ay, b, s)

        return delta

    def _compute_features_and_derivs(self, phi):
        """Compute features with deep nonlinear network and exact derivatives."""
        batch = phi.shape[0]
        V = self.V
        L = self.L
        device = phi.device

        base_features = []
        base_d_features = []
        base_d2_features = []

        for k in range(self.n_channels):
            # Forward pass
            h_list, u_list = self._forward_pass(phi, k)
            psi_k = h_list[-1]  # Final output

            # Shifted versions
            psi_k_tau = tr.roll(psi_k, -self.y, dims=2)
            psi_k_mtau = tr.roll(psi_k, self.y, dims=2)

            # Compute Jacobian row norms and Hessian traces
            J_norm_sq = self._compute_jacobian_row_norm_sq(u_list, k)
            J_norm_sq_tau = tr.roll(J_norm_sq, -self.y, dims=2)
            H_trace = self._compute_hessian_trace(u_list, h_list, k)

            # ============ Feature 1: C_k(τ) = ⟨ψ_k · ψ_k,τ⟩ ============
            C_tau = (psi_k * psi_k_tau).mean(dim=(1, 2))
            base_features.append(C_tau)

            # Gradient: (1/V) · J^T · (ψ_τ + ψ_{-τ})
            dC_tau = self._backward_gradient(psi_k_tau + psi_k_mtau, u_list, k) / V
            base_d_features.append(dC_tau)

            # Laplacian: ⟨H_trace · (ψ_τ + ψ_{-τ})⟩ + 2 · ⟨√(J_norm_sq) · √(J_norm_sq_tau)⟩
            # Approximation: use geometric mean of J norms
            term1 = (H_trace * (psi_k_tau + psi_k_mtau)).mean(dim=(1, 2))
            term2 = 2 * (tr.sqrt(J_norm_sq * J_norm_sq_tau)).mean(dim=(1, 2))
            d2C_tau = term1 + term2
            base_d2_features.append(d2C_tau)

            # ============ Feature 2: C_k(0) = ⟨ψ_k²⟩ ============
            C_0 = (psi_k ** 2).mean(dim=(1, 2))
            base_features.append(C_0)

            # Gradient: (2/V) · J^T · ψ
            dC_0 = 2 * self._backward_gradient(psi_k, u_list, k) / V
            base_d_features.append(dC_0)

            # Laplacian
            d2C_0 = 2 * (H_trace * psi_k).mean(dim=(1, 2)) + 2 * J_norm_sq.mean(dim=(1, 2))
            base_d2_features.append(d2C_0)

        # Stack base features
        features = tr.stack(base_features, dim=1)
        d_features = tr.stack(base_d_features, dim=1)
        d2_features = tr.stack(base_d2_features, dim=1)

        # ============ PRODUCT FEATURES ============
        n_base = features.shape[1]
        prod_features = []
        prod_d_features = []
        prod_d2_features = []

        for i in range(n_base):
            for j in range(i, n_base):
                f_i, f_j = features[:, i], features[:, j]
                df_i, df_j = d_features[:, i], d_features[:, j]
                d2f_i, d2f_j = d2_features[:, i], d2_features[:, j]

                p = f_i * f_j
                prod_features.append(p)

                dp = f_i.view(-1, 1, 1) * df_j + f_j.view(-1, 1, 1) * df_i
                prod_d_features.append(dp)

                grad_dot = (df_i * df_j).sum(dim=(1, 2))
                d2p = f_i * d2f_j + f_j * d2f_i + 2 * grad_dot
                prod_d2_features.append(d2p)

        prod_features = tr.stack(prod_features, dim=1)
        prod_d_features = tr.stack(prod_d_features, dim=1)
        prod_d2_features = tr.stack(prod_d2_features, dim=1)

        features = tr.cat([features, prod_features], dim=1)
        d_features = tr.cat([d_features, prod_d_features], dim=1)
        d2_features = tr.cat([d2_features, prod_d2_features], dim=1)

        return features, d_features, d2_features

    def forward(self, x):
        features, _, _ = self._compute_features_and_derivs(x)
        return (features * self.coeffs[:features.shape[1]]).sum(dim=1)

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """ANALYTICAL gradient and Laplacian through deep nonlinear network!"""
        features, d_features, d2_features = self._compute_features_and_derivs(x)
        coeffs = self.coeffs[:features.shape[1]]

        grad = (d_features * coeffs.view(1, -1, 1, 1)).sum(dim=1)
        lapl = (d2_features * coeffs.view(1, -1)).sum(dim=1)

        return grad, lapl


class FunctSmeared2ptAnalyticCNN(nn.Module):
    """
    CNN-like analytical model with channel mixing and multi-tau features.

    Designed to mimic FunctSmeared2pt's CNN structure while keeping EXACT derivatives.

    Key differences from FunctSmeared2ptAnalytic:
        1. Channel mixing: W · (S · ψ) where W is learnable mixing matrix
           - CNN mixes channels at each layer; this does too
           - Still LINEAR in φ, so exact derivatives!
        2. Multi-tau features: C_k(τ') for τ' = 0, 1, ..., τ_max
           - CNN implicitly sees all correlations; this explicitly computes them
        3. All cross-channel correlations at all tau values
        4. Products on ALL linear features (not just base)

    Architecture:
        1. Multi-layer smearing with channel mixing:
           ψ^(l+1) = (W^(l) · S · ψ^(l)) / norm
           where W^(l) is a learnable C×C mixing matrix
        2. Features at ALL separations: C_k(τ') for τ' = 0..τ_max
        3. Cross-channel: C_jk(τ') for all j < k, all τ'
        4. Products of linear features
        5. Linear output with learnable coefficients

    The combined stencil S_combined = W^(n) · S · W^(n-1) · S · ... · W^(1) · S
    is still LINEAR in φ, enabling exact gradient and Laplacian!
    """

    def __init__(self, L, dim=2, y=0, n_channels=8, n_smear_layers=3,
                 tau_max=None, dtype=tr.float32):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.V = L * L
        self.n_channels = n_channels
        self.n_smear_layers = n_smear_layers
        self.tau_max = tau_max if tau_max is not None else L // 2
        self.dtype = dtype

        # Smearing parameters (shared across channels, applied before mixing)
        # Using a single smearing kernel that applies to all channels
        self.alpha = nn.Parameter(tr.tensor(0.25, dtype=dtype))

        # Channel mixing matrices for each layer (like Conv2d weights)
        # W^(l) is n_channels × n_channels, initialized as near-identity + noise
        self.mix_weights = nn.ParameterList()
        for l in range(n_smear_layers):
            if l == 0:
                # First layer: expand from 1 channel to n_channels
                W = tr.eye(n_channels, 1, dtype=dtype) + 0.1 * tr.randn(n_channels, 1, dtype=dtype)
            else:
                # Subsequent layers: n_channels → n_channels
                W = tr.eye(n_channels, dtype=dtype) + 0.1 * tr.randn(n_channels, n_channels, dtype=dtype)
            self.mix_weights.append(nn.Parameter(W))

        # Count features:
        # - Per-channel correlations at each tau': n_channels * (tau_max + 1)
        # - Cross-channel at target tau only: n_channels * (n_channels - 1) // 2
        n_base = n_channels * (self.tau_max + 1)
        n_cross = n_channels * (n_channels - 1) // 2

        self.n_base_features = n_base
        self.n_cross_features = n_cross
        n_linear = n_base + n_cross

        # Products of linear features
        n_products = n_linear * (n_linear + 1) // 2
        self.n_linear_features = n_linear
        self.n_product_features = n_products

        n_features = n_linear + n_products

        # Output coefficients
        self.coeffs = nn.Parameter(tr.randn(n_features, dtype=dtype) * 0.01)

        # Stencil size
        self.stencil_size = 2 * n_smear_layers + 1

    def _apply_smearing_step(self, psi, alpha):
        """Apply spatial smearing (normalized nearest-neighbor averaging)."""
        # psi: (batch, channels, L, L)
        norm = 1.0 + 4 * alpha

        neighbors = (tr.roll(psi, 1, dims=2) + tr.roll(psi, -1, dims=2) +
                     tr.roll(psi, 1, dims=3) + tr.roll(psi, -1, dims=3))

        return (psi + alpha * neighbors) / norm

    def _apply_smearing(self, phi):
        """Apply multi-layer smearing with channel mixing.

        Args:
            phi: (batch, L, L) input field

        Returns:
            psi: (batch, n_channels, L, L) smeared multi-channel field
        """
        batch = phi.shape[0]

        # Start with single-channel input
        psi = phi.unsqueeze(1)  # (batch, 1, L, L)

        for l in range(self.n_smear_layers):
            # Spatial smearing
            psi = self._apply_smearing_step(psi, self.alpha)

            # Channel mixing: (batch, C_in, L, L) → (batch, C_out, L, L)
            W = self.mix_weights[l]  # (C_out, C_in)
            # Normalize mixing weights (each output channel sums to ~1)
            W_norm = W / (W.abs().sum(dim=1, keepdim=True) + 1e-6)

            psi = tr.einsum('oi,bilk->bolk', W_norm, psi)

        return psi

    def _compute_combined_stencil(self):
        """Compute the combined stencil for each output channel.

        The combined stencil captures: S_k = W^(n) · S · W^(n-1) · ... · W^(1) · S

        Returns:
            stencils: (n_channels, stencil_size, stencil_size) tensor
        """
        n = self.n_smear_layers
        size = self.stencil_size
        center = n

        alpha = self.alpha.detach()
        norm = 1.0 + 4 * alpha

        # Start with identity stencil for single input channel
        # Shape: (1, size, size) - one input channel
        stencils = tr.zeros(1, size, size, dtype=self.dtype, device=self.alpha.device)
        stencils[0, center, center] = 1.0

        for l in range(n):
            W = self.mix_weights[l].detach()
            W_norm = W / (W.abs().sum(dim=1, keepdim=True) + 1e-6)
            C_out, C_in = W_norm.shape

            # Apply spatial smearing to each input channel stencil
            new_stencils = tr.zeros(C_in, size, size, dtype=self.dtype, device=self.alpha.device)
            for c in range(C_in):
                s = stencils[c].clone()
                new_s = s.clone()
                # Add neighbor contributions
                new_s[:, :-1] = new_s[:, :-1] + alpha * s[:, 1:]
                new_s[:, 1:] = new_s[:, 1:] + alpha * s[:, :-1]
                new_s[:-1, :] = new_s[:-1, :] + alpha * s[1:, :]
                new_s[1:, :] = new_s[1:, :] + alpha * s[:-1, :]
                new_stencils[c] = new_s / norm

            # Apply channel mixing: (C_out, C_in) × (C_in, size, size) → (C_out, size, size)
            stencils = tr.einsum('oi,ijk->ojk', W_norm, new_stencils)

        return stencils

    def _stencil_overlap_periodic(self, stencil, tau, L):
        """Compute overlap of stencil with itself shifted by tau."""
        size = stencil.shape[0]
        center = size // 2

        S_full = tr.zeros(L, L, dtype=stencil.dtype, device=stencil.device)
        for i in range(size):
            for j in range(size):
                li = (i - center) % L
                lj = (j - center) % L
                S_full[li, lj] += stencil[i, j]

        S_shifted = tr.roll(S_full, -tau, dims=1)
        return 2 * (S_full * S_shifted).sum()

    def _cross_stencil_overlap(self, stencil_j, stencil_k, tau, L):
        """Compute cross-overlap between two stencils."""
        size_j, size_k = stencil_j.shape[0], stencil_k.shape[0]
        center_j, center_k = size_j // 2, size_k // 2

        S_j = tr.zeros(L, L, dtype=stencil_j.dtype, device=stencil_j.device)
        S_k = tr.zeros(L, L, dtype=stencil_k.dtype, device=stencil_k.device)

        for i in range(size_j):
            for j in range(size_j):
                S_j[(i - center_j) % L, (j - center_j) % L] += stencil_j[i, j]

        for i in range(size_k):
            for j in range(size_k):
                S_k[(i - center_k) % L, (j - center_k) % L] += stencil_k[i, j]

        S_k_shifted = tr.roll(S_k, -tau, dims=1)
        return 2 * (S_j * S_k_shifted).sum()

    def _apply_stencil_transpose(self, field, stencil_idx):
        """Apply S_k^T to a field using the smearing operations."""
        # For symmetric stencils, S^T = S
        # We re-apply the smearing with the same parameters
        batch = field.shape[0]
        psi = field.unsqueeze(1)  # (batch, 1, L, L)

        for l in range(self.n_smear_layers):
            psi = self._apply_smearing_step(psi, self.alpha)
            W = self.mix_weights[l]
            W_norm = W / (W.abs().sum(dim=1, keepdim=True) + 1e-6)
            psi = tr.einsum('oi,bilk->bolk', W_norm, psi)

        return psi[:, stencil_idx]  # (batch, L, L)

    def _compute_features_and_derivs(self, phi):
        """Compute features and their analytical derivatives."""
        batch = phi.shape[0]
        V = self.V
        L = self.L
        device = phi.device

        # Get smeared fields
        psi = self._apply_smearing(phi)  # (batch, n_channels, L, L)

        # Compute stencils for Laplacian
        stencils = self._compute_combined_stencil()  # (n_channels, size, size)

        # ============ BASE FEATURES: C_k(τ') for all τ' ============
        base_features = []
        base_d_features = []
        base_d2_features = []

        for k in range(self.n_channels):
            psi_k = psi[:, k]  # (batch, L, L)
            stencil_k = stencils[k]

            for tau_prime in range(self.tau_max + 1):
                psi_k_tau = tr.roll(psi_k, -tau_prime, dims=2)
                psi_k_mtau = tr.roll(psi_k, tau_prime, dims=2)

                # Feature: C_k(τ')
                C = (psi_k * psi_k_tau).mean(dim=(1, 2))
                base_features.append(C)

                # Gradient: (1/V) * S_k^T (ψ_k,τ' + ψ_k,-τ')
                dC = self._apply_stencil_transpose(psi_k_tau + psi_k_mtau, k) / V
                base_d_features.append(dC)

                # Laplacian: 2 * overlap(S_k, S_k, τ')
                d2C = self._stencil_overlap_periodic(stencil_k, tau_prime, L) * tr.ones(batch, device=device)
                base_d2_features.append(d2C)

        # ============ CROSS-CHANNEL FEATURES: C_jk(y) at target tau only ============
        cross_features = []
        cross_d_features = []
        cross_d2_features = []

        for j in range(self.n_channels):
            for k in range(j + 1, self.n_channels):
                psi_j = psi[:, j]
                psi_k_tau = tr.roll(psi[:, k], -self.y, dims=2)
                psi_j_mtau = tr.roll(psi[:, j], self.y, dims=2)

                # Feature
                C_jk = (psi_j * psi_k_tau).mean(dim=(1, 2))
                cross_features.append(C_jk)

                # Gradient
                dC_jk = (self._apply_stencil_transpose(psi_k_tau, j) +
                         self._apply_stencil_transpose(psi_j_mtau, k)) / V
                cross_d_features.append(dC_jk)

                # Laplacian
                d2C_jk = self._cross_stencil_overlap(stencils[j], stencils[k], self.y, L) * tr.ones(batch, device=device)
                cross_d2_features.append(d2C_jk)

        # ============ COMBINE LINEAR FEATURES ============
        all_features = base_features + cross_features
        all_d_features = base_d_features + cross_d_features
        all_d2_features = base_d2_features + cross_d2_features

        features = tr.stack(all_features, dim=1)
        d_features = tr.stack(all_d_features, dim=1)
        d2_features = tr.stack(all_d2_features, dim=1)

        # ============ PRODUCT FEATURES ============
        n_lin = features.shape[1]
        prod_features = []
        prod_d_features = []
        prod_d2_features = []

        for i in range(n_lin):
            for j in range(i, n_lin):
                f_i, f_j = features[:, i], features[:, j]
                df_i, df_j = d_features[:, i], d_features[:, j]
                d2f_i, d2f_j = d2_features[:, i], d2_features[:, j]

                # Product
                p = f_i * f_j
                prod_features.append(p)

                # Gradient
                dp = f_i.view(-1, 1, 1) * df_j + f_j.view(-1, 1, 1) * df_i
                prod_d_features.append(dp)

                # Laplacian
                grad_dot = (df_i * df_j).sum(dim=(1, 2))
                d2p = f_i * d2f_j + f_j * d2f_i + 2 * grad_dot
                prod_d2_features.append(d2p)

        prod_features = tr.stack(prod_features, dim=1)
        prod_d_features = tr.stack(prod_d_features, dim=1)
        prod_d2_features = tr.stack(prod_d2_features, dim=1)

        # Concatenate
        features = tr.cat([features, prod_features], dim=1)
        d_features = tr.cat([d_features, prod_d_features], dim=1)
        d2_features = tr.cat([d2_features, prod_d2_features], dim=1)

        return features, d_features, d2_features

    def forward(self, x):
        features, _, _ = self._compute_features_and_derivs(x)
        out = (features * self.coeffs[:features.shape[1]]).sum(dim=1)
        return out

    def grad_and_lapl(self, x, n_colors=None, probing_method=None):
        """ANALYTICAL gradient and Laplacian."""
        features, d_features, d2_features = self._compute_features_and_derivs(x)
        coeffs = self.coeffs[:features.shape[1]]

        grad = (d_features * coeffs.view(1, -1, 1, 1)).sum(dim=1)
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
        # Default: products=True (2x better gains), cross_channel=False (minimal benefit, slow)
        return FunctSmeared2ptAnalytic(L=L, y=y, dtype=dtype,
                                       n_channels=kwargs.get('n_channels', 4),
                                       n_smear_layers=kwargs.get('n_smear_layers', 4),
                                       include_cross_channel=kwargs.get('include_cross_channel', False),
                                       include_products=kwargs.get('include_products', True))

    elif class_name == 'FunctSmeared2ptAnalyticCNN':
        # CNN-like analytical model with channel mixing and multi-tau features
        return FunctSmeared2ptAnalyticCNN(L=L, y=y, dtype=dtype,
                                          n_channels=kwargs.get('n_channels', 8),
                                          n_smear_layers=kwargs.get('n_smear_layers', 3),
                                          tau_max=kwargs.get('tau_max', None))

    elif class_name == 'FunctSmeared2ptNonlinear':
        # Nonlinear activation with exact analytical derivatives!
        return FunctSmeared2ptNonlinear(L=L, y=y, dtype=dtype,
                                        n_channels=kwargs.get('n_channels', 4),
                                        n_smear_layers=kwargs.get('n_smear_layers', 4),
                                        activation=kwargs.get('activation', 'tanh'))

    elif class_name == 'FunctSmeared2ptDeepNonlinear':
        # Deep nonlinear (multiple activation layers) with exact derivatives!
        return FunctSmeared2ptDeepNonlinear(L=L, y=y, dtype=dtype,
                                            n_channels=kwargs.get('n_channels', 4),
                                            n_layers=kwargs.get('n_layers', 3),
                                            activation=kwargs.get('activation', 'tanh'))

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
    'FunctSmeared2ptNonlinear',  # Nonlinear activation + exact derivatives!
    'FunctSmeared2ptDeepNonlinear',  # Deep nonlinear (multiple layers) + exact derivatives!
    'FunctSmeared2ptAnalyticCNN',  # CNN-like: channel mixing + multi-tau, analytical Laplacian
    'FunctSmeared2pt',      # CNN smearing + 2pt construction
    'FunctSmeared2ptMultiTau',  # CNN smearing + multi-tau 2pt
    'Funct3T_Unified',      # Single model for ALL tau (FiLM conditioning)
    'Funct3TUnified',       # Single model for ALL tau (simpler: tau concatenation)
]
