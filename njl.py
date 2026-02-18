#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:30:00 2025

@author: Yamil Cahuana Medrano
"""

import numpy as np
import torch as tr

import numpy as np
import torch as tr

class field:
    def __init__(self, V, l, m, Nf, batch_size=1, device="cpu", dtype=tr.float64): 
        self.V = tuple(V)
        self.L = V[0]
        self.Vol = np.prod(V)
        self.Nf = Nf
        self.lam = l
        self.mass = m
        self.Bs = batch_size
        self.device = device
        self.dtype = dtype
        self.cdtype = tr.complex128 if dtype == tr.float64 else tr.complex64
        
        # Verificación de seguridad inicial
        if self.lam == 0: raise ValueError("Lambda cannot be zero!")
        
        self.setup_gamma()

    def setup_gamma(self):
        g0 = tr.tensor([[0., 1.], [1., 0.]], device=self.device, dtype=self.cdtype)
        g1 = tr.tensor([[0., -1j], [1j, 0.]], device=self.device, dtype=self.cdtype)
        id2 = tr.eye(2, device=self.device, dtype=self.cdtype)
        self.gammas = [g0, g1]
        self.proj_minus = [(id2 - g) * 0.5 for g in self.gammas]
        self.proj_plus  = [(id2 + g) * 0.5 for g in self.gammas]


    def hotStart(self):
        """
        Generate a hot start configuration for the sigma field
        Output: sigma of size (B xL x L)
        """
        sigma=tr.normal(0.0,1.0, [self.Bs,self.V[0],self.V[1]], dtype=self.dtype,device=self.device)
        return sigma
    
    def refreshP(self):
        """
            momentum for hamiltonian dynamics for sigma field
            Output: P of size (B,L,L)
        """
        P = tr.normal(0.0,1.0,[self.Bs,self.V[0],self.V[1]],dtype=self.dtype,device=self.device)
        return P

    def evolveQ(self,dt,P,Q):
        """leapfrog update of a scalar field"""
        return Q + dt*P


    def kinetic(self,P):
        return tr.einsum('bxy,bxy->b',P,P)/2.0

    def coldStart(self):
        """
        Generate a cold start configuration for the sigma field
        Output: sigma of size (B x L x L), values in {-1, 1}
        """
        sigma=tr.ones([self.Bs,self.V[0],self.V[1]], dtype=self.dtype,device=self.device)
        return sigma
    

    def build_D_unscaled(self, sigma):
        # Verifica si entra basura
        if tr.isnan(sigma).any(): raise ValueError("NaN detected in sigma inside build_D")
            
        L = self.L
        Bs = self.Bs
        dim_spinor = 2
        vol = L * L
        matrix_dim = vol * dim_spinor
        
        diag_term = (self.mass + 2.0 + sigma).to(self.cdtype)
        diag_flat = diag_term.view(Bs, -1)
        
        D = tr.zeros((Bs, matrix_dim, matrix_dim), dtype=self.cdtype, device=self.device)
        idx = tr.arange(vol, device=self.device)
        
        for s in range(dim_spinor):
            rows = 2 * idx + s
            D[:, rows, rows] = diag_flat
            
        for mu in range(2):
            x, y = tr.meshgrid(tr.arange(L), tr.arange(L), indexing='ij')
            x, y = x.to(self.device), y.to(self.device)
            
            if mu == 0:
                x_plus, x_minus = (x + 1) % L, (x - 1) % L
                y_curr, y_curr_m = y, y
            else:
                x_plus, x_minus = x, x
                y_curr, y_curr_m = (y + 1) % L, (y - 1) % L
            
            i_curr = (x * L + y).flatten()
            i_plus = (x_plus * L + y_curr).flatten()
            i_minus = (x_minus * L + y_curr_m).flatten()
            
            P_minus = -1.0 * self.proj_minus[mu]
            P_plus  = -1.0 * self.proj_plus[mu]
            
            for s1 in range(2):
                for s2 in range(2):
                    rows = 2 * i_curr + s1
                    D[:, rows, 2*i_plus+s2] += P_minus[s1, s2]
                    D[:, rows, 2*i_minus+s2] += P_plus[s1, s2]      
        return D

    def generate_phi(self, sigma):
        D = self.build_D_unscaled(sigma)
        dim = 2 * self.L * self.L
        chi = tr.randn((self.Bs, dim), device=self.device, dtype=self.cdtype)
        Ddag = D.transpose(-2, -1).conj()
        self.curr_phi = tr.einsum('bji, bj -> bi', Ddag, chi)
        if tr.isnan(self.curr_phi).any(): print("WARNING: NaNs generated in phi!")
        return self.curr_phi

    def solve_robust(self, A, b):
        """Solver que NUNCA devuelve NaNs (o lo intenta con todas sus fuerzas)"""
        # 1. Jitter (Ruido) siempre activo para estabilidad numérica en float64
        jitter_eps = 1e-10
        jitter = jitter_eps * tr.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        
        try:
            return tr.linalg.solve(A + jitter, b)
        except RuntimeError:
            print("Solve failed even with jitter. Increasing jitter...")
            jitter = 1e-6 * tr.eye(A.shape[-1], device=A.device, dtype=A.dtype)
            return tr.linalg.solve(A + jitter, b)

    def fermion_action(self, sigma):
        D = self.build_D_unscaled(sigma)
        Ddag = D.transpose(-2, -1).conj()
        Q = tr.matmul(Ddag, D)
        
        X = self.solve_robust(Q, self.curr_phi.unsqueeze(-1)).squeeze(-1)
        
        self.last_solve_X = X 
        Sf = tr.sum(tr.conj(self.curr_phi) * X, dim=1).real
        return Sf

    def action(self, sigma):
        # Look for nans
        if tr.isnan(sigma).any(): raise ValueError("NaN detected in sigma inside action")
        
        Sb = (self.Nf / (2 * self.lam)) * tr.sum(sigma**2, dim=(1,2))
        Sf = self.fermion_action(sigma)
        return Sb + Sf

    def force(self, sigma):
        #bosonic part
        F_bos = - (self.Nf / self.lam) * sigma
        
        #fermionic part
        #Error corrected: always recalculate X because sigma has changed in the integrator
        self.fermion_action(sigma)
        X = self.last_solve_X
        
        #Fallback if the solver exploded and returned None or something weird
        if tr.isnan(X).any(): 
            return tr.zeros_like(sigma)

        D = self.build_D_unscaled(sigma)
        Y = tr.einsum('bij, bj -> bi', D, X)
        
        X_view = X.view(self.Bs, self.L, self.L, 2)
        Y_view = Y.view(self.Bs, self.L, self.L, 2)
        dot_prod = tr.sum(tr.conj(X_view) * Y_view, dim=3)
        
        #Factor 2
        F_ferm = 2.0 * dot_prod.real
        
        #all
        return F_bos + F_ferm


    def measure_correlators(self, sigma):
        """
        Calcula los correladores de Pión y Sigma proyectados a momento cero.
        Retorna: (C_pi_t, C_sigma_t) con shape (Batch, L)
        """
        L = self.L
        Vol = self.Vol
        Bs = self.Bs
        
        #1. Build Dirac matrix for this sigma field
        # D has shape (Bs, 2*Vol, 2*Vol)
        D = self.build_D_unscaled(sigma)
        
        #2. Calculate the propagator S(x, 0) using a point source
        #Invert D * psi = source for each spin index at the origin.
        propagators = []
        
        for s in range(2): # Loop over spin indices of the source (up, down)
            # Vector source: all zeros, 1 at the origin (0,0)
            rhs = tr.zeros((Bs, Vol*2), dtype=self.cdtype, device=self.device)
            # The flat index of the origin (x=0, y=0) with spin s is simply s
            rhs[:, s] = 1.0 
            
            # Solve the linear system: S = D^-1 * rhs
            # Use solve_robust to avoid crashes if the matrix is singular
            psi = self.solve_robust(D, rhs)
            propagators.append(psi)

        #3. Reconstruct the propagator tensor S
        #Stack the solutions. Shape: (Bs, Vol*2, 2_source)
        S_flat = tr.stack(propagators, dim=-1)
        
        #Reshape to separate space (x,y) and spin (s_sink)
        #Final S shape: (Bs, L, L, 2_sink, 2_source)
        S = S_flat.view(Bs, L, L, 2, 2)
        
        #4. Wick contractions
        
        # --- PION (Pseudo scalar) ---
        # Correlator = Trace( S * S_dagger )
        # This is the sum of the absolute values squared of all elements
        Pion_xy = tr.sum(tr.abs(S)**2, dim=(3, 4)) 
        
        # --- CANAL SIGMA (Escalar) ---
        # Correlador = Traza( S * gamma5 * S_dagger * gamma5 )
        # Definimos gamma5 = diag(1, -1) consistente con tus gammas g0, g1
        g5 = tr.tensor([[1, 0], [0, -1]], dtype=self.cdtype, device=self.device)
        
        # Conjugada transpuesta (intercambiando índices de espín 3 y 4)
        S_dag = tr.conj(S.permute(0, 1, 2, 4, 3))
        
        # Operación S_back = g5 * S_dag * g5
        # Usamos einsum para multiplicar las matrices de 2x2 espín
        S_back = tr.einsum('ij, bxyjk, kl -> bxyil', g5, S_dag, g5)
        
        # Traza final: Tr(S * S_back)
        Sigma_xy = tr.einsum('bxyij, bxyji -> bxy', S, S_back).real

        # 5. Proyección a Momento Cero (p=0)
        # Sumamos sobre el volumen espacial para obtener C(t).
        # Asumimos dim=2 es 'x' (espacio) y dim=1 es 't' (tiempo).
        
        C_pi_t = tr.sum(Pion_xy, dim=2)      # Shape: (Bs, L_tiempo)
        C_sigma_t = tr.sum(Sigma_xy, dim=2)  # Shape: (Bs, L_tiempo)
        
        return C_pi_t, C_sigma_t



    def measure_momentum_correlators(self, sigma, k_list=[0, 1]):
        """
        Calculates Pion and Sigma correlators for specific momentum modes.
        
        Args:
            sigma: The auxiliary field configuration.
            k_list: List of integer momentum modes (p = 2*pi*k / L).
            
        Returns:
            results: A dictionary containing 'pion' and 'sigma' dictionaries,
                     keyed by the momentum mode 'k'.
                     Example: results['pion'][1] is the C_pi(t) for p=1.
        """
        L = self.L
        Vol = self.Vol
        Bs = self.Bs
        
        # 1. Construct Dirac Operator for the current configuration
        # D shape: (Batch, 2*Vol, 2*Vol)
        D = self.build_D_unscaled(sigma)
        
        # 2. Calculate Propagator S(x, 0) using a Point Source at the origin
        # We solve D * psi = source for each source spin component.
        propagators = []
        
        for s in range(2): # Loop over source spins (up/down)
            # Create source vector: zero everywhere, 1.0 at origin (0,0)
            rhs = tr.zeros((Bs, Vol*2), dtype=self.cdtype, device=self.device)
            rhs[:, s] = 1.0 
            
            # Solve linear system: S = D^-1 * source
            # Using solve_robust to handle potential singular matrices
            psi = self.solve_robust(D, rhs)
            propagators.append(psi)

        # 3. Reconstruct Propagator Tensor
        # Stack solutions. Shape becomes: (Batch, Vol*2, 2_source_spins)
        S_flat = tr.stack(propagators, dim=-1)
        
        # Reshape to separate spatial/temporal indices from spin indices
        # Final S shape: (Batch, Time, Space, Spin_sink, Spin_source)
        # Assuming L[0] is Time, L[1] is Space (or vice versa, L=L)
        S = S_flat.view(Bs, L, L, 2, 2)
        
        # 4. Perform Meson Contractions (Wick's Theorem)
        
        # --- PION CHANNEL (Pseudo-scalar) ---
        # Operator: P = psibar * gamma5 * psi
        # Correlation: tr(S * S_dagger)
        # We sum over spin indices (dims 3 and 4)
        Pion_xy = tr.sum(tr.abs(S)**2, dim=(3, 4)) 
        
        # --- SIGMA CHANNEL (Scalar) ---
        # Operator: S = psibar * I * psi
        # Correlation: tr(S * gamma5 * S_dagger * gamma5)
        
        # Define gamma5 = diag(1, -1) in chiral basis
        g5 = tr.tensor([[1, 0], [0, -1]], dtype=self.cdtype, device=self.device)
        
        # Conjugate transpose of S (swapping spin dims 3 and 4)
        S_dag = tr.conj(S.permute(0, 1, 2, 4, 3))
        
        # S_back = g5 * S_dag * g5
        # Using einsum for 2x2 matrix multiplication on spin indices
        S_back = tr.einsum('ij, bxyjk, kl -> bxyil', g5, S_dag, g5)
        
        # Final Trace: Tr(S * S_back)
        Sigma_xy = tr.einsum('bxyij, bxyji -> bxy', S, S_back).real

        # 5. Momentum Projection (Fourier Transform)
        # We project the spatial dimension onto momentum modes.
        # C(t, p) = Sum_x C(t, x) * exp(-i * p * x)
        
        results = {'pion': {}, 'sigma': {}}
        
        # Create spatial coordinate vector [0, 1, ..., L-1]
        x_coords = tr.arange(L, device=self.device, dtype=self.dtype)
        
        for k in k_list:
            # Define momentum value: p = 2*pi*k / L
            p_val = 2.0 * np.pi * k / L
            
            # Calculate phase factor: exp(-i * p * x)
            # Casting to complex is important here
            phase = tr.exp(-1j * p_val * x_coords)
            
            # Perform projection summing over spatial dimension (dim=2)
            # Broadcasting: 'phase' applies to the spatial axis for all batches/times
            
            # For Pion
            C_pi_p = tr.sum(Pion_xy.to(self.cdtype) * phase, dim=2)
            results['pion'][k] = C_pi_p.real # Store real part
            
            # For Sigma
            C_sig_p = tr.sum(Sigma_xy.to(self.cdtype) * phase, dim=2)
            results['sigma'][k] = C_sig_p.real # Store real part
            
        return results

