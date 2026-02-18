#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:30:00 2025

@author: Yamil Cahuana Medrano
"""

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
        sigma=tr.normal(0.0,0.1, [self.Bs,self.V[0],self.V[1]], dtype=self.dtype,device=self.device)
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
        jitter_eps = 1e-11
        jitter = jitter_eps * tr.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        
        try:
            return tr.linalg.solve(A + jitter, b)
        except RuntimeError:
            print("Solve failed even with jitter. Increasing jitter...")
            jitter = 1e-9 * tr.eye(A.shape[-1], device=A.device, dtype=A.dtype)
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
        if tr.isnan(sigma).any(): raise ValueError("NaN detected in sigma inside action")
        
        Sb = (self.Nf / (2 * self.lam)) * tr.sum(sigma**2, dim=(1,2))
        Sf = self.fermion_action(sigma)
        return Sb - Sf

    def force(self, sigma):
        #bosonic force
        F_bos = - (self.Nf / self.lam) * sigma
        
        self.fermion_action(sigma)
        X = self.last_solve_X
        
        if tr.isnan(X).any(): 
            return tr.zeros_like(sigma)

        D = self.build_D_unscaled(sigma)
        Y = tr.einsum('bij, bj -> bi', D, X)
        
        X_view = X.view(self.Bs, self.L, self.L, 2)
        Y_view = Y.view(self.Bs, self.L, self.L, 2)
        dot_prod = tr.sum(tr.conj(X_view) * Y_view, dim=3)
        
        F_ferm = 2.0 * dot_prod.real
        
        return F_bos - F_ferm

