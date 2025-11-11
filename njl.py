#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:30:00 2025

@author: Yamil Cahuana Medrano
"""

import numpy as np
import torch as tr


class njl:
    def __init__(self,V,l,m,Nf,batch_size=1,device="cpu",dtype=tr.float32): 
        self.V = tuple(V) # lattice size
        self.L = V[0]  # assuming square lattice
        self.kappa = 1/(2*(m + 4))  # Wilson parameter?
        self.Vol = np.prod(V)
        self.Nd = len(V)
        self.Nf = Nf  # number of fermion flavors
        self.lam = l # the coupling
        self.mass  = m
        self.Bs=batch_size
        self.device=device
        self.dtype=dtype
        if dtype==tr.float32:
            self.cdtype=tr.cfloat
        else:
            self.cdtype=tr.cdouble

    def gamma_matrices(self):
        gamma0 = tr.tensor([[0., 1.], [1., 0.]])         # sigma_x
        gamma1 = tr.tensor([[0., -1j], [1j, 0.]])        # sigma_y
        return [gamma0.to(tr.cdouble), gamma1.to(tr.cdouble)]

    def site_index(self, x, y, L):
        return (x % L) + L * (y % L)

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
        """leapfrog update of field"""
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
    

    def build_DW_wilson(self, sigma):
        """
        Build the 2D Wilson Dirac operator D_W(x,y) for a batch of sigma fields.
        CHANGE: still with for loops, this is inefficient. Check flavor structure and sigma field symmetry
        sigma shape: (Bs, L, L)
        Output: D shape: (Bs, 2L^2, 2L^2), complex dtype the 2 is because of spinor components not flavors.
        """
        L = self.L
        Bs = self.Bs
        kappa = self.kappa
        V = L * L
        dim = 2 * V

        D = tr.zeros((Bs, dim, dim), dtype=self.cdtype, device=self.device)
        gamma = self.gamma_matrices()
        eye2 = tr.eye(2, dtype=self.cdtype, device=self.device)

        for x in range(L):
            for y in range(L):
                i = self.site_index(x, y, L)

                # Diagonal: (1 + 2kappa * sigma[b,x,y])
                diag_val = (1.0 + 2.0 * kappa * sigma[:, x, y]).to(self.cdtype)  # shape: (Bs,)

                for a in range(2):
                    idx = 2 * i + a
                    D[:, idx, idx] = diag_val  # broadcast to batch

                # Off-diagonal: hopping terms
                for mu in range(2):
                    dx, dy = (1, 0) if mu == 0 else (0, 1)
                    gamma_mu = gamma[mu].to(self.device)

                    j_plus = self.site_index(x + dx, y + dy, L)
                    j_minus = self.site_index(x - dx, y - dy, L)

                    hop_plus = (eye2 - gamma_mu) * (-kappa * 0.5)  # shape (2, 2)
                    hop_minus = (eye2 + gamma_mu) * (-kappa * 0.5)

                    for a in range(2):
                        for b in range(2):
                            idx_from = 2 * i + a
                            idx_p = 2 * j_plus + b
                            idx_m = 2 * j_minus + b

                            D[:, idx_from, idx_p] += hop_plus[a, b]
                            D[:, idx_from, idx_m] += hop_minus[a, b]

        return D

    def pseudofermion_field(self,Dirac_op):
        """
        Generate pseudofermion field phi with equations on gattringer 8.1.3
        Output: phi of size (Bs, 2*L*L), complex dtype

        """
        V=self.L*self.L
        dim = 2 * V
        phi_real = tr.randn([self.Bs, dim], dtype=self.dtype, device=self.device)
        phi_imag = tr.randn([self.Bs, dim], dtype=self.dtype, device=self.device)
        phi = phi_real + 1j * phi_imag
        #Dp= Dirac_op.conj().transpose(-2,-1)

        phi = tr.einsum('bxy,by->bx',Dirac_op, phi)

        return phi
    def fermion_action(self, Dirac_op, phi):

        """
        Compute the fermion action S_f = phi^dagger (D_W^dagger D_W)^{-1} phi
        Output: action value for each batch element, size (Bs,) or we can sum over batch and return a scalar
        Complete this function
        """
        return 0

    def action(self, sigma, phi,Dirac_op):
        sig2=sigma*sigma
        A = tr.sum((self.Nf/2*self.lam)*sig2,dim=(1,2))+ self.fermion_action(Dirac_op, phi)
        return A