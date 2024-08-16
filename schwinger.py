#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 2023

@author: Ben Slimmer
"""

import numpy as np;
import torch as tr;
import update as h;
import integrators as i;
import time;
import matplotlib.pyplot as plt;
import pandas as pd;

gamma = tr.tensor([[[0.0, 1.0], [1.0, 0.0]], [[0, -1.0j], [1.0j,0]]])
g5 = tr.tensor([[1.0, 0], [0, -1.0]], dtype=tr.complex64)

class schwinger():


    def __init__(self,V,l,m,batch_size=1,device="cpu",dtype=tr.float32): 
        self.V = tuple(V) # lattice size
        self.Vol = np.prod(V)
        self.Nd = len(V)
        self.lam = l # the coupling
        self.mass  = m
        self.mtil = m + self.Nd
        self.Bs=batch_size
        self.device=device
        self.dtype=dtype


    #Section for Dirac operator and some needed functionality for manipulating it: ************************************************
    


    #Inputs: self, gauge field u
    #Outputs: Dirac operator
    def diracOperator(self, u):
        d0 = tr.zeros([self.Bs, self.V[0]*self.V[1]*self.Nd, self.V[0]*self.V[1]*self.Nd], dtype=tr.complex64)
        d= d0.to_sparse()
       
        #enumeration of lattice sites
        p_f = tr.tensor(np.arange(self.V[0]*self.V[1]))
        #Reshape to match lattice geometry
        p =tr.reshape(p_f, (self.V[0], self.V[1]))

        for mu in [0, 1]:
            #Forward shifted indices
            p_s =  tr.roll(p, shifts = -1, dims=mu)
            #Flatten the 2D reps of lattice/field to one dimension
            p_sf = tr.reshape(p_s, (-1,))
            u_f = tr.reshape(u[:, mu, :, :], (self.Bs, self.V[0]*self.V[1]))
            d_dir = tr.zeros([self.Bs, self.V[0]*self.V[1], self.V[0]*self.V[1]], dtype=tr.complex64)
            d_dir[:, p_f, p_sf] = u_f
            d = d + tr.kron(d_dir, -0.5*(tr.eye(2) - gamma[mu])).to_sparse()

            #Backwards shifted indices
            p_s =  tr.roll(p, shifts = +1, dims=mu)
            #Flatten the 2D reps of lattice/field to one dimension
            p_sf = tr.reshape(p_s, (-1,))
            u_f = tr.reshape(tr.conj(tr.roll(u[:, mu, :, :], shifts=1, dims=mu+1)), (self.Bs, self.V[0]*self.V[1]))
            d_dir = tr.zeros([self.Bs, self.V[0]*self.V[1], self.V[0]*self.V[1]], dtype=tr.complex64)
            d_dir[:, p_f, p_sf] = u_f
            d = d + tr.kron(d_dir, -0.5*(tr.eye(2) + gamma[mu])).to_sparse()

        #Mass terms
        d_dir = tr.zeros([self.Bs, self.V[0]*self.V[1], self.V[0]*self.V[1]], dtype=tr.complex64)
        d_dir[:, p_f, p_f] = self.mtil
        d = d+ tr.kron(d_dir, tr.eye(2)).to_sparse()
            

        return d


    #Inputs:, self, Bx(VolxNd)x(VolxNd) dirac operator d
    #outputs: Bx(VolxNd) pseudofermion field
    def generate_Pseudofermions(self, d):
        x = tr.normal(0, 2.0*np.pi, [self.Bs, self.V[0]*self.V[1]*self.Nd], dtype=tr.complex64)
        p = tr.einsum('bxy, by-> bx', d.to_dense(), tr.exp(x))
        return p

    #Input: self, Bx(VolxNd)x(VolxNd) dirac operator d, Bx(VolxNd) pseudofermion field p
    #Output: Bx1 fermio action
    def fermionAction(self, d, p):
        p1 = tr.einsum('bxy, by -> bx', d.to_dense(), p)
        return tr.einsum('bx, bx -> b', tr.conj(p), p1)

#********************************************************************************************************************************


#Gauge theory ****************************************************************************************************************
    #Inputs: self, BxLxLxmu gauge field u 
    #Outputs: Pure gauge action, Bx1
    #TODO: Generalize to higher dimension lattices
    def gaugeAction(self, u):
        #plaquette matrix
        pl = u[:,0,:,:]*tr.roll(u[:,1,:,:], shifts=-1, dims=1) \
            * tr.conj(tr.roll(u[:,0,:,:], shifts=-1, dims=2))*tr.conj(u[:,1,:,:]) 
        
        S = (1/self.lam**2) *(tr.sum(tr.real(tr.ones_like(pl) - pl),dim=(1,2)))
        return S



    #TODO: Write fermion action to be able to include it in total action
    #Inputs: self, tuple q of theory
    #Outputs: Action of the given configuration
    def action(self, q):
        #Check if fermions are part of theory
        if len(q) == 1:
            return self.gaugeAction(q[0]).type(self.dtype)
        else:
            #Absolute value for the slightly complex fermion action?
            return self.gaugeAction(q[0]).type(self.dtype) + tr.abs(self.fermionAction(q[2], q[1])).type(self.dtype)
    

    #TODO: Write fermion force contribution
    #Inputs: self, tuple of gauge field, fermion field, dirac operator
    #Output: force tensor for updating momentum tensor in HMC
    def force(self, q):
        #Isolate gauge field
        u = q[0]
        #A matrix of 'staples'
        a = tr.zeros_like(q[0])


        
        a[:,0,:,:]  = tr.roll(u[:,1,:,:], shifts=-1, dims=1) * tr.conj(tr.roll(u[:,0,:,:], shifts=-1, dims=2))*tr.conj(u[:,1,:,:]) \
                      + tr.conj(tr.roll(u[:,1,:,:], shifts=(-1,1), dims= (1,2)))*tr.conj(tr.roll(u[:,0,:,:], shifts=1, dims=2)) * tr.roll(u[:,1,:,:], shifts=1, dims=2)
        a[:,1,:,:] = tr.conj(tr.roll(u[:,0,:,:], shifts=(1,-1), dims=(1,2))) * tr.conj(tr.roll(u[:,1,:,:], shifts=1, dims=1)) * tr.roll(u[:,0,:,:], shifts=1, dims=1) \
                     + tr.roll(u[:,0,:,:], shifts=-1, dims=2) * tr.conj(tr.roll(u[:,1,:,:], shifts=-1, dims=1)) * tr.conj(u[:,0,:,:])
        #gauge action contribution
        #Additional negative sign added from Hamilton's eqs.
        fg = (-1.0)*(-1.0j* (1.0/self.lam**2)/2.0)* (u*a - tr.conj(a)*tr.conj(u))

        #TODO: fermion action contribution
        #Quenched approximation
        ff = tr.zeros_like(u)
            


        #force is already be real... force the cast for downstream errors
        return (tr.real(fg+ ff)).type(self.dtype)
        #return fg + ff


    def kinetic(self,P):
        return tr.einsum('buxy,buxy->b',P,P)/2.0 
    
    #HMC refresh of momentum
    def refreshP(self):
        P = tr.normal(0.0,1.0,[self.Bs,2,self.V[0],self.V[1]],dtype=self.dtype,device=self.device)
        return P
    
    #HMC position evolve 
    def evolveQ(self,dt,P,Q):
        #Update the gauge field itself
        u_upd = Q[0]*tr.exp(1.0j*dt*P)
        #Update dirac operator with new field if fermions are in theory
        if len(Q) == 1:
            return (u_upd,)
        else:
            d_upd = self.diracOperator(u_upd)
            return (u_upd, Q[1], d_upd)
    
    #Input: self
    #Output: Hot started U[1] gauge field of batch*lattice size.
        #the final dimension of 2 is for link variable in t, then x direction
    def hotStart(self):
        alpha = tr.normal(0.0, 2*np.pi, [self.Bs, 2, self.V[0], self.V[1]],
                      dtype=self.dtype,device=self.device)
        u = tr.exp(1.0j*alpha)
        return u
    
    #Input: self
    #Output: Cold started U[1] gauge field of batch*lattice size.
        #the final dimension of 2 is for link variable in t, then x direction
    def coldStart(self):
        return tr.ones(self.Bs, 2, self.V[0], self.V[1], dtype=tr.complex64, device=self.device)
    
    #Input: Inverse dirac operator
    #Output: Batch x Vector of Correlation functions for each time slice
    #Note- assumes spatial momentum zero for propogating state
    def pi_plus_correlator(self, d_inv):

        ev = tr.zeros([self.Bs, self.V[0]])
        c = tr.zeros(self.Bs)
        for nt in np.arange(self.V[0]):
            c = tr.zeros(self.Bs)
            for nx in np.arange(self.V[1]):
                #Must be doubled to account for dirac space!
                n = 2*(self.V[1] * nt + nx)
                s1 = d_inv[:, n:n+2, 0:2]
                s2 = tr.einsum('bxy, byz -> bxz', s1, s1.conj().transpose(1,2))
                #B length vector
                c = c - tr.sum(s2, dim=(1,2))
            #BxL tensor        
            ev[:, nt] = c


        return ev / np.sqrt(1.0*self.V[1])
                
        
        # #Dirac inverse in tensor format
        # d_inv_tsr = tr.reshape(d_inv, [self.Bs, self.V[0], self.V[1], self.V[0], self.V[1], 2, 2])

        # #BxLxLx2x2 tensors
        # sf = d_inv_tsr[:, 0, 0, :, :, :, :]
        # sb = d_inv_tsr[:,:,:, 0, 0, :, :]

        # #BxLxL tensor
        # #TODO: Same as abs(D^-1)^2 ?
        # c = -1.0*tr.einsum('ij, bxyjk, km, bxymi -> bxy', g5, sf, g5, sb)



        # #BxL tensor
        # return tr.sum(c, dim=2) /np.sqrt(1.0*self.V[1])



    