#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 2023

@author: Ben Slimmer
"""

import numpy as np;
import scipy as sp;
import torch as tr;
import update as h;
import integrators as i;
import time;
import matplotlib.pyplot as plt;
import pandas as pd;
from numba import njit, jit


gamma = tr.tensor([[[1.0,0], [0,-1.0]], [[0,1.0], [1.0,0.0]]], dtype=tr.complex64)
g5 = tr.tensor([[0, -1.0j], [1.0j, 0]])


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


    #Fermions/Dirac operator and some needed functionality for manipulating them: ************************************************
    


    #Inputs: self, gauge field u
    #Outputs: Dirac operator
    def diracOperator(self, u):
        d0 = tr.zeros([self.Bs, self.V[0]*self.V[1]*self.Nd, self.V[0]*self.V[1]*self.Nd], dtype=tr.complex64)
        d= d0.to_sparse()
       
        #enumeration of lattice sites-flat
        p_f = tr.tensor(np.arange(self.V[0]*self.V[1]))
        #Reshape to match lattice geometry
        p =tr.reshape(p_f, (self.V[0], self.V[1]))

        #Apply antiperiodic boundary condition in time to the gauge fields:
        #This is a convuluted way to do it, but an inplace operation breaks autograd...
        bc = tr.zeros([self.Bs, 2, self.V[0], self.V[1]], dtype=tr.complex64)
        bc[:, 0, self.V[0]-1, :] = -2.0*u[:, 0, self.V[0] - 1, :]
        u_bc = u + bc


        for mu in [0, 1]:
            #Forward shifted indices
            p_s =  tr.roll(p, shifts = -1, dims=mu)
            #Flatten the 2D reps of lattice/field to one dimension
            p_sf = tr.reshape(p_s, (-1,))
            u_f = tr.reshape(u_bc[:, mu, :, :], (self.Bs, self.V[0]*self.V[1]))
            d_dir = tr.zeros([self.Bs, self.V[0]*self.V[1], self.V[0]*self.V[1]], dtype=tr.complex64)
            d_dir[:, p_f, p_sf] = u_f
            d = d + tr.kron(d_dir, -0.5*(tr.eye(2) - gamma[mu])).to_sparse()

            #Backwards shifted indices
            p_s =  tr.roll(p, shifts = +1, dims=mu)
            #Flatten the 2D reps of lattice/field to one dimension
            p_sf = tr.reshape(p_s, (-1,))
            u_f = tr.reshape(tr.conj(tr.roll(u_bc[:, mu, :, :], shifts=1, dims=mu+1)), (self.Bs, self.V[0]*self.V[1]))
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
        
        #Scalar random numbers
        x = tr.normal(0.0,1.0,[self.Bs, self.V[0]*self.V[1]*self.Nd],dtype=tr.complex64,device=self.device)
        
        p = tr.einsum('bxy, by-> bx', d.to_dense(), x)

        return p

    #Input: self, Bx(VolxNd)x(VolxNd) dirac operator d, Bx(VolxNd) pseudofermion field p
    #Output: Bx1 fermion action
    def fermionAction(self, d, p):
        d_dense = d.to_dense()
        m =tr.inverse(tr.einsum('bxy, byz-> bxz', d_dense, d_dense.conj().transpose(1,2)))
        p1 = tr.einsum('bxy, by -> bx', m, p)
        return tr.einsum('bx, bx -> b', tr.conj(p), p1)

#********************************************************************************************************************************


#Gauge theory ****************************************************************************************************************
  
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
        #the first dimension of 2 is for direction, first link variable in t, then x direction
    def coldStart(self):
        u= tr.ones(self.Bs, 2, self.V[0], self.V[1], dtype=tr.complex64, device=self.device)

        return u  
  
  
    #Inputs: self, BxmuxLxL gauge field u 
    #Outputs: Pure gauge action, Bx1
    def gaugeAction(self, u):
        #plaquette matrix
        pl = u[:,0,:,:]*tr.roll(u[:,1,:,:], shifts=-1, dims=1) \
            * tr.conj(tr.roll(u[:,0,:,:], shifts=-1, dims=2))*tr.conj(u[:,1,:,:]) 
        
        S = (1/self.lam**2) *(tr.sum(tr.real(tr.ones_like(pl) - pl),dim=(1,2)))
        return S


#HMC functions *****************************************************************************/


    #Inputs: self, tuple q of theory
    #Outputs: Action of the given configuration
    def action(self, q):
        #Check if fermions are part of theory
        if len(q) == 1:
            return self.gaugeAction(q[0]).type(self.dtype)
        else:
            #Absolute value for the slightly complex fermion action?
            return self.gaugeAction(q[0]).type(self.dtype) + tr.abs(self.fermionAction(q[2], q[1])).type(self.dtype)
    


    #Inputs: self, tuple of gauge field, fermion field, dirac operator
    #Output: force tensor for updating momentum tensor in HMC
    #Faster implementation of force function for HMC
    def force(self, q):
        #Isolate gauge field
        u = q[0]
        #A tensor of 'staples'
        a = tr.zeros_like(q[0])

        
        a[:,0,:,:]  = tr.roll(u[:,1,:,:], shifts=-1, dims=1) * tr.conj(tr.roll(u[:,0,:,:], shifts=-1, dims=2))*tr.conj(u[:,1,:,:]) \
                      + tr.conj(tr.roll(u[:,1,:,:], shifts=(-1,1), dims= (1,2)))*tr.conj(tr.roll(u[:,0,:,:], shifts=1, dims=2)) * tr.roll(u[:,1,:,:], shifts=1, dims=2)
        a[:,1,:,:] = tr.conj(tr.roll(u[:,0,:,:], shifts=(1,-1), dims=(1,2))) * tr.conj(tr.roll(u[:,1,:,:], shifts=1, dims=1)) * tr.roll(u[:,0,:,:], shifts=1, dims=1) \
                     + tr.roll(u[:,0,:,:], shifts=-1, dims=2) * tr.conj(tr.roll(u[:,1,:,:], shifts=-1, dims=1)) * tr.conj(u[:,0,:,:])
        #gauge action contribution
        fg = (-1.0j* (1.0/self.lam**2)/2.0)* (u*a - tr.conj(a)*tr.conj(u))


        #If fermions aren't present, simply return force of gauge field
        if len(q) == 1:
            #Force is already real, force cast for downstream errors
            #Additional negative sign added from Hamilton's eqs.
            return (-1.0)*tr.real(fg).type(self.dtype)
        #start = time.time()
        #Otherwise, compute force of the fermion fields
        d_dense = q[2].to_dense()
        f = q[1]

        #Dirac Op. derivative- very similar to dirac operator itself without mass term:
        #Empty bx2x(V)x(Vx2)x(Vx2)
        dd = tr.zeros([self.Bs, self.Nd, self.V[0]*self.V[1], self.V[0]*self.V[1]*self.Nd, self.V[0]*self.V[1]*self.Nd], dtype=tr.complex64)
       
        #enumeration of lattice sites
        p_f = tr.tensor(np.arange(self.V[0]*self.V[1]))
        #Reshape to match lattice geometry
        p =tr.reshape(p_f, (self.V[0], self.V[1]))

        #Apply antiperiodic boundary condition in time to the gauge fields:
        u_bc = u.clone().detach()
        u_bc[:, 0, self.V[0]-1, :] = -1.0*u[:, 0, self.V[0]-1, :]

        for mu in [0,1]:
             #Forward shifted indices
            p_s =  tr.roll(p, shifts = -1, dims=mu)
            #Flatten the 2D reps of lattice/field to one dimension
            p_sf = tr.reshape(p_s, (-1,))
            u_f = tr.reshape(u_bc[:, mu, :, :], (self.Bs, self.V[0]*self.V[1]))
            for i in np.arange(len(p_f)):
                dd[:,mu, p_f[i], 2*p_f[i]:2*p_f[i]+2, 2*p_sf[i]:2*p_sf[i]+2] = dd[:,mu, p_f[i], 2*p_f[i]:2*p_f[i]+2, 2*p_sf[i]:2*p_sf[i]+2] + \
                tr.einsum("b, xy -> bxy",u_f[:,i], (-0.5j*(tr.eye(2) - gamma[mu])))

            #backward shifted indices
            p_s =  tr.roll(p, shifts = +1, dims=mu)
            #Flatten the 2D reps of lattice/field to one dimension
            p_sf = tr.reshape(p_s, (-1,))
            u_f = tr.reshape(tr.conj(tr.roll(u_bc[:, mu, :, :], shifts=1, dims=mu+1)), (self.Bs, self.V[0]*self.V[1]))
            for i in np.arange(len(p_f)):
                dd[:,mu, p_sf[i], 2*p_f[i]:2*p_f[i]+2, 2*p_sf[i]:2*p_sf[i]+2] = dd[:,mu, p_sf[i], 2*p_f[i]:2*p_f[i]+2, 2*p_sf[i]:2*p_sf[i]+2] + \
                      tr.einsum("b, xy -> bxy",u_f[:,i], (0.5j*(tr.eye(2) + gamma[mu])))

        #With Dirac operator gauge link derivative constructed, compute the full fermion force
        
        #D D^dagger inverse - maybe can store this earlier on to reuse?
        ddi = tr.inverse(tr.einsum('bxy, bzy->bxz', d_dense, d_dense.conj()))
        v = tr.einsum('bxy, by ->bx', ddi, f)
        
        a = tr.einsum('bx, bimxy, bzy, bz->bim',v.conj(), dd, d_dense.conj(), v)

        ff = -1.0 *(a + a.conj())

        #Match to geometry of gauge force object
        ff= tr.reshape(ff, [self.Bs, 2, self.V[0], self.V[1]])    

        #fermion force has small complex components.. take real part only?
        #Additional negative sign added from Hamilton's eqs.
        return ((-1.0)*tr.real(fg+ ff)).type(self.dtype)    

 

    #Testing an autograd approach to building force
    def autograd_force(self, q):
        #Gauge theory derivative is quick, just build it analytically
        #Isolate gauge field
        u = q[0]
        #A matrix of 'staples'
        a = tr.zeros_like(q[0])


        
        a[:,0,:,:]  = tr.roll(u[:,1,:,:], shifts=-1, dims=1) * tr.conj(tr.roll(u[:,0,:,:], shifts=-1, dims=2))*tr.conj(u[:,1,:,:]) \
                      + tr.conj(tr.roll(u[:,1,:,:], shifts=(-1,1), dims= (1,2)))*tr.conj(tr.roll(u[:,0,:,:], shifts=1, dims=2)) * tr.roll(u[:,1,:,:], shifts=1, dims=2)
        a[:,1,:,:] = tr.conj(tr.roll(u[:,0,:,:], shifts=(1,-1), dims=(1,2))) * tr.conj(tr.roll(u[:,1,:,:], shifts=1, dims=1)) * tr.roll(u[:,0,:,:], shifts=1, dims=1) \
                     + tr.roll(u[:,0,:,:], shifts=-1, dims=2) * tr.conj(tr.roll(u[:,1,:,:], shifts=-1, dims=1)) * tr.conj(u[:,0,:,:])
        #gauge action contribution
        fg = (-1.0j* (1.0/self.lam**2)/2.0)* (u*a - tr.conj(a)*tr.conj(u))


        #If fermions aren't present, simply return force of gauge field
        if len(q) == 1:
            #Force is already real, force cast for downstream errors
            #Additional negative sign added from Hamilton's eqs.
            return (-1.0)*tr.real(fg).type(self.dtype)
        
        #Otherwise, run the autograd procedure on the fermion action
        d_dense = q[2].to_dense()
        f = q[1]
        #Seek derivative wrt gauge field...
        x_tensor = u.detach()
        x_tensor.requires_grad_()
        y = self.fermionAction(self.diracOperator(x_tensor), f)
        y.backward(tr.ones_like(y))
        ff = x_tensor.grad

        #Match to geometry of gauge force object
        ff= tr.reshape(ff, [self.Bs, 2, self.V[0], self.V[1]])

        #fermion force has small complex components.. take real part only?
        #Additional negative sign added from Hamilton's eqs.
        return ((-1.0)*tr.real(fg+ ff)).type(self.dtype)




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
    
    
    #Input: Inverse dirac operator
    #Output: Batch x Vector of Correlation functions for each time slice
    #Note- assumes spatial momentum zero for propogating state
    def pi_plus_correlator(self, d_inv):

        ev = tr.zeros([self.Bs, self.V[0]])
        #c = tr.zeros(self.Bs)
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
            # values of c are verified as real- cast them to real to avoid error message        
            ev[:, nt] = tr.real(c)


        return ev / np.sqrt(1.0*self.V[1])
    



# Domain Decomposition Development *******************************************


    #input: Lattice configuration, timeslice for domain boundaries
    #Output: Domain Decompostion based block-banded Dirac operator
    #Note assumes 2 subdomains, with width 2 boundaries for naive implementation
    def bb_DiracOperator(self, q, xcut_1, xcut_2):
        #enumeration of lattice sites-flat
        p_f = tr.tensor(np.arange(self.V[0]*self.V[1]))
        #Reshape to match lattice geometry
        p =tr.reshape(p_f, (self.V[0], self.V[1]))

        #define 2 width boundary region indices
        b1 = p[xcut_1:xcut_1+2, :].reshape(-1,)
        b2 = p[xcut_2:xcut_2+2, :].reshape(-1,)

        #Subdomain indices
        s1 = p[0:xcut_1, :].reshape(-1,)
        s2 = p[xcut_1 + 2:xcut_2,:].reshape(-1,)

        #Reordered indices for block-banded structure
        ri = tr.cat([b1,b2,s1,s2])

        d = q[2].to_dense()

        #Block diagonal dirac operator
        bd_d = tr.zeros_like(d)

        #Naive block banded construction
        i = 0
        for n in ri:
            #O(n) approach- in development
            #Fill in Dirac space matrix at each re-ordered spacetime index
            # bd_d[:, 2*i,0::2] = tr.index_select(d[:, 2*n, :], 1, 2*ri)
            # bd_d[:,2*i, 1::2] = tr.index_select(d[:, 2*n, :], 1, 2*ri + 1)
            # bd_d[:, 2*i+1,0::2] = tr.index_select(d[:, 2*n+1, :], 1, 2*ri)
            # bd_d[:,2*i+1, 1::2] = tr.index_select(d[:, 2*n+1, :], 1, 2*ri + 1)

            
            # #Naive O(n^2) algorithm
            j=0
            for m in ri:
                #Take 2x2 matrix of Dirac space for each index
                bd_d[:, 2*i:2*i+2, 2*j:2*j+2] = d[:, 2*n:2*n+2, 2*m:2*m+2]
                j+=1
            i+=1

        #Return sparse, as that is how normal operator is treated
        return bd_d.to_sparse()
    
    #input: Lattice configuration, timeslice for domain boundaries
    #Output: Domain Decompostion based factorized propogator matrix between subdomains
    #Note assumes 2 subdomains for naive implementation
    def dd_Factorized_Propogator(self, q, xcut_1, xcut_2):

        bb_d = self.bb_DiracOperator(q, xcut_1, xcut_2)

        bb_d = bb_d.to_dense()
        
        #Isolate sub matrices
        #Assumes 2 width 2 timeslice boundaries
        d00 = bb_d[:,0:8*self.V[1], 0:8*self.V[1]]
        d01 = bb_d[:, 0:8*self.V[1], 8*self.V[1]:]
        d10 = bb_d[:, 8*self.V[1]:, 0:8*self.V[1]]
        d11 = bb_d[:, 8*self.V[1]:, 8*self.V[1]:]

        #Schur complement
        s11 = d11 - tr.einsum('bij, bjk, bkm->bim', d10, tr.inverse(d00), d01)

        #Factorized propogator for points in subdomains
        fp = tr.inverse(s11)
        return fp
    
    #Input: Factorized propogator matrix, cut timeslices assuming width 2
    #Output: Batch x Vector of Correlation functions for each time slice
    #Note- assumes spatial momentum zero for propogating state
    def dd_Pi_Plus_Correlator(self, fp, xcut_1, xcut_2):

        ev = tr.zeros([self.Bs, self.V[0]])
        #lattice site index of subdomains
        n=0
        #Time slice on full lattice
        nt = 0
        #Traverse each lattice site in subdomains
        while n < fp.size(dim=1)-2:
            c = tr.zeros(self.Bs)
            #Each spatial index on a timeslice
            for nx in np.arange(self.V[1]):
                s1 = fp[:, n:n+2, 0:2]
                s2 = tr.einsum('bxy, bzy -> bxz', s1, s1.conj())

                #B length vector
                c = c - tr.sum(s2, dim=(1,2))
                #Iterate to next spatial site
                n += 2

            #BxL tensor      
            ev[:, nt] = tr.real(c)
            #If next time slice is a boundary slice, skip them in correlator vector
            if (nt + 1 == xcut_1 or nt+1 ==xcut_2):
                nt += 3
            else:
                nt +=1

        return ev / np.sqrt(1.0*self.V[1])
 
    #Input: Batch of square matrices, rank of approximation
    #Output: Neumann series approximated inverse of matrix batch
    def neumann_Inverse(self, m, r):

        #rank 0 term- batch of identity matrices
        x = tr.eye(m.size(dim=1))
        x = x.reshape((1, m.size(dim=1), m.size(dim=1)))
        x = x.repeat(self.Bs, 1, 1)

        #rank 1 term - Identity minus matrix
        base = x - m

        sum = base + x

        if r == 1:
            return sum
        
        #rank 2 and higher terms
        term = base
        for i in np.arange(2, r+1):
            term = tr.einsum('bxy, byz-> bxz', term, base)
            sum = sum + term
        
        return sum
    

    #input: Lattice configuration, timeslice for domain boundaries, rank of approximation, source index
    #Output: Domain Decompostion based factorized propogator matrix between subdomains
    #Note: assumes 2 subdomains for naive implementation
    #TODO: In development
    def dd_Approx_Propogator(self, q, xcut_1, xcut_2, r, sx):

        bb_d = self.bb_DiracOperator(q, xcut_1, xcut_2)

        bb_d = bb_d.to_dense()
        
        #Isolate sub matrices
        #Assumes 2 width 2 timeslice boundaries
        d00 = bb_d[:,0:8*self.V[1], 0:8*self.V[1]]
        d01 = bb_d[:, 0:8*self.V[1], 8*self.V[1]:]
        d10 = bb_d[:, 8*self.V[1]:, 0:8*self.V[1]]
        d11 = bb_d[:, 8*self.V[1]:, 8*self.V[1]:]

        #Schur complement
        s11 = d11 - tr.einsum('bij, bjk, bkm->bim', d10, self.neumann_Inverse(d00, r), d01)

        # build batch of solution vectors for point source
        source_vec = tr.zeros(self.Bs, d11.size(dim=1))
        #Adjust sx for boundary as needed-assumes just two boundaries, with one on end
        if(sx > 2*xcut_1*self.V[1]):
            sx = sx - 4*self.V[1]
        source_vec[:, sx]
        s11_np = s11.numpy()
        source_vec_np = source_vec.numpy()

        #TODO: Is there a better way to batch solve these inexactly?
        prop = tr.zeros([self.Bs, d11.size(dim=2)], dtype=tr.complex64)
        for b in np.arange(self.Bs):
            prop[b, :] = tr.from_numpy(sp.sparse.linalg.bicgstab(s11_np[b, :, :], source_vec_np[b, :])[0])

        #Return propogator vectors solved via BICGSTAB
        return prop


            
    

