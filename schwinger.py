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
    #TODO: complex x?
    def generate_Pseudofermions(self, d):
        
        #Scalar random numbers
        #x = tr.normal(0.0,1.0,[self.Bs, self.V[0]*self.V[1]*self.Nd],dtype=tr.complex64,device=self.device)

        #random complex numbers
        real = tr.randn([self.Bs, self.V[0]*self.V[1]*self.Nd], dtype=tr.float64)
        imag = tr.randn([self.Bs, self.V[0]*self.V[1]*self.Nd], dtype=tr.float64)
        x = tr.complex(real, imag)
        
        p = tr.einsum('bxy, by-> bx', d.to_dense(), x)

        return p

    #Input: self, Bx(VolxNd)x(VolxNd) dirac operator d, Bx(VolxNd) pseudofermion field p
    #Output: Bx1 fermion action
    def fermionAction(self, d, p):
        d_dense = d.to_dense()
        d_inv = tr.inverse(d_dense)
        d_dag_inv = tr.inverse(d_dense.conj().transpose(1,2))
        return tr.einsum('bx, bxy, byz, bz->b', tr.conj(p), d_dag_inv, d_inv, p)

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
    
    #Input: Inverse dirac operator, source timeslices to average over, spatial momentum
    #timeslice max
    #Output: Batch x Vector of Correlation functions for each time slice
    def exact_Pion_Correlator(self, d_inv, s_range, p=0.0, ts_max = -1):

        #Default timeslice max-set to full length of lattice
        if ts_max==-1:
            ts_max = self.V[0]

        ev = tr.zeros([self.Bs, ts_max])


        #Set source on each selected timeslices and average later
        for ts in s_range:
            #spacetime lattice index of source
            sx = self.V[1]*ts

            for nt in np.arange(ts_max):
                c = tr.zeros(self.Bs)
                for nx in np.arange(self.V[1]):
                    #Must be doubled to account for dirac space!
                    n = 2*(self.V[1] * nt + nx)

                    s1 = d_inv[:, n:n+2, 2*sx:2*sx+2]
                    s2 = tr.einsum('bxy, byz -> bxz', s1, s1.conj().transpose(1,2))

                    #B length vector
                    c = c - np.exp(-1.0j*nx*p)*tr.sum(s2, dim=(1,2))
                #BxL tensor
                # values of c are verified as real- cast them to real to avoid error message        
                ev[:, nt] = ev[:, nt] +  tr.real(c)

        #Average over the sources
        ev = ev / (len(s_range))

        return ev / np.sqrt(1.0*self.V[1])


    #Uses BICGStab to find the inexact propogator, rather than the exact inverse matrix
    #Created for comparision purposes with the schur complement propogator
    def approx_Propogator(self, d, sx):

        # build batch of solution vectors for point source - one for each dirac index
        source_vec = tr.zeros(self.Bs, d.size(dim=1))
        source_vec2 = tr.zeros(self.Bs, d.size(dim=1))
      
        #Mark the point source in the solution vector-
        source_vec[:, 2*sx] = 1.0
        source_vec2[:,2*sx+1] = 1.0
        d_np = d.to_dense().numpy()
        source_vec_np = source_vec.numpy()
        source_vec2_np = source_vec2.numpy()

        #Additional dimension for Dirac space
        prop = tr.zeros([self.Bs, d.size(dim=2),2], dtype=tr.complex64)
        for b in np.arange(self.Bs):
            #Note- transposing Dirac matrix seems to work? Is the whole structure backwards?
            prop[b, :, 0] = tr.from_numpy(sp.sparse.linalg.bicgstab(d_np[b, :, :], source_vec_np[b, :], tol=1e-09)[0])
            prop[b, :, 1] = tr.from_numpy(sp.sparse.linalg.bicgstab(d_np[b, :, :], source_vec2_np[b, :], tol=1e-09)[0])

        #Return propogator vectors solved via BICGSTAB
        return prop


    #Input: Dirac operator, source timeslices to average over
    #Output: Batch x Vector of Correlation functions for each time slice
    #Note- assumes spatial momentum zero for propogating state
    #Propogator found with BICGStab rather than exact inverse
    def pion_Correlator(self, d, s_range):

        ev = tr.zeros([self.Bs, self.V[0]])

        for sx in s_range:
            sx = self.V[1] * sx

            fp = self.approx_Propogator(d, sx)

            #lattice site index of subdomains
            n=0

            #Ordering of timeslices
            ts = np.arange(self.V[0])

            #Traverse each lattice site in subdomains
            for nt in ts:
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
                ev[:, nt] = ev[:, nt] + tr.real(c)

        #Average over the sources
        ev = ev / (len(s_range))

        return ev / np.sqrt(1.0*self.V[1])
    
    #Input: Inverse dirac operator, source timeslices to average over, spatial momentum
    #timeslice max
    #Output: Batch x Vector of singlet Correlation functions for each time slice
    #TODO: Needs testing
    def exact_Singlet_Correlator(self, d_inv, s_range, p=0.0, ts_max = -1):

        #Default timeslice max-set to full length of lattice
        if ts_max==-1:
            ts_max = self.V[0]

        ev = tr.zeros([self.Bs, ts_max])


        #Set source on each selected timeslices and average later
        for ts in s_range:
            #spacetime lattice index of source
            sx = self.V[1]*ts

            for nt in np.arange(ts_max):
                c = tr.zeros(self.Bs)
                for nx in np.arange(self.V[1]):
                    #Must be doubled to account for dirac space!
                    n = 2*(self.V[1] * nt + nx)

                    s1 = d_inv[:, n:n+2, 2*sx:2*sx+2]
                    s2 = tr.einsum('bxy, byz -> bxz', s1.conj().transpose(1,2), s1)
                    s3 = tr.einsum('xy, byx-> b', g5, d_inv[:, n:n+2, n:n+2]) * \
                    tr.einsum('xy, byx-> b', g5, d_inv[:, 2*sx:2*sx+2, 2*sx:2*sx+2])

                    #B length vector
                    c = c + np.exp(-1.0j*nx*p)*(-1.0*tr.sum(s2, dim=(1,2)) - 2.0*s3) 
                #BxL tensor
                #TODO: See below- are values of c going to still be real?
                # values of c are verified as real- cast them to real to avoid error message        
                ev[:, nt] = ev[:, nt] +  tr.real(c)

        #Average over the sources
        ev = ev / (len(s_range))

        return ev / np.sqrt(1.0*self.V[1])
    



# Domain Decomposition Development *******************************************

    #Input: Batch of square matrices, rank of approximation
    #Output: Neumann series approximated inverse of matrix batch
    def neumann_Inverse(self, m, r):

        #rank 0 term- batch of identity matrices
        x = tr.eye(m.size(dim=1), dtype=tr.complex64)
        #x = x.reshape((1, m.size(dim=1), m.size(dim=1)))
        x = x.repeat(self.Bs, 1, 1)

        if r == 0:
            return x

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


# Block banded Dirac operator/Schur complement based observables

    #input: Lattice configuration, timeslice for domain boundaries, boundary width
    #Output: Domain Decompostion based block-banded Dirac operator
    #Note assumes 2 subdomains, with width 2 boundaries for naive implementation
    def bb_DiracOperator(self, q, xcut_1, xcut_2, bw):
        #enumeration of lattice sites-flat
        p_f = tr.tensor(np.arange(self.V[0]*self.V[1]))
        #Reshape to match lattice geometry
        p =tr.reshape(p_f, (self.V[0], self.V[1]))

        #define 2 width boundary region indices
        b1 = p[xcut_1:xcut_1+bw, :].reshape(-1,)
        b2 = p[xcut_2:xcut_2+bw, :].reshape(-1,)

        #Subdomain indices
        s1 = p[0:xcut_1, :].reshape(-1,)
        s2 = p[xcut_1 + bw:xcut_2,:].reshape(-1,)
        if (xcut_2 +bw != self.V[0]):
            s1 = tr.cat((p[xcut_2+bw:, :].reshape(-1), s1), dim=0)

        #Reordered indices for block-banded structure
        ri = tr.cat([b1,b2,s1,s2])

        #Use Dirac operator for dynamical fermions
        #Need to build it if its a quenched model
        if len(q) == 1:
            d = self.diracOperator(q[0]).to_dense()
        else:
            d = q[2].to_dense()

        #Block diagonal dirac operator
        bd_d = tr.zeros_like(d)

        #Naive block banded construction
        i = 0
        for n in ri:  
            # #Naive O(n^2) algorithm
            j=0
            for m in ri:
                #Take 2x2 matrix of Dirac space for each index
                bd_d[:, 2*i:2*i+2, 2*j:2*j+2] = d[:, 2*n:2*n+2, 2*m:2*m+2]
                j+=1
            i+=1

        #Return sparse, as that is how normal operator is treated
        return bd_d.to_sparse()
    
    #input: Lattice configuration, timeslice for domain boundaries, boundary width
    #optional input for pre-computed/approximated d00 inverse
    #Output: Domain Decompostion based factorized propogator matrix between subdomains
    #Note assumes 2 subdomains for naive implementation
    def dd_Schur_Propogator(self, q, xcut_1, xcut_2, bw, d00_inv=tr.zeros(2)):

        bb_d = self.bb_DiracOperator(q, xcut_1, xcut_2,bw)

        bb_d = bb_d.to_dense()
        
        #Isolate sub matrices
        #Assumes 2 timeslice boundaries of width bw
        d00 = bb_d[:,0:4*bw*self.V[1], 0:4*bw*self.V[1]]
        d01 = bb_d[:, 0:4*bw*self.V[1], 4*bw*self.V[1]:]
        d10 = bb_d[:, 4*bw*self.V[1]:, 0:4*bw*self.V[1]]
        d11 = bb_d[:, 4*bw*self.V[1]:, 4*bw*self.V[1]:]

        #If function is not given a matrix for the inverse of the boundary region,
        #compute it exactly
        if tr.numel(d00_inv)== 0:
            d00_inv = tr.inverse(d00)

        #Schur complement
        s11 = d11 - tr.einsum('bij, bjk, bkm->bim', d10, d00_inv, d01)

        #Factorized propogator for points in subdomains
        fp = tr.inverse(s11)
        return fp
    

    #Input: inverse Schur complement, starting timeslice of boundarys, boundary width
    #range of source timeslices to average over
    #Output: Hadron correlator computed using inverse of exact Schur complement
    def dd_Exact_Pion_Correlator(self, s_inv, xcut_1, xcut_2, bw, s_range, p=0.0):

        ev = tr.zeros([self.Bs, self.V[0]])

        for sx in s_range:
            sx = self.V[1] * sx

            #lattice site index of subdomains
            n=0

            #Corrected ordering of timeslices if neccesary
            ts = np.arange(self.V[0])
            if (xcut_2 + bw < self.V[0]):
                ts = np.roll(ts, self.V[0] - (xcut_2+bw))

            #Traverse each lattice site in subdomains
            for nt in ts:
                #Skip timeslice if its in boundary area
                if (nt not in np.arange(xcut_1, xcut_1+bw) and nt not in np.arange(xcut_2, xcut_2+bw)):
                    c = tr.zeros(self.Bs)
                    #Each spatial index on a timeslice
                    for nx in np.arange(self.V[1]):
                        s1 = s_inv[:, n:n+2, 2*sx:2*sx+2]
                        s2 = tr.einsum('bxy, bzy -> bxz', s1, s1.conj())

                        #B length vector
                        c = c - np.exp(-1.0j*nx*p)*tr.sum(s2, dim=(1,2))
                        #Iterate to next spatial site
                        n += 2

                    #BxL tensor      
                    ev[:, nt] = ev[:, nt] + tr.real(c)

        #Average over the sources
        ev = ev / (len(s_range))

        return ev / np.sqrt(1.0*self.V[1])
    
    #Input: inverse Schur complement, starting timeslice of boundarys, boundary width
    #range of source timeslices to average over
    #Output: Hadron correlator computed using inverse of exact Schur complement
    #TODO: Needs testing
    def dd_Exact_Singlet_Correlator(self, s_inv, xcut_1, xcut_2, bw, s_range, p=0.0):

        ev = tr.zeros([self.Bs, self.V[0]])

        for sx in s_range:
            sx = self.V[1] * sx

            #lattice site index of subdomains
            n=0

            #Corrected ordering of timeslices if neccesary
            ts = np.arange(self.V[0])
            if (xcut_2 + bw < self.V[0]):
                ts = np.roll(ts, self.V[0] - (xcut_2+bw))

            #Traverse each lattice site in subdomains
            for nt in ts:
                #Skip timeslice if its in boundary area
                if (nt not in np.arange(xcut_1, xcut_1+bw) and nt not in np.arange(xcut_2, xcut_2+bw)):
                    c = tr.zeros(self.Bs)
                    #Each spatial index on a timeslice
                    for nx in np.arange(self.V[1]):
                        s1 = s_inv[:, n:n+2, 2*sx:2*sx+2]
                        s2 = tr.einsum('bxy, bzy -> bxz', s1, s1.conj())
                        s3 = tr.einsum('xy, byx-> b', g5, s_inv[:, n:n+2, n:n+2]) * \
                        tr.einsum('xy, byx-> b', g5, s_inv[:, 2*sx:2*sx+2, 2*sx:2*sx+2])

                        #B length vector
                        c = c + np.exp(-1.0j*nx*p)*(-1.0*tr.sum(s2, dim=(1,2)) - 2.0*s3)

                        #Iterate to next spatial site
                        n += 2

                    #BxL tensor      
                    ev[:, nt] = ev[:, nt] + tr.real(c)

        #Average over the sources
        ev = ev / (len(s_range))

        return ev / np.sqrt(1.0*self.V[1])

        



    
    #Input: block banded D operator, cut timeslices, boundary width, Neumann approx. rank
    #source timeslice range
    #Output: Batch x Vector of Correlation functions for each time slice
    #Note- assumes spatial momentum zero for propogating state
    def dd_Pion_Correlator(self, bb_d, xcut_1, xcut_2, bw, r, s_range):

        ev = tr.zeros([self.Bs, self.V[0]])

        for sx in s_range:
            sx = self.V[1] * sx

            fp = self.dd_Approx_Propogator(bb_d, xcut_1, xcut_2, bw, r, sx)

            #lattice site index of subdomains
            n=0

            #Corrected ordering of timeslices if neccesary
            ts = np.arange(self.V[0])
            if (xcut_2 + bw < self.V[0]):
                ts = np.roll(ts, self.V[0] - (xcut_2+bw))

            #Traverse each lattice site in subdomains
            for nt in ts:
                #Skip timeslice if its in boundary area
                if (nt not in np.arange(xcut_1, xcut_1+bw) and nt not in np.arange(xcut_2, xcut_2+bw)):
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
                    ev[:, nt] = ev[:, nt] + tr.real(c)

        #Average over the sources
        ev = ev / (len(s_range))

        return ev / np.sqrt(1.0*self.V[1])
 

    #input: block banded D operator, timeslice for domain boundaries, boundary width rank of approximation, source index
    #Output: Domain Decompostion based factorized propogator matrix between subdomains
    #Note: assumes 2 subdomains for naive implementation
    def dd_Approx_Propogator(self, bb_d, xcut_1, xcut_2, bw, r, sx):


        bb_d = bb_d.to_dense()
        
        #Isolate sub matrices
        #Assumes 2 width 2 timeslice boundaries
        d00 = bb_d[:,0:4*bw*self.V[1], 0:4*bw*self.V[1]]
        d01 = bb_d[:, 0:4*bw*self.V[1], 4*bw*self.V[1]:]
        d10 = bb_d[:, 4*bw*self.V[1]:, 0:4*bw*self.V[1]]
        d11 = bb_d[:, 4*bw*self.V[1]:, 4*bw*self.V[1]:]

        #Schur complement
        s11 = d11 - tr.einsum('bij, bjk, bkm->bim', d10, self.neumann_Inverse(d00, r), d01)

        # build batch of solution vectors for point source - one for each dirac index
        source_vec = tr.zeros(self.Bs, d11.size(dim=1))
        source_vec2 = tr.zeros(self.Bs, d11.size(dim=1))
        #Adjust sx for boundary as needed-assumes just two boundaries
        if(sx > xcut_2*self.V[1]):
            sx = sx - 2*bw*self.V[1]
        elif(sx > xcut_1*self.V[1]):
            sx = sx - bw*self.V[1]
        #Also adjust potential movement of the timeslice
        if(xcut_2 + bw < self.V[0]):
            sx = sx + (self.V[0] - (xcut_2 + bw))*self.V[1]
        #Mark the point source in the solution vector-
        source_vec[:, 2*sx] = 1.0
        source_vec2[:,2*sx+1] = 1.0
        s11_np = s11.numpy()
        source_vec_np = source_vec.numpy()
        source_vec2_np = source_vec2.numpy()

        #Additional dimension for Dirac space
        prop = tr.zeros([self.Bs, d11.size(dim=2),2], dtype=tr.complex64)
        for b in np.arange(self.Bs):

            prop[b, :, 0] = tr.from_numpy(sp.sparse.linalg.bicgstab(s11_np[b, :, :], source_vec_np[b, :], tol=1e-9)[0])
            prop[b, :, 1] = tr.from_numpy(sp.sparse.linalg.bicgstab(s11_np[b, :, :], source_vec2_np[b, :], tol=1e-9)[0])

        #Return propogator vectors solved via BICGSTAB
        return prop


# Two Level HMC Functions *******************************************

    #Input: conjugate momentum tensor, boundary timeslices, boundary width
    #Output: conjugate momentum with boundary timeslice momentum frozen   
    def dd_Freeze_P(self, p, xcut_1, xcut_2, bw):
        p[:, :, xcut_1:xcut_1+bw, :] =  0.0
        p[:, :, xcut_2:xcut_2+bw, :] = 0.0
        return p

        
    def dd_Force(self, q, xcut_1, xcut_2, bw, r=-1):
        #Compute gauge force as in the full model

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

        #Zero out the force on the frozen links
        fg[:,:,xcut_1:xcut_1+bw, :] = 0.0
        fg[:,:,xcut_2:xcut_2+bw, :] = 0.0

        #If fermions aren't present, simply return force of gauge field
        if len(q) == 1:
            #Force is already real, force cast for downstream errors
            #Additional negative sign added from Hamilton's eqs.
            return (-1.0)*tr.real(fg).type(self.dtype)
        
        #TODO: dynamical fermion implementation




