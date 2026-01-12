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
        real = tr.randn([self.Bs, self.V[0]*self.V[1]*self.Nd], dtype=tr.float32)
        imag = tr.randn([self.Bs, self.V[0]*self.V[1]*self.Nd], dtype=tr.float32)
        x = tr.complex(real, imag).to(tr.complex64)
        
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
    
    #Input:Lattice configuration, timeslices of boundaries, boundary width, num. of subspace
    #eigenvectors sought after
    #Output: Most significant k eigenvectors of the complement space- deflation
    def complement_Deflation_Eigenvectors(self, q, xcut_1, k, ov = 1):
        #Isolate complement Dirac matrix
        d = self.diracOperator(q[0]).to_dense()

        if ov == 1:

            #complement space
            d00 = d[:, 0:xcut_1*2*self.V[1], 0:xcut_1*2*self.V[1]]
        else:
            d_rolled = tr.roll(d, ((ov-1)*2*self.V[1], (ov-1)*2*self.V[1]), dims=(1,2))
            d00 = d_rolled[:, 0:(xcut_1+2*(ov-1))*2*self.V[1], 0:(xcut_1+2*(ov-1))*2*self.V[1]]


        boundary_only = True
        if boundary_only:
            bound_d = tr.zeros((self.Bs, 2*2*ov*self.V[1], 2*2*ov*self.V[1]), dtype=tr.complex64)

            bound_d[:, :2*ov*self.V[1], :2*ov*self.V[1]] = d00[:, :2*ov*self.V[1], :2*ov*self.V[1]]
            bound_d[:, 2*ov*self.V[1]:, 2*ov*self.V[1]:] = d00[:, -2*ov*self.V[1]:, -2*ov*self.V[1]:]

            m = tr.einsum('byx, byz-> bxz', tr.conj(bound_d), bound_d)

        else:
            #Compute eigenvectors of D^dag D:
            #Lets try computing it in the complement space
            m = tr.einsum('byx, byz-> bxz', tr.conj(d00), d00)

        #Try eigh and see if it doesn't overwhelm memory
        L, V = tr.linalg.eigh(m)

        #Sort and return k with lowest eigenvalue
        #sorted_L, sorted_ind = tr.sort(tr.abs(L), descending=True)
        sorted_L, sorted_ind = tr.sort(tr.real(L), descending = False)


        projs = tr.zeros((self.Bs, k+1, 2*self.V[0]*self.V[1]), dtype=tr.complex64)
        
        for b in np.arange(self.Bs):
            transverse = tr.eye(2*self.V[0]*self.V[1])
            for x in np.arange(k):
                if boundary_only:
                    temp = tr.zeros_like(projs[b,x,:], dtype=tr.complex64)
                    temp[:2*ov*self.V[1]] = V[b, :2*ov*self.V[1], sorted_ind[b,x]]
                    temp[-2*ov*self.V[1]:] = V[b, -2*ov*self.V[1]:, sorted_ind[b,x]]
                else:
                    temp = tr.zeros_like(projs[b,x,:], dtype=tr.complex64)
                    temp[:(xcut_1+2*(ov-1))*2*self.V[1]] = V[b, :, sorted_ind[b, x]]

                projs[b, x, :] = tr.roll(temp, -(ov-1)*2*self.V[1], 0)

                transverse = transverse - tr.outer(projs[b,x,:], tr.conj(projs[b,x,:]))/tr.dot(projs[b,x,:], tr.conj(projs[b,x,:]))

            #Collapse transverse space into a vector
            t_vec = transverse.sum(dim=0)
            #Save as final vector
            #projs[b, k, :] = t_vec
            projs[b, k, :] = t_vec/tr.norm(t_vec)

            #Try Graham Schmidt orthogonalization for transverse space instead
            # ones vector for transverse space to keep deterministic
            transverse = tr.ones_like(projs[b,k-1, :], dtype=tr.complex64)
            #transverse = tr.rand_like(projs[b,k-1, :]) + 1.0j*tr.rand_like(projs[b,k-1, :])

            # orthogonalization
            for x in range(k):
                v = projs[b, x, :]
                transverse = transverse - tr.dot(tr.conj(v), transverse) * v

            # re-orthogonalization pass 
            for x in range(k):
                v = projs[b, x, :]
                transverse = transverse - tr.dot(tr.conj(v), transverse) * v

            # Normalize
            norm = tr.norm(transverse)
            transverse = transverse / norm

            projs[b, k, :] = transverse

            #Now try a very coarse transverse space in the empty center
            # B = 2 * ov * self.V[1]
            # N = 2 * self.V[0] * self.V[1]

            # t = tr.zeros(N, dtype=tr.complex64)
            # t[B:N-B] = 1.0   # or tr.randn(N - 2*B)

            # #t = t / tr.norm(t)
            # projs[b, k, :] = t



        return projs
    
    #Input:Lattice configuration, timeslices of boundaries, boundary width, num. of subspace
    #eigenvectors sought after
    #Output: Most significant k distillation
    def boundary_Distillation_Eigenvectors(self, q, xcut_1, k, ov = 1):
        u = q[0]


        
        #Includes Dirac Space in Laplacian

        #Construct the full laplacian then slice it
        u = q[0]

        laplacian = tr.zeros((self.Bs, 2*self.V[0]*self.V[1], 2*self.V[0]*self.V[1]), dtype=tr.complex64)

        #enumeration of lattice sites-flat
        p_f = tr.tensor(np.arange(self.V[0]*self.V[1]))
        #Reshape to match lattice geometry
        p =tr.reshape(p_f, (self.V[0], self.V[1]))

        bc = tr.zeros([self.Bs, 2, self.V[0], self.V[1]], dtype=tr.complex64)
        bc[:, 0, self.V[0]-1, :] = -2.0*u[:, 0, self.V[0] - 1, :]
        u_bc = u + bc

        for mu in [0,1]:
            #Forward shifted indices
            p_s =  tr.roll(p, shifts = -1, dims=mu)
            #Flatten the 2D reps of lattice/field to one dimension
            p_sf = tr.reshape(p_s, (-1,))
            u_f = tr.reshape(u_bc[:, mu, :, :], (self.Bs, self.V[0]*self.V[1]))
            d_dir = tr.zeros([self.Bs, self.V[0]*self.V[1], self.V[0]*self.V[1]], dtype=tr.complex64)
            d_dir[:, p_f, p_sf] = u_f
            laplacian = laplacian + tr.kron(d_dir, tr.eye(2))

            #Backwards shifted indices
            p_s =  tr.roll(p, shifts = +1, dims=mu)
            #Flatten the 2D reps of lattice/field to one dimension
            p_sf = tr.reshape(p_s, (-1,))
            u_f = tr.reshape(tr.conj(tr.roll(u_bc[:, mu, :, :], shifts=1, dims=mu+1)), (self.Bs, self.V[0]*self.V[1]))
            d_dir = tr.zeros([self.Bs, self.V[0]*self.V[1], self.V[0]*self.V[1]], dtype=tr.complex64)
            d_dir[:, p_f, p_sf] = u_f
            laplacian = laplacian + tr.kron(d_dir, tr.eye(2))

        #Diagonal
        d_dir = tr.zeros([self.Bs, self.V[0]*self.V[1], self.V[0]*self.V[1]], dtype=tr.complex64)
        d_dir[:, p_f, p_f] = -4.0
        laplacian = laplacian+ tr.kron(d_dir, tr.eye(2))

        #Reduce to the complement domain
        laplacian_roll = tr.roll(laplacian, ((ov-1)*2*self.V[1], (ov-1)*2*self.V[1]), dims=(1,2))

        r_l = laplacian_roll[:, 0:(xcut_1+2*(ov-1))*2*self.V[1], 0:(xcut_1+2*(ov-1))*2*self.V[1]]
        

        #Or try just the boundaries
        boundary_only = True
        if boundary_only:
            bound_laplacian = tr.zeros((self.Bs, 2*2*ov*self.V[1], 2*2*ov*self.V[1]), dtype=tr.complex64)

            bound_laplacian[:, :2*ov*self.V[1], :2*ov*self.V[1]] = r_l[:, :2*ov*self.V[1], :2*ov*self.V[1]]
            bound_laplacian[:, 2*ov*self.V[1]:, 2*ov*self.V[1]:] = r_l[:, -2*ov*self.V[1]:, -2*ov*self.V[1]:]

            L, V = tr.linalg.eigh(-1.0*bound_laplacian)
        else:
            L, V = tr.linalg.eigh(-1.0*r_l)
        sorted_L, sorted_ind = tr.sort(tr.real(L), descending=False)
        print(sorted_L[0, :])

        projs = tr.zeros((self.Bs, k+1, 2*self.V[0]*self.V[1]), dtype=tr.complex64)

        for b in np.arange(self.Bs):

            for x in np.arange(k):
                if boundary_only:
                    temp = tr.zeros_like(projs[b,x,:], dtype=tr.complex64)
                    temp[:2*ov*self.V[1]] = V[b, :2*ov*self.V[1], sorted_ind[b,x]]
                    temp[-2*ov*self.V[1]:] = V[b, -2*ov*self.V[1]:, sorted_ind[b,x]]
                else:
                    temp = tr.zeros_like(projs[b,x,:], dtype=tr.complex64)
                    temp[:(xcut_1+2*(ov-1))*2*self.V[1]] = V[b, :, sorted_ind[b, x]]
                
                projs[b, x, :] = tr.roll(temp, -(ov-1)*2*self.V[1], 0)

             #Try Graham Schmidt orthogonalization for transverse space instead
            # vector of ones for transverse space
            transverse = tr.ones(2*self.V[0]*self.V[1], dtype=tr.complex64)

            # orthogonalization
            for x in range(k):
                v = projs[b, x, :]
                transverse = transverse - tr.dot(tr.conj(v), transverse) * v

            # re-orthogonalization pass 
            for x in range(k):
                v = projs[b, x, :]
                transverse = transverse - tr.dot(tr.conj(v), transverse) * v

            # Normalize
            norm = tr.norm(transverse)
            transverse = transverse / norm

            projs[b, k, :] = transverse


        projs_orthonormal, r = tr.linalg.qr(tr.transpose(projs, 1,2), mode='reduced')

        
        #print(projs[0, k, :])


        

        return tr.transpose(projs_orthonormal, 1, 2)

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
    
    #Performs index reordering of the fermion vector to match the block banded
    #Dirac operator
    def bb_FermionVector(self, f, xcut_1, xcut_2, bw):
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

        #Block diagonal dirac operator
        bb_f = tr.zeros_like(f)

        #Naive block banded construction
        i = 0
        for n in ri:
            bb_f[:, 2*i:2*i+2] = f[:, 2*n:2*n+2]
            i += 1

        #Return sparse, as that is how normal operator is treated
        return bb_f

    
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
    

    #Input: Lattice configuration, timeslice for boundaries, and boundary width
    #Output:A low rank approximant version of the Schur complement of the Dirac matrix
    #for points in subdomains
    def dd_low_Rank_Schur_Complement(self, q, xcut_1, xcut_2, bw, d00_inv=tr.zeros(2)):

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

        nlcl = tr.pca_lowrank(tr.einsum('bij, bjk, bkm->bim', d10, d00_inv, d01))
        
        s11 = d11 - tr.pca_lowrank(tr.einsum('bij, bjk, bkm->bim', nlcl[0], tr.diag_embed(nlcl[1]), nlcl[2]))

        return s11
    
    #Implementing Giusti's subdomain construction schur complement for testing
    def schur_Complement_Giusti(self, q, xcut_1):

        d = self.diracOperator(q[0]).to_dense()


        complement = d[:, 0:xcut_1*2*self.V[1], 0:xcut_1*2*self.V[1]]
        bulk = d[:,xcut_1*self.V[1]*2:, xcut_1*self.V[1]*2:]

        d01 = d[:, 0:xcut_1*2*self.V[1], xcut_1*self.V[1]*2:]
        d10 = d[:, xcut_1*self.V[1]*2:, 0:xcut_1*2*self.V[1]]

        schur = complement - tr.einsum('bij, bjk, bkm->bim', d01, tr.inverse(bulk), d10)

        return schur

    #factorized propogator using unapproximated Schur complement in the 
    #complement subdomain
    def exact_Factorized_Propogator(self, q, xcut_1, xcut_2, bw, projs):
        #First seperate sections of the Dirac operator
        d = self.diracOperator(q[0]).to_dense()

        #Contains second subdomain and both frozen boundaries
        bulk = d[:,xcut_1*self.V[1]*2:, xcut_1*self.V[1]*2:]

        s1_inv = tr.inverse(self.schur_Complement_Giusti(q, xcut_1))
        bulk_inv = tr.inverse(bulk)

        #Boundary interface
        d10 = d[:, xcut_1*self.V[1]*2:, 0:xcut_1*2*self.V[1]]

        #Product for left factorization piece
        bulk_prod = tr.einsum('bxy, byz->bxz', bulk_inv, d10)


        #Will be same for each of the equal sized overlapping subdomains
        #Dirac indices remain layered into spatial indices here
        subdomain_index_ct = tr.numel(s1_inv[0,0,:])
        bulk_index_ct = tr.numel(bulk[0,0,:])

        ensembleL = tr.zeros([self.Bs, tr.numel(projs[:,0]), bulk_index_ct], dtype=tr.complex64)
        ensembleR = tr.zeros([self.Bs, tr.numel(projs[:,0]), subdomain_index_ct], dtype=tr.complex64)
        summed = tr.zeros([self.Bs, bulk_index_ct, subdomain_index_ct])

        if projs.ndim==2:
            num_proj = tr.numel(projs[:,0])
        else:
            num_proj = tr.numel(projs[0,0,:])

        for xi in tr.arange(tr.numel(projs[:,0])):
            #Need to update to allow for batch of projectors
            if projs.ndim == 2:
                p = projs[xi, :]
                p = p.repeat(self.Bs, 1)
            else:
                p = projs[:, :, xi]

            #Slice projector relevant to complement subdomain size
            projector = p[:, :(xcut_1)*self.V[1]*2]

            ensembleR[:, xi, :] = tr.einsum('bx, bxy->by', tr.conj(projector), s1_inv)

            ensembleL[:, xi, :] = tr.einsum('bxy, by-> bx', bulk_prod, projector)

            summed = summed + tr.einsum('bx, by-> bxy', ensembleL[:, xi, :], 
                                        ensembleR[:, xi, :])
        
        return ensembleL, ensembleR, summed



    
    #Input lattice configuration, frozen timeslices beginnings, boundary width, projection vectors, overlap T/F
    #Output:  batch x intermediate ensemble of factorized propogator contributions to all points on the lattice, one for each subdomain. Contains only first order contributions
    #New version of function based on projectors
    def factorized_Propogator_Proj(self, q, xcut_1, xcut_2, projs, ov=1):

        if ov != 1:
            #First seperate sections of the Dirac operator
            d = self.diracOperator(q[0]).to_dense()

            d_rolled = d.roll(((ov-1)*self.V[1]*2, (ov-1)*self.V[1]*2),
                                dims=(1,2))
            
            #Contains all dirac matrix points within the first subdomain
            s1 = d_rolled[:, :(2*(ov-1) + xcut_1)*self.V[1]*2, 
                          :(2*(ov-1) + xcut_1)*self.V[1]*2]
            
            #Contains second subdomain and both frozen boundaries
            bulk = d[:,xcut_1*self.V[1]*2:, xcut_1*self.V[1]*2:]

            s1_inv = tr.inverse(s1)
            bulk_inv = tr.inverse(bulk)

            #Boundary interface
            d_rolled = d.roll((ov-1)*self.V[1]*2, 2)
            d10 = d_rolled[:, xcut_1*self.V[1]*2:, :(2*(ov-1) + xcut_1)*self.V[1]*2]

            #Product for left factorization piece
            bulk_prod = tr.einsum('bxy, byz->bxz', bulk_inv, d10)

            subdomain_index_ct = tr.numel(s1[0,0,:])
            bulk_index_ct = tr.numel(bulk[0,0,:])

            if projs.ndim==2:
                num_proj = tr.numel(projs[:,0])
            else:
                num_proj = tr.numel(projs[0,:,0])

            ensembleL = tr.zeros([self.Bs, num_proj, bulk_index_ct], dtype=tr.complex64)
            ensembleR = tr.zeros([self.Bs, num_proj, subdomain_index_ct], dtype=tr.complex64)
            summed = tr.zeros([self.Bs, bulk_index_ct, subdomain_index_ct])

            for xi in tr.arange(num_proj):
                #Need to update to allow for batch of projectors
                if projs.ndim == 2:
                    p = projs[xi, :]
                    p = p.repeat(self.Bs, 1)
                else:
                    p = projs[:, xi, :]

                #roll projector for right piece
                p_r = p.roll((ov-1)*self.V[1]*2, 1)

                #Slice projector relevant to complement subdomain size
                projector = p_r[:, :(xcut_1+2*(ov-1))*self.V[1]*2]

                ensembleR[:, xi, :] = tr.einsum('bx, bxy->by', tr.conj(projector), s1_inv)

                ensembleL[:, xi, :] = tr.einsum('bxy, by-> bx', bulk_prod, projector)
                

                summed = summed + tr.einsum('bx, by-> bxy', ensembleL[:, xi, :], 
                                            ensembleR[:, xi, :])

            #Remove overlap spaces from the right complement domain
            ensembleR =ensembleR[:, :, (ov-1)*2*self.V[1]:(self.V[0]+1 -ov)*2*self.V[1]]
            summed = summed[:, :, (ov-1)*2*self.V[1]:(self.V[0]+1 -ov)*2*self.V[1]]

            
            return ensembleL, ensembleR, summed
        
        #Original function assuming minimum overlap below

        #First seperate sections of the Dirac operator
        d = self.diracOperator(q[0]).to_dense()

        #Contains all dirac matrix points within the first subdomain
        s1 = d[:, :xcut_1*self.V[1]*2, :xcut_1*self.V[1]*2]

        #Contains second subdomain and both frozen boundaries
        bulk = d[:,xcut_1*self.V[1]*2:, xcut_1*self.V[1]*2:]

        s1_inv = tr.inverse(s1)
        bulk_inv = tr.inverse(bulk)

        #Boundary interface
        d10 = d[:, xcut_1*self.V[1]*2:, 0:xcut_1*2*self.V[1]]

        #Product for left factorization piece
        bulk_prod = tr.einsum('bxy, byz->bxz', bulk_inv, d10)

        # plt.spy(bulk_prod[0,:,:])

        # plt.show()

        #Will be same for each of the equal sized overlapping subdomains
        #Dirac indices remain layered into spatial indices here
        subdomain_index_ct = tr.numel(s1[0,0,:])
        bulk_index_ct = tr.numel(bulk[0,0,:])

        if projs.ndim==2:
            num_proj = tr.numel(projs[:,0])
        else:
            num_proj = tr.numel(projs[0,:,0])

        ensembleL = tr.zeros([self.Bs, num_proj, bulk_index_ct], dtype=tr.complex64)
        ensembleR = tr.zeros([self.Bs, num_proj, subdomain_index_ct], dtype=tr.complex64)
        summed = tr.zeros([self.Bs, bulk_index_ct, subdomain_index_ct])

        for xi in tr.arange(num_proj):
            #Need to update to allow for batch of projectors
            if projs.ndim == 2:
                p = projs[xi, :]
                p = p.repeat(self.Bs, 1)
            else:
                p = projs[:, xi, :]


            #Slice projector relevant to complement subdomain size
            projector = p[:, :(xcut_1)*self.V[1]*2]

            ensembleR[:, xi, :] = tr.einsum('bx, bxy->by', tr.conj(projector), s1_inv)

            ensembleL[:, xi, :] = tr.einsum('bxy, by-> bx', bulk_prod, projector)

            summed = summed + tr.einsum('bx, by-> bxy', ensembleL[:, xi, :], 
                                        ensembleR[:, xi, :])
        
        return ensembleL, ensembleR, summed



    
    
    #Input: lattice configuration, frozen timeslices, boundary width, intermediate indices, overlap T/F
    #of the Dirac operator, intermediate points
    #Output:  batch x intermediate ensemble of factorized propogator contributions to all points on the lattice, one for each subdomain. Contains only first order contributions
    #NOTE: This approach has a bug and will produce incorrect results. Kept as reference
    #for now but may be removed later
    def factorized_Propogator(self, q, xcut_1, xcut_2, bw, inter, overlap):


        #Attempt 2 - based on Giuisti factorization scheme
        #We're gonna take a full inverse of the subdomain and return a list
        #of the hadron correlator from each boundary point

        #First seperate sections of the Dirac operator
        d = self.diracOperator(q[0]).to_dense()


        #Contains all dirac matrix points within the first subdomain
        #roll the entire end boundary if working with a thick overlap,
        #Only the edge of the boundary for no overlap
        if overlap == True:
            rolled_d = d.roll(shifts=(bw*self.V[1]*2, bw*self.V[1]*2), dims=(1,2))
            s1 = rolled_d[:, :(xcut_1+2*bw)*self.V[1]*2, :(xcut_1+2*bw)*self.V[1]*2]
        else:
            rolled_d = d.roll(shifts=(self.V[1]*2, self.V[1]*2), dims=(1,2))        
            s1 = d[:, :(xcut_1+2)*self.V[1]*2, :(xcut_1+2)*self.V[1]*2]
        #Contains second subdomain and both frozen boundaries

        bulk = d[:,xcut_1*self.V[1]*2:, xcut_1*self.V[1]*2:]


        s1_inv = tr.inverse(s1)
        bulk_inv = tr.inverse(bulk)

        #Will be same for each of the equal sized overlapping subdomains
        subdomain_index_ct = int(tr.numel(s1[0,0,:])/2)
        bulk_index_ct = int(tr.numel(bulk[0,0,:])/2)

        ensembleL = tr.zeros([self.Bs, tr.numel(inter), bulk_index_ct,2], dtype=tr.complex64)
        ensembleR = tr.zeros([self.Bs, tr.numel(inter), subdomain_index_ct, 2], dtype=tr.complex64)
        summed = tr.zeros([self.Bs, bulk_index_ct, subdomain_index_ct, 2, 2])
        for xi in tr.arange(tr.numel(inter)):

            if overlap == True:
                if inter[xi] > 2*xcut_2*self.V[1]:
                    adj_inter = inter[xi] - 2*(self.V[0] - bw)*self.V[1]
                else:
                    adj_inter = inter[xi] + 2*bw*self.V[1]
            else:
                if inter[xi] > 2*xcut_2*self.V[1]:
                    adj_inter = inter[xi] - 2*(self.V[0]-1)*self.V[1]
                else:
                    adj_inter = inter[xi] +2*self.V[1]

            ensembleR[:,xi, :, :] = s1_inv[:, adj_inter, :].view((self.Bs, subdomain_index_ct, 2))

            #Isolate intermediate Dirac entry for boundary crossing
            d_int=d[:, inter[xi], inter[xi]]

            #Compute offset for bulk matrix
            offset = 2*(xcut_1)*self.V[1]
            
            ensembleL[:, xi, :, :] = tr.einsum('bxi, b->bxi', bulk_inv[:, :, inter[xi]-offset].view(
                (self.Bs, bulk_index_ct, 2)),
                                  d_int)
            
            summed = summed + tr.einsum('bxa, byc-> bxyac', ensembleL[:, xi, :, :],
                                        ensembleR[:, xi, :, :])
            
        return ensembleL, ensembleR, summed
    
    #Input:Batch of factorized propogators
    #Output: Measurements of two point correlator binned by timeslice separation'
    #TODO:Testing adding a momentum
    def measure_Factorized_Two_Point_Correlator(self, q, f_propogator, xcut_1, xcut_2, bw, p = 0.0, ov=1):
        
        d_inv = tr.inverse(self.diracOperator(q[0]).to_dense())
        max_length = int((self.V[0])/2)
        
        #1/2 for middle, then x2 for dirac space
        mid_x = self.V[1]

        #Accounts for shifting of the lattice indices in the factorizing process
        sink_adj = -2*(xcut_1)*self.V[1]
        
        sink_ts = xcut_1+bw

        factorized_corr_magnitude = [None]* (max_length)
        corr_magnitude = [None] * (max_length)

        source_max = xcut_1 + (ov-1)

        while sink_ts < self.V[0]-bw:
            source_ts = 0
            while source_ts < xcut_1:
                source_x = mid_x + 2*self.V[1]*source_ts
                sink_x = mid_x + 2*self.V[1]*sink_ts + sink_adj
                factorized_prop = f_propogator[:, sink_x:sink_x+2, source_x:source_x+2]
                factorized_corr = -1.0*tr.sum(tr.einsum('bij, bkj-> bik', factorized_prop, factorized_prop.conj()), dim=(1,2))*np.exp(-1.0j*p*mid_x*0.5)

                #Take an unfactorized measurement as well
                sink_x = sink_x - sink_adj
                prop = d_inv[:, sink_x:sink_x+2, source_x:source_x+2]
                corr = -1.0*tr.sum(tr.einsum('bij, bkj-> bik', prop, prop.conj()), dim=(1,2))*np.exp(-1.0j*p*mid_x*0.5)

                #Save correlator measurements to matrix
                #First record distance between timeslices
                dist = min(sink_ts - source_ts, source_ts + self.V[0] - sink_ts)
                if corr_magnitude[dist-1] == None:
                    factorized_corr_magnitude[dist-1] = factorized_corr
                    corr_magnitude[dist- 1] = corr
                else:
                    # temp = tr.cat((factorized_corr_magnitude[dist- bw -1], factorized_corr), 0)
                    temp = tr.cat((factorized_corr_magnitude[dist-1], factorized_corr), 0)
                    factorized_corr_magnitude[dist-1] = temp
                    #factorized_corr_magnitude[dist - 1, :] = tr.cat((factorized_corr_magnitude[dist - 1, :],
                    #                                                      factorized_corr), dim=0)
                    temp = tr.cat((corr_magnitude[dist-1], corr), 0)
                    corr_magnitude[dist-1] = temp
                    #corr_magnitude[dist - 1, :] = tr.cat((corr_magnitude[dist - 1, :],
                    #                                                      corr), dim=0)
                source_ts += 1
            print("sink position: ", sink_ts)
            sink_ts += 1



        return factorized_corr_magnitude, corr_magnitude
    
    #TODO: Check this still works with newest bugfix
    def measure_Factorized_Pion_Correlator(self, q, f_propogator, xcut_1, xcut_2, bw, p = 0.0, ov=1, factorized_only=False):

        max_length = int((self.V[0])/2)

        if factorized_only == False:
            d_inv = tr.inverse(self.diracOperator(q[0]).to_dense())
        corr = [None] * (max_length)

        factorized_corr = [None]* (max_length)
        
        #1/2 for middle, then x2 for dirac space
        mid_x = self.V[1]

        #Accounts for shifting of the lattice indices in the factorizing process
        sink_adj = -2*(xcut_1)*self.V[1]
        
        sink_t = xcut_1+ bw

        while sink_t < self.V[0] - bw:
            source_t = 0
            while source_t < xcut_1:
                f_c = tr.zeros(self.Bs)
                c = tr.zeros(self.Bs)
                for sink_x in tr.arange(self.V[1]):
                    #TODO: Why does using more than one source spatial index break this?
                    #Probably has to do with distance between points: encode a source-sink difference?
                    for source_x in (0,):
                        source_ind = 2*self.V[1]*source_t + 2*source_x
                        sink_ind = 2*self.V[1]*sink_t + 2*sink_x + sink_adj
                        factorized_prop = f_propogator[:, sink_ind:sink_ind+2, source_ind:source_ind+2]
                        f_c = f_c -  tr.sum(tr.einsum('bij, bkj-> bik', factorized_prop, factorized_prop.conj()), dim=(1,2)) \
                            *np.exp(-1.0j*p*sink_x)

                        #Take an unfactorized measurement as well if needed
                        if factorized_only == False:
                            sink_ind = sink_ind - sink_adj
                            prop = d_inv[:, sink_ind:sink_ind+2, source_ind:source_ind+2]
                            c = c - tr.sum(tr.einsum('bij, bkj-> bik', prop, prop.conj()), dim=(1,2)) \
                                *np.exp(-1.0j*p*sink_x)

                dist = min(sink_t - source_t, source_t + self.V[0] - sink_t)

                #Multiply by fourier transform factor 
                c = (1.0/np.sqrt(1.0*self.V[1]))  * c
                f_c = (1.0/np.sqrt(1.0*self.V[1])) * f_c

                if corr[dist-1] == None:
                    factorized_corr[dist -1] = f_c
                    corr[dist - 1] = c
                else:
                    temp = tr.cat((factorized_corr[dist -1], f_c), 0)
                    factorized_corr[dist -1] = temp
                    #factorized_corr_magnitude[dist - 1, :] = tr.cat((factorized_corr_magnitude[dist - 1, :],
                    #                                                      factorized_corr), dim=0)
                    temp = tr.cat((corr[dist -1], c), 0)
                    corr[dist-1] = temp

                source_t += 1
            #print('sink position: ', sink_t)
            sink_t += 1
        if factorized_only == False:
            return factorized_corr, corr
        else:
            return factorized_corr




    

    #TODO: Compute systematic error between approx and full propogator approach
    #Input: field configuration, frozen boundary timeslices and width, inverse of Dirac matrix in boundaries,
    #source indices, pion momentum
    #Output: difference in the measurements, to be averaged later
    def dd_Pion_Systematic_Error(self, q, xcut_1, xcut_2, bw, d00_inv, s_range, p):
        
        d = self.diracOperator(q[0]).to_dense()
        exact_correlator = self.exact_Pion_Correlator(tr.inverse(d), s_range,p)

        s_inv = self.dd_Schur_Propogator(q, xcut_1, xcut_2, bw, d00_inv)

        dd_correlator = self.dd_Exact_Pion_Correlator(s_inv, xcut_1, xcut_2, bw, s_range, p)

        #avg_diff = tr.mean(exact_correlator, dim=0) - tr.mean(dd_correlator, dim=0)
        #diff_err = tr.sqrt(tr.square(tr.std(exact_correlator, dim=0)) + tr.square(tr.std(dd_correlator, dim=0)))

        return exact_correlator - dd_correlator
        
    

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
    #Output: Domain Decompostion based factorized propogator matrix between subdomains using BICGStab rather than 
    #the exact inverse
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
        p[:,:,xcut_1:xcut_1+bw -1, :] = 0.0
        p[:,1,xcut_1+bw-1,:] = 0.0
        p[:,:,xcut_2:xcut_2+bw-1, :] = 0.0
        p[:,1,xcut_2+bw-1,:] = 0.0
        return p
    

    def localized_Fermion_Action(self, d, f):
        d = d.to_dense()
        return tr.einsum('bx,bx->b', f.conj(), f) - tr.einsum('bi, bij, bkj, bk->b', f.conj(), d, d.conj(), f) 

    #Input: lattice configuration, frozen timeslices and boundary widths
    #Output: Approximated action of the configuration based on a tridiagonalized 
    #Schur complement
    def dd_Action(self, q, xcut_1, xcut_2, bw):
        #Check if fermions are part of theory
        if len(q) == 1:
            return self.gaugeAction(q[0]).type(self.dtype)
        else:
            #Absolute value for the slightly complex fermion action?
            bb_d = self.bb_DiracOperator(q, xcut_1, xcut_2, bw)

            #For using a factorized approximate action- in development
            #return self.gaugeAction(q[0]).type(self.dtype) + \
            #tr.abs(self.blockdiagonal_Fermion_Action(bb_d, q[1], xcut_1, xcut_2, bw).type(self.dtype))

            #Using the exact action for testing
            return self.gaugeAction(q[0]).type(self.dtype) + tr.abs(self.fermionAction(q[2], q[1])).type(self.dtype)
            

    #Input:field configuration, frozen timeslice indices, boundary width, placeholder for nonlocal approx
    #Output:Tensor of force for gauge field HMC
    #nlf:Non-local force- may be used for cross-subdomain approximation in a parallelization scheme
    # TODO:Finish development and write description of finalized function    
    def dd_Force(self, q, xcut_1, xcut_2, bw, nlf=None):

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
        # fg[:,:,xcut_1:xcut_1+bw, :] = 0.0
        # fg[:,:,xcut_2:xcut_2+bw, :] = 0.0
        # fg[:,0,xcut_1-1, :] = 0.0
        # fg[:,0,xcut_2-1,:] = 0.0
        fg[:,:,xcut_1:xcut_1+bw -1, :] = 0.0
        fg[:,1,xcut_1+bw-1,:] = 0.0
        fg[:,:,xcut_2:xcut_2+bw-1, :] = 0.0
        fg[:,1,xcut_2+bw-1,:] = 0.0

        #If fermions aren't present, simply return force of gauge field
        if len(q) == 1:
            #Force is already real, force cast for downstream errors
            #Additional negative sign added from Hamilton's eqs.
            return (-1.0)*tr.real(fg).type(self.dtype)
    
        #TODO: dynamical fermion implementation- in progress

        #Identify reordering of index points
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


        #Construct Dirac operator derivative as normal:
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


        #reorder the lattice indices of the derivative tensor
        bb_dd = tr.zeros_like(dd, dtype=tr.complex64)
        i = 0
        for n in ri:  
            # #Naive O(n^2) algorithm
            j=0
            for m in ri:
                #Take 2x2 matrix of Dirac space for each index
                bb_dd[:, :, :, 2*i:2*i+2, 2*j:2*j+2] = dd[:,:,:, 2*n:2*n+2, 2*m:2*m+2]
                j+=1
            i+=1       

        #Now we want to peel off the pieces we are interested in and construct the  
        #derivative of the Schur complement
        

        bb_d = self.bb_DiracOperator(q, xcut_1, xcut_2,bw)

        bb_d = bb_d.to_dense()
    
        #Isolate sub matrices
        #Assumes 2 timeslice boundaries of width bw
        d00 = bb_d[:,0:4*bw*self.V[1], 0:4*bw*self.V[1]]
        d01 = bb_d[:, 0:4*bw*self.V[1], 4*bw*self.V[1]:]
        d10 = bb_d[:, 4*bw*self.V[1]:, 0:4*bw*self.V[1]]
        d11 = bb_d[:, 4*bw*self.V[1]:, 4*bw*self.V[1]:]


        #dd00 = bb_dd[:,:,:,0:4*bw*self.V[1], 0:4*bw*self.V[1]]
        dd01 = bb_dd[:,:,:, 0:4*bw*self.V[1], 4*bw*self.V[1]:]
        dd10 = bb_dd[:,:,:,4*bw*self.V[1]:, 0:4*bw*self.V[1]]
        dd11 = bb_dd[:,:,:, 4*bw*self.V[1]:, 4*bw*self.V[1]:]

        #If non-local force elements are given, use them, if not generate them
        #For inserting an approximation for parallelization in later development
        if nlf == None:
            sd11 = dd11 - (tr.einsum('bimxy, byz, bza->bimxa', dd10, tr.inverse(d00), d01) +
                        tr.einsum('bxy, byz, bimza->bimxa', d10, tr.inverse(d00), dd01))
        else:
            sd11 = dd11 - nlf


        #Schur complement
        s11 = d11 - tr.einsum('bij, bjk, bkm->bim', d10, tr.inverse(d00), d01)

        
        #tri_s11 = self.schur_Tridiagonal(s11).to_dense()
        tri_s11=s11
        #tri_s11 = self.schur_Block_Diagonal(s11).to_dense()

        bb_f = self.bb_FermionVector(q[1], xcut_1, xcut_2, bw)

        #fermions for points inside subdomains
        sd_f = bb_f[:, 4*bw*self.V[1]:]


        #For now as progress, we just attempt to compute force on the whole lattice
        #Will do separation for parallelization later

        #SS^dagger inverse
        ssi = tr.inverse(tr.einsum('bxy, bzy-> bxz', tri_s11, tri_s11.conj()))
        v = tr.einsum('bxy, by-> bx', ssi, sd_f)

        a = tr.einsum('bx, bimxy, bzy, bz-> bim', v.conj(), sd11, tri_s11.conj(), v)

        ff = -1.0*(a + a.conj())


        ff = tr.reshape(ff, [self.Bs, 2, self.V[0], self.V[1]]) 

        #Zero out the force on the frozen links
        ff[:,:,xcut_1:xcut_1+bw -1, :] = 0.0
        ff[:,1,xcut_1+bw-1,:] = 0.0
        ff[:,:,xcut_2:xcut_2+bw-1, :] = 0.0
        ff[:,1,xcut_2+bw-1,:] = 0.0

        #TODO: Check on the complex components of ff
        return ((-1.0)*tr.real(fg+ ff).type(self.dtype))
        

    #input: Dirac operator, psuedofermion vector, rank of inverse approximation
    #output:Approximated fermionic contribution to action
    #approximation based on a block Schur complement
    def blockdiagonal_Fermion_Action(self, bd, f, xcut_1, xcut_2, bw, d00_inv=tr.zeros(2)):
        bb_d = bd.to_dense()
        
        #Isolate sub matrices
        #Assumes 2 timeslice boundaries of width bw
        d00 = bb_d[:,0:4*bw*self.V[1], 0:4*bw*self.V[1]]
        d01 = bb_d[:, 0:4*bw*self.V[1], 4*bw*self.V[1]:]
        d10 = bb_d[:, 4*bw*self.V[1]:, 0:4*bw*self.V[1]]
        d11 = bb_d[:, 4*bw*self.V[1]:, 4*bw*self.V[1]:]

        #reorder psuedofermion vector
        bb_f = self.bb_FermionVector(f, xcut_1, xcut_2, bw)

        #If function is not given a matrix for the inverse of the boundary region,
        #compute it exactly
        if d00_inv.count_nonzero()== 0:
            d00_inv = tr.inverse(d00)

        #Schur complement
        s11 = d11 - tr.einsum('bij, bjk, bkm->bim', d10, d00_inv, d01)

        #tri_s11  = self.schur_Tridiagonal(s11)
        #tri_s11 = s11
        tri_s11 = self.schur_Block_Diagonal(s11)

        tri_s11_inv = tr.inverse(tri_s11.to_dense())

        #fermion vector in the subdomains
        sd_f = bb_f[:, 4*bw*self.V[1]:]

        return tr.einsum('bx, bxy, byz, bz->b', tr.conj(sd_f), tr.conj(tr.transpose(tri_s11_inv, 1,2))
                         , tri_s11_inv, sd_f)
    

    #input: Block diagonal Dirac operator, psuedofermion vector, boundary
    # timesslices and boundary width. Inverse of boundary dirac operator if already computed
    #Output: Fermion action based on the 
    def schur_Fermion_Action(self, bd, f, xcut_1, xcut_2, bw, d00_inv=tr.zeros(2)):
        bb_d = bd.to_dense()
        
        #Isolate sub matrices
        #Assumes 2 timeslice boundaries of width bw
        d00 = bb_d[:,0:4*bw*self.V[1], 0:4*bw*self.V[1]]
        d01 = bb_d[:, 0:4*bw*self.V[1], 4*bw*self.V[1]:]
        d10 = bb_d[:, 4*bw*self.V[1]:, 0:4*bw*self.V[1]]
        d11 = bb_d[:, 4*bw*self.V[1]:, 4*bw*self.V[1]:]

        #reorder psuedofermion vector
        bb_f = self.bb_FermionVector(f, xcut_1, xcut_2, bw) 

        #If function is not given a matrix for the inverse of the boundary region,
        #compute it exactly
        if tr.numel(d00_inv)== 0:
            d00_inv = tr.inverse(d00)

        #Schur complement
        s11 = d11 - tr.einsum('bij, bjk, bkm->bim', d10, d00_inv, d01)

        s11_inv = tr.inverse(s11)

        #fermion vector in the subdomains
        sd_f = bb_f[:, 4*bw.self.V[1]]

        #TODO: Conjugate?
        return tr.einsum('bx, bxy, byz, bz->b', tr.conj(sd_f), tr.transpose(s11_inv, 1,2)
                         , s11_inv, sd_f)
    
    #Takes the Schur complement and returns the sparse tri-diagonal approximation of it
    def schur_Tridiagonal(self, s):
        mask = tr.zeros([s.size(dim=1), s.size(dim=1)], dtype=tr.bool)

        #Construct a boolean block tridiagonal mask
        for x in np.arange(0, s.size(dim=1), 2):
            if x == 0:
                mask[x:x+2, x:x+4] = True
            elif x == s.size(dim=1) -2 :
                mask[x:x+2, x-2:x+2] = True
            else:
                mask[x:x+2, x-2:x+4] = True
            
        mask = mask.unsqueeze(0)

        batch_mask = mask.repeat(self.Bs, 1,1)

        s_d = s.to_dense()

        return tr.where(batch_mask, s_d, tr.zeros_like(s_d)).to_sparse()
    
    #Takes the Schur complement and discards elements non on the block diagonal
    #corresponding to point pairs within subdomains
    def schur_Block_Diagonal(self,s):
        
        m_int = tr.zeros([s.size(dim=1), s.size(dim=2)])

        half = int(s.size(dim=1)/2)
        #TODO: +1 on indexing below or not?
        m_int[0:half, 0:half] = tr.ones_like(m_int[0:half, 0:half])
        m_int[half:, half:] = tr.ones_like(m_int[half:, half:])

        #mask = tr.tensor(m_int, dtype=tr.bool)
        mask =m_int.to(dtype=tr.bool)

        mask = mask.unsqueeze(0)

        batch_mask = mask.repeat(self.Bs, 1,1)

        s_d = s.to_dense()

        return tr.where(batch_mask, s_d, tr.zeros_like(s_d)).to_sparse()

            