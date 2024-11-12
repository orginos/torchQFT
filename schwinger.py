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
        
        #Otherwise, compute force of the fermion fields
        d_dense = q[2].to_dense()
        f = q[1]

        #Dirac Op. derivative- very similar to dirac operator itself without mass term:
        #Empty bx2x(V)x(Vx2)x(Vx2)
        dd = tr.zeros([self.Bs, self.Nd, self.V[0]*self.V[1], self.V[0]*self.V[1]*self.Nd, self.V[0]*self.V[1]*self.Nd], dtype=tr.complex64)
        dd= dd.to_sparse()
       
        #enumeration of lattice sites
        p_f = tr.tensor(np.arange(self.V[0]*self.V[1]))
        #Reshape to match lattice geometry
        p =tr.reshape(p_f, (self.V[0], self.V[1]))

        #Apply antiperiodic boundary condition in time to the gauge fields:
        u_bc = u.clone().detach()
        u_bc[:, 0, self.V[0]-1, :] = -1.0*u[:, 0, self.V[0]-1, :]

        for mu in [0, 1]:
            #Forward shifted indices
            p_s =  tr.roll(p, shifts = -1, dims=mu)
            #Flatten the 2D reps of lattice/field to one dimension
            p_sf = tr.reshape(p_s, (-1,))
            u_f = tr.reshape(u_bc[:, mu, :, :], (self.Bs, self.V[0]*self.V[1]))
            d_dir = tr.zeros([self.Bs, self.Nd, self.V[0]*self.V[1], self.V[0]*self.V[1], self.V[0]*self.V[1]], dtype=tr.complex64)
            d_dir[:, mu, p_f, p_f, p_sf] = u_f
            #Note included imaginary number below due to derivative
            dd = dd  + tr.kron(d_dir, -0.5j*(tr.eye(2) - gamma[mu])).to_sparse()
            #dd = tr.add(tr.kron(d_dir, -0.5j*(tr.eye(2) - gamma[mu])), dd)

            #Backwards shifted indices
            p_s =  tr.roll(p, shifts = +1, dims=mu)
            #Flatten the 2D reps of lattice/field to one dimension
            p_sf = tr.reshape(p_s, (-1,))
            u_f = tr.reshape(tr.conj(tr.roll(u_bc[:, mu, :, :], shifts=1, dims=mu+1)), (self.Bs, self.V[0]*self.V[1]))
            d_dir = tr.zeros([self.Bs,self.Nd, self.V[0]*self.V[1], self.V[0]*self.V[1], self.V[0]*self.V[1]], dtype=tr.complex64)
            d_dir[:, mu, p_sf, p_f, p_sf] = u_f
            #Note included imaginary number below due to derivative
            dd = dd + tr.kron(d_dir, 0.5j*(tr.eye(2) + gamma[mu])).to_sparse()
            #dd = tr.add(tr.kron(d_dir, 0.5j*(tr.eye(2) + gamma[mu])), dd)

        dd_dense= dd.to_dense()

        #With Dirac operator gauge link derivative constructed, compute the full fermion force
        m = tr.einsum('bimxy, byz-> bimxz', dd_dense, d_dense.conj().transpose(1,2)) + \
              tr.einsum('bxy, bimyz->bimxz', d_dense, dd_dense.conj().transpose(3,4))
        
        #D D^dagger inverse - maybe can store this earlier on to reuse?
        ddi =tr.inverse(tr.einsum('bxy, byz-> bxz', d_dense, d_dense.conj().transpose(1,2)))
        v = tr.einsum('bxy, by ->bx', ddi, f)
        
        ff = -1.0*tr.einsum('bx, bimxy, by-> bim', v.conj(), m, v)

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
    


                
        



    