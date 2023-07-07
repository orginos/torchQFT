#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 2022

@author: Ben Slimmer
"""

import numpy as np;
import torch as tr;
import update as h;
import integrators as i

gamma = tr.tensor([[[0.0, -1.0j], [-1.0j, 0.0]], [[0, -1], [1,0]], [[1,0], [0,-1]]])

class schwinger():


    def __init__(self,V,l,m,batch_size=1,device="cpu",dtype=tr.float32): 
        self.V = tuple(V) # lattice size
        self.Vol = np.prod(V)
        self.Nd = len(V)
        self.lam = l # the coupling
        self.mass  = m
        self.mtil = m + 2*self.Nd
        self.Bs=batch_size
        self.device=device
        self.dtype=dtype

    
    #Inputs: self, gauge field u 
    #Outputs: Pure gauge action
    def gaugeAction(self, u):
        #plaquette matrix
        pl = u[:,:,:,0]*tr.roll(u[:,:,:,1], shifts=1, dims=1) \
            * tr.conj(tr.roll(u[:,:,:,0], shifts=1, dims=2))*tr.conj(u[:,:,:,1]) 
        S = (2/self.lam**2) *(self.V[0]*self.V[1] + np.real(tr.sum(pl,dim=(1,2))))
        return S

    #Inputs: self, gauge field u
    #Outputs: Dirac operator
    def diracOperator(self, u):
        #Sub-sum for adding terms of different directions
        sub =tr.zeros([self.Bs, self.V[0], self.V[1], 2, 2], dtype=tr.complex64)
        #Identity matrix unsqueezed for batch size
        b_eye = tr.unsqueeze(tr.eye(n=self.V[0], m=self.V[1]), 0)
        #TODO: Check on below dimensions for the pytorch roll function:
        #roll_dim = [2, 1]

        for mu in range(0, self.Nd):
            u_dir = u[:,:,:,mu]

            naive = (1.0/2)*(tr.roll(u_dir, shifts=1, dims=mu+1) \
            - tr.conj(u_dir))
            wilson = (-1.0/2)*(tr.roll(u_dir, shifts=1, dims=mu+1) \
            + tr.conj(u_dir))
            sub += tr.einsum('ij,bxy->bxyij', (gamma[mu]), naive) + tr.einsum('ij,bxy->bxyij', (tr.eye(2)), wilson)
        return sub + tr.einsum('ij,bxy->bxyij', (self.mtil*tr.eye(2)), b_eye)


    #Inputs: self, gauge field u, dirac operator d, spinor lattice psi
    def fermionAction(self, u, d, psi):

        #Updated version with seperate dirac operator function...
        first = tr.einsum('bxyij, bxyj->bxyi', d, psi)
        return 1.0j*tr.einsum('bxyi,bxyi->b',tr.conj(psi), first)


    #TODO: Fix fermion action to be able to include it in total action
    #Inputs: self, the gauge field u
    #Outputs: Action of the given configuration
    def action(self, u):
        return self.gaugeAction(u).type(self.dtype)
    
    

    #TODO: Write fermion force contribution
    #Inputs: self, gauge field u
    #Output: force tensor for updating momentum tensor in HMC
    def force(self, u):
        #'staple' A matrix
        a = tr.zeros_like(u)

        a[:,:,:,0]  = tr.roll(u[:,:,:,1], shifts = 1, dims=(1))*tr.conj(tr.roll(u[:,:,:,0], shifts= 1, dims=(2)))*tr.conj(u[:,:,:,1]) \
                        + tr.conj(tr.roll(u[:,:,:,1], shifts=(1, -1), dims=(1,2)))*tr.conj(tr.roll(u[:,:,:,0], shifts=-1, dims=2))*tr.roll(u[:,:,:,1], shifts = -1, dims=2)
        a[:,:,:,1] = tr.roll(u[:,:,:,0], shifts=1, dims=(2)) *tr.conj(tr.roll(u[:,:,:,1], shifts=1, dims=(1)))*tr.conj(u[:,:,:,0]) \
                        + tr.conj(tr.roll(u[:,:,:,0], shifts=(-1,1), dims=(1,2)))*tr.conj(tr.roll(u[:,:,:,1], shifts=-1,dims=1))*tr.roll(u[:,:,:,0],shifts=-1,dims=1)
        #gauge action contribution
        fg = (-1.0j* (2/self.lam**2)/12.0)* (u*a - tr.conj(a)*tr.conj(u))

        #TODO: fermion action contribution
        #Free theory case for now
        ff = tr.zeros_like(u)

        #force should already be real... force the cast for downstream errors
        return (tr.real(fg+ ff)).type(self.dtype)


    def kinetic(self,P):
        return tr.einsum('bxyz,bxyz->b',P,P)/2.0 
    
    #HMC refresh of momentum
    def refreshP(self):
        P = tr.normal(0.0,1.0,[self.Bs,self.V[0],self.V[1], 2],dtype=self.dtype,device=self.device)
        return P
    
    #HMC position evolve 
    def evolveQ(self,dt,P,Q):
        return Q*tr.exp(1.0j*dt*P)
    
    #Input: self
    #Output: Hot started U[1] gauge field of batch*lattice size.
        #the final dimension of 2 is for link variable in t, then x direction
    def hotStart(self):
        alpha = tr.normal(0.0, 2*np.pi, [self.Bs, self.V[0], self.V[1], 2],
                      dtype=self.dtype,device=self.device)
        u = tr.exp(1.0j*alpha)
        return u
    
    #Input: self
    #Output: normalized Dirac spinor lattice
    def spinorLattice(self):
        p = tr.exp(1.0j*tr.normal(0.0,2*np.pi, [self.Bs, self.V[0], self.V[1], 2],
                        dtype=self.dtype,device=self.device))
        return p
        
    

def main():
    import time
    import matplotlib.pyplot as plt
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    L=128
    batch_size=1
    lam =0.1
    mass= 0.1
    sch = schwinger([L,L],lam,mass,batch_size=batch_size)

    u = sch.hotStart()
    d=sch.diracOperator(u)

    psi = sch.spinorLattice()



    #Test plot of randomized link values in x direction
    if(False):
        #arccos of real part of the link -> randomized parameter
        plt.imshow(tr.arccos(tr.real(u[0,:,:,0])), cmap='hot', interpolation='nearest')
        plt.show()

    #Test spinor plot
    if(False):
        #arccos of real part of the link -> randomized parameter
        plt.imshow(tr.arccos(tr.real(u[0,:,:,0])), cmap='hot', interpolation='nearest')
        plt.show()


    
    #test dirac operator
    if(False):
        d = sch.diracOperator(u)
        print(d.size())
        print(tr.det(d).size())
        print(tr.det(d)[0,:5,:5])
        d_dag =tr.conj(tr.transpose(d, 3, 4))
        tr.inverse(tr.einsum('bxyij, bxyjk->bxyik', d, d_dag))



    #Verify action functions run without error
    #Runs, but values are unverified
    if(True):
        gS = sch.gaugeAction(u)
        print("Gauge action: ", gS)

        #This returns a mostly imaginary action.. needs review
        fS = sch.fermionAction(u,d,psi)
        print("Fermion action: ", fS)

    #test force function
    if(False):
        print(sch.force(u))


    mn2 = i.minnorm2(sch.force,sch.evolveQ,7,1.0)

    sim = h.hmc(sch, mn2)

    #Test sim
    if (True):
        # plt.imshow(tr.arccos(tr.real(u[0,:,:,0])), cmap='hot', interpolation='nearest')
        # plt.show()
        u_upd = sim.evolve(u, 100)
        plt.imshow(tr.arccos(tr.real(u_upd[0,:,:,0])), cmap='hot', interpolation='nearest')
        plt.show()
    



if __name__ == "__main__":
   main()