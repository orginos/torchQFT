#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 2022

@author: Ben Slimmer
"""

import numpy as np;
import torch as tr;
import update as hmc;
import integrators as i

gamma = tr.tensor([[[0.0, -1.0j], [-1.0j, 0.0]], [[0, -1], [1,0]], [[1,0], [0,1]]])

class schwinger():
    #Both action functions are UNTESTED... work in progress  
    
    #Inputs: self, gauge field u 
    #Outputs: Pure gauge action
    def gaugeAction(self, u):
        #plaquette matrix
        pl = u[:,:,:,0]*tr.roll(u[:,:,:,1], shifts=1, dims=1) \
            * tr.conj(tr.roll(u[:,:,:,0], shifts=1, dims=2))*tr.conj(u[:,:,:,1]) 
        S = (2/self.lam**2) *(self.V[0]*self.V[1] + np.real(tr.sum(pl,dim=(1,2))))
        return S

    #Inputs: self, gauge field u, spinor lattice psi
    def fermionAction(self, u, psi):
        sub =tr.zeros([self.Bs, self.V[0], self.V[1], 2])
        #Fermion action over each dimension summed, including multiplicaton with gamma matrix
        for mu in range(0, self.Nd):
            #below expands a unitary dimension 4th dimension to carry out element wise multiplication of U on the spinor
            u_mult = tr.unsqueeze(u[:,:,:,mu], 3)
            pre = tr.einsum('ij, bxyj->bxyi',gamma[mu], (1.0/2)*(u_mult*tr.roll(psi, shifts=1, dims=(mu+1)) \
                                                           -tr.conj(tr.roll(u_mult, shifts=-1, dims=(mu+1)))*tr.roll(psi, shifts=-1, dims=(mu+1))))
            sub = sub + tr.einsum('ij, bxyj->bxyi',gamma[mu], (1.0/2)*(u_mult*tr.roll(psi, shifts=1, dims=(mu+1)) \
                                                           -tr.conj(tr.roll(u_mult, shifts=-1, dims=(mu+1)))*tr.roll(psi, shifts=-1, dims=(mu+1))))
        return 1.0j *tr.einsum('bxyz,bxyz -> b', tr.conj(psi), sub)


    #TODO: Write an expression for the gauge + fermion action
    #Inputs: self, the fermion field configuration q
    #Outputs: Action of the given configuration
    def action(self, q):
        return self.gaugeAction() + self.fermionAction()


    def kinetic(self,P):
        return tr.einsum('bxy,bxy->b',P,P)/2.0 
    
    #HMC refresh of momentum 
    def refreshP(self):
        P = tr.normal(0.0,1.0,[self.Bs,self.V[0],self.V[1]],dtype=self.dtype,device=self.device)
        return P
    
    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    
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
    def generateSpinors(self):
        p = tr.normal(0.0,2*np.pi, [self.Bs, self.V[0], self.V[1], 2],
                        dtype=self.dtype,device=self.device)
        psi = (1.0/np.sqrt(2.0)) * tr.exp(1.0j*p)
        return psi
    
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

    psi = sch.generateSpinors()



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

    #Verify action functions run without error
    #Runs, but values are unverified
    if(True):
        gS = sch.gaugeAction(u)
        print(gS)

        #This returns a mostly imaginary action.. needs review
        fS = sch.fermionAction(u, psi)
        print(fS)

    # mn2 = i.minnorm2(sch.force,sch.evolveQ,7,1.0)



if __name__ == "__main__":
   main()