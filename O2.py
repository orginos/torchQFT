#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue March 24 9:38:24 2023

@author: Kostas Orginos
"""

import numpy as np
import torch as tr

class O2():
    def action(self,phi):
        A = self.Nd*self.Vol*tr.ones(phi.shape[0],device=self.device)
        for mu in range(1,self.Nd+1):
            A = A - tr.sum(tr.cos(phi-tr.roll(phi,shifts=-1,dims=mu)),dim=(1,2))
        return self.beta*A
    
    def force(self,phi):
        F = tr.zeros_like(phi)
        for mu in range(1,self.Nd+1):
            F +=  -tr.sin(phi - tr.roll(phi,shifts= 1,dims=mu)) -  tr.sin(phi - tr.roll(phi,shifts=-1,dims=mu))
        return self.beta*F
    

    def refreshP(self):
        P = tr.normal(0.0,1.0,[self.Bs,self.V[0],self.V[1]],dtype=self.dtype,device=self.device)
        return P

    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    
    def kinetic(self,P):
        return tr.sum(P*P,dim=(1,2))/2.0 

    def hotStart(self):
        sigma=tr.normal(0.0,2.0*np.pi,[self.Bs,self.V[0],self.V[1]],
                        dtype=self.dtype,device=self.device)
        return sigma

    #following equations 4,5,6 of https://arxiv.org/pdf/1210.6116.pdf
    def helicity_modulus(self,phi):
        mu=1
        c = tr.sum(tr.cos(phi-tr.roll(phi,shifts=-1,dims=mu)),dim=(1,2))
        s = tr.sum(tr.sin(phi-tr.roll(phi,shifts=-1,dims=mu)),dim=(1,2))
        c *= 1.0/self.Vol
        s *= 1.0/self.Vol
        Y = c - self.beta*self.Vol *s*s
        return Y
        
    def __init__(self,V,beta,batch_size=1,device="cpu",dtype=tr.float32): 
            self.V = tuple(V) # lattice size
            self.Vol = np.prod(V)
            self.Nd = len(V)
            self.beta = beta # the coupling
            self.Bs=batch_size
            self.device=device
            self.dtype=dtype
            self.N = 2 # does O2 only


def main():
    import time
    import matplotlib.pyplot as plt
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    L=128
    batch_size=1
    beta = 0.5
    o = O2([L,L],beta,batch_size=batch_size)

    phi=o.hotStart()
    plt.imshow(phi[0,:,:], cmap='hot', interpolation='nearest')
    plt.show()

    tic=time.perf_counter()
    Niter=10000
    for k in range(Niter):
        o.action(phi)
    toc=time.perf_counter()
    print(f"action time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")

    tic=time.perf_counter()
    for k in range(Niter):
        o.force(phi)
    toc=time.perf_counter()
    print(f"force time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")    

    P = o.refreshP()

    tic=time.perf_counter()
    for k in range(Niter):
        o.kinetic(phi)
    toc=time.perf_counter()
    print(f"kinetic time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")    

if __name__ == "__main__":
   main()
    

            



