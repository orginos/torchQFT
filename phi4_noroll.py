#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  10 9:38:24 2019

@author: Kostas Orginos
"""

import numpy as np
import torch as tr

class phi4():
    def old_action(self,phi):
        A = 0.5*self.mtil*tr.einsum('bxy,bxy->b',phi,phi) + (self.lam/24.0)*tr.einsum('bxy,bxy->b',phi*phi,phi*phi)
        for mu in range(1,self.Nd+1):
            A = A - tr.einsum('bxy,bxy->b',phi,tr.roll(phi,shifts=-1,dims=mu))
        return A

    def roll_action(self,phi):
        phi2 = phi*phi
        A = tr.sum((0.5*self.mtil + (self.lam/24.0)*phi2)*phi2,dim=(1,2))
        for mu in range(1,self.Nd+1):
            A = A - tr.sum(phi*tr.roll(phi,shifts=-1,dims=mu),dim=(1,2))
        return A 
               
    def action(self,phi):
        phi2 = phi*phi
        A = tr.sum((0.5*self.mtil + (self.lam/24.0)*phi2)*phi2,dim=(1,2))
        phi_c = phi.view(phi.shape[0],1,phi.shape[1],phi.shape[2])
        A = A - 0.5*tr.sum(phi_c*self.nn(phi_c),dim=(1,2,3))
        return A
    
    def old_force(self,phi):
        F = -self.mtil*phi - self.lam*phi**3/6.0
        for mu in range(1,self.Nd+1):
            F +=  tr.roll(phi,shifts= 1,dims=mu)+tr.roll(phi,shifts=-1,dims=mu)
        return F

    def roll_force(self,phi):
        F = (-self.mtil- (self.lam/6.0)*phi*phi)*phi
        for mu in range(1,self.Nd+1):
            F +=  tr.roll(phi,shifts= 1,dims=mu)+tr.roll(phi,shifts=-1,dims=mu)
        return F

    def force(self,phi):
        F = (-self.mtil- (self.lam/6.0)*phi*phi)*phi
        phi_c = phi.view(phi.shape[0],1,phi.shape[1],phi.shape[2])
        return F+self.nn(phi_c).squeeze()
    

    def refreshP(self):
        P = tr.normal(0.0,1.0,[self.Bs,self.V[0],self.V[1]],dtype=self.dtype,device=self.device)
        return P

    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    
    def kinetic(self,P):
        return tr.einsum('bxy,bxy->b',P,P)/2.0 ;

    def hotStart(self):
        sigma=tr.normal(0.0,1.0,
                        [self.Bs,self.V[0],self.V[1]],
                        dtype=self.dtype,device=self.device)
        return sigma
    
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
            # the  nearest neighbors
            self.nn = tr.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),stride=1,
                                   padding=1,padding_mode='circular',bias=False)
            mask = [[0.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,0.0]]
            self.nn.weight = tr.nn.Parameter(tr.tensor([[mask]],dtype=self.dtype,device=self.device),requires_grad=False)




def main():
    import time
    import matplotlib.pyplot as plt
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    L=128
    batch_size=4
    lam =0.1
    mass= 0.1
    o = phi4([L,L],lam,mass,batch_size=batch_size)

    phi=o.hotStart()
    #plt.imshow(phi[0,:,:], cmap='hot', interpolation='nearest')
    #plt.show()

    tic=time.perf_counter()
    Niter=10000
    for k in range(Niter):
        oldA=o.old_action(phi)
    toc=time.perf_counter()
    print(f"old_action time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")

    tic=time.perf_counter()
    Niter=10000
    for k in range(Niter):
        rollA=o.roll_action(phi)
    toc=time.perf_counter()
    print(f"roll_action time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")

    tic=time.perf_counter()
    Niter=10000
    for k in range(Niter):
        A=o.action(phi)
    toc=time.perf_counter()
    print(f"action time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")

    print("convAction diff: ",((    A-oldA)/o.Vol).numpy())
    print("rollAction diff: ",((rollA-oldA)/o.Vol).numpy())
    
    tic=time.perf_counter()
    for k in range(Niter):
        oldF=o.old_force(phi)
    toc=time.perf_counter()
    print(f"old_force time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")

    tic=time.perf_counter()
    for k in range(Niter):
        rollF=o.roll_force(phi)
    toc=time.perf_counter()
    print(f"roll_force time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")

    tic=time.perf_counter()
    for k in range(Niter):
        convF=o.force(phi)
    toc=time.perf_counter()
    print(f"force time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")
    
    convDiff = (convF - oldF).abs().mean(dim=(1,2))
    rollDiff = (rollF - oldF).abs().mean(dim=(1,2))
    print("convForce diff: ",convDiff.numpy())
    print("rollForce diff: ",rollDiff.numpy())
    
    P = o.refreshP()

    tic=time.perf_counter()
    for k in range(Niter):
        o.kinetic(phi)
    toc=time.perf_counter()
    print(f"kinetic time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")    

if __name__ == "__main__":
   main()
    

            



