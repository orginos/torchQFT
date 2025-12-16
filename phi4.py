#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  10 9:38:24 2019

@author: Kostas Orginos
"""

import numpy as np
import torch as tr

class phi4():
    def action(self,phi):
        #A = 0.5*self.mtil*tr.einsum('bxy,bxy->b',phi,phi) + (self.lam/24.0)*tr.einsum('bxy,bxy->b',phi**2,phi**2)
        #for mu in range(1,self.Nd+1):
        #    A = A - tr.einsum('bxy,bxy->b',phi,tr.roll(phi,shifts=-1,dims=mu))

        phi2 = phi*phi
        A = tr.sum((0.5*self.mtil + (self.lam/24.0)*phi2)*phi2,dim=(1,2))
        for mu in range(1,self.Nd+1):
            A = A - tr.sum(phi*tr.roll(phi,shifts=-1,dims=mu),dim=(1,2))
        return A
    
    def force(self,phi):
        F = -self.mtil*phi - self.lam*phi**3/6.0
        for mu in range(1,self.Nd+1):
            F +=  tr.roll(phi,shifts= 1,dims=mu)+tr.roll(phi,shifts=-1,dims=mu)
        return F
    

    def refreshP(self):
        P = tr.normal(0.0,1.0,[self.Bs,self.V[0],self.V[1]],dtype=self.dtype,device=self.device)
        return P

    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    
    def kinetic(self,P):
        return tr.einsum('bxy,bxy->b',P,P)/2.0

    def coldStart(self):
        sigma=tr.ones([self.Bs,self.V[0],self.V[1]], dtype=self.dtype,device=self.device)
        return sigma

    def hotStart(self):
        sigma=tr.normal(0.0,1.0,[self.Bs,self.V[0],self.V[1]],dtype=self.dtype,device=self.device)
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


class phi4_c1:
    def action(self,phi_c):
        rphis=[]
        rphis.append(phi_c)
        iii=0
        for pi in reversed(self.pics):
            #print(pi.shape)
            rphi = self.rg.refine(rphis[iii],pi)
            rphis.append(rphi)
            iii+=1
        phi_f = rphis[-1]
        #evaluate coarse field in action of rg
        #print(phi_f.shape,"shape of fine field")
        return self.sg.action(phi_f)
        #if I dont add the .sum() I got a grad for the batch system, it seems to me that we include that in the force property the batch is summed?

    def force(self,phi_c):
        #phi_c.requires_grad_(True)
        x_tensor = phi_c.clone()
        x_tensor.requires_grad_()
        SS = self.action(x_tensor)

        SS.sum().backward()

        if tr.isnan(SS).any():#torch derivative of the action
            print("nan locations:",tr.isnan(SS).nonzero())
            self.phi_fail=phi_c
        return -x_tensor.grad
    
    def refreshP(self):
        P = tr.normal(0.0,1.0,self.phis[-1].shape)#only difference with fine level
        return P

    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    
    def kinetic(self,P):
        return tr.einsum('bxy,bxy->b',P,P)/2.0

    def generate_cfg_levels(self,phi11,level):#run every time we need to contruct deeper or superficial levels
        #run a configuration
        self.level=level
        phis=[]
        pis=[]
        phicopy=phi11.clone()
        print("shape of the original field",phicopy.shape)
        phis.append(phicopy)
        for _ in range(level):
            print("coarsening level ",_," field shape ",phicopy.shape)
            phic,pic = self.rg.coarsen(phicopy)
            phis.append(phic)
            pis.append(pic)
            phicopy=phic
        self.phis=phis
        self.pics=pis

        #reversed
        rphis=[]
        rphis.append(phis[-1])
        for phics,pis in zip(reversed(phis),reversed(pis)):
            rphi = self.rg.refine(phics,pis)
            rphis.append(rphi)
        self.rphis=rphis

    def __init__(self,sgg,rgg):
        self.sg = sgg#theory? in the finest level
        self.rg = rgg#projector to coarse level



def main():
    import time
    import matplotlib.pyplot as plt
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    L=128
    batch_size=1
    lam =0.1
    mass= 0.1
    o = phi4([L,L],lam,mass,batch_size=batch_size)

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
    

            



