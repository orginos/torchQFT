#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  10 9:38:24 2019

@author: Kostas Orginos
"""

import numpy as np
import torch as tr
import phi4  as p

class phi4_rg(p.phi4):
    def action(self,phi):
        dd = phi - self.cF 
        return super().action(phi) + 0.5*self.kappa*tr.einsum('bxy,bxy->b',dd,dd)
    
    def force(self,phi):
        return super().force(phi)-self.kappa*(phi-self.cF)

    #averages the field inside a 2x2 block
    def block(self,phi):
        return tr.nn.AvgPool2d(2, stride=2)(phi)

    #picks the even points only
    def coarsen(self,phi):
        return tr.gather(tr.gather(phi,1,self.indX),2,self.indY)

    #fills a fine lattice from a coarse with constant values inside a block
    def refine(self,phi2):
        syF = tr.zeros(self.Bs,self.V2[0],self.V[1],dtype=self.dtype).scatter_(2,self.indY,phi2).scatter_(2,self.oindY,phi2)
        sF = tr.zeros(self.Bs,self.V[0],self.V[1],dtype=self.dtype).scatter_(1,self.indX,syF).scatter_(1,self.oindX,syF)
        return  sF
    
    
    def __init__(self,V,l,m,k,cF,batch_size=1,device="cpu",dtype=tr.float32):
            super().__init__(V,l,m,batch_size,device,dtype)

            #project the local coarse field to the fine lattice
            self.V2 =  (int(V[0]/2), int(V[1]/2))
            V2 = self.V2

            X=np.arange(0,V[0],2)
            Y=np.arange(0,V[1],2)
            fooX = np.repeat(X[:,np.newaxis],V[1],axis=1)
            fooX = np.repeat(fooX[np.newaxis,:,:],self.Bs,axis=0)
            self.indX = tr.tensor(fooX)
            fooY = np.repeat(Y[np.newaxis,:],V2[0],axis=0)
            fooY = np.repeat(fooY[np.newaxis,:,:],self.Bs,axis=0)
            self.indY = tr.tensor(fooY)

            oX = X + 1
            oY = Y + 1
            fooX = np.repeat(oX[:,np.newaxis],V[1],axis=1)
            fooX = np.repeat(fooX[np.newaxis,:,:],self.Bs,axis=0)
            self.oindX = tr.tensor(fooX)
            fooY = np.repeat(oY[np.newaxis,:],V2[0],axis=0)
            fooY = np.repeat(fooY[np.newaxis,:,:],self.Bs,axis=0)
            self.oindY = tr.tensor(fooY)

            self.cF = self.refine(cF) 
            if(cF.shape[0] != self.Bs) or (cF.shape[1] != V2[0]) or (cF.shape[2] != V2[1]):
                print("The shape of the coarse field does not match")
                print("Expected ", self.Bs, self.V2)
                print("Got      ", cF.shape)
            self.kappa = k


def main():
    import time
    import matplotlib.pyplot as plt
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    L=128
    batch_size=1
    lam =0.1
    mass= 0.1
    kappa=2
    o2 = p.phi4([int(L/2),int(L/2)],lam,mass,batch_size=batch_size)
    cF = o2.hotStart()
    
    o = phi4_rg([L,L],lam,mass,kappa,cF,batch_size=batch_size)

    phi=o.hotStart()
    plt.imshow(phi[0,:,:], cmap='hot', interpolation='nearest')
    plt.show()

    phi2 = o.block(phi)
    plt.imshow(phi2[0,:,:], cmap='hot', interpolation='nearest')
    plt.show()

    rphi = o.refine(phi2)
    plt.imshow(rphi[0,:,:], cmap='hot', interpolation='nearest')
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
    

            



