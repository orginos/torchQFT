#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  10 9:38:24 2019

@author: Kostas Orginos
"""

import numpy as np
import torch as tr
import phi4  as p

#implements Renormalization Group transformations
class phi4_rg_trans():

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

    #fills a fine lattice from a coarse with gaussian noise inside a block
    def noisy_refine(self,phi2):
        gnoise=tr.randn_like(phi2)
        syF = tr.zeros(self.Bs,self.V2[0],self.V[1],dtype=self.dtype).scatter_(2,self.indY,phi2).scatter_(2,self.oindY,gnoise)
        # I need new  noise  shape to be used in the next stage
        # yes I am replacing gausian random numbers with new ones!
        # noise is just noise...
        gnoise  = tr.randn_like(syF)      
        sF = tr.zeros(self.Bs,self.V[0],self.V[1],dtype=self.dtype).scatter_(1,self.indX,syF).scatter_(1,self.oindX,gnoise)
        return  sF
    
    #blocking factor is always 2
    def __init__(self,V,batch_size=1,device="cpu",dtype=tr.float32):
        
            self.Bs = batch_size
            self.device = device
            self.dtype = dtype
            
            #project the local coarse field to the fine lattice
            self.V2 =  (int(V[0]/2), int(V[1]/2))
            self.V = V
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


#implements RG transformations using masks
class phi4_masked_rg():
    #blocking factor is always 2
    def __init__(self,V,batch_size=1,device="cpu",dtype=tr.float32):
        
            self.Bs = batch_size
            self.device = device
            self.dtype = dtype
            self.V=V
            #project the local coarse field to the fine lattice
            self.V2 =  (int(V[0]/2), int(V[1]/2))
            self.V = V
            V2 = self.V2
            
            X=np.arange(0,V[0],2)
            Y=np.arange(0,V[1],2)
            tt = tr.zeros(self.Bs,V[0],V[1],dtype=tr.bool)
            ttX = tt.clone()
            ttY = tt.clone()
            ttY[:,:,Y] = True
            ttX[:,X,:] = True
            self.cmask = tr.logical_and(ttX,ttY)
            self.fmask = tr.logical_not(self.cmask)
            self.fine_shape = (self.Bs,V[0],V[1])
            self.coarse_shape = (self.Bs,V2[0],V2[1])
            self.coarse_size  = np.prod(self.coarse_shape)
            self.fine_size    = np.prod(self.fine_shape)

    #picks the even points only
    def coarsen(self,phi):
        return tr.reshape(tr.masked_select(phi,self.cmask),self.coarse_shape)

    def fine_zeros(self,phi2):
        rphi = tr.zeros(self.fine_shape)
        rphi[self.cmask]=phi2.view(self.coarse_size)
        return rphi

    #replaces the coarse points in a fine field phi by the coarse field phi2
    def fine_replace(self,phi2,phi):
        iphi = phi.clone()
        iphi[self.cmask]=phi2.view(self.coarse_size)
        return iphi
    


            
def main():
    import time
    import matplotlib.pyplot as plt
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    L=256
    batch_size=1
    lam =0.1
    mass= 0.1
    o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
    o2 = p.phi4([int(L/2),int(L/2)],lam,mass,batch_size=batch_size)
    cF = o2.hotStart()
    
    rg = phi4_rg_trans([L,L],batch_size=batch_size)

    phi=o.hotStart()
    plt.imshow(phi[0,:,:], cmap='hot', interpolation='nearest')
    plt.show()

    phi2 = rg.block(phi)
    plt.imshow(phi2[0,:,:], cmap='hot', interpolation='nearest')
    plt.show()

    rphi = rg.refine(phi2)
    plt.imshow(rphi[0,:,:], cmap='hot', interpolation='nearest')
    plt.show()

    rphi = rg.noisy_refine(phi2)
    plt.imshow(rphi[0,:,:], cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == "__main__":
   main()
    

            



