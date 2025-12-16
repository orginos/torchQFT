#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  10 9:38:24 2019

@author: Kostas Orginos
I have just modified it to work on arbitrary dimensions and do checks

"""

import numpy as np
import torch as tr

class phi4():
    def action(self,phi):
        #A = 0.5*self.mtil*tr.einsum('bxy,bxy->b',phi,phi) + (self.lam/24.0)*tr.einsum('bxy,bxy->b',phi**2,phi**2)
        #for mu in range(1,self.Nd+1):
        #    A = A - tr.einsum('bxy,bxy->b',phi,tr.roll(phi,shifts=-1,dims=mu))

        phi2 = phi*phi
        #A = tr.sum((0.5*self.mtil + (self.lam/24.0)*phi2)*phi2,dim=(1,2))
        #for mu in range(1,self.Nd+1):
        #    A = A - tr.sum(phi*tr.roll(phi,shifts=-1,dims=mu),dim=(1,2))

            
        A = tr.sum((0.5*self.mtil + (self.lam/24.0)*phi2)*phi2, dim=tuple(range(1, self.Nd+1)))
        for mu in range(1, self.Nd+1):
            A = A - tr.sum(phi * tr.roll(phi, shifts=-1, dims=mu), dim=tuple(range(1, self.Nd+1)))
        return A
    
    def force(self,phi):
        F = -self.mtil*phi - self.lam*phi**3/6.0
        for mu in range(1,self.Nd+1):
            F +=  tr.roll(phi,shifts= 1,dims=mu)+tr.roll(phi,shifts=-1,dims=mu)
        return F


    def refreshP(self):
        shape = [self.Bs] + list(self.V)
        P = tr.normal(0.0,1.0,shape,dtype=self.dtype,device=self.device)
        return P

    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    
    def kinetic(self,P):
        spatial_dims = ''.join(chr(ord('a') + i) for i in range(self.Nd))#abcd for d=4
        einsum_str = f"z{spatial_dims},z{spatial_dims}->z"
        return tr.einsum(einsum_str,P,P)/2.0

    def hotStart(self):
        shape = [self.Bs] + list(self.V)
        sigma = tr.normal(0.0, 1.0, shape, dtype=self.dtype, device=self.device)
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


class NonLinearRGlayer(nn.Module):
    def __init__(self, channels=1, hidden_channels=8, batch_size=1):
        super(NonLinearRGlayer, self).__init__()
        self.batch_size = batch_size
        
        # Restrictor: one small conv + downsampling
        self.restrict_net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=2)
        )

        # Prolongator: upsampling + conv
        self.prolong_net = nn.Sequential(
            nn.ConvTranspose2d(channels, hidden_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1)
        )

    def coarsen(self, f):
        ff = f.view(f.shape[0], 1, f.shape[1], f.shape[2])  # B x 1 x H x W
        c = self.restrict_net(ff)
        r = ff - self.prolong_net(c)
        if self.batch_size == 1:
            return c.squeeze(1), r.squeeze(1)
        else:
            return c.squeeze(), r.squeeze()

    def refine(self, c, r):
        cc = c.view(c.shape[0], 1, c.shape[1], c.shape[2])
        rr = r.view(r.shape[0], 1, r.shape[1], r.shape[2])
        f_rec = self.prolong_net(cc) + rr
        if self.batch_size == 1:
            return f_rec.squeeze(1)
        else:
            return f_rec.squeeze()
    


def main():
    import time
    import matplotlib.pyplot as plt
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    L=32
    batch_size=1
    lam =0.1
    mass= 0.1
    o = phi4([L],lam,mass,batch_size=batch_size, device=device, dtype=tr.float32)

    phi=o.hotStart()
    #plt.imshow(phi[0,:,:,1,2].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    #plt.show()
    print(f"phi shape {phi.shape} dtype {phi.dtype} device {phi.device}")
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