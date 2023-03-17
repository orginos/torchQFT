#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu March  16 20:54:24 2023

@author: Kostas Orginos
"""

import numpy as np
import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter

import phi4  as p

# In place of realNVP just for testing
class Identity(nn.Module):
    def __init__(self, prior):
        super(Identity, self).__init__()
        self.prior = prior
    def g(self, z):
        x=z
        return x
    def f(self,x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp 
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        x = self.g(z)
        return x

    
class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = tr.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = tr.nn.ModuleList([nets() for _ in range(len(mask))])
    
    # this is the forward start from noise target
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * tr.exp(s) + t)
        return x
    
    # this is backward from target to noise
    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * tr.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp #+ self.C
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        #logp = self.prior.log_prob(z)
        x = self.g(z)
        return x

class RGlayer(nn.Module):
    def __init__(self):
        super(RGlayer, self).__init__()
        self.restrict = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,2),stride=2,bias=False)
        self.restrict.weight = tr.nn.Parameter(tr.tensor([[[[1.0,0.0],[0.0,0.0]]]]),requires_grad=False)

        self.prolong = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=(2,2),stride=2,bias=False)
        self.prolong.weight = tr.nn.Parameter(tr.tensor([[[[1.0,0.0],[0.0,0.0]]]]),requires_grad=False)

        
    def coarsen(self,f):
        ff = f.view(f.shape[0],1,f.shape[1],f.shape[2])
        return self.restrict(ff).squeeze()
    
    def refine(self,c):
        cc = c.view(c.shape[0],1,c.shape[1],c.shape[2])
        return self.prolong(cc).squeeze()
    
def test_realNVP():
    import time
    import matplotlib.pyplot as plt
    

    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    L=4
    V=L*L
    batch_size=1000
    lam =0.5
    mass= -0.2
    o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
    X = np.array(np.arange(L))[:,np.newaxis]
    Y = np.array(np.arange(L))[np.newaxis,:]
    X = np.repeat(X,L,axis=1)
    Y = np.repeat(Y,L,axis=0)
    mm = (X+Y)%2 # even odd mask
    mm = mm.reshape(V)
    nets = lambda: nn.Sequential(nn.Linear(V, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, V), nn.Tanh())
    nett = lambda: nn.Sequential(nn.Linear(V, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, V))
    # the number of masks determines layers
    Nlayers = 3
    masks = tr.from_numpy(np.array([mm, 1-mm] * Nlayers).astype(np.float32))
    normal = distributions.Normal(tr.zeros(V),tr.ones(V))
    prior= distributions.Independent(normal, 1)
    flow = RealNVP(nets, nett, masks, prior)

    optimizer = tr.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=1e-4)
    for t in range(5001):   
        #with torch.no_grad():
        z = prior.sample((batch_size, 1)).squeeze()
        x = flow.g(z) # generate a sample
        loss = (flow.log_prob(x)+o.action(x.view(batch_size,L,L))).mean() # KL divergence (or not?)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step() 
        if t % 500 == 0:
            #print(z.shape)
            print('iter %s:' % t, 'loss = %.3f' % loss)
    print("Testing Reversibility")
    z = prior.sample((1000, 1)).squeeze()
    x = flow.g(z)
    zz,j=flow.f(x)

    rev_check = (tr.abs(zz-z)/V).mean()
    print("Should be zero if reversible: ",rev_check.detach().numpy())

    diff = o.action(x.view(x.shape[0],L,L))+flow.log_prob(x)
    print("max  action diff: ", tr.max(diff.abs()).detach().numpy())
    print("mean action diff: ", diff.mean().detach().numpy())
    print("std  action diff: ", diff.std().detach().numpy())
    
def test_Identity():
    import time
    import matplotlib.pyplot as plt
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    L=4
    V=L*L
    batch_size=1000
    lam =0.5
    mass= -0.2
    o  = p.phi4([L,L],lam,mass,batch_size=batch_size)

    normal = distributions.Normal(tr.zeros(V),tr.ones(V))
    prior= distributions.Independent(normal, 1)
    
    flow = Identity(prior)

    
    print("Testing Reversibility")
    z = prior.sample((1000, 1)).squeeze()
    x = flow.g(z)
    zz,j=flow.f(x)

    rev_check = (tr.abs(zz-z)/V).mean()
    print("Should be zero if reversible: ",rev_check.detach().numpy())

    diff = o.action(x.view(x.shape[0],L,L))+flow.log_prob(x)
    print("max  action diff: ", tr.max(diff.abs()).detach().numpy())
    print("mean action diff: ", diff.mean().detach().numpy())
    print("std  action diff: ", diff.std().detach().numpy())

def test_RGlayer():
    import time
    import matplotlib.pyplot as plt
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    L=8
    V=L*L
    batch_size=4
    lam =0.5
    mass= -0.2
    o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
    rg = RGlayer()

    phi = o.hotStart()
    phi2 = rg.coarsen(phi)
    rphi = rg.refine(phi2)
    
    plt.pcolormesh(phi[0,:,:],cmap='hot')
    plt.show()
    plt.pcolormesh(phi2[0,:,:],cmap='hot')
    plt.show()
    plt.pcolormesh(rphi[0,:,:],cmap='hot')
    plt.show()
    print(phi[0,:,:])
    print(phi2[0,:,:])
    print(rphi[0,:,:])
    
def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='realNVP')
    
    args = parser.parse_args()
    if(args.t=='realNVP'):
        print("Testing realNVP")
        test_realNVP()
    if(args.t=='id'):
        print("Testing Identity")
        test_Identity()
    if(args.t=="rg"):
        print("Testing RG Layer")
        test_RGlayer()
        
    else:
        print("Nothing to test")


  
    

if __name__ == "__main__":
    main()
    

            



