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

    def forward(self,z):
        return self.g(z)
    
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        x = self.g(z)
        return x

    
class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior,data_dims=(1)):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = tr.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = tr.nn.ModuleList([nets() for _ in range(len(mask))])
        self.data_dims=data_dims
        
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
            log_det_J -= s.sum(dim=self.data_dims)
        return z, log_det_J

    def forward(self,z):
        return self.g(z)
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp #+ self.C
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        #logp = self.prior.log_prob(z)
        x = self.g(z)
        return x

# This is a convolutional Flow Layer which takes in a bijector
# it applies it to 2x2 block shifts the field and
# applies the second instance of the bijector to the shifted field
# then it shifts back (which may not be needed but do it for sanity)
# it allows to repeat the process arbitrary number of steps
# the default is 1
class ConvFlowLayer(nn.Module):
    def __init__(self,size,bijector,Nsteps=1):
        super(ConvFlowLayer, self).__init__()
        self.Nsteps=Nsteps
        self.bj = tr.nn.ModuleList([bijector() for _ in range(2*Nsteps)])
        # for now the kernel is kept 2x2 and stride is 2 so it only works on lattices with
        # a power of 2 dimensions
        fold_params = dict(kernel_size=(2,2), dilation=1, padding=0, stride=(2,2))
        self.unfold = nn.Unfold(**fold_params)
        self.fold = nn.Fold(size,**fold_params)
        # this could be an arbitrary class with a sample method
        # self.prior=prior
        # I see no need for sampling from a prior for this layer
        
    # noise to fields
    def forward(self,z):
        uf = self.unfold(z.view(z.shape[0],1,z.shape[1],z.shape[2])).transpose(2,1)
        for k in range(self.Nsteps):
            sf = self.unfold(tr.roll(self.fold(self.bj[2*k  ].g(uf).transpose(2,1)),dims=(2,3),shifts=(-1,-1))).transpose(2,1)
            uf = self.unfold(tr.roll(self.fold(self.bj[2*k+1].g(sf).transpose(2,1)),dims=(2,3),shifts=( 1, 1))).transpose(2,1)
        x = self.fold(uf.transpose(2,1)).squeeze()
        return x

    # fields to noise
    def backward(self,x):
        log_det_J=x.new_zeros(x.shape[0])
        #HERE IS WHERE WE HAVE FUN!
        # add the extra dimension for unfolding
        z = x.view(x.shape[0],1,x.shape[1],x.shape[2])
        for k in reversed(range(self.Nsteps)):
            #shift  and unfold
            sz = self.unfold(tr.roll(z,dims=(2,3),shifts=(-1,-1))).transpose(2,1)
            #unfold and flow 
            ff,J = self.bj[2*k+1].f(sz)
            log_det_J += J
            #fold shift unfold
            sz = self.unfold(tr.roll(self.fold(ff.transpose(2,1)),dims=(2,3),shifts=(1,1))).transpose(2,1)
            # flow backwards
            ff,J = self.bj[2*k].f(sz)
            log_det_J += J 
            #fold
            z = self.fold(ff.transpose(2,1))
            
        z = z.squeeze()
        return z,log_det_J

    def log_prob(self,x):
        z, logp = self.backward(x)
        #return self.prior.log_prob(z) + logp
        return logp # we do not have a prior distribution for this layer 

# no need for sampling for this layer
#    def sample(self, batchSize): 
#        z = self.prior.sample((batchSize, 1))
#        x = self.forward(z)
#        return x

#prepares RealNVP for the Convolutional Flow Layer
def FlowBijector(Nlayers=3):
    mm = np.array([1,0,0,1])
    tV = mm.size
    nets = lambda: nn.Sequential(nn.Linear(tV, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, tV), nn.Tanh())
    nett = lambda: nn.Sequential(nn.Linear(tV, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, tV))

    # the number of masks determines layers
    #Nlayers = 3
    masks = tr.from_numpy(np.array([mm, 1-mm] * Nlayers).astype(np.float32))
    normal = distributions.Normal(tr.zeros(tV),tr.ones(tV))
    prior= distributions.Independent(normal, 1)
    return  RealNVP(nets, nett, masks, prior, data_dims=(1,2))


# this is an invertible RG transformation
# it preseves the residual fine degrees of freedom
class RGlayer(nn.Module):
    def __init__(self,transformation_type="select"):
        super(RGlayer, self).__init__()
        if(transformation_type=="select"):
            mask_c = [[1.0,0.0],[0.0,0.0]]
            mask_r = [[1.0,0.0],[0.0,0.0]]
        elif(transformation_type=="average"):
            mask_c = [[0.25,0.25],[0.25,0.25]]
            mask_r = [[1.00,1.00],[1.00,1.00]]
        else:
            print("Uknown RG blocking transformation. Using default.")
            mask_c = [[1.0,0.0],[0.0,0.0]]
            mask_r = [[1.0,0.0],[0.0,0.0]]
                  
        
        self.restrict = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,2),stride=2,bias=False)
        self.restrict.weight = tr.nn.Parameter(tr.tensor([[mask_c]]),requires_grad=False)

        self.prolong = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=(2,2),stride=2,bias=False)
        self.prolong.weight = tr.nn.Parameter(tr.tensor([[mask_r]]),requires_grad=False)

        
    def coarsen(self,f):
        ff = f.view(f.shape[0],1,f.shape[1],f.shape[2])
        c = self.restrict(ff)
        r = ff-self.prolong(c)
        return c.squeeze(),r.squeeze()
    
    def refine(self,c,r):
        cc = c.view(c.shape[0],1,c.shape[1],c.shape[2])
        rr = r.view(c.shape[0],1,r.shape[1],r.shape[2])
        return (self.prolong(cc)+rr).squeeze()


class MGflow(nn.Module):
     def __init__(self,flow,rg,):
         super(MGflow, self).__init__()
         self.flow = flow

def test_realNVPjacobian():
    import time
    import matplotlib.pyplot as plt
    

    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    L=8
    V=L*L
    batch_size=4
    lam =0.5
    mass= -0.2
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

    z = prior.sample((batch_size, 1)).squeeze()
    x = flow.g(z)
    #print(x)
    zz,J = flow.f(x)
    print("Diff: ",(zz-z).abs().mean())
    print("RealNVP jacobian: ",J.detach().numpy())
    def jwrap(x):
        z,j=flow.f(x)
        return z
    
    print("Shape of x: ",x.shape)
    torchJacM = tr.autograd.functional.jacobian(jwrap,x)
    print("Autograd jacobian matrix shape:",torchJacM.shape)
    log_dets = []
    diffs = []
    for k in range(batch_size):
        foo = torchJacM[k,:,k,:].squeeze()
        ldet  = np.log(foo.det().numpy())
        log_dets.append(ldet)
        diffs.append(np.abs(ldet - J[k].detach().numpy())/V)

    print("log(Jacobians): ",log_dets)
    print("Differences   : ",diffs)
                     
                     

    
    #print("log(Jacobian) :", )
    #foo = torchJacM[1,:,1,:].squeeze()
    #print("Jacobian matrix: ", foo)
    #print("log(Jacobian) :", np.log(foo.det().numpy()))
    
    
    
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

    phi = o.hotStart()

    print("Testing the select rule")
    rg = RGlayer()
    phi2,ff = rg.coarsen(phi)
    rphi = rg.refine(phi2,ff)
    
    plt.pcolormesh(phi[0,:,:],cmap='hot')
    plt.show()
    plt.pcolormesh(phi2[0,:,:],cmap='hot')
    plt.show()
    plt.pcolormesh(rphi[0,:,:],cmap='hot')
    plt.show()
    print(phi[0,:,:])
    print(phi2[0,:,:])
    print(rphi[0,:,:])
    rev_check = (tr.abs(rphi-phi)/V).mean()
    print("Should be zero if reversible: ",rev_check.detach().numpy())

    print("\n*****\nTesting the average rule")
    rg = RGlayer("average")
    phi2,ff = rg.coarsen(phi)
    rphi = rg.refine(phi2,ff)
    
    plt.pcolormesh(phi[0,:,:],cmap='hot')
    plt.show()
    plt.pcolormesh(phi2[0,:,:],cmap='hot')
    plt.show()
    plt.pcolormesh(rphi[0,:,:],cmap='hot')
    plt.show()
    print(phi[0,:,:])
    print(phi2[0,:,:])
    print(rphi[0,:,:])
    rev_check = (tr.abs(rphi-phi)).mean()
    print("Should be zero if reversible: ",rev_check.detach().numpy())

def test_ConvFlowLayerJacobian():
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

    phi = o.hotStart()

    #print(FlowBijector())
    #print("OK")
    cf = ConvFlowLayer([L,L],FlowBijector)
    fphi = cf(phi)
    rphi,J = cf.backward(fphi)

    print("The Jacobian is: ", J.detach().numpy())
    def jwrap(x):
        z,_=cf.backward(x)
        return z

    x=fphi
    print("Shape of x: ",x.shape)
    torchJacM = tr.autograd.functional.jacobian(jwrap,x)
    print("Autograd jacobian matrix shape:",torchJacM.shape)
    torchJacM = torchJacM.reshape(batch_size,V,batch_size,V)
    print("Autograd jacobian matrix reshaped:",torchJacM.shape)
    log_dets = []
    diffs = []
    for k in range(batch_size):
        foo = torchJacM[k,:,k,:].squeeze()
        ldet  = np.log(foo.det().numpy())
        log_dets.append(ldet)
        diffs.append(np.abs(ldet - J[k].detach().numpy())/V)

    print("log(Jacobians): ",log_dets)
    print("Differences   : ",diffs)
    
def test_ConvFlowLayer():
    import time
    import matplotlib.pyplot as plt
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    L=32
    V=L*L
    batch_size=4
    lam =0.5
    mass= -0.2
    o  = p.phi4([L,L],lam,mass,batch_size=batch_size)

    phi = o.hotStart()

    #print(FlowBijector())
    #print("OK")
    cf = ConvFlowLayer([L,L],FlowBijector)
    #print("OK2")
    
    fphi = cf(phi)
    rphi,J = cf.backward(fphi)

    print("The Jacobian is: ", J.detach().numpy())
    ttJ = cf.log_prob(fphi)
    print("The Jacobian is: ", ttJ.detach().numpy())
    
    rev_check = (tr.abs(rphi-phi)).mean()
    print("Should be zero if reversible: ",rev_check.detach().numpy())
    
    #print(fphi.shape)

    plt.pcolormesh(phi[0,:,:],cmap='hot')
    plt.show()
    
    plt.pcolormesh(fphi.detach()[0,:,:],cmap='hot')
    plt.show()


    plt.pcolormesh(rphi.detach()[0,:,:],cmap='hot')
    plt.show()
    
def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='realNVP')
    
    args = parser.parse_args()
    if(args.t=='realNVP'):
        print("Testing realNVP")
        test_realNVP()
    elif(args.t=='id'):
        print("Testing Identity")
        test_Identity()
    elif(args.t=="rg"):
        print("Testing RG Layer")
        test_RGlayer()
    elif(args.t=="cflow"):
        print("Testing Convolutional Flow Layer")
        test_ConvFlowLayer()
    elif(args.t=="realNVPjac"):
        print("Testing RealNVP Jacobian")
        test_realNVPjacobian()
    elif(args.t=="cflowjac"):
        print("Testing Convolutional Flow Jacobian")
        test_ConvFlowLayerJacobian()
        
    else:
        print("Nothing to test")


  
    

if __name__ == "__main__":
    main()
    

            



