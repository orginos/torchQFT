# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:04:12 CET 2025

@author: Kostas Orginos
"""
# Collection of flows that I use frequently
# so that I know what I am doing and not have to read the docs
#  of other implementations of the same ideas
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm 

import time
print("Torch Version: ",tr.__version__)
print("Numpy Version: ",np.__version__)

if tr.backends.mps.is_available():
    device = tr.device("mps")
else:
    print ("MPS device not found.")
    device = "cpu"
    
print("Accelerator divice: ",device)


class Shift2D(nn.Module):
    def __init__(self, shift: tuple):
        """
        A layer that shifts a 2D input tensor along height and width.
        
        Args:
        - shift (tuple): (shift_down, shift_right), positive values shift right/down.
        """
        super().__init__()
        self.shift = shift
        self.ishift = (-shift[0],-shift[1]) # inverse shift
        

    def forward(self, x):
        return tr.roll(x, shifts=self.shift, dims=(-2, -1))
        
    #inverse returns the jacobian as well
    #assume batch dimension being the 0-th 
    #last 2 dimensions are the spatial 2D 
    def inverse(self, x):
        return tr.roll(x, shifts=self.ishift, dims=(-2, -1)),tr.zeros(x.shape[0],device=x.device)

# one thing at a time
class ConvRealNVP(nn.Module):
    def __init__(self, prior,shape,conv_layers,mask,activation=nn.ReLU(),device="cpu"):
        super(ConvRealNVP, self).__init__()
        self.device=device
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        #self.s = nn.Sequential()
        #self.t = nn.Sequential()
        self.s = nn.ModuleList()
        self.t = nn.ModuleList()
        self.shape = shape
        
        # set up the t and s networks
        for i in range(len(mask)):
            s = nn.Sequential()
            t = nn.Sequential()
            in_dim =1
            k=0  
            out_dim = 4*in_dim
            layer = nn.Conv2d(in_dim,out_dim, kernel_size=2, stride=2) 
            s.add_module('s-lin'+str(k),layer)
            s.add_module('s-act'+str(k),activation)
            layer = nn.Conv2d(in_dim,out_dim, kernel_size=2, stride=2) 
            t.add_module('t-lin'+str(k),layer)
            t.add_module('t-act'+str(k),activation)
            in_dim=out_dim
            k+=1
            for l in conv_layers:
                layer = nn.Conv2d(in_dim,l, kernel_size=1, stride=1)
                s.add_module('s-lin'+str(k),layer)
                s.add_module('s-act'+str(k),activation)
                layer = nn.Conv2d(in_dim,l, kernel_size=1, stride=1)
                t.add_module('t-lin'+str(k),layer)
                t.add_module('t-act'+str(k),activation)
                in_dim=l
                k+=1
            layer = nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=1)   
            s.add_module('s-lin'+str(k),layer)
            s.add_module('s-act'+str(k),activation)
            layer = nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=1) 
            t.add_module('t-lin'+str(k),layer)
            t.add_module('t-act'+str(k),activation)
            k+=1
            # final step make a lattice of the original shape
            layer = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=2, stride=2)
            s.add_module('s-linT'+str(k),layer)
            s.add_module('s-tanh'+str(k),nn.Tanh())
            layer = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=2, stride=2)
            t.add_module('t-linT'+str(k),layer)
            t.add_module('t-act'+str(k),activation)
            self.s.append(s)
            self.t.append(t)

        
    def forward(self,x):
        x=x.unsqueeze(1)
        z = x
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * tr.exp(s) + t)
        
        return x.squeeze()

    def inverse(self,x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        z = z.unsqueeze(1)
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])    
            z = (1 - self.mask[i]) * (z - t) * tr.exp(-s) + z_
            log_det_J -= s.sum(dim=(2,3)).squeeze()
            
        return z.squeeze(), log_det_J
        
    def log_prob(self,x):
        z, logp = self.backward(x)
        return self.prior.log_prob(z.flatten(start_dim=1)) + logp 
        #return self.prior.log_prob(z.view(x.shape[0],np.prod(x.shape[1:3]))) + logp 
    
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1)).view([batchSize]+self.shape).to(self.device)
        #logp = self.prior.log_prob(z)
        x = self.forward(z)
        return x

    def prior_sample(self,batchSize):
        return self.prior.sample((batchSize, 1)).view([batchSize]+self.shape).to(self.device)
        
#use what I know it works
class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = tr.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = tr.nn.ModuleList([nets() for _ in range(len(mask))])
        #z = self.prior_sample(1)
        self.shape = (mask.shape[1],) # expects 1d data
        
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

    def forward(self,z):
        return self.g(z)
    def inverse(self,x):
        return self.f(x)
        
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp #+ self.C
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        #logp = self.prior.log_prob(z)
        x = self.g(z)
        return x

    def prior_sample(self,batchSize):
        return self.prior.sample((batchSize, 1)).squeeze()

class shapedRealNVP(RealNVP):
    def __init__(self, shape,nets, nett, mask, prior):
        super(shapedRealNVP, self).__init__(nets, nett, mask, prior)
        self.shape = shape
        self.V = np.prod(shape)
        
    def forward(self,z):
        return super().forward(z.view(z.shape[0],self.V)).view([z.shape[0]] + self.shape)
        
    def inverse(self,x):
        z,J = super().inverse(x.view(x.shape[0],self.V))
        return z.view([x.shape[0]] + self.shape),J
        
    def log_prob(self,x):
        return super().log_prob(x.view(x.shape[0],self.V))
        
    def sample(self,Nb):
        return super().sample(Nb).view([Nb] + self.shape)
        
    def prior_sample(self,Nb):
        return super().prior_sample(Nb).view([Nb] + self.shape)
    

        
# the inverse returns the jacobian
class FlowModel(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)  # List of flows
        self.shape = self.flows[0].shape
        
    def forward(self, x):
        for flow in self.flows:
            x = flow(x)
        return x

    def inverse(self, x):
        log_det_jac = tr.zeros(x.shape[0],device=x.device)
        for flow in reversed(self.flows):
            x,j = flow.inverse(x)
            log_det_jac+=j
      
        return x,log_det_jac
        
    def log_prob(self,x):
        z, logp = self.inverse(x)
        return self.prior.log_prob(z) + logp #+ self.C

    def sample(self, batchSize): 
        z = self.flows[0].prior_sample(batchSize)
        #logp = self.prior.log_prob(z)
        x = self.forward(z)
        return x

    def prior_sample(self,batchSize):
        return self.flows[0].prior_sample(batchSize)


class ParityNet(nn.Module):
    def __init__(self, net,Parity=+1):
        super(ParityNet,self).__init__()
        self.Parity = Parity
        self.net = net
    def forward(self,x):
        return 0.5*(self.net(x) + self.Parity*self.net(-x))




def test_jacobian(model):
    import time
    import matplotlib.pyplot as plt
    
    print("The model has data shape: ",model.shape)
    V = np.prod(model.shape)
    
    
    batch_size=4
    z = model.prior_sample(batch_size)
    x = model(z)
    #print(x)
    zz,J = model.inverse(x)
    print("Reversibility Diff: ",(zz-z).norm()/z.norm())
    print("log(Jacobian)     : ",J.to("cpu").detach().numpy())
    def jwrap(x):
        z,j=model.inverse(x)
        return z
   
    print("Shape of x: ",x.shape)
    torchJacM = tr.autograd.functional.jacobian(jwrap,x)
    print("Autograd jacobian matrix shape:",torchJacM.shape)
    log_dets = []
    diffs = []
    for k in range(batch_size):
        if(len(model.shape)==1):
            foo = torchJacM[k,:,k,:].squeeze()
        elif  (len(model.shape)==2):
            foo = torchJacM[k,:,:,k,:,:].squeeze().view(V,V)
        else:
            foo = tr.eye(V).unsqueeze(0).expand(batch_size, -1, -1)
        ldet  = np.log(foo.det().numpy())
        log_dets.append(ldet)
        diffs.append(np.abs(ldet - J[k].detach().numpy())/V)

    print("log(Jacobian) : ",tr.tensor(log_dets).detach().numpy())
    print("Differences   : ",tr.tensor(diffs).detach().numpy())

def test_jacobians():
    L=8
    V=L*L
    batch_size=4
    X = np.array(np.arange(L))[:,np.newaxis]
    Y = np.array(np.arange(L))[np.newaxis,:]
    X = np.repeat(X,L,axis=1)
    Y = np.repeat(Y,L,axis=0)
    mm = (X+Y)%2 # even odd mask
    lm = mm.reshape(V)
    Nlayers = 3
    masks_LxL = tr.from_numpy(np.array([mm, 1-mm] * Nlayers).astype(np.float32))
    masks_V   = tr.from_numpy(np.array([lm, 1-lm] * Nlayers).astype(np.float32))

    
    nets = lambda: ParityNet(nn.Sequential(nn.Linear(V, 2*V), nn.LeakyReLU(), nn.Linear(2*V, 2*V), nn.LeakyReLU(), nn.Linear(2*V, V), nn.Tanh()), Parity=+1)
    nett = lambda: ParityNet(nn.Sequential(nn.Linear(V, 2*V), nn.LeakyReLU(), nn.Linear(2*V, 2*V), nn.LeakyReLU(), nn.Linear(2*V, V)), Parity = -1)

    normal = tr.distributions.Normal(tr.zeros(V),tr.ones(V))
    prior= tr.distributions.Independent(normal, 1)
    
    flow_model = FlowModel([RealNVP(nets, nett, masks_V, prior),RealNVP(nets, nett, masks_V, prior)])

    print("\n* Testing  FlowModel made by RealNVPs")
    test_jacobian(flow_model)
    print("-------------------")
    
    cf = lambda : ConvRealNVP(prior,shape=[L,L],conv_layers=[8,8],mask=tr.tensor(mm),device="cpu",activation=nn.GELU())

    conv_flow_model = FlowModel([cf(),Shift2D((1,1)),cf(),Shift2D((-1,-1))])

    print("\n* Testing  FlowModel made by ConvRealNVPs")
    test_jacobian(conv_flow_model)
    print("-------------------")
    
    shaped_flow_model = FlowModel([shapedRealNVP([L,L],nets, nett, masks_V, prior),shapedRealNVP([L,L],nets, nett, masks_V, prior)])
    print("\n* Testing  FlowModel made by shapedRealNVPs")
    test_jacobian(shaped_flow_model)
    print("-------------------")

def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='jac')
    test_jacobians()

if __name__ == "__main__":
    main()
    
