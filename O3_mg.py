#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  27 11:00:00 2024

The model is O(3) in 2D.
RG transformations 
sigma[batch,spin,x,y] is the structure of the tensor containing the field

@author: Kostas Orginos
"""

import torch as tr
import numpy as np
from torch import nn

class RGlayer(nn.Module):
    def __init__(self,transformation_type="select",N=3):
        super(RGlayer, self).__init__()
        if(transformation_type=="select"):
            mask_c = [[1.0,0.0],[0.0,0.0]]
            mask_r = [[1.0,1.0],[1.0,1.0]]
        elif(transformation_type=="average"):
            mask_c = [[0.25,0.25],[0.25,0.25]]
            mask_r = [[1.00,1.00],[1.00,1.00]]
        else:
            print("Uknown RG blocking transformation. Using default.")
            mask_c = [[1.0,0.0],[0.0,0.0]]
            mask_r = [[1.0,0.0],[0.0,0.0]]
                  
        # We need this for debuging
        self.type = transformation_type
        
        self.restrict = nn.Conv2d(groups=N,in_channels=N, out_channels=N, kernel_size=(2,2),stride=2,bias=False)
        self.restrict.weight = tr.nn.Parameter(tr.tensor([[mask_c]]).repeat(N,1,1,1),requires_grad=False)
        self.prolong = nn.ConvTranspose2d(groups=N,in_channels=N,out_channels=N,kernel_size=(2,2),stride=2,bias=False)
        self.prolong.weight = tr.nn.Parameter(tr.tensor([[mask_r]]).repeat(N,1,1,1),requires_grad=False)

    def coarsen(self,f):
        c = self.restrict(f)
        c = c/tr.norm(c,dim=1).view(c.shape[0],1,c.shape[2],c.shape[3])
        fc = self.prolong(c)
        i_one_p_dot = 1.0/(1.0+tr.einsum('bsxy,bsxy->bxy',fc,f))
        A = tr.einsum('bsxy,brxy->bsrxy',fc,f)
        A = A - A.transpose(1,2)
        A2 = tr.einsum('bxy,bskxy,bkrxy->bsrxy',i_one_p_dot,A,A)
        r = tr.eye(3,3).view(1,3,3,1,1) + A + A2
        return c,r
    
    def refine(self,c,r):
        # rotate with the transpose
        return tr.einsum('brsxy,brxy->bsxy',r,self.prolong(c))


def test_RGlayer(file):
    import time
    import matplotlib.pyplot as plt
    import O3 as s
    import update as u
    import integrators as i
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    Nwarm =10
    L=16
    lat=[L,L]
    V=L*L
    batch_size=4
    beta = 1.263
    o  = s.O3(lat,beta,batch_size=batch_size)
    
    sigma = o.hotStart()
    if(file != 'no-load'):
        print("Loading fields from: ",file)
        sigma=tr.load(file)
        
    mn2 = i.minnorm2(o.force,o.evolveQ,7,1.0)
    hmc = u.hmc(T=o,I=mn2)
    sigma = hmc.evolve(sigma,Nwarm)
    
    sig2img = (sigma[0].permute((1,2,0))+1.0)/2.0
    print(sigma[0].shape,sig2img.shape)
    plt.imshow(sig2img, interpolation='nearest')
    plt.show()

    RG = RGlayer(transformation_type="average")
    c,r=RG.coarsen(sigma)
    ff=RG.refine(c,r)
    print("Reversibility check: ",tr.norm(sigma-ff).sum()/tr.norm(sigma))
    print("Orthogonality check: ",tr.norm(tr.einsum('brkxy,bskxy->brsxy',r,r) - tr.eye(3).view(1,3,3,1,1)))

    #print(sigma-ff)
    #save the last field configuration
    tr.save(sigma,'o3_'+str(lat[0])+"_"+str(lat[1])+"_b"+str(beta)+"_bs"+str(o.Bs)+".pt")
    
def main():
    import argparse
    import sys
    

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='rg')
    parser.add_argument('-l', default='no-load')

    args = parser.parse_args()
    if(args.t=="rg"):
        print("Testing RG Layer")
        test_RGlayer(args.l)
    else:
        print("Nothing to test")
        
if __name__ == "__main__":
   main()
    
            



