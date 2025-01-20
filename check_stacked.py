import numpy as np
import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter
import phi4_mg as m
import phi4 as p
import integrators as i
import update as u

import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import sys
    
import time
from stacked_model import *


def tr_jackknife(d,func):
     # d is a torch tensor with the first dimension being a batch
     # function is evaluated in the data
     r =tr.zeros_like(d)
     for j in range(d.shape[0]):
          tt = tr.cat((d[:j],d[(j+1):]),0)
          #print(tt,tt.std())
          r[j]=func(tt).mean()

     return r


Jstd  = lambda x : tr.std(x)*np.sqrt(x.shape[0]-1)

def ver_print(verb,*args):
     #print(verb)
     if verb:
          print(*args)
          
def check_stacked_phi4(args):

     file=args.f
     if(args.f=="no-load"):
          load_flag=False
     else:
          load_flag=True
          file=args.f

     device = "cuda" if tr.cuda.is_available() else "cpu"
     ver_print(args.ver,f"Using {device} device")
     
     depth = args.d
     L=args.L
     batch_size=args.b
     Nconvs = args.nc

     V=L*L
     lam =args.g
     mass=args.m
          
     o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
     phi = o.hotStart()

     #set up a prior
     normal = distributions.Normal(tr.zeros(V,device=device),tr.ones(V,device=device))
     prior= distributions.Independent(normal, 1)

     width=args.w
     Nlayers=args.nl

     #non parity symmetric (default) bijector
     bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
     if(args.fbj):
          bij_list = []
          for k in range(2*Nconvs):
               bij_list.append(m.FlowBijector(Nlayers=Nlayers,width=width))
          bij = m.BijectorFactory(bij_list).bij

     #use parity symmetric bijector
     if(args.sbj):
          ver_print(args.ver,"Using parity symmetric bijector")
          bij = lambda: m.FlowBijectorParity(Nlayers=Nlayers,width=width)
          if(args.fbj):
               bij_list = []
               for k in range(2*Nconvs):
                    bij_list.append(m.FlowBijectorParity(Nlayers=Nlayers,width=width))
               bij = m.BijectorFactory(bij_list).bij

     mg = lambda : m.MGflow([L,L],bij,m.RGlayer("average"),prior,Nconvs=Nconvs)
     models = []

     ver_print(args.ver,"Initializing ",depth," stages")
     for d in range(depth):
          models.append(mg())
        
     sm = SuperModel(models,target =o.action )

     c=0
     for tt in sm.parameters():
          #ver_print(args.ver,tt.shape)
          if tt.requires_grad==True :
               c+=tt.numel()
     ver_print(args.ver,"parameter count: ",c)

     sm.load_state_dict(tr.load(file))
     sm.eval()

     ver_print(args.ver,"starting model check")
     x=sm.sample(batch_size)
     diff = sm.diff(x).detach()
     for b in range(1,args.sb):
        x=sm.sample(batch_size)
        diff = tr.cat((diff,sm.diff(x).detach()),0)
                      
     m_diff = diff.mean()     
     diff -= m_diff

     std =  diff.std().numpy()
     e_std = tr_jackknife(diff,tr.std).std().numpy()*np.sqrt(diff.shape[0]-1)
     ver_print(args.ver,"max  action diff: ", tr.max(diff.abs()).numpy())
     ver_print(args.ver,"min  action diff: ", tr.min(diff.abs()).numpy())
     ver_print(args.ver,"mean action diff: ", m_diff.detach().numpy())
     ver_print(args.ver,"std  action diff: ",std,"+/-",e_std)
     #compute the reweighting factor
     foo = tr.exp(-diff)
     #ver_print(args.ver,foo)
     w = foo/tr.mean(foo)

     rw = w.mean().numpy()
     e_rw = w.std().numpy()
     ver_print(args.ver,"mean re-weighting factor: " , rw)
     ver_print(args.ver,"std  re-weighting factor: " , e_rw)

     ESS = (foo.mean())**2/(foo*foo).mean()
     ver_print(args.ver,"ESS                     : " , ESS.numpy())

     return std,e_std, rw,e_rw, ESS


def main():
     parser = argparse.ArgumentParser()
     parser.add_argument('-f'  , default='no-load')
     parser.add_argument('-d'  , type=int  , default=1    )
     parser.add_argument('-L'  , type=int  , default=16   )
     parser.add_argument('-m'  , type=float, default=-0.5 )
     parser.add_argument('-g'  , type=float, default=1.0  )
     parser.add_argument('-b'  , type=int  , default=128    )
     parser.add_argument('-w'  , type=int  , default=16   )
     parser.add_argument('-nl' , type=int  , default=1    )
     parser.add_argument('-sb' , type=int  , default=1    )
     parser.add_argument('-nc' , type=int  , default=1    )
     parser.add_argument('-fbj', type=bool , default=False)
     parser.add_argument('-sbj', type=bool , default=False)
     parser.add_argument('-ver', type=bool , default=False)
     
     args = parser.parse_args()

     args.ver=True
     check_stacked_phi4(args)
     
if __name__ == "__main__":
     main()
    



   
