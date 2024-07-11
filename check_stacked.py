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


parser = argparse.ArgumentParser()
parser.add_argument('-f' , default='no-load')
parser.add_argument('-d' , type=int  , default=1   )
parser.add_argument('-L' , type=int  , default=16  )
parser.add_argument('-m' , type=float, default=-0.5)
parser.add_argument('-g' , type=float, default=1.0 )
parser.add_argument('-b' , type=int  , default=128 )
parser.add_argument('-w' , type=int  , default=16  )
parser.add_argument('-nl', type=int  , default=1   )
parser.add_argument('-sb' , type=int  , default=1    )
parser.add_argument('-nc' , type=int  , default=1    )
parser.add_argument('-fbj', type=bool , default=False)



args = parser.parse_args()

file=args.f
if(args.f=="no-load"):
     load_flag=False
else:
    load_flag=True
    file=args.f

device = "cuda" if tr.cuda.is_available() else "cpu"
print(f"Using {device} device")

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
bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
if(args.fbj):
     bij_list = []
     for k in range(2*Nconvs):
          bij_list.append(m.FlowBijector(Nlayers=Nlayers,width=width))
     bij = m.BijectorFactory(bij_list).bij

mg = lambda : m.MGflow([L,L],bij,m.RGlayer("average"),prior,Nconvs=Nconvs)
models = []

print("Initializing ",depth," stages")
for d in range(depth):
    models.append(mg())
        
sm = SuperModel(models,target =o.action )

c=0
for tt in sm.parameters():
    #print(tt.shape)
    if tt.requires_grad==True :
        c+=tt.numel()
print("parameter count: ",c)

sm.load_state_dict(tr.load(file))
sm.eval()

print("starting model check")
validate(batch_size,args.sb,'foo',sm)

