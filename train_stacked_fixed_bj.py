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
parser.add_argument('-f' ,             default='no-load')
parser.add_argument("-cuda", type=int, default=-1)

parser.add_argument('-d' , type=int,   default=1    )
parser.add_argument('-e' , type=int,   default=1000 )
parser.add_argument('-L' , type=int,   default=16   )
parser.add_argument('-m' , type=float, default=-0.5 )
parser.add_argument('-g' , type=float, default=1.0  )
parser.add_argument('-b' , type=int,   default=4    )
parser.add_argument('-nb', type=int,   default=4    ) # number different batch sizes to use
parser.add_argument('-lr', type=float, default=1e-4 )
parser.add_argument('-w' , type=int,   default=16   )
parser.add_argument('-nl', type=int,   default=2    )

args = parser.parse_args()

file=args.f
if(args.f=="no-load"):
     load_flag=False
else:
    load_flag=True
    file=args.f

cuda=args.cuda
device = tr.device("cpu" if cuda<0 else "cuda:"+str(cuda))
print(f"Using {device} device")

#device = "cuda" if tr.cuda.is_available() else "cpu"
#print(f"Using {device} device")

depth = args.d
L=args.L
batch_size=args.b
epochs= args.e

V=L*L
lam =args.g
mass=args.m

o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
phi = o.hotStart()

#set up a prior
normal = distributions.Normal(tr.zeros(V).to(device),tr.ones(V).to(device))
prior= distributions.Independent(normal, 1)

width=args.w
Nlayers=args.nl
fixed_Bijector1 = m.FlowBijector(Nlayers=Nlayers,width=width)
fixed_Bijector2 = m.FlowBijector(Nlayers=Nlayers,width=width)

bij = m.BijectorFactory(fixed_Bijector1,fixed_Bijector2)

mg = lambda : m.MGflow([L,L],bij.bij,m.RGlayer("average"),prior).to(device)
#mg=mg.to(device)
models = []

print("Initializing ",depth," stages")
for d in range(depth):
    models.append(mg())
        
sm = SuperModel(models,target =o.action )
sm=sm.to(device)
c=0
for tt in sm.parameters():
    #print(tt.shape)
    if tt.requires_grad==True :
        c+=tt.numel()
print("parameter count: ",c)

tag = str(L)+"_m"+str(mass)+"_l"+str(lam)+"_w_"+str(width)+"_l_"+str(Nlayers)+"_st_"+str(depth)+"_fbj"
if(load_flag):
    sm.load_state_dict(tr.load(file))
    sm.eval()

    
for b in batch_size*(2**np.arange(args.nb)):
     print("Running with batch_size = ",b, " and learning rate= ",args.lr)
     loss_hist=trainSM(sm,levels=[], epochs=epochs,batch_size=b,super_batch_size=1,learning_rate=args.lr)
     tt = tag+"_b"+str(b)
     plot_loss(loss_hist,tt)
     validate(1024,tt,sm)

if(not load_flag):
    file = "sm_phi4_"+tag+".dict"
tr.save(sm.state_dict(), file)
   
