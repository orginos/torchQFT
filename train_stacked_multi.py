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
parser.add_argument('-e' , type=int  , default=1000)
parser.add_argument('-L' , type=int  , default=16  )
parser.add_argument('-m' , type=float, default=-0.5)
parser.add_argument('-g' , type=float, default=1.0 )
parser.add_argument('-b' , type=int  , default=4   )
parser.add_argument('-nb', type=int  , default=4   ) # number batch sizes to use
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-w' , type=int  , default=16  )
parser.add_argument('-nl', type=int  , default=1   )
parser.add_argument('-sb', type=int  , default=1   )

args = parser.parse_args()

file=args.f
if(args.f=="no-load"):
     load_flag=False
else:
    load_flag=True
    file=args.f

NUM_GPU = tr.cuda.device_count()
if NUM_GPU <1 :
     device = "cpu"
else:
     device = "cuda"
     
print(f"Using {device} device")
print(f"Number of GPUS {NUM_GPU}")


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
bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
mg = lambda : m.MGflow([L,L],bij,m.RGlayer("average"),prior)
models = []

print("Initializing ",depth," stages")
for d in range(depth):
    models.append(mg())
        
sm = SuperModel(models,target =o.action )

tag = str(L)+"_m"+str(mass)+"_l"+str(lam)+"_w_"+str(width)+"_l_"+str(Nlayers)+"_st_"+str(depth)
#load the serial model first
#I always save the serial model
if(load_flag):
    sm.load_state_dict(tr.load(file))
    sm.eval()

# Make the model parallel if I have more than 1 GPUs
if NUM_GPU > 1:
     	sm = MyDataParallel(sm)
#	sm = tr.nn.DataParallel(sm)
        
sm=sm.to(device)
#sm.eval()
c=0
for tt in sm.parameters():
    #print(tt.shape)
    if tt.requires_grad==True :
        c+=tt.numel()
print("parameter count: ",c)



#sm.train()
for b in batch_size*(2**np.arange(args.nb)):
     print("Running with mini_batch_size = ",b," on ",NUM_GPU, " GPUs -> batch_size = ",b*NUM_GPU," and learning rate= ",args.lr)
     loss_hist=trainSM(sm,levels=[], epochs=epochs,batch_size=b*NUM_GPU,super_batch_size=args.sb,learning_rate=args.lr)
     tt = tag+"_b"+str(b)
     plot_loss(loss_hist,tt)
     validate(b*NUM_GPU,args.sb,tt,sm)


# This way it saves the scalar model
if(not load_flag):
    file = "sm_phi4_"+tag+".dict"
if NUM_GPU >1 :
     tr.save(sm.module.state_dict(), file)
else:
     tr.save(sm.state_dict(), file)
   
