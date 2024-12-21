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
parser.add_argument('-f'  , default='no-load')
parser.add_argument('-d'  , type=int  , default=1    )
parser.add_argument('-e'  , type=int  , default=1000 )
parser.add_argument('-L'  , type=int  , default=16   )
parser.add_argument('-m'  , type=float, default=-0.5 )
parser.add_argument('-g'  , type=float, default=1.0  )
parser.add_argument('-b'  , type=int  , default=4    )
parser.add_argument('-nb' , type=int  , default=4    ) # number batch sizes to use
parser.add_argument('-lr' , type=float, default=1e-4 )
parser.add_argument('-nlr', type=int  , default=4    ) # number of learning rates
parser.add_argument('-w'  , type=int  , default=16   )
parser.add_argument('-nl' , type=int  , default=1    )
parser.add_argument('-sb' , type=int  , default=1    )
parser.add_argument('-nc' , type=int  , default=1    )
parser.add_argument('-fbj', type=bool , default=False)
parser.add_argument('-sbj', type=bool , default=False)
parser.add_argument('-dev', type=int  , default=-1)

args = parser.parse_args()

file=args.f
if(args.f=="no-load"):
     load_flag=False
else:
    load_flag=True
    file=args.f



device = tr.device("cpu")
# Check that MPS is available
if not tr.backends.mps.is_available():
    if not tr.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
     if (args.dev >= 0 ):
          device = tr.device("mps")

# Check that CUDA is available
if not tr.cuda.is_available():
    if not tr.backends.cuda.is_built():
        print("CUDA not available because the current PyTorch install was not "
              "built with CUDA enabled.")
    else:
        print("CUDA not available or you do not have a GPU on this machine.")

else:
     if (args.dev >= 0 ):
          device = tr.device("cuda:"+str(args.dev))


    
print(f"Using {device} device")

depth = args.d
L=args.L
batch_size=args.b
epochs= args.e
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
     print("Using parity symmetric bijector")
     bij = lambda: m.FlowBijectorParity(Nlayers=Nlayers,width=width)
     if(args.fbj):
          bij_list = []
          for k in range(2*Nconvs):
               bij_list.append(m.FlowBijectorParity(Nlayers=Nlayers,width=width))
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

tag = str(L)+"_m"+str(mass)+"_l"+str(lam)+"_w_"+str(width)+"_l_"+str(Nlayers)+"_nc_"+str(Nconvs)+"_st_"+str(depth)
if(args.fbj):
     tag = tag + "_fbj"
if(args.sbj):
     tag = tag + "_sbj"
if(load_flag):
    sm.load_state_dict(tr.load(file))
    sm.eval()

sm.to(device)

if(not load_flag):
    file = "sm_phi4_"+tag+".dict"
for b in batch_size*(2**np.arange(args.nb)):
     for lr in args.lr*(0.1**np.arange(args.nlr)):
          print("Running with batch_size = ",b, " and learning rate= ",lr)
          loss_hist=trainSM(sm,levels=[], epochs=epochs,batch_size=b,super_batch_size=args.sb,learning_rate=args.lr)
          batch_size = b*args.sb
          tt = tag+"_b"+str(batch_size)+"_lr_"+str(lr)
          plot_loss(loss_hist,tt)
          validate(b,10*args.sb,tt,sm)
          tr.save(sm.state_dict(), file)



   
