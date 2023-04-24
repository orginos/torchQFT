import numpy as np
import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter
import phi4_mg as m
import phi4 as p
import integrators as i
import update as u
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
import time
from stacked_model import *


parser = argparse.ArgumentParser()
parser.add_argument('-f'               , default='no-load')
parser.add_argument("-cuda", type=int  , default=-1)
parser.add_argument('-d'   , type=int  , default=1   )
parser.add_argument('-L'   , type=int  , default=16  )
parser.add_argument('-m'   , type=float, default=-0.5)
parser.add_argument('-g'   , type=float, default=1.0 )
parser.add_argument('-b'   , type=int  , default=128 )
parser.add_argument('-w'   , type=int  , default=16  )
parser.add_argument('-nl'  , type=int  , default=1   )

args = parser.parse_args()

file=args.f
if(args.f=="no-load"):
     load_flag=False
else:
    load_flag=True
    file=args.f

''' 
file = args.f
if(args.cuda=="-1"):
    load_flag = False
    cuda = int(np.array(file["cuda"]))
    d = int(np.array(file["d"]))
    L = int(np.array(file["L"]))
    mass = float(np.array(file["m"]))
    g = float(np.array(file["g"]))
    b = int(np.array(file["b"]))
    w = int(np.array(file["w"]))
    nl = int(np.array(file["nl"]))
else:
    load_flag = True
    cuda = args.cuda
    d = args.d
    L = args.L
    mass = args.m
    g = args.g
    b = args.b
    w = args.w
    nl = args.nl
    file=args.f
'''


cuda=args.cuda
device = tr.device("cpu" if cuda<0 else "cuda:"+str(cuda))
print(f"Using {device} device")

#device = tr.device("cpu" if cuda<0 else "cuda:"+str(cuda))
#print(f"Using {device} device")

depth = args.d
L=args.L
batch_size=args.b

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
sm = sm.to(device)

c=0
for tt in sm.parameters():
    #print(tt.shape)
    if tt.requires_grad==True :
        c+=tt.numel()
print("parameter count: ",c)

#sm.load_state_dict(tr.load(file))
sm.eval()

print("starting model check")
validate(batch_size,'foo',sm)

