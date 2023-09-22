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

def jackknife(d):
    # d is the list containing data
    # NOTE: it works with python lists not numpy arrays
    #
    N=len(d) -1 
    ss = sum(d)
    r=[]
    for n in range(len(d)):
        r.append((ss-d[n])/N)

    return r

def tr_jackknife(d,func):
     # d is a torch tensor with the first dimension being a batch
     # function is evaluated in the data
     r =tr.zeros_like(d)
     for j in range(d.shape[0]):
          tt = tr.cat((d[:j],d[(j+1):]),0)
          #print(tt,tt.std())
          r[j]=func(tt).mean()

     return r

def check_model(L=16,
                lam=1.0,
                mass=-0.5,
                batch_size=1024,
                width=4,
                Nlayers=1,
                depth=1,
                path='./trained_models'):

    V=L*L
    o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
    phi = o.hotStart()

    #set up a prior
    normal = distributions.Normal(tr.zeros(V,device=device),tr.ones(V,device=device))
    prior= distributions.Independent(normal, 1)

    bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
    mg = lambda : m.MGflow([L,L],bij,m.RGlayer("average"),prior)
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

    tag = str(L)+"_m"+str(mass)+"_l"+str(lam)+"_w_"+str(width)+"_l_"+str(Nlayers)+"_st_"+str(depth)

    file = path+"/sm_phi4_"+tag+".dict"

    sm.load_state_dict(tr.load(file))
    sm.eval()

    print("Model check")
    x=sm.sample(batch_size)
    diff = sm.diff(x).detach()
    m_diff = diff.mean()
    diff -= m_diff

    Jstd  = lambda x : tr.std(x)*np.sqrt(x.shape[0]-1)
    e_s_std = tr_jackknife(diff,tr.std).std().numpy()*np.sqrt(diff.shape[0]-1)
    s_std = diff.std().numpy()

    print("max  action diff: ", tr.max(diff.abs()).numpy())
    print("min  action diff: ", tr.min(diff.abs()).numpy())
    print("mean action diff: ", m_diff.detach().numpy())
    print("std  action diff: ", s_std, " +/- ", e_s_std)
    #compute the reweighting factor
    foo = tr.exp(-diff)
    #print(foo)
    w = foo/tr.mean(foo)

    print("mean re-weighting factor: " , w.mean().numpy())
    print("std  re-weighting factor: " , w.std().numpy())

    logbins = np.logspace(np.log10(1e-3),np.log10(1e3),int(w.shape[0]/10))
    _=plt.hist(w,bins=logbins)
    plt.xscale('log')
    plt.title('Reweighting factor distribution')
    plt.xlabel('Reweighting factor ')
    plt.savefig(path+"/sm_rw_"+tag+".pdf")
    #plt.show()
    plt.close()
    _=plt.hist(diff.detach(),bins=int(w.shape[0]/10))
    plt.title('ΔS distribution')
    plt.xlabel('ΔS')
    plt.savefig(path+"/sm_ds_"+tag+".pdf")
    #plt.show()
    plt.close()

    table = {width : {Nlayers : (s_std.item(),e_s_std)} }
    print(f"table[{width:2d}].update({{{Nlayers:1d} : ({s_std:2.6f},{e_s_std:2.6f})}})")
    return s_std



parser = argparse.ArgumentParser()
parser.add_argument('-p' , default="./trained_models/")
parser.add_argument('-d' , type=int  , default=1   )
parser.add_argument('-L' , type=int  , default=16  )
parser.add_argument('-m' , type=float, default=-0.5)
parser.add_argument('-g' , type=float, default=1.0 )
parser.add_argument('-b' , type=int  , default=1024)
#parser.add_argument('-w' , type=int  , default=16  )
#parser.add_argument('-nl', type=int  , default=1   )


args = parser.parse_args()
depth = args.d
path = args.p

device = "cuda" if tr.cuda.is_available() else "cpu"
print(f"Using {device} device")

L=args.L
batch_size=args.b


lam =args.g
mass=args.m

Width = [4,8,16,32]
Nlayers = [1,2,3]
color = { 1: "tab:red", 2: "tab:green", 3 :  "tab:orange"}
table ={}
Ntries = 16
mean_table = {}
std_table = {}
for w in Width:
    mean_table.update({w:{}})
    std_table.update({w:{}})
    table.update({w:{}})
    for l in Nlayers:
        table[w].update({l : []})
        for k in range(Ntries):
            ds = check_model(L=L,
                             lam=lam,
                             mass=mass,
                             batch_size=args.b,
                             width=w,
                             Nlayers=l,
                             depth=depth,
                             path=args.p)
            table[w][l].append(ds)
        mean_table[w].update({l : np.mean(table[w][l])})
        std_table[w].update({l : np.std(table[w][l])})


#print(table)
for l in Nlayers:
    y = []
    min_y = []
    max_y = []
    for w in Width:
        y.append (mean_table[w][l])
        min_y.append(mean_table[w][l]-std_table[w][l])
        max_y.append(mean_table[w][l]+std_table[w][l])
        
    plt.plot(Width,y,color = color[l], marker='o',label='Bijector Layer No: '+str(l))
    plt.fill_between(Width, min_y, max_y, color = color[l],alpha=0.2)

plt.xlabel('Width')
plt.ylabel('std(ΔS)')
plt.title("Number of MG Layers: 1")
plt.legend(loc='center right')
#plt.show()


file="ds_L"+str(L)+"_g"+str(lam)+"_m"+str(mass)+"_st"+str(depth)+".pdf"
plt.savefig(file)
