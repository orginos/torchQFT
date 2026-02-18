import numpy as np
import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter
import phi4_mg as m
import phi4 as s
import integrators as integ
import update as u

import mgmc as mgmc


import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import sys
    
import time
from stacked_model import *

import Gamma_error as gm

import matplotlib.pyplot as plt
import matplotlib as mpl

import os
os.environ["PATH"] = "/sciclone/home/yacahuanamedra/texlive/bin/x86_64-linux:" + os.environ["PATH"]

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')
import pickle


from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

mpl.rc('font', **font)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

mpl.rc('font', **font)


parser = argparse.ArgumentParser()


parser.add_argument('-Nskip' , type=int,   default=1 , help="number of skipped trajectories between measurements")
parser.add_argument('-lam'     , type=float, default=2.4 , help="Coupling constant")
parser.add_argument('-batch', type=int,   default=10  , help="HMC batch size")
parser.add_argument('-L'     , type=int,   default=8, help="Lattice size")
parser.add_argument('-dev'   , type=int,   default=-1 , help="Device number, -1 for CPU, 0,1,... for GPU")
parser.add_argument("-warm",   type=int,   default=1000 , help="Number of warmup HMC steps")
parser.add_argument("-meas",   type=int,   default=10000 , help="Number of measurement HMC steps")
parser.add_argument('-m'     , type=float, default=-0.5, help="Mass parameter")
parser.add_argument('-mcmg', type=str, default="hmc", help="hmc:use HMC, mgmc:use MGMC and rnvp:use MGMC+rnvp (or deepsets)")
parser.add_argument('-train', type=str, default="none", help="use 'DeltaS or KL' as arguments to train MGMC+rnvp (or deepsets)")
parser.add_argument('-Nlayers', type=int, default=4, help="Number of layers in the RNVP")
parser.add_argument('-width', type=int, default=4, help="Width of the RNVP or deepsets")
parser.add_argument('-Nlevels', type=int, default=2, help="define number of levels that are going to be used in the mcmg")
parser.add_argument('-snet', type=str, default="yes", help="use 'yes' or 'no' as arguments to define rnvp or deepsets respectively")

args = parser.parse_args()

#print arguments
print(args)

print("entre al programa",flush=True)


L=args.L
lat = [L,L]
V=L*L

# This set of params is very very close to critical.
lam = args.lam
Nwarm = args.warm
Nmeas = args.meas
Nskip = args.Nskip
batch_size = args.batch
mass = args.m
Nlevels = args.Nlevels
Nlayers = args.Nlayers
width = args.width
snet = args.snet



dtype = tr.float64



#device = args.dev
if args.dev == -1:
    device = "cuda"
else:
    device = "cpu"

#### new set up
normal = distributions.Normal(tr.zeros(V,dtype=dtype,device=device),tr.ones(V,dtype=dtype,device=device))
prior= distributions.Independent(normal, 1)

normal_c=distributions.Normal(tr.zeros(V//4,dtype=dtype,device=device),tr.ones(V//4,dtype=dtype,device=device))
prior_c=distributions.Independent(normal_c,1)


Vol = np.prod(lat)
sg = s.phi4(lat,lam,mass,batch_size=batch_size,device=device,dtype=dtype)
phi = sg.hotStart()

#clone phi
phi2 = phi.clone()

#typeof_mgf = "rnvp"
if args.mcmg == "mgmc" or args.mcmg == "hmc":
    typeof_mgf = "normal"
else:
    typeof_mgf = "rnvp"
if args.snet == "yes":
    FLOW=lambda: mgmc.FlowBijector_rnvp(Nlayers=Nlayers,width=width,device=device,ttp=dtype)
else:
    FLOW=lambda: mgmc.FlowBijector_deepsets(Nlayers=Nlayers,width=width,device=device,ttp=dtype)


mn2 = integ.minnorm2(sg.force,sg.evolveQ,20,1.0)
print(phi.shape,Vol,tr.mean(phi),tr.std(phi))
hmc_f = u.hmc(T=sg,I=mn2,verbose=False)

#FLOW=lambda: mgmc.FlowBijector_rnvp(Nlayers=Nlayers,width=width,device=device,ttp=dtype)


mgf=mgmc.MGflow1([L,L],FLOW,mgmc.RGlayer1("average",batch_size=batch_size,dtype=dtype,device=device),prior_c,depth=Nlevels,mode=typeof_mgf)#.double()

sgc = mgmc.phi4_c1(sg,mgf,device=sg.device,dtype=sg.dtype,mode="rnvp")
mn2c = integ.minnorm2(sgc.force,sgc.evolveQ,20,1.0)
hmc_c = u.hmc(T=sgc,I=mn2c,verbose=False)

mgf.generate_levels(phi)

c=0
for tt in mgf.parameters():
    #print(tt.shape)
    c+=tt.numel()

print("parameter count: ",c)






if args.mcmg == "hmc":
    lC2p, lchi_m, E, av_phi, phi = gm.get_observables_hist(sg, hmc_f, phi, Nwarm, Nmeas, Nskip)

elif args.mcmg == "mgmc":
    #lC2p, lchi_m, E, av_phi, phi = mgmc.get_observables_MCMG(sg,mgf, hmc_f,sgc, mn2c, hmc_c, phi, Nwarm, 100, pp="no",mode=typeof_mcmg)
    lC2p, lchi_m, E, av_phi, phi = mgmc.get_observables_MCMG(sg,mgf, hmc_f,sgc, mn2c, hmc_c, phi, Nwarm, Nmeas, pp="no",mode="normal")

elif args.mcmg == "rnvp":
    #lC2p, lchi_m, E, av_phi, phi = mgmc.get_observables_MCMG(sg,mgf, hmc_f, phi, Nwarm, Nmeas, pp="no",mode="rnvp")
    print("training......",flush=True)
    if args.train == "DeltaS":
        #mgmc.train_multigrid_DeltaS(mgf,phi,sg,steps=100)
        loss_history, mgf = mgmc.train_multigrid_DeltaS(mgf,phi,sg,steps=501)
    elif args.train == "KL":
        #loss_history ,mgf = mgmc.train_multigrid_KL(mgf,phi,sgc,steps=501)
        loss_history, mgf = mgmc.train_multigrid_KL(mgf,phi,sgc,steps=501)
    else:
        pass
    lC2p, lchi_m, E, av_phi, phi = mgmc.get_observables_MCMG(sg,mgf, hmc_f,sgc, mn2c, hmc_c, phi, Nwarm, Nmeas, pp="no",mode="rnvp")

# Determine the method label for the filename
method_label = args.mcmg
if args.mcmg == "rnvp":
    if args.snet == "yes":
        method_label = "rnvp_flow"
    else: # args.snet == "no"
        method_label = "deepsets_flow"

# Construct the output directory path
output_dir = f"phi4data_{method_label}"
os.makedirs(output_dir, exist_ok=True)

#tau_phi1,tau_suscept1 = get_autocorrelationtime(av_phi, lchi_m)
#map traces to cpu and do the analysis there
results_av = gm.gamma_method_with_replicas(gm.split_first_dim_to_list(tr.stack(av_phi).T.unsqueeze(2).to("cpu")), lambda A: A[0], max_lag=1300)
results_lchi = gm.gamma_method_with_replicas(gm.split_first_dim_to_list(tr.stack(lchi_m).T.unsqueeze(2).to("cpu")), lambda A: A[0], max_lag=1300)   

print("Gamma results for average phi:")
print(f"F = {results_av['value']:.6f} ± {results_av['dvalue']:.6f} (±{results_av['ddvalue']:.6f})")
print(f"tau_int = {results_av['tau_int']:.3f} ± {results_av['dtau_int']:.3f}")
print(f"W_opt = {results_av['W_opt']}, Q = {results_av['Q']}")
print("Gamma results for susceptibility:")
print(f"F = {results_lchi['value']:.6f} ± {results_lchi['dvalue']:.6f} (±{results_lchi['ddvalue']:.6f})")
print(f"tau_int = {results_lchi['tau_int']:.3f} ± {results_lchi['dtau_int']:.3f}")
print(f"W_opt = {results_lchi['W_opt']}, Q = {results_lchi['Q']}")


tau_phi1=results_av['tau_int']
tau_suscept1=results_lchi['tau_int']
dtau_phi1=results_av['dtau_int']
dtau_suscept1=results_lchi['dtau_int']
phi_av_mean=results_av['value']
phi_av_std=results_av['dvalue']
sucept_mean=results_lchi['value']
sucept_std=results_lchi['dvalue']

# save results to a text file with out removing previous data, but specify if is hmc or mgmc or rnvp with jacobian or not
output_filename = os.path.join(output_dir, f"MCMG_results_L{L}_lam{lam}.txt")

with open(output_filename, "a") as f:
    #names of columns
    if os.path.getsize(output_filename) == 0:    
        f.write("#mass Nwarm Nmeas Nskip batch_size phi_av_mean phi_av_std tau_phi1 dtau_phi1 sucept_mean sucept_std tau_suscept1 dtau_suscept1\n")
    f.write(f"{mass} {Nwarm} {Nmeas} {Nskip} {batch_size} {phi_av_mean} {phi_av_std} {tau_phi1} {dtau_phi1} {sucept_mean} {sucept_std} {tau_suscept1} {dtau_suscept1}\n")
