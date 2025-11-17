import numpy as np
import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter
import phi4_mg as m
import phi4 as s
import integrators as i
import update as u

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

###### Architectures and parameters #################################

### this is the first attempt at a non-linear restriction/prolongation operator but not invertible
class NonLinearRGlayer(nn.Module):
    def __init__(self, channels=1, hidden_channels=8, batch_size=1):
        super(NonLinearRGlayer, self).__init__()
        self.batch_size = batch_size
        
        # Restrictor: one small conv + downsampling
        self.restrict_net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=2)
        )

        # Prolongator: upsampling + conv
        self.prolong_net = nn.Sequential(
            nn.ConvTranspose2d(channels, hidden_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1)
        )

    def coarsen(self, f):
        ff = f.view(f.shape[0], 1, f.shape[1], f.shape[2])  # B x 1 x H x W
        c = self.restrict_net(ff)
        r = ff - self.prolong_net(c)
        if self.batch_size == 1:
            return c.squeeze(1), r.squeeze(1)
        else:
            return c.squeeze(), r.squeeze()

    def refine(self, c, r):
        cc = c.view(c.shape[0], 1, c.shape[1], c.shape[2])
        rr = r.view(r.shape[0], 1, r.shape[1], r.shape[2])
        f_rec = self.prolong_net(cc) + rr
        if self.batch_size == 1:
            return f_rec.squeeze(1)
        else:
            return f_rec.squeeze()




parser = argparse.ArgumentParser()


parser.add_argument('-Nskip' , type=int,   default=1 , help="number of skipped trajectories between measurements")
parser.add_argument('-lam'     , type=float, default=2.4 , help="Coupling constant")
parser.add_argument('-batch', type=int,   default=10  , help="HMC batch size")
parser.add_argument('-L'     , type=int,   default=8, help="Lattice size")
parser.add_argument('-dev'   , type=int,   default=-1 , help="Device number, -1 for CPU, 0,1,... for GPU")
parser.add_argument("-warm",   type=int,   default=1000 , help="Number of warmup HMC steps")
parser.add_argument("-meas",   type=int,   default=10000 , help="Number of measurement HMC steps")
parser.add_argument('-m'     , type=float, default=-0.5, help="Mass parameter")

args = parser.parse_args()

L=args.L
lat = [L,L]

# This set of params is very very close to critical.
lam = args.lam
#
Nwarm = args.warm
Nmeas = args.meas
Nskip = args.Nskip
batch_size = args.batch
mass = args.m
device = args.dev

Vol = np.prod(lat)
sg = s.phi4(lat,lam,mass,batch_size=batch_size,device=device)
phi = sg.hotStart()
mn2 = i.minnorm2(sg.force,sg.evolveQ,7,1.0)
print(phi.shape,Vol,tr.mean(phi),tr.std(phi))
hmc = u.hmc(T=sg,I=mn2,verbose=False)


lC2p, lchi_m, E, av_phi, phi = gm.get_observables_hist(sg, hmc, phi, Nwarm, Nmeas, Nskip)
#tau_phi1,tau_suscept1 = get_autocorrelationtime(av_phi, lchi_m)
#map traces to cpu and do the analysis there
results_av = gm.gamma_method_with_replicas(gm.split_first_dim_to_list(tr.stack(av_phi).T.unsqueeze(2).to(device)), lambda A: A[0])
results_lchi = gm.gamma_method_with_replicas(gm.split_first_dim_to_list(tr.stack(lchi_m).T.unsqueeze(2).to(device)), lambda A: A[0])

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

#save results to a text file with out removing previous data
with open("MCMG_results_L"+str(L)+"_lam"+str(lam)+".txt", "a") as f:
    #names of columns
    f.write("#mass Nwarm Nmeas Nskip batch_size phi_av_mean phi_av_std tau_phi1 dtau_phi1 sucept_mean sucept_std tau_suscept1 dtau_suscept1\n")
    f.write(f"{mass} {Nwarm} {Nmeas} {Nskip} {batch_size} {phi_av_mean} {phi_av_std} {tau_phi1} {dtau_phi1} {sucept_mean} {sucept_std} {tau_suscept1} {dtau_suscept1}\n")

