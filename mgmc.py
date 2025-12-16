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


# tex setup
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



# Vcycle  for normal and real nvp implementation
def V_cycle(phi_o,sgc_c,mgf_c,hmc_f,m1,m2,Nskip=1,mode="normal"):#mode="rnvp"
    phi_o= hmc_f.evolve(phi_o,m1)
    sgc = phi4_c1(sgc_c,mgf_c,device=sgc_c.device,dtype=sgc_c.dtype,mode=mode)
    mn2c = i.minnorm2(sgc.force,sgc.evolveQ,7,1.0)
    hmcc = u.hmc(T=sgc,I=mn2c,verbose=True)
    sgc.generate_cfg_levels(phi_o)
    phic=sgc.phis[-1]
    phic_up=hmcc.evolve(phic,Nskip)
    #now go back up
    for sss in reversed(range(mgf_c.depth)):
        phic_up= mgf_c.rg.refine(phic_up,sgc.pics[sss])

    #print("shape of fine field ",phic_up.shape)
    phic_up=hmc_f.evolve(phic_up,m2)
    return phic_up, hmcc.AcceptReject

#get_observables_MCMG(sg,mgf, hmc, phi, Nwarm, Nmeas, pp=)

def get_observables_MCMG(sg,mgf, hmc, phi, Nwarm, Nmeas,pp="print",mode="normal"):

    tic=time.perf_counter()
    Vol=sg.Vol
    lat=[phi.shape[1], phi.shape[2]]
    toc=time.perf_counter()

    print(f"time {(toc - tic)*1.0e6/Nwarm:0.4f} micro-seconds per HMC trajecrory")

    lC2p = []
    lchi_m = []
    E = []
    av_phi = []
    phase=tr.tensor(np.exp(1j*np.indices(tuple(lat))[0]*2*np.pi/lat[0]),dtype=sg.dtype,device=sg.device)
    for k in range(Nmeas):
        ttE = sg.action(phi)/Vol
        E.extend(ttE)
        av_sigma = tr.mean(phi.view(sg.Bs,Vol),axis=1)
        av_phi.extend(av_sigma)
        chi_m = av_sigma*av_sigma*Vol
        p1_av_sig = tr.mean(phi.view(sg.Bs,Vol)*phase.view(1,Vol),axis=1)
        C2p = tr.real(tr.conj(p1_av_sig)*p1_av_sig)*Vol
        if(k%10==0) and pp=="print":
            print("k= ",k,"(av_phi,chi_m, c2p, E) ", av_sigma.mean().detach().numpy(),chi_m.mean().detach().numpy(),C2p.mean().detach().numpy(),ttE.mean().detach().numpy())
        lC2p.extend(C2p)
        lchi_m.extend(chi_m)
        ## HMC update but also V cycle
        phi,accept=V_cycle(phi,sg,mgf,hmc,1,1,Nskip=Nskip,mode=mode)

    return lC2p, lchi_m, E, av_phi, phi



class MGflow1(nn.Module):
    def __init__(self,size,bijector,rg,prior,Nconvs=1,depth=None):
        super(MGflow1, self).__init__()
        self.prior=prior
        self.rg=rg
        self.size = size
        minSize = min(size)
        print("Initializing MGflow module with size: ",minSize)
        if depth==None:
            self.depth = int(np.log(minSize)/np.log(2))
        else:
            self.depth = depth
        print("Using depth: ", self.depth)
        print("Using rg type: ",rg.type)
        sizes = []
        for k in range(self.depth):
            sizes.append([int(size[i]/(2**k)) for i in range(len(size))])
            print("(depth, size): ", k, sizes[-1])
            
            
        # the module list are ordered from fine to coarse
        self.cflow=tr.nn.ModuleList([m.ConvFlowLayer(sizes[k],bijector,Nconvs) for k in range(self.depth)])

    #noise to fields
    def forward(self,z):
        x = z
        
        # can I use lists and still expect autgrad to work?
        fines = []
        #take the noise to the coarsest level
        for k in range(self.depth-1):
            c,f =self.rg.coarsen(x)
            #print(c.shape,f.shape)
            x=c
            fines.append(f)
        #print("Number of fine levels: ", len(fines))
        # now reverse order to get back to fine
        # x should now be coarsest possible
        #print("Size of x: ", x.shape)
        for k in range(self.depth-1,0,-1):
            #print(k)
            fx=self.cflow[k](x)
            x=self.rg.refine(fx,fines[k-1])
        fx = self.cflow[0](x)
        #print("Size of fx at the end:",fx.shape)
        
        return fx

    #fields to noise
    def backward(self,x):
        log_det_J=x.new_zeros(x.shape[0])

        # can I use lists and still expect autgrad to work?
        fines = []
        for k in range(self.depth-1):
            #print(k,"shape(x)",x.shape)
            fx,J = self.cflow[k].backward(x)
            log_det_J += J
            cx,ff = self.rg.coarsen(fx)
            fines.append(ff)
            x=cx
        #print("end","shape(x)",x.shape)
        #for k in range(len(fines)):
            #print(k,"shape of fines",fines[k].shape)
        fx,J = self.cflow[self.depth-1].backward(x)
        log_det_J += J
        #move the noise to the finest level
        for k in range(self.depth-2,-1,-1):
            #print(k,"sizes", fx.shape,fines[k].shape)
            z=self.rg.refine(fx,fines[k])
            #print("Size of z at the end:",z.shape)
            #print("fx at the end:",fx.shape)  
            fx=z
        return fx,log_det_J

    def log_prob(self,x):
        z, logp = self.backward(x)
        #print("In log prob z.shape: ", z.shape)
        #print("In log prob z.shape: ", z.shape)
        return self.prior.log_prob(z.flatten(start_dim=1)) + logp

    def sample(self, batchSize): 
        #z = self.prior.sample((batchSize, 1)).reshape(batchSize,self.size[0],self.size[1])
        z = self.prior_sample(batchSize)
        x = self.forward(z)
        return x

    # generate a sample from the prior
    def prior_sample(self,batch_size):
        return self.prior.sample((batch_size,1)).reshape(batch_size,self.size[0],self.size[1])


class phi4_c1:
    def action(self,phi_c):
        rphis=[]
        rphis.append(phi_c)
        iii=0
        logdet_total=tr.zeros(phi_c.shape[0],dtype=self.dtype,device=self.device)
        for pi in reversed(self.pics):
            #print(pi.shape)
            rphi = self.rg.refine(rphis[iii],pi)
            if self.mode=="rnvp":
                #flowback through the network
                ######This was the code before
                #rphi,logdet = self.mgf.cflow[self.level-1-iii].backward(rphi)
                #logdet_total+=logdet
                #print("log_det",logdet,"level",self.level-1-iii)
                rphi = self.mgf.cflow[self.level-1-iii].forward(rphi)
                logdet_total+=self.mgf.cflow[self.level-1-iii].log_prob(rphi)
            #print("log_det",logdet,"level",self.level-1-iii)
            rphis.append(rphi)
            iii+=1
        #phi_f = rphis[-1]
        #evaluate coarse field in action of rg
        #print(phi_f.shape,"shape of fine field")
        return self.sg.action(rphi)-logdet_total ## NANs
        #if I dont add the .sum() I got a grad for the batch system, it seems to me that we include that in the force property the batch is summed?

    def force(self, phi_c):
        x_tensor = phi_c.clone().requires_grad_()

        S = self.action(x_tensor)
        grad = tr.autograd.grad(S.sum(), x_tensor, retain_graph=False)[0]

        if grad is None:
            print("[ERROR] Gradient is None.")
            raise RuntimeError("autograd.grad returned None.")
        #release memory on gpu
        return -grad.detach()
    
    def refreshP(self):
        P = tr.normal(0.0,1.0,self.phis[-1].shape).to(self.device).to(self.dtype)#only difference with fine level
        return P

    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    
    def kinetic(self,P):
        return tr.einsum('bxy,bxy->b',P,P)/2.0

    def generate_cfg_levels(self,phi11):#run every time we need to contruct deeper or superficial levels
        #run a configuration
        self.level = self.mgf.depth
        phis=[]
        pis=[]
        phicopy=phi11.clone().to(self.device).to(self.dtype)
        
        #print("shape of the original field",phicopy.shape)
        phis.append(phicopy)
        for step in range(self.level):
            if self.mode=="rnvp":
                phicopy= self.mgf.cflow[step].forward(phicopy)
                #print("device phicopy ",phicopy.device," dtype ",phicopy.dtype)
            #print("coarsening level ",_," field shape ",phicopy.shape)
            phic,pic = self.rg.coarsen(phicopy)

            phis.append(phic)
            pis.append(pic)
            phicopy=phic
        self.phis=phis
        self.pics=pis

        #reversed
        rphis=[]
        
        rphis.append(phis[-1])
        #self.mgf.cflow[step].backward(rphis[0])
        sss=0
        for phics,pis in zip(reversed(phis),reversed(pis)):
            
            rphi = self.rg.refine(phics,pis)
            #flowback through the network
            if self.mode=="rnvp":
                rphi,logdet = self.mgf.cflow[self.level-1-sss].backward(rphi)
                #print("log_det",logdet,"level",self.level-1-sss)
            sss+=1
            
            rphis.append(rphi)
        self.rphis=rphis

    def __init__(self,sgg,mgf,device="cpu",dtype=tr.float64,mode="rnvp"):
        self.sg = sgg #theory? in the finest level
        self.mgf = mgf #neural net
        self.rg = mgf.rg #projector to coarse level
        self.mode = mode
        print("multigrid is done by: ",self.mode)
        self.device = device
        self.dtype = dtype


def main():
    import integrators as i

    device = tr.device("cuda")
    dtype=tr.float32
    L=128
    lat = [L,L]
    V=L*L
    # This set of params is very very close to critical.
    lam = 2.4
    mas = -0.55

    normal = distributions.Normal(tr.zeros(V,dtype=dtype,device=device),tr.ones(V,dtype=dtype,device=device))
    prior= distributions.Independent(normal, 1)


    Nwarm = 1
    Nmeas = 1000
    Nskip = 1
    batch_size = 10

    Vol = np.prod(lat)
    sg = s.phi4(lat,lam,mas,batch_size=batch_size,device=device,dtype=dtype)
    phi = sg.hotStart()
    mn2 = i.minnorm2(sg.force,sg.evolveQ,10,1.0)
    print(phi.shape,Vol,tr.mean(phi),tr.std(phi))
    hmc = u.hmc(T=sg,I=mn2,verbose=False)

    FLOW=lambda: m.FlowBijectorParity(Nlayers=1,width=32)

    mgf=MGflow1([L,L],FLOW,m.RGlayer("average",batch_size=batch_size,dtype=dtype,device=device),prior,depth=1).to(device)#.double()

