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


#still need to change the code of phi_coarse try Vcycle instead
def V_cycle(phi_o,sgc_f,mgf_c,hmc_f,sgc,hmc_c,m1,m2,Nskip=1,mode="normal"):#mode="rnvp"
    phi_o= hmc_f.evolve(phi_o,m1)
    
    sgc.generate_cfg_levels(phi_o)
    
    phi_c,logdet_c=mgf_c.backward(phi_o)
    phi_c1=hmc_c.evolve(phi_c,Nskip)
    #now go back up
    phi_f1,logdet_f=mgf_c.forward(phi_c1)
    #print("shape of fine field ",phic_up.shape)
    phi_f1=hmc_f.evolve(phi_f1,m2)
    
    return phi_f1, hmc_c.AcceptReject

#get_observables_MCMG(sg,mgf, hmc, phi, Nwarm, Nmeas, pp=)

def get_observables_MCMG(sg,mgf, hmc_f, sgc, mn2c, hmc_c, phi, Nwarm, Nmeas,pp="print",mode="normal"):

    tic=time.perf_counter()
    Vol=sg.Vol
    lat=[phi.shape[1], phi.shape[2]]
    toc=time.perf_counter()

    #sgc = phi4_c1(sg,mgf,device=sg.device,dtype=sg.dtype,mode=mode)
    #mn2c = i.minnorm2(sgc.force,sgc.evolveQ,20,1.0)
    #hmc_c = u.hmc(T=sgc,I=mn2c,verbose=True)

    print(f"time {(toc - tic)*1.0e6/Nwarm:0.4f} micro-seconds per HMC trajecrory")

    lC2p = []
    lchi_m = []
    E = []
    av_phi = []
    phase=tr.tensor(np.exp(1j*np.indices(tuple(lat))[0]*2*np.pi/lat[0]),dtype=sg.dtype,device=sg.device)
    
    # Run sampling in no_grad mode to save memory. 
    # Force calculation in HMC will still work because it uses local requires_grad_().
    with tr.no_grad():
        for k in tqdm(range(Nmeas)):
            ttE = sg.action(phi)/Vol
            E.append(ttE)
            av_sigma = tr.mean(phi.view(sg.Bs,Vol),axis=1)
            av_phi.append(av_sigma)
            chi_m = av_sigma*av_sigma*Vol
            p1_av_sig = tr.mean(phi.view(sg.Bs,Vol)*phase.view(1,Vol),axis=1)
            C2p = tr.real(tr.conj(p1_av_sig)*p1_av_sig)*Vol
            if(k%10==0) and pp=="print":
                print("k= ",k,"(av_phi,chi_m, c2p, E) ", av_sigma.mean().detach().numpy(),chi_m.mean().detach().numpy(),C2p.mean().detach().numpy(),ttE.mean().detach().numpy())
            lC2p.append(C2p)
            lchi_m.append(chi_m)
            ## HMC update but also V cycle
            #phi_o,sgc_f,mgf_c,hmc_f,sgc,hmc_c,m1,m2,Nskip=1,mode="normal"
            phi,accept=V_cycle(phi,sg,mgf,hmc_f,sgc,hmc_c,1,1,Nskip=1,mode=mode)
            #print("step:",k,flush=True)

    return lC2p, lchi_m, E, av_phi, phi


def epsilon_test_coarse(sg,mgf,sgc,phi):
    print("Initializing epsilon test")
    eps1=[]
    deltaH=[]
    mgf.eval()

    sgc.generate_cfg_levels(phi)
    phic,logdet=mgf.backward(phi)
    Pc=sgc.refreshP()

    K=sgc.kinetic(Pc)
    V=sgc.action(phic)
    Hinit=K+V
    print("initial energy",Hinit)

    for rk in np.geomspace(20,120,20):
        k=int(rk)
        dt=1.0/k
        print("we are in the",k,"step")
        mn2c = integ.minnorm2(sgc.force,sgc.evolveQ,k,1.0)
        Pc2,phic2= mn2c.integrate(Pc,phic)
        
        
        Hfinal=sgc.kinetic(Pc2)+sgc.action(phic2)
        DH=tr.abs(Hfinal-Hinit)
        eps1.append(dt)
        deltaH.append(DH)


    DelH=tr.stack(deltaH).to("cpu")
    e=tr.tensor(eps1).to("cpu")

    return e,DelH


def train_multigrid_DeltaS(mgf,phi,sg,steps=501):
    # Optimizador
    with tr.no_grad():
        mgf.generate_levels(phi)
    # Aumentamos weight_decay (L2) a 1e-3 para penalizar pesos grandes y evitar explosión
    optimizer = tr.optim.Adam([p for p in mgf.parameters() if p.requires_grad], lr=1e-5,weight_decay=1e-2)
    scheduler = tr.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=1, pct_start=0.1, anneal_strategy='cos', div_factor=25, max_lr=1e-3, epochs=steps)

    loss_history = []

    # Configuraciones
    super_batch_size = 4 # Si lo usas para acumular gradientes
    mgf.train() # IMPORTANTE: Activa dropouts o batchnorms si los hubiera

    print("Training multigrid model...")

    for t in range(steps):   
        #optimizer.zero_grad() 
        
        batch_loss_acum = 0
        batch_energy_acum = 0
        batch_logdet_acum = 0
        
        for b in range(super_batch_size):

            # 2. Forward eficiente
            phi_c,log_det_c = mgf.backward(phi)
            
            xi = tr.randn_like(phi_c) 
            
            # PASO B: Perturbar z
            phi_c1 = phi_c + 0.1 * xi
            
            phi1, log_det_f1 = mgf.forward(phi_c1)
            phi2, log_det_f2 = mgf.forward(phi_c)

            action_val = sg.action(phi1)
            log_det_J = log_det_f2

            effective_energy_orig = sg.action(phi2)-log_det_f2
            effective_energy_probe = sg.action(phi1)-log_det_f1
            
            loss_probe = ((effective_energy_probe - effective_energy_orig)**2).mean()
            
            # Total Loss: SOLO Probe Loss (sin KL)
            loss = loss_probe
            
            # Normalizamos por el super_batch
            loss = loss / super_batch_size 
            
            tr.nn.utils.clip_grad_norm_(mgf.parameters(), max_norm=10000.0)
            loss.backward() 
            
            # ACUMULAR ESTADÍSTICAS PARA MONITOREO (sin re-calcular nada)
            batch_loss_acum += loss.item() * super_batch_size # loss ya estaba dividida
            batch_energy_acum += action_val.mean().item()
            batch_logdet_acum += log_det_J.mean().item()

        # Optimizador y Scheduler
        optimizer.step()
        
        # Calcular promedios del super-batch
        epoch_loss = batch_loss_acum / super_batch_size
        epoch_energy = batch_energy_acum / super_batch_size
        epoch_logdet = batch_logdet_acum / super_batch_size
        
        loss_history.append(epoch_loss)
        
        # IMPORTANTE: Scheduler step debe ir DESPUÉS de optimizer.step()
        # Para OneCycleLR, step() se llama en cada iteración (no en cada época si steps_per_epoch=len(loader))
        # Aquí cada 't' cuenta como un paso.
        scheduler.step()
        
        if t % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            # Mostramos promedios reales del batch que acabamos de procesar
            print(f"Iter {t:04d} | Loss: {epoch_loss:.2f} | lr: {current_lr:.1e} | "
                    f"E: {epoch_energy:.2f} | LDet: {epoch_logdet:.2f}",flush=True)
    return loss_history, mgf


def train_multigrid_KL(mgf,phi,sgc,steps=1001,super_batch_size=10):
    # Optimizador
    optimizer = tr.optim.Adam([p for p in mgf.parameters() if p.requires_grad], lr=1e-3,weight_decay=1e-4)
    scheduler = tr.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=1,pct_start=0.1,anneal_strategy='cos',div_factor=25,max_lr=1e-2,epochs=steps)
    loss_history = []
    batch_size=phi.shape[0]
    # Configuraciones
    # Si lo usas para acumular gradientes
    mgf.train() # IMPORTANTE: Activa dropouts o batchnorms si los hubiera
    with tr.no_grad():
        mgf.generate_levels(phi)


    print("Iniciando entrenamiento...")

    for t in range(steps):   
        optimizer.zero_grad() # Limpiamos gradientes ANTES del super-batch
        
        # Beta Annealing: De 5.0 (fuerza la energía) a 1.0 (física real) en 800 épocas
        #progreso = min(1.0, t / 800.0)
        #beta = 5.0 * (1 - progreso) + 1.0 * progreso
        
        batch_loss_acum = 0
        
        # Opción A: Acumulación de Gradientes (Ahorra memoria RAM)
        for b in range(super_batch_size):
            # 1. Muestrear z
            phi_c=mgf.backward(phi)### I have to change the prior sample function for coarse level!!!!!
            z = mgf.prior_sample_coarse(batch_size)
            # 2. Forward eficiente (Obtenemos x y el LogDet en UNA pasada)
            # NO llamamos a mgf.log_prob(x) después.
            x, log_det_J = mgf.forward(z) 
            
            # 3. Calcular log q(x)
            # Obtenemos la probabilidad cruda
            z_flat = z.flatten(start_dim=1)
            raw_lp = mgf.prior.log_prob(z_flat)

            # Verificamos si necesitamos sumar dimensiones
            if raw_lp.dim() > 1:
                log_p_z = raw_lp.sum(dim=1) # Caso Normal(0,1) estándar
            else:
                log_p_z = raw_lp            # Caso MultivariateNormal (ya viene sumado)
            
            # Ahora sí calculamos log q(x)
            log_q_x = log_p_z - log_det_J
            
            # 4. Calcular Acción (Física)
            # S(x) es equivalente a -log p_target(x)
            action_val = sgc.action(z)
            
            # 5. KL Divergence con Beta Annealing
            loss = (log_q_x + action_val).mean()
            
            # Normalizamos por el super_batch para que el learning rate sea consistente
            loss = loss / super_batch_size 
            
            tr.nn.utils.clip_grad_norm_(mgf.parameters(), max_norm=100.0)
            # 6. Backward inmediato (libera el grafo computacional paso a paso)
            loss.backward() 
            
            batch_loss_acum += loss.item()

        # Paso del optimizador (una vez procesados los 10 micro-batches)
        optimizer.step()
        scheduler.step()
        
        loss_history.append(batch_loss_acum)
        with tr.no_grad():
            # 1. Energía (Acción Física): Queremos que baje
            mean_energy = action_val.mean().item()
            
            # 2. Entropía (Variedad): Queremos que suba (o no baje demasiado)
            # H[q] = - E[log q(x)]
            mean_entropy = -log_q_x.mean().item()
            
            # 3. Determinante Promedio (Diagnóstico de expansión)
            # Nos dice cuánto está "estirando" el espacio el flujo
            mean_logdet = log_det_J.mean().item()
        if t % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Iter {t:04d} | Loss: {loss.item():.2f} | lr: {current_lr:.1e} | "
                    f"E: {mean_energy:.2f} | H: {mean_entropy:.2f} | LDet: {mean_logdet:.2f}")
                    
    return loss_history, mgf


def FlowBijector_DeepSet(Nlayers=3,width=256,ttp=tr.float64,device="cpu"):
    mm = np.array([1,0,0,1])
    tV = mm.size
    nets = lambda: ZeroNet().to(device).to(ttp)
    def nett():
        seq = nn.Sequential(nn.Linear(tV,width),nn.Linear(width,tV),nn.Tanh(),Rescale(scale=0.1,trainable=False))
        seq.apply(init_xavier)
        return seq.to(device).to(ttp)
    masks = tr.from_numpy(np.array([mm, 1-mm] * Nlayers)).to(device).to(ttp)
    normal = distributions.Normal(tr.zeros(tV,dtype=ttp,device=device),tr.ones(tV,dtype=ttp,device=device))
    prior= distributions.Independent(normal, 1)
    return RealNVP_1(nets, nett, masks, prior, data_dims=(1,2))

def FlowBijector_rnvp(Nlayers=3,width=256,ttp=tr.float64,device="cpu"):
    mm = np.array([1,0,0,1])
    tV = mm.size
    #nets = lambda: ZeroNet().to(device).to(ttp)
    def nets():
        seq = nn.Sequential(nn.Linear(tV,width),nn.Linear(width,tV),Cos_nn(),Rescale(scale=0.1,trainable=False))
        seq.apply(init_xavier)
        return seq.to(device).to(ttp)
    def nett():
        seq = nn.Sequential(nn.Linear(tV,width),nn.Linear(width,tV),nn.Tanh(),Rescale(scale=0.1,trainable=False))
        seq.apply(init_xavier)
        return seq.to(device).to(ttp)
    masks = tr.from_numpy(np.array([mm, 1-mm] * Nlayers)).to(device).to(ttp)
    normal = distributions.Normal(tr.zeros(tV,dtype=ttp,device=device),tr.ones(tV,dtype=ttp,device=device))
    prior= distributions.Independent(normal, 1)
    return RealNVP_1(nets, nett, masks, prior, data_dims=(1,2))

class phi4_c1:
    def action(self, phi_c):
        # Initialize logdet_total in the correct device
        #logdet_total = tr.zeros(phi_c.shape[0], dtype=self.dtype, device=self.device)
        phi_c,logdet_total=self.mgf.forward(phi_c)

        return self.sg.action(phi_c) - logdet_total
        #if I dont add the .sum() I got a grad for the batch system, it seems to me that we include that in the force property the batch is summed?

    def force(self, phi_c, create_graph=False, retain_graph=False):
        with tr.enable_grad():
            x_tensor = phi_c.clone().requires_grad_()

            S = self.action(x_tensor)
            # print("Action S:", S.item())
            grad = tr.autograd.grad(S.sum(), x_tensor, create_graph=create_graph, retain_graph=retain_graph)[0]
            if tr.isnan(grad).any():
                # If NaN, return zeros to avoid crash, let HMC reject via energy check (if possible) or fail gracefully later
                print("WARNING: NaN gradient detected in force. Returning zeros.")
                return tr.zeros_like(grad)
            

            #release memory on gpu
            if create_graph:
                return -grad
            return -grad.detach()
    
    def refreshP(self):
        P = tr.normal(0.0,1.0,self.mgf.coarse_Pshape).to(self.device).to(self.dtype)#only difference with fine level
        return P

    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    
    def kinetic(self,P):
        return tr.einsum('bxy,bxy->b',P,P)/2.0

    def generate_cfg_levels(self,phi11):#run every time we need to construct deeper or superficial levels
        self.mgf.generate_levels(phi11)

    def __init__(self,sgg,mgf,device="cpu",dtype=tr.float64,mode="rnvp"):
        self.sg = sgg #theory? in the finest level
        self.mgf = mgf #neural net
        self.rg = mgf.rg #projector to coarse level
        self.mode = mode
        print("multigrid is done by123: ",self.mode)
        self.device = device
        self.dtype = dtype



class MGflow1(nn.Module):
    def __init__(self, size, bijector, rg, prior, Nconvs=2, depth=None,mode="rnvp"):
        super(MGflow1, self).__init__()

        self.prior = prior
        self.rg = rg
        self.size = size
        self.mode= mode
        minSize = min(size)
        print("Initializing MGflow module with size: ", minSize)
        if depth == None:
            self.depth = int(np.log(minSize) / np.log(2))
        else:
            self.depth = depth
        print("Using depth: ", self.depth)
        print("Using rg type: ", rg.type)
        sizes = []
        for k in range(self.depth):
            sizes.append([int(size[i] / (2**k)) for i in range(len(size))])
            print("(depth, size): ", k, sizes[-1])

        # the module list are ordered from fine to coarse
        self.cflow = tr.nn.ModuleList([ConvFlowLayer(sizes[k], bijector, Nconvs) for k in range(self.depth)])

    def generate_levels(self,phi_fine):
        pi = []
        phi_original = phi_fine.clone()

        for k in range(self.depth):
            #transform with realnvp
            #print("level",k)
            if self.mode=="rnvp":
                phi_fine,logdet = self.cflow[k].backward(phi_fine)
            phi_fine, f = self.rg.coarsen(phi_fine)
            pi.append(f)
        self.coarse_Pshape=phi_fine.shape
        #now do the opposite the forward map
        for k in reversed(range(self.depth)):
            #print("level",k)
            phi_fine = self.rg.refine(phi_fine,pi[k])
            if self.mode=="rnvp":
                phi_fine, logdet = self.cflow[k].forward(phi_fine)
        #comparison of phi original and phi fine
        diff=phi_original-phi_fine
        self.pi=pi
        #print("lenght of list of flows",len(self.cflow),"lenght of pi",len(pi))
        return phi_fine,diff
            
        
    # coarse to fine
    def forward(self, z):
        
        log_det_total = z.new_zeros(z.shape[0])#check size,
        
        for k in reversed(range(self.depth)):
            #print("level forward",k)
            z = self.rg.refine(z,self.pi[k])
            if self.mode=="rnvp":
                z, logdet = self.cflow[k].forward(z)
                log_det_total+=logdet
        
        return z, log_det_total  # <--- RETORNAMOS AMBOS

    # fine to coarse
    def backward(self, x):
        log_det_J = x.new_zeros(x.shape[0])

        for k in range(self.depth):
            #print("levels back",k)
            # ConvFlowLayer.backward ya devuelve (z, J_negativo)
            if self.mode=="rnvp":    
                x, J = self.cflow[k].backward(x) 
                log_det_J += J
            x, ff = self.rg.coarsen(x)
            
        return x, log_det_J

    def log_prob(self,x):
        z, logp = self.backward(x)
        return self.prior.log_prob(z.flatten(start_dim=1)) + logp

    def sample(self, batchSize): 
        #z = self.prior.sample((batchSize, 1)).reshape(batchSize,self.size[0],self.size[1])
        z = self.prior_sample(batchSize)
        x = self.forward(z)
        return x

    # generate a sample from the prior
    def prior_sample(self,batch_size):
        return self.prior.sample((batch_size,1)).reshape(batch_size,self.size[0],self.size[1])

    def prior_sample_coarse(self,batch_size):
        return self.prior.sample((batch_size,1)).reshape(batch_size,self.coarse_Pshape[1],self.coarse_Pshape[2])


    
class Rescale(nn.Module):
    def __init__(self, scale=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(tr.tensor(scale))
        self.scale.requires_grad = trainable
    def forward(self, x):
        return x * self.scale

class ZeroNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        # Returns zeros matching the input shape
        return x.new_zeros(x.shape)

def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Cos_nn(nn.Module):
    def forward(self, x):
        return tr.cos(x)

class RGlayer1(nn.Module):
    def __init__(self, transformation_type="select", batch_size=1, dtype=tr.float64, device="cpu"):
        super(RGlayer1, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

        if transformation_type == "select":
            mask_c = [[1.0, 0.0], [0.0, 0.0]]
            mask_r = [[1.0, 1.0], [1.0, 1.0]]
        elif transformation_type == "average":
            mask_c = [[0.25, 0.25], [0.25, 0.25]]
            mask_r = [[1.0, 1.0], [1.0, 1.0]]
        else:
            print("Unknown RG blocking transformation. Using default.")
            mask_c = [[1.0, 0.0], [0.0, 0.0]]
            mask_r = [[1.0, 0.0], [0.0, 0.0]]

        self.type = transformation_type

        self.restrict = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=2, bias=False)
        self.restrict.weight = nn.Parameter(tr.tensor([[mask_c]], dtype=self.dtype, device=self.device), requires_grad=False)
        self.prolong = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=2, bias=False)
        self.prolong.weight = nn.Parameter(tr.tensor([[mask_r]], dtype=self.dtype, device=self.device), requires_grad=False)

    def coarsen(self, f):
        #print("coarsen device: ", f.device, f.dtype)
        ff = f.view(f.shape[0], 1, f.shape[1], f.shape[2])
        c = self.restrict(ff)
        #no grad pi
        r = ff - self.prolong(c)
        #r.
        return (c.squeeze(1), r.squeeze(1)) if self.batch_size == 1 else (c.squeeze(), r.squeeze())

    def refine(self, c, r):
        cc = c.view(c.shape[0], 1, c.shape[1], c.shape[2])
        rr = r.view(r.shape[0], 1, r.shape[1], r.shape[2])
        return (self.prolong(cc) + rr).squeeze(1) if self.batch_size == 1 else (self.prolong(cc) + rr).squeeze()


class RealNVP_1(nn.Module):
    def __init__(self, nets, nett, mask, prior,data_dims=(1)):
        super(RealNVP_1, self).__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = tr.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = tr.nn.ModuleList([nets() for _ in range(len(mask))])
        self.data_dims=data_dims
        
    # this is the forward start from noise target
    def g(self, z):
        log_det_J_for, x = z.new_zeros(z.shape[0]), z

        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * tr.exp(s) + t)
            log_det_J_for += s.sum(dim=self.data_dims)
        return x, log_det_J_for
    
    # this is backward from target to noise
    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * tr.exp(-s) + z_
            log_det_J -= s.sum(dim=self.data_dims)
        return z, log_det_J

    def forward(self,z):
        return self.g(z)
    
    def backward(self,x):
        return self.f(x)
    
    def log_probf(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp #+ self.C
    
    def log_probg(self,z):
        x, logp = self.g(z)
        return self.prior.log_prob(z) - logp
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        #logp = self.prior.log_prob(z)
        x = self.g(z)
        return x
    
class ConvFlowLayer(nn.Module):
    def __init__(self,size,bijector,Nsteps=1):
        super(ConvFlowLayer, self).__init__()
        self.Nsteps=Nsteps
        self.bj = tr.nn.ModuleList([bijector() for _ in range(2*Nsteps)])
        # for now the kernel is kept 2x2 and stride is 2 so it only works on lattices with
        # a power of 2 dimensions
        fold_params = dict(kernel_size=(2,2), dilation=1, padding=0, stride=(2,2))
        
        self.unfold = nn.Unfold(**fold_params)
        self.fold = nn.Fold(size,**fold_params)

    def forward(self, z):
        log_det_J = z.new_zeros(z.shape[0]) # Acumulador
        
        # Unfold inicial
        uf = self.unfold(z.view(z.shape[0],1,z.shape[1],z.shape[2])).transpose(2,1)
        
        for k in range(self.Nsteps):

            input_vec = uf 
            
            out_vec, ld_local = self.bj[2*k].g(input_vec) 
            
            if ld_local.dim() == 1:
                log_det_J += ld_local
            else:
                log_det_J += ld_local.sum(dim=1)
            
            shifted = self.unfold(tr.roll(self.fold(out_vec.transpose(2,1)),dims=(2,3),shifts=(-1,-1))).transpose(2,1)
            
            # --- Bijector Odd (2*k+1) ---
            out_vec_2, ld_local_2 = self.bj[2*k+1].g(shifted)
            if ld_local.dim()==1:
                log_det_J += ld_local_2
            else:
                log_det_J += ld_local_2.sum(dim=1)
            
            uf = self.unfold(tr.roll(self.fold(out_vec_2.transpose(2,1)),dims=(2,3),shifts=( 1, 1))).transpose(2,1)
            
        x = self.fold(uf.transpose(2,1)).squeeze(1)
        return x, log_det_J

    # fields to noise
    def backward(self,x):
        log_det_J=x.new_zeros(x.shape[0])
        #HERE IS WHERE WE HAVE FUN!
        # add the extra dimension for unfolding
        z = x.view(x.shape[0],1,x.shape[1],x.shape[2])
        for k in reversed(range(self.Nsteps)):
            #shift  and unfold
            sz = self.unfold(tr.roll(z,dims=(2,3),shifts=(-1,-1))).transpose(2,1)
            #unfold and flow
            ff,J = self.bj[2*k+1].f(sz)
            log_det_J += J
            #fold shift unfold
            sz = self.unfold(tr.roll(self.fold(ff.transpose(2,1)),dims=(2,3),shifts=(1,1))).transpose(2,1)
            # flow backwards
            ff,J = self.bj[2*k].f(sz)
            log_det_J += J 
            #fold
            z = self.fold(ff.transpose(2,1))
            
        z = z.squeeze(1)#change back to Batch,2d
        return z,log_det_J

    def log_prob_back(self,x):
        z, logp = self.backward(x)
        #return self.prior.log_prob(z) + logp
        return logp # we do not have a prior distribution for this layer 



class ConvFlowLayer_1(nn.Module):
    def __init__(self,size,bijector,Nsteps=1):
        super(ConvFlowLayer_2, self).__init__()
        self.Nsteps=Nsteps
        self.bj = tr.nn.ModuleList([bijector() for _ in range(2*Nsteps)])
        # for now the kernel is kept 2x2 and stride is 2 so it only works on lattices with
        # a power of 2 dimensions
        fold_params = dict(kernel_size=(2,2), dilation=1, padding=0, stride=(2,2))
        
        self.unfold = nn.Unfold(**fold_params)
        self.fold = nn.Fold(size,**fold_params)
        
    # noise to fields
    def forward(self, z):
        log_det_J = z.new_zeros(z.shape[0]) # Acumulador
        
        # Unfold inicial
        uf = self.unfold(z.view(z.shape[0],1,z.shape[1],z.shape[2])).transpose(2,1)
        
        for k in range(self.Nsteps):
            # Paso 1: Flow (sin shift)
            
            # --- Bijector Par (2*k) ---
            input_vec = uf 
            out_vec, ld_local = self.bj[2*k].g(input_vec) 
            
            if ld_local.dim() == 1:
                log_det_J += ld_local
            else:
                log_det_J += ld_local.sum(dim=1)
            
            # Sin shift
            shifted = out_vec 
            
            # --- Bijector Impar (2*k+1) ---
            out_vec_2, ld_local_2 = self.bj[2*k+1].g(shifted)
            if ld_local.dim()==1:
                log_det_J += ld_local_2
            else:
                log_det_J += ld_local_2.sum(dim=1)
            
            # Preparar para siguiente ciclo
            uf = out_vec_2
            
        x = self.fold(uf.transpose(2,1)).squeeze(1)
        return x, log_det_J

    # fields to noise
    def backward(self,x):
        log_det_J=x.new_zeros(x.shape[0])
        z = x.view(x.shape[0],1,x.shape[1],x.shape[2])
        
        # Unfold inicial
        sz = self.unfold(z).transpose(2,1)
        
        for k in reversed(range(self.Nsteps)):
            # Flow backwards impar
            ff,J = self.bj[2*k+1].f(sz)
            log_det_J += J
            
            sz = ff
            
            # Flow backwards par
            ff,J = self.bj[2*k].f(sz)
            log_det_J += J 
            
            sz = ff
            
        z = self.fold(sz.transpose(2,1))
        z = z.squeeze(1)
        return z,log_det_J

    def log_prob(self,x):
        z, logp = self.backward(x)
        return logp 




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

