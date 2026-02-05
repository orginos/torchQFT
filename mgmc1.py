import numpy as np
import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter
import phi4_mg as m


#!/usr/local/bin/python3
import time
import numpy as np
import torch as tr
import phi4 as s
import integrators as i
import update as u

import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import sys

import time
from stacked_model import *
import Gamma_error as gm

### latex plots

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

### class definitions

import torch as tr
import torch.nn as nn

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
        r = ff - self.prolong(c)
        return (c.squeeze(1), r.squeeze(1)) if self.batch_size == 1 else (c.squeeze(), r.squeeze())

    def refine(self, c, r):
        cc = c.view(c.shape[0], 1, c.shape[1], c.shape[2])
        rr = r.view(r.shape[0], 1, r.shape[1], r.shape[2])
        return (self.prolong(cc) + rr).squeeze(1) if self.batch_size == 1 else (self.prolong(cc) + rr).squeeze()
    

class phi4_c1:
    def action(self, phi_c):
        rphis = []
        rphis.append(phi_c)
        iii = 0
        # Inicializar logdet en el dispositivo correcto
        logdet_total = tr.zeros(phi_c.shape[0], dtype=self.dtype, device=self.device)
        
        for pi in reversed(self.pics):
            # 1. Refinamiento lineal (RG inverso)
            rphi = self.rg.refine(rphis[iii], pi)
            
            if self.mode == "rnvp":
                # 2. Corrección con el Flow (Forward)
                # Ahora obtenemos el determinante AQUÍ, sin llamar a backward
                rphi, logdet_layer = self.mgf.cflow[self.level-1-iii].forward(rphi)
                
                # REGLA DE SIGNOS PARA LA ACCIÓN EFECTIVA:
                # S_eff(phi_c) = S(phi_f) - log |det J|
                # Como 'forward' va de coarse -> fine (generativo), el Jacobiano
                # mide la expansión de volumen. Debemos RESTAR ese volumen.
                # (O sumar si definiste log_det_J con signo negativo, revisa esto abajo).
                
                # Si log_det_layer es positivo (expansión), restamos al log-likelihood
                # que equivale a SUMAR a la Acción (porque Acción ~ -log_prob).
                # S = -log_prob => log_prob_new = log_prob_old - log_det
                # S_new = - (log_prob_old - log_det) = S_old + log_det
                
                # En tu código anterior sumabas log_prob, lo cual es:
                # log_prob = prior + log_det_backward
                # log_det_backward = - log_det_forward
                # Así que restabas el log_det_forward.
                
                # Si usamos la Acción directamente:
                logdet_total -= logdet_layer # Checar signo según convención S
                
            rphis.append(rphi)
            iii += 1
            
        rphi_fine = rphis[-1]
        
        # Retornamos la acción total
        # Action física + Corrección jacobiana
        return self.sg.action(rphi_fine) - logdet_total
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
            
            # Gradient Clipping for Stability
            max_norm = 10000.0
            norm = grad.norm()
            if norm > max_norm:
                # print(f"Clipping gradient: norm {norm.item()} -> {max_norm}")
                grad = grad * (max_norm / norm)

            #release memory on gpu
            if create_graph:
                return -grad
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
        # print("coarsening level init field shape ",phicopy.shape)
        for step in range(self.level):
            if self.mode=="rnvp":
                
                phicopy,logdet = self.mgf.cflow[step].backward(phicopy)
                #create an extra dimension for batch
                #if self.sg.Bs==1:
                #    phicopy=phicopy.unsqueeze(0)
                
                #print("device phicopy ",phicopy.device," dtype ",phicopy.dtype)
            #print("coarsening level ",_," field shape ",phicopy.shape)
            #print("coarsening level ",step," field shape ",phicopy.shape)
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
                rphi = self.mgf.cflow[self.level-1-sss].forward(rphi)
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

class MGflow1(nn.Module):
    def __init__(self, size, bijector, rg, prior, Nconvs=2, depth=None):
        super(MGflow1, self).__init__()
        # ... (Tu __init__ se queda IGUAL) ...
        self.prior = prior
        self.rg = rg
        self.size = size
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

    # noise to fields (MODIFICADO para devolver log_det)
    def forward(self, z):
        x = z
        # Inicializamos el acumulador del determinante
        # Nota: El signo depende de tu convención. 
        # Si forward es z->x (expansión), el log_det suele ser positivo.
        log_det_total = x.new_zeros(x.shape[0]) 
        
        fines = []
        
        # 1. Bajamos a la escala más gruesa (Coarsening)
        # Nota: RG linear tiene Jacobiano constante (generalmente se ignora o cancela si es unitario)
        for k in range(self.depth - 1):
            c, f = self.rg.coarsen(x)
            x = c
            fines.append(f)
            
        # 2. Subimos y aplicamos flujos (Refining + Flow)
        # x está en la escala más gruesa
        for k in range(self.depth - 1, 0, -1):
            # APLICAMOS FLOW: x_new = g(x_old)
            # IMPORTANTE: ConvFlowLayer.forward debe devolver (fx, J)
            fx, J = self.cflow[k].forward(x) 
            
            # Acumulamos el determinante
            log_det_total += J 
            
            # Refinamos mezclando con los residuos guardados
            x = self.rg.refine(fx, fines[k - 1])
            
        # Última capa de flujo en la escala fina
        fx, J = self.cflow[0].forward(x)
        log_det_total += J
        
        return fx, log_det_total  # <--- RETORNAMOS AMBOS

    # fields to noise (Se mantiene casi igual, solo verifica ConvFlowLayer)
    def backward(self, x):
        log_det_J = x.new_zeros(x.shape[0])
        fines = []
        
        for k in range(self.depth - 1):
            # ConvFlowLayer.backward ya devuelve (z, J_negativo)
            fx, J = self.cflow[k].backward(x) 
            log_det_J += J
            
            cx, ff = self.rg.coarsen(fx)
            fines.append(ff)
            x = cx
            
        fx, J = self.cflow[self.depth - 1].backward(x)
        log_det_J += J
        
        for k in range(self.depth - 2, -1, -1):
            z = self.rg.refine(fx, fines[k])
            fx = z
            
        return fx, log_det_J

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
        # this could be an arbitrary class with a sample method
        # self.prior=prior
        # I see no need for sampling from a prior for this layer
        
    # noise to fields
    #def forward(self,z):
    #    uf = self.unfold(z.view(z.shape[0],1,z.shape[1],z.shape[2])).transpose(2,1)
    #    for k in range(self.Nsteps):
    #        sf = self.unfold(tr.roll(self.fold(self.bj[2*k  ].g(uf).transpose(2,1)),dims=(2,3),shifts=(-1,-1))).transpose(2,1)
    #        uf = self.unfold(tr.roll(self.fold(self.bj[2*k+1].g(sf).transpose(2,1)),dims=(2,3),shifts=( 1, 1))).transpose(2,1)
    #    x = self.fold(uf.transpose(2,1)).squeeze(1)
    #    return x
    # Modificamos forward para devolver log_det_J también
    def forward(self, z):
        log_det_J = z.new_zeros(z.shape[0]) # Acumulador
        
        # Unfold inicial
        uf = self.unfold(z.view(z.shape[0],1,z.shape[1],z.shape[2])).transpose(2,1)
        
        for k in range(self.Nsteps):
            # Paso 1: Shift y Flow
            # Nota: Al llamar a bj[].g(uf), necesitamos que .g devuelva (out, log_det)
            # Asumo que estás usando el RealNVP_1 modificado que te di antes.
            
            # --- Bijector Par (2*k) ---
            # OJO: Aquí hay un detalle técnico. 
            # Si tu RealNVP actúa sobre los parches (dimensión 4), el determinante que devuelve
            # es por cada parche. Hay que sumar sobre todos los parches.
            
            # Preparar input
            input_vec = uf # (Batch, N_patches, 4)
            
            # Aplicar g (ahora devuelve x y log_det)
            out_vec, ld_local = self.bj[2*k].g(input_vec) 
            
            # Sumar log_det de todos los parches
            if ld_local.dim() == 1:
                log_det_J += ld_local
            else:
                log_det_J += ld_local.sum(dim=1)
            
            # Fold/Shift lógica compleja...
            shifted = self.unfold(tr.roll(self.fold(out_vec.transpose(2,1)),dims=(2,3),shifts=(-1,-1))).transpose(2,1)
            
            # --- Bijector Impar (2*k+1) ---
            out_vec_2, ld_local_2 = self.bj[2*k+1].g(shifted)
            if ld_local.dim()==1:
                log_det_J += ld_local_2
            else:
                log_det_J += ld_local_2.sum(dim=1)
            
            # Preparar para siguiente ciclo (shift back)
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

# no need for sampling for this layer
#    def sample(self, batchSize): 
#        z = self.prior.sample((batchSize, 1))
#        x = self.forward(z)
#        return x

#prepares RealNVP for the Convolutional Flow Layer
def FlowBijector(Nlayers=3,width=256):
    mm = np.array([1,0,0,1])
    tV = mm.size
    nets = lambda: nn.Sequential(nn.Linear(tV, width), nn.LeakyReLU(), nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, tV), nn.Tanh())
    nett = lambda: nn.Sequential(nn.Linear(tV, width), nn.LeakyReLU(), nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, tV))

    # the number of masks determines layers
    #Nlayers = 3
    masks = tr.from_numpy(np.array([mm, 1-mm] * Nlayers).astype(np.float32))
    normal = distributions.Normal(tr.zeros(tV),tr.ones(tV))
    prior= distributions.Independent(normal, 1)
    return  RealNVP(nets, nett, masks, prior, data_dims=(1,2))

#prepares RealNVP for the Convolutional Flow Layer
def FlowBijectorParity(Nlayers=3,width=256):
    mm = np.array([1,0,0,1])
    tV = mm.size
    nets = lambda: ParityNet(nn.Sequential(nn.Linear(tV, width), nn.LeakyReLU(), nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, tV), nn.Tanh()),Parity=+1)
    nett = lambda: ParityNet(nn.Sequential(nn.Linear(tV, width), nn.LeakyReLU(), nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, tV)),Parity=-1)

    # the number of masks determines layers
    #Nlayers = 3
    masks = tr.from_numpy(np.array([mm, 1-mm] * Nlayers).astype(np.float32))
    normal = distributions.Normal(tr.zeros(tV),tr.ones(tV))
    prior= distributions.Independent(normal, 1)
    return  RealNVP(nets, nett, masks, prior, data_dims=(1,2))

#prepares RealNVP for the Convolutional Flow Layer
def FlowBijector_3layers(Nlayers=3,width=256):
    mm = np.array([1,0,0,1])
    tV = mm.size
    nets = lambda: nn.Sequential(nn.Linear(tV, width), nn.LeakyReLU(), nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, tV), nn.Tanh())
    nett = lambda: nn.Sequential(nn.Linear(tV, width), nn.LeakyReLU(), nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, tV))

    # the number of masks determines layers
    #Nlayers = 3
    masks = tr.from_numpy(np.array([mm, 1-mm] * Nlayers).astype(np.float32))
    normal = distributions.Normal(tr.zeros(tV),tr.ones(tV))
    prior= distributions.Independent(normal, 1)
    return  RealNVP(nets, nett, masks, prior, data_dims=(1,2))


#utility class that allows to fix bijectors between layers
class BijectorFactory():
     def __init__(self, bj_list):
          self.counter=-1
          self.bj_list = bj_list
     def bij(self):
         self.counter+=1 
         bj_index = self.counter % len(self.bj_list)
         print("Adding Bijector ",bj_index)
         return self.bj_list[bj_index]

# to be used to make a realNVP with specific parity properties
class ParityNet(nn.Module):
    def __init__(self, net,Parity=+1):
        super(ParityNet,self).__init__()
        self.Parity = Parity
        self.net = net
    def forward(self,x):
        return 0.5*(self.net(x) + self.Parity*self.net(-x))

class ZeroNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        # Returns zeros matching the input shape
        return x.new_zeros(x.shape)

def FlowBijectorParity1(Nlayers=3,width=256,ttp=tr.float64,device="cpu"):
    mm = np.array([1,0,0,1])
    tV = mm.size
    # Use ZeroNet for scaling (Additive Coupling) - LogDet is always 0
    nets = lambda: ParityNet(nn.Sequential(nn.Linear(tV, width), nn.Tanh(), nn.Linear(width, tV), nn.Tanh(),Rescale(scale=0.001)),Parity=+1).to(device).to(ttp)#ZeroNet().to(device).to(ttp)
    # Keep Tanh and Rescale(0.001) for translation
    nett = lambda: ParityNet(nn.Sequential(nn.Linear(tV, width),nn.Tanh(), nn.Linear(width, tV),Rescale(scale=0.001)),Parity=-1).to(device).to(ttp)
    masks = tr.from_numpy(np.array([mm, 1-mm] * Nlayers)).to(device).to(ttp)
    normal = distributions.Normal(tr.zeros(tV,dtype=ttp,device=device),tr.ones(tV,dtype=ttp,device=device))
    prior= distributions.Independent(normal, 1)
    return  RealNVP_1(nets, nett, masks, prior, data_dims=(1,2))


#prepares RealNVP for the Convolutional Flow Layer
class Rescale(nn.Module):
    def __init__(self, scale=0.001, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(tr.tensor(scale))
        self.scale.requires_grad = trainable
    def forward(self, x):
        return x * self.scale


# Identity Bijector for Debugging
class IdentityFlow(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def g(self, z):
        # Forward: z -> x. Returns x, log_det (B,)
        return z, z.new_zeros(z.shape[0])

    def f(self, x):
        # Backward: x -> z. Returns z, log_det (B,)
        return x, x.new_zeros(x.shape[0]) 

def FlowIdentity():
    return IdentityFlow()


class ZeroNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        # Returns zeros matching the input shape
        return x.new_zeros(x.shape)

def FlowBijectorParity1(Nlayers=3,width=256,ttp=tr.float64,device="cpu"):
    mm = np.array([1,0,0,1])
    tV = mm.size
    # Use ZeroNet for scaling (Additive Coupling) - LogDet is always 0
    nets = lambda: ZeroNet()
    # Keep Tanh and Rescale(0.001) for translation
    nett = lambda: ParityNet(nn.Sequential(nn.Linear(tV, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, tV),Rescale(scale=0.001)),Parity=-1).to(device).to(ttp)
    masks = tr.from_numpy(np.array([mm, 1-mm] * Nlayers)).to(device).to(ttp)
    normal = distributions.Normal(tr.zeros(tV,dtype=ttp,device=device),tr.ones(tV,dtype=ttp,device=device))
    prior= distributions.Independent(normal, 1)
    return  RealNVP_1(nets, nett, masks, prior, data_dims=(1,2))

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

#still need to change the code of phi_coarse try Vcycle instead
def V_cycle(phi_o,sgc_c,mgf_c,hmc_f,m1,m2,Nskip=1,mode="normal"):#mode="rnvp"
    phi_o= hmc_f.evolve(phi_o,m1)
    sgc = phi4_c1(sgc_c,mgf_c,device=sgc_c.device,dtype=sgc_c.dtype,mode=mode)
    mn2c = i.minnorm2(sgc.force,sgc.evolveQ,200,1.0)
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
    
    # Run sampling in no_grad mode to save memory. 
    # Force calculation in HMC will still work because it uses local requires_grad_().
    with tr.no_grad():
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



import integrators as i


def train_force_based(sg, mgf, epochs=100, lr=1e-4, batch_size=64):
    """
    Train the flow using Force Matching (L_force) and Sensitivity (L_sens) losses.
    L_force: Minimizes the squared norm of the coarse force (stiffness).
    L_sens: Minimizes the variance of the coarse force with respect to fine fluctuations.
    """
    import torch.optim as optim
    
    # Ensure flow is in training mode
    mgf.train()
    
    # Optimizer for the flow parameters
    optimizer = optim.Adam([p for p in mgf.parameters() if p.requires_grad], lr=lr)
    
    print(f"Starting Force-Based Training for {epochs} epochs...")
    
    # Create a phi4_c1 wrapper for force calculation
    # We use mode="rnvp" to ensure the flow is used in the action calculation
    sgc = phi4_c1(sg, mgf, device=sg.device, dtype=sg.dtype, mode="rnvp")
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Sample from the Flow's prior (z) and generate fine fields (phi_fine)
        # This gives us a batch of configurations 'proposed' by the current flow.
        z = mgf.prior_sample(batch_size)
        phi_fine, log_det = mgf(z) # Forward pass
        
        # 2. Decompose phi_fine into phi_coarse and pics (fluctuations)
        # We perform the RG coarsening step.
        # generate_cfg_levels populates sgc.phis (levels) and sgc.pics (fluctuations)
        sgc.generate_cfg_levels(phi_fine)
        
        # Get the coarsest field (input to the force calculation)
        phi_coarse = sgc.phis[-1]
        
        # 3. Calculate L_force (Stiffness)
        # Force on the coarse field given the CURRENT fluctuations (pics)
        # We need create_graph=True because we want to differentiate the force wrt flow params.
        force_original = sgc.force(phi_coarse, create_graph=True)
        
        # Loss 1: Minimize magnitude of coarse force (smoother landscape)
        loss_force = (force_original.norm(dim=1)**2).mean()
        
        # 4. Calculate L_sens (Sensitivity)
        # We want the force to be insensitive to the fluctuations (pics).
        # We create a 'perturbed' set of fluctuations.
        # Assuming pics are roughly N(0,1) in a good RG, we can sample standard normal noise.
        # Or simpler: Permute the pics in the batch dimension to "refresh" them.
        
        # Save original pics to restore later
        original_pics = [p.clone() for p in sgc.pics]
        
        # Permute fluctuations (mix r' across batch)
        perm_idx = tr.randperm(batch_size, device=sg.device)
        for i in range(len(sgc.pics)):
            sgc.pics[i] = sgc.pics[i][perm_idx]
            
        # Recalculate force with PERMUTED fluctuations
        force_perturbed = sgc.force(phi_coarse, create_graph=True)
        
        # Loss 2: Minimize difference between forces (variance wrt r')
        loss_sens = ((force_original - force_perturbed).norm(dim=1)**2).mean()
        
        # Restore original pics (good practice)
        for i in range(len(sgc.pics)):
            sgc.pics[i] = original_pics[i]
            
        # Total Loss
        # Weighting can be adjusted. Start with 1:1.
        total_loss = loss_force + loss_sens
        
        # Backward and Step
        total_loss.backward()
        
        # Clip gradients to prevent explosion during training
        tr.nn.utils.clip_grad_norm_(mgf.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={total_loss.item():.4f} (Force={loss_force.item():.4f}, Sens={loss_sens.item():.4f})")
            
    print("Training complete.")
    return loss_history

def main():
    device = tr.device('cuda:0')
    dtype=tr.float64
    L=32
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
    batch_size = 200

    Vol = np.prod(lat)
    sg = s.phi4(lat,lam,mas,batch_size=batch_size,device=device,dtype=dtype)
    phi = sg.hotStart()


    #clone phi
    phi2 = phi.clone()


    mn2 = i.minnorm2(sg.force,sg.evolveQ,14,1.0)
    print(phi.shape,Vol,tr.mean(phi),tr.std(phi))
    hmc = u.hmc(T=sg,I=mn2,verbose=False)

    FLOW=lambda: FlowBijectorParity1(Nlayers=4,width=64,device=device,ttp=dtype)

    mgf=MGflow1([L,L],FLOW,RGlayer1("average",batch_size=batch_size,dtype=dtype,device=device),prior,depth=1).to(device)#.double()



# Optimizador
optimizer = tr.optim.Adam([p for p in mgf.parameters() if p.requires_grad], lr=1e-2,weight_decay=1e-4)
scheduler = tr.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1001)
loss_history = []

# Configuraciones
super_batch_size = 20 # Si lo usas para acumular gradientes
mgf.train() # IMPORTANTE: Activa dropouts o batchnorms si los hubiera

print("Iniciando entrenamiento...")

for t in range(1001):   
    optimizer.zero_grad() # Limpiamos gradientes ANTES del super-batch
    
    # Beta Annealing: De 5.0 (fuerza la energía) a 1.0 (física real) en 800 épocas
    #progreso = min(1.0, t / 800.0)
    #beta = 5.0 * (1 - progreso) + 1.0 * progreso
    beta = 1.0
    
    batch_loss_acum = 0
    
    # Opción A: Acumulación de Gradientes (Ahorra memoria RAM)
    for b in range(super_batch_size):
        # 1. Muestrear z
        z = mgf.prior_sample(batch_size)
        
        # 2. Forward eficiente (Obtenemos x y el LogDet en UNA pasada)
        # NO llamamos a mgf.log_prob(x) después.
        x, log_det_J = mgf(z) 
        
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
        action_val = sg.action(x)
        
        # --- PROBE-MOVE ACTION-CHANGE LOSS (NUEVO) ---
        # Objetivo: Que cambios 'gruesos' en z produzcan cambios mínimos en la Acción.
        # "Coarse Move": Perturbación de baja frecuencia en z.
        
        # Generar ruido grueso (Coarse Noise)
        # Hacemos downsample -> ruido -> upsample para simular movimiento en escala gruesa
        scale_factor = 4 # Factor de escala para definir "grueso" (ajustable)
        coarse_size = [s // scale_factor for s in mgf.size]
        xi_coarse = tr.randn(batch_size, 1, *coarse_size, device=mgf.prior.mean.device, dtype=mgf.prior.mean.dtype)
        # Upsample al tamaño fino (interpolación nearest para bloques)
        delta_xi = tr.nn.functional.interpolate(xi_coarse, size=mgf.size, mode='nearest')
        delta_xi = delta_xi.flatten(start_dim=1) # Aplanar para sumar a z (que es plano fuera de forward?)
        
        # Nota: z aquí es (batch, 1, L, L) o plano?
        # prior_sample devuelve (batch, 1, L, L). mgf(z) usa eso.
        # Pero flatten se usó para log_prob. z original sigue siendo tensor 4D?
        # Revisamos mgf.prior_sample: return ...reshape(batch, L, L) -> (batch, 1, L, L)?
        # mgf.prior_sample en mgmc1.py: (batch, 1, L, L).
        
        # Perturbar z
        z_probe = z + 0.1 * delta_xi.view_as(z) # delta = 0.1 (pequeña perturbación)
        
        # Forward del Probe
        x_probe, _ = mgf(z_probe)
        action_probe = sg.action(x_probe)
        
        # Loss Probe: Minimizar varianza del cambio de acción
        loss_probe = ((action_probe - action_val)**2).mean()
        
        # Total Loss: SOLO Probe Loss (sin KL)
        loss = loss_probe
        # ---------------------------------------------
        
        # Normalizamos por el super_batch
        loss = loss / super_batch_size 
 
        
        tr.nn.utils.clip_grad_norm_(mgf.parameters(), max_norm=10.0)
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

    #epsilon test variables...
    #clone phi
    #phi_init=phi.clone()
    #P=sg.refreshP()
    #K=sg.kinetic(P)
    #V=sg.action(phi_init)
    #Hinit=K+V

    import integrators as i
    #list
    eps1=[]
    deltaH=[]


    mgf.eval()
    phi= hmc.evolve(phi,1)
    sgc = phi4_c1(sg,mgf,device=sg.device,dtype=sg.dtype,mode="rnvp")
    #print(phi.view(phi.shape[0], 1, phi.shape[1], phi.shape[2]).shape)
    #print("phi",phi.shape)
    sgc.generate_cfg_levels(phi)


    phic=sgc.phis[-1]
    Pc=sgc.refreshP()

    K=sgc.kinetic(Pc)
    V=sgc.action(phic)
    Hinit=K+V
    print("initial energy",Hinit)

    for rk in np.geomspace(50,100,20):
        k=int(rk)
        dt=1.0/k
        print("we are in the",k,"step")
        mn2c = i.minnorm2(sgc.force,sgc.evolveQ,k,1.0)
        Pc2,phic2= mn2c.integrate(Pc,phic)
        
        
        Hfinal=sgc.kinetic(Pc2)+sgc.action(phic2)
        DH=tr.abs(Hfinal-Hinit)
        eps1.append(dt)
        deltaH.append(DH)
        #hmcc = u.hmc(T=sgc,I=mn2c,verbose=True)
        
        #phic_up=hmcc.evolve(phic,1)
        #now go back up
        #for sss in reversed(range(mgf.depth)):
            #phic_up= mgf.rg.refine(phic_up,sgc.pics[sss])

        
        #print("shape of fine field ",phic_up.shape)
        #phic_up=hmc.evolve(phic_up,1)

    DelH=tr.stack(deltaH).to("cpu")
    e=tr.tensor(eps1).to("cpu")

    plt.plot(e,DelH)

