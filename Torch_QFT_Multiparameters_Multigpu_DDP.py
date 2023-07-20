import numpy as np
import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
import time
import h5py
import utils
import flow
from scipy.stats import norm
import os
import re


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader

import os


# PHI4 CLASS #
class phi4():
    def __init__(self,V,l,m,rack,train_data,dtype=tr.float32): 
        self.V = tuple(V)
        self.Vol = np.prod(V)
        self.Nd = len(V)
        self.lam = l 
        self.mass  = m
        self.mtil = m + 2*self.Nd
        self.Bs=train_data.dataset[0]
        self.device=rack
        self.dtype=dtype
    def action(self,phi):
        phi2 = phi*phi
        A = tr.sum((0.5*self.mtil + (self.lam/24.0)*phi2)*phi2,dim=(1,2))
        for mu in range(1,self.Nd+1):
            A = A - tr.sum(phi*tr.roll(phi,shifts=-1,dims=mu),dim=(1,2))
        return A
    def force(self,phi):
        F = -self.mtil*phi - self.lam*phi**3/6.0
        for mu in range(1,self.Nd+1):
            F +=  tr.roll(phi,shifts= 1,dims=mu)+tr.roll(phi,shifts=-1,dims=mu)
        return F
    def refreshP(self):
        P = tr.normal(0.0,1.0,[self.Bs,self.V[0],self.V[1]],dtype=self.dtype,device=self.device)
        return P
    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    def kinetic(self,P):
        return tr.einsum('bxy,bxy->b',P,P)/2.0 ;
    def hotStart(self):
        sigma=tr.normal(0.0,1.0,[self.Bs,self.V[0],self.V[1]],dtype=self.dtype,device=self.device)
        return sigma




# RealNVP - ConvFlowLayer - RGlayer - MGflow, CLASSES #
class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior,data_dims=(1)):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = tr.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = tr.nn.ModuleList([nets() for _ in range(len(mask))])
        self.data_dims=data_dims
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * tr.exp(s) + t)
        return x
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
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp 
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        x = self.g(z)
        return x





class ConvFlowLayer(nn.Module):
    def __init__(self,size,bijector,Nsteps=1):
        super(ConvFlowLayer, self).__init__()
        self.Nsteps=Nsteps
        self.bj = tr.nn.ModuleList([bijector() for _ in range(2*Nsteps)])
        fold_params = dict(kernel_size=(2,2), dilation=1, padding=0, stride=(2,2))
        self.unfold = nn.Unfold(**fold_params)
        self.fold = nn.Fold(size,**fold_params)
    def forward(self,z):
        uf = self.unfold(z.view(z.shape[0],1,z.shape[1],z.shape[2])).transpose(2,1)
        for k in range(self.Nsteps):
            sf = self.unfold(tr.roll(self.fold(self.bj[2*k  ].g(uf).transpose(2,1)),dims=(2,3),shifts=(-1,-1))).transpose(2,1)
            uf = self.unfold(tr.roll(self.fold(self.bj[2*k+1].g(sf).transpose(2,1)),dims=(2,3),shifts=( 1, 1))).transpose(2,1)
        x = self.fold(uf.transpose(2,1)).squeeze()
        return x
    def backward(self,x):
        log_det_J=x.new_zeros(x.shape[0])
        z = x.view(x.shape[0],1,x.shape[1],x.shape[2])
        for k in reversed(range(self.Nsteps)):
            sz = self.unfold(tr.roll(z,dims=(2,3),shifts=(-1,-1))).transpose(2,1)
            ff,J = self.bj[2*k+1].f(sz)
            log_det_J += J
            sz = self.unfold(tr.roll(self.fold(ff.transpose(2,1)),dims=(2,3),shifts=(1,1))).transpose(2,1)
            ff,J = self.bj[2*k].f(sz)
            log_det_J += J 
            z = self.fold(ff.transpose(2,1))
        z = z.squeeze()
        return z,log_det_J
    def log_prob(self,x):
        z, logp = self.backward(x)
        return logp





def FlowBijector(Nlayers=3,width=256):
    mm = np.array([1,0,0,1])
    tV = mm.size
    nets = lambda: nn.Sequential(nn.Linear(tV, width), nn.LeakyReLU(), nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, tV), nn.Tanh())
    nett = lambda: nn.Sequential(nn.Linear(tV, width), nn.LeakyReLU(), nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, tV))
    masks = tr.from_numpy(np.array([mm, 1-mm] * Nlayers).astype(np.float32))
    normal = distributions.Normal(tr.zeros(tV),tr.ones(tV))
    prior= distributions.Independent(normal, 1)
    return  RealNVP(nets, nett, masks, prior, data_dims=(1,2))






class RGlayer(nn.Module):
    def __init__(self,transformation_type="select"):
        super(RGlayer, self).__init__()
        if(transformation_type=="select"):
            mask_c = [[1.0,0.0],[0.0,0.0]]
            mask_r = [[1.0,0.0],[0.0,0.0]]
        elif(transformation_type=="average"):
            mask_c = [[0.25,0.25],[0.25,0.25]]
            mask_r = [[1.00,1.00],[1.00,1.00]]
        else:
            print("Uknown RG blocking transformation. Using default.")
            mask_c = [[1.0,0.0],[0.0,0.0]]
            mask_r = [[1.0,0.0],[0.0,0.0]]
        self.type = transformation_type
        self.restrict = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,2),stride=2,bias=False)
        self.restrict.weight = tr.nn.Parameter(tr.tensor([[mask_c]]),requires_grad=False)
        self.prolong = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=(2,2),stride=2,bias=False)
        self.prolong.weight = tr.nn.Parameter(tr.tensor([[mask_r]]),requires_grad=False)
    
    def coarsen(self,f):
        ff = f.view(f.shape[0],1,f.shape[1],f.shape[2])
        c = self.restrict(ff)
        r = ff-self.prolong(c)
        return c.squeeze(),r.squeeze()
   
    def refine(self,c,r):
        cc = c.view(c.shape[0],1,c.shape[1],c.shape[2])
        rr = r.view(c.shape[0],1,r.shape[1],r.shape[2])
        return (self.prolong(cc)+rr).squeeze()





class MGflow(nn.Module):
    def __init__(self,size,bijector,rg,prior):
        super(MGflow, self).__init__()
        self.prior=prior
        self.rg=rg
        self.size = size
        minSize = min(size)
        print("Initializing MGflow module wiht size: ",minSize)
        self.depth = int(np.log(minSize)/np.log(2))
        print("Using depth: ", self.depth)
        print("Using rg type: ",rg.type)
        sizes = []
        for k in range(self.depth):
            sizes.append([int(size[i]/(2**k)) for i in range(len(size))])
            print("(depth, size): ", k, sizes[-1])
        self.cflow=tr.nn.ModuleList([ConvFlowLayer(sizes[k],bijector) for k in range(self.depth)])
    
    def forward(self,z):
        x = z
        fines = []
        for k in range(self.depth-1):
            c,f =self.rg.coarsen(x)
            x=c
            fines.append(f)
        for k in range(self.depth-1,0,-1):
            fx=self.cflow[k](x)
            x=self.rg.refine(fx,fines[k-1])
        fx = self.cflow[0](x)        
        return fx
    
    def backward(self,x):
        log_det_J=x.new_zeros(x.shape[0])
        fines = []
        for k in range(self.depth-1):
            fx,J = self.cflow[k].backward(x)
            log_det_J += J
            cx,ff = self.rg.coarsen(fx)
            fines.append(ff)
            x=cx
        fx,J = self.cflow[self.depth-1].backward(x)
        log_det_J += J
        for k in range(self.depth-2,-1,-1):
            z=self.rg.refine(fx,fines[k])
            fx=z  
        return z,log_det_J
    
    def log_prob(self,x):
        z, logp = self.backward(x)
        return self.prior.log_prob(z.flatten(start_dim=1)) + logp
    
    def sample(self, batchSize): 
        z = self.prior_sample(batchSize)
        x = self.forward(z)
        return x
    
    def prior_sample(self,batch_size):
        return self.prior.sample((batch_size,1)).reshape(batch_size,self.size[0],self.size[1])




# SuperModel (stacked) CLASS #        
class SuperModel(nn.Module):
    def __init__(self,models,target):
        super(SuperModel, self).__init__()
        self.size = models[0].size
        self.models=nn.ModuleList(models)
        #self.No = len(models)
        self.prior = models[0].prior 
        self.target=target
    
    def forward(self,z):
        x=z
        for k in range(len(self.models)):
            x=self.models[k].forward(x)
        return x
    
    def backward(self,x):
        log_det_J=x.new_zeros(x.shape[0])
        z=x
        for k in range(len(self.models)-1,-1,-1):
            z,J=self.models[k].backward(z)
            log_det_J+=J
        return z,log_det_J
    
    def log_prob(self,x):
        z, logp = self.backward(x)
        return self.prior.log_prob(z.flatten(start_dim=1)) + logp     
    
    def sample(self, batchSize): 
        z = self.prior_sample(batchSize)
        x = self.forward(z)
        return x
    
    def prior_sample(self,batch_size):
        return self.prior.sample((batch_size,1)).reshape(batch_size,self.size[0],self.size[1])
    
    def loss(self,x):
        return (self.log_prob(x)+self.target(x)).mean()
    
    def diff(self,x):
        return self.log_prob(x)+self.target(x)
    

def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset=[dataset],
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler([dataset])
    )



def trainSM(SuperM, doc, tag, total_epochs, learning_rate, save_every, train_data, rank, levels=[]):
    tic = time.perf_counter()
    params = []
    if levels==[] :
        params = [p for p in SuperM.module.parameters() if p.requires_grad==True]
    else:
        for l in levels:
            params.extend([p for p in SuperM.module.models[l].parameters() if p.requires_grad==True])
    print("Number of parameters to train is: ",len(params))
    optimizer = tr.optim.Adam(params, lr=learning_rate)
    loss_history = []
    pbar = tqdm(range(total_epochs))
    for epoch in range(total_epochs):
        b_sz = train_data.dataset[0]
        print(f"[GPU{rank}] Epoch {epoch} | TrainingSize: {b_sz} | Steps: {len(train_data)}", end="")
        train_data.sampler.set_epoch(epoch)

        z = SuperM.module.prior_sample(train_data.dataset[0])
        x = SuperM(z) 
        tloss = SuperM.module.loss(x) 
        for b in range(1,train_data.batch_size):
            z = SuperM.module.prior_sample(train_data.dataset[0])
            x = SuperM(z)
            tloss += SuperM.module.loss(x)
        loss =tloss/train_data.batch_size    
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_history.append(loss.cpu().detach().numpy())
        print(' | Loss: %.3f' % loss)

        #pbar.set_postfix({' | loss': loss.cpu().detach().numpy()})
        #with h5py.File(rootFolder+"records/"+"TorchQFT_"+"L_"+str(L)+"_l_"+str(lam)+"_m_"+str(mass)+"_w_"+str(width)+"_n_"+str(Nlayers)+"_e_"+str(t)+".h5","w") as f:  
        #f.create_dataset("loss_history",data=loss.cpu().detach().numpy())
        #print(loss_history[-1])
        #if t % 10 == 0:
        #    toc=time.perf_counter()
        #    print('iter %s:' % t, 'loss = %.3f' % loss,'time = %.3f' % (toc-tic),'seconds')
        #    tic=time.perf_counter()
        if epoch % save_every == 0:
            #print('Epoch %s:' % t, 'loss = %.3f' % loss)
            #with h5py.File(folder+"records/"+"TorchQFT_"+tag+".h5","w") as f:  
            #ckp = SuperM.module.state_dict()
            ckp = SuperM.state_dict()
            #print("goal"+folderSaveEvery)
            PATH = "TorchQFT_"+tag+"_checkpoint_"+str(epoch)+".pt"
            tr.save(ckp, PATH)
            #f.create_dataset("loss_history",data=loss.cpu().detach().numpy())
    toc = time.perf_counter()
    doc.write("\t"+str(toc-tic))
    print(f"Time {(toc - tic):0.4f} seconds, Training checkpoint saved")
    return loss_history






def plot_loss(lh,title,folder):
    plt.plot(np.arange(len(lh)),lh)
    plt.xlabel("epoch")
    plt.ylabel("KL-divergence")
    plt.title("Training history of MG super model ")
    plt.savefig(folder+"sm_tr_"+title+".png")
    plt.close()





def validate(test_size,title,mm,folder,width,Nlayers,doc):
    x=mm.module.sample(test_size)
    #diff = mm.diff(x).detach()
    diff = mm.module.diff(x).detach().cpu()
    m_diff = diff.cpu().mean()
    diff -= m_diff
    print("max  action diff: ", tr.max(diff.abs()).numpy())
    print("min  action diff: ", tr.min(diff.abs()).numpy())
    print("mean action diff: ", m_diff.detach().numpy())
    print("std  action diff: ", diff.std().numpy())
    foo = tr.exp(-diff)
    w = foo/tr.mean(foo)
    print("mean re-weighting factor: " , w.mean().numpy())
    print("std  re-weighting factor: " , w.std().numpy())
    doc.write("\t"+str(tr.max(diff.abs()).numpy())+"\t"+str(tr.min(diff.abs()).numpy())+"\t"+str(m_diff.detach().numpy())+"\t"+str(diff.std().numpy())+"\t"+str(w.mean().numpy())+"\t"+str(w.std().numpy())+"\n")
    logbins = np.logspace(np.log10(1e-3),np.log10(1e3),int(w.shape[0]/10))
    _=plt.hist(w,bins=logbins)
    plt.xscale('log')
    plt.title('Reweighting factor')
    plt.savefig(folder+"sm_rw_"+title+".png")
    plt.close()
    _=plt.hist(diff.detach(),bins=int(w.shape[0]/10))
    plt.title('ΔS distribution')
    plt.savefig(folder+"sm_ds_"+title+".png")
    plt.close()
    doc.flush()
    os.fsync(doc.fileno())






#function gia arxikopoihsh rank kai world_size
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    tr.cuda.set_device(rank)









# MAIN DEF OF PROGRAM#
#def main(rank:int, world_size: int, folder: None, name:None, load:None, depth_size: int, total_epochs: int, lattice_size: int, mass_size: float, lambda_size: float, training_size: int, number_of_training_size: int, batch_size: int, learning_rate:float, number_of_learning_rate: int, width_size:int, number_of_width: int, nLayers_size: int, number_of_layers: int, save_every: int):    
def main(rank:int, world_size: int, depth_size: int, total_epochs: int, lattice_size: int, mass_size: float, lambda_size: float, training_size: int, number_of_training_size: int, batch_size: int, learning_rate:float, number_of_learning_rate: int, width_size:int, number_of_width: int, nLayers_size: int, number_of_layers: int, save_every: int):    
    
    ddp_setup(rank, world_size)
    print("Function: Setup - OK | rank:",rank," | world_size:",world_size)
    
    ''' 
    #if args.folder is None:
    if folder is None:

        rootFolder = "./models/device"+str(rank)+"_depth_size"+str(depth_size)+"_total_epochs"+str(total_epochs)+"_save_every"+str(save_every)+"_lattice_size"+str(lattice_size)+"_mass_size"+str(mass_size)+"_lambda_size"+str(lambda_size)+"_training_size"+str(training_size)+"_number_of_training_size"+str(number_of_training_size)+"_batch_size"+str(batch_size)+"_learning_rate"+str(learning_rate)+"_number_of_learning_rate"+str(number_of_learning_rate)+"_width_size"+str(width_size)+"_number_of_width"+str(number_of_width)+"_nlayers"+str(nLayers_size)+"_number_of_layers"+str(number_of_layers)+"/"
        print("No specified saving path, using",rootFolder)
    else:
        rootFolder = args.folder
    if rootFolder[-1] != '/':
        rootFolder += '/'
    utils.createWorkSpace(rootFolder)
    if args.load:
        with h5py.File(rootFolder+"/parameters.hdf5","r") as f:
            load_flag = False
            #device=int(np.array(f["device"]))
            depth_size=int(np.array(f["depth_size"]))
            total_epochs=int(np.array(f["total_epochs"]))
            lattice_size=int(np.array(f["lattice_size"]))
            mass_size=float(np.array(f["mass_size"]))
            lambda_size=float(np.array(f["lambda_size"]))
            trainingSize=int(np.array(f["training_size"]))
            number_of_training_size=int(np.array(f["number_of_training_size"]))
            batch_size=int(np.array(f["batch_size"]))
            learningRate=float(np.array(f["learning_rate"]))
            number_of_learning_rate=int(np.array(f["number_of_learning_rate"]))
            width=int(np.array(f["width_size"]))
            number_of_width=int(np.array(f["number_of_width"]))
            nLayers=int(np.array(f["nLayers_size"]))
            number_of_layers=int(np.array(f["number_of_layers"]))
            save_every=int(np.array(f["save_every"]))
    else:
        load_flag = True
        #device=args.device
        depth_size=args.depth_size
        total_epochs=args.total_epochs
        lattice_size=args.lattice_size
        mass_size=args.mass_size
        lambda_size=args.lambda_size
        trainingSize=args.training_size
        number_of_training_size=args.number_of_training_size
        batch_size=args.batch_size
        learningRate=args.learning_rate
        number_of_learning_rate=args.number_of_learning_rate
        width=args.width_size
        number_of_width=args.number_of_width
        nLayers=args.nLayers_size
        number_of_layers=args.number_of_layers
        save_every=args.save_every
        with h5py.File(rootFolder+"parameters.hdf5","w") as f:
            #f.create_dataset("device",data=args.device)
            f.create_dataset("depth_size",data=args.depth_size)
            f.create_dataset("total_epochs",data=args.total_epochs)
            f.create_dataset("lattice_size",data=args.lattice_size)
            f.create_dataset("mass_size",data=args.mass_size)
            f.create_dataset("lambda_size",data=args.lambda_size)
            f.create_dataset("trainingSize",data=args.training_size)
            f.create_dataset("number_of_training_size",data=args.number_of_training_size)
            f.create_dataset("batch_size",data=args.batch_size)
            f.create_dataset("learningRate",data=args.learning_rate)
            f.create_dataset("number_of_learning_rate",data=args.number_of_learning_rate)
            f.create_dataset("width",data=args.width_size)
            f.create_dataset("number_of_width",data=args.number_of_width)
            f.create_dataset("nLayers",data=args.nLayers_size)
            f.create_dataset("number_of_layers",data=args.number_of_layers)
            f.create_dataset("save_every",data=args.save_every)
    '''
    #cuda=device
    #device=tr.device("cpu" if cuda<0 else "cuda:"+str(cuda))
    #print(f"Using {device} device")

    depth_size=depth_size
    total_epochs=total_epochs
    L=lattice_size
    mass_size=mass_size
    lambda_size=lambda_size
    trainingSize=training_size
    number_of_training_size=number_of_training_size
    learningRate=learning_rate
    number_of_learning_rate=number_of_learning_rate
    width=width_size
    number_of_width=number_of_width
    nLayers=nLayers_size
    number_of_layers=number_of_layers
    save_every=save_every
    
    train_data = prepare_dataloader(training_size, batch_size)
    V=L*L
    o=phi4([L,L],lambda_size,mass_size, rank, train_data)
    phi=o.hotStart()
    normal=distributions.Normal(tr.zeros(V).to(rank),tr.ones(V).to(rank))
    prior=distributions.Independent(normal, 1)
    #doc = open(rootFolder+"/MGflow.txt", "w")
    doc = open("MGflow.txt", "w")

    #doc = open(rootFolder+"/MGflow_"+tag+".txt", "w")
    doc.write("Device\tTotal_Epochs\tSave_Every\tLattice_Size\tLambda_Size\tMass_Size\tDepth_Size\tWidth_Size\tNLayers\tTraining_Size\tBatch_Size\tLearning_Rate\tParameters\tTime\tMax_Action_Diff\tMin_Action_Diff\tMean_Action_Diff\tStd_Action_Diff\tMean_ReWeighting_Factor\tStd_ReWeighting_Factor\n")

    for nLayers in nLayers_size+(np.arange(number_of_layers)):
        for width in width_size*(2**np.arange(number_of_width)):
            #tag="device"+str(device)+"_depth"+str(depth)+"_epochs"+str(epochs)+"_lattice"+str(lattice)+"_mass"+str(mass)+"_lambda"+str(lam)+"_batchSize"+str(batchSize)+"_numberOfBatches"+str(numberOfBatches)+"_learningRate"+str(learningRate)+"_width"+str(width)+"_numberOfWidth"+str(numberOfWidth)+"_nLayers"+str(nLayers)

            bij=lambda: FlowBijector(Nlayers=nLayers_size,width=width_size)
            mg=lambda: MGflow([L,L],bij,RGlayer("average"),prior)
            models=[]
            print("Initializing ",depth_size," stages")
            for d in range(depth_size):
                models.append(mg())        
            sm=SuperModel(models,target=o.action).to(rank)
            sm=DDP(sm, device_ids=[rank])
            c=0
            for tt in sm.parameters():
                if tt.requires_grad==True :
                    c+=tt.numel()
            print("Parameter count: ",c)
            #if(args.load):
            import os
            import glob
            #name = max(glob.iglob(rootFolder+'model/*.model'), key=os.path.getctime)
            #name = max(glob.iglob('model/*.model'), key=os.path.getctime)

            #print("load saving at "+name)
            #sm.load_state_dict(tr.load(name))
            sm.eval()
            savePath=None
            save=True
            if savePath is None:
                savePath = "./models/"
            for trainingSize in training_size*(2**np.arange(number_of_training_size)):
                for learningRate in learning_rate*(10**np.arange(number_of_learning_rate)):
                    tag="_bs"+str(training_size)+"_lr"+str(learningRate)+"_w"+str(width)+"_nl"+str(nLayers)
                    #folderSaveEvery=rootFolder
                    folderSaveEvery=None
                    #folder=rootFolder+'plot/'

                    folder=''
                    print("Running with training size=",trainingSize,  "width=",width,  "nlayers=",nLayers,  " & learning rate=",learningRate)        
                    #doc.write(str(rank)+"\t"+str(total_epochs)+"\t"+str(save_every)+"\t"+str(lattice_size)+"\t"+str(lambda_size)+"\t"+str(mass_size)+"\t"+str(depth_size)+"\t"+str(width_)+"\t"+str(nLayers)+"\t"+str(trainingSize)+"\t"+str(learningRate)+"\t"+str(c))
                    #loss_hist=trainSM(sm,doc,folderSaveEvery,tag,levels=[],epochs=total_epochs,saveEvery=save_every,batch_size=batchSize,batch_size=batch_size,learning_rate=learning_rate)
                    loss_hist=trainSM(sm, doc, tag, total_epochs, learning_rate, save_every, train_data, rank, levels=[])

                    tt = tag
                    #tt = "_batchSize"+str(batchSize)                    
                    plot_loss(loss_hist,tt,folder)
                    validate(1024,tt,sm,folder,width,nLayers,doc)
                    ''' 
                    if args.name is None:
                        name= "TorchQFT_"+"lattice"+str(lattice_size)+"_lambda"+str(lambda_size)+"_mass"+str(mass_size)+tag
                    else:
                        name = args.name
                    if(not args.load):
                        name= "TorchQFT_"+"lattice"+str(lattice_size)+"_lambda"+str(lambda_size)+"_mass"+str(mass_size)+tag
                    #tr.save(sm.state_dict(), rootFolder+'model/'+name+'.model')
                    tr.save(sm.state_dict(), 'model/'+name+'.model')
                    '''
    doc.close()
    destroy_process_group()







# _name_ MAIN CALL #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    #parser.add_argument("--folder", type=dir, default=None)
    #parser.add_argument("--name", default=None)
    #parser.add_argument("--load", action='store_true')
    #parser.add_argument("--device", type=int, default=-1)
    parser.add_argument('--depth_size', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=1000)
    parser.add_argument('--lattice_size', type=int, default=16)
    parser.add_argument('--mass_size', type=float, default=-0.5)
    parser.add_argument('--lambda_size', type=float, default=1.0)
    parser.add_argument('--training_size', type=int, default=64)
    parser.add_argument('--number_of_training_size', type=int, default=1)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--number_of_learning_rate', type=float, default=1)
    parser.add_argument('--width_size', type=int, default=16)
    parser.add_argument('--number_of_width', type=int, default=1)
    parser.add_argument('--nLayers_size', type=int, default=2)
    parser.add_argument('--number_of_layers', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=100)
    args = parser.parse_args()
    

    world_size = tr.cuda.device_count()
    #mp.spawn(main, args=(world_size, args.folder, args.name, args.load, args.depth_size, args.total_epochs, args.lattice_size, args.mass_size, args.lambda_size, args.training_size, args.number_of_training_size, args.batch_size, args.learning_rate, args.number_of_learning_rate, args.width_size, args.number_of_width, args.nLayers_size, args.number_of_layers, args.save_every), nprocs=world_size)
    #main(args.folder, args.name, args.load, args.device, args.depth, args.epochs, args.lattice, args.mass, args.lam, args.batchSize, args.numberOfBatches, args.learningRate, args.numbersOfLearningRate,args.width, args.numberOfWidth, args.nLayers, args.numberOfLayers, args.saveEvery)
    mp.spawn(main, args=(world_size, args.depth_size, args.total_epochs, args.lattice_size, args.mass_size, args.lambda_size, args.training_size, args.number_of_training_size, args.batch_size, args.learning_rate, args.number_of_learning_rate, args.width_size, args.number_of_width, args.nLayers_size, args.number_of_layers, args.save_every), nprocs=world_size, join=True)

