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



# PHI4 CLASS #
class phi4():
    def __init__(self,V,l,m,device, batch_size=1,dtype=tr.float32): 
        self.V = tuple(V)
        self.Vol = np.prod(V)
        self.Nd = len(V)
        self.lam = l 
        self.mass  = m
        self.mtil = m + 2*self.Nd
        self.Bs=batch_size
        self.device=device
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
    
def trainSM(SuperM,doc,folderSaveEvery,tag,levels=[],epochs=100, saveEvery=10,batch_size=16,super_batch_size=1,learning_rate=1.0e-4):
    tic = time.perf_counter()
    params = []
    if levels==[] :
        params = [p for p in SuperM.parameters() if p.requires_grad==True]
    else:
        for l in levels:
            params.extend([p for p in SuperM.models[l].parameters() if p.requires_grad==True])
    print("Number of parameters to train is: ",len(params))
    optimizer = tr.optim.Adam(params, lr=learning_rate)
    loss_history = []
    pbar = tqdm(range(epochs))
    for t in pbar:   
        z = SuperM.prior_sample(batch_size)
        x = SuperM(z) 
        tloss = SuperM.loss(x) 
        for b in range(1,super_batch_size):
            z = SuperM.prior_sample(batch_size)
            x = SuperM(z)
            tloss += SuperM.loss(x)
        loss =tloss/super_batch_size    
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_history.append(loss.cpu().detach().numpy())
        pbar.set_postfix({'loss': loss.cpu().detach().numpy()})
        #with h5py.File(rootFolder+"records/"+"TorchQFT_"+"L_"+str(L)+"_l_"+str(lam)+"_m_"+str(mass)+"_w_"+str(width)+"_n_"+str(Nlayers)+"_e_"+str(t)+".h5","w") as f:  
        #f.create_dataset("loss_history",data=loss.cpu().detach().numpy())
        #print(loss_history[-1])
        #if t % 10 == 0:
        #    toc=time.perf_counter()
        #    print('iter %s:' % t, 'loss = %.3f' % loss,'time = %.3f' % (toc-tic),'seconds')
        #    tic=time.perf_counter()
        if t % saveEvery == 0:
            #print('Epoch %s:' % t, 'loss = %.3f' % loss)
            #with h5py.File(folder+"records/"+"TorchQFT_"+tag+".h5","w") as f:  
            #ckp = SuperM.module.state_dict()
            ckp = SuperM.state_dict()
            #print("goal"+folderSaveEvery)
            PATH = folderSaveEvery+"/records/TorchQFT_"+tag+"_checkpoint_"+str(t)+".pt"
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
def validate(batch_size,title,mm,folder,width,Nlayers,doc):
    x=mm.sample(batch_size)
    diff = mm.diff(x).detach()
    diff = mm.diff(x).detach().cpu()
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





# MAIN DEF OF PROGRAM#
def main(folder, name, load, device: int, depth: int, epochs: int, lattice: int, mass: float, lam: float, batchSize: int, numberOfBatches: int, learningRate:float, numbersOfLearningRate: int, width:int, numberOfWidth: int, nLayers: int, numberOfLayers: int, saveEvery: int):    
    if args.folder is None:
        rootFolder = "./models/device"+str(device)+"_depth"+str(depth)+"_epochs"+str(epochs)+"_saveEvery"+str(saveEvery)+"_lattice"+str(lattice)+"_mass"+str(mass)+"_lambda"+str(lam)+"_batchsize"+str(batchSize)+"_numberbatches"+str(numberOfBatches)+"_learningrate"+str(learningRate)+"_nOfLr"+str(numbersOfLearningRate)+"_width"+str(width)+"_numberOfWidth"+str(numberOfWidth)+"_nlayers"+str(nLayers)+"_nOfLayers"+str(numberOfLayers)+"/"
        print("No specified saving path, using",rootFolder)
    else:
        rootFolder = args.folder
    if rootFolder[-1] != '/':
        rootFolder += '/'
    utils.createWorkSpace(rootFolder)
    if args.load:
        with h5py.File(rootFolder+"/parameters.hdf5","r") as f:
            load_flag = False
            device=int(np.array(f["device"]))
            depth=int(np.array(f["depth"]))
            epochs=int(np.array(f["epochs"]))
            lattice=int(np.array(f["lattice"]))
            mass=float(np.array(f["mass"]))
            lam=float(np.array(f["lam"]))
            batchSize=int(np.array(f["batchSize"]))
            numberOfBatches=int(np.array(f["numberOfBatches"]))
            learningRate=float(np.array(f["learningRate"]))
            numbersOfLearningRate=int(np.array(f["numbersOfLearningRate"]))
            width=int(np.array(f["width"]))
            numberOfWidth=int(np.array(f["numberOfWidth"]))
            nLayers=int(np.array(f["nLayers"]))
            numberOfLayers=int(np.array(f["numberOfLayers"]))
            saveEvery=int(np.array(f["saveEvery"]))
    else:
        load_flag = True
        device=args.device
        d=args.depth
        e=args.epochs
        lattice=args.lattice
        mass=args.mass
        lam=args.lam
        batchSize=args.batchSize
        numberOfBatches=args.numberOfBatches
        learningRate=args.learningRate
        numbersOfLearningRate=args.numbersOfLearningRate
        width=args.width
        numberOfWidth=args.numberOfWidth
        nLayers=args.nLayers
        numberOfLayers=args.numberOfLayers
        saveEvery=args.saveEvery
        with h5py.File(rootFolder+"parameters.hdf5","w") as f:
            f.create_dataset("device",data=args.device)
            f.create_dataset("depth",data=args.depth)
            f.create_dataset("epochs",data=args.epochs)
            f.create_dataset("lattice",data=args.lattice)
            f.create_dataset("mass",data=args.mass)
            f.create_dataset("lam",data=args.lam)
            f.create_dataset("batchSize",data=args.batchSize)
            f.create_dataset("numberOfBatches",data=args.numberOfBatches)
            f.create_dataset("learningRate",data=args.learningRate)
            f.create_dataset("numbersOfLearningRate",data=args.numbersOfLearningRate)
            f.create_dataset("width",data=args.width)
            f.create_dataset("numberOfWidth",data=args.numberOfWidth)
            f.create_dataset("nLayers",data=args.nLayers)
            f.create_dataset("numberOfLayers",data=args.numberOfLayers)
            f.create_dataset("saveEvery",data=args.saveEvery)

    cuda=device
    device=tr.device("cpu" if cuda<0 else "cuda:"+str(cuda))
    print(f"Using {device} device")
    depth=depth
    epochs=epochs
    L=lattice
    mass=mass
    lam=lam
    batch_size=batchSize
    numberOfBatches=numberOfBatches
    learning_rate=learningRate
    numbersOfLearningRate=numbersOfLearningRate
    width_=width
    numberOfWidth=numberOfWidth
    n_layers=nLayers
    numberOfLayers=numberOfLayers
    savePeriod=saveEvery
    V=L*L
    o=phi4([L,L],lam,mass,device=device,batch_size=batch_size)
    phi=o.hotStart()
    normal=distributions.Normal(tr.zeros(V).to(device),tr.ones(V).to(device))
    prior=distributions.Independent(normal, 1)
    doc = open(rootFolder+"/MGflow.txt", "w")
    #doc = open(rootFolder+"/MGflow_"+tag+".txt", "w")
    doc.write("Device\tEpochs\tSaveEvery\tLattice\tLambda\tMass\tDepth\tWidth\tNLayers\tBatchSize\tLearningRate\tParameters\tTime\tMaxActionDiff\tMinActionDiff\tMeanActionDiff\tStdActionDiff\tMeanReWeightingFactor\tStdReWeightingFactor\n")

    for nLayers in n_layers+(np.arange(numberOfLayers)):
        for width in width_*(2**np.arange(numberOfWidth)):
            #tag="device"+str(device)+"_depth"+str(depth)+"_epochs"+str(epochs)+"_lattice"+str(lattice)+"_mass"+str(mass)+"_lambda"+str(lam)+"_batchSize"+str(batchSize)+"_numberOfBatches"+str(numberOfBatches)+"_learningRate"+str(learningRate)+"_width"+str(width)+"_numberOfWidth"+str(numberOfWidth)+"_nLayers"+str(nLayers)

            bij=lambda: FlowBijector(Nlayers=n_layers,width=width_)
            mg=lambda: MGflow([L,L],bij,RGlayer("average"),prior)
            models=[]
            print("Initializing ",depth," stages")
            for d in range(depth):
                models.append(mg())        
            sm=SuperModel(models,target=o.action )
            sm=sm.to(device)
            c=0
            for tt in sm.parameters():
                if tt.requires_grad==True :
                    c+=tt.numel()
            print("Parameter count: ",c)
            if(args.load):
                import os
                import glob
                name = max(glob.iglob(rootFolder+'model/*.model'), key=os.path.getctime)
                print("load saving at "+name)
                sm.load_state_dict(tr.load(name))
                sm.eval()
            savePath=None
            save=True
            if savePath is None:
                savePath = "./models/"
            for batchSize in batch_size*(2**np.arange(numberOfBatches)):
                for learningRate in learning_rate*(10**np.arange(numbersOfLearningRate)):
                    tag="_bs"+str(batchSize)+"_lr"+str(learningRate)+"_w"+str(width)+"_nl"+str(nLayers)
                    folderSaveEvery=rootFolder
                    folder=rootFolder+'plot/'
                    print("Running with batch size=",batchSize,  "width=",width,  "nlayers=",nLayers,  " & learning rate=",learningRate)        
                    doc.write(str(device)+"\t"+str(epochs)+"\t"+str(saveEvery)+"\t"+str(lattice)+"\t"+str(lam)+"\t"+str(mass)+"\t"+str(depth)+"\t"+str(width)+"\t"+str(nLayers)+"\t"+str(batchSize)+"\t"+str(learningRate)+"\t"+str(c))
                    loss_hist=trainSM(sm,doc,folderSaveEvery,tag,levels=[],epochs=epochs,saveEvery=saveEvery,batch_size=batchSize,super_batch_size=1,learning_rate=learningRate)
                    tt = tag
                    #tt = "_batchSize"+str(batchSize)                    
                    
                    plot_loss(loss_hist,tt,folder)
                    validate(1024,tt,sm,folder,width,nLayers,doc)
                    if args.name is None:
                        name= "TorchQFT_"+"lattice"+str(lattice)+"_lambda"+str(lam)+"_mass"+str(mass)+tag
                    else:
                        name = args.name
                    if(not args.load):
                        name= "TorchQFT_"+"lattice"+str(lattice)+"_lambda"+str(lam)+"_mass"+str(mass)+tag
                    tr.save(sm.state_dict(), rootFolder+'model/'+name+'.model')
    doc.close()








# _name_ MAIN CALL #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("--folder", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lattice', type=int, default=16)
    parser.add_argument('--mass', type=float, default=-0.5)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--batchSize', type=int, default=4)
    parser.add_argument('--numberOfBatches', type=int, default=6)
    parser.add_argument('--learningRate', type=float, default=0.0001)
    parser.add_argument('--numbersOfLearningRate', type=float, default=3)
    parser.add_argument('--width' , type=int, default=4)
    parser.add_argument('--numberOfWidth', type=int, default=5)
    parser.add_argument('--nLayers', type=int, default=1)
    parser.add_argument('--numberOfLayers', type=int, default=4)
    parser.add_argument('--saveEvery', type=int, default=100)
    args = parser.parse_args()
    main(args.folder, args.name, args.load, args.device, args.depth, args.epochs, args.lattice, args.mass, args.lam, args.batchSize, args.numberOfBatches, args.learningRate, args.numbersOfLearningRate,args.width, args.numberOfWidth, args.nLayers, args.numberOfLayers, args.saveEvery)

