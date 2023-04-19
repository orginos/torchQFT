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

import time

class SuperModel(nn.Module):
    def __init__(self,models,target):
        super(SuperModel, self).__init__()
        self.size = models[0].size
        self.models=nn.ModuleList(models)
        self.No = len(models)
        self.prior = models[0].prior # keep the first model's prior as the prior
        self.target=target # the is the target negative log(probability) otherwise known as the action
        
    #noise to fields
    def forward(self,z):
        x=z
        for k in range(len(self.models)):
            x=self.models[k].forward(x)
        return x
    
    #fields to noise
    def backward(self,x):
        log_det_J=x.new_zeros(x.shape[0])
        z=x
        for k in range(len(self.models)-1,-1,-1):
            z,J=self.models[k].backward(z)
            log_det_J+=J
        
        return z,log_det_J
            
    def log_prob(self,x):
        z, logp = self.backward(x)
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

    def loss(self,x):
        return (self.log_prob(x)+self.target(x)).mean()

def trainSM( SuperM, levels=[], epochs=100,batch_size=16,super_batch_size=1,learning_rate=1.0e-4):
    params = []
    if levels==[] :
        params = [p for p in SuperM.parameters() if p.requires_grad==True]
    else:
        for l in levels:
            params.extend([p for p in SuperM.models[l].parameters() if p.requires_grad==True])
    print("Number of parameters to train is: ",len(params))
    optimizer = tr.optim.Adam(params, lr=learning_rate)
    loss_history = []
    #tic=time.perf_counter()
    pbar = tqdm(range(epochs))
    for t in pbar:   
        z = SuperM.prior_sample(batch_size)
        x = SuperM(z) # generate a sample
        tloss = SuperM.loss(x) #(SuperM.log_prob(x)+o.action(x)).mean() # KL divergence (or not?)
        for b in range(1,super_batch_size):
            z = SuperM.prior_sample(batch_size)
            x = SuperM(z) # generate a sample
            tloss += SuperM.loss(x)#(sm.log_prob(x)+o.action(x)).mean() # KL divergence (or not?)
        loss =tloss/super_batch_size    
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_history.append(loss.detach().numpy())
        pbar.set_postfix({'loss': loss.detach().numpy()})
        #print(loss_history[-1])
        #if t % 10 == 0:
        #    toc=time.perf_counter()
        #    print('iter %s:' % t, 'loss = %.3f' % loss,'time = %.3f' % (toc-tic),'seconds')
        #    tic=time.perf_counter()
    return loss_history


def plot_loss(lh):
    plt.plot(np.arange(len(lh)),lh)
    plt.xlabel("epoch")
    plt.ylabel("KL-divergence")
    plt.title("Training history of MG super model ")
    plt.show()
    #title = "L="+str(L)+"-batch="+str(batch_size)+"-LR="+str(learning_rate)

def validate(batch_size,mm):
    x=mm.sample(batch_size)
    diff = (o.action(x)+mm.log_prob(x)).detach()
    m_diff = diff.mean()
    diff -= m_diff
    print("max  action diff: ", tr.max(diff.abs()).numpy())
    print("min  action diff: ", tr.min(diff.abs()).numpy())
    print("mean action diff: ", m_diff.detach().numpy())
    print("std  action diff: ", diff.std().numpy())
    #compute the reweighting factor
    foo = tr.exp(-diff)
    #print(foo)
    w = foo/tr.mean(foo)

    print("mean re-weighting factor: " , w.mean().numpy())
    print("std  re-weighting factor: " , w.std().numpy())
    
    #plt.show()
    #title = "L"+str(L)+"-batch"+str(batch_size)+"-LR"+str(learning_rate)
    #plt.savefig("sm_mg_train_"+title+".pdf")
    #plt.show()
    #plt.close()
    logbins = np.logspace(np.log10(1e-3),np.log10(1e3),int(w.shape[0]/10))
    _=plt.hist(w,bins=logbins)
    plt.xscale('log')
    plt.title('Reweighting factor')
    #plt.savefig("sm_mg_rw_"+title+".pdf")
    plt.show()
    #plt.close()
    _=plt.hist(diff.detach(),bins=int(w.shape[0]/10))
    plt.title('ΔS distribution')
    #plt.savefig("sm_mg_ds_"+title+".pdf")


def test_reversibility():
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    L=16
    batch_size=32
    V=L*L
    lam =1.0
    mass= -0.2
    o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
    phi = o.hotStart()
    #set up a prior
    normal = distributions.Normal(tr.zeros(V,device=device),tr.ones(V,device=device))
    prior= distributions.Independent(normal, 1)

    width=16
    Nlayers=1
    bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
    mg = lambda : m.MGflow([L,L],bij,m.RGlayer("average"),prior)
    sm = SuperModel([mg(),mg()],target =o.action )

    c=0
    for tt in sm.parameters():
        #print(tt.shape)
        if tt.requires_grad==True :
            c+=tt.numel()
      
    print("parameter count: ",c)

    tic = time.perf_counter()
    x=sm.sample(128)
    z,J=sm.backward(x)
    xx=sm.forward(z)
    dd = tr.sum(tr.abs(xx -x)).detach()
    toc=time.perf_counter()
    print("Should be zero: ",dd/(x.shape[0]*x.shape[1]*x.shape[2]))
    print(f"Time {(toc - tic):0.4f} seconds")

    tic = time.perf_counter()
    z = sm.prior_sample(128)
    x=sm.forward(z)
    zz,J=sm.backward(x)
    dd = tr.sum(tr.abs(zz -z)).detach()
    toc=time.perf_counter()
    print("Should be zero: ",dd/(x.shape[0]*x.shape[1]*x.shape[2]))
    print(f"Time {(toc - tic):0.4f} seconds")

def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='revers')
    parser.add_argument('-l', default='no-load')

    args = parser.parse_args()
    if(args.t=='revers'):
        print("Testing reversibility")
        test_reversibility()
    

if __name__ == "__main__":
    main()
    
