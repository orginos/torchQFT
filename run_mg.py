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

import argparse
import sys
    
parser = argparse.ArgumentParser()
parser.add_argument('-l', default='no-load')
args = parser.parse_args()

file = args.l
if(args.l=="no-load"):
    load_flag = False
else:
    load_flag = True
    
    
device = "cuda" if tr.cuda.is_available() else "cpu"
print(f"Using {device} device")

L=16

V=L*L
batch_size=16
super_batch_size = 2
lam =1.0
mass= -0.2
o  = p.phi4([L,L],lam,mass,batch_size=batch_size)

phi = o.hotStart()
    
#set up a prior
normal = distributions.Normal(tr.zeros(V),tr.ones(V))
prior= distributions.Independent(normal, 1)

width=256
Nlayers=1
bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
mg = m.MGflow([L,L],bij,m.RGlayer("average"),prior)
#print("The flow Model: ", mg)

if(load_flag):
    mg.load_state_dict(tr.load(file))
    mg.eval()
    
print(mg)
c=0
for tt in mg.parameters():
    #print(tt.shape)
    if tt.requires_grad==True :
        c+=tt.numel()
        
print("parameter count: ",c)

learning_rate = 1.0e-4
optimizer = tr.optim.Adam([p for p in mg.parameters() if p.requires_grad==True], lr=learning_rate)

loss_history = []
for t in range(101):   
    #with torch.no_grad():
    #z = prior.sample((batch_size,1)).squeeze().reshape(batch_size,L,L)
    
    z = mg.prior_sample(batch_size)
    x = mg(z) # generate a sample
    tloss = (mg.log_prob(x)+o.action(x)).mean() # KL divergence (or not?)
    for b in range(1,super_batch_size):
        z = mg.prior_sample(batch_size)
        x = mg(z) # generate a sample
        tloss += (mg.log_prob(x)+o.action(x)).mean() # KL divergence (or not?)
    loss =tloss/super_batch_size
        
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    loss_history.append(loss.detach().numpy())
    #print(loss_history[-1])
    if t % 10 == 0:
        #print(z.shape)
        print('iter %s:' % t, 'loss = %.3f' % loss)

x=mg.sample(1024)
diff = (o.action(x)+mg.log_prob(x)).detach()
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
    
plt.plot(np.arange(len(loss_history)),loss_history)
plt.xlabel("epoch")
plt.ylabel("KL-divergence")
title = "L="+str(L)+"-batch="+str(batch_size)+"-LR="+str(learning_rate)
plt.title("Training history of MG model "+title)
#plt.show()
title = "L"+str(L)+"-batch"+str(batch_size)+"-LR"+str(learning_rate)
plt.savefig("mg_train_"+title+".pdf")
#plt.show()
plt.close()
logbins = np.logspace(np.log10(1e-3),np.log10(1e3),int(w.shape[0]/10))
plt.hist(w,bins=logbins)
plt.xscale('log')
plt.savefig("mg_rw_"+title+".pdf")
#plt.show()
plt.close()
plt.hist(diff.detach(),bins=int(w.shape[0]/10))
plt.savefig("mg_ds_"+title+".pdf")
#plt.show()
#save the model
if(not load_flag):
    file = "phi4_"+str(L)+"_m"+str(mass)+"_l"+str(lam)+"_nvp_w"+str(width)+"_n"+str(Nlayers)+".dict"
tr.save(mg.state_dict(), file)
    
