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
import h5py
import utils
import flow
#from flow import Flow
import subprocess
from matplotlib import colors
import io
import time
start_time = time.time()


parser = argparse.ArgumentParser(description="QFT version 1.0")
parser.add_argument("-folder", default=None)
parser.add_argument("-name", default=None, help='name of flow')
group = parser.add_argument_group('Initializing parameters')
group.add_argument("-load", action='store_true', help="if load from folder")
group.add_argument("-cuda", type=int, default=-1, help="default device: cpu")
group = parser.add_argument_group('Ising parameters')
group.add_argument("-L",type=int, default=16, help="default linear size: 16")
group.add_argument("-la",type=float, default=0.5, help="default lam: 0.5")
group.add_argument("-ma",type=float, default=-0.2, help="default mass: -0.2")
group = parser.add_argument_group('Model parameters')
group.add_argument("-bs", type=int, default=32, help="default batch size: 32")
group.add_argument("-sbs", type=int, default=1, help="default super batch size: 1")
group.add_argument("-nl", type=int, default=1, help="default number of layers: 1")
group.add_argument("-w", type=int, default=8, help="default width of layers: 8")
group.add_argument("-lr", type=float, default=0.0001, help="default learning rate: 0.0001")
#group.add_argument("-sp", type=int, default=10, help="save period")
#group.add_argument("-op", type=str, default="optim.SGD", help="default optimizer: SGD")
group.add_argument("-e", type=int, default=1000, help="default epochs: 1000")
args = parser.parse_args()
if args.folder is None:
    rootFolder = './models/L_'+str(args.L)+"_l_"+str(args.la)+"_m_"+str(args.ma)+"_bs_"+str(args.bs)+"_sbs_"+str(args.sbs)+"_nl_"+str(args.nl)+"_w_"+str(args.w)+"_lr_"+str(args.lr)+"_e_"+str(args.e)+"/"        
    print("No specified saving path, using",rootFolder)
else:
    rootFolder = args.folder
if rootFolder[-1] != '/':
    rootFolder += '/'
utils.createWorkSpace(rootFolder)
#file = args.load
#if(args.load=="no-load"):
if args.load:
    with h5py.File(rootFolder+"/parameters.hdf5","r") as f:
        load_flag = False
        cuda = int(np.array(f["cuda"]))
        L = int(np.array(f["L"]))
        la = float(np.array(f["la"]))
        ma = float(np.array(f["ma"]))
        bs = int(np.array(f["bs"]))
        sbs = int(np.array(f["sbs"]))
        nl = int(np.array(f["nl"]))
        w = int(np.array(f["w"]))
        lr = float(np.array(f["lr"]))
        #sp = int(np.array(f["sp"]))
        #op = str(np.array(file["op"]))
        e = int(np.array(f["e"]))
else:
    load_flag = True
    cuda = args.cuda
    L = args.L
    la = args.la
    ma = args.ma
    bs = args.bs
    sbs = args.sbs
    nl = args.nl
    w = args.w
    lr = args.lr
    #sp = args.sp
    #op = args.op
    e = args.e
    with h5py.File(rootFolder+"parameters.hdf5","w") as f:
        f.create_dataset("cuda",data=args.cuda)
        f.create_dataset("L",data=args.L)
        f.create_dataset("la",data=args.la)
        f.create_dataset("ma",data=args.ma)
        f.create_dataset("bs",data=args.bs)
        f.create_dataset("sbs",data=args.sbs)
        f.create_dataset("nl",data=args.nl)
        f.create_dataset("w",data=args.w)
        f.create_dataset("lr",data=args.lr)
        #f.create_dataset("sp",data=args.sp)
        f.create_dataset("e",data=args.e)



''' 
if args.load:
    import os
    import glob
    name = max(glob.iglob(rootFolder+'savings/*.saving'), key=os.path.getctime)
    print("load saving at "+name)
    saved = tr.load(name)
    mg.load(saved)
'''


device = tr.device("cpu" if cuda<0 else "cuda:"+str(cuda))
print(f"Using {device} device")


L=L
V=L*L
batch_size=bs
super_batch_size = sbs
lam = la
mass= ma
o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
phi = o.hotStart()
#set up a prior
normal = distributions.Normal(tr.zeros(V).to(device),tr.ones(V).to(device))
prior= distributions.Independent(normal, 1)
width=w
Nlayers=nl
epochs_list=e
#saveSteps=sp
bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
mg = m.MGflow([L,L],bij,m.RGlayer("average"),prior)
#print("The flow Model: ", mg)
mg = mg.to(device)
#if(load_flag):
if(args.load):
    import os
    import glob
    name = max(glob.iglob(rootFolder+'model/*.model'), key=os.path.getctime)
    print("load saving at "+name)
    mg.load_state_dict(tr.load(name))
    mg.eval()
print(mg)
c=0
for tt in mg.parameters():
    #print(tt.shape)
    if tt.requires_grad==True :
        c+=tt.numel()
print("Device",device,"for",e,"epochs")        
print("QFT for: L:",L,",lam:",lam,",mass:",mass)
doc = open(rootFolder+"MGflow_L_"+str(L)
         +"_l_"+str(lam)
         +"_m_"+str(mass)
         +"_par_"+str(c)
         +"_bs_"+str(batch_size)
         +"_sbs_"+str(super_batch_size)                        
         +"_nl_"+str(Nlayers)
         +"_w_"+str(width)
         +"_lr"+str(lr)
         +"_e_"+str(e)+".txt", "w")
doc.write("epoch\tloss\tc\tbatch_size\tsuper_batch_size\tnlayers\twidth\tlearning_rate\tmax_action_diff\tmin_action_diff\tmean_action_diff\tstd_action_diff\tmean_re-weighting_factor\tstd_re-weighting_factor\ttime_execution\n")
print("Total parameters:",c,",batch_size=",batch_size,",super_batch_size=",super_batch_size,",nlayers=",Nlayers,",width=",width,",learning_rate=",lr,",epochs=",e)      

savePath=None
save=True
if savePath is None:
    savePath = "./models/"
learning_rate = lr
optimizer = tr.optim.SGD([p for p in mg.parameters() if p.requires_grad==True], lr=learning_rate)
#optimizer = tr.optim.Adam([p for p in mg.parameters() if p.requires_grad==True], lr=learning_rate)
loss_history = []
for t in range(e):   
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
    loss_history.append(loss.cpu().detach().numpy())
    #print(loss_history[-1])
    if t % 10 == 0:
        #print(z.shape)
        print('iter %s:' % t, 'loss = %.3f' % loss)
        with h5py.File(rootFolder+"records/"+"TorchQFT_"+"L_"+str(L)+"_l_"+str(lam)+"_m_"+str(mass)+"_w_"+str(width)+"_n_"+str(Nlayers)+"_e_"+str(t)+".h5","w") as f:  
            f.create_dataset("loss_history",data=loss.cpu().detach().numpy())

x=mg.sample(1024)
diff = (o.action(x)+mg.log_prob(x)).detach().cpu()
m_diff = diff.mean().cpu()
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
print("execution time=%.2f sec" % (time.time() - start_time))
doc.write(str(t)+"\t"+str(loss.cpu().detach().numpy())+"\t"+str(c)+"\t"+str(batch_size)+"\t"+str(super_batch_size)+"\t"+str(Nlayers)+"\t"+str(width)+"\t"+str(learning_rate)+"\t"+str(tr.max(diff.abs()).numpy())+"\t"+str(tr.min(diff.abs()).numpy())+"\t"+str(m_diff.detach().numpy())+"\t"+str(diff.std().numpy())+"\t"+str(w.mean().numpy())+"\t"+str(w.std().numpy())+"\t"+str((time.time() - start_time))+"\n")

doc.close()


plt.plot(np.arange(len(loss_history)),loss_history)
plt.xlabel("epoch")
plt.ylabel("KL-divergence")
title = "L_"+str(L)+"_batch_"+str(batch_size)+"_lr_"+str(learning_rate)
plt.title("Training history of MG model "+title)
#plt.show()
title = "L_"+str(L)+"_batch_"+str(batch_size)+"_lr_"+str(learning_rate)
plt.savefig(rootFolder+'plot/'+"mg_train_"+title+".pdf")
plt.savefig(rootFolder+'plot/'+"mg_train_"+title+".png")
#plt.show()
plt.close()

logbins = np.logspace(np.log10(1e-3),np.log10(1e3),int(w.shape[0]/10))
plt.hist(w,bins=logbins)
plt.xscale('log')
plt.ylabel("Frequency")
plt.xlabel("Number of samples")
plt.savefig(rootFolder+'plot/'+"mg_rw_"+title+".pdf")
plt.savefig(rootFolder+'plot/'+"mg_rw_"+title+".png")
#plt.show()
plt.close()

plt.hist(diff.detach(),bins=int(w.shape[0]/10))
plt.ylabel("Frequency")
plt.xlabel("Number of samples")
plt.savefig(rootFolder+'plot/'+"mg_ds_"+title+".pdf")
plt.savefig(rootFolder+'plot/'+"mg_ds_"+title+".png")
#plt.show()
#save the model
plt.close()

if args.name is None:
    #name = "TorchQFT"+'_nl'+str(nl)+'_w'+str(w)+'_Ising'
    name= "TorchQFT_"+"L_"+str(L)+"_l_"+str(lam)+"_m_"+str(mass)+"_w_"+str(width)+"_n_"+str(Nlayers)
else:
    name = args.name

if(not args.load):
    name= "TorchQFT_"+"L_"+str(L)+"_l_"+str(lam)+"_m_"+str(mass)+"_w_"+str(width)+"_n_"+str(Nlayers)
tr.save(mg.state_dict(), rootFolder+'model/'+name+'.model')
#mg.save(mg, "model.h5")



#keepSavings = 3
''' 
def cleanSaving(epoch):
    if epoch >= keepSavings*saveSteps:
        cmd =["rm","-rf",savePath+"savings/"+flow.name+"Saving_epoch"+str(epoch-keepSavings*saveSteps)+".saving"]
        subprocess.check_call(cmd)
        cmd =["rm","-rf",savePath+"records/"+flow.name+"Record_epoch"+str(epoch-keepSavings*saveSteps)+".hdf5"]
        subprocess.check_call(cmd)
'''
