#LIBRARIES
import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.init as init
from torch import distributions
from torch.nn.parameter import Parameter
import torch.optim as optim
import phi4_mg as m
import phi4 as p
import integrators as i
import update as u
import time
import matplotlib.pyplot as plt
import argparse
import sys
from argparse import ArgumentParser
import io
import time
import os
import gc

#TESTS FOR GPU VRAM ALLOCATION
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:22"
#PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:<value>
#PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:20"

#INITIALIZE TIME
start_time = time.time()
total_time =time.time()

#MAIN ARGUMENTS
parser = argparse.ArgumentParser(description="QFT version 1.0")
group = parser.add_argument_group('Initializing parameters')
group.add_argument("-cuda", type=int, default=-1, help="default device: cpu")
group.add_argument("-load", default='no-load', help="default folder: models")
group = parser.add_argument_group('Model parameters')
group.add_argument("-epochs", type=int, default=1001, help="default epochs: 1000")
group = parser.add_argument_group('Ising parameters')
group.add_argument("-L",type=int, default=16, help="default linear size: 16")
args = parser.parse_args()
file = args.load
if(args.cuda=="-1"):
    load_flag = False
    cuda = int(np.array(file["cuda"]))
    epochs = int(np.array(file["epochs"]))
    L = int(np.array(file["L"]))
else:
    load_flag = True
    cuda = args.cuda
    epochs = args.epochs
    L = args.L

#DEVICE INITIALIZATION
device = tr.device("cpu" if cuda<0 else "cuda:"+str(cuda))

#MODEL DEVELOPMENT
#LEARNING PROBLEM PARAMETERS
L=L
V=L*L
lam=0.5
mass=-0.2
#MODEL PARAMETERS
batch_size_list=[128, 256, 512, 1024]
super_batch_size_list=[64, 128, 256, 512]
Nlayers_list=[1, 2, 3, 4, 5]
width_list=[16, 32, 64, 128, 256]
learning_rate_list=[0.0001, 0.001, 0.01, 0.1]
optimizer_list=[optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta, optim.Adam, optim.Adamax, optim.NAdam]
epochs_list=epochs
#OTHER PARAMETERS
'''
momentum_list= [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
weight_list= [init.uniform_, init.normal_, init.zeros_, init.xavier_normal_, init.xavier_uniform_, init.kaiming_normal_, init.kaiming_uniform_]
activation_list=[nn.LeakyReLU, nn.Identity, nn.ReLU, nn.ELU, nn.ReLU6, nn.GELU, nn.Softplus, nn.Softsign, nn.Tanh, nn.Sigmoid, nn.Hardsigmoid]
weight_constraint_list= [1.0, 2.0, 3.0, 4.0, 5.0]
dropout_rate_list= [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_neurons_list= [1, 5, 10, 15, 20, 25, 30]
'''
#MODEL TRAINING
print("\nDevice",device,"for",epochs,"epochs")        
print("QFT for: L:",L,",lam:",lam,",mass:",mass)
f = open("MGflow_L_"+str(L)+"_lam_"+str(lam)+"_mass_"+str(mass)+"_epochs_"+str(epochs)+".txt", "w")
f.write("c\tbatch_size\tsuper_batch_size\tnlayers\twidth\tlearning_rate\toptimizer\tmax_action_diff\tmin_action_diff\tmean_action_diff\tstd_action_diff\tmean_re-weighting_factor\tstd_re-weighting_factor\ttime_execution\n")
for batch_size in batch_size_list:
    for super_batch_size in super_batch_size_list:
        for Nlayers in Nlayers_list:
            for width in width_list:
                for learning_rate in learning_rate_list:
                    for optimizer in optimizer_list:
                        o = p.phi4([L,L],lam,mass,batch_size=batch_size)
                        phi = o.hotStart()
                        normal = distributions.Normal(tr.zeros(V).to(device),tr.ones(V).to(device))
                        prior= distributions.Independent(normal, 1)
                        bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
                        mg = m.MGflow([L,L],bij,m.RGlayer("average"),prior)
                        mg = mg.to(device)
                        gc.collect()
                        tr.cuda.empty_cache()
                        if(load_flag):
                            mg.eval()
                        c=0
                        for tt in mg.parameters():
                            if tt.requires_grad==True :
                                c+=tt.numel()
                        print("\nTotal parameters:",c,",batch_size=",batch_size,",super_batch_size=",super_batch_size,",nlayers=",Nlayers,",learning_rate=",learning_rate,",width=",width,",optimizer=",optimizer.__name__)      
                        learning_rate=learning_rate
                        optimizer = optimizer([p for p in mg.parameters() if p.requires_grad==True], lr=learning_rate)
                        loss_history = []
                        for epochs in range(epochs_list):   
                            z = mg.prior_sample(batch_size)
                            x = mg(z) 
                            tloss = (mg.log_prob(x)+o.action(x)).mean()
                            for b in range(1,super_batch_size):
                                z = mg.prior_sample(batch_size)
                                x = mg(z)
                                tloss += (mg.log_prob(x)+o.action(x)).mean() 
                            loss = tloss/super_batch_size
                            optimizer.zero_grad()
                            loss.backward(retain_graph=True)
                            optimizer.step()
                            loss_history.append(loss.cpu().detach().numpy())
                            if epochs % 10 == 0:
                                print("Epoch "+str(epochs)+"/"+str(epochs_list)+" - loss: "+str(loss.cpu().detach().numpy()))
                        #MODEL TEST/VALIDATION
                        x=mg.sample(1024)
                        diff = (o.action(x)+mg.log_prob(x)).detach().cpu()
                        m_diff = diff.cpu().mean()
                        diff -= m_diff
                        foo = tr.exp(-diff)
                        w = foo/tr.mean(foo)
                        f.write(str(c)+"\t"+str(batch_size)+"\t"+str(super_batch_size)+"\t"+str(Nlayers)+"\t"+str(width)+"\t"+str(learning_rate)+"\t"+str(optimizer.__class__.__name__)+"\t"+str(tr.max(diff.abs()).numpy())+"\t"+str(tr.min(diff.abs()).numpy())+"\t"+str(m_diff.detach().numpy())+"\t"+str(diff.std().numpy())+"\t"+str(w.mean().numpy())+"\t"+str(w.std().numpy())+"\t"+str((time.time() - start_time))+"\n")
                        print("Model results: max_action_diff=",tr.max(diff.abs()).numpy(),",min_action_diff=", tr.min(diff.abs()).numpy(),",mean_action_diff=",m_diff.detach().numpy(),",std_action_diff=",diff.std().numpy(),",mean_re-weighting_factor=",w.mean().numpy(),",std_re-weighting_factor=",w.std().numpy(),",execution_time=%.2f sec" % (time.time() - start_time)+"\n")
                        start_time = time.time()

#END
f.close()
print("Total execution time: %.2f sec" % (time.time() - total_time)+"\n")


