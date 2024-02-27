#Run command
#python3 -m torch.distributed.run --standalone --nproc_per_node=8 train_stacked.py

#Run&Load command
#python3 -m torch.distributed.run --standalone --nproc_per_node=8 train_stacked.py -f sm_phi4_32_m-0.5_l1.9_w_8_l_2_st_1.dict -e 1000

#run
#sudo rmmod nvidia_uvm
#sudo modprobe nvidia_uvm
#which helps Ubuntu system after it was suspended.


from ast import arg
import numpy as np
import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter
import phi4_mg as m
import phi4 as p
import integrators as i
import update as u
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
import time
from stacked_model import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os  
import os.path


parser = argparse.ArgumentParser()
parser.add_argument('-f' , default='no-load')
parser.add_argument('-d' , type=int  , default=1   )
parser.add_argument('-e' , type=int  , default=1000)
parser.add_argument('-L' , type=int  , default=16  )
parser.add_argument('-m' , type=float, default=-0.5)
parser.add_argument('-g' , type=float, default=1.1 )
parser.add_argument('-b' , type=int  , default=128 )
parser.add_argument('-nb', type=int  , default=1   )
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-w' , type=int  , default=8   )
parser.add_argument('-nl', type=int  , default=2   )
parser.add_argument('-sb', type=int  , default=16  )
parser.add_argument('-se', type=int  , default=100 )
#parser.add_argument('-le', type=int  , default=0 )

args = parser.parse_args()

''' 
path = './LatticeQCD_Results/' 
try:  
    os.mkdir(path)  
except OSError as error:  
    print(error)   
'''

file = args.f
if(args.f == "no-load"):
    load_flag = False
    #file = path
    print("No specified saving path, using",file)
else:
    load_flag = True
    file = args.f


dist.init_process_group("nccl")
rank = dist.get_rank()
num_gpus = tr.cuda.device_count()
device_id = int(os.environ['ROCM_VISIBLE_DEVICES'])
print(f"Start running basic DDP example on rank {rank} on device {device_id} out of {num_gpus}.")


depth = args.d
epochs = args.e
L=args.L
mass=args.m
lam =args.g
batch_size=args.b
number_of_batches=args.nb
learning_rate=args.lr
width=args.w
number_of_layers=args.nl
super_batch=args.sb
save_every=args.se
#last_epoch=args.le


V=L*L
o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
phi = o.hotStart()


#set up a prior
normal = distributions.Normal(tr.zeros(V,device=device_id),tr.ones(V,device=device_id))
prior= distributions.Independent(normal, 1)


bij = lambda: m.FlowBijector(Nlayers=number_of_layers,width=width)
mg = lambda : m.MGflow([L,L],bij,m.RGlayer("average"),prior)
models = []


print("Initializing ",depth," stages")
for d in range(depth):
    models.append(mg())


#model = ToyModel().to(device_id)
#ddp_model = DDP(model, device_ids=[device_id])


sm = SuperModel(models,target=o.action).to(device_id)
sm = DDP(sm, device_ids=[device_id])


c=0
for tt in sm.parameters():
    #print(tt.shape)
    if tt.requires_grad==True :
        c+=tt.numel()
print("parameter count: ",c)


#tag = "L"+str(L)+"_m"+str(mass)+"_l"+str(lam)+"_w"+str(width)+"_nl"+str(number_of_layers)+"_st"+str(depth)+"_bs"+str(batch_size)+"_sb"+str(super_batch)+"_e"+str(epochs)+"_se"+str(save_every)+"_lr"+str(learning_rate)
tag = "L"+str(L)+"_m"+str(mass)+"_l"+str(lam)+"_w"+str(width)+"_nl"+str(number_of_layers)

path = 'sm_phi4_'+tag+'/'
try:  
    os.mkdir(path)  
except OSError as error:  
    print(error)   


last_epoch=0

if(load_flag):
    checkpoint = tr.load(file)
    sm.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer_state=checkpoint['optimizer_state_dict']
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    #sm.module.load_state_dict(tr.load(file))
    sm.eval()
else:
    optimizer_state=None


#documentTraining.write(str(device_id)+"\t"+str(epochs)+"\t"+str(saveEvery)+"\t"+str(lattice)+"\t"+str(lam)+"\t"+str(mass)+"\t"+str(depth)+"\t"+str(width)+"\t"+str(nLayers)+"\t"+str(batchSize)+"\t"+str(learningRate)+"\t"+str(c))
#txt_training = open("MG_flow_training_per_Epoch.txt", "w")
#txt_training.write("Device\tEpochs\tSave_Every\tLattice\tGamma\tMass\tDepth\tWidth\tNumber_of_Layers\tBatch_Size\tLearning_Rate\tParameters\tParams\tTime\tMax_Action_Diff\tMin_Action_Diff\tMean_Action_Diff\tStd_Action_Diff\tMean_ReWeighting_Factor\tStd_ReWeighting_Factor\tMean_Minus_Std\tMean_Plus_Std\tLoss_KL_Diverge\tESS\n")



if os.path.isfile(path+"/sm_phi4_"+tag+"_training_validation_steps.txt"):
    txt_training_validation_steps = open(path+"/sm_phi4_"+tag+"_training_validation_steps.txt", "a")
else:
    txt_training_validation_steps = open(path+"/sm_phi4_"+tag+"_training_validation_steps.txt", "a")
    txt_training_validation_steps.write("Epoch\tTraining_Max_Action_Diff\tTraining_Min_Action_Diff\tTraining_Mean_Action_Diff\tTraining_Std_Action_Diff\tTraining_Mean_Re_Weighting_Factor\tTraining_Std_Re_Weighting_Factor\tTraining_Mean_Minus_Std\tTraining_Mean_Plus_Std\tTraining_Loss_KL_diverge\tTraining_Loss_Ess\tValidation_Max_Action_Diff\tValidation_Min_Action_Diff\tValidation_Mean_Action_Diff\tValidation_Std_Action_Diff\tValidation_Mean_Re_Weighting_Factor\tValidation_Std_Re_Weighting_Factor\tValidation_Mean_Minus_Std\tValidation_Mean_Plus_Std\tValidation_Loss_KL_diverge\tValidation_Loss_Ess\n")

for b in batch_size*(2**np.arange(number_of_batches)):
    
     print("Running with batch_size = ", b, " and learning rate= ", learning_rate)
     #loss_hist, std_hist, mean_hist, ess_hist, optimizer, loss = trainSM(sm, tag, txt_training_steps, levels=[], epochs=epochs, batch_size=b, super_batch_size= super_batch, learning_rate=learning_rate, save_every=save_every)
     loss_training_history, loss_validation_history, std_training_history, std_validation_history, ess_training_history, ess_validation_history,  optimizer, training_loss, validation_loss = trainSM(sm, tag, path, txt_training_validation_steps, levels=[], rank=rank, epochs=epochs, last_epoch=last_epoch,batch_size=b, super_batch_size= super_batch, learning_rate=learning_rate, save_every=save_every, optimizer_state=optimizer_state)
     batch_size = (b*super_batch)

     last_epoch+=epochs

     #if (rank==0):
        #tt = tag+"_b"+str(batch_size)
       #tt = tag
       #plot_loss(loss_training_history, loss_validation_history, tt, path)
       #plot_std(std_training_history, std_validation_history,tt,save_every, path)
       #plot_ess(ess_training_history,ess_validation_history,tt,save_every, path)
       # testing(b,b*super_batch,tt,sm.module,epochs, path)


if(not load_flag):
    #file = "sm_phi4_"+tag+".dict"
    file = path+"sm_phi4_"+tag+".pt"


if rank==0:
    #tr.save(sm.module.state_dict(), file)
    tr.save({'epoch':last_epoch, 'model_state_dict':sm.module.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),'loss':training_loss},file)

dist.destroy_process_group()
