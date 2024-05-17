from importlib.machinery import SourceFileLoader
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
import os
import torch.distributed as dist
import psutil
import torchvision
#from torchview import draw_graph
import torchsummary
#import visualkeras
#from torchviz import make_dot, make_dot_from_trace
process = psutil.Process()


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
        #random_seed=42
        #tr.cuda.manual_seed(random_seed)
        z = self.prior_sample(batchSize)
        x = self.forward(z)
        return x


    # generate a sample from the prior
    def prior_sample(self,batch_size):
        #random_seed=42
        #tr.cuda.manual_seed(random_seed)
        return self.prior.sample((batch_size,1)).reshape(batch_size,self.size[0],self.size[1])


    def loss(self,x):
        return (self.log_prob(x)+self.target(x)).mean()


    def diff(self,x):
        return self.log_prob(x)+self.target(x)


class triviality():
    def __init__(self,mm,batch_size=1,device="cpu",dtype=tr.float32): 
        self.model = mm
        self.Bs=batch_size
        self.device=device
        self.dtype=dtype

    
    def action(self,z):
        x=self.model.forward(z)
        _,J = self.model.backward(x)
        return self.model.target(x) + J
        #return self.model.log_prob(x) + self.model.target(x)


    #approximate the force by just a quadratic potential
    #if trivialization is exact for phi^2 this is exact
    def force(self,z):
        return -z


    def refreshP(self):
        P = tr.normal(0.0,1.0,[self.Bs,self.model.size[0],self.model.size[1]],dtype=self.dtype,device=self.device)
        return P


    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    

    def kinetic(self,P):
        return tr.sum(P*P,dim=(1,2))/2.0


def trainSM(SuperM, tag, path, txt_training_validation_steps, levels=[], rank=0, epochs=100, batch_size=16, super_batch_size=1, learning_rate=1.0e-4, beta=0.9, save_every=100, c=None, optimizer_state=None):

    #print("rank", rank)
    tic = time.perf_counter()
    params = []

    if levels==[] :
        params = [p for p in SuperM.parameters() if p.requires_grad==True]
    else:
        for l in levels:
            params.extend([p for p in SuperM.models[l].parameters() if p.requires_grad==True])
    print("Number of parameters to train is: ",len(params))
    #print("Memory used {:.0f}".format(process.memory_info().rss/1000000,"MB"))
    #model_graph = draw_graph(SuperM, input_size=(4,4), graph_name='SuperM', roll=True)
    #model_graph.visual_graph
    #print(SuperM)
    #torchsummary.summary(SuperM, input_size=(64,64))
    #visualkeras.layered_view(SuperM).show()
    optimizer = tr.optim.Adam(params, lr=learning_rate, betas=(beta, 0.999))
    
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)


    loss_training_history = []
    std_training_history = []
    mean_training_history = []
    ess_training_history = []

    loss_validation_history = []
    std_validation_history = []
    mean_validation_history = []
    ess_validation_history = []

    #tic=time.perf_counter()
    
    #diff=0
    
    pbar = tqdm(range(epochs))

    for t in pbar:
        
        loss = 0.0
        loss_validation=0.0
        optimizer.zero_grad()

        for b in range(0,super_batch_size):
            
            z = SuperM.module.prior_sample(batch_size)
            x = SuperM(z) 
            #print("Training_set_size:",x.numel())

            tloss = SuperM.module.loss(x)/super_batch_size
            tloss.backward()
            loss+=tloss

            x_validation=SuperM.module.sample(batch_size)
            #print("Validation_set_size:",x_validation.numel())
            vloss = SuperM.module.loss(x_validation)/super_batch_size
            vloss.backward()
            loss_validation+=vloss

        dist.all_reduce(loss)
        dist.all_reduce(loss_validation)

        optimizer.step()

        loss_training_history.append(loss.cpu().detach().numpy())
        loss_validation_history.append(loss_validation.cpu().detach().numpy())

        diff_training = SuperM.module.diff(x).detach()
        diff_validation = SuperM.module.diff(x_validation).detach()

        if rank==0:

            PATH = path+"sm_phi4_"+tag+"_checkpoint_"+str(t)+".pt"
            pbar.set_postfix({'training_loss': loss.cpu().detach().numpy(), ' | validation_loss': loss_validation.cpu().detach().numpy()})
            #pbar.set_postfix({'loss': loss.cpu().detach().numpy()})

            m_diff_training = diff_training.mean()
            m_diff_training = m_diff_training.cpu()     
            diff_training -= m_diff_training
            diff_training = diff_training.cpu()
            foo_training = tr.exp(-diff_training)
            w_training = foo_training/tr.mean(foo_training)
            ess_training = (foo_training.mean())**2/(foo_training*foo_training).mean()
            mean_training_minus_std = m_diff_training.detach().numpy()-diff_training.std().numpy()
            mean_training_plus_std = m_diff_training.detach().numpy()+diff_training.std().numpy()
            ''' 
            print("training_max_action_diff: ", tr.max(diff_training.abs()).numpy())
            print("training_min_action_diff: ", tr.min(diff_training.abs()).numpy())
            print("training_mean_action_diff: ", m_diff_training.detach().numpy())
            print("training_std_action_diff: ", diff_training.std().numpy())
            print("training_mean_re_weighting_factor: ", w_training.mean().numpy())
            print("training_std_re_weighting_factor: ", w_training.std().numpy())
            print("training_ess: ", ess_training.numpy(),"\n")
            '''

            m_diff_validation = diff_validation.mean()
            m_diff_validation = m_diff_validation.cpu()     
            diff_validation -= m_diff_validation
            diff_validation = diff_validation.cpu()
            foo_validation = tr.exp(-diff_validation)
            w_validation = foo_validation/tr.mean(foo_validation)
            ess_validation = (foo_validation.mean())**2/(foo_validation*foo_validation).mean()
            mean_validation_minus_std=m_diff_validation.detach().numpy()-diff_validation.std().numpy()
            mean_validation_plus_std=m_diff_validation.detach().numpy()+diff_validation.std().numpy()
            '''
            print("validation_max_action_diff: ", tr.max(diff_validation.abs()).numpy())
            print("validation_min_action_diff: ", tr.min(diff_validation.abs()).numpy())
            print("validation_mean_action_diff: ", m_diff_validation.detach().numpy())
            print("validation_std_action_diff: ", diff_validation.std().numpy())
            print("validation_mean_re_weighting_factor: ", w_validation.mean().numpy())
            print("validation_std_re_weighting_factor: ", w_validation.std().numpy())
            print("validation_ess: ", ess_validation.numpy())
            '''
            
            txt_training_validation_steps.write(str(t)+"\t"+str(tr.max(diff_training.abs()).numpy())+"\t"+str(tr.min(diff_training.abs()).numpy())+"\t"+str(m_diff_training.detach().numpy())+"\t"+str(diff_training.std().numpy())+"\t"+str(w_training.mean().numpy())+"\t"+str(w_training.std().numpy())+"\t"+str(mean_training_minus_std)+"\t"+str(mean_training_plus_std)+"\t"+str(loss.cpu().detach().numpy())+"\t"+str(ess_training.numpy())+"\t"+str(tr.max(diff_validation.abs()).numpy())+"\t"+str(tr.min(diff_validation.abs()).numpy())+"\t"+str(m_diff_validation.detach().numpy())+"\t"+str(diff_validation.std().numpy())+"\t"+str(w_validation.mean().numpy())+"\t"+str(w_validation.std().numpy())+"\t"+str(mean_validation_minus_std)+"\t"+str(mean_validation_plus_std)+"\t"+str(loss.cpu().detach().numpy())+"\t"+str(ess_validation.numpy())+"\n")
            txt_training_validation_steps.flush()
            os.fsync(txt_training_validation_steps.fileno())
            
            mean_training_history.append(m_diff_training.detach().numpy())
            std_training_history.append(diff_training.std().numpy())
            ess_training_history.append(ess_training.numpy())
            
            mean_validation_history.append(m_diff_validation.detach().numpy())
            std_validation_history.append(diff_validation.std().numpy())
            ess_validation_history.append(ess_validation.numpy())

            if t % save_every == 0:
                tr.save(SuperM.module.state_dict(), PATH)

                #tr.save({'epoch':epochs,'model_state_dict':SuperM.module.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':loss,},PATH)

    if rank==0:


        tr.save({'epoch':epochs,'model_state_dict':SuperM.module.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':loss},PATH)


        m_diff_training = diff_training.mean()
        m_diff_training = m_diff_training.cpu()     
        diff_training -= m_diff_training
        diff_training = diff_training.cpu()
        foo_training = tr.exp(-diff_training)
        w_training = foo_training/tr.mean(foo_training)
        ess_training = (foo_training.mean())**2/(foo_training*foo_training).mean()
        mean_training_minus_std = m_diff_training.detach().numpy()-diff_training.std().numpy()
        mean_training_plus_std = m_diff_training.detach().numpy()+diff_training.std().numpy()
        print("training_max_action_diff:", tr.max(diff_training.abs()).numpy())
        print("training_min_action_diff:", tr.min(diff_training.abs()).numpy())
        print("training_mean_action_diff:", m_diff_training.detach().numpy())
        print("training_std_action_diff:", diff_training.std().numpy())
        print("training_mean_re_weighting_factor:", w_training.mean().numpy())
        print("training_std_re_weighting_factor:", w_training.std().numpy())
        print("training_ess:", ess_training.numpy(),"\n")

        m_diff_validation = diff_validation.mean()
        m_diff_validation = m_diff_validation.cpu()     
        diff_validation -= m_diff_validation
        diff_validation = diff_validation.cpu()
        foo_validation = tr.exp(-diff_validation)
        w_validation = foo_validation/tr.mean(foo_validation)
        ess_validation = (foo_validation.mean())**2/(foo_validation*foo_validation).mean()
        mean_validation_minus_std=m_diff_validation.detach().numpy()-diff_validation.std().numpy()
        mean_validation_plus_std=m_diff_validation.detach().numpy()+diff_validation.std().numpy()
        print("validation_max_action_diff:", tr.max(diff_validation.abs()).numpy())
        print("validation_min_action_diff:", tr.min(diff_validation.abs()).numpy())
        print("validation_mean_action_diff:", m_diff_validation.detach().numpy())
        print("validation_std_action_diff:", diff_validation.std().numpy())
        print("validation_mean_re_weighting_factor:", w_validation.mean().numpy())
        print("validation_std_re_weighting_factor:", w_validation.std().numpy())
        print("validation_ess: ", ess_validation.numpy())

        txt_training_validation_steps.write(str(t+1)+"\t"+str(tr.max(diff_training.abs()).numpy())+"\t"+str(tr.min(diff_training.abs()).numpy())+"\t"+str(m_diff_training.detach().numpy())+"\t"+str(diff_training.std().numpy())+"\t"+str(w_training.mean().numpy())+"\t"+str(w_training.std().numpy())+"\t"+str(mean_training_minus_std)+"\t"+str(mean_training_plus_std)+"\t"+str(loss.cpu().detach().numpy())+"\t"+str(ess_training.numpy())+"\t"+str(tr.max(diff_validation.abs()).numpy())+"\t"+str(tr.min(diff_validation.abs()).numpy())+"\t"+str(m_diff_validation.detach().numpy())+"\t"+str(diff_validation.std().numpy())+"\t"+str(w_validation.mean().numpy())+"\t"+str(w_validation.std().numpy())+"\t"+str(mean_validation_minus_std)+"\t"+str(mean_validation_plus_std)+"\t"+str(loss.cpu().detach().numpy())+"\t"+str(ess_validation.numpy())+"\n")
        txt_training_validation_steps.flush()
        os.fsync(txt_training_validation_steps.fileno())

        mean_training_history.append(m_diff_training.detach().numpy())
        std_training_history.append(diff_training.std().numpy())
        ess_training_history.append(ess_training.numpy())
        
        mean_validation_history.append(m_diff_validation.detach().numpy())
        std_validation_history.append(diff_validation.std().numpy())
        ess_validation_history.append(ess_validation.numpy())

    toc = time.perf_counter()
    print(f"Time {(toc - tic):0.4f} seconds","\n")

    return loss_training_history, loss_validation_history, std_training_history, std_validation_history, ess_training_history, ess_validation_history, optimizer, loss, loss_validation
    #return loss_training_history, std_training_history, mean_training_history, ess_training_history, optimizer, loss

def plot_avg(path,L,lam,mass, batch_size, super_batch_size, mm):
    # magnetic moment = \integral_x mean(x) exp(-S(x)) p(x) dx / \integral_x exp(-S(x)) p(x) dx
    #           \approx [ sum_x mean(x) exp(-S(x)) ] / [ sum_x exp(-S(x)) ], where x drawn with prob p(x)
    #                 = [ sum_x exp(-S(x) - logZ + log(mean(x))) ], where logZ = log [sum_x exp(-S(x)) ]

    o = p.phi4([L,L],lam,mass,batch_size=batch_size)
    V=L*L
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('magnetic moment')
    mm_pos_orig = tr.Tensor([])
    mm_neg_orig = tr.Tensor([])
    for b in range(super_batch_size):
        x = o.hotStart().detach()
        action = o.action(x)
        logZ = tr.logsumexp(-action, 0)
        mean = x.mean(dim=(1,2))
        neg_mean = tr.argwhere(tr.minimum(mean, tr.Tensor([0.0])))
        pos_mean = tr.argwhere(tr.maximum(mean, tr.Tensor([0.0])))
        magnetic_moment += tr.exp(tr.logsumexp(-action[pos_mean] - logZ + tr.log(mean[pos_mean]), 0)) - tr.exp(tr.logsumexp(-action[neg_mean] - logZ + tr.log(-mean[neg_mean]), 0)))
        mm_pos_orig = tr.cat((mm_pos_orig, tr.squeeze(-action[pos_mean] - logZ + tr.log(mean[pos_mean]))))
        mm_neg_orig = tr.cat((mm_neg_orig, tr.squeeze(-action[neg_mean] - logZ + tr.log(-mean[neg_mean]))))

    print("magnetic moment", magnetic_moment)
    ax[0,0].hist(mm_pos_orig, bins=100)
    ax[0,1].hist(mm_neg_orig, bins=100)
    ax[0,0].set(ylabel="frequency")

    # magnetic moment = \integral_x mean(x) exp(-S(x)) p(x) dx / \integral_x exp(-S(x)) p(x) dx
    #           \approx [ sum_x mean(x) exp(-S(x)) ] / [ sum_x exp(-S(x)) ], where x drawn with prob p(x)
    #                 = [ sum_x exp(-S(x) - logZ + log(mean(x))) ], where logZ = log [sum_x exp(-S(x)) ]

    mm_pos_orig = tr.Tensor([])
    mm_neg_orig = tr.Tensor([])
    for b in range(super_batch_size):
        x = mm.sample(batch_size).detach().to(device="cpu")
        action = mm.diff(x).detach().to(device="cpu")
        logZ = tr.logsumexp(-action, 0).to(device="cpu")
        mean = x.mean(dim=(1,2)).to(device="cpu")
        neg_mean = tr.argwhere(tr.minimum(mean, tr.Tensor([0.0])))
        pos_mean = tr.argwhere(tr.maximum(mean, tr.Tensor([0.0])))
        magnetic_moment += tr.exp(tr.logsumexp(-action[pos_mean] - logZ + tr.log(mean[pos_mean]), 0)) - tr.exp(tr.logsumexp(-action[neg_mean] - logZ + tr.log(-mean[neg_mean]), 0)))
        mm_pos_orig = tr.cat((mm_pos_orig, tr.squeeze(-action[pos_mean] - logZ + tr.log(mean[pos_mean]))))
        mm_neg_orig = tr.cat((mm_neg_orig, tr.squeeze(-action[neg_mean] - logZ + tr.log(-mean[neg_mean]))))

    print("magnetic moment with model", magnetic_moment)
    ax[1,0].hist(mm_pos_orig, bins=100)
    ax[1,1].hist(mm_neg_orig, bins=100)
    ax[1,0].set(ylabel="frequency")

    fig.savefig(path+"sm_magmom.pdf", dpi=300)

def plot_loss(lh,vh,title, path):
    plt.plot(np.arange(len(lh)),lh,label='Training')
    plt.plot(np.arange(len(vh)),vh,label='Validation')
    plt.xlabel("epoch")
    plt.ylabel("KL-divergence")
    plt.title("Training-Validation KL-diverge history of MG super model")
    plt.legend(fancybox=True, loc='best', prop={'size': 7}, framealpha=1)
    #plt.show()
    plt.savefig(path+"sm_training_validation_KL_"+title+".pdf", dpi=300)
    plt.close()


def plot_std(stdh,stdvh,title,save_every, path):
    x_training=np.arange(len(stdh))
    #x_training=x_training*save_every
    y_training=stdh
    y_validation=stdvh
    plt.plot(x_training,y_training, label='Training')
    plt.plot(x_training,y_validation, label='Validation')
    plt.xlabel("epoch")
    plt.ylabel("std")
    plt.title("Training-Validation STD history of MG super model")
    plt.legend(fancybox=True, loc='best', prop={'size': 7}, framealpha=1)
    plt.savefig(path+"sm_training_validation_STD_"+title+".pdf", dpi=300)
    plt.close()

    plt.yscale('log')
    plt.plot(x_training,y_training, label='Training')
    plt.plot(x_training,y_validation, label='Validation')
    plt.xlabel("epoch")
    plt.ylabel("std")
    plt.title("Training-Validation Log-std history of MG super model")
    plt.legend(fancybox=True, loc='best', prop={'size': 7}, framealpha=1)
    plt.savefig(path+"sm_training_validation_LOG_STD_"+title+".pdf", dpi=300)
    plt.close()


def plot_ess(essh,essvh,title,save_every, path):
    x_training=np.arange(len(essh))
    #x_training=x_training*save_every
    y_training=essh
    y_validation=essvh
    plt.plot(x_training,y_training, label='Training')
    plt.plot(x_training,y_validation, label='Validation')
    plt.xlabel("epoch")
    plt.ylabel("ess")
    plt.title("Training-Validation ESS history of MG super model")
    plt.legend(fancybox=True, loc='best', prop={'size': 7}, framealpha=1)
    plt.savefig(path+"sm_training_validation_ESS_"+title+".pdf", dpi=300)
    plt.close()


def testing(batch_size,super_batch_size,title,mm,epochs, path):
    
    txt_testing_steps = open(path+"sm_phi4_"+title+"_testing_steps.txt", "w")
    txt_testing_steps.write("Epochs\tMax_Action_Diff\tMin_Action_Diff\tMean_Action_Diff\tStd_Action_Diff\tMean_ReWeighting_Factor\tStd_ReWeighting_Factor\tMean_Minus_Std\tMean_Plus_Std\tLoss_KL_diverge\tLoss_Ess\n")
    
    x=mm.sample(batch_size)
    print("Test_set_size:",x.numel())
     #torch.Size([3,5]).numel() 
    diff = mm.diff(x).detach()
    
    for b in range(1,super_batch_size):
        x=mm.sample(batch_size)
        diff = tr.cat((diff,mm.diff(x).detach()),0)           
    
    m_diff = diff.mean()
    m_diff=m_diff.cpu()     
    diff -= m_diff
    diff = diff.cpu()
    foo = tr.exp(-diff)
    w = foo/tr.mean(foo)
    ess = (foo.mean())**2/(foo*foo).mean()
    mean_test_minus_std=m_diff.detach().numpy()-diff.std().numpy()
    mean_test_plus_std=m_diff.detach().numpy()+diff.std().numpy()

    print("test_max_action_diff:", tr.max(diff.abs()).numpy())
    print("test_min_action_diff:", tr.min(diff.abs()).numpy())
    print("test_mean_action_diff:", m_diff.detach().numpy())
    print("test_std_action_diff:", diff.std().numpy())
    print("test_mean_re_weighting_factor:", w.mean().numpy())
    print("test_std_re_weighting_factor:", w.std().numpy())
    print("test_ess:", ess.numpy())

    txt_testing_steps.write(str(epochs)+"\t"+str(tr.max(diff.abs()).numpy())+"\t"+str(tr.min(diff.abs()).numpy())+"\t"+str(m_diff.detach().numpy())+"\t"+str(diff.std().numpy())+"\t"+str(w.mean().numpy())+"\t"+str(w.std().numpy())+"\t"+str(mean_test_minus_std)+"\t"+str(mean_test_plus_std)+"\t"+str(ess.numpy())+"\n")
    txt_testing_steps.flush()
    os.fsync(txt_testing_steps.fileno())

    logbins = np.logspace(np.log10(1e-3),np.log10(1e3),int(w.shape[0]/10))
    _=plt.hist(w,bins=logbins)
    plt.xscale('log')
    plt.title('Test set reweighting factor')
    plt.savefig(path+"sm_testing_RW_"+title+".pdf", dpi=300)
    #plt.show()
    plt.close()

    _=plt.hist(diff.detach(),bins=int(w.shape[0]/10))
    plt.title('Test set ΔS distribution')
    plt.savefig(path+"sm_testing_DS_"+title+".pdf", dpi=300)
    #plt.show()
    plt.close()


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


def test_hmc(file,depth=2):
    import time

    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    L=16
    batch_size=128
    V=L*L
    lam =1.1
    mass= -0.5
    o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
    phi = o.hotStart()
    #set up a prior
    normal = distributions.Normal(tr.zeros(V,device=device),tr.ones(V,device=device))
    prior= distributions.Independent(normal, 1)

    width=32
    Nlayers=2
    bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
    mg = lambda : m.MGflow([L,L],bij,m.RGlayer("average"),prior)
    models = []
    print("Initializing ",depth," stages")
    for d in range(depth):
        models.append(mg())
        
    sm = SuperModel(models,target =o.action )

    c=0
    for tt in sm.parameters():
        #print(tt.shape)
        if tt.requires_grad==True :
            c+=tt.numel()

    tag = str(L)+"_m"+str(mass)+"_l"+str(lam)+"_st_"+str(depth)
    sm.load_state_dict(tr.load(file))
    sm.eval()

    validate(batch_size,tag,sm)
    triv = triviality(sm,batch_size=batch_size)
    z = sm.prior_sample(batch_size)
    #m_action = triv.action(z)
    #p_action = - sm.prior.log_prob(z.flatten(start_dim=1))
    #diff = m_action - p_action
    #print(m_action)
    #print(diff - diff.mean())

    mn2 = i.minnorm2(triv.force,triv.evolveQ,6,1.0)
    
    hmc = u.hmc(T=triv,I=mn2,verbose=False)
    Nwarm=10
    Nskip=2
    Nmeas=1000
    tic=time.perf_counter()
    z = hmc.evolve(z,Nwarm)
    toc=time.perf_counter()
    print(f"time {(toc - tic)/Nwarm:0.4f} seconds per HMC trajecrory")
    print("Acceptance rate: ",hmc.calc_Acceptance())

    for k in range(Nmeas):
        av_z = tr.mean(z,dim=(1,2))
        print(k," av_z",av_z.mean().detach().numpy(), " std_z: ",av_z.std().detach().numpy()," full std: ",z.std().detach().numpy())
        tic=time.perf_counter()
        z = hmc.evolve(z,Nskip)
        toc=time.perf_counter()
        print(f"time {(toc - tic)/Nwarm:0.4f} seconds per HMC trajecrory","| Acceptance rate: ",hmc.calc_Acceptance())
    

def test_train(depth=1,epochs=100,load_flag=False,file="model.dict"):
    import time
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    L=16
    batch_size=128
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
    models = []
    print("Initializing ",depth," stages")
    for d in range(depth):
        models.append(mg())
        
    sm = SuperModel(models,target =o.action )

    c=0
    for tt in sm.parameters():
        #print(tt.shape)
        if tt.requires_grad==True :
            c+=tt.numel()

    tag = str(L)+"_m"+str(mass)+"_l"+str(lam)+"_st_"+str(depth)
    if(load_flag):
        sm.load_state_dict(tr.load(file))
        sm.eval()
    print("parameter count: ",c)
    
    for b in [4,8,16,32]:
        loss_hist=trainSM(sm,levels=[], epochs=epochs,batch_size=b,super_batch_size=1)
        tt = tag+"_b"+str(b)
        plot_loss(loss_hist,tt)
        validate(1024,tt,sm)

    if(not load_flag):
        file = "sm_phi4_"+str(L)+"_m"+str(mass)+"_l"+str(lam)+"_st_"+str(depth)+".dict"
    tr.save(sm.state_dict(), file)
   

def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='revers')
    parser.add_argument('-l', default='no-load')
    parser.add_argument('-d', type=int,default=1)
    parser.add_argument('-e', type=int,default=1000)
    
    args = parser.parse_args()
    if(args.t=='revers'):
        print("Testing reversibility")
        test_reversibility()
    elif(args.t=="train"):
        print("Testing MGflow training")
        if(not args.l=="no-load"):
            test_train(args.d,args.e,True,args.l)
        else:
            test_train(args.d,args.e)
    elif(args.t=="hmc"):
        test_hmc(args.l,args.d)
    else:
        print("Nothing to test")

if __name__ == "__main__":
    main()
