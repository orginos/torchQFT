#!/usr/local/bin/python3
## JUST TESTING FUNCTIONALITY
## 
import numpy as np

import matplotlib.pyplot as plt
import torch as tr
import integrators as integ
import update as u

class hmc_sampler():
    def __init__(self, init_cnf,Nhmc=4,Nmd=5,integrator="minnorm2",action=lambda x: 0.5*tr.sum(x*x,dim=1) ,verbose=True) :
        self.state= init_cnf
        self.Nb   = init_cnf.shape[0] 
        self.action = action
        self.field_shape = init_cnf[0].shape
        self.shape = init_cnf.shape
        self.I = getattr(integ,integrator)(self.force,self.evolveQ,Nmd,1.0) #integrator
        self.Nhmc=Nhmc
        self.hmc = u.hmc(T=self,I=self.I,verbose=verbose)
    

    def set_action(self,action=lambda x: 0.5*tr.sum(x*x,dim=1),Nwarm=10):
        self.action = action
        self.warm_up(Nwarm)
        
    def force(self,x):
        x.requires_grad_(True)
        A = self.action(x)
        return -tr.autograd.grad(A,x,grad_outputs=tr.ones_like(A), create_graph=True)[0]


    def refreshP(self):
        P = tr.normal(0.0,1.0,self.shape)
        return P
    def evolveQ(self,dt,P,Q):
        return Q + dt*P

    def kinetic(self,P):
        return tr.einsum('b...,b...->b',P,P)/2.0 ;

    def warm_up(self,Nwarm):
        self.state = self.hmc.evolve(self.state,Nwarm).detach()

    def epsilon_test(self,iN=1,fN=3,steps=50,integrator="minnorm2"):
        # use the state
        x=[]
        y=[]
        ey = []
        P = self.refreshP()
        Hi=self.kinetic(P)+ self.action(self.state)
        for rk in np.logspace(iN,fN,steps):
            k=int(rk)
            dt = 1.0/k
            print("Using dt= ",dt)
            l = getattr(integ,integrator)(self.force,self.evolveQ,k,1.0)
            PP,QQ = l.integrate(P,self.state)
            Hf = self.kinetic(PP)+ self.action(QQ)
            DH = tr.abs(Hf - Hi).detach()
            x.append(dt) 
            y.append(DH.mean().item())
            ey.append(DH.std().item()/np.sqrt(DH.shape[0]-1))
        return x,y,ey
            
    
    def sample(self,shape):
        N = np.prod(shape)
        # sample as many times as needed to get the shape elements.
        tt = int(N / self.Nb)
        if(N % self.Nb != 0):
            tt += 1
            
        #print(self.Nb,N,tt)
        foo=self.hmc.evolve(self.state,self.Nhmc)
        total = foo.clone().detach()
        for k in range(tt-1):
            foo = self.hmc.evolve(foo,self.Nhmc).detach()
            total = tr.cat([total,foo],dim=0)
        #print(total.shape)
        tt_shape = tuple([*shape,*tuple(self.field_shape)])
        #print(tt_shape)
        result = total[:N].view(tt_shape)
        self.state = foo
        return result

def main():
    m2 = -1.00
    lam = 1.0
    action = lambda x :  tr.sum(0.5*m2*x*x + lam/24.0*x*x*x*x,dim=1)
    hmc = hmc_sampler(tr.ones(16,2),integrator="minnorm4pf4", Nmd=5,action=action,verbose=False)
    x = tr.randn(5,2)
    print("If zero force is correct: ",(hmc.force(x)+m2*x + lam/6.0*x*x*x).norm().item())

    hmc.warm_up(1000)
    print("Acceptance rate: ", hmc.hmc.calc_Acceptance())
    hmc.hmc.AcceptReject = []
    boo = hmc.sample((1024,))

    print("The mean and variance for the distribution is: ",boo.mean().item(),boo.std().item())
    print("Acceptance rate: ", hmc.hmc.calc_Acceptance())
    plt.hist(boo.view(np.prod(list(boo.shape)),1).detach().numpy(),100)
    plt.show()

    
    
if __name__ == "__main__":
    main()
  
