import torch as tr
import O3 as s
import integrators as i
import update as u
import numpy as np

import matplotlib.pyplot as plt

import rungeKutta as rk

class O3flow():
    #allow for time dependence in the flow action
    # and the resunting generator
    def S(self,t,sigma): # flow action
        A = self.Nd*self.Vol*tr.ones(sigma.shape[0],device=self.device)
        # normalize the action so that it is zero when the spins are aligned
        #although flattening is not taking anytime ...
        # I will explicitelly make the code 2d
        for mu in range(2,self.Nd+2):
            A = A - tr.einsum('bsxy,bsxy->b',sigma,tr.roll(sigma,shifts=-1,dims=mu))
        return (self.C[0]+self.C[1]*t*t)*A

    # Z is the gradient (not the negative gradient) of the flow action
    def Z(self,t,sigma): # flow generator
        h = tr.zeros_like(sigma)
        for mu in range(2,self.Nd+2):
              h =h + tr.roll(sigma,shifts=-1,dims=mu) + tr.roll(sigma,shifts=1,dims=mu)
        F =  tr.einsum('bixy,bjxy->bijxy',h,sigma) -  tr.einsum('bixy,bjxy->bijxy',sigma,h)
         
        return -(self.C[0]+self.C[1]*t*t)*F
        
    def __init__(self,C,V,batch_size=1,device="cpu",dtype=tr.float32):
        self.C = C # the coefficients
        self.Nd = len(V)
        self.Vol = np.prod(V)
        self.V = V # a tuple with the lattice dimensions
        self.device=device
        self.dtype=dtype
        self.Bs=batch_size # batch size
        self.N =3 

    def hotStart(self):
        sigma=tr.normal(0.0,1.0,
                        [self.Bs,self.N,self.V[0],self.V[1]],
                        dtype=self.dtype,device=self.device)
        # this is not exactly random ....
        # I need to cut the corners to get a uniformly random vector
        n=self.dot(sigma,sigma).view(self.Bs,1,self.V[0],self.V[1])
        return sigma/tr.sqrt(n)

    def dot(self,s1,s2):
        return tr.einsum('bsxy,bsxy->bxy',s1,s2)
    
# multiplies an O(3) matrix to a vector
def GxV(R,Q):
    return tr.einsum('bkjxy,bjxy->bkxy',R,Q)

# multiplies two group  elements
def GxG(R,Q):
    return tr.einsum('bkjxy,bjmxy->bkmxy',R,Q)

    
lat=[16,16]

o = O3flow(tr.tensor([-1.0,1.5]),lat,2)

# normalize the spins

sigma = o.hotStart()
 

print("Initial action: ",o.S(0.0,sigma))

rk4 = rk.lieGroupRK4(GxG,GxV,o.Z)

N0 = 10
h0 = 1.0/N0
gf_sigma0 = rk4.integrate(0.0, sigma, 1.0, N0)
act0=o.S(1.0,gf_sigma0)
print("Final action: ",h0,act0)

N0 = 100
h0 = 1.0/N0
gf_sigma0 = rk4.integrate(0.0, sigma, 1.0, N0)
act0=o.S(1.0,gf_sigma0)
print("Final action: ",h0,act0)

N0 = 1000
h0 = 1.0/N0
gf_sigma0 = rk4.integrate(0.0, sigma, 1.0, N0)
act0=o.S(1.0,gf_sigma0)
print("Final action: ",h0,act0)

#sg = s.O3(G,1.0)
#

#gf_sigma0 = rk.rungeKuttaCompact(0.0, sigma, 1.0, N0,sg.force,sg.evolveQ)
#act0=sg.action(gf_sigma0)
#print("Final action: ",h0,act0,-o3f.S(1.0,gf_sigma0))
