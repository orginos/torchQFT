#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  17 17:00:00 2022

The model is O(3) in 2D. We will simulate with Hybrid Monte Carlo.
For this reason we define the derivative of the spin along the \alpha
direction in the tangent space as
\partial_\alpha sigma =L_\alpha sigma
where L is the anti-hermitian generator of 3D rotations

L_1 = [ 0  0 0 ; 0 0 -1 ;  0 1 0]
L_2 = [ 0  0 1 ; 0 0  0 ; -1 0 0]
L_3 = [ 0 -1 0 ; 1 0  0 ;  0 0 0]

the action of the model is 
S = -\beta \sim_x \sum_\mu sigma(x)^T*sigma(x+\mu)

Will use torch tensors with a batch dimension to describe the model.
sigma[batch,spin,x,y] is the structure of the tensor containing the field

@author: Kostas Orginos
"""

import torch as tr
import numpy as np

class O3():
    # sigma is a torch tensor
    
    def action(self,sigma):
        A = self.Nd*self.Vol*tr.ones(sigma.shape[0],device=self.device)
        # normalize the action so that it is zero when the spins are aligned
        #although flattening is not taking anytime ...
        # I will explicitelly make the code 2d
        for mu in range(2,self.Nd+2):
            A = A - tr.einsum('bsxy,bsxy->b',sigma,tr.roll(sigma,shifts=-1,dims=mu))
        return self.beta*A

    def force(self,sigma):
        F = tr.zeros_like(sigma)
        Lsig = -tr.einsum('bsxy,sra->braxy',sigma,self.L)
        for mu in range(2,2+self.Nd):
            F+=tr.roll(sigma,shifts= 1,dims=mu)+tr.roll(sigma,shifts=-1,dims=mu)
            #F=F+tr.einsum('bsaxy,bsxy->baxy',Lsig,
            #              tr.roll(sigma,shifts= 1, dims=mu)+
            #              tr.roll(sigma,shifts=-1, dims=mu))
        F=tr.einsum('bsaxy,bsxy->baxy',Lsig,F)
        #cross is a factor of 2 slower than einsum!
        #note that multiplying L and then doing a dot product
        # has many more flops!
        #F=tr.cross(sigma,F,dim=1)
        return self.beta*F

    #here I have a problem... how do I know the batch size?
    #I need to set it in the constructor...
    # again here I explicitely make it 2-d
    def refreshP(self):
        P = tr.normal(0.0,1.0,[self.Bs,self.N,self.V[0],self.V[1]],
                      dtype=self.dtype,device=self.device)
        return P

    def kinetic(self,P):
        return tr.einsum('bsxy,bsxy->b',P,P)/2.0 ;

    #MM = eye(3,3) + sin(w)*A/w + 1/2*sin(w/2)^2*A*A/(w/2)^2
    #with w = norm(p)
    # A= p.L
    def expo(self,P):
        E=tr.eye(self.N,self.N,dtype=self.dtype,device=self.device).view(1,self.N,self.N,1,1)
        #print(E.shape)
        norm = tr.sqrt(tr.einsum('baxy,baxy->bxy',P,P))
        sin = tr.sin(norm)/norm
        sin2 = 2*(tr.sin(norm/2)/norm)**2
        #build the matrix A
        A  = tr.einsum('ija,baxy->bijxy',self.L,P)
        AA = tr.einsum('bikxy,bkjxy->bijxy',A,A)
        R= E + tr.einsum('bijxy,bxy->bijxy',A,sin) +  tr.einsum('bijxy,bxy->bijxy',AA,sin2)
        return R

    def evolveQ(self,dt,P,Q):
        R = self.expo(dt*P)
        return  tr.einsum('bsrxy,brxy->bsxy',R,Q)

    # multiplies an O(3) matrix to a vector
    def mult(self,R,Q):
        return tr.einsum('bsrxy,brxy->bsxy',R,Q)

    #dot product of two spins on every point on the lattice
    def dot(self,s1,s2):
        return tr.einsum('bsxy,bsxy->bxy',s1,s2)
    
    def Q(self,sigma):
        # implementation of the geometric definition of the
        # topological charge from
        # Berg & Luscher: https://doi.org/10.1016/0550-3213(81)90568-X
        #
        def sA(s1,s2,s3):
            s12 = self.dot(s1,s2)
            s23 = self.dot(s2,s3)
            s13 = self.dot(s1,s3)
            rho2 = 2.0*(1.0+s12)*(1.0+s23)*(1.0 + s13)
            #print("rho2 = ", rho2)
            rho2 = tr.where(rho2>0, rho2, tr.ones_like(rho2))
            rho = tr.sqrt(rho2)
            s123 = 1j*self.dot(s1,tr.cross(s2,s3,dim=1))
            #print("rho = ", rho)
            #print("rho2 = ", rho2)
            
            num = 1.0 + s12+s13+s23 +s123
            #print("num = ", num)
            sA = tr.where(tr.abs(num) >1.0e-15 ,num/rho,tr.ones_like(num))
            #sA = (1.0 + s12+s13+s23 +s123)/rho
            #print("sA= ",sA)
            #print("abs(sA)= ",np.abs(sA))
            sA = -2.0*1j*tr.log(sA)
            return tr.real(sA )

        if(self.Nd!=2):
            print("Topological charge in defined in 2D")
            return 0

        # the corners of the two triangles (together with sigma)
        sig0 = tr.roll(sigma,shifts=-1,dims=2) # mu=0
        sig1 = tr.roll(sigma,shifts=-1,dims=3) # mu=1
        sig01 = tr.roll(sig0,shifts=-1,dims=3) # mu=1
        q = sA(sigma,sig0,sig01) + sA(sigma,sig01,sig1)
        dims=range(1,len(q.shape))
        return tr.sum(q,dim=tuple(dims))/(4.0*np.pi)


    def coldStart(self):
        sigma=tr.ones([self.Bs,self.N,self.V[0],self.V[1]],dtype=self.dtype,device=self.device)
        n=self.dot(sigma,sigma).view(self.Bs,1,self.V[0],self.V[1])
        return sigma/tr.sqrt(n)
        
    def hotStart(self):
        sigma=tr.normal(0.0,1.0,
                        [self.Bs,self.N,self.V[0],self.V[1]],
                        dtype=self.dtype,device=self.device)
        # this is not exactly random ....
        # I need to cut the corners to get a uniformly random vector
        n=self.dot(sigma,sigma).view(self.Bs,1,self.V[0],self.V[1])
        return sigma/tr.sqrt(n)
    
    def __init__(self,V,beta,batch_size=1,device="cpu",dtype=tr.float32): 
            self.V = V # a tuple with the lattice dimensions
            self.Nd = len(V)
            self.Vol = np.prod(V)
            self.beta = beta # the coupling
            self.device=device
            self.dtype=dtype
            self.Bs=batch_size # batch size
            self.N = 3 # only the O(3) is simulated here
            # the generators of the group
            self.L  = tr.tensor([[[  0,  0,  0],
                                 [  0,  0, -1],
                                 [  0,  1,  0]],
                                [[  0,  0,  1],
                                 [  0,  0,  0],
                                 [ -1,  0,  0]],
                                [[  0, -1,  0],
                                 [  1,  0,  0],
                                 [ 0,  0,  0]]],
                                dtype=self.dtype,
                                device=self.device)


def main():
    import time
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    L=128
    batch_size=1
    #get a unit norm random sigma
    #tr.rand(batch_size, 3,L, L, device=device)
    #n=tr.einsum('bsxy,bsxy->bxy',sigma,sigma).view(batch_size,1,L,L)
    #sigma = sigma/tr.sqrt(n)
    beta=1.263
    o = O3([L,L],beta,batch_size=batch_size)
    sigma=o.hotStart()
    tic=time.perf_counter()
    Niter=10000
    for k in range(Niter):
        o.action(sigma)
    toc=time.perf_counter()
    print(f"action time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")

    tic=time.perf_counter()
    for k in range(Niter):
        o.force(sigma)
    toc=time.perf_counter()
    print(f"force time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")    

    P = o.refreshP()

    tic=time.perf_counter()
    for k in range(Niter):
        o.kinetic(sigma)
    toc=time.perf_counter()
    print(f"kinetic time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")    

    tic=time.perf_counter()
    Niter=5000
    for k in range(Niter):
        o.expo(P)
    toc=time.perf_counter()
    print(f"expo time {(toc - tic)*1.0e6/Niter:0.4f} micro-seconds")    

    ev_sigma=o.evolveQ(1.0,P,sigma)

    q = o.Q(ev_sigma)
    if(batch_size>1):
        for b in range(batch_size):
            print(f"Topological charge of evolved sigma in batch {b}: {q[b]:0.4f}")
    else:
            print(f"Topological charge of evolved sigma: {q[0]:0.4f}")
    
    
if __name__ == "__main__":
   main()
    
            



