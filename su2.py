#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  17 9:38:24 2019

The model is SU(2) in 4D. The generators of the group are 
 tau_k. They are anti-hermitian and take to be the following:

tau_1 = i/2 .* [ 0   1 ;  1  0 ]
tau_2 = i/2 .* [ 0   i ; -i  0 ]
tau_3 = i/2 .* [ 1   0 ;  0 -1 ]

the action of the model is 
S = -\beta \sum_x \sum_\mu_\nu P_\mu\nu(x)

with P_\mu\nu is the plaquette.

The fields will be layed out as 
U[b,i,j,x,mu]
with i, j are color indices x : (xyzt) is space time and mu is direction of the link
b is the batch dimension

@author: Kostas Orginos
"""

import torch as tr
import torch.nn as nn

import numpy as np

dtype = tr.float32
ctype = tr.complex64
device = "cpu"
tau=tr.tensor([0.0])
tau0 = 0.5*tr.tensor([[1,0],
                       [0,1]])
def reset_tau():
    return 0.5j*tr.tensor([[[0,1],
                            [1,0]],
                           [[  0, 1j],
                            [-1j,  0]],
                           [[1, 0],
                            [0,-1]]])
tau = reset_tau()

basis = tr.concat((tau0[None,:,:],tau),dim=0)

# multiplies an SU(2) matrix to a vector
def GxV(R,Q):
    return tr.einsum('bkjxyzt,bjxyzt->bkxyzt',R,Q)

# multiplies two group  elements
def GxG(R,Q):
    return tr.einsum('bkjxy,bjmxy->bkmxy',R,Q)


# multiplies Lie group matrices in matrix form
def TxT(L,M):
    return tr.einsum('ik,kj->ij',L,M)

def TdxT(L,M):
    return tr.einsum('ki,kj->ij',tr.conj(L),M)

def coeffs(U):
    b=tr.empty([4])
    for k in range(4):
        sign = -1
        if (k==0): sign = 1 
        b[k] = tr.real(2.0*sign*tr.einsum('ii',TxT(basis[k],U)))
    return b

def Mat(b):
    return tr.sum(b[:,None,None] * basis,dim=0)

def cMat(z):
    U = tr.empty(2,2,dtype=z.dtype)
    U[0,:] = z
    U[1,0] = tr.conj(-z[1])
    U[1,1] = tr.conj( z[0])
    return U

# this layout is bacxyztd
# d : mu
# b : batch
# xyzt: space-time
# ac : color
# I call it norm layout
class gauge_field():
    def __init__(self,lat,Nbatch=1,dtype=tr.complex64,device="cpu"): 
         self.dtype = dtype
         self.device = device
         self.lat = lat
         self.Vol = np.prod(lat)
         self.Nd = len(lat)
         self.Nc = 2
         self.shape = [Nbatch]+[self.Nc,self.Nc]+lat+[len(lat)]
    def empty(self):
        return tr.empty(self.shape,device=self.device,dtype=self.dtype)
    def zero(self):
        return tr.zeros(self.shape,device=self.device,dtype=self.dtype)
    def zeroCM(self):
        return tr.zeros(self.shape[0:-1],device=self.device,dtype=self.dtype)
        
    def cold(self):
        shape = [1,2,2]+len(self.shape[3:])*[1]
        one = tr.eye(2,2,device=self.device,dtype=self.dtype).view(shape)
        return one.repeat([self.shape[0],1,1]+self.lat+[self.Nd])
    def hot(self):
        v = tr.randn([self.shape[0],4]+self.shape[3:])
        v = 2.0*v/v.norm(dim=1).view([self.shape[0],1]+self.shape[3:]).type(self.dtype)
        return tr.einsum('mac,bm...->bac...',basis,v)

    def random_gauge(self):
        v = tr.randn([self.shape[0]]+self.shape[3:])
        v = 2.0*v/v.norm(dim=1).view([self.shape[0]]+self.shape[3:]).type(self.dtype)
        return tr.einsum('mac,bm...->bac...',basis,v)
        
    def trace_plaquett(self,U,mu,nu):
        P = tr.empty([U.shape[0]]+list(U.shape[3:-1]),dtype=U.dtype)
        P = tr.einsum('bij...,bjl...,bml...,bim...->b',
                      U[...,mu],
                      U[...,nu].roll(dims=mu+3,shifts=-1),
                      U[...,mu].roll(dims=nu+3,shifts=-1).conj(),
                      U[...,nu].conj())
        return P

    def staple(self,U,mu,nu):
        S = tr.einsum('bij...,bjl...,bml...->bim...',
                      U[...,nu],
                      U[...,mu].roll(dims=nu+3,shifts=-1),
                      U[...,nu].roll(dims=mu+3,shifts=-1).conj())
        S+= tr.einsum('bji...,bjl...,blm...->bim...',
                      U[...,nu].roll(dims=nu+3,shifts=+1).conj(),
                      U[...,mu].roll(dims=nu+3,shifts=+1),
                      U[...,nu].roll(dims=(mu+3,nu+3),shifts=(-1,1)))
        return S

    def multCM(U,V):
        return tr.einsum('bik...,bkj...->bij...',U,V)
    def multCMdag(U,V):
        return tr.einsum('bik...,bjk...->bij...',U,V.conj())
        
    def link(U,mu):
        return U[...,mu]
         
    def add_force(self,F,P,mu):
        F[...,mu] += P

    def plaquett_force(self,U):
        F = self.empty()
        for mu in range(self.Nd):
            S = self.zeroCM()
            for nu in range(self.Nd):
                if(nu!=mu):
                    S += self.staple(U,mu,nu)
            #self.gf.add_force(F,self.gf.staple(U,mu,nu),mu)
            F[...,mu] = tr.einsum('bik...,bjk...->bij...',U[...,mu],S.conj())
        F=0.5*(F-F.transpose(1,2).conj()) # no need to subtract the trace     
        return F
        # NEED TO CHECK CORRECTNESS 
    
    def LieProject(self,F):
        F = 0.5*(F-F.transpose(1,2).conj())
        trace = 0.5*tr.einsum('bii...->b...',F)
        print(trace.shape)
        # subtract the trace
        F[:,0,0] -= trace
        F[:,1,1] -= trace
        # if it is used only for force there is no need for trace subtraction as it is automatically traceless because we are in SU(2) 
        # If you use this on a general 2x2 matrix then you need to subtract the trace
        return F 
        
# this layout is dbxyztac
# d : mu
# b : batch
# xyzt: space-time
# ac : color
# I call it alt layout
class gauge_field_alt():
    def __init__(self,lat,Nbatch=1,dtype=tr.complex64,device="cpu"): 
         self.dtype = dtype
         self.device = device
         self.lat = lat
         self.Vol = np.prod(lat)
         self.Nd = len(lat)
         self.Nc = 2
         self.shape = [len(lat),Nbatch]+lat+[self.Nc,self.Nc]
        
    def empty(self):
        return tr.empty(self.shape,device=self.device,dtype=self.dtype)
    def zero(self):
        return tr.zeros(self.shape,device=self.device,dtype=self.dtype)
    def zeroCM(self):
        return tr.zeros(self.shape[1:],device=self.device,dtype=self.dtype)
    def cold(self):
        shape = (len(self.shape)-2)*[1] + [2,2]
        #print(shape)
        one = tr.eye(2,2,device=self.device,dtype=self.dtype).view(shape)
        #ss = [self.shape[0],1,1]+self.lat+[self.Nd]
        #print(ss)
        return one.repeat([self.Nd,self.shape[1]]+self.lat+[1,1])
    def hot(self):
        v = tr.randn(self.shape[0:-2]+[4])
        print(v.shape)
        v = 2.0*v/v.norm(dim=len(self.shape)-2).view((self.shape[0:-2]+[1])).type(self.dtype)
        return tr.einsum('mac,...m->...ac',basis,v)
        
    def random_gauge(self):
        v = tr.randn(self.shape[0:-2])
        print(v.shape)
        v = 2.0*v/v.norm(dim=len(self.shape)-2).view((self.shape[0:-2])).type(self.dtype)
        return tr.einsum('mac,...m->...ac',basis,v)
    
#    @tr.compile
    def trace_plaquett(self,U,mu,nu):
        P = tr.empty([U.shape[0]]+list(U.shape[3:-1]),dtype=U.dtype)
        P = tr.einsum('b...ij,b...jl,b...ml,b...im->b',
                      U[mu],
                      U[nu].roll(dims=mu+1,shifts=-1),
                      U[mu].roll(dims=nu+1,shifts=-1).conj(),
                      U[nu].conj())
        return P

    def staple(self,U,mu,nu):
        S = tr.einsum('...ij,...jl,...ml->...im',
                      U[nu],
                      U[mu].roll(dims=nu+1,shifts=-1),
                      U[nu].roll(dims=mu+1,shifts=-1).conj())
        S+= tr.einsum('...ji,...jl,...lm->...im',
                      U[nu].roll(dims=nu+1,shifts=+1).conj(),
                      U[mu].roll(dims=nu+1,shifts=+1),
                      U[nu].roll(dims=(mu+1,nu+1),shifts=(-1,1)))
        return S

    def link(U,mu):
        return U[mu]

    def multCM(U,V):
        return tr.einsum('...ik,...kj->...ij',U,V)
    def multCMdag(U,V):
        return tr.einsum('...ik,...jk->...ij',U,V.conj())
        
    def add_force(self,F,P,mu):
        F[mu] += P

    def plaquett_force(self,U):
        F = self.empty()
        for mu in range(self.Nd):
            S = self.zeroCM()
            for nu in range(self.Nd):
                if(nu!=mu):
                    S += self.staple(U,mu,nu)
            #self.gf.add_force(F,self.gf.staple(U,mu,nu),mu)
            F[mu] = tr.einsum('...ik,...jk->...ij',U[mu],S.conj())
        F=0.5*(F-F.transpose(6,7).conj()) # no need to subtract the trace     
        return F
        # NEED TO CHECK CORRECTNESS 
        
    def LieProject(self,F):
        F = 0.5*(F-F.transpose(6,7).conj())
        trace = 0.5*tr.einsum('...ii->...',F)
        # subtract the trace
        F[...,0,0] -= trace
        F[...,1,1] -= trace
        return F 
        
def alt_to_norm(U):
    return  tr.einsum('dbxyztak->bakxyztd',U)
    
def norm_to_alt(U):
    return  tr.einsum('bakxyztd->dbxyztak',U)

class SU2():
    def __init__(self,beta,gauge_field_type,Nbatch=1):
        self.beta = beta # the coupling
        self.gf = gauge_field_type
    
    def action(self,U):
        A = self.gf.Nd*(self.gf.Nd-1)*self.gf.Vol/2.0
        # normalize the action so that it is zero when the links are 1
        #print(A)
        for mu in range(self.gf.Nd):
            for nu in range(mu+1,self.gf.Nd):
                A = A - self.gf.trace_plaquett(U,mu,nu).real/float(self.gf.Nc)
        return self.beta*A

    def force(self,U):
        return self.gf.plaquett_force(U)*(self.beta/float(self.gf.Nc))
    
def main():
    import matplotlib.pyplot as plt

    # test Lie algebra
    for k in range(4):
        print(basis[k,:,:].numpy())

    print("Check Lie algebra")
    for k0 in range(3):
        k1 = (k0+1)%3
        k2 = (k0+2)%3
        d = TxT(tau[k0,:,:],tau[k1,:,:]) - TxT(tau[k1,:,:],tau[k0,:,:]) - tau[k2,:,:] 
        print(k0,k1,k2," : ", d.sum().numpy())

    print("Check Lie algebra")
    for k0 in range(3):
        k1 = (k0+1)%3
        k2 = (k0+2)%3
        d = TxT(tau[k0,:,:],tau[k1,:,:])  - 0.5*tau[k2,:,:] 
        print(k0,k1,k2," : ", d.sum().numpy()  )

    print("Check anti-commutator")
    
    for k0 in range(3):
        for k1 in range(3):
            d = TxT(tau[k0,:,:],tau[k1,:,:]) + TxT(tau[k1,:,:],tau[k0,:,:])
            print(k0,k1, " : ", d.numpy())
            

    a = tr.normal(0.0,2.0,[4])
    a = 2*a/tr.norm(a)
    print(a)
    U = Mat(a) #tr.sum(a[:,None,None] * basis,dim=0)
    print(U)
    print("Should be identity")
    print(TdxT(U,U))
    print(a.norm()/2)
    #check coefficient extraction
    b=coeffs(U)
    print(a-b)

    b = tr.normal(0.0,2.0,[4])
    b = 2*b/tr.norm(b)

    Ub = Mat(b)

    Uc = TxT(U,Ub)
    c= coeffs(Uc)
    print(c,c.norm()/2)

    #Uc = TxT(Ub,U)
    #c= coeffs(Uc)
    #print(c,c.norm()/2)

    print(0.5*(a[0]*b[0] - tr.sum(a[1:]*b[1:])))
    for k in range(3):
        cc = 0.5*(a[0]*b[k+1] +  a[k+1]*b[0])
        s=1
        for l in range(k+1,3):
            cc += -0.5*s*(a[k]*b[l] - a[l]*b[k])
            s*=-1
        print(cc)


    #Let's use two complex numbers
    z = tr.normal(0,2,[2]) + 1j *  tr.normal(0,2,[2])
    z = z/tr.norm(z)
    print(z)

    U = cMat(z)
    print("Must be identity")
    print(TdxT(U,U))

    z1 = tr.normal(0,2,[2]) + 1j *  tr.normal(0,2,[2])
    z1 = z1/tr.norm(z1)

    Ub = cMat(z1)
    Uc = TxT(U,Ub)

    zc = Uc[0,:]

    zzc = tr.zeros([2],dtype=ctype)
    zzc[0] = z[0]*z1[0] - z[1]*tr.conj(z1[1])
    zzc[1] = z[0]*z1[1] + z[1]*tr.conj(z1[0])

    print("Must be zero")
    print(zzc-zc)


    
    
if __name__ == "__main__":
   main()

        
        
    

    
