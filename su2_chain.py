#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Oct 21 13:20:18 EDT 2024

The model is SU(2) chiral chain in 2D (otherwise known as the O(4) spin model).
The generators of the group are  tau_k. They are anti-hermitian and take to be
the following:

tau_1 = i/2 .* [ 0   1 ;  1  0 ]
tau_2 = i/2 .* [ 0   i ; -i  0 ]
tau_3 = i/2 .* [ 1   0 ;  0 -1 ]

the action of the model is 
S = -beta sum_(x,mu) 1/4[ Tr u(x) u(x+mu)^(+)  + h.c.]

this coresponds in terms of unit length spins to the action
S_spin = -beta sum_(x,mu) s(x) s(x+mu) with s being uning norm spin

The fields will be layed out as 
U[b,x,i,j]
with i, j are color indices and x : (xy) is space 
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

        
# this layout is dbxyac
# b : batch
# xy: space
# ac : color
class field():
    def __init__(self,lat,Nbatch=1,dtype=tr.complex64,device="cpu"): 
         self.dtype = dtype
         self.device = device
         self.lat = lat
         self.Vol = np.prod(lat)
         self.Nd = len(lat)
         self.Nc = 2
         self.shape = [Nbatch]+lat+[self.Nc,self.Nc]
         self.ci = [len(self.shape)-2, len(self.shape)-1]# color indices
         self.eps = tr.finfo(dtype).eps
         #self.eye = tr.eye(2).expand(10, 128, 128, -1, -1)

    def empty(self):
        return tr.empty(self.shape,device=self.device,dtype=self.dtype)
    def zero(self):
        return tr.zeros(self.shape,device=self.device,dtype=self.dtype)
    def cold(self):
        shape = (len(self.shape)-2)*[1] + [2,2]
        #print(shape)
        one = tr.eye(2,2,device=self.device,dtype=self.dtype).view(shape)
        #ss = [self.shape[0],1,1]+self.lat+[self.Nd]
        #print(ss)
        return one.repeat([self.Nd,self.shape[1]]+self.lat+[1,1])
    def hot(self):
        v = tr.randn(self.shape[0:-2]+[4])
        #print(v.shape)
        v = 2.0*v/v.norm(dim=len(self.shape)-2).view((self.shape[0:-2]+[1])).type(self.dtype)
        return tr.einsum('mac,...m->...ac',basis,v)
        
#    @tr.compile
    def trace_nn_prod(self,U,mu):
        return tr.einsum('b...ik,b...ik->b',U,U.roll(dims=mu+1,shifts=-1).conj())

    def multCM(self,U,V):
        return tr.einsum('...ik,...kj->...ij',U,V)
    def multCMdag(self,U,V):
        return tr.einsum('...ik,...jk->...ij',U,V.conj())
        
    def nn_force(self,U):
        H = self.zero()
        for mu in range(self.Nd):
            #F += tr.einsum('...ik,...jk->...ij',U,U.roll(dims=mu+1,shifts=-1).conj())
            H += U.roll(dims=mu+1,shifts=-1) + U.roll(dims=mu+1,shifts=+1)
        F = tr.einsum('...ik,...jk->...ij',U,H.conj())
        F=0.5*(F-F.transpose(self.ci[0],self.ci[1]).conj()) # no need to subtract the trace     
        return -0.5*F
        # NEED TO CHECK CORRECTNESS 

    def coeffs(self,U):
        b=tr.empty(list(U.shape[:-2])+[4])
        b = -tr.real(2.0*tr.einsum('kij,...ji->...k',basis,U))
        b[...,0] = - b[...,0]
        return b
    def mat(self,b):
        return tr.einsum('kij,...k->...ij',basis, tr.complex(b,tr.zeros_like(b)))
    
    def reunit(self,A):
        U,S,Vh = tr.linalg.svd(A)
        return U@Vh

    def LieProject(self,F):
        F = 0.5*(F-F.transpose(self.ci[0],self.ci[1]).conj())
        trace = 0.5*tr.einsum('...ii->...',F)
        # subtract the trace
        F[...,0,0] -= trace
        F[...,1,1] -= trace
        return F 

#    def expo(self,P):
#        nn = tr.norm(P,dim=(self.ci[0],self.ci[1]))/np.sqrt(2)
#        E = tr.einsum('...,ij->...ij',tr.cos(nn),tr.eye(2,2)) + tr.einsum('...,...ij->...ij',tr.sin(nn)/nn,P) 
#        return E

#assumes last two indices are where the matrix is
    def expo(self,P):
        nn = (tr.norm(P,dim=(-1,-2))/np.sqrt(2)+tr.finfo(P.dtype).eps).unsqueeze(-1).unsqueeze(-1)   
        return tr.cos(nn)*tr.eye(2, dtype=P.dtype, device=P.device).expand_as(P) + P*(tr.sin(nn)/nn) 


    
    def traceSquared(self,P):
        return tr.einsum('b...ij,b...ij->b',P,P.conj()).real

    def evolveQ(self,dt,P,Q):
        R = self.expo(dt*P)
        return  tr.einsum('...ik,...kj->...ij',R,Q)


class SU2chain():
    def __init__(self,beta,field_type):
        self.beta = beta # the coupling
        self.f    = field_type
        
    def action(self,U):
        A = self.f.Nd*self.f.Vol
        # normalize the action so that it is zero when the spins are 1
        #print(A)
        for mu in range(self.f.Nd):
                A = A - self.f.trace_nn_prod(U,mu).real/float(self.f.Nc)
        return self.beta*A

    def force(self,U):
        return self.f.nn_force(U)*(self.beta/float(self.f.Nc))

    def refreshP(self):
        P = tr.normal(0.0,1.0,self.f.shape,dtype=self.f.dtype,device=self.f.device)
        P -= P.transpose(self.f.ci[0],self.f.ci[1]).conj() 
        trace = 0.5*tr.einsum('...ii->...',P)
        P[...,0,0]-=trace
        P[...,1,1]-=trace
        P *= 0.5
        return P

    def kinetic(self,P):
        return self.f.traceSquared(P)
                
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

        
        
    

    
