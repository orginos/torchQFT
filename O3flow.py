#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:28:06 EDT 2021

Implement flow actions for the O(3) model.
I will use the generator free formalism for now... 

@author: Kostas Orginos
"""
import torch as tr
import torch.nn as nn

import numpy as np

import O3 as o3

dtype = tr.float32
device = "cpu"
L=tr.tensor([0.0])
def reset_L():
    return tr.tensor([[[  0,  0,  0],
                     [  0,  0, -1],
                     [  0,  1,  0]],
                    [[  0,  0,  1],
                     [  0,  0,  0],
                     [ -1,  0,  0]],
                    [[  0, -1,  0],
                     [  1,  0,  0],
                     [ 0,  0,  0]]],
                   dtype=dtype,
                   device=device)
L = reset_L()

# multiplies an O(3) matrix to a vector
def GxV(R,Q):
    return tr.einsum('bkjxy,bjxy->bkxy',R,Q)

# multiplies two group  elements
def GxG(R,Q):
    return tr.einsum('bkjxy,bjmxy->bkmxy',R,Q)


class FlowActionTerm(nn.Module):
    def __init__(self,cc=[0],EvenOdd=True):
        super(FlowActionTerm, self).__init__()
        self.EO=EvenOdd
        self.C = nn.ParameterList([nn.Parameter(tr.tensor(c)) for c in cc])

        
    #evaluate the coefficient at flowtime t
    # the result is C[0] is the hiest power
    # interprents the coefficients as even or odd powered polynominum
    def Coeff(self,t):
        t2=t*t;
        val = self.C[0]
        for k in range(len(self.C)-1):
            val = val*t2 +self.C[k+1]
        if (self.EO == False):
            val = t*val
        return val


class Psi0(FlowActionTerm):
    def __init__(self,cc=[1.0]):
        FlowActionTerm.__init__(self,cc,EvenOdd=True)
        
    def action(self,s,t):
        # I have a batch dimension first
        A = tr.zeros(s.shape[0],device=device)
        # I will explicitelly make the code 2d
        for mu in range(2,s.dim()):
            A += tr.einsum('bsxy,bsxy->b',s,tr.roll(s,shifts=-1,dims=mu))
        
        return self.Coeff(t)*A

    def grad(self,s,t):
        F = tr.zeros_like(s)
        Lsig = -tr.einsum('bsxy,sra->braxy',s,L)
        for mu in range(2,s.dim()):
            F+=tr.roll(s,shifts= 1,dims=mu)+tr.roll(s,shifts=-1,dims=mu)
        F=tr.einsum('bsaxy,bsxy->baxy',Lsig,F)
        
        return self.Coeff(t)*F

    # this one is simple... it is an eigenfunction of the laplacian
    def lapl(self,s,t):
        return -4.0*self.action(s,t)
    
class Psi2(FlowActionTerm):
    def __init__(self,cc=[1.0]):
        FlowActionTerm.__init__(self,cc,EvenOdd=False)
        
    def action(self,s,t):
        # I have a batch dimension first
        A = tr.zeros(s.shape[0],device=device)
        # first make the s1
        s1 = tr.zeros_like(s) #-( 2.0*(s.dim()-2) )*s 
        for mu in range(2,s.dim()):
            s1+=tr.roll(s,shifts=-1,dims=mu)+tr.roll(s,shifts=1,dims=mu)
        for mu in range(2,s.dim()):
            A += tr.einsum('bsxy,bsxy->b',s,tr.roll(s1,shifts=-1,dims=mu))

        V = 2.0*tr.ones(s.shape[0],device=device,dtype=dtype)
        for mu in range(2,s.dim()):
            V *= s.shape[mu] 
        # subtract the constant
        A -= V
        return self.Coeff(t)*A

    def grad(self,s,t):
        # first make the s1
        s1 = tr.zeros_like(s)#-( 2.0*(s.dim()-2) )*s 
        for mu in range(2,s.dim()):
            s1+=tr.roll(s,shifts=-1,dims=mu)+tr.roll(s,shifts=1,dims=mu)
        F = tr.zeros_like(s)
        Lsig = -tr.einsum('bsxy,sra->braxy',s,L)
        for mu in range(2,s.dim()):
            F+=tr.roll(s1,shifts= 1,dims=mu)+tr.roll(s1,shifts=-1,dims=mu)
        F=tr.einsum('bsaxy,bsxy->baxy',Lsig,F)
        
        return self.Coeff(t)*F

    # this one is simple... it is an eigenfunction of the laplacian
    def lapl(self,s,t):
        return -4.0*self.action(s,t)

class Psi11_l(FlowActionTerm):
    def __init__(self,cc=[1.0]):
        FlowActionTerm.__init__(self,cc,EvenOdd=False)
        
    def action(self,s,t):
        # I have a batch dimension first
        A = tr.zeros(s.shape[0],device=device)
        for mu in range(2,s.dim()):
            b = tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=mu))
            A += tr.einsum('bxy,bxy->b',b,b)

        V = 2.0/3.0*tr.ones(s.shape[0],device=device,dtype=dtype)
        for mu in range(2,s.dim()):
            V *= s.shape[mu] 
        # subtract the constant
        A -= V
        
        return self.Coeff(t)*A

    def grad(self,s,t):
        F = tr.zeros_like(s)
        Lsig = -tr.einsum('bsxy,sra->braxy',s,L)
        for mu in range(2,s.dim()):
            bp = tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=mu))
            bm = tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=+1,dims=mu))
            F+=tr.einsum('bsxy,bxy->bsxy',tr.roll(s,shifts= 1,dims=mu),bm)
            F+=tr.einsum('bsxy,bxy->bsxy',tr.roll(s,shifts=-1,dims=mu),bp)
        F=tr.einsum('bsaxy,bsxy->baxy',Lsig,F)
        
        return 2.0*self.Coeff(t)*F

    # this one is simple... it is an eigenfunction of the laplacian
    def lapl(self,s,t):
        return -12.0*self.action(s,t)
    
class Psi11t(FlowActionTerm):
    def __init__(self,cc=[1.0]):
        FlowActionTerm.__init__(self,cc,EvenOdd=False)

    def action(self,s,t):
        # first make the bonds
        b = tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=2) + tr.roll(s,shifts=1,dims=2))
        for mu in range(3,s.dim()):
            b += tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=mu) + tr.roll(s,shifts=1,dims=mu))
        
        return self.Coeff(t)*tr.einsum('bxy,bxy->b',b,b)

    def grad(self,s,t):
        # first make the bonds
        b = tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=2) + tr.roll(s,shifts=1,dims=2))
        for mu in range(3,s.dim()):
            b += tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=mu) + tr.roll(s,shifts=1,dims=mu))
        bs = tr.einsum('bsxy,bxy->bsxy',s,b)
        Fs = tr.zeros_like(s)
        Fbs = tr.zeros_like(s)
        Lsig = -tr.einsum('bsxy,sra->braxy',s,L)
        for mu in range(2,s.dim()):
            Fs +=tr.roll(s ,shifts= 1,dims=mu)+tr.roll(s ,shifts=-1,dims=mu)
            Fbs+=tr.roll(bs,shifts= 1,dims=mu)+tr.roll(bs,shifts=-1,dims=mu) 
        F = tr.einsum('baxy,bxy->baxy',Fs,b) + Fbs
        F=tr.einsum('bsaxy,bsxy->baxy',Lsig,F)
        
        return 2.0*self.Coeff(t)*F

    # this one is not simple...
    # I experimentally found the right constant to subract
    # and the right weights for Psi11 and Psi2
    # We need to check the normalization of what I programed and what
    # I have in the notes ... there seem to be a factor of two over all
    # scaling missing is various places
    def lapl(self,s,t):
        ps11_l = Psi11_l([1.0])
        ps2    = Psi2([1.0])
        A = -4.0*ps11_l.action(s,1.0) + 4.0*ps2.action(s,1.0)

        #A = A
        #print(A)
        # the constant
        V = 40.0/3.0*tr.ones(s.shape[0],device=device,dtype=dtype)
        for mu in range(2,s.dim()):
            V *= s.shape[mu] 
        # subtract the constant
        A += V
        #print(A)
        A *= self.Coeff(t) 
        return A-10.0*self.action(s,t)


#this is now an eigenfunction of the Lie Laplacian.
# NOTE: I found all the coefficients experimentaly and they are not equal to those in the old notes...
# I will fix the algebra so that I get the right answer...
class Psi11(FlowActionTerm):
    def __init__(self,cc=[1.0]):
        FlowActionTerm.__init__(self,cc,EvenOdd=False)
        self.p11t = Psi11t([1.0])
        self.p11l = Psi11_l([1.0])
        self.psi2 = Psi2([1.0])
        
        
    def action(self,s,t):
        V = np.prod(list(s.shape[2:s.dim()]))
        A = self.p11t.action(s,t) -2.0*self.p11l.action(s,t) - 2.0/3.0*self.psi2.action(s,t) - 4.0/3.0*V
        
        return self.Coeff(t)*A

    def grad(self,s,t):
        return self.Coeff(t)*(self.p11t.grad(s,t) -2.0*self.p11l.grad(s,t) -2.0/3.0*self.psi2.grad(s,t))

    def lapl(self,s,t):
        return self.Coeff(t)*(self.p11t.lapl(s,t) -2.0*self.p11l.lapl(s,t) -2.0/3.0*self.psi2.lapl(s,t))

class FlowAction(nn.Module):
    def __init__(self,cc=[Psi0([1.0])]):
        super(FlowAction, self).__init__()
        self.T=nn.ModuleList([p for p in cc])
        
    def action(self,s,t):
        A = tr.zeros(s.shape[0],device=device,dtype=dtype)
        for p in self.T:
            A += p.action(s,t)
        return A
    def grad(self,s,t):
        F = tr.zeros_like(s)
        for p in self.T:
            F += p.grad(s,t)
        return F
    def lapl(self,s,t):
        A = tr.zeros(s.shape[0],device=device,dtype=dtype)
        for p in self.T:
            A += p.lapl(s,t)
        return A
        
def NumericalLaplacian(s,action,eps):
    #print(s.size()[2:4])
    o = o3.O3(s.size()[2:4],1.0,batch_size=s.size()[0],device=device)

    Lap = tr.zeros(s.shape[0],device=device)
    A = action(s,1.0);
    for a in range(3):
        for x in range(s.size()[2]):
            for y in range(s.size()[3]):
                f = tr.ones_like(s)*1.0e-15
                f[:,a,x,y]= tr.ones(s.shape[0],device=device)
                #print(f.size())
                sp = o.evolveQ(eps,f,s)
                sm = o.evolveQ(-eps,f,s)
                Ap = action(sp,1.0)
                Am = action(sm,1.0)
                #print(Am,Ap,A)
                Lap += (Ap + Am - 2*A)/eps**2
    return Lap

            
def main():
    import matplotlib.pyplot as plt

    # test polynomium evaluation
    a = FlowActionTerm([1,-1],EvenOdd=True)
    t = np.arange(-5,5,0.01)
    y = a.Coeff(t)
    plt.plot(t,y,t,t**2-1)

    b = FlowActionTerm([1,-1],EvenOdd=False)
    y = b.Coeff(t)
    plt.plot(t,y,t,t*(t**2-1))
    plt.show()

    
if __name__ == "__main__":
   main()

        
        
    

    
