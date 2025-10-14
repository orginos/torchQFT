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

#dtype = tr.float32
dtype = tr.get_default_dtype()   # respects global default
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
L = reset_L().to(dtype=dtype)       # create with current default

def _L_like(s):
    global L
    if L.dtype != s.dtype or L.device != s.device:
        L = reset_L().to(dtype=s.dtype, device=s.device)
    return L

# multiplies an O(3) matrix to a vector
def GxV(R,Q):
    return tr.einsum('bkjxy,bjxy->bkxy',R,Q)

# multiplies two group  elements
def GxG(R,Q):
    return tr.einsum('bkjxy,bjmxy->bkmxy',R,Q)


def so3_expo(P):
        E=tr.eye(3,3,dtype=P.dtype,device=P.device).view(1,3,3,1,1)
        #print(E.shape)
        norm = tr.sqrt(tr.einsum('baxy,baxy->bxy',P,P))
        sin = tr.sin(norm)/norm
        sin2 = 2*(tr.sin(norm/2)/norm)**2
        #build the matrix A
        A  = tr.einsum('ija,baxy->bijxy',L,P)
        AA = tr.einsum('bikxy,bkjxy->bijxy',A,A)
        R= E + tr.einsum('bijxy,bxy->bijxy',A,sin) +  tr.einsum('bijxy,bxy->bijxy',AA,sin2)
        return R

def so3_evolve(dt,P,Q):
        R = so3_expo(dt*P)
        return  tr.einsum('bsrxy,brxy->bsxy',R,Q)

from abc import ABC, abstractmethod

class Functional(ABC):
    """
    Abstract base defining a functional interface with three methods:
      - action(x):      scalar/value of the functional at x
      - grad(x):        gradient at x
      - lapl(x):        Laplacian at x
    """

    @abstractmethod
    def action(self, x):
        """Compute the action/value at x."""
        raise NotImplementedError

    @abstractmethod
    def grad(self, x):
        """Compute the gradient at x."""
        raise NotImplementedError

    def mgrad(self,x):
        return -self.grad(x)
    
    @abstractmethod
    def lapl(self, x):
        """Compute the Laplacian at x."""
        raise NotImplementedError

    def mlapl(self,x): # minus the laplacian
        return self.lapl(x) 

    def grad_lapl(self,x):
        """Compute the grad of the Laplacian at x."""
        raise NotImplementedError
    
class Psi0(Functional):
    def action(self,s):
        # I have a batch dimension first
        A = tr.zeros(s.shape[0],device=s.device)
        # I will explicitelly make the code 2d
        for mu in range(2,s.dim()):
            A += tr.einsum('bsxy,bsxy->b',s,tr.roll(s,shifts=-1,dims=mu))
        return 2.0*A # matches the paper definition with +/- sums
    
    def grad(self,s):
        F = tr.zeros_like(s)
        Lsig = -tr.einsum('bsxy,sra->braxy',s,L)
        for mu in range(2,s.dim()):
            F+=tr.roll(s,shifts= 1,dims=mu)+tr.roll(s,shifts=-1,dims=mu)
        F=tr.einsum('bsaxy,bsxy->baxy',Lsig,F)
        
        return 2.0*F

    # this one is simple... it is an eigenfunction of the laplacian
    def lapl(self,s):
        return -4.0*self.action(s)

    def grad_lapl(self,s):
        return -4.0*self.grad(s)
    
    
class Psi2(Functional):
    def action(self,s):
        # I have a batch dimension first
        A = tr.zeros(s.shape[0],device=device)
        # first make the s1
        s1 = tr.zeros_like(s) #-( 2.0*(s.dim()-2) )*s 
        for mu in range(2,s.dim()):
            s1+=tr.roll(s,shifts=-1,dims=mu)+tr.roll(s,shifts=1,dims=mu)
        for mu in range(2,s.dim()):
            A += tr.einsum('bsxy,bsxy->b',s,tr.roll(s1,shifts=-1,dims=mu))

        V = 2.0*tr.ones(s.shape[0],device=s.device,dtype=s.dtype)
        for mu in range(2,s.dim()):
            V *= s.shape[mu] 
        # subtract the constant
        A -= V
        return 2.0*A  # matches the paper definition with +/- sums

    def grad(self,s):
        # first make the s1
        s1 = tr.zeros_like(s)#-( 2.0*(s.dim()-2) )*s 
        for mu in range(2,s.dim()):
            s1+=tr.roll(s,shifts=-1,dims=mu)+tr.roll(s,shifts=1,dims=mu)
        F = tr.zeros_like(s)
        Lsig = -tr.einsum('bsxy,sra->braxy',s,L)
        for mu in range(2,s.dim()):
            F+=tr.roll(s1,shifts= 1,dims=mu)+tr.roll(s1,shifts=-1,dims=mu)
        F=tr.einsum('bsaxy,bsxy->baxy',Lsig,F)
        
        return 2.0*F # matches the paper definition with +/- sums

    # this one is simple... it is an eigenfunction of the laplacian
    def lapl(self,s):
        return -4.0*self.action(s)

    def grad_lapl(self,s):
        return -4.0*self.grad(s)
    
class Psi11_l(Functional):
    def action(self,s):
        # I have a batch dimension first
        A = tr.zeros(s.shape[0],device=s.device)
        for mu in range(2,s.dim()):
            b = tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=mu))
            A += 2.0*tr.einsum('bxy,bxy->b',b,b) # for the backward term as well

        V = 4.0/3.0*tr.ones(s.shape[0],device=device,dtype=dtype)
        for mu in range(2,s.dim()):
            V *= s.shape[mu] 
        # subtract the constant
        A -= V
        
        return A

    def grad(self,s):
        F = tr.zeros_like(s)
        Lsig = -tr.einsum('bsxy,sra->braxy',s,L)
        for mu in range(2,s.dim()):
            bp = tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=mu))
            bm = tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=+1,dims=mu))
            F+=tr.einsum('bsxy,bxy->bsxy',tr.roll(s,shifts= 1,dims=mu),bm)
            F+=tr.einsum('bsxy,bxy->bsxy',tr.roll(s,shifts=-1,dims=mu),bp)
        F=tr.einsum('bsaxy,bsxy->baxy',Lsig,F)
        
        return 4.0*F # 2 for the power and 2 the to match the paper definition with +/- sums

    # this one is simple... it is an eigenfunction of the laplacian
    def lapl(self,s):
        return -12.0*self.action(s)

    def grad_lapl(self,s):
        return -12.0*self.grad(s)
    
class Psi11t(Functional):

#    def action(self,s):
#        # first make the bonds
#        b = tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=2) + tr.roll(s,shifts=1,dims=2))
#        for mu in range(3,s.dim()):
#            b += tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=mu) + tr.roll(s,shifts=1,dims=mu))
        
#        return tr.einsum('bxy,bxy->b',b,b)

#slow
#    def grad(self,s):
#        # first make the bonds
#        b = tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=2) + tr.roll(s,shifts=1,dims=2))
#        for mu in range(3,s.dim()):
#            b += tr.einsum('bsxy,bsxy->bxy',s,tr.roll(s,shifts=-1,dims=mu) + tr.roll(s,shifts=1,dims=mu))
#        bs = tr.einsum('bsxy,bxy->bsxy',s,b)
#        Fs = tr.zeros_like(s)
#        Fbs = tr.zeros_like(s)
#        Lsig = -tr.einsum('bsxy,sra->braxy',s,L)
#        for mu in range(2,s.dim()):
#            Fs +=tr.roll(s ,shifts= 1,dims=mu)+tr.roll(s ,shifts=-1,dims=mu)
#            Fbs+=tr.roll(bs,shifts= 1,dims=mu)+tr.roll(bs,shifts=-1,dims=mu) 
#        F = tr.einsum('baxy,bxy->baxy',Fs,b) + Fbs
#        F=tr.einsum('bsaxy,bsxy->baxy',Lsig,F)
#        
#        return -2.0*F

    def action(self,s):
        M = dot3(s,s1_field(s)).unsqueeze(1)
        return (M*M).flatten(1).sum(-1)
    
    def grad(self,s):
        s1 = s1_field(s)
        M = dot3(s,s1).unsqueeze(1)
        sM = s*M
        V = s1_field(sM)+ s1*M   
        return 2*tr.cross(s,V, dim=1)
   


    # this one is not simple...
    # I experimentally found the right constant to subract
    # and the right weights for Psi11 and Psi2
    # We need to check the normalization of what I programed and what
    # I have in the notes ... there seem to be a factor of two over all
    # scaling missing is various places
    def lapl(self,s):
        ps11_l = Psi11_l()
        ps2    = Psi2()
        A = -2.0*ps11_l.action(s) + 2.0*ps2.action(s)

        #A = A
        #print(A)
        # the constant
        V = 40.0/3.0*tr.ones(s.shape[0],device=s.device,dtype=s.dtype)
        for mu in range(2,s.dim()):
            V *= s.shape[mu] 
        # subtract the constant
        A += V
        #print(A)
        return A-10.0*self.action(s)

    def grad_lapl(self,s):
        ps11_l = Psi11_l()
        ps2    = Psi2()
        return -2.0*ps11_l.grad(s) + 2.0*ps2.grad(s) - 10.0*self.grad(s)

#this is now an eigenfunction of the Lie Laplacian.
# NOTE: I found all the coefficients experimentaly and they are not equal to those in the old notes...
# I will fix the algebra so that I get the right answer...
class Psi11(Functional):
    def __init__(self):
        self.p11t = Psi11t()
        self.p11l = Psi11_l()
        self.psi2 = Psi2()
        
    def action(self,s):
        V = np.prod(list(s.shape[2:s.dim()]))
        A = self.p11t.action(s) - self.p11l.action(s) - 1.0/3.0*self.psi2.action(s) - 4.0/3.0*V
        
        return A

    def grad(self,s):
        return self.p11t.grad(s) - self.p11l.grad(s) -1.0/3.0*self.psi2.grad(s)

    def lapl(self,s):
        return self.p11t.lapl(s) - self.p11l.lapl(s) -1.0/3.0*self.psi2.lapl(s)

    def grad_lapl(self,s):
        return self.p11t.grad_lapl(s) - self.p11l.grad_lapl(s) -1.0/3.0*self.psi2.grad_lapl(s)

# Now the basis for O(t^2) flow action:
#specialize for 2D and SO(3)
DIRS = [(1,0),(-1,0),(0,1),(0,-1)]
def roll(t,dx,dy): return tr.roll(tr.roll(t,dx,-2),dy,-1)
def dot3(a,b): return (a*b).sum(dim=1)
def s1_field(s):
    # s^(1)_x = sum over directed nearest neighbors
    return sum(roll(s, dx, dy) for dx, dy in DIRS)

def s2_field(s):
    # s^(2)_x = sum_μ ( s^(1)_{x+μ} - s_x )
    s1 = s1_field(s)
    return sum(roll(s1, dx, dy) for dx, dy in DIRS) - 4 * s

class Psi3(Functional):
    def action(self,s):
        acc=0; 
        for d1 in DIRS:
            for d2 in DIRS:
                for d3 in DIRS:
                    acc += dot3(s, roll(s,d1[0]+d2[0]+d3[0], d1[1]+d2[1]+d3[1]))
        return acc.flatten(1).sum(-1)

    def grad(self,s):
        f = tr.zeros_like(s)
        for x1,y1 in DIRS:
            for x2,y2 in DIRS:
                for x3,y3 in DIRS:
                    f -= tr.cross(roll(s,x1+x2+x3, y1+y2+y3),s,dim=1)
        return 2.0*f
    
    def lapl(self,s):
        return -4.0*self.action(s) 

    def grad_lapl(self,s):
        return -4.0*self.grad(s)
    
class Psi21(Functional):
    def action(self,s):
        acc = 0.0
        for dx1, dy1 in DIRS:
            for dx2, dy2 in DIRS:
                d12 = dot3(s, roll(s, dx1+dx2, dy1+dy2))
                for dx3, dy3 in DIRS:
                    acc = acc + d12 * dot3(s, roll(s, dx3, dy3))
        return acc.flatten(1).sum(-1)


    def grad(self,s):
        s1 = s1_field(s)
        s2 = s1_field(s1) #- 4.0*s
        #print(s2.shape,(dot3(s,s1).unsqueeze(1)).shape)
        d1 = dot3(s,s1).unsqueeze(1)
        d2 = dot3(s,s2).unsqueeze(1)
        V  = s2*d1+sum(roll(sum(roll(s*d1,x,y) for x,y in DIRS),x,y) for x,y in DIRS)
        V += s1*d2 + sum(roll(s*d2,x,y) for x,y in DIRS) 
        f = tr.cross(s,V,dim=1)
        return f
    
    def lapl(self,s):
        return -(10*self.action(s) - 2*Psi3().action(s) -16*Psi0().action(s))

    def grad_lapl(self,s):
        return -(10*self.grad(s) - 2*Psi3().grad(s) -16*Psi0().grad(s))

class Psi12d(Functional):
    def action(self,s):
        s1 = s1_field(s)
        acc = dot3(s,s1)*dot3(s1,s1)
        return acc.flatten(1).sum(-1)
    
    def grad(self,s):
        s1 = s1_field(s)
        Q = dot3(s1,s1).unsqueeze(1)
        M = dot3(s,s1).unsqueeze(1)
        V = s1*Q + sum(roll(Q*s+2.0*M*s1,x,y) for x,y in DIRS)
        f = tr.cross(s,V,dim=1)
        return f
    
    def lapl(self,s):
        return -(8*self.action(s) -32*Psi0().action(s) +4*Psi12l().action(s))

    def grad_lapl(self,s):
        return -(8*self.grad(s) -32*Psi0().grad(s) +4*Psi12l().grad(s))

    
class Psi111(Functional):
    def action(self,s):
        s1 = s1_field(s)
        acc = dot3(s,s1)**3
        return acc.flatten(1).sum(-1)
    
    def grad(self,s):
        s1 = s1_field(s)
        M2 = dot3(s,s1).unsqueeze(1)**2
        V = s1*M2 + sum(roll(s*M2,x,y) for x,y in DIRS)
        f = 3.0*tr.cross(s,V,dim=1)
        return f
    
    def lapl(self,s):
        return -(18*self.action(s) - 6*Psi12d().action(s) -24*Psi0().action(s) + 6*Psi1l1().action(s))

    def grad_lapl(self,s):
        return -(18*self.grad(s) - 6*Psi12d().grad(s) -24*Psi0().grad(s) + 6*Psi1l1().grad(s))

   
    
class Psi111c(Functional):
    def action(self,s):
        acc = 0.0
        for dx1, dy1 in DIRS:
            s1 = roll(s, dx1, dy1); b1 = dot3(s, s1)
            for dx2, dy2 in DIRS:
                s2 = roll(s1, dx2, dy2); b2 = dot3(s1, s2)
                for dx3, dy3 in DIRS:
                    s3 = roll(s2, dx3, dy3); b3 = dot3(s2, s3)
                    acc = acc + b1 * b2 * b3
        return acc.flatten(1).sum(-1)


    def grad(self,s):
        """
        Analytic Lie derivative field Xi for psi111_c:
        Xi[:, a, x, y] = ∂_x^a psi111_c(s)  for a=0,1,2 (x-,y-,z- generators).
        Uses the identity ∂_x^a F = (L^a s_x)·(∂F/∂s_x) = (s_x × grad_x F)_a.

        Args:
             s: [B, 3, Lx, Ly], unit vectors along dim=1 (O(3) spins).

        Returns:
             Xi: [B, 3, Lx, Ly], Lie-derivative components at each site.
        """
         
        V = tr.zeros_like(s)  # sitewise gradient dF/ds_x

        for dx1, dy1 in DIRS:
            s1 = roll(s, dx1, dy1); b0 = dot3(s,  s1)
            for dx2, dy2 in DIRS:
                s2 = roll(s1, dx2, dy2); b1 = dot3(s1, s2)
                for dx3, dy3 in DIRS:
                    s3 = roll(s2, dx3, dy3); b2 = dot3(s2, s3)

                    p12 = (b1 * b2).unsqueeze(1)
                    p02 = (b0 * b2).unsqueeze(1)
                    p01 = (b0 * b1).unsqueeze(1)

                    # y0 = x
                    V = V + s1 * p12

                    # y1 = x - (dx1, dy1)
                    to_y1 = s * p12 + s2 * p02
                    V = V + roll(to_y1, -dx1, -dy1)

                    # y2 = x - (dx1+dx2, dy1+dy2)
                    to_y2 = s1 * p02 + s3 * p01
                    V = V + roll(to_y2, -(dx1 + dx2), -(dy1 + dy2))

                    # y3 = x - (dx1+dx2+dx3, dy1+dy2+dy3)
                    to_y3 = s2 * p01
                    V = V + roll(to_y3, -(dx1 + dx2 + dx3), -(dy1 + dy2 + dy3))

        # Lie components: ∂^a psi = (L^a s)·V = (s × V)_a
        return tr.cross(s,V, dim=1)  # [B, 3, Lx, Ly]

    
    def lapl(self,s):
        return -(16*self.action(s) -4*Psi21().action(s) -16*Psi0().action(s) -4*Psi12l().action(s) + 8*Psi1l1().action(s))

    def grad_lapl(self,s):
        return -(16*self.grad(s) -4*Psi21().grad(s) -16*Psi0().grad(s) -4*Psi12l().grad(s) + 8*Psi1l1().grad(s))
    
class Psi12l(Functional):

    def action(self,s):
        return sum(dot3(s,roll(s,x,y))*sum(dot3(s,roll(s,x+w,y+z)) for w,z in DIRS) for x,y in DIRS).flatten(1).sum(-1)
    
    def grad(self,s):
        V  = sum(roll(s,x,y)*sum(dot3(s,roll(s,x+w,y+z)).unsqueeze(1) for w,z in DIRS) for x,y in DIRS)
        V += sum(roll(s,x,y)*sum(dot3(roll(s,x,y),roll(s,w,z)).unsqueeze(1) for w,z in DIRS) for x,y in DIRS)
        V += sum((dot3(s,roll(s,x,y)).unsqueeze(1)*sum(roll(s,x+w,y+z) for w,z in DIRS)) for x,y in DIRS)
        V += s1_field(sum((dot3(s,roll(s,x,y)).unsqueeze(1)*roll(s,x,y)) for x,y in DIRS))
        return tr.cross(s,V, dim=1)
        
    def lapl(self,s):
        return -(10*self.action(s) - 12*Psi0().action(s))

    def grad_lapl(self,s):
        return -(10*self.grad(s) - 12*Psi0().grad(s))
  
class Psi1l1(Functional):
    def action(self,s):
        bonds = [dot3(s, roll(s, x, y)) for x, y in DIRS]
        acc = sum(b**2 for b in bonds)*sum(b for b in bonds)
        return acc.flatten(1).sum(-1)
    
    def B(self,s,x,y):
        return dot3(s,roll(s,x,y)).unsqueeze(1)

    def grad(self,s):
        s1 = s1_field(s)
        M = dot3(s,s1).unsqueeze(1)
        K = sum(self.B(s,x,y)**2 for x,y in DIRS)
        V  = 2.0*sum(self.B(s,x,y)*roll(s,x,y) for x,y in DIRS)*M
        V += K*s1
        V += sum((2*self.B(s,x,y)*roll(M,x,y)+ roll(K,x,y))*roll(s,x,y) for x,y in DIRS)
        
        return tr.cross(s,V, dim=1)
   
    def lapl(self,s):
        return -(20*self.action(s) -4*Psi12l().action(s) +4*Psi111l().action(s) -20*Psi0().action(s))

    def grad_lapl(self,s):
        return -(20*self.grad(s) -4*Psi12l().grad(s) +4*Psi111l().grad(s) -20*Psi0().grad(s))

class Psi111l(Functional):
    def action(self,s):
        acc = sum(dot3(s,roll(s,x,y))**3 for x,y in DIRS)
        return acc.flatten(1).sum(-1)
        
    def B(self,s,x,y):
        return dot3(s,roll(s,x,y)).unsqueeze(1)

    def grad(self,s):
        V = sum(self.B(s,x,y)**2*roll(s,x,y) + self.B(s,-x,-y)**2*roll(s,-x,-y) for x,y in DIRS)  
        return 3.0*tr.cross(s,V, dim=1)     
       
    def lapl(self,s):
        return -(24*self.action(s) - 12*Psi0().action(s))

    def grad_lapl(self,s):
        return -(24*self.grad(s) - 12*Psi0().grad(s))
 
    
class FlowAction(nn.Module):
    def __init__(self,cc=[Psi0()]):
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

    def grad_lapl(self,s,t):
        A = tr.zeros(s.shape[0],device=device,dtype=dtype)
        for p in self.T:
            A += p.grad_lapl(s,t)
        return A


def LieDeriv(func,s):
    r = s.detach().clone().requires_grad_(True)
    #print(r.shape)
    F = func(r)
    g = tr.autograd.grad(F,r,grad_outputs=tr.ones_like(F), create_graph=True)[0]
    return tr.cross(s, g, dim=1)
    #return tr.einsum('bi...,aij,bj...->ba...',g,L,s) # slower

def _distance_R_coloring(Lx, Ly, R=2, pbc=True):
    """Greedy distance-R coloring on an Lx×Ly torus (pbc=True) or grid (pbc=False).
    Returns colors[ Lx, Ly ] with values in {0,...,C-1} and C."""
    colors = [[-1]*Ly for _ in range(Lx)]
    # Manhattan offsets within radius R (excluding self)
    offs = [(dx,dy) for dx in range(-R,R+1) for dy in range(-R,R+1)
            if (abs(dx)+abs(dy) <= R and not (dx==0 and dy==0))]
    C = 0
    for i in range(Lx):
        for j in range(Ly):
            forbidden = set()
            for dx,dy in offs:
                ii = (i+dx) % Lx if pbc else i+dx
                jj = (j+dy) % Ly if pbc else j+dy
                if 0 <= ii < Lx and 0 <= jj < Ly:
                    c = colors[ii][jj]
                    if c != -1:
                        forbidden.add(c)
            # smallest non-negative not in forbidden
            c = 0
            while c in forbidden: c += 1
            colors[i][j] = c
            if c+1 > C: C = c+1
    #import numpy as np
    return tr.from_numpy(np.array(colors, dtype='int64')), C

def LieLaplacian(func, s, colors, *, eps=1e-12):
    """
    Exact physics-sign Lie Laplacian L^2 F = -Δ_{S^2} F using distance-R probing via a
    precomputed coloring.

    Args
    ----
    func   : callable(r) -> tensor[B], differentiable; one scalar per batch sample.
    s      : tensor[B, 3, Lx, Ly], unit spins (approximately). Channel-first.
    colors : tensor or numpy array [Lx, Ly] of integer color ids. Sites with the same
             color must be separated by Manhattan distance > R (>= d_min) that covers
             the support of H_xy. Color ids can be 0..C-1 or 1..C.

    Returns
    -------
    L2        : tensor[B], exact L^2 F per batch.
    n_colors  : int, number of distinct colors actually used.

    Notes
    -----
    - Complexity: 2 * n_colors HVPs (vs 2 * Lx * Ly in the site-local method).
    - This is exact provided the coloring radius matches the operator range.
    - Uses float64 for stable second derivatives.
    """
    # ---- prep & first derivatives
    r = s.detach().to(tr.float64).requires_grad_(True)
    B, C, Lx, Ly = r.shape
    assert C == 3, "Expected channel-first spins [B, 3, Lx, Ly]."

    # Colors -> torch on device, zero-based ids, and shape check
    if not tr.is_tensor(colors):
        colors = tr.from_numpy(colors)
    colors = colors.to(device=r.device)
    assert colors.shape == (Lx, Ly), (
        f"colors shape {tuple(colors.shape)} must match spatial shape (Lx, Ly)=({Lx},{Ly}). "
        "If your generator returns (Ny, Nx), pass colors.T or reorder accordingly."
    )
    colors = colors - colors.min()              # ensure start at 0
    n_colors = int(colors.max().item()) + 1

    # One scalar per batch
    Fb = func(r)                                # [B]
    g, = tr.autograd.grad(Fb.sum(), r, create_graph=True)  # [B,3,Lx,Ly]

    # ---- tangent frame at each site (orthonormal), then FREEZE directions
    ex = tr.zeros_like(r); ex[:, 0] = 1.0
    ey = tr.zeros_like(r); ey[:, 1] = 1.0
    t1 = tr.cross(r, ex, dim=1)
    need_fb = t1.norm(dim=1, keepdim=True) < 1e-8
    t1 = tr.where(need_fb, tr.cross(r, ey, dim=1), t1)
    t1 = t1 / (t1.norm(dim=1, keepdim=True) + eps)
    t2 = tr.cross(r, t1, dim=1)
    t1 = t1.detach(); t2 = t2.detach()

    sum_all = (1, 2, 3)                         # sum over spin+space
    trHP = tr.zeros(B, dtype=r.dtype, device=r.device)

    # ---- probe per color: EXACT diag block trace (no same-color couplings within range)
    for c in range(n_colors):
        mask = (colors == c).view(1, 1, Lx, Ly).to(dtype=r.dtype, device=r.device)
        v1 = (t1 * mask).detach()
        v2 = (t2 * mask).detach()

        dot1 = (g * v1).sum()
        Hv1, = tr.autograd.grad(dot1, r, retain_graph=True)
        trHP += (Hv1 * v1).sum(dim=sum_all)

        dot2 = (g * v2).sum()
        Hv2, = tr.autograd.grad(dot2, r, retain_graph=True)
        trHP += (Hv2 * v2).sum(dim=sum_all)

    # ---- Δ = tr(H P) - 2 s·g, then L^2 = -Δ
    sdotg = (r * g).sum(dim=sum_all)            # [B]
    L2 = 2.0 * sdotg - trHP
    return L2.detach()


def LieLaplacian_stoch(
    func,
    s,
    colors,
    *,
    nsamples: int = 1,
    eps: float = 1e-12,
    dtype=tr.float64,
):
    """
    Unbiased (and exact when coloring radius is sufficient) estimator of
    L^2 F = -Δ_{S^2} F for a batched scalar functional on a product of S^2.

    Assumptions:
      - Channel-first spins: s: [B, 3, Lx, Ly], approximately unit.
      - func(s) -> [B] (one scalar per batch).
      - 'colors': [Lx, Ly] integer ids (0..C-1 or arbitrary ints). All sites with the same
        color should be separated by Manhattan distance > operator range for exactness.

    Args:
      nsamples : number of Hutchinson draws (variance ↓ as 1/nsamples).
      eps      : small number for tangent normalization.
      dtype    : float64 recommended for stable second derivatives.

    Returns:
      L2_mean : [B] estimate of L^2 F
      stats   : dict with
                  'std'     : [B] (sample std, unbiased; 0 if nsamples==1)
                  'se'      : [B] (std error; 0 if nsamples==1)
                  'ncolors' : int
    """
    # -- prep
    r = s.detach().to(dtype).requires_grad_(True)
    B, C, Lx, Ly = r.shape
    assert C == 3, "Expected channel-first spins [B,3,Lx,Ly]."

    # Normalize/pack colors -> torch, on device, 0..ncolors-1
    if not tr.is_tensor(colors):
        colors = tr.as_tensor(colors)
    colors = colors.to(device=r.device)
    assert colors.shape == (Lx, Ly), \
        f"colors shape {tuple(colors.shape)} must match (Lx, Ly)=({Lx},{Ly})."
    # compress ids to 0..K-1
    uniq = tr.unique(colors)
    #print(uniq)
    remap = {int(u.item()): i for i, u in enumerate(uniq)}
    tt = tr.empty_like(colors, dtype=tr.int64)
    for k, v in remap.items():
        tt[(colors == k)] = v
    ncolors = int(tt.max().item()) + 1
    colors = tt
    #print(colors)
    #print(ncolors)
    # -- first derivatives
    Fb = func(r)                                       # [B]
    g, = tr.autograd.grad(Fb.sum(), r, create_graph=True)

    # -- tangent frame at each site, freeze the directions
    ex = tr.zeros_like(r); ex[:, 0] = 1.0
    ey = tr.zeros_like(r); ey[:, 1] = 1.0
    t1 = tr.cross(r, ex, dim=1)
    need_fb = t1.norm(dim=1, keepdim=True) < 1e-8
    t1 = tr.where(need_fb, tr.cross(r, ey, dim=1), t1)
    t1 = t1 / (t1.norm(dim=1, keepdim=True) + eps)
    t2 = tr.cross(r, t1, dim=1)
    t1 = t1.detach(); t2 = t2.detach()

    sum_all = (1, 2, 3)
    sdotg = (r * g).sum(dim=sum_all)                   # [B]

    per_sample = []
    for s_idx in range(nsamples):
        trHP = tr.zeros(B, dtype=r.dtype, device=r.device)

        for c in range(ncolors):
            mask = (colors == c).view(1, 1, Lx, Ly).to(dtype=r.dtype, device=r.device)

            # Z2 noise per site (independent across batch & sites)
            z1 = (tr.randint(0, 2, (B, 1, Lx, Ly), device=r.device) * 2 - 1).to(r.dtype)
            z2 = (tr.randint(0, 2, (B, 1, Lx, Ly), device=r.device) * 2 - 1).to(r.dtype)

            v1 = (t1 * (mask * z1)).detach()          # uses only t1 -> exact w/ perfect coloring
            v2 = (t2 * (mask * z2)).detach()          # uses only t2 -> exact w/ perfect coloring

            # H v1
            dot1 = (g * v1).sum()
            Hv1, = tr.autograd.grad(dot1, r, retain_graph=True)
            trHP += (Hv1 * v1).sum(dim=sum_all)

            # H v2
            last_color = (c == ncolors - 1)
            last_sample = (s_idx == nsamples - 1)
            retain = not (last_color and last_sample)  # free graph at the very end
            dot2 = (g * v2).sum()
            Hv2, = tr.autograd.grad(dot2, r, retain_graph=retain)
            trHP += (Hv2 * v2).sum(dim=sum_all)

        L2_est = 2.0 * sdotg - trHP                    # [B]
        per_sample.append(L2_est)

    if nsamples == 1:
        return per_sample[0].detach(), {'std': tr.zeros_like(sdotg), 'se': tr.zeros_like(sdotg), 'ncolors': ncolors}
    else:
        L2_stack = tr.stack(per_sample, dim=0)         # [S, B]
        L2_mean = L2_stack.mean(dim=0)
        # Unbiased std over samples; avoid NaN for S=1 guarded above
        L2_std = L2_stack.std(dim=0, unbiased=True)
        L2_se  = L2_std / (nsamples ** 0.5)
        return L2_mean.detach(), {'std': L2_std.detach(), 'se': L2_se.detach(), 'ncolors': ncolors}


def NumericalLaplacian(s,action,eps):
    #print(s.size()[2:4])
    
    Lap = tr.zeros(s.shape[0],device=device)
    A = action(s,1.0);
    for a in range(3):
        for x in range(s.size()[2]):
            for y in range(s.size()[3]):
                f = tr.ones_like(s)*1.0e-15
                f[:,a,x,y]= tr.ones(s.shape[0],device=device)
                #print(f.size())
                sp = so3_evolveQ(eps,f,s)
                sm = so3_evolveQ(-eps,f,s)
                Ap = action(sp,1.0)
                Am = action(sm,1.0)
                #print(Am,Ap,A)
                Lap += (Ap + Am - 2*A)/eps**2
    return Lap

def uniform_spin(batch,lat):
    shape = list((batch,3,*lat))
    s = tr.randn(shape)
    s = s/s.norm(dim=1,keepdim=True).clamp_min(1e-14)
    return s
    
import time
def testDeriv(*Funcs):
    beta=1.2345
    lat = [8,8]
    Nd = len(lat)
    Vol = np.prod(lat)
    Bs = 32
    sigma = uniform_spin(Bs,lat).to(dtype=dtype)

    print("Working lattice: ",sigma.shape)

    for f in Funcs:
        autoG = LieDeriv(f.action,sigma)
        g = f.grad(sigma)
        r = (autoG-g).norm()/g.norm()
        print("Testing gradient for: ",f.__class__.__name__,r.item())

    h = 0.01
    for f in Funcs:
        g=f.grad(sigma)
        rs = so3_evolve(h,g,sigma)
        delta = ((f.action(rs) - f.action(sigma))/h).detach()
        sign = (delta/delta.abs()).mean().item()
        print("Testing gradient sign for: ",f.__class__.__name__,sign)
                 
    for f in Funcs:
        tic = time.perf_counter()
        for k in range(1000):
            LieDeriv(f.action,sigma)
        toc = time.perf_counter()
        print(f.__class__.__name__,": ",f"Elapsed time: {toc - tic:.6f} msec")


def testLaplacian():
    beta=1.2345
    lat = [8,8]
    Nd = len(lat)
    Vol = np.prod(lat)
    Bs = 32
    sigma = uniform_spin(Bs,lat)

    print("Working lattice: ",sigma.shape)
    colors, ncolors = _distance_R_coloring(lat[0], lat[1], R=3)  # [Lx,Ly]
    colors = colors.to(dtype=tr.int64, device=sigma.device)

    for f in (Psi0(),Psi2(),Psi11_l(),Psi11t(),Psi11(),Psi3(),Psi21(),Psi12d(),Psi111(),Psi111l(),Psi12l(),Psi111c(),Psi1l1()):
        autoLap = LieLaplacian(f.action,sigma,colors)
        lap = f.lapl(sigma)
        r = (autoLap+lap).norm()/lap.norm()
        eig = (autoLap/f.action(sigma)).mean()
        print("Testing laplacian for: ",f.__class__.__name__,f"Ev:\t {eig.item():4.4f} Resid: {r.item():.6e}")

    for f in (Psi0(),Psi2(),Psi11_l(),Psi11t(),Psi11(),Psi3()):
        tic = time.perf_counter()
        for k in range(1000):
            LieLaplacian(f.action,sigma,colors)
        toc = time.perf_counter()
        print(f.__class__.__name__,": ",f"Elapsed time: {toc - tic:.6f} msec")


def testGradLapl(*Funcs):
    beta=1.2345
    lat = [8,8]
    Nd = len(lat)
    Vol = np.prod(lat)
    Bs = 32
    sigma = uniform_spin(Bs,lat).to(dtype=dtype)

    print("Working lattice: ",sigma.shape)

    for f in Funcs:
        autoG = LieDeriv(f.lapl,sigma)
        g = f.grad_lapl(sigma)
        r = (autoG-g).norm()/g.norm()
        print("Testing gradient of the Laplacian for: ",f.__class__.__name__, " : ",r.item())

                 
    for f in Funcs:
        tic = time.perf_counter()
        for k in range(1000):
            LieDeriv(f.lapl,sigma)
        toc = time.perf_counter()
        print(f.__class__.__name__,": ",f"Elapsed time: {toc - tic:.6f} msec")

    
from dataclasses import dataclass,field
from typing import List
import matplotlib.pyplot as plt

@dataclass
class Starget:
    beta : float
    coef : float = field(init=False)
    
    def __post_init__(self):
        self.coef = -0.5*self.beta

    def __call__(self,s):
        return self.coef*Psi0().action(s)

    def grad(self,s):
        return self.coef*Psi0().grad(s)

    def mgrad(self,s,t):
        return -self.grad(s,t)
    
    def mlapl(self,s):
        return -self.coef*Psi0.lapl(s)

@dataclass
class Sflow0:
    beta : float
    coef : float = field(init=False)
    
    def __post_init__(self):
        self.coef = self.beta/8.0

    def __call__(self,s):
        return self.coef*Psi0().action(s)

    def grad(self,s):
        return self.coef*Psi0().grad(s)

    def mgrad(self,s,t):
        return -self.grad(s,t)
    
    def mlapl(self,s):
        return -self.coef*Psi0().lapl(s)

@dataclass
class Sflow1:
    beta : float
    c    : List[float] = field(init=False)
    psi  : List = field(init=False)
    
    def __post_init__(self):
        coef = self.beta**2/8.0
        self.c = [1.0/3.0*coef, -1.0/5.0*coef, -1.0/6.0*coef]
        self.psi = [Psi2(),Psi11(),Psi11_l()]
        
    def __call__(self,s):
        r = tr.zeros(s.shape[0],dtype=s.dtype,device=s.device)
        for c, p in zip(self.c, self.psi):
            r += c * p.action(s)
        return r


    def grad(self,s):
        r = tr.zeros_like(s)
        for c, p in zip(self.c, self.psi):
            r += c * p.grad(s)
        return r

    def mgrad(self,s,t):
        return -self.grad(s,t)
    
    def mlapl(self,s):
        r = tr.zeros(s.shape[0],dtype=s.dtype,device=s.device)
        for c, p in zip(self.c, self.psi):
            r -= c * p.lapl(s)
        return r


@dataclass
class Sflow2:
    beta   : float
    lat    : List[int]
    c      : List[float] = field(init=False)
    psi    : List = field(init=False)
    colors : tr.Tensor = field(init=False)
    
    def __post_init__(self):
        coef = self.beta**3/8.0
        self.c = [-1489.0/3000.0*coef, 
                  +  29.0/ 200.0*coef, 
                  -  11.0/ 100.0*coef,
                  -   1.0/  30.0*coef,
                  +   2.0/  90.0*coef,
                  +   1.0/  40.0*coef,
                  +  41.0/1500.0*coef,
                  -   7.0/ 300.0*coef,
                  +   7.0/1800.0*coef
                  ]
        self.psi = [Psi0(),
                    Psi3(),
                    Psi21(),
                    Psi12d(),
                    Psi111(),
                    Psi111c(),
                    Psi12l(),
                    Psi1l1(),
                    Psi111l()]

        self.colors, ncolors = _distance_R_coloring(self.lat[0], self.lat[1], R=3)  # [Lx,Ly]
        
    def __call__(self,s):
        r = tr.zeros(s.shape[0],dtype=s.dtype,device=s.device)
        for c, p in zip(self.c, self.psi):
            r += c * p.action(s)
        return r


    def grad(self,s):
        r = tr.zeros_like(s)
        for c, p in zip(self.c, self.psi):
            r += c * p.grad(s)
        return r

    def mgrad(self,s,t):
        return -self.grad(s,t)
    
    def mlapl(self,s):
        colors = self.colors.to(dtype=tr.int64, device=s.device)
        autoLap = LieLaplacian(self.__call__,s,colors)
        return autoLap


@dataclass
class SflowO1:
    beta : float
    S0 : Sflow0 = field(init=False)
    S1 : Sflow1 = field(init=False)
    
    def __post_init__(self):
        self.S0 = Sflow0(self.beta)
        self.S1 = Sflow1(self.beta)
            
    def __call__(self,s,t):
        return self.S0(s) + t*self.S1(s)

    def grad(self,s,t):
        return self.S0.grad(s) + t*self.S1.grad(s)

    def mgrad(self,s,t):
        return -self.grad(s,t)
    
    def mlapl(self,s,t):
        return self.S0.mlapl(s) + t*self.S1.mlapl(s)

@dataclass
class SflowO2:
    beta : float
    lat  : List[int]
    S0 : Sflow0 = field(init=False)
    S1 : Sflow1 = field(init=False)
    S2 : Sflow2 = field(init=False)
    def __post_init__(self):
        self.S0 = Sflow0(self.beta)
        self.S1 = Sflow1(self.beta)
        self.S2 = Sflow2(self.beta,self.lat)
        
    def __call__(self,s,t):
        return self.S0(s) + t*(self.S1(s) + t*self.S2(s))

    def grad(self,s,t):
        return self.S0.grad(s) + t*(self.S1.grad(s)+ t*self.S2.grad(s))

    def mgrad(self,s,t):
        return -self.grad(s,t)
    
    def mlapl(self,s,t):
        return self.S0.mlapl(s) + t*(self.S1.mlapl(s)+ t*self.S2.mlapl(s))
    
    
def testLuscher0():
    print("Testing 0th order Luscher equation")
    beta=1.2345
    lat = [8,8]
    Nd = len(lat)
    Vol = np.prod(lat)
    Bs = 32
    sigma=uniform_spin(Bs,lat)

    colors, ncolors = _distance_R_coloring(lat[0], lat[1], R=1)  # [Lx,Ly]
    colors = colors.to(dtype=tr.int64, device=sigma.device)

    print("Working lattice: ",sigma.shape)
    #lap = LieLaplacian(Psi0().action,sigma,colors)
    #S = -0.5*beta*Psi0().action(sigma) # the right handside
    #ss = Starget(beta)
    #S = ss(sigma)
    #LL = beta/8.0 * lap # zeroth order flow-action
    #r = LL + S
    St = Starget(beta)
    Sf = Sflow0(beta)
    S = St(sigma)
    r = LieLaplacian(Sf,sigma,colors) +S
    print("Residual: ",(r.norm()/S.norm()).item())


    
def testLuscher1():
    print("Testing 1st order Luscher equation")
    beta=1.2345
    lat = [8,8]
    Nd = len(lat)
    Vol = np.prod(lat)
    Bs = 32
    sigma=uniform_spin(Bs,lat)
    
    colors, ncolors = _distance_R_coloring(lat[0], lat[1], R=2)  # [Lx,Ly]
    colors = colors.to(dtype=tr.int64, device=sigma.device)

    print("Working lattice: ",sigma.shape)
    St = Starget(beta)
    Sf0 = Sflow0(beta)
    Sf1 = Sflow1(beta)
    C1 = - 2.0/3.0 * beta**2 * Vol
    rhs = (Sf0.grad(sigma)*St.grad(sigma)).sum(tuple(range(1,sigma.dim())))
    #r = LieLaplacian(Sf1,sigma,colors) + rhs -C1
    r = Sf1.mlapl(sigma) + rhs - C1
    #print(r)
    print("Residual: ",(r.norm()/rhs.norm()).item())



def testLuscher2():
    print("Testing 2nd order Luscher equation")
    beta=1.2345
    lat = [8,8]
    Nd = len(lat)
    Vol = np.prod(lat)
    Bs = 32
    sigma=uniform_spin(Bs,lat)
    
    print("Working lattice: ",sigma.shape)
    St = Starget(beta)
    #Sf0 = Sflow0(beta)
    Sf1 = Sflow1(beta)
    Sf2 = Sflow2(beta,lat)
    #C1 = - 2.0/3.0 * beta**2 * Vol
    C2  = 0 
    rhs = (Sf1.grad(sigma)*St.grad(sigma)).sum(tuple(range(1,sigma.dim())))
    #r = LieLaplacian(Sf1,sigma,colors) + rhs -C1
    lap = Sf2.mlapl(sigma)
    r = lap + rhs
    #r -= r.min()
    print("mLap: ",lap)
    print("dot : ",rhs)
    print("res : ",r)
    print("std res: ",r.std().item())
    print("Residual: ",(r.norm()/rhs.norm()).item())

    
def testLuscher():
    print("Testing Luscher equation using O(1) flow")
    beta=1.2345
    lat = [8,8]
    Nd = len(lat)
    Vol = np.prod(lat)
    Bs = 64
    sigma=uniform_spin(Bs,lat)
    
    print("Working lattice: ",sigma.shape)
    St = Starget(beta)
    Sf = SflowO1(beta)
    C1 = - 2.0/3.0 * beta**2 * Vol
    r=[]
    std = []
    ftime = np.linspace(0,1.0,20)
    for t in ftime:
        rr = St(sigma) + Sf.mlapl(sigma,t) + t*(St.grad(sigma)*Sf.grad(sigma,t)).sum(tuple(range(1,sigma.dim())))-t*C1
        r.append(rr.norm().item()/sigma.shape[0])
        std.append(rr.std().item()/np.sqrt(sigma.shape[0]))
    print(ftime,r,std)
    p = np.polyfit(ftime, r, deg=2)
    print("Fitted polynomium: ",p)
    x = np.linspace(0,1.0,100)
    y = np.polyval(p,x)
    sp = np.polyfit(ftime,std,deg=2)
    dy = np.polyval(sp,x)
    
    plt.plot(x,y,color='orange')
    # plot shaded error band
    plt.fill_between(x, y - dy, y + dy, color='orange', alpha=0.3)
    plt.plot(ftime, r,'.',color='red')
    
    #plt.show()

    #second order action
    print("Testing Luscher equation using O(2) flow")
    Sf2 = SflowO2(beta,lat)
    r=[]
    std=[]
    for t in ftime:
        rr = St(sigma) + Sf2.mlapl(sigma,t) + t*(St.grad(sigma)*Sf2.grad(sigma,t)).sum(tuple(range(1,sigma.dim())))-t*C1
        r.append(rr.norm().item()/sigma.shape[0])
        std.append(rr.std().item()/np.sqrt(sigma.shape[0]))
    
    p = np.polyfit(ftime, r, deg=3)
    x = np.linspace(0,1.0,100)
    y = np.polyval(p,x)
    sp = np.polyfit(ftime,std,deg=3)
    dy = np.polyval(sp,x)
    print("Fitted polynomium: ",p)
    plt.plot(x,y,color='cyan')
    # plot shaded error band
    plt.fill_between(x, y - dy, y + dy, color='cyan', alpha=0.3)
    plt.plot(ftime, r,'.',color='blue')
    
    plt.show()
    
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
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='deriv')
    
    args = parser.parse_args()

    if(args.t=='deriv'):
        print("Testing Lie Derivative")
        testDeriv(Psi0(),Psi2(),Psi11_l(),Psi11t(),Psi11())
    elif(args.t=='deriv2'):
        print("Testing Lie Derivative (O(t^2))")
        testDeriv(Psi3(),Psi21(),Psi12d(),Psi111(),Psi111c(),Psi12l(),Psi1l1(),Psi111l())

    if(args.t=='grad_lapl'):
        print("Testing Lie Derivative of the Laplacian")
        testGradLapl(Psi0(),Psi2(),Psi11_l(),Psi11t(),Psi11())
    elif(args.t=='grad_lapl2'):
        print("Testing Lie Derivative of the Laplacian (O(t^2))")
        testGradLapl(Psi3(),Psi21(),Psi12d(),Psi111(),Psi111c(),Psi12l(),Psi1l1(),Psi111l())
     
    elif(args.t=='lapl'):
        print("Testing Laplacian")
        testLaplacian()
    elif(args.t=='l0'):
        testLuscher0()
    elif(args.t=='l1'):
        testLuscher1()
    elif(args.t=='l2'):
        testLuscher2()
    elif(args.t=='luscher'):
        testLuscher()
        
    else:
        print("Nothing to test")

        

        
        
    

    
# =============================================================================
# Lie–group Munthe–Kaas integrators for O(3) spins (S^2) with Jacobian tracking
# Convention: grad(s,t) -> ω(s,t) in so(3); step: s -> exp(h*hat(ω)) s
# mlapl(s,t) = -ΔS; ascent ⇒ div f = +Δ = -mlapl
# =============================================================================

import torch as tr

def _unit_retract(s, eps=1e-12):
    n = tr.clamp(tr.norm(s, dim=1, keepdim=True), min=eps)
    return s / n

def _rotate_by_omega(s, omega, h, eps=1e-12):
    a = h * omega
    theta = tr.clamp(tr.norm(a, dim=1, keepdim=True), min=eps)
    u = a / theta
    c = tr.cos(theta)
    si = tr.sin(theta)
    ux = tr.cross(u, s, dim=1)
    uds = (u * s).sum(dim=1, keepdim=True)
    return s * c + ux * si + u * uds * (1.0 - c)

def _dexpinv_so3(v, w):
    vw  = tr.cross(v, w, dim=1)
    vvw = tr.cross(v, vw, dim=1)
    return w - 0.5 * vw + (1.0/12.0) * vvw

def _call_maybe_t(obj, name, s, t):
    fn = getattr(obj, name)
    with tr.enable_grad():
        try:    return fn(s, t)
        except TypeError: return fn(s)

def _rkmk_step_lie(flow, s, t, h, butcher):
    stages = len(butcher["b"])
    ks = []
    for i in range(stages):
        v_i = tr.zeros_like(s) if i == 0 else sum(butcher["a"][i][j] * ks[j] for j in range(i))
        s_i = _rotate_by_omega(s, v_i, 1.0)
        t_i = t + butcher["c"][i] * h
        omega_i = _call_maybe_t(flow, "grad", s_i, t_i)
        k_i = _dexpinv_so3(v_i, h * omega_i)
        ks.append(k_i)
    v = sum(butcher["b"][i] * ks[i] for i in range(stages))
    s_next = _rotate_by_omega(s, v, 1.0)
    return _unit_retract(s_next)

def integrate_rkmk2(flow, s0, *, t0=0.0, t1=1.0, n_steps=200):
    butcher = {"c":[0.0,0.5],
               "a":[[0.0,0.0],[0.5,0.0]],
               "b":[0.0,1.0]}
    s=_unit_retract(s0); h=(t1-t0)/float(n_steps); t=t0
    t_grid = tr.linspace(t0,t1,n_steps+1, device=s.device, dtype=s.dtype)
    with tr.no_grad():
        for _ in range(n_steps):
            s = _rkmk_step_lie(flow,s,t,h,butcher); t += h
    return s, t_grid

# ========= Lie–group implicit midpoint (time-reversible, 2nd order) =========
# Solve u from:  u = h * ω( exp(0.5*u) s_n , t + 0.5*h )
# Advance:       s_{n+1} = exp(u) s_n
# Self-adjoint ⇒ Ψ_h^{-1} = Ψ_{-h} (to solver tolerance).
def _lgim_solve_u(flow, s, t, h, *, tol=1e-10, max_iters=50, damping=1.0):
    """
    Fixed-point iteration for u in algebra (same shape as s):
        u = h * omega( exp(0.5 u) s , t+0.5 h )
    'damping' in (0,1] for robustness; 1.0 is standard.
    """
    u = tr.zeros_like(s)
    th = t + 0.5 * h
    for _ in range(max_iters):
        s_mid = _rotate_by_omega(s, u, 0.5)                  # exp(0.5*u) s
        omega_mid = _call_maybe_t(flow, "grad", s_mid, th)   # ω(s_mid, t+0.5h)
        u_new = damping * (h * omega_mid) + (1.0 - damping) * u
        # termination on algebra increment change
        if tr.max(tr.norm(u_new - u, dim=1)) < tol:
            u = u_new
            break
        u = u_new
    return u

def integrate_lgim(flow, s0, *, t0=0.0, t1=1.0, n_steps=200, tol=1e-10, max_iters=50, damping=1.0):
    s = _unit_retract(s0); h = (t1 - t0) / float(n_steps); t = t0
    t_grid = tr.linspace(t0, t1, n_steps + 1, device=s.device, dtype=s.dtype)
    with tr.no_grad():
        for _ in range(n_steps):
            u = _lgim_solve_u(flow, s, t, h, tol=tol, max_iters=max_iters, damping=damping)
            s = _unit_retract(_rotate_by_omega(s, u, 1.0))
            t += h
    return s, t_grid

def integrate_lgim_with_logdet(flow, s0, *, t0=0.0, t1=1.0, n_steps=200, tol=1e-10, max_iters=50, damping=1.0):
    # ascent + mlapl=-Δ  ⇒  div f = +Δ = -mlapl, evaluated at the midpoint
    s = _unit_retract(s0); h = (t1 - t0) / float(n_steps); t = t0
    B = s.size(0); logdet = tr.zeros(B, device=s.device, dtype=s.dtype)
    with tr.no_grad():
        for _ in range(n_steps):
            u = _lgim_solve_u(flow, s, t, h, tol=tol, max_iters=max_iters, damping=damping)
            s_mid = _rotate_by_omega(s, u, 0.5)
            tau_mid = - _call_maybe_t(flow, "mlapl", s_mid, t + 0.5*h)  # [B]
            logdet = logdet + h * tau_mid
            s = _unit_retract(_rotate_by_omega(s, u, 1.0))
            t += h
    return s, logdet

# Back to higher orderd RK type integrators
def integrate_rkmk3(flow, s0, *, t0=0.0, t1=1.0, n_steps=200):
    butcher = {"c":[0.0,0.5,1.0],
               "a":[[0,0,0],[0.5,0,0],[-1,2,0]],
               "b":[1/6,2/3,1/6]}
    s=_unit_retract(s0); h=(t1-t0)/float(n_steps); t=t0
    t_grid = tr.linspace(t0,t1,n_steps+1, device=s.device, dtype=s.dtype)
    with tr.no_grad():
        for _ in range(n_steps):
            s = _rkmk_step_lie(flow,s,t,h,butcher); t += h
    return s, t_grid

def integrate_rkmk4(flow, s0, *, t0=0.0, t1=1.0, n_steps=200):
    butcher = {"c":[0.0,0.5,0.5,1.0],
               "a":[[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]],
               "b":[1/6,1/3,1/3,1/6]}
    s=_unit_retract(s0); h=(t1-t0)/float(n_steps); t=t0
    t_grid = tr.linspace(t0,t1,n_steps+1, device=s.device, dtype=s.dtype)
    with tr.no_grad():
        for _ in range(n_steps):
            s = _rkmk_step_lie(flow,s,t,h,butcher); t += h
    return s, t_grid

integrate_rk4 = integrate_rkmk4  # alias

# ========= Gauss–Legendre 2-stage collocation (order 4), symmetric =========
# Stages at c = 1/2 ± sqrt(3)/6; A = [[1/4, 1/4 - r],[1/4 + r, 1/4]], b=[1/2,1/2], r = sqrt(3)/6.
# MK formulation:
#   v_i = sum_j A_{ij} k_j
#   k_i = dexp^{-1}_{v_i}( h * ω( exp(v_i) s_n , t + c_i h ) )
# Then v = sum_i b_i k_i, s_{n+1} = exp(v) s_n.

def _rkmk_gl4_step(flow, s, t, h, *, tol=1e-10, max_iters=50, damping=1.0):
    r = (3.0 ** 0.5) / 6.0
    A11, A12 = 0.25, 0.25 - r
    A21, A22 = 0.25 + r, 0.25
    c1, c2 = 0.5 - r, 0.5 + r
    b1 = b2 = 0.5

    k1 = tr.zeros_like(s)
    k2 = tr.zeros_like(s)

    for _ in range(max_iters):
        v1 = A11 * k1 + A12 * k2
        v2 = A21 * k1 + A22 * k2
        s1 = _rotate_by_omega(s, v1, 1.0)
        s2 = _rotate_by_omega(s, v2, 1.0)
        t1 = t + c1 * h
        t2 = t + c2 * h
        w1 = _call_maybe_t(flow, "grad", s1, t1)
        w2 = _call_maybe_t(flow, "grad", s2, t2)
        k1_new = _dexpinv_so3(v1, h * w1)
        k2_new = _dexpinv_so3(v2, h * w2)
        # damped Picard update
        dk1 = k1_new - k1
        dk2 = k2_new - k2
        k1 = k1 + damping * dk1
        k2 = k2 + damping * dk2
        if max(tr.max(tr.norm(dk1, dim=1)).item(), tr.max(tr.norm(dk2, dim=1)).item()) < tol:
            break

    v = b1 * k1 + b2 * k2
    s_next = _unit_retract(_rotate_by_omega(s, v, 1.0))
    return s_next

def integrate_rkmk_gl4(flow, s0, *, t0=0.0, t1=1.0, n_steps=200, tol=1e-10, max_iters=50, damping=1.0):
    s = _unit_retract(s0); h = (t1 - t0) / float(n_steps); t = t0
    t_grid = tr.linspace(t0, t1, n_steps + 1, device=s.device, dtype=s.dtype)
    with tr.no_grad():
        for _ in range(n_steps):
            s = _rkmk_gl4_step(flow, s, t, h, tol=tol, max_iters=max_iters, damping=damping)
            t += h
    return s, t_grid

def integrate_rkmk_gl4_with_logdet(flow, s0, *, t0=0.0, t1=1.0, n_steps=200, tol=1e-10, max_iters=50, damping=1.0):
    # ascent + mlapl=-Δ  ⇒  per-step τ = h * (b1*tau1 + b2*tau2), tau_i = -mlapl(s_i, t_i)
    r = (3.0 ** 0.5) / 6.0
    A11, A12 = 0.25, 0.25 - r
    A21, A22 = 0.25 + r, 0.25
    c1, c2 = 0.5 - r, 0.5 + r
    b1 = b2 = 0.5

    s = _unit_retract(s0); h = (t1 - t0) / float(n_steps); t = t0
    B = s.size(0); logdet = tr.zeros(B, device=s.device, dtype=s.dtype)
    with tr.no_grad():
        for _ in range(n_steps):
            # coupled stage solve
            k1 = tr.zeros_like(s); k2 = tr.zeros_like(s)
            for _it in range(max_iters):
                v1 = A11 * k1 + A12 * k2
                v2 = A21 * k1 + A22 * k2
                s1 = _rotate_by_omega(s, v1, 1.0)
                s2 = _rotate_by_omega(s, v2, 1.0)
                t1 = t + c1 * h
                t2 = t + c2 * h
                w1 = _call_maybe_t(flow, "grad", s1, t1)
                w2 = _call_maybe_t(flow, "grad", s2, t2)
                k1_new = _dexpinv_so3(v1, h * w1)
                k2_new = _dexpinv_so3(v2, h * w2)
                dk1 = k1_new - k1; dk2 = k2_new - k2
                k1 = k1 + damping * dk1; k2 = k2 + damping * dk2
                if max(tr.max(tr.norm(dk1, dim=1)).item(), tr.max(tr.norm(dk2, dim=1)).item()) < tol:
                    break
            # accumulate divergence at stage states
            tau1 = - _call_maybe_t(flow, "mlapl", s1, t1)  # [B]
            tau2 = - _call_maybe_t(flow, "mlapl", s2, t2)  # [B]
            logdet = logdet + h * (b1 * tau1 + b2 * tau2)
            # advance
            v = b1 * k1 + b2 * k2
            s = _unit_retract(_rotate_by_omega(s, v, 1.0))
            t += h
    return s, logdet

# ---------- with log|det J| ----------
_FLOW_SIGN_FOR_LOGDET = -1.0  # ascent with mlapl = -Δ

def _rkmk_step_lie_with_logdet(flow, s, t, h, butcher):
    ks, taus = [], []
    stages = len(butcher["b"])
    for i in range(stages):
        v_i = tr.zeros_like(s) if i==0 else sum(butcher["a"][i][j]*ks[j] for j in range(i))
        s_i = _rotate_by_omega(s, v_i, 1.0)
        t_i = t + butcher["c"][i]*h
        omega_i = _call_maybe_t(flow, "grad", s_i, t_i)
        k_i = _dexpinv_so3(v_i, h*omega_i)
        ks.append(k_i)
        tau_i = _FLOW_SIGN_FOR_LOGDET * _call_maybe_t(flow, "mlapl", s_i, t_i)  # [B]
        taus.append(tau_i)
    v = sum(butcher["b"][i]*ks[i] for i in range(stages))
    s_next = _unit_retract(_rotate_by_omega(s, v, 1.0))
    tau_step = sum(butcher["b"][i]*taus[i] for i in range(stages))  # [B]
    return s_next, tau_step

def integrate_rkmk2_with_logdet(flow, s0, *, t0=0.0, t1=1.0, n_steps=200):
    butcher = {"c":[0.0,0.5],
               "a":[[0.0,0.0],[0.5,0.0]],
               "b":[0.0,1.0]}
    s=_unit_retract(s0); h=(t1-t0)/float(n_steps); t=t0
    logdet = tr.zeros(s.size(0), device=s.device, dtype=s.dtype)
    with tr.no_grad():
        for _ in range(n_steps):
            s, tau = _rkmk_step_lie_with_logdet(flow, s, t, h, butcher)
            logdet = logdet + h * tau
            t += h
    return s, logdet

def integrate_rkmk3_with_logdet(flow, s0, *, t0=0.0, t1=1.0, n_steps=200):
    butcher = {"c":[0.0,0.5,1.0],
               "a":[[0,0,0],[0.5,0,0],[-1,2,0]],
               "b":[1/6,2/3,1/6]}
    s=_unit_retract(s0); h=(t1-t0)/float(n_steps); t=t0
    logdet = tr.zeros(s.size(0), device=s.device, dtype=s.dtype)
    with tr.no_grad():
        for _ in range(n_steps):
            s, tau = _rkmk_step_lie_with_logdet(flow, s, t, h, butcher)
            logdet = logdet + h * tau
            t += h
    return s, logdet

def integrate_rkmk4_with_logdet(flow, s0, *, t0=0.0, t1=1.0, n_steps=200):
    butcher = {"c":[0.0,0.5,0.5,1.0],
               "a":[[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]],
               "b":[1/6,1/3,1/3,1/6]}
    s=_unit_retract(s0); h=(t1-t0)/float(n_steps); t=t0
    logdet = tr.zeros(s.size(0), device=s.device, dtype=s.dtype)
    with tr.no_grad():
        for _ in range(n_steps):
            s, tau = _rkmk_step_lie_with_logdet(flow, s, t, h, butcher)
            logdet = logdet + h * tau
            t += h
    return s, logdet

# ---------- Dormand–Prince 5(4) MK (adaptive) ----------
def integrate_rkmk_dp5(flow, s0, *, t0=0.0, t1=1.0, rtol=1e-5, atol=1e-7,
                        h_init=None, h_min=None, h_max=None, safety=0.9,
                        max_steps=100000, record=False):
    c = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
    a = [[0,0,0,0,0,0,0],
         [1/5,0,0,0,0,0,0],
         [3/40,9/40,0,0,0,0,0],
         [44/45,-56/15,32/9,0,0,0,0],
         [19372/6561,-25360/2187,64448/6561,-212/729,0,0,0],
         [9017/3168,-355/33,46732/5247,49/176,-5103/18656,0,0],
         [35/384,0,500/1113,125/192,-2187/6784,11/84,0]]
    b5 = [35/384,0,500/1113,125/192,-2187/6784,11/84,0]
    b4 = [5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40]

    s=_unit_retract(s0); device, dtype=s.device, s.dtype
    t=t0; direction=1.0 if t1>=t0 else -1.0; T=abs(t1-t0)

    if h_init is None: h = T/50.0 if T>0 else 1.0
    else: h = abs(h_init)
    if h_min is None: h_min = T/10_000.0 if T>0 else 1e-6
    if h_max is None: h_max = T/5.0 if T>0 else 1.0

    steps=0
    ts=[t] if record else None
    states=[s.clone()] if record else None

    with tr.no_grad():
        while (t-t0)*direction < T and steps < max_steps:
            h = min(h, (t1-t)*direction)
            ks=[]
            for i in range(7):
                v_i = sum(a[i][j]*ks[j] for j in range(i)) if i>0 else tr.zeros_like(s)
                s_i = _rotate_by_omega(s, v_i, 1.0)
                t_i = t + c[i]*h
                omega_i = _call_maybe_t(flow, "grad", s_i, t_i)
                k_i = _dexpinv_so3(v_i, h*omega_i)
                ks.append(k_i)

            v5 = sum(b5[i]*ks[i] for i in range(7))
            v4 = sum(b4[i]*ks[i] for i in range(7))
            e_v = v5 - v4
            scale = atol + rtol * tr.maximum(tr.norm(v5, dim=1, keepdim=True), tr.tensor(1.0, device=device, dtype=dtype))
            err = tr.sqrt(tr.mean(((e_v/scale)**2)))

            accept = (err<=1.0) or (abs(h)<=1.01*abs(h_min))
            if accept:
                s = _unit_retract(_rotate_by_omega(s, v5, 1.0))
                t = t + h
                steps += 1
                if record:
                    ts.append(t); states.append(s.clone())

            factor = 2.0 if err==0.0 else safety * float(err ** (-1.0/5.0))
            h = min(max(h_min, abs(h) * max(0.2, min(5.0, factor))), h_max)
            h *= direction

    info={"n_steps":steps,"t_final":t,"accepted":(abs(t-t1)<1e-12)}
    if record: return s, (tr.tensor(ts,dtype=dtype,device=device), states), info
    return s, info

def integrate_rkmk_dp5_with_logdet(flow, s0, *, t0=0.0, t1=1.0, rtol=1e-5, atol=1e-7,
                                   h_init=None, h_min=None, h_max=None, safety=0.9,
                                   max_steps=100000):
    c = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
    a = [[0,0,0,0,0,0,0],
         [1/5,0,0,0,0,0,0],
         [3/40,9/40,0,0,0,0,0],
         [44/45,-56/15,32/9,0,0,0,0],
         [19372/6561,-25360/2187,64448/6561,-212/729,0,0,0],
         [9017/3168,-355/33,46732/5247,49/176,-5103/18656,0,0],
         [35/384,0,500/1113,125/192,-2187/6784,11/84,0]]
    b5 = [35/384,0,500/1113,125/192,-2187/6784,11/84,0]
    b4 = [5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40]

    s=_unit_retract(s0); device, dtype=s.device, s.dtype
    t=t0; direction=1.0 if t1>=t0 else -1.0; T=abs(t1-t0)
    logdet = tr.zeros(s.size(0), device=device, dtype=dtype)

    if h_init is None: h = T/50.0 if T>0 else 1.0
    else: h = abs(h_init)
    if h_min is None: h_min = T/10_000.0 if T>0 else 1e-6
    if h_max is None: h_max = T/5.0 if T>0 else 1.0

    steps=0
    with tr.no_grad():
        while (t-t0)*direction < T and steps < max_steps:
            h = min(h, (t1-t)*direction)
            ks=[]; taus=[]
            for i in range(7):
                v_i = sum(a[i][j]*ks[j] for j in range(i)) if i>0 else tr.zeros_like(s)
                s_i = _rotate_by_omega(s, v_i, 1.0)
                t_i = t + c[i]*h
                omega_i = _call_maybe_t(flow, "grad", s_i, t_i)
                k_i = _dexpinv_so3(v_i, h*omega_i)
                ks.append(k_i)
                tau_i = (-1.0) * _call_maybe_t(flow, "mlapl", s_i, t_i)  # ascent: div = -mlapl
                taus.append(tau_i)
            v5 = sum(b5[i]*ks[i] for i in range(7))
            v4 = sum(b4[i]*ks[i] for i in range(7))
            e_v = v5 - v4
            scale = atol + rtol * tr.maximum(tr.norm(v5, dim=1, keepdim=True), tr.tensor(1.0, device=device, dtype=dtype))
            err = tr.sqrt(tr.mean(((e_v/scale)**2)))
            accept = (err<=1.0) or (abs(h)<=1.01*abs(h_min))
            if accept:
                s = _unit_retract(_rotate_by_omega(s, v5, 1.0))
                tau_step = sum(b5[i]*taus[i] for i in range(7))
                logdet = logdet + h * tau_step
                t = t + h
                steps += 1
            factor = 2.0 if err==0.0 else 0.9 * float(err ** (-1.0/5.0))
            h = min(max(h_min, abs(h) * max(0.2, min(5.0, factor))), h_max); h *= direction

    info={"n_steps":steps,"t_final":t,"accepted":(abs(t-t1)<1e-12)}
    return s, logdet, info
