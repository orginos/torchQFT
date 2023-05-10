#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:48:03 EDT 2022

Copied from pyQFT

It should work as is

@author: Kostas Orginos
"""

import torch as tr

# Finds value of y for a given x using step size h 
# and initial value y0 at x0. 
def rungeKutta(x0, y0, x, n,dydx): 
    # Count number of iterations using step size or 
    # step height h 
    h = (x - x0)/n  
    # Iterate for number of iterations 
    y = y0 
    for i in range(1, n + 1):
        #print(i)
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * dydx(x0, y) 
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1) 
        k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2) 
        k4 = h * dydx(x0 + h, y + k3) 
  
        # Update next value of y 
        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 
  
        # Update next value of x 
        x0 = x0 + h 
    return y

# compact fields
# it solves the equation
# Y' = Z(Y) Y
#
# Finds value of y for a given x using step size h 
# and initial value y0 at x0.
# the following functions are needed.
# Z is the generator of the flow in the ODE (see above)
# note that this algorith assumes no time dependence for the flow generator
# evoY evolves the Y fields (which may have a complicated evolution)
#
def rungeKuttaCompact(x0, y0, x, Nint,Z,evoY): 
    # Count number of iterations using step size or 
    # step height h 
    h = (x - x0)/Nint 
    # Iterate for number of iterations 
    Y = y0 
    for i in range(0, Nint):
        #print(i)
        "Apply Runge Kutta Formulas to find next value of y"
        #W0 = y
        Z0 = Z(Y)
        W1=evoY(h,0.25*Z0,Y)
        Z1 = Z(W1)
        W2=evoY(h,8.0/9.0*Z1- 17.0/36.0*Z0,W1)
        Z2 = Z(W2)
        Y = evoY(h,3.0/4.0*Z2 - 8.0/9.0*Z1 + 17.0/36.0*Z0,W2)
        # Update next value of x 
        x0 = x0 + h
        # not needed because Z does not depend on x
    
    return Y


# NOT WORKING IN TORCH
# General Explicit Runge-Kutta Munthe-Kaas algorithm  implemented
# with a class that takes in the a, b, c constants defining the method
# Z is a function Z(x,y) where x is the "time" and y is the dependent variable
class RungeKutta():
    def __init__(self,a,b,q,GxG,GxV,Z,NexpoTaylor=10):
        # a is an s x s matrix and
        # b the final linear compbination of k's
        self.Z = Z       # the generator of the flow
        self.GxG = GxG # the group element multiplication
        self.GxV = GxV # group element (or Lie algebra element)
        # multiplication on manifold point
        
        # Bernulli/k! upt to k =8  (even only)
        # 2, 4, 6, 8
        self.Bern_o_k_fact = [ 1.0/12.0,-1.0/720,1.0/30240.0,-1.0/12909600 ]
        
        self.NexpoTaylor = NexpoTaylor

        #print(b.shape)
        self.s = b.shape[0] # the number of stages of the RK method
        self.q = q # the order of the RK method
        self.a =a
        self.b =b
        self.c = tr.empty_like(b)
        # this is the definition of c for RK methods
        #for i in range(len(b)):
        #     self.c[i] = tr.sum(a[i,:])
        self.c = tr.sum(a,dim=1)
        #print(a,self.c)
        
        
    # Horner scheme for Lie algebra element exponentiation
    # and applied on y
    # ie. exp(u)*y
    def expo(self,u,y):    
        M=y
        for k in range(self.NexpoTaylor,0,-1):
            M = (1.0/k)*self.GxV(u,M) + y
        return M
    
    def adj(self,u,v):
        return self.GxG(u,v) - self.GxG(v,u)
     
    def dexpinv(self,u,v):
        p = self.adj(u,v)
        r = v - 0.5*p
        for j in range(2,self.q-1,2):
            p = self.adj(u,p)
            r = r + self.Bern_o_k_fact[int(j/2)]*p
            p = self.adj(u,p)
        return r
    
    def integrate(self,x0, y0, x, Nint):
        # Count number of iterations using step size or 
        # step height h 
        h = (x - x0)/Nint 
        # Iterate for number of iterations 
        y = y0
        #k=[None]*self.q
        #print("k = ",k)
        zero = 0.0*self.Z(0.0,y)# just get the zero lie algebra element ...
        xx=x0
        #print(zero)
        #k = tr.zeros([self.s])
        for l in range(0, Nint):
            v = zero
            for i in range(self.s):
                u=zero
                print(i,self.s)
                for j in range(i):
                    print(k.shape,h,u.shape)
                    u = u + h *self.a[i,j]*k[j]
                kk = self.Z(xx+h*self.c[i],self.expo(u,y))
                k = self.dexpinv(u,kk)
                print(kk.shape)
                #print(i,self.dexpinv(u,kk))
                v = v + h*self.b[i]*k
            y=self.expo(v,y)
            xx += h # advance the time    
        return y
    
#classic 4-th order RungeKutta
class lieGroupRK4(RungeKutta):
    def __init__(self,GxG,GxV,Z,NexpoTaylor=10):
        q=4
        a = tr.tensor([[0. , 0. , 0., 0.],
                      [0.5, 0. , 0., 0.],
                      [0. , 0.5, 0., 0.],
                      [0. , 0. , 1., 0.]])
        b = tr.tensor([1.0/6,1.0/3,1.0/3,1.0/6])   
        RungeKutta.__init__(self,a,b,q,GxG,GxV,Z,NexpoTaylor=10)


def test_rk():
    import phi4 as s
    import numpy as np
    
    L=32
    batch_size=2
    lam =1.9
    mass= -0.5
    o = s.phi4([L,L],lam,mass,batch_size=batch_size)

    phi0 = o.hotStart()

    print("Initial action: ", o.action(phi0).numpy())
    dydx = lambda x,y: o.force(y)
    ndydx = lambda x,y: -o.force(y)
    for n in np.arange(10,101):
        phi1 = rungeKutta(0, phi0, 1.0, n ,dydx)
        fa = o.action(phi1).numpy()
        iphi0 = rungeKutta(0, phi1, 1.0, n ,ndydx)
        ia = o.action(iphi0).numpy()
        print("Nsteps = ",n," Final action: ", fa, " Reversed action: ",ia)

 

if __name__ == "__main__":
   test_rk()
    


