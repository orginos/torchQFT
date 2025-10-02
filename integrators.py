#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  19 9:38:24 2022

Copied from pyQFT

@author: Kostas Orginos
"""

import numpy as np
def simple_evolveP(dt,F,P):
    try:
        return P+dt*F
    except RuntimeError:
        print("RuntimeError in simple_evolveP")
        print("dt:",dt,"F:",F.shape,"P:",P.shape)
        return P+dt*F

class integrator():
    def __init__(self,force,evolveQ,Nmd,t,evolveP):
        self.force   = force
        self.Nmd     = Nmd
        self.dt      = t/Nmd
        self.evolveQ = evolveQ
        self.evolveP = evolveP
    
class leapfrog(integrator):
    def __init__(self,force,evolveQ,Nmd,t,evolveP=simple_evolveP):
        super().__init__(force,evolveQ,Nmd,t,evolveP)
             
    def integrate(self,p,q):
        #p=p + 0.5*self.force(q)*self.dt
        p = self.evolveP(0.5*self.dt,self.force(q),p) 
        for t in range(1,self.Nmd):
            q = self.evolveQ(self.dt,p,q) #q + p*self.dt
            #p = p + self.force(q)*self.dt
            p=self.evolveP(self.dt,self.force(q),p)
        q = self.evolveQ(self.dt,p,q) #q + p*self.dt
        #p = p + 0.5*self.force(q)*self.dt
        p=self.evolveP(0.5*self.dt,self.force(q),p)
                       
        return p,q

class minnorm2(integrator):
    def __init__(self,force,evolveQ,Nmd,t,evolveP=simple_evolveP,lam=0.1931833275037836):
        super().__init__(force,evolveQ,Nmd,t,evolveP)
        self.lam=lam

    def integrate(self,p,q):
        for t in range(0,self.Nmd):
            #p = p + self.force(q)*self.dt*self.lam
            if q.isnan().any():
                print("NaN detected first, q in minnorm2", t)
            if p.isnan().any():
                print("NaN detected first, p in minnorm2", t)
            if self.force(q).isnan().any():
                print("NaN detected first, force(q) in minnorm2", t)
                print(self.force(q).shape)
            p=self.evolveP(self.dt*self.lam,self.force(q),p)
            q = self.evolveQ(0.5*self.dt,p,q) #q + 0.5*self.dt*p
            self.qtest = q
            self.ptest = p
            if q.isnan().any():
                print("NaN detected second, q in minnorm2 ", t)
            if p.isnan().any():
                print("NaN detected second, p in minnorm2 ", t)
            if self.force(q).isnan().any():
                print("NaN detected second, force(q) in minnorm2", t )
                print("location of NAN:",self.force(q).isnan().nonzero(),self.force(q)[self.force(q).isnan()])
            #p = p + self.force(q)*self.dt*(1.0-2.0*self.lam)
            p=self.evolveP(self.dt*(1.0-2.0*self.lam),self.force(q),p)
            q = self.evolveQ(0.5*self.dt,p,q) #q + 0.5*self.dt*p
            self.qtest1 = q
            self.ptest1 = p
            if q.isnan().any():
                print("NaN detected third, q in minnorm2 ", t)
            if p.isnan().any():
                print("NaN detected third, p in minnorm2 ", t)
            if self.force(q).isnan().any():
                print("NaN detected third, force(q) in minnorm2", t )
                print("location of NAN:",self.force(q).isnan().nonzero(),self.force(q)[self.force(q).isnan()])
            #p = p + self.force(q)*self.dt*self.lam
            p=self.evolveP(self.dt*self.lam,self.force(q),p)
            self.ptest2 = p

        return p,q
    
class minnorm4pf4(integrator):
    def __init__(self,force,evolveQ,Nmd,t,evolveP=simple_evolveP,
                 rho=0.1786178958448091,
                 theta=-0.06626458266981843,
                 lam=0.7123418310626056):
        super().__init__(force,evolveQ,Nmd,t,evolveP)
        self.lam=lam
        self.rho=rho
        self.the=theta
        
    def integrate(self,p,q):
        for t in range(0,self.Nmd):
            p=self.evolveP(self.dt*self.rho,self.force(q),p)
            q=self.evolveQ(self.dt*self.lam,p,q) 
            p=self.evolveP(self.dt*self.the,self.force(q),p)
            q=self.evolveQ(self.dt*(1.0-2.0*self.lam)/2.0,p,q) 
            p=self.evolveP(self.dt*(1.0-2.0*(self.the+self.rho)),self.force(q),p)
            q=self.evolveQ(self.dt*(1.0-2.0*self.lam)/2.0,p,q)
            p=self.evolveP(self.dt*self.the,self.force(q),p)
            q=self.evolveQ(self.dt*self.lam,p,q)
            p=self.evolveP(self.dt*self.rho,self.force(q),p)
                                   
        return p,q
    
# implements the magnetic leapfrog    
class m_leapfrog(integrator):
    def __init__(self,force,evolveQ,Nmd,t,applyInvG,applyExpG,evolveP=simple_evolveP):
        super().__init__(force,evolveQ,Nmd,t,evolveP)
        # G is a sparse matrix implementing the non-Hamiltonian dynamics
        self.applyInvG=applyInvG
        self.applyExpG=applyExpG
        
    def integrate(self,p,q):
        #p=p + 0.5*self.force(q)*self.dt
        p = self.evolveP(0.5*self.dt,self.force(q),p) 
        for t in range(1,self.Nmd):
            q = self.evolveQ(self.dt,p,q) #q + p*self.dt
            #p = p + self.force(q)*self.dt
            p=self.evolveP(self.dt,self.force(q),p)
        q = self.evolveQ(self.dt,p,q) #q + p*self.dt
        #p = p + 0.5*self.force(q)*self.dt
        p=self.evolveP(0.5*self.dt,self.force(q),p)
                       
        return p,q


