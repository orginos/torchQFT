#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  19 9:38:24 2022

Copied from pyQFT

@author: Kostas Orginos
"""

import numpy as np
def simple_evolveP(dt,F,P):
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
            p=self.evolveP(self.dt*self.lam,self.force(q),p)
            q = self.evolveQ(0.5*self.dt,p,q) #q + 0.5*self.dt*p
            #p = p + self.force(q)*self.dt*(1.0-2.0*self.lam)
            p=self.evolveP(self.dt*(1.0-2.0*self.lam),self.force(q),p)
            q = self.evolveQ(0.5*self.dt,p,q) #q + 0.5*self.dt*p
            #p = p + self.force(q)*self.dt*self.lam
            p=self.evolveP(self.dt*self.lam,self.force(q),p)
                       
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


# r-RESPA (nested) integrator using leapfrog for the fast force
class rrespa_leapfrog(integrator):
    def __init__(self, fast_force, slow_force, evolveQ, Nouter, t, Ninner=4, evolveP=simple_evolveP):
        super().__init__(fast_force, evolveQ, Nouter, t, evolveP)
        self.fast_force = fast_force
        self.slow_force = slow_force
        self.Ninner = int(Ninner)

    def integrate(self, p, q):
        dt = self.dt
        dt_inner = dt / self.Ninner
        for _ in range(self.Nmd):
            # slow half-step
            p = self.evolveP(0.5 * dt, self.slow_force(q), p)
            # inner fast steps (leapfrog)
            for _ in range(self.Ninner):
                p = self.evolveP(0.5 * dt_inner, self.fast_force(q), p)
                q = self.evolveQ(dt_inner, p, q)
                p = self.evolveP(0.5 * dt_inner, self.fast_force(q), p)
            # slow half-step
            p = self.evolveP(0.5 * dt, self.slow_force(q), p)
        return p, q


# r-RESPA (nested) integrator using minnorm2 for the fast force
class rrespa_minnorm2(integrator):
    def __init__(self, fast_force, slow_force, evolveQ, Nouter, t, Ninner=4, evolveP=simple_evolveP, lam=0.1931833275037836):
        super().__init__(fast_force, evolveQ, Nouter, t, evolveP)
        self.fast_force = fast_force
        self.slow_force = slow_force
        self.Ninner = int(Ninner)
        self.lam = lam

    def integrate(self, p, q):
        dt = self.dt
        dt_inner = dt / self.Ninner
        for _ in range(self.Nmd):
            # slow half-step
            p = self.evolveP(0.5 * dt, self.slow_force(q), p)
            # inner fast steps (minnorm2)
            for _ in range(self.Ninner):
                p = self.evolveP(dt_inner * self.lam, self.fast_force(q), p)
                q = self.evolveQ(0.5 * dt_inner, p, q)
                p = self.evolveP(dt_inner * (1.0 - 2.0 * self.lam), self.fast_force(q), p)
                q = self.evolveQ(0.5 * dt_inner, p, q)
                p = self.evolveP(dt_inner * self.lam, self.fast_force(q), p)
            # slow half-step
            p = self.evolveP(0.5 * dt, self.slow_force(q), p)
        return p, q
