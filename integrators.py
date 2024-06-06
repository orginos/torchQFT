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
    
    #For CS Methods project
    def dd_Integrate(self, p, q, xcut):
        #For pure gauge theory, evolution is local-
        #no need to run seperate processes for subdomains
        #Complete the first pass normally:
        # p=self.evolveP(self.dt*self.lam,self.force(q),p)
        # q = self.evolveQ(0.5*self.dt,p,q) #q + 0.5*self.dt*p
        # #p = p + self.force(q)*self.dt*(1.0-2.0*self.lam)
        # p=self.evolveP(self.dt*(1.0-2.0*self.lam),self.force(q),p)
        # q = self.evolveQ(0.5*self.dt,p,q) #q + 0.5*self.dt*p
        # #p = p + self.force(q)*self.dt*self.lam
        # p=self.evolveP(self.dt*self.lam,self.force(q),p)
        q0=q.clone()
        p0 = p.clone()
        #Freeze the boundary for the rest of the updates
        for t in range(0,self.Nmd):
            #p = p + self.force(q)*self.dt*self.lam
            p=self.evolveP(self.dt*self.lam,self.force(q),p)
            #Freeze boundary
            p[:,xcut, :, :] = p0[:,xcut, :,:]
            p[:,0, :, :] = p0[:,0, :,:]
            q = self.evolveQ(0.5*self.dt,p,q) #q + 0.5*self.dt*p
            #Freeze boundary
            q[:,xcut, :, :] = q0[:,xcut, :,:]
            q[:,0, :, :] = q0[:,0, :,:]
            #p = p + self.force(q)*self.dt*(1.0-2.0*self.lam)
            p=self.evolveP(self.dt*(1.0-2.0*self.lam),self.force(q),p)
            #Freeze boundary
            p[:,xcut, :, :] = p0[:,xcut, :,:]
            p[:,0, :, :] = p0[:,0, :,:]
            q = self.evolveQ(0.5*self.dt,p,q) #q + 0.5*self.dt*p
            #Freeze boundary
            q[:,xcut, :, :] = q0[:,xcut, :,:]
            q[:,0, :, :] = q0[:,0, :,:]
            #p = p + self.force(q)*self.dt*self.lam
            p=self.evolveP(self.dt*self.lam,self.force(q),p)
            #Freeze boundary
            p[:,xcut, :, :] = p0[:,xcut, :,:]
            p[:,0, :, :] = p0[:,0, :,:]


        return p, q
        
    
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


