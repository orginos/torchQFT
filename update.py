#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday May 20 11:00:42 EDT 2022

Copied from pyQFT

@author: Kostas Orginos
"""

import torch as tr
import integrators as i
from copy import deepcopy
import numpy as np

class hmc:
    def __init__(self,T,I,verbose=True):
        self.T = T # the "theory"
        self.I = I # the integrator
        self.verbose=verbose
        self.AcceptReject = [] # accept/reject history

    def calc_Acceptance(self):
        if(len(self.AcceptReject)>0):
            return np.mean(self.AcceptReject)
        else:
            return 0
    
    #Original evolve function
    def evolve(self,q,N):
        qshape =tuple([q.shape[0]]+[1]*(len(q.shape)-1))
        for k in range(N):
            q0=q.clone() # copy the q
            p0 = self.T.refreshP()
            H0 = self.T.kinetic(p0) + self.T.action(q0)
            p,q = self.I.integrate(p0,q0)
            Hf = self.T.kinetic(p) + self.T.action(q)
            DH = Hf - H0
            acc_prob=tr.where(DH<0,tr.ones_like(DH),tr.exp(-DH))
            R=tr.rand_like(acc_prob)
            Acc_flag = (R<acc_prob)
            AR = tr.where(Acc_flag,tr.ones_like(DH),tr.zeros_like(DH))
            self.AcceptReject.extend(AR.tolist())
            q = tr.where(Acc_flag.view(qshape),q,q0)
            
            if(self.verbose):
                av_ar = tr.mean(AR)
                print(" HMC: ",k," DH= ",DH.tolist()," A/R= ",Acc_flag.tolist()," Pacc= ",av_ar.item())
                    
        return q

    #Evolve function for handling gauge fields and fermion fields as a tuple
    def evolve_f(self,q,N, f=False):
        qshape =tuple([q[0].shape[0]]+[1]*(len(q[0].shape)-1))
        #Acceptance flag shapes needed with fermion field/dirac operator
        if f == True:
            fshape = tuple([q[1].shape[0]]+[1]*(len(q[1].shape)-1))
            dshape = tuple([q[2].shape[0]]+[1]*(len(q[2].shape)-1))

        for k in range(N):
            #Generate psuedofermions if neccesary
            if f == True:
                f_upd = self.T.generate_Pseudofermions(q[2])
                q = (q[0], f_upd, q[2])
            q0=tuple(q) # copy the q
            p0 = self.T.refreshP()
            H0 = self.T.kinetic(p0) + self.T.action(q0)
            p,q = self.I.integrate(p0,q)
            Hf = self.T.kinetic(p) + self.T.action(q)
            DH = Hf - H0
            acc_prob=tr.where(DH<0,tr.ones_like(DH),tr.exp(-DH))
            R=tr.rand_like(acc_prob)
            Acc_flag = (R<acc_prob)
            AR = tr.where(Acc_flag,tr.ones_like(DH),tr.zeros_like(DH))
            self.AcceptReject.extend(AR.tolist())
            u = tr.where(Acc_flag.view(qshape),q[0],q0[0])
            #This is hacky... must be better way
            if f == False:
                q = (u,)
            else:
                ff = tr.where(Acc_flag.view(fshape),q[1],q0[1])
                #tr.where doesn't accept dense matrices
                d_dense = q[2].to_dense()
                d0_dense = q0[2].to_dense()
                d = tr.where(Acc_flag.view(dshape),d_dense,d0_dense)
                q = (u, ff, d.to_sparse())
            # q = tr.where(Acc_flag.view(qshape),q,q0)

            
            if(self.verbose):
                av_ar = tr.mean(AR)
                print(" HMC: ",k," DH= ",DH.tolist()," A/R= ",Acc_flag.tolist()," Pacc= ",av_ar.item())
                    
        return q
    
    #Naive DD evolve - added for numerical methods project
    def dd_Evolve(self,q, N, xcut):
        qshape =tuple([q.shape[0]]+[1]*(len(q.shape)-1))
        for k in range(N):
            q0=q.clone() # copy the q
            p0 = self.T.refreshP()
            H0 = self.T.kinetic(p0) + self.T.action(q0)
            if(k ==0):
                p,q = self.I.integrate(p0,q0)
            else:
                p,q = self.I.dd_Integrate(p0,q0, xcut)
            Hf = self.T.kinetic(p) + self.T.action(q)
            DH = Hf - H0
            acc_prob=tr.where(DH<0,tr.ones_like(DH),tr.exp(-DH))
            R=tr.rand_like(acc_prob)
            Acc_flag = (R<acc_prob)
            AR = tr.where(Acc_flag,tr.ones_like(DH),tr.zeros_like(DH))
            self.AcceptReject.extend(AR.tolist())
            q = tr.where(Acc_flag.view(qshape),q,q0)
            
            if(self.verbose):
                av_ar = tr.mean(AR)
                print(" HMC: ",k," DH= ",DH.tolist()," A/R= ",Acc_flag.tolist()," Pacc= ",av_ar.item())
                    
        return q

    
# NOT YET CONVERTED TO TORCH
class smd:
    def __init__(self,T,I,gamma):
        self.T = T # the "theory"
        self.I = I # the integrator
        self.c1 = np.exp(-gamma*I.Nmd*I.dt) # the gamma factor
        self.c2 = np.sqrt(1.0- self.c1**2)
        # according to Luscher's paper (https://arxiv.org/pdf/1105.4749.pdf)
        # the integrator needs to be called with 1 step
        
    def evolve(self,q,N):
        q0 = np.zeros(q.shape,dtype=q.dtype)
        p = self.T.refreshP()
        for k in range(N):
            q0=deepcopy(q) # copy the q
            p0 = self.c1*p + self.c2*self.T.refreshP()
            H0 = self.T.kinetic(p0) + self.T.action(q0)
            p,q = self.I.integrate(p0,q0)
            Hf = self.T.kinetic(p) + self.T.action(q)
            DH = Hf - H0
            if DH<0 :
                Pacc = 1
                A="A"
            else :
                R = np.exp(-DH)
                x=np.random.uniform(0,1,1)
                if x < R :
                    Pacc=R
                    A = "A"
                else:
                    A = "R"
                    Pacc = 1 - R
                    q[:] =  q0 # reject
                    p[:] = -p0 # flip momentum
            print( " SMD: ",k," DH= ",DH," Pacc=",Pacc,"  ",A)
                    
        return q
