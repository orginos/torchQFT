#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 2022

@author: Ben Slimmer
"""

import numpy as np;
import torch as tr;
import update as h;
import integrators as i;
import time;
import matplotlib.pyplot as plt;

gamma = tr.tensor([[[0.0, -1.0j], [-1.0j, 0.0]], [[0, -1], [1,0]], [[1,0], [0,-1]]])
g5 = tr.tensor([[1.0+0.0j, 0], [0, -1.0+0.0j]])

class schwinger():


    def __init__(self,V,l,m,batch_size=1,device="cpu",dtype=tr.float32): 
        self.V = tuple(V) # lattice size
        self.Vol = np.prod(V)
        self.Nd = len(V)
        self.lam = l # the coupling
        self.mass  = m
        self.mtil = m + self.Nd
        self.Bs=batch_size
        self.device=device
        self.dtype=dtype


    #Section for Dirac operator and some needed functionality for manipulating it: ************************************************
    # Halted development on this section to focus on pure gauge

    #Inputs: self, gauge field u
    #Outputs: Dirac operator
    def diracOperator(self, u):

        d = tr.zeros([self.Bs, self.V[0], self.V[1], self.V[0], self.V[1], 2,2], dtype=tr.complex64)
        #for loop to fill entries- probably a way to optimize with pytorch.
        for i in range(0, self.V[0]):
            for j in range(0, self.V[1]):
                d[:,i, j, i, j, :, :] = self.mtil*tr.eye(2)
                #Manage special cases of periodic boundary conditions w/ conditionals
                # -1/2 terms are to eliminate doublers, assuming spacing a =1
                if(i == 0):
                   d[:,i,j, self.V[0]-1, j, :, :] = -1.0*tr.conj(u[:,self.V[0]-1, j, 0]) * gamma[0] \
                    -0.5* tr.eye(2)
                else:
                    d[:,i,j, i-1, j, :, :] = tr.conj(u[:,i-1, j, 0]) * gamma[0] \
                    -0.5* tr.eye(2)
                if(i == self.V[0]-1):
                   d[:,i,j, 0, j, :, :] = -1.0*tr.conj(u[:,i, j, 0]) * gamma[0] \
                   -0.5* tr.eye(2)
                else:
                     d[:,i,j, i+1, j, :, :] = u[:,i, j, 0] * gamma[0] \
                     -0.5* tr.eye(2)
                if(j == 0):
                   d[:,i,j, i, self.V[1]-1, :, :] = tr.conj(u[:,i, self.V[1]-1, 1]) * gamma[1] \
                   -0.5* tr.eye(2)
                else:
                    d[:,i,j, i, j-1, :, :] = tr.conj(u[:,i, j-1, 0]) * gamma[1] \
                    -0.5* tr.eye(2)
                if(j == self.V[1]-1):
                   d[:,i,j, i, 0, :, :] = u[:,i, j, 1] * gamma[1] \
                   -0.5* tr.eye(2)
                else:
                     d[:,i,j, i, j+1, :, :] = u[:,i, j, 1] * gamma[1]  \
                     -0.5* tr.eye(2)
                    

        #Outdated/incorrect Dirac implementation for review below

        #Sub-sum for adding terms of different directions
        # sub =tr.zeros([self.Bs, self.V[0], self.V[1], 2, 2], dtype=tr.complex64)
        # #Identity matrix unsqueezed for batch size
        # b_eye = tr.unsqueeze(tr.eye(n=self.V[0], m=self.V[1]), 0)

        # for mu in range(0, self.Nd):
        #     u_dir = u[:,:,:,mu]
        #     naive = (1.0/2)*(tr.roll(u_dir, shifts=1, dims=mu+1) \
        #     - tr.conj(u_dir))
        #     wilson = (-1.0/2)*(tr.roll(u_dir, shifts=1, dims=mu+1) \
        #     + tr.conj(u_dir))

        #     sub += tr.einsum('ij,bxy->bxyij', (gamma[mu]), naive) + tr.einsum('ij,bxy->bxyij', (tr.eye(2)), wilson)
        # return sub + tr.einsum('ij,bxy->bxyij', (self.mtil*tr.eye(2)), tr.ones(self.Bs, self.V[0], self.V[1]))
        return d

    #Inputs: self, gauge field u
    #Output: Operator derivative
    def diracDerivative(self, u):
        partiald = tr.zeros([self.Bs, self.V[0], self.V[1], 2, 2, 2], dtype=tr.complex64)
        for mu in range(0, self.Nd):
            u_dir = u[:,:,:,mu]
            partiald[:,:,:,0,:,:] = (1.0j/2)*(-1.0*tr.einsum('bxy, ij -> bxyij', tr.roll(u_dir,1,dims=(mu+1)), (tr.eye(2) - gamma[mu])) 
                            + tr.einsum('bxy, ij -> bxyij', tr.conj(u_dir), (tr.eye(2) + gamma[mu])))
        return partiald
    
    
    #Inputs: self and m: a general BxLxLxmux2X2 tensor in position, direction, dirac space respectively
    #Needed specifically for the hermitian conjugate of the derivative of Dirac
    #operator for force calculation
    #Outputs: Hermitian conjugate for dirac and position indices
    def p_dagger(self, m):
        return tr.conj(tr.transpose(tr.transpose(m, 4, 5), 1, 2))  
    
    #Returns Hermitian conjugate of Dirac Operator
    #Input: BxLxLxLxLx2x2 tensor
    def d_dagger(self, d):
        return tr.conj(tr.transpose(tr.transpose(tr.transpose(d, 1, 2), 3, 4), 5, 6))
    
    
    #Inverse of momentum space dirac operator:
    #NOTE: Assumes diagonal momentum
    def dp_inverse(self, p, u):
        sum = tr.zeros(self.Bs, 2,2)
        matsum = tr.zeros(self.Bs, u.size(dim=1), u.size(dim=2), 2, 2)
        for mu in range(0, self.Nd):
            matsum = 0.5*(np.exp(1.0j*p[mu])*tr.einsum('ij, bxy-> bxyij',(gamma[mu] - tr.eye(2)),u[:,:,:,mu]) - \
                 np.exp(-1.0j*p[mu])*tr.einsum('ij, bxy-> bxyij', (gamma[mu] + tr.eye(2)),tr.conj(tr.roll(u[:,:,:,mu],1, mu+1))))
        for i in range(0, u.size(dim=1)):
            for j in range(0, u.size(dim=2)):
                sum = sum + matsum[:,i, j,:,:]
        sum = (1/(u.size(dim=1)*u.size(dim=2)))* sum + tr.eye(2)*self.mtil
        return tr.inverse(sum)
        
    
    def d_product(self, d , dd):
        return tr.einsum('bxymnij, byznojk ->bxzmoik', d, dd)
    

    #A naive implementation of the dirac inverse at specified coordinates
    def dirac_inverse(self, u, m0, m1, n0, n1):

        sum = tr.zeros(self.Bs, 2,2, dtype=tr.complex64)
        for p0 in np.arange(-self.V[0]/2.0 + 1.0, self.V[0]/2.0, 1.0):
            for p1 in np.arange(-self.V[1]/2.0 + 1.0, self.V[1]/2.0, 1.0):
                sum += self.dp_inverse([p0, p1], u)*np.exp(1.0j*((n0-m0)*p0+ (n1-m1)*p1))
        return sum/(u.size(dim=1)*u.size(dim=2))

        

    #A naive implementation of a domain-decomposed propogator
    def DD_dirac_inverse(self, d, u, xcut, b, m0, m1, n0, n1):
        #Define subdomains
        u1 = u[:, 0:(xcut+1), :, :]
        u2 = u[:, xcut:, :,:]
        sum = tr.zeros(self.Bs, 2, 2, dtype=tr.complex64)
        #Steps of factorized propogator
        for i in b:
            s1 = self.dirac_inverse(u1, m0, m1, xcut, i)
            s2 = tr.einsum("bxy, byz -> bxz", d[:, xcut, i, xcut+1, i], s1)
            s3 = tr.einsum('bxy, byz -> bxz', self.dirac_inverse(u2, 0, i, n0-xcut, n1), s2)
            sum = sum + s3
        return sum

#********************************************************************************************************************************


#Gauge theory ****************************************************************************************************************
    #Inputs: self, BxLxLxmu gauge field u 
    #Outputs: Pure gauge action
    #TODO: Generalize to higher dimension lattices
    def gaugeAction(self, u):
        #plaquette matrix
        pl = u[:,:,:,0]*tr.roll(u[:,:,:,1], shifts=-1, dims=1) \
            * tr.conj(tr.roll(u[:,:,:,0], shifts=-1, dims=2))*tr.conj(u[:,:,:,1]) 
        
        S = (1/self.lam**2) *(tr.sum(tr.real(tr.ones_like(pl) - pl),dim=(1,2)))
        return S



    #TODO: Write fermion action to be able to include it in total action
    #Inputs: self, the gauge field u
    #Outputs: Action of the given configuration
    def action(self, u):
        return self.gaugeAction(u).type(self.dtype)
    

    #TODO: Write fermion force contribution
    #Inputs: self, gauge field u
    #Output: force tensor for updating momentum tensor in HMC
    def force(self, u):
        #A matrix of 'staples'
        a = tr.zeros_like(u)


        #TODO:Generalize to higher dimension lattices
        a[:,:,:,0]  = tr.roll(u[:,:,:,1], shifts=-1, dims=1) * tr.conj(tr.roll(u[:,:,:,0], shifts=-1, dims=2))*tr.conj(u[:,:,:,1]) \
                      + tr.conj(tr.roll(u[:,:,:,1], shifts=(-1,1), dims= (1,2)))*tr.conj(tr.roll(u[:,:,:,0], shifts=1, dims=2)) * tr.roll(u[:,:,:,1], shifts=1, dims=2)
        a[:,:,:,1] = tr.conj(tr.roll(u[:,:,:,0], shifts=(1,-1), dims=(1,2))) * tr.conj(tr.roll(u[:,:,:,1], shifts=1, dims=1)) * tr.roll(u[:,:,:,0], shifts=1, dims=1) \
                     + tr.roll(u[:,:,:,0], shifts=-1, dims=2) * tr.conj(tr.roll(u[:,:,:,1], shifts=-1, dims=1)) * tr.conj(u[:,:,:,0])
        #gauge action contribution
        #Additional negative sign added from Hamilton's eqs.
        fg = (-1.0)*(-1.0j* (1.0/self.lam**2)/2.0)* (u*a - tr.conj(a)*tr.conj(u))

        #TODO: fermion action contribution
        #Free theory case for now
        ff = tr.zeros_like(u)
            


        #force is already be real... force the cast for downstream errors
        return (tr.real(fg+ ff)).type(self.dtype)
        #return fg + ff


    def kinetic(self,P):
        return tr.einsum('bxyz,bxyz->b',P,P)/2.0 
    
    #HMC refresh of momentum
    def refreshP(self):
        P = tr.normal(0.0,1.0,[self.Bs,self.V[0],self.V[1], 2],dtype=self.dtype,device=self.device)
        return P
    
    #HMC position evolve 
    def evolveQ(self,dt,P,Q):
        return Q*tr.exp(1.0j*dt*P)
    
    #Input: self
    #Output: Hot started U[1] gauge field of batch*lattice size.
        #the final dimension of 2 is for link variable in t, then x direction
    def hotStart(self, a):
        alpha = tr.normal(0.0, 2*np.pi, [self.Bs, self.V[0], self.V[1], 2],
                      dtype=self.dtype,device=self.device)
        u = tr.exp(1.0j*a*alpha)
        return u
    
    #Input: self
    #Output: Cold started U[1] gauge field of batch*lattice size.
        #the final dimension of 2 is for link variable in t, then x direction
    def coldStart(self):
        return tr.ones(self.Bs, self.V[0], self.V[1], 2)
    
    #Input: self
    #Output: normalized Dirac spinor lattice
    #TODO - Implement dynamical fermion generation
    def spinorLattice(self, d):
       
        return 0
        
    


def main():
    plt.rcParams['text.usetex'] = True
    
    device = "cuda" if tr.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    L=16
    batch_size=1
    #Coupling
    lam =0.5
    mass= 0.1
    #Spacing
    a = 1
    sch = schwinger([L,L],lam,mass,batch_size=batch_size)

    u = sch.hotStart(a)
    u_cold = sch.coldStart()
    d=sch.diracOperator(u)
    d_cold = sch.diracOperator(u_cold)

    mn2 = i.minnorm2(sch.force,sch.evolveQ,50,1.0)

    sim = h.hmc(sch, mn2)

    #Test plot of randomized link values in x direction
    if(False):
        #arccos of real part of the link -> randomized parameter
        plt.imshow(tr.real(u[0,:,:,0]), cmap='hot', interpolation='nearest')
        plt.show()


    #Verify action functions run without error
    #Runs, but values are unverified
    if(False):
        gS = sch.gaugeAction(u)
        print("Gauge action: ", gS)


    #test force function
    if(False):
        print("Force: ",sch.force(u))




    #Test sim
    if (False):
        # plt.imshow(tr.arccos(tr.real(u[0,:,:,0])), cmap='hot', interpolation='nearest')
        # plt.show()
        u_upd = sim.evolve(u, 100)
        plt.imshow(tr.real(u_upd[0,:,:,0]), cmap='hot', interpolation='nearest')
        plt.show()

    #Test stepsize
    if(True):
        #Average over a batch of configurations
        #Parameters imitate that of Duane 1987 HMC Paper
        L=8
        batch_size=1000
        lam =1.015
        mass= 0.1
        sch = schwinger([L,L],lam,mass,batch_size=batch_size)

        u = sch.hotStart(a)

        e2 = []
        dH = []
        Herr= [] 

        for e in np.linspace(0.05, 1, 100):
            lf = i.leapfrog(sch.force,sch.evolveQ,1,e)
            sim = h.hmc(sch, lf, False)
            e2.append(e**2)
            sdH = np.abs(sim.step_DH(u))
            dH.append(tr.mean(sdH))
            Herr.append(tr.std(sdH))
            

        lineary = np.linspace(dH[0], dH[99], 100)
        linearx = np.linspace(e2[0], e2[99], 100)

        fig, ax1 = plt.subplots(1,1)

        ax1.errorbar(e2, dH, yerr=Herr)
        ax1.plot(linearx, lineary, '--r')
        ax1.set_ylabel(r'$|\Delta H|$')
        ax1.set_xlabel(r'$\epsilon^2$')
        plt.show()

    #Plaquette average - In progress
    if(False):
        #Average over a batch of configurations
        #Parameters imitate that of Duane 1987 HMC Paper
        L=8
        batch_size=1
        lam =1.015
        mass= 0.1
        sch = schwinger([L,L],lam,mass,batch_size=batch_size)

        u = sch.hotStart(a)

        pl_avg = []
        #Need to generalize to larger batch size
        #pl_err= []

        #TODO: Generalize to >0 batch size
        for e in np.linspace(0.05, 0.16, 100):
            #Tune leap frog to desired step size
            lf = i.leapfrog(sch.force,sch.evolveQ,50,50.0*e)
            sim = h.hmc(sch, lf, True)
            #Evolve, say, 10 steps of HMC- not clear from paper how many
            #HMC steps are run
            u_upd = sim.evolve(u, 10)
            #Generate plaquette matrix of new config
            pl = u_upd[:,:,:,0]*tr.roll(u_upd[:,:,:,1], shifts=-1, dims=1) \
            * tr.conj(tr.roll(u_upd[:,:,:,0], shifts=-1, dims=2))*tr.conj(u_upd[:,:,:,1]) 
            #Average
            pl_avg.append(tr.mean(pl))

        fig, ax1 = plt.subplots(1,1)

        ax1.plot(np.linspace(0.5, 16, 100), pl_avg)
        ax1.set_ylabel(r'Plaquette Average')
        ax1.set_xlabel(r'$\epsilon$')
        plt.show()

        
            

        










    



if __name__ == "__main__":
   main()