#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 2024

@author: Ben Slimmer
"""

#File for tests of the Schwinger model code

import numpy as np;
import torch as tr;
import update as h;
import integrators as i;
import schwinger as s;
import time;
import matplotlib.pyplot as plt;
import pandas as pd;

#Gauge theory HMC integrator test
def dH_eps2():
    #Average over a batch of configurations
    #Parameters imitate that of Duane 1987 HMC Paper
    L=8
    batch_size=1000
    lam = np.sqrt(1.0/4.0)
    mass= 0.4
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u0 = (sch.hotStart(),)

    e2 = []
    dh = []
    h_err= []

    p0 = sch.refreshP()
    H0 = sch.action(u0) + sch.kinetic(p0)

    for n in np.arange(10, 51):
        im2 = i.minnorm2(sch.force, sch.evolveQ, n, 1.0)
        p, u = im2.integrate(p0, u0)
        H = sch.action(u) + sch.kinetic(p)
        dh.append(tr.mean(H - H0))
        h_err.append(tr.std(H - H0) / np.sqrt(batch_size - 1))
        e2.append((1.0/n)**2)
        print(n)

    fig, ax1 = plt.subplots(1,1)

    ax1.errorbar(e2, dh, yerr=h_err)
    ax1.set_ylabel(r'$\Delta H$')
    ax1.set_xlabel(r'$\epsilon^2$')
    plt.show()

#Testing equlibration of gauge theory HMC
def pure_gauge_Savg():
    #Gauge theory plaquette average
    #Average over a batch of configurations
    #Parameters imitate that of Duane 1987 HMC Paper
    #16^2 lattice to more closely align with Lang 1986 paper
    L=16
    batch_size=1000 
    lam =np.sqrt(1.0/0.970)
    mass= 0.1
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u = sch.hotStart()

    pl_avg = []
    pl_err = []

    cpl_avg = []
    cpl_err = []
    cu = sch.coldStart()

    #Gauge theory- tuple contains one element, the gauge field
    q = (u,)
    cq = (cu,)

    #Average action
    S0 = sch.gaugeAction(u) / float(L**2)
    pl_avg.append(tr.mean(S0))
    pl_err.append(tr.std(S0)/np.sqrt(tr.numel(S0) - 1))
    cS0 = sch.gaugeAction(cu) / float(L**2)
    cpl_avg.append(tr.mean(cS0))
    cpl_err.append(tr.std(cS0)/np.sqrt(tr.numel(cS0) - 1))


    
    for n in np.arange(0, 50):
        #Tune integrator to desired step size
        im2 = i.minnorm2(sch.force,sch.evolveQ,50, 1.0)
        sim = h.hmc(sch, im2, False)
        #Evolve, one HMC step
        q = sim.evolve_f(q, 1)
        cq = sim.evolve_f(cq, 1)
        u = q[0]
        cu = cq[0]
        #Average action
        S = sch.gaugeAction(u) / float(L**2)
        pl_avg.append(tr.mean(S))
        pl_err.append(tr.std(S)/np.sqrt(tr.numel(S) - 1))
        cS = sch.gaugeAction(cu) / float(L**2)
        cpl_avg.append(tr.mean(cS))
        cpl_err.append(tr.std(cS)/np.sqrt(tr.numel(cS) - 1))

    fig, ax1 = plt.subplots(1,1)

    ax1.errorbar(np.arange(51), pl_avg, pl_err, label="Hot Start")
    ax1.errorbar(np.arange(51), cpl_avg, cpl_err, label = "Cold Start")
    ax1.set_ylabel(r'$\langle S \rangle$')
    ax1.set_xlabel(r'n')
    ax1.legend(loc='lower right')
    plt.show()

#Quick look at D operator of trivial gauge field
def trivial_D_inspect():
    #Simple Dirac operator test
    #Checking for sparsity and correct non-zero entries of a trivial gauge field
    #Looks good by eye inspection
    L=4
    batch_size=1 
    lam =np.sqrt(1.0/0.970)
    mass= 0.1
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u = sch.coldStart()

    d = sch.diracOperator(u)
    df = pd.DataFrame(d.to_dense()[0,:,:])
    df.to_csv("Testfile.csv")

#Plotting D operator eigenvalues to confirm correct complex conjugate behavior
def D_eigenvalues():
    #Check for correct complex conjugate behavior of D operator eigenvalues
    batch_size=1
    #lam =np.sqrt(1.0/0.970)
    lam = np.sqrt(1.0/4.0)
    mass= 0.5
    L = 16
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u = sch.coldStart()
    d = sch.diracOperator(u)

    eig, v = tr.linalg.eig(d.to_dense())
    print(tr.numel(eig))

    fig, ax1 = plt.subplots(1,1)
    ax1.scatter(eig.real, eig.imag)
    plt.show()

#Testing functionality of D operator and related fermion action - quenched approximation
def fermion_action():
    #Average over a batch of configurations
    #Parameters imitate that of Duane 1987 HMC Paper
    #16^2 lattice to more closely align with Lang 1986 paper
    L=8
    batch_size=100 
    lam =np.sqrt(1.0/4.0)
    mass= 0.1
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u = (sch.hotStart(),)

    pl_avg = []
    pl_err = []

    cpl_avg = []
    cpl_err = []
    cu = (sch.coldStart(),)

    d = sch.diracOperator(u[0])
    cd =sch.diracOperator(cu[0])

    f = sch.generate_Pseudofermions(d)
    cf = sch.generate_Pseudofermions(cd)

    #Define q as a tuple of gauge field, pseudofermions, and dirac operator
    q = (u[0], f, d)
    cq = (cu[0], cf, cd)

    #Average action
    S0 = sch.action(q) / float(L**2)
    pl_avg.append(tr.mean(S0))
    pl_err.append(tr.std(S0)/np.sqrt(tr.numel(S0) - 1))
    cS0 = sch.action(cq) / float(L**2)
    cpl_avg.append(tr.mean(cS0))
    cpl_err.append(tr.std(cS0)/np.sqrt(tr.numel(cS0) - 1))


    k=20
    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,50, 1.0)
    sim = h.hmc(sch, im2, False)
    for n in np.arange(0, k):
        
        #Evolve gauge field, one HMC step
        u = sim.evolve_f(u, 1)
        cq = sim.evolve_f(cu, 1)

        #Update dirac operator + psuedofermions
        d = sch.diracOperator(u[0])
        cd =sch.diracOperator(cu[0])
        f = sch.generate_Pseudofermions(d)
        cf = sch.generate_Pseudofermions(cd)
        q = (u[0], f, d)
        cq = (cu[0], f, d)



        #Average action
        S = sch.action(q) / float(L**2)
        pl_avg.append(tr.mean(S))
        pl_err.append(tr.std(S)/np.sqrt(tr.numel(S) - 1))
        cS = sch.action(cq) / float(L**2)
        cpl_avg.append(tr.mean(cS))
        cpl_err.append(tr.std(cS)/np.sqrt(tr.numel(cS) - 1))
        print(n)

    fig, ax1 = plt.subplots(1,1)

    ax1.errorbar(np.arange(k+1), pl_avg, pl_err, label="Hot Start")
    ax1.errorbar(np.arange(k+1), cpl_avg, cpl_err, label = "Cold Start")
    ax1.set_ylabel(r'$\langle S \rangle$')
    ax1.set_xlabel(r'n')
    ax1.legend(loc='lower right')
    #ax1.set_xlim(2, 50)
    #ax1.set_ylim(pl_avg[k]-5, pl_avg[5]+500)
    plt.show() 

#Determining pi plus mass in quenched approximation using effective mass curve
def quenched_pi_plus_mass():
    #Measurement process -given function for correlator of pi plus
    batch_size=1000
    #lam =np.sqrt(1.0/0.970)
    lam = np.sqrt(1.0/4.0)
    mass= 0.4
    L = 32
    L2 = 16
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Average correlation function for each lattice timeslice
    c_list = []
    c_err = []

    u = sch.hotStart()
    #Need to make a tuple
    q =(u,)

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,50, 1.0)
    sim = h.hmc(sch, im2, False)
    
    #Bring lattice to equilibrium
    q = sim.evolve_f(q, 5, False)



    #Measurement process- n measurements on 1000 batches
    for n in np.arange(10):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())


        #Measure correlation function for different lengths
        #for nt in np.arange(0, L):
            
        #Vector of time slice correlations
        cl = sch.pi_plus_correlator(d_inv)
        if n ==0:
            c = cl
        else:
            c= tr.cat((c, cl), 0)
        print(n)

    #Average over all batches and configurations measured
    c_avg = tr.mean(c, dim=0)
    c_err = (tr.std(c, dim=0)/np.sqrt(tr.numel(c[:,0]) - 1))

    m_eff = np.log(c_avg[:L-1]/c_avg[1:]) 

    print(m_eff)


    fig, ax1 = plt.subplots(1,1)

    ax1.set_yscale('log', nonpositive='clip')


    ax1.errorbar(np.arange(0, L), tr.abs(c_avg), tr.abs(c_err), ls="", marker=".")

    plt.show()

#Test functionality of full dynamical model, beginning with force of fermion fields
def fermion_force():
    
    L=8
    batch_size=1
    lam =np.sqrt(1.0/4.0)
    mass= 0.4
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    #Define q as a tuple of the gauge field, psuedofermion field, Dirac operator
    q = (u, f, d)

    #Check it runs without error first
    #sch.force(q)

    # A few update steps
    #Seeing huge increases in configuration energy, new config always rejected
    im2 = i.minnorm2(sch.force,sch.evolveQ,100, 1.0)
    sim = h.hmc(sch, im2, True)

    q = sim.evolve_f(q, 5, True)


#Testing equilibration of full dynamical model
def dynamical_action():
    #Plaquette average of dynamical fermion theory
    L=8
    batch_size=10
    lam =np.sqrt(1.0/4)
    mass= 0.4
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u = sch.hotStart()

    pl_avg = []
    pl_err = []

    cpl_avg = []
    cpl_err = []
    cu = sch.coldStart()

    #full dynamical theory- requires psuedofermions and dirac operator in tuple
    d = sch.diracOperator(u)
    cd = sch.diracOperator(cu)
    f = sch.generate_Pseudofermions(d)
    cf = sch.generate_Pseudofermions(cd)
    q = (u, f, d)
    cq = (cu, cf, cd)

    #Average action
    S0 = sch.action(q) / float(L**2)
    pl_avg.append(tr.mean(S0))
    pl_err.append(tr.std(S0)/np.sqrt(tr.numel(S0) - 1))
    cS0 = sch.action(cq) / float(L**2)
    cpl_avg.append(tr.mean(cS0))
    cpl_err.append(tr.std(cS0)/np.sqrt(tr.numel(cS0) - 1))

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,25, 1.0)
    sim = h.hmc(sch, im2, False)

    steps = 20
    for n in np.arange(0, steps):
        #Evolve, one HMC step
        q = sim.evolve_f(q, 1, True)
        cq = sim.evolve_f(cq, 1, True)
        u = q[0]
        cu = cq[0]
        #Average action
        S = sch.action(q) / float(L**2)
        pl_avg.append(tr.mean(S))
        pl_err.append(tr.std(S)/np.sqrt(tr.numel(S) - 1))
        cS = sch.action(cq) / float(L**2)
        cpl_avg.append(tr.mean(cS))
        cpl_err.append(tr.std(cS)/np.sqrt(tr.numel(cS) - 1))
        print(n)

    fig, ax1 = plt.subplots(1,1)

    ax1.errorbar(np.arange(steps + 1), pl_avg, pl_err, label="Hot Start")
    ax1.errorbar(np.arange(steps + 1), cpl_avg, cpl_err, label = "Cold Start")
    ax1.set_ylabel(r'$\langle S \rangle$')
    ax1.set_xlabel(r'n')
    ax1.legend(loc='lower right')
    plt.show()

#Testing Monte Carlo integration of full dynamical model
def dynamical_dH_vs_eps2():
    L=4
    batch_size=50
    lam =np.sqrt(1.0/4.0)
    mass= 0.4
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u0 = sch.hotStart()

    d = sch.diracOperator(u0)

    f = sch.generate_Pseudofermions(d)

    #Define q as a tuple of the gauge field, psuedofermion field, Dirac operator
    q0 = (u0, f, d)

    e2 = []
    dh = []
    h_err= []

    p0 = sch.refreshP()
    H0 = sch.action(q0) + sch.kinetic(p0)

    for n in np.arange(10, 51):
        im2 = i.minnorm2(sch.force, sch.evolveQ, n, 1.0)
        p, q = im2.integrate(p0, q0)
        H = sch.action(q) + sch.kinetic(p)
        dh.append(tr.mean(H - H0))
        h_err.append(tr.std(H - H0) / np.sqrt(batch_size - 1))
        e2.append((1.0/n)**2)
        print(n)

    fig, ax1 = plt.subplots(1,1)

    ax1.errorbar(e2, dh, yerr=h_err)
    ax1.set_ylabel(r'$\Delta H$')
    ax1.set_xlabel(r'$\epsilon^2$')
    ax1.set_title(r'$\beta=4$, $m=0.4$')
    plt.show()



def main():
    #dH_eps2()
    #quenched_pi_plus_mass()
    #fermion_action()
    #fermion_force()
    #pure_gauge_Savg()
    #dynamical_action()
    dynamical_dH_vs_eps2()







    



if __name__ == "__main__":
   main()