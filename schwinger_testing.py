#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 2024

@author: Ben Slimmer
"""

#File for tests of the Schwinger model code

import numpy as np;
import scipy as sp;
import torch as tr;
import update as h;
import integrators as i;
import schwinger as s;
import time;
import matplotlib.pyplot as plt;
import pandas as pd;
#High performance compiler
from numba import njit, jit

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
    batch_size=10 
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
        sim = h.hmc(sch, im2, True)
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
    L = 16
    L2 = 8
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Average correlation function for each lattice timeslice
    c_list = []
    c_err = []

    u = sch.hotStart()
    #Need to make a tuple
    q =(u,)

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,30, 1.0)
    sim = h.hmc(sch, im2, False)
    
    #Bring lattice to equilibrium
    q = sim.evolve_f(q, 5)



    #Measurement process- n measurements on 1000 batches
    for n in np.arange(10):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())


        #Measure correlation function for different lengths
        #for nt in np.arange(0, L):
            
        #Vector of time slice correlations
        cl = sch.exact_Pion_Correlator(d_inv)
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
    ax1.set_title(str(batch_size) + ' batch '+ str(L) + 'x' + str(L2) + r' lattice, $\beta$ = ' + str(1/lam**2) +' m=' + str(mass))

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
    im2 = i.minnorm2(sch.force,sch.evolveQ,100, 1.0)
    sim = h.hmc(sch, im2, True)

    q = sim.evolve_f(q, 5)


#Testing equilibration of full dynamical model
def dynamical_action():
    #Plaquette average of dynamical fermion theory
    L=8
    batch_size=100
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
        q = sim.evolve_f(q, 1)
        cq = sim.evolve_f(cq, 1)
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
    ax1.set_ylabel(r'$\langle S \rangle$', fontsize=32)
    ax1.yaxis.set_tick_params(labelsize=28)
    ax1.set_xlabel(r'n', fontsize=32)
    ax1.xaxis.set_tick_params(labelsize=28)
    ax1.legend(loc='lower right')
    ax1.set_title(str(batch_size) + ' batch '+ str(L) + 'x' + str(L) + r' lattice, $\beta$ = ' + str(1/lam**2) +' m=' + str(mass))
    plt.show()


#Testing Monte Carlo integration of full dynamical model
def dynamical_dH_vs_eps2():
    L=4
    batch_size=100
    lam =np.sqrt(1.0/10.0)
    mass= 0.02
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u0 = sch.hotStart()

    d = sch.diracOperator(u0)

    f = sch.generate_Pseudofermions(d)

    #Define q as a tuple of the gauge field, psuedofermion field, Dirac operator
    q0 = (u0, f, d)

    e2 = []
    lf_dh = []
    lf_herr= []
    mn_dh = []
    mn_herr = []

    p0 = sch.refreshP()
    H0 = sch.action(q0) + sch.kinetic(p0)

    for n in np.arange(10, 51):
        imlf = i.leapfrog(sch.force, sch.evolveQ, n, 1.0)
        im2 = i.minnorm2(sch.force, sch.evolveQ, n, 1.0)
        lf_p, lf_q = imlf.integrate(p0, q0)
        mn_p, mn_q = im2.integrate(p0, q0)
        lf_H = sch.action(lf_q) + sch.kinetic(lf_p)
        mn_H = sch.action(mn_q) + sch.kinetic(mn_p)
        lf_dh.append(tr.mean(lf_H - H0))
        lf_herr.append(tr.std(lf_H - H0) / np.sqrt(batch_size - 1))
        mn_dh.append(tr.mean(mn_H - H0))
        mn_herr.append(tr.std(mn_H - H0) / np.sqrt(batch_size - 1))
        e2.append((1.0/n)**2)
        print(n)

    #fig, (ax1, ax2) = plt.subplots(2,1)
    fig, ax2 = plt.subplots(1,1)
    fig.suptitle(r'$\beta=$'+str(1/lam**2) + r', ' + r'$m_0=$' + str(mass), fontsize=32)
    # ax1.errorbar(e2, lf_dh, yerr=lf_herr)
    # ax1.set_ylabel(r'$\Delta H$', fontsize=14)
    # ax1.set_xlabel(r'$\epsilon^2$')
    # ax1.set_title('Leapfrog integrator')

    #ax2.errorbar(e2, mn_dh, yerr=mn_herr)
    ax2.errorbar(np.arange(10,51)**2, mn_dh, yerr=mn_herr)
    ax2.yaxis.set_tick_params(labelsize=28)
    ax2.set_ylabel(r'$\Delta H$', fontsize=32)
    ax2.xaxis.set_tick_params(labelsize=28)
    ax2.set_xlabel(r'Integration steps$^2$', fontsize=32)
    #ax2.set_title('Min norm squared integrator')

    plt.show()


#Full dynamical model spectroscopy calculation, using explicit construction of fermion force
def pi_plus_mass():
    #Measurement process -given function for correlator of pi plus
    batch_size=5
    #lam =np.sqrt(1.0/0.970)
    lam = np.sqrt(1.0/3.0)
    mass= 0.05
    L = 16
    L2 = 8
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Average correlation function for each lattice timeslice
    c_list = []
    c_err = []

    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    #Define q as a tuple of the gauge field, psuedofermion field, Dirac operator
    q =(u, f, d)

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,5, 1.0)
    sim = h.hmc(sch, im2, True)
    
    #Bring lattice to equilibrium
    q = sim.evolve_f(q, 50)



    #Measurement process- n measurements on batches
    for n in np.arange(10):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())

            
        #Vector of time slice correlations
        cl = sch.exact_Pion_Correlator(d_inv)
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
    ax1.set_title(str(batch_size) + ' batch '+ str(L) + 'x' + str(L2) + r' lattice, $\beta$ = ' + str(1/lam**2) +' m=' + str(mass))

    plt.show()

#Testing autograd implementation of force and its speed improvement
def autograd_test():
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

    #Runs successfully
    #sch.autograd_force(q0)

    print("4x4 lattice")
    #Time autograd
    start = time.time()
    sch.autograd_force(q0)
    end = time.time()
    #0.035379 s
    print("Autograd: ", end - start)

    #Time analytic construction
    start = time.time()
    sch.force(q0)
    end = time.time()
    #0.065135 s - twice as long
    print("Analytic: ", end - start)

    #Try a larger lattice...
    print("8x8 lattice")
    L=8
    batch_size=50
    lam =np.sqrt(1.0/4.0)
    mass= 0.4
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u0 = sch.hotStart()

    d = sch.diracOperator(u0)

    f = sch.generate_Pseudofermions(d)

    #Define q as a tuple of the gauge field, psuedofermion field, Dirac operator
    q0 = (u0, f, d)

    #Runs successfully
    #sch.autograd_force(q0)

    #Time autograd
    start = time.time()
    sch.autograd_force(q0)
    end = time.time()
    #0.130516 s
    print("Autograd: ", end - start)

    #Time analytic construction
    start = time.time()
    sch.force(q0)
    end = time.time()
    #7.02905 s - Huge difference!!
    print("Analytic: ", end - start)

#Autograd force integrator behaves poorly
def autograd_dH_vs_eps2():
    L=6
    batch_size=100
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
        im2 = i.leapfrog(sch.autograd_force, sch.evolveQ, n, 1.0)
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

#Check equilibrium
#Takes a few more steps than the analytic force approach, but does equilibrate
def autograd_dynamical_action():
    #Total action of dynamical fermion theory
    L=6
    batch_size=100
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
    S0 = sch.action(q)
    pl_avg.append(tr.mean(S0))
    pl_err.append(tr.std(S0)/np.sqrt(tr.numel(S0) - 1))
    cS0 = sch.action(cq)
    cpl_avg.append(tr.mean(cS0))
    cpl_err.append(tr.std(cS0)/np.sqrt(tr.numel(cS0) - 1))

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.autograd_force,sch.evolveQ,30, 1.0)
    sim = h.hmc(sch, im2, False)

    steps = 60
    for n in np.arange(0, steps):
        #Evolve, one HMC step
        q = sim.evolve_f(q, 1)
        cq = sim.evolve_f(cq, 1)
        u = q[0]
        cu = cq[0]
        #Average action
        S = sch.action(q)
        pl_avg.append(tr.mean(S))
        pl_err.append(tr.std(S)/np.sqrt(tr.numel(S) - 1))
        cS = sch.action(cq)
        cpl_avg.append(tr.mean(cS))
        cpl_err.append(tr.std(cS)/np.sqrt(tr.numel(cS) - 1))
        print(n)

    fig, ax1 = plt.subplots(1,1)

    ax1.errorbar(np.arange(steps + 1), pl_avg, pl_err, label="Hot Start")
    ax1.errorbar(np.arange(steps + 1), cpl_avg, cpl_err, label = "Cold Start")
    ax1.set_ylabel(r'$S_{total}$')
    ax1.set_xlabel(r'n')
    ax1.legend(loc='lower right')
    plt.show()

#Compare results of pi plus mass calculation using explicit and autograd approach on a small lattice
def autograd_pi_plus_comparison():
    #Measurement process -given function for correlator of pi plus
    batch_size=10
    #lam =np.sqrt(1.0/0.970)
    lam = np.sqrt(1.0/4.0)
    mass= 0.4
    L = 16
    L2 = 8
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Average correlation function for each lattice timeslice
    #For explicit force integrator
    e_c_list = []
    e_c_err = []

    #For autograd integrator
    a_c_list = []
    a_c_err = []

    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    #Define q as a tuple of the gauge field, psuedofermion field, Dirac operator
    q =(u, f, d)

    #Tune integrator to desired step size
    #Explicit force integrator
    e_im2 = i.minnorm2(sch.force,sch.evolveQ,30, 1.0)
    e_sim = h.hmc(sch, e_im2, False)

    #Autograd integrator
    a_im2 = i.minnorm2(sch.autograd_force,sch.evolveQ,30, 1.0)
    a_sim = h.hmc(sch, a_im2, False)
    
    #Bring lattice to equilibrium
    e_q = e_sim.evolve_f(q, 20)

    a_q = a_sim.evolve_f(q, 20)



    #Measurement process- n measurements on the batches
    for n in np.arange(10):
        #Discard some in between
        e_q = e_sim.evolve_f(e_q, 10)

        a_q = a_sim.evolve_f(a_q, 10)
        
        e_d = sch.diracOperator(e_q[0])
        e_d_inv = tr.linalg.inv(e_d.to_dense())

        a_d = sch.diracOperator(a_q[0])
        a_d_inv = tr.linalg.inv(a_d.to_dense())


        #Measure correlation function for different lengths
        #for nt in np.arange(0, L):
            
        #Vector of time slice correlations
        e_cl = sch.exact_Pion_Correlator(e_d_inv)
        a_cl = sch.exact_Pion_Correlator(a_d_inv)
        if n ==0:
            e_c = e_cl
            a_c = a_cl
        else:
            e_c= tr.cat((e_c, e_cl), 0)
            a_c= tr.cat((a_c, a_cl), 0)
        print(n)

    #Average over all batches and configurations measured
    e_c_avg = tr.mean(e_c, dim=0)
    e_c_err = (tr.std(e_c, dim=0)/np.sqrt(tr.numel(e_c[:,0]) - 1))

    a_c_avg = tr.mean(a_c, dim=0)
    a_c_err = (tr.std(a_c, dim=0)/np.sqrt(tr.numel(a_c[:,0]) - 1))

    fig, ax1 = plt.subplots(1,1)

    ax1.set_yscale('log', nonpositive='clip')


    ax1.errorbar(np.arange(0, L), tr.abs(e_c_avg), tr.abs(e_c_err), ls="", marker=".", label = "Explicit Force")
    ax1.errorbar(np.arange(0, L), tr.abs(a_c_avg), tr.abs(a_c_err), ls="", marker=".", label = "Autograd")

    ax1.legend(loc='lower right')

    plt.show()

#Full dynamical model spectroscopy calculation, using autograd construction of fermion force
def autograd_pi_plus_mass():
    #Measurement process -given function for correlator of pi plus
    batch_size=5
    #lam =np.sqrt(1.0/0.970)
    lam = np.sqrt(1.0/5.0)
    mass= 0.2*lam
    L = 20
    L2 = 20
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Average correlation function for each lattice timeslice
    c_list = []
    c_err = []

    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    #Define q as a tuple of the gauge field, psuedofermion field, Dirac operator
    q =(u, f, d)

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.autograd_force,sch.evolveQ,30, 1.0)
    sim = h.hmc(sch, im2, True)
    
    #Bring lattice to equilibrium
    q = sim.evolve_f(q, 30)



    #Measurement process- n measurements on the batches
    for n in np.arange(10):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())


        #Measure correlation function for different lengths
        #for nt in np.arange(0, L):
            
        #Vector of time slice correlations
        cl = sch.exact_Two_Point_correlator(d_inv)
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
    

#Fit function for pion triplet
#TODO: N_T is hardcoded- way to pass in fitting process?
def f_pi_triplet(x, m, A):
    N_T = 10
    return A* (np.exp(-m *x) + np.exp(-(N_T - x)*m))

#Fit for analytical pion mass as a function of quark mass and coupling to fit critical mas
def f_analytical_pi_triplet(x, mc):
    #Coupling
    lam = 1.0/np.sqrt(10.0)
    return 2.066*np.power(((x-mc)/lam)**2, 1.0/3.0) * lam


#Fit function for finite volume effects
def f_FV_Mass(L, m_inf, A):
    return m_inf + A*(np.sqrt(m_inf)/np.sqrt(L))*np.exp(-m_inf*L)

#Effective mass curve looks OK to eye test- run a fit on the curve to estimate mass
def pion_triplet_fit():
    #Measurement process -given function for correlator of pi plus
    batch_size=30
    #lam =np.sqrt(1.0/0.970)
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass... Need critical mass offset for analytical comparison
    mass= -0.08*lam
    L = 10
    L2 = 8
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)


    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    #Define q as a tuple of the gauge field, psuedofermion field, Dirac operator
    q =(u, f, d)

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,25, 1.0)
    sim = h.hmc(sch, im2, True)
    
    #Equilibration
    q = sim.evolve_f(q, 50)



    #Measurement process- nm measurements on batches
    nm = 10
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())

            
        #Vector of time slice correlations
        cl = sch.exact_Pion_Correlator(d_inv, (0,), p=1*(2*np.pi)/L2)
        if n ==0:
            c = cl
        else:
            c= tr.cat((c, cl), 0)
        print(n)

    #Average over all batches and configurations measured
    c_avg = tr.mean(c, dim=0)
    c_err = (tr.std(c, dim=0)/np.sqrt(tr.numel(c[:,0]) - 1))

    #Write dataframe of data

    df = pd.DataFrame([c_avg.detach().numpy(), c_err.detach().numpy()])
    #TODO: Write more descriptive datafile name
    df.to_csv("output.csv", index = False)

    #Fit the effective mass curve
    popt, pcov = sp.optimize.curve_fit(f_pi_triplet, np.arange(0, L), tr.abs(c_avg), sigma = tr.abs(c_err))
    print(popt)
    print(pcov)

    m_eff = np.log(c_avg[:L-1]/c_avg[1:]) 

    print(m_eff)


    fig, ax1 = plt.subplots(1,1)

    ax1.set_yscale('log', nonpositive='clip')


    ax1.errorbar(np.arange(0, L), tr.abs(c_avg), tr.abs(c_err), ls="", marker=".")
    ax1.set_title(str(batch_size * nm) + ' config '+ str(L) + 'x' + str(L2) + r' lattice, $\beta$ = ' + str(1/lam**2) +' m=' + str(mass))

    plt.show()

#For importing previously generated correlation function data, and producing/plotting a fit
def import_fit():
    df = pd.read_csv('output.csv')
    a = df.to_numpy()


    fig, ax1 = plt.subplots(1,1)

    ax1.set_yscale('log', nonpositive='clip')


    #Fit the effective mass curve
    #Select only the time slices near the center of the lattice
    popt, pcov = sp.optimize.curve_fit(f_pi_triplet, np.arange(3,7), np.abs(a[0, 3:7]), sigma = np.abs(a[1, 3:7]))
    print(popt)
    print(pcov)

    #Plot the fit & data
    #ax1.plot(np.linspace(0, 10, 100), f_pi_triplet(np.linspace(0, 10, 100), *popt))
    ax1.errorbar(np.arange(0, 10), np.abs(a[0]), np.abs(a[1]), ls="", marker=".", markersize=10, elinewidth=2.0)

    ax1.set_ylabel(r'$C(t)$', fontsize=32)
    ax1.set_xlabel(r'$n_t$', fontsize=32)

    ax1.yaxis.set_tick_params(labelsize=28)
    ax1.xaxis.set_tick_params(labelsize=28)

    ax1.set_title(r'$p= \frac{4 \pi}{L}$', fontsize=32)

    ax1.set_ylim(0.000001, 1)

    plt.show()


#Fit FV effects
def FV_Fitting():
    #Match filename
    df = pd.read_csv('FV_fit_beta=10_m0=0.csv')

    fig, ax1 = plt.subplots(1,1)

    popt, pcov = sp.optimize.curve_fit(f_FV_Mass, df['L'], df['m'], sigma=df['std_dev'])
    print(popt)
    print(pcov)

    #Plot fit and data
    ax1.plot(np.linspace(2, 14, 100), f_FV_Mass(np.linspace(2,14,100), *popt))
    ax1.errorbar(df['L'], df['m'], df['std_dev'], ls="", marker=".")

    ax1.set_ylabel(r'$m_\pi$')
    ax1.set_xlabel(r'$L$')

    plt.show()


#Fitting critical mass of Wilson-Dirac operator for given coupling
def fit_critical_mass():
    #Match filename
    df = pd.read_csv('beta=10_L=10_m0_scan.csv')
    #Coupling
    lam = 1.0/np.sqrt(10.0)

    fig, ax1 = plt.subplots(1,1)
    
    #Fit to lowest mo values
    df_fit = df.loc[(df['m/g'] <= 0.3) & (df['m/g']>=0.2)]
    #df_fit = df
    popt, pcov = sp.optimize.curve_fit(f_analytical_pi_triplet, df_fit['m0'], df_fit['mpi exp'], sigma = df_fit['std_dev'], p0=(-0.12))
    print(popt)
    print(pcov)
    

    #Plot the fit
    ax1.plot((np.linspace(popt, 0.3, 1000000) -popt)/lam, f_analytical_pi_triplet(np.linspace(popt, 0.3, 1000000), *popt)/lam, linewidth=3.0)
    ax1.set_xlim(0.1, 0.5)



    ax1.errorbar((df['m0']-popt)/lam, df['mpi exp']/lam, df['std_dev']/lam, ls="", marker=".", markersize=10.0, elinewidth=2.0)

    ax1.set_ylabel(r'$m_\pi/e$', fontsize=32)
    ax1.set_xlabel(r'$m_q/e$', fontsize=32)

    ax1.yaxis.set_tick_params(labelsize=28)
    ax1.xaxis.set_tick_params(labelsize=28)

    ax1.set_title(r'300 Configurations, $\beta =10$', fontsize=32)

    plt.show()

#TODO: Needs review
def topological_Charge_Distribution():
    #Measurement process -given function for correlator of pi plus
    batch_size=30
    #lam =np.sqrt(1.0/0.970)
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass... Need critical mass offset for analytical comparison
    mass= -0.08*lam
    L = 10
    L2 = 8
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)


    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    #Define q as a tuple of the gauge field, psuedofermion field, Dirac operator
    q =(u, f, d)

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,25, 1.0)
    sim = h.hmc(sch, im2, True)
    
    #Equilibration
    q = sim.evolve_f(q, 50)


    charge = tr.sum(tr.real(tr.log(q[0])/1.0j) % (2.0*np.pi), dim=(1,2,3))/(2.0*np.pi*L*L2) - 1.0
    #Measurement process- nm measurements on batches
    nm = 20
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        charge = tr.cat((charge, tr.sum(tr.real(tr.log(q[0])/1.0j) % (2.0*np.pi), dim=(1,2,3))/(2.0*np.pi*L*L2) -1.0))

    fig, ax1 = plt.subplots(1,1)

    df = pd.DataFrame(charge.detach().numpy())
    #TODO: Write more descriptive datafile name
    df.to_csv("output.csv", index = False)

    ax1.hist(charge)

    plt.show()



def main():
    #dH_eps2()
    #quenched_pi_plus_mass()
    #fermion_action()
    #fermion_force()
    #pure_gauge_Savg()
    #trivial_D_inspect()
    #dynamical_action()
    dynamical_dH_vs_eps2()
    #pi_plus_mass()
    #autograd_test()
    #autograd_dH_vs_eps2()
    #autograd_dynamical_action()
    #pi_plus_comparison()
    #autograd_pi_plus_mass()
    #pion_triplet_fit()
    #import_fit()
    #fit_critical_mass()
    #FV_Fitting()
    #topological_Charge_Distribution()
    
    #force_speed_test()








    



if __name__ == "__main__":
   main()