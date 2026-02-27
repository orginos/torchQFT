#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 2024

@author: Ben Slimmer
"""

#File for testing domain decompositon based code for Schwinger model

import numpy as np;
import scipy as sp;
import torch as tr;
import update as h;
import integrators as i;
import schwinger as s;
import time;
import matplotlib.pyplot as plt;
import pandas as pd;

#Plotting settings 
plt.rcParams.update({
    # ---- Figure size ----
#    "figure.figsize": (3.4, 6.5),   # tall 3-panel column figure
    "figure.dpi": 300,

    # ---- Fonts ----
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # ---- Lines ----
    "lines.linewidth": 1.5,
    "lines.markersize": 5,

    # ---- Axes ----
    "axes.linewidth": 1.2,

    # ---- Ticks ----
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,

    # ---- Legend ----
    "legend.frameon": False,

    # ---- Errorbars ----
    "errorbar.capsize": 3,
})

#Check that the rearranged Dirac operator has block diagonal property
def block_Diagonal_Check():
    batch_size=1
    #lam =np.sqrt(1.0/0.970)
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass... Need critical mass offset for analytical comparison
    mass= 0.05
    L = 10
    L2 = 10
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)


    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)
    
    q= (u,f,d)
    
    #Regular Dirac operator
    d = sch.diracOperator(u).to_dense()
    plt.spy(d[0,:,:])

    plt.show()

    #Block diagonal
    dd = sch.bb_DiracOperator(q, 3, 8, 2).to_dense()
    plt.spy(dd[0,:,:])

    plt.show()

    #Isolate sub matrices
    #Assumes 2 width 2 timeslice boundaries
    d00 = dd[0,0:4*2*L, 0:4*2*L]
    d01 = dd[0, 0:4*2*L, 4*2*L:]
    d10 = dd[0, 4*2*L:, 0:4*2*L]
    d11 = dd[0, 4*2*L:, 4*2*L:]

    #Schur complement
    s11 = d11 - tr.einsum('ij, jk, km->im', d10, tr.inverse(d00), d01)


    plt.spy(d11)
    plt.show()

    d11_inv = tr.inverse(d11)
    plt.spy(d11_inv)
    plt.show()

    
    plt.spy(tr.inverse(s11) - tr.inverse(d11))
    plt.show()

#Checking if exact factorized propogator matches with traditional approach
def propogator_Comparison():
    batch_size=1
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass... Need critical mass offset for analytical comparison
    mass= -0.06*lam
    L = 10
    L2 = 10
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)


    u = sch.hotStart()

    d = sch.diracOperator(u)

    d_inv = tr.inverse(d.to_dense())

    f = sch.generate_Pseudofermions(d)

    q = (u, f, d)

    s_inv = sch.dd_Schur_Propogator(q, 3, 8, 2)

    #Compare propogator of first time slice of first subdomain
    # Very nearly match! Error of 10^-9
    print(d_inv[0,1,0:2*L2] - s_inv[0,1,0:2*L2])

#Checking if we can anticipate Neumann series to converge
#Check full D operator norm, and D00 norm
def dirac_Operator_Norm():
    batch_size= 10
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.0
    L = 10
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 3
    xcut_2 = 8
    #Approximation rank
    r=2


    u = sch.hotStart()
    q= (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)
    #Equilibration
    q = sim.evolve_f(q, 25)

    d = sch.bb_DiracOperator(q, xcut_1, xcut_2).to_dense()

    #Check the spectral radius of identity - matrix of interest:
    x = tr.eye(d.size(dim=1))
    x = x.reshape((1, d.size(dim=1), d.size(dim=1)))
    x = x.repeat(batch_size, 1, 1)

    eig = tr.linalg.eig(x - d)
    print("Full D operator: ", tr.max(tr.abs(eig[0]), dim=1))

    #What about the norm of the sub matrix we are approximating?
    d00 = d[:,0:8*L, 0:8*L]

    x = tr.eye(d00.size(dim=1))
    x = x.reshape((1, d00.size(dim=1), d00.size(dim=1)))
    x = x.repeat(batch_size, 1, 1)

    eig = tr.linalg.eig(x - d00)
    print("D00", tr.max(tr.abs(eig[0]), dim=1))

    #Does the dynamical model also not have a convergent D00 - yes
    u = sch.hotStart()
    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    q = (u, f, d)
    im2 = i.minnorm2(sch.force,sch.evolveQ,10, 1.0)
    sim = h.hmc(sch, im2, False)
    #Equilibration
    q = sim.evolve_f(q, 25)

    d = sch.bb_DiracOperator(q, xcut_1, xcut_2).to_dense()
    d00 = d[:,0:8*L, 0:8*L]

    x = tr.eye(d00.size(dim=1))
    x = x.reshape((1, d00.size(dim=1), d00.size(dim=1)))
    x = x.repeat(batch_size, 1, 1)

    eig = tr.linalg.eig(x - d00)
    print("Dynamical D00: ", tr.max(tr.abs(eig[0]), dim=1))



#Checking basic functionality of approximated factorized propogator
def approx_Propogator_Testing():
    batch_size=2
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.0
    L = 32
    L2 = 32
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 14
    xcut_2 = 30
    r = 1


    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    q = (u, f, d)

    bb_d = sch.bb_DiracOperator(q, xcut_1, xcut_2)

    #Schur complement based propogator
    p = sch.dd_Approx_Propogator(bb_d, xcut_1, xcut_2, r, 40)

    #plt.spy(p[0, 24*32:32*32, :])
    #plt.show()

    #Full propogator
    p2 = sch.approx_Propogator(d.to_dense(), 20)

    d_inv = tr.inverse(d.to_dense())

    s_inv = sch.dd_Schur_Propogator(q, 3, 8)

    #Does it match to the exact solution?- not very closely, only to ~10^-4
    #print(s_inv[0,0:60,40:42] - p[0,0:60,:])

    #How does the factorized match with the full approximated propogator?
    #Also off between 10^-4 and 10^-5
    #print(p[0,0:40,:] - p2[0,0:40, :])

    #How about the full exact propogator vs the full approx propogator
    #This starts strong near sources(10^-6~10^-7) but gets weaker at other end of lattice
    #(10^-2~10^-3)
    #print(p2[0,0:40,:] - d_inv[0,0:40, 40:42])


    #Subdomain propogators match in the exact factorization -  up to 10^-9
    #print(s_inv[0,60:100,0:2] - d_inv[0,100:140,0:2])
    #print(p[0,:,:])

#Comparing the 2 point function generated with full propogator and Schur complement
#based subdomain propogators in the quenched approximation
def quenched_two_point_Comparison():
    batch_size= 30
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.00
    L = 16
    L2 = 16
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 6
    xcut_2 = 14
    bw=2
    #Neumann Approximation rank
    r=0


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)
    
    #Does it run?
    #bb_d = sch.bb_DiracOperator(q, xcut_1, xcut_2)
    #fp = sch.dd_Approx_Propogator(bb_d, xcut_1, xcut_2, r, 0)
    #sch.dd_Two_Point_Correlator(q, xcut_1, xcut_2, r)

    #Equilibration
    q = sim.evolve_f(q, 25)


    #Measurement process- Compare nm measurements on batches
    nm = 10
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        d = sch.diracOperator(q[0])
        s_inv = sch.dd_Schur_Propogator(q, xcut_1, xcut_2, bw)
        d_inv = tr.linalg.inv(d.to_dense())

            
        #Vector of time slice correlations
        cl = sch.exact_Pion_Correlator(d_inv, (3,))
        cl2 = sch.dd_Exact_Pion_Correlator(s_inv, xcut_1, xcut_2, bw, (3,))
        if n ==0:
            c = cl
            c2= cl2
        else:
            c= tr.cat((c, cl), 0)
            c2 = tr.cat((c2, cl2),0)
        print(n)


    #Average over all batches and configurations measured
    c_avg = tr.mean(c, dim=0)
    c_err = (tr.std(c, dim=0)/np.sqrt(tr.numel(c[:,0]) - 1))

    c2_avg = tr.mean(c2, dim=0)
    c2_err = (tr.std(c2, dim=0)/np.sqrt(tr.numel(c2[:,0]) - 1))

    #Write dataframe of data

    df = pd.DataFrame([c_avg.detach().numpy(), c_err.detach().numpy(), c2_avg.detach().numpy(), c2_err.detach().numpy()])
    #TODO: Write more descriptive datafile name
    df.to_csv("output.csv", index = False)

    fig, ax1 = plt.subplots(1,1)

    ax1.set_yscale('log', nonpositive='clip')


    ax1.errorbar(np.arange(0, L), tr.abs(c_avg), tr.abs(c_err), ls="", marker=".")
    ax1.errorbar(np.arange(0, L), tr.abs(c2_avg), tr.abs(c2_err), ls="", marker=".")

    plt.show()

#Plot imported data on 2 pt functions, difference, and error differences to assess approximation
def plot_Two_Point_Difference():
    data = np.genfromtxt('m0=-0.04.csv', delimiter=',')
    
    
    fig, ax1 = plt.subplots(1,1)
    ax1.set_yscale('log', nonpositive='clip')

    ax1.errorbar(data[0], np.abs(data[1]), data[2], label=r"C(t)")
    ax1.errorbar(data[0], np.abs(data[3]), data[4], label=r"C'(t)")

    ax1.plot(data[0], data[5], label=r"C(t) - C'(t)")
    ax1.plot(data[0], data[6], label=r"Err[C(t)] - Err[C'(t)]")

    plt.title(r"$m_0 = -0.04$, $\beta = 10$, 32x32 100 configs")
    plt.legend(loc=4)

    plt.show()




#Comparing the 2 point function generated with full propogator and Schur complement
#based subdomain propogators
def two_Point_Decay_Comparison():
    batch_size=5
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.0
    L = 10
    L2 = 10
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 2
    xcut_2 = 7
    bw = 2
    #Approximation rank
    #r=2


    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    q = (u, f, d)

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,5, 1.0)
    sim = h.hmc(sch, im2, False)
    
    #Does it run?
    #fp = sch.dd_Approx_Propogator(q, xcut_1, xcut_2, r, 0)
    #sch.dd_Pion_Correlator(fp, xcut_1, xcut_2, 0)

    #Equilibration
    q = sim.evolve_f(q, 25)



    #Measurement process- Compare nm measurements on batches
    nm = 1
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())
        fp = sch.dd_Schur_Propogator(q, xcut_1, xcut_2,bw)

            
        #Vector of time slice correlations
        cl = sch.exact_Pion_Correlator(d_inv, (3,))
        cl2 = sch.dd_Exact_Pion_Correlator(fp, xcut_1, xcut_2, bw (3,))
        if n ==0:
            c = cl
            c2= cl2
        else:
            c= tr.cat((c, cl), 0)
            c2 = tr.cat((c2, cl2),0)
        print(n)

    #Average over all batches and configurations measured
    c_avg = tr.mean(c, dim=0)
    c_err = (tr.std(c, dim=0)/np.sqrt(tr.numel(c[:,0]) - 1))

    c2_avg = tr.mean(c2, dim=0)
    c2_err = (tr.std(c2, dim=0)/np.sqrt(tr.numel(c2[:,0]) - 1))

    #Write dataframe of data

    df = pd.DataFrame([c_avg.detach().numpy(), c_err.detach().numpy(), c2_avg.detach().numpy(), c2_err.detach().numpy()])
    #TODO: Write more descriptive datafile name
    df.to_csv("output.csv", index = False)

    fig, ax1 = plt.subplots(1,1)

    ax1.set_yscale('log', nonpositive='clip')


    ax1.errorbar(np.arange(0, L), tr.abs(c_avg), tr.abs(c_err), ls="", marker=".")
    ax1.errorbar(np.arange(0, L), tr.abs(c2_avg), tr.abs(c2_err), ls="", marker=".")

    plt.show()

#Fit function for pion triplet
#TODO: N_T is hardcoded- way to pass in fitting process?
def f_pi_triplet(x, m, A):
    N_T = 16
    return A* (np.exp(-m *x) + np.exp(-(N_T - x)*m))

#Fitting function for 
def fit_Exact_and_Approx():
    df = pd.read_csv('output.csv')
    a = df.to_numpy()


    fig, ax1 = plt.subplots(1,1)

    ax1.set_yscale('log', nonpositive='clip')


    #Fit the effective mass curve for the exact computation and approximation
    #Select only the time slices near the center of the lattice
    exact_popt, exact_pcov = sp.optimize.curve_fit(f_pi_triplet, np.concatenate((np.arange(3,7),np.arange(9,12))), np.concatenate((np.abs(a[0, 3:7]), np.abs(a[0, 9:12]))), sigma = np.concatenate((np.abs(a[1, 3:7]), np.abs(a[1, 9:12]))))
    print("Exact")
    print(exact_popt)
    print(exact_pcov)

    approx_popt, approx_pcov = sp.optimize.curve_fit(f_pi_triplet, np.concatenate((np.arange(3,7),np.arange(9,12))), np.concatenate((np.abs(a[2, 3:7]), np.abs(a[2, 9:12]))), sigma = np.concatenate((np.abs(a[3, 3:7]), np.abs(a[3, 9:12]))))
    print("Approximation:")
    print(approx_popt)
    print(approx_pcov)

    #Plot the fit & data
    ax1.plot(np.linspace(0, 16, 100), f_pi_triplet(np.linspace(0, 16, 100), *exact_popt), 'b')
    ax1.errorbar(np.arange(0, 16), np.abs(a[0]), np.abs(a[1]), ls="", fmt="b.", label="Exact")

    ax1.plot(np.linspace(0, 16, 100), f_pi_triplet(np.linspace(0, 16, 100), *approx_popt), 'r')
    ax1.errorbar(np.arange(0, 16), np.abs(a[2]), np.abs(a[3]), ls="", fmt="r.", label="Approx.")

    ax1.legend()

    plt.show()

#TODO: In Development
def compute_Propogator_Correction():
    #First we compute the correlator using the approximation scheme:
    batch_size= 100
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.08
    L = 32
    L2 = 32
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 14
    xcut_2 = 30
    bw=2
    #Neumann Approximation rank
    r=0

    s_range = (7,)


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    print("Approx. calculation: ")
    #Equilibration
    q = sim.evolve_f(q, 25)

    #Measurement process- Compare nm measurements on batches
    nm = 10
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        bb_d = sch.bb_DiracOperator(q, xcut_1, xcut_2, bw)

            
        #Vector of time slice correlations
        cl = sch.dd_Pion_Correlator(bb_d, xcut_1, xcut_2, bw, r, s_range)
        if n ==0:
            c= cl
        else:
            c = tr.cat((c, cl),0)
            
        print(n)

    #Now Compute the Exact observable
    batch_size= 10
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.08
    L = 32
    L2 = 32
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    print("Exact calculation: ")
    #Equilibration
    q = sim.evolve_f(q, 25)

    #Measurement process- Compare nm measurements on batches
    nm = 10
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())
            
        #Vector of time slice correlations
        cl = sch.pion_Correlator(sch.diracOperator(q[0]), s_range)

        if n ==0:
            c2= cl
        else:
            c2 = tr.cat((c2, cl),0)
        print(n)


    #Save produced correlator data
    #Average over all batches and configurations measured
    c_avg = tr.mean(c, dim=0)
    c_err = (tr.std(c, dim=0)/np.sqrt(tr.numel(c[:,0]) - 1))

    c2_avg = tr.mean(c2, dim=0)
    c2_err = (tr.std(c2, dim=0)/np.sqrt(tr.numel(c2[:,0]) - 1))

    #Write dataframe of data

    df = pd.DataFrame([c_avg.detach().numpy(), c_err.detach().numpy(), c2_avg.detach().numpy(), c2_err.detach().numpy()])
    #TODO: Write more descriptive datafile name
    df.to_csv("output.csv", index = False)




#Computing systematic error correction
#Note- this is NOT a differing Monte Carlo process- just the factorized observable
#TODO: In development
def compute_Pion_Mass_Correction():
   #First we compute the correlator using the approximation scheme:
    batch_size= 100
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.08
    L = 32
    L2 = 32
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 14
    xcut_2 = 30
    bw=2
    #Neumann Approximation rank
    r=0

    s_range = (7,)


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    #print("Approx. calculation: ")
    #Equilibration
    q = sim.evolve_f(q, 50)

    #Measurement process- Compare nm measurements on batches
    nm = 1
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10)
        bb_d = sch.bb_DiracOperator(q, xcut_1, xcut_2, bw)

            
        #Vector of time slice correlations
        cl = sch.dd_Pion_Correlator(bb_d, xcut_1, xcut_2, bw, r, s_range)
        if n ==0:
            c= cl
        else:
            c = tr.cat((c, cl),0)

        d = sch.diracOperator(q[0])
            
        #Vector of time slice correlations
        cl = sch.exact_Pion_Correlator(tr.inverse(d.to_dense()), s_range)

        if n ==0:
            c2= cl
        else:
            c2 = tr.cat((c2, cl),0)
        
        print(n)

    #Save produced correlator data
    #Average over all batches and configurations measured
    c_avg = tr.mean(c, dim=0)
    c_err = (tr.std(c, dim=0)/np.sqrt(tr.numel(c[:,0]) - 1))

    c2_avg = tr.mean(c2, dim=0)
    c2_err = (tr.std(c2, dim=0)/np.sqrt(tr.numel(c2[:,0]) - 1))

    #Write dataframe of data

    df = pd.DataFrame([c_avg.detach().numpy(), c_err.detach().numpy(), c2_avg.detach().numpy(), c2_err.detach().numpy()])
    #TODO: Write more descriptive datafile name
    df.to_csv("output.csv", index = False)


#Fit function for pion triplet
#TODO: N_T is hardcoded- way to pass in fitting process?
def f_pi_triplet(x, m, A):
    s = 3.0
    N_T = 16.0
    return A* (np.exp(-m *(x-s)) + np.exp(-(N_T - (x-s))*m))

def import_correction_fit():
    df = pd.read_csv('output.csv')
    a = df.to_numpy()


    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.set_yscale('log', nonpositive='clip')
    ax2.set_yscale('log', nonpositive='clip')


    #Fit the effective mass curve
    #Select only the time slices near the center of the lattice
    popt, pcov = sp.optimize.curve_fit(f_pi_triplet, np.arange(0,5), np.abs(a[0, 0:5]), sigma = np.abs(a[1, 0:5]))
    print("Approx:")
    print(popt)
    print(pcov)

    #Plot the fit & data
    #ax1.plot(np.linspace(0, 10, 100), f_pi_triplet(np.linspace(0, 10, 100), *popt))
    ax1.errorbar(np.arange(0, 32), np.abs(a[0]), np.abs(a[1]), ls="", marker=".", markersize=10, elinewidth=2.0)

    ax1.set_ylabel(r'$C(t)$', fontsize=32)
    ax1.set_xlabel(r'$n_t$', fontsize=32)

    ax1.yaxis.set_tick_params(labelsize=28)
    ax1.xaxis.set_tick_params(labelsize=28)


    #Fit the effective mass curve
    #Select only the time slices near the center of the lattice
    popt, pcov = sp.optimize.curve_fit(f_pi_triplet, np.arange(0,5), np.abs(a[2, 0:5]), sigma = np.abs(a[3, 0:5]))
    print("Exact:")
    print(popt)
    print(pcov)

    #Plot the fit & data
    #ax1.plot(np.linspace(0, 10, 100), f_pi_triplet(np.linspace(0, 10, 100), *popt))
    ax2.errorbar(np.arange(0, 32), np.abs(a[2]), np.abs(a[3]), ls="", marker=".", markersize=10, elinewidth=2.0)

    ax2.set_ylabel(r'$C(t)$', fontsize=32)
    ax2.set_xlabel(r'$n_t$', fontsize=32)

    ax2.yaxis.set_tick_params(labelsize=28)
    ax2.xaxis.set_tick_params(labelsize=28)




    plt.show()


#Testing Integrator behaves as an Omelyan integratior with boundary freezing
def dd_Integrator_dH():
    #Average over a batch of configurations
    L=16
    batch_size=30
    lam = np.sqrt(1.0/10.0)
    mass= -0.08*lam
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)

    u0 = (sch.hotStart(),)

    e2 = []
    dh = []
    h_err= []

    xcut1 = 6
    xcut2= 14
    bw = 2

    p0 = sch.refreshP()
    p0 = sch.dd_Freeze_P(p0, xcut1, xcut2, bw)
    H0 = sch.action(u0) + sch.kinetic(p0)

    for n in np.arange(10, 51):
        im2 = i.minnorm2(sch.dd_Force, sch.evolveQ, n, 1.0)
        p, u = im2.dd_Integrate(p0, u0, xcut1, xcut2, bw)
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

#Testing how dd integration affects action:
def dd_Integrated_Action():
    L=16
    batch_size=30
    lam = np.sqrt(1.0/10.0)
    mass= -0.08*lam
    sch = s.schwinger([L,L],lam,mass,batch_size=batch_size)



    u = sch.hotStart()

    pl_avg = []
    pl_err = []


    #Gauge theory- tuple contains one element, the gauge field
    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,50, 1.0)
    sim = h.hmc(sch, im2, False)

    q = sim.evolve_f(q, 50)

    #Average action
    S0 = sch.gaugeAction(q[0]) / float(L**2)
    pl_avg.append(tr.mean(S0))
    pl_err.append(tr.std(S0)/np.sqrt(tr.numel(S0) - 1))

    lvl2_im2 = i.minnorm2(sch.dd_Force,sch.evolveQ,20, 1.0)
    lvl2_sim = h.hmc(sch, lvl2_im2, False)

    nm = 150
    for n in np.arange(0, nm):
        #Tune integrator to desired step size
        #Evolve, one HMC step
        if(n < 100):
            q = sim.evolve_f(q, 1)
        else:
            q = lvl2_sim.second_Level_Evolve(q, 1, 6, 14, 2)
            
        u = q[0]
        #Average action
        S = sch.gaugeAction(u) / float(L**2)
        pl_avg.append(tr.mean(S))
        pl_err.append(tr.std(S)/np.sqrt(tr.numel(S) - 1))

    fig, ax1 = plt.subplots(1,1)

    ax1.errorbar(np.arange(nm+1), pl_avg, pl_err, label="Hot Start")
    ax1.set_ylabel(r'$\langle S \rangle$')
    ax1.set_xlabel(r'n')
    ax1.legend(loc='lower right')
    plt.show()



#Compare gauge configuration correlation between 1-level and 2-level integrator produced configurations
#TODO: Insufficient, ignores possibility of global gauge transformation that leaves the 
#correlator invariant. Needs reworking
def config_Correlation():
    batch_size= 10
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 16
    L2 = 16
    pm = 2.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 6
    xcut_2 = 14
    bw=2

    s_range= (3,)


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    lvl2_im2 = i.minnorm2(sch.dd_Force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)
    lvl2_sim = h.hmc(sch, lvl2_im2, False)

    #Typical equilibration
    q0 = sim.evolve_f(q, 200)
    q = tuple(q0)


    # Two level integration
    #Level 0 configurations per batch
    n0 = 30

    #Level 1 configurations per batch
    n1 = 10

    #Gauge config data arrays:
    ad0 = tr.zeros(batch_size, n0*n1, L,L)
    ad1 = tr.zeros(batch_size, n0*n1, L, L)

    nd = 0

    for n in np.arange(n0):

        #Thermalize several steps between 2nd level integration
        q = sim.evolve_f(q, 50)
        q1 = tuple(q)

        for nx in np.arange(n1):
        
        #For each subdomain:
            # Evolve locally while maintaining boundaries
            # For quenched model, its actually more efficient to update the whole lattice
            qc = tuple(q)
            q = lvl2_sim.second_Level_Evolve(q, 50, xcut_1, xcut_2, bw)
            #Measure config correlation:
            ad0[:, nd, :, :] = tr.abs(tr.log(q[0][:, 0, :,:]) - tr.log(qc[0][:,0,:,:]))
            ad1[:,nd,:,:] = tr.abs(tr.log(q[0][:, 1, :,:]) - tr.log(qc[0][:,1,:,:]))
            nd += 1


        #Return to configuration before 2nd level integration
        q = tuple(q1)

        print(n)

    #Average over all batches and samples measured
    ad0_avg_2lvl = tr.mean(ad0, dim=(0,1))
    ad1_avg_2lvl = tr.mean(ad1, dim=(0,1))


    #Single level integration
    q=q0
    nm = 300

    #Gauge config data arrays:
    ad0 = tr.zeros(batch_size, nm, L,L)
    ad1 = tr.zeros(batch_size, nm, L, L)

    for n in np.arange(nm):
        q0= tuple(q)
        #Discard some in between
        q= sim.evolve_f(q, 50)
        ad0[:, n, :, :] = tr.abs(tr.log(q[0][:, 0, :,:]) - tr.log(qc[0][:,0,:,:]))
        ad1[:,n,:,:] = tr.abs(tr.log(q[0][:, 1, :,:]) - tr.log(qc[0][:,1,:,:]))
        
        print(n)

    #Average over all batches and samples measured
    ad0_avg_1lvl = tr.mean(ad0, dim=(0,1))
    ad1_avg_1lvl = tr.mean(ad1, dim=(0,1))

    s0 = tr.cat((ad0_avg_2lvl[:xcut_1, :], ad0_avg_2lvl[xcut_1+bw:xcut_2, :]), dim=0)
    s1 = tr.cat((ad1_avg_2lvl[:xcut_1, :], ad1_avg_2lvl[xcut_1+bw:xcut_2, :]), dim=0)
    print("Two Level: ")
    print("Avg: ", tr.mean(s0), tr.mean(s1))
    print("Std: ", tr.std(s0), tr.std(s1))


    print("One Level: ")
    print("Avg: ", tr.mean(ad0_avg_1lvl), tr.mean(ad1_avg_1lvl))
    print("Std: ", tr.std(ad0_avg_1lvl), tr.std(ad1_avg_1lvl))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    axes = ((ax1, ax2), (ax3, ax4))

    im = ax1.imshow(ad0_avg_1lvl, vmin=0.0, vmax = 2.0*np.pi)
    im = ax2.imshow(ad1_avg_1lvl, vmin=0.0, vmax = 2.0*np.pi)
    im = ax3.imshow(ad0_avg_2lvl, vmin=0.0, vmax = 2.0*np.pi)
    im = ax4.imshow(ad1_avg_2lvl, vmin=0.0, vmax = 2.0*np.pi)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()
    
            

def plot_Signal_To_Noise():
    #Match filename
    d = np.loadtxt('output.csv', delimiter=',')
    L= 16

    fig, ax1 = plt.subplots(1,1)

    ax1.set_yscale('log', nonpositive='clip')

    level_2 = np.divide(np.abs(d[1,:]), d[2,:])
    #level_2 = np.abs(d[1,:])
    level_1 = np.divide(np.abs(d[3,:]), d[4,:])
    #level_1 = np.abs(d[3,:])

    ax1.scatter(np.arange(0,L), level_2, s=500, marker='o', c='red', label='DD-HMC')
    ax1.scatter(np.arange(0,L), level_1, s=500, marker='o', c='black', label='HMC')
    #ax1.set_title(r'Signal to noise ratio of a high momentum pion 2 point correlator', fontsize=36)
    ax1.set_xlabel(r'$t$', fontsize=48)
    ax1.set_ylabel(r'$\frac{C(t)}{\mathrm{err}[C(t)]}$', fontsize=48)
    ax1.tick_params(axis='both', which='major', labelsize=24)
    ax1.legend(loc='lower right', fontsize=30)
    plt.show()

#Plot exact/approx correlator difference from imported csv data
def plot_correlator_difference():
    #Match filename
    df = pd.read_csv('correlator approx data.csv')

    fig, ax1 = plt.subplots(1,1)

    ax1.set_yscale('log', nonpositive='clip')

    ax1.scatter(df['timeslice'], df['exact'], label=r'$C(t)$')
    ax1.scatter(df['timeslice'], df['exact err'], label=r'$Err [ C(t) ]$')

    ax1.scatter(df['timeslice approx'], df['approx'], label=r'$\bar{C}(t)$')
    ax1.scatter(df['timeslice approx'], df['approx err'], label=r'$Err [ \bar{C}(t) ]$')

    ax1.scatter(df['timeslice approx'], df['diff'], label=r'$C(t)-\bar{C}(t)$')
    ax1.scatter(df['timeslice approx'], df['err diff'], label=r'$Err [C(t) - \bar{C}(t) ]$')

    ax1.set_xlabel(r'$n_t$', fontsize=32)

    ax1.yaxis.set_tick_params(labelsize=28)
    ax1.xaxis.set_tick_params(labelsize=28)

    ax1.set_title(r'300 Configurations w/ Correlation length $\approx 4.5$', fontsize=32)

    ax1.legend(loc='lower right')
    plt.show()


#Compares autocorrelation time between one-level and two-level integrators
#Uses hadronic correlator, topological charge, or avg. plaquette action
def autocorrelation_Comparison():
    batch_size= 200
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 4
    L2 = 4
    pm = 0.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 7
    xcut_2 = 14
    bw=2

    s_range= (3,)


    u = sch.hotStart()

    #Quenched theory
    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,30, 1.0)
    lvl2_im2 = i.minnorm2(sch.dd_Force,sch.evolveQ,30, 1.0)
    sim = h.hmc(sch, im2, False)
    lvl2_sim = h.hmc(sch, lvl2_im2, False)

    #Typical equilibration
    q0 = sim.evolve_f(q, 200)
    q = tuple(q0)

    #Start with one-level integrator
    nm = 50000
    measures = tr.zeros(batch_size,nm)
    #timeslice of measurement
    ts = 0
    print("One-level measurement:")
    for n in np.arange(nm):
        #Take one step and measure
        q= sim.evolve_f(q, 1)
        #d_inv = tr.inverse(sch.diracOperator(q[0]).to_dense())
            
        #Vector of time slice pion correlator
        #cl = sch.exact_Pion_Correlator(d_inv, s_range, p=p)

        #measures[:,n] = cl[:,ts]

        #Measuring topological charge
        #charge = tr.sum(tr.real(tr.log(q[0])/1.0j) % (2.0*np.pi), dim=(1,2,3))/(2.0*np.pi*L*L2) - 1.0
        #measures[:,n] = charge

        #Measuring avg plaquette action:
        measures[:,n]= sch.gaugeAction(q[0]) / float(L**2)


        if (n % 100 == 0):
            print(n)

    np.savetxt("one_level_autoc_measurements.csv",measures.numpy(), delimiter=',')
    #Compute autocorrelation across the chain
    #For excluding last few autocorrelation measurements due to low statistics
    cutoff = 20000

    #Timeslice of measurement
    ts = 0

    print("Autocorrelation computation:")
    m = tr.mean(measures, dim=1)
    auto_corr = tr.zeros(batch_size, nm-cutoff)
    auto_corr[:,0] = tr.var(measures, dim=1)

    prod = measures * measures
    auto_corr[:,0] = (tr.mean(prod, dim=1) - tr.mean(measures, dim=1)*tr.mean(measures, dim=1)) \
        / tr.sqrt(tr.var(measures, dim=1)* tr.var(measures, dim=1))
    #Computing autocorrelation:
    #Exclude last several measurements due to low statistics
    for t in np.arange(1, nm-cutoff):
        #Product of measurements t steps apart
        shifted = tr.roll(measures, -t, dims=1)
        prod = measures * shifted
        #Autocorrelation, cutting off elements which have
        # rolled from opposite end of the vector
        auto_corr[:,t] = (tr.mean(prod[:,:-t], dim=1) - tr.mean(measures[:,:-t], dim=1)*tr.mean(shifted[:,:-t], dim=1)) \
        / tr.sqrt(tr.var(measures[:,:-t], dim=1)* tr.var(shifted[:,:-t], dim=1))
    fig, ax1 = plt.subplots(1, 1)

    ax1.set_yscale('log', nonpositive='clip')

    # avg_autocorr = tr.mean(tr.abs(auto_corr)/tr.reshape(auto_corr[:,0], (-1, 1)), dim=0)
    # err_autocorr = tr.std(tr.abs(auto_corr)/tr.reshape(auto_corr[:,0], (-1, 1)), dim=0)/np.sqrt(tr.numel(auto_corr[:,0]) - 1)
    avg_autocorr = tr.mean(tr.abs(auto_corr), dim=0)
    err_autocorr = tr.std(tr.abs(auto_corr), dim=0)/np.sqrt(tr.numel(auto_corr[:,0]) - 1)

    ax1.errorbar(np.arange(nm-cutoff), avg_autocorr, err_autocorr, ls="", marker=".")

    #ax1.plot(np.arange(nm-offset), tr.mean(auto_corr, dim=0)/auto_corr[0], label="1 Level")

    ax1.set_title("Autocorrelation comparison")
    ax1.set_ylabel(r"$Corr(X_0, X_t)$")
    ax1.set_xlabel(r"HMC Step $t$")
    ax1.legend(loc='upper right')


    plt.show()

    

    #Two level integrator
    q = tuple(q0)
    twolvl_measures = tr.zeros(batch_size,nm)
    print("Two-level measurement: ")
    for n in np.arange(nm):
        #Take one step
        q= lvl2_sim.second_Level_Evolve(q, 1, xcut_1, xcut_2, bw)
        # s_inv = sch.dd_Schur_Propogator(q, xcut_1,xcut_2, bw)
            
        # # Measure using schur complement
        # cl = sch.dd_Exact_Pion_Correlator(s_inv,xcut_1,xcut_2, bw, s_range,p=p)

        #d_inv = tr.inverse(sch.diracOperator(q[0]).to_dense())
            
        #Vector of time slice correlations
        #cl = sch.exact_Pion_Correlator(d_inv, s_range, p=p)

        #twolvl_measures[n] = cl[0, ts]

        #Measuring topological charge
        #charge = tr.sum(tr.real(tr.log(q[0])/1.0j) % (2.0*np.pi), dim=(1,2,3))/(2.0*np.pi*L*L2) - 1.0
        #twolvl_measures[n] = charge[0]

        #Measuring avg plaquette action:
        twolvl_measures[:,n]= sch.gaugeAction(q[0]) / float(L**2)

        if (n % 100 == 0):
            print(n)

    #Compute autocorrelation across the chain
    print("Autocorrelation computation:")
    m = tr.mean(twolvl_measures, dim=1)
    auto_corr2 = tr.zeros(batch_size, nm-cutoff)
    auto_corr2[:, 0] = tr.var(twolvl_measures, dim=1)

    prod = twolvl_measures * twolvl_measures
    auto_corr2[:,0] = (tr.mean(prod, dim=1) - tr.mean(twolvl_measures, dim=1)*tr.mean(twolvl_measures, dim=1)) \
        / tr.sqrt(tr.var(twolvl_measures, dim=1)* tr.var(twolvl_measures, dim=1))


    #Computing autocorrelation:
    #Exclude last several measurements due to low statistics
    for t in np.arange(1, nm-cutoff):
        #Product of measurements t steps apart
        shifted = tr.roll(twolvl_measures, -t, dims=1)
        prod = twolvl_measures * shifted
        #Autocorrelation, cutting off elements which have
        # rolled from opposite end of the vector
        auto_corr2[:,t] = (tr.mean(prod[:,:-t], dim=1) - tr.mean(twolvl_measures[:,:-t], dim=1)*tr.mean(shifted[:,:-t], dim=1)) \
        / tr.sqrt(tr.var(twolvl_measures[:,:-t], dim=1)* tr.var(shifted[:,:-t], dim=1))


    avg_autocorr2 = tr.mean(tr.abs(auto_corr2), dim=0)
    err_autocorr2 = tr.std(tr.abs(auto_corr2), dim=0)/np.sqrt(tr.numel(auto_corr2[:,0]) - 1)
    
    fig, ax1 = plt.subplots(1, 1)

    ax1.set_yscale('log', nonpositive='clip')

    #TODO: Fix plotting; align with one-level code

    ax1.errorbar(np.arange(nm-cutoff), avg_autocorr, err_autocorr, ls="", marker=".", label="One Level")
    ax1.errorbar(np.arange(nm-cutoff), avg_autocorr2, err_autocorr2, ls="", marker=".", label="Two Level")

    ax1.set_title("Autocorrelation comparison")
    ax1.set_ylabel(r"$\Gamma (X_0, X_t)$")
    ax1.set_xlabel(r"HMC Step $t$")
    ax1.legend(loc='upper right')


    plt.show()


def correlator_Dist():
    #Import the files
    two_lvl = tr.tensor(np.loadtxt("two_level_correlator.csv", delimiter=','))
    one_lvl = tr.tensor(np.loadtxt("one_level_correlator.csv", delimiter=','))

    fig, (ax1,ax2) = plt.subplots(2,1)

    fig.suptitle("Timeslice 5")

    # counts,bin_edges = np.histogram(one_lvl[:, 10],30)
    # bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.

    # ax1.scatter(bin_centres, counts, marker='o')

    # counts,bin_edges = np.histogram(two_lvl[:, 10],30)
    # bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.

    # ax2.scatter(bin_centres, counts, marker='o')

    ax1.hist(one_lvl[:, 5], bins=100)
    ax2.hist(two_lvl[:, 5],bins=100)

    ax1.set_title("One Level")
    ax2.set_title("Two Level")
    ax1.set_xlim(-0.08, 0.08)
    ax2.set_xlim(-0.08, 0.08)
    

    plt.show()

#Takes square matrix and produces a k-rank approximation of its inverse based on eigenvalue
#Decomposition
#Input: batch of square pytorch tensors, rank of approx
# k = -1 input corresponds to exact inverse
def eig_Inv_Approx(m, k):

    if k == -1:
        return tr.inverse(m)
    
    #We anticipate that the given matrix will be g5 hermitian:
    id = tr.eye(int(m.size(dim=1)/2))
    id = id.reshape((1, int(m.size(dim=1)/2), int(m.size(dim=1)/2)))
    id = id.repeat(m.size(dim=0), 1, 1)

    g5 = tr.tensor([[0, -1.0j], [1.0j, 0]])

    g5m = tr.einsum('bxy, byz->bxz', tr.kron(id, g5), m)

    L,V = tr.linalg.eig(g5m)
    #print(V)
    
    inv_l = 1.0/L

    A = tr.diag_embed(inv_l, dim1=1, dim2=2)

    sorted_L, sorted_ind = tr.sort(tr.abs(inv_l), descending=True)

    mask = tr.zeros_like(inv_l, dtype=tr.bool)
    for b in np.arange(inv_l.size(dim=0)):
        mask[b, :] =  tr.abs(inv_l[b,:]) >= sorted_L[b, k-1]
    
    #expand the mask to encapsulate full columns
    exp_mask = tr.zeros([mask.size(0), mask.size(1), mask.size(1)], dtype=tr.bool)
    for b in np.arange(mask.size(0)):
        for x in np.arange(mask.size(1)):
            exp_mask[b,:,x] = mask[b,x]

    #mask = tr.abs(inv_l) >= sorted_L[:, k-1]

    filtered_A = tr.where(exp_mask, A, tr.tensor(0))

    #print(V)

    #print(V_T)
    filtered_V = tr.where(exp_mask, V, tr.tensor(0))
    # return tr.einsum("bij, bjk, bkl -> bil", filtered_V,
    #                 filtered_A, tr.transpose(filtered_V, 1,2))

    #No filtering added- should match exact inverse exactly
    return tr.einsum("bij, bjk, bkl, blm -> bim", filtered_V,
                    filtered_A, tr.conj(tr.transpose(filtered_V, 1,2)), tr.kron(id, g5))


def test_inv_approx():
    batch_size=1
    lam=1/10.0
    mass= -0.08*lam
    L = 2
    L2 = 2
    pm = 0.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 7
    xcut_2 = 14
    bw=2

    s_range= (3,)


    u = sch.hotStart()

    d=sch.diracOperator(u).to_dense()

    #Full eigenvalue decomp reproduces taking the inverse directly
    print(tr.abs(tr.einsum("bxy, byz->bxz", d, eig_Inv_Approx(d, 8)) - 
          tr.einsum("bxy, byz->bxz", d, eig_Inv_Approx(d, -1))) <0.000001)


def factorized_Observable_Systematic_Error_Test():
    batch_size= 30
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 16
    L2 = 16
    pm = 2.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 6
    xcut_2 = 14
    bw=2

    s_range= (3,)


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    #Typical equilibration
    q0 = sim.evolve_f(q, 200)
    q = tuple(q0)

    nm = 20
    c= tr.zeros(batch_size, nm, L)
    err = tr.zeros(batch_size, nm, L)
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 50)

        bb_d = sch.bb_DiracOperator(q, xcut_1, xcut_2, bw).to_dense()
        d00 = bb_d[:,0:4*bw*L, 0:4*bw*L2]

        #Approximate the inverse
        #For initial testing, use the exact inverse
        #It comes out to a difference around machine error
        #d00_inv = tr.inverse(d00)

        #Lets try an actual approximation to the inverse
        d00_inv = eig_Inv_Approx(d00, 36)

        err[:, n, :] = sch.dd_Pion_Systematic_Error(q, xcut_1, xcut_2, bw, d00_inv, s_range, p)

        d_inv = tr.inverse(sch.diracOperator(q[0]).to_dense())

        #Example of an error reading
        print(err[0,n, :])
            
        #Vector of time slice correlations
        cl = sch.exact_Pion_Correlator(d_inv, s_range, p=p)

        c[:, n, :] = cl

        print(n)

        #Compute a systematic error to apply to the DD measurement per timeslice
        avg_sys = tr.mean(err, dim=(0,1)) #Empty?!
        err_sys = tr.std(err, dim=(0,1)) /np.sqrt(tr.numel(err[:,:,0]) - 1) # length 16


    #Print the error to examine size when all measurements are completed
    print("Computed systematic error: ",avg_sys, err_sys)


#TODO: In Development- needs error correction
#2-lvl factorized observable with multilevel integration
def quenched_Global_Twolvl_Observable():
    batch_size= 30
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 16
    L2 = 16
    pm = 2.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 6
    xcut_2 = 14
    bw=2

    s_range= (3,)


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    lvl2_im2 = i.minnorm2(sch.dd_Force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)
    lvl2_sim = h.hmc(sch, lvl2_im2, False)

    #Typical equilibration
    q0 = sim.evolve_f(q, 200)
    q = tuple(q0)


    # Two level integration
    #Level 0 configurations per batch
    n0 = 20

    #Level 1 configurations per batch
    n1 = 10

    #Ensemble averaged correlation function data array:
    c= tr.zeros(batch_size, n0*n1*n1, L)
    #index for adding correlator measurement to sample
    cx = 0
    #For measuring systematic error of approximate propogator
    err = tr.zeros(batch_size, n0, L)


    for n in np.arange(n0):

        #Thermalize several steps between 2nd level integration
        q = sim.evolve_f(q, 50)
        q1 = tuple(q)

        bb_d = sch.bb_DiracOperator(q1, xcut_1, xcut_2, bw).to_dense()

        d00 = bb_d[:,0:4*bw*L, 0:4*bw*L2]

        #Approximate the inverse
        #d00_inv = eig_Inv_Approx(d00, 36)
        d00_inv=tr.inverse(d00)

        #Compute systematic error
        err[:, n, :] = sch.dd_Pion_Systematic_Error(q, xcut_1, xcut_2, bw, d00_inv, s_range, p)

        #To store 2nd-level gauge configs for ensemble averaging
        #batch, config no., lattice indexes
        u_lvl2 = tr.zeros([batch_size, n1, 2, L, L2], dtype=tr.complex64)
        for nx in np.arange(n1):
            #For each subdomain:
            # Evolve locally while maintaining boundaries
            # For quenched model, its actually more efficient to update the whole lattice
            #Save gauge config
            q = lvl2_sim.second_Level_Evolve(q, 50, xcut_1, xcut_2, bw)
            u_lvl2[:, nx, :, :, :] = q[0]

        #Now ensemble average each combo of level 2 subdomains
        u_measure = q[0].clone().detach()
        for x in np.arange(n1):
            u_measure[:, :, 0:xcut_1, :] = u_lvl2[:,x,:, 0:xcut_1,:]
            for y in np.arange(n1):
                u_measure[:, :, xcut_1+bw:xcut_2, :] = u_lvl2[:,y,:,xcut_1+bw:xcut_2,:]

                #Measuring each spliced together config
                q = (u_measure,)
                s_inv = sch.dd_Schur_Propogator(q, xcut_1,xcut_2, bw, d00_inv)
                cl = sch.dd_Exact_Pion_Correlator(s_inv,xcut_1,xcut_2, bw, s_range,p=p)
            
                #Vector of time slice correlations
                c[:, cx, :] = cl
                cx +=1

        #Return to configuration before 2nd level integration
        q = tuple(q1)
        print(n)

    #Save produced correlator data for one batch
    c_np = c[0,:,:].numpy()
    np.savetxt("two_level_correlator.csv",c_np, delimiter=',')
    #Average over all batches and configurations measured
    c_avg = tr.mean(c, dim=(0,1))
    c_err = (tr.std(c, dim=(0,1))/np.sqrt(tr.numel(c[:,:,0]) - 1))

    #Save systematic error and adjust measurement
    #Compute a systematic error to apply to the DD measurement per timeslice
    avg_sys = tr.mean(err, dim=(0,1))
    err_sys = tr.std(err, dim=(0,1)) /np.sqrt(tr.numel(err[:,:,0]) - 1)
    correction = pd.DataFrame([avg_sys, err_sys])
    np.savetxt("two_level_correlator_systematic_error", correction.to_numpy(), delimiter=',')

    adj_c_avg = tr.mean(c_avg + avg_sys)
    adj_c_err = tr.sqrt(tr.square(c_err) + tr.square(err_sys))
    print(adj_c_avg.size())
    print(adj_c_err.size())


    #Single level integration
    q=q0
    nm = 200
    c2= tr.zeros(batch_size, nm, L)
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 50)
        d_inv = tr.inverse(sch.diracOperator(q[0]).to_dense())
            
        #Vector of time slice correlations
        cl = sch.exact_Pion_Correlator(d_inv, s_range, p=p)

        c2[:, n, :] = cl

        print(n)


    #Save produced correlator data for one batch
    c_np = c2[0,:,:].numpy()
    np.savetxt("one_level_correlator.csv",c_np, delimiter=',')
    #Average over all batches and configurations measured
    c2_avg = tr.mean(c2, dim=(0,1))
    c2_err = (tr.std(c2, dim=(0,1))/np.sqrt(tr.numel(c2[:,:,0]) - 1))

    #Write dataframe of data
    df = pd.DataFrame([c_avg.detach().numpy(), c_err.detach().numpy(), c2_avg.detach().numpy(), c2_err.detach().numpy()])
    #df = pd.DataFrame([adj_c_avg.detach().numpy(), adj_c_err.detach().numpy(), c2_avg.detach().numpy(), c2_err.detach().numpy()])
    #TODO: Write more descriptive datafile name
    df.to_csv("output.csv", index = False)

    a = df.to_numpy()


    #Fit data 
    popt_2l, pcov_2l = sp.optimize.curve_fit(f_pi_triplet, np.arange(8,11), np.abs(a[0, 8:11]), sigma = np.abs(a[1, 8:11]))
    print("Two level:")
    print(popt_2l)
    print(pcov_2l)

    popt_1l, pcov_1l = sp.optimize.curve_fit(f_pi_triplet, np.arange(8,11), np.abs(a[2, 8:11]), sigma = np.abs(a[3, 8:11]))
    print("One level:")
    print(popt_1l)
    print(pcov_1l)

    fig, (ax1, ax2) = plt.subplots(2,1)

    ax1.set_yscale('log', nonpositive='clip')
    ax2.set_yscale('log', nonpositive='clip')

    ax1.plot(np.linspace(0, L, 100), f_pi_triplet(np.linspace(0, L, 100), *popt_2l))
    ax2.plot(np.linspace(0, L, 100), f_pi_triplet(np.linspace(0, L, 100), *popt_1l))

    ax1.errorbar(np.arange(0, L), tr.abs(c_avg), tr.abs(c_err), ls="", marker=".", label=r'2-level')
    ax2.errorbar(np.arange(0, L), tr.abs(c2_avg), tr.abs(c2_err), ls="", marker=".", label=r'1-level')
    ax1.set_title(r'p= '+str(pm) + r'$\times \frac{2\pi}{L}$ pion correlator')

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

    plt.show()

def test_Factorized_Propogator():
    batch_size= 10
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 32
    L2 = 16
    pm = 2.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 11
    xcut_2 = 27
    bw=5

    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    #Typical equilibration
    q0 = sim.evolve_f(q, 200)
    q = tuple(q0)

    #Level 0 configurations tested on
    n0 = 10

    #set of intermediate points for ensemble avg.

    int1 = tr.arange((xcut_1)*L2+1, (xcut_1+bw)*L2)
    
    int2 = tr.arange((xcut_2)*L2+1, (xcut_2+bw)*L2)

    #nter = tr.tensor((base_1, base_1+4, base_1+6, base_1+12,base_2, base_2+4, base_2+6, base_2+12))
    inter = tr.cat([int1, int2])
    
    source_x = 5*L2 + 8
    sink_x = (xcut_1+5+bw)*L2 + 8

    #Ensemble averaged correlation function data array:
    c= tr.zeros(batch_size, n0)
    c2= tr.zeros(batch_size, n0)
    for n in np.arange(n0):

        #Thermalize several steps between 2nd level integration
        q = sim.evolve_f(q, 50)
        q1 = tuple(q)
        factorized_L, factorized_R = sch.factorized_Propogator(q, xcut_1, xcut_2, bw, inter, True)

        #left_std = tr.std(ensemble_L, dim=1)
        #right_std = tr.std(ensemble_R, dim=1)


        #Kronecker product of dirac entries by intermediate

        left_avg = factorized_L.view(batch_size, tr.numel(inter), (xcut_1+2*bw)*L2, 2,2)
        right_avg = factorized_R.view(batch_size, tr.numel(inter), (xcut_1+2*bw)*L2, 2,2)

        combined = tr.einsum('bixag, biygd-> bixyad', left_avg, right_avg)

        #Accounts for shifting of the lattice indices in the factorizing process
        sink_adj = (xcut_1)*L2
        source_adj = bw*L2
        for x in tr.arange(tr.numel(inter)):
            p1 = combined[:, x, sink_x-sink_adj, source_x+source_adj, :, :]
            for y in tr.arange(tr.numel(inter)):
                p2 = combined[:, y, sink_x-sink_adj, source_x+source_adj, :, :]
                c[:, n] =  c[:, n] + tr.sum(tr.einsum("bij, bkj->bik", p1, p2.conj()), dim=(1,2))


        # propogator= prop_matrix[:, sink_x-sink_adj, source_x+source_adj, :, :]

        # c[:, n] = tr.sum(tr.einsum("bij, bkj->bik", propogator, propogator.conj()), dim=(1,2))

        d_inv = tr.inverse(sch.diracOperator(q[0]).to_dense())
        propogator = d_inv[:, 2*sink_x:2*sink_x+2, 2*source_x:2*source_x+2]
            

        c2[:, n] = tr.sum(tr.einsum("bij, bkj->bik", propogator, propogator.conj()), dim=(1,2))
        
        print(n)
    
    print("Factorized: ",tr.mean(c), tr.std(c)/np.sqrt(tr.numel(c)-1))
    print("Full: ", tr.mean(c2), tr.std(c2)/np.sqrt(tr.numel(c2)-1))


def factorized_Pion_Comparison():
    batch_size= 30
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 32
    L2 = 16
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 11
    xcut_2 = 27
    bw=5

    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    #Typical equilibration
    q0 = sim.evolve_f(q, 200)
    q = tuple(q0)


    # Two level integration
    #Level 0 configurations tested on
    n0 = 10

    #set of intermediate points for ensemble avg.

    #Intermediate based on overlap or not
    overlap=True
    if overlap:
        #int1 = tr.arange((xcut_1)*L2+1, (xcut_1+bw)*L2)
        #int2 = tr.arange((xcut_2)*L2+1, (xcut_2+bw)*L2)
        int1 = tr.arange((xcut_1+2)*L2, (xcut_1+3)*L2)
        int2 = tr.arange((xcut_2+2)*L2, (xcut_2+3)*L2)
    else:
        int1 = tr.arange((xcut_1)*L2, (xcut_1+1)*L2)
        int2 = tr.arange((L-1)*L2, L*L2)
    inter = tr.cat([int1, int2])
    
    source_x = 5*L2 + 8

    #Ensemble averaged correlation function data array:
    c= tr.zeros(batch_size, n0, xcut_2-xcut_1 - bw)
    c2= tr.zeros(batch_size, n0, xcut_2-xcut_1-bw)
    for n in np.arange(n0):
        #Thermalize before measuring
        q = sim.evolve_f(q, 50)
        factorized_L, factorized_R = sch.factorized_Propogator(q, xcut_1, xcut_2, bw, inter, overlap)


        #Kronecker product of dirac entries by intermediate

        #left_avg = factorized_L.view(batch_size, tr.numel(inter), (xcut_1+2*bw)*L2, 2,2)
        left_avg = factorized_L
        #right_avg = factorized_R.view(batch_size, tr.numel(inter), (xcut_1+2*bw)*L2, 2,2)
        right_avg = factorized_R

        combined = tr.einsum('bixag, biygd-> bixyad', left_avg, right_avg)
        
        if overlap == True:
            source_adj = bw*L2
        else:
            source_adj= L2

        sink_adj = (xcut_1)*L2

        for nt in np.arange(xcut_1 + bw, xcut_2):
            #Just one point in spatial dimension since full contraction is expensive
            for nx in np.arange(8,9):
                #Factorized propogator measurment
                sink = nt*L2 + nx - sink_adj
                for x in tr.arange(tr.numel(inter)):
                    p1 = combined[:, x, sink, source_x+source_adj, :, :]
                    for y in tr.arange(tr.numel(inter)):
                        p2 = combined[:, y, sink, source_x+source_adj, :, :]
                        c[:, n, nt-xcut_1 -bw] =  c[:, n, nt-xcut_1 -bw] + tr.sum(tr.einsum("bij, bkj->bik", p1, p2.conj()), dim=(1,2))

                d_inv = tr.inverse(sch.diracOperator(q[0]).to_dense())

                #Full propogator
                sink = sink + sink_adj
                propogator = d_inv[:, 2*sink:2*sink+2, 2*source_x:2*source_x+2]
                c2[:, n, nt-xcut_1-bw] = tr.sum(tr.einsum("bij, bkj->bik", propogator, propogator.conj()), dim=(1,2))
        print(n)

    factorized_avg = tr.mean(c, dim=(0,1))
    factorized_err = tr.std(c, dim=(0,1))/np.sqrt(tr.numel(c[:,:,0]))

    full_avg = tr.mean(c2, dim=(0,1))
    full_err = tr.std(c2, dim=(0,1))/np.sqrt(tr.numel(c2[:,:,0]))

    bias = tr.abs(factorized_avg - full_avg)
    bias_err = tr.std(tr.abs(c-c2), dim=(0,1))/np.sqrt(tr.numel(c[:,:,0]))

    correlation = tr.mean(c*c2, dim=(0,1)) - tr.mean(c, dim=(0,1))*tr.mean(c2, dim=(0,1))

    correlation = correlation /(tr.std(c, dim=(0,1)) * tr.std(c2, dim=(0,1)))

    correlation_err = tr.sqrt((1-tr.square(correlation))/(tr.numel(c[:,:,0]) - 2))

    fig=plt.figure()
    
    gs = plt.GridSpec(3,1, figure= fig)

    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2,0])

    ax1.errorbar(tr.arange(0, tr.numel(factorized_avg)) +xcut_1+bw, factorized_avg, factorized_err,
                marker= '.', ms=12, label='Factorized')
    ax1.errorbar(tr.arange(0,tr.numel(full_avg))+xcut_1+bw, full_avg, full_err,
                marker='.', ms=12, label='Full')
    ax1.errorbar(tr.arange(tr.numel(bias))+xcut_1+bw, bias, bias_err,
                marker= '.', ms=12, label="Bias")
    
    ax1.plot(tr.arange(0, tr.numel(factorized_avg)) +xcut_1+bw, factorized_err,
                marker= '.', ms=12, label='Factorized Err.')
    ax1.plot(tr.arange(0,tr.numel(full_avg))+xcut_1+bw, full_err,
                marker='.', ms=12, label='Full Prop. Err.')
    ax1.plot(tr.arange(tr.numel(bias))+xcut_1+bw, bias_err,
                marker= '.', ms=12, label="Bias Err.")

    ax2.errorbar(tr.arange(tr.numel(bias))+xcut_1+bw, tr.abs(correlation), correlation_err,
                  marker='.', lw=2.0, ms=12)

    ax1.set_yscale('log', nonpositive='clip')
    ax2.set_yscale('log', nonpositive='clip')

    ax1.legend(loc='lower right')
    if overlap==True:
        ax1.set_title('Thick overlap boundaries, n='+ str(batch_size*n0), fontsize=30)
    else:
        ax1.set_title('Thin boundaries, n='+ str(batch_size*n0), fontsize=30)

    ax1.set_ylabel('Magnitude',fontsize=20)
    #ax1.set_xlabel("Timeslice in Subdomain 2", fontsize=30)


    ax2.set_ylabel("Correlation",
                   fontsize=20)

    ax2.set_xlabel("Timeslice in Subdomain 2", fontsize=30)

    #Save correlator data
    factorized = tr.stack((tr.arange(0,tr.numel(full_avg))+xcut_1+bw,
                           factorized_avg, factorized_err, full_avg,
                           full_err, bias, bias_err, correlation, correlation_err), dim=0).numpy()
    np.savetxt('factorized_data.csv', factorized, delimiter=',')


    plt.show()
        

#Testing that the projector based factorization approach reproduces
#results of the naive indexing approach 
def compare_Factorizations():
    batch_size= 30
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.07*lam
    L = 32
    L2 = 16
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 11
    xcut_2 = 27
    bw=5

    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    #Typical equilibration
    q0 = sim.evolve_f(q, 200)
    q = tuple(q0)

    #Intermediate based on overlap or not
    #Use only a single dirac index and lattice index combo for intermediate
    overlap=False
    if overlap:
        int1 = tr.arange((xcut_1+2)*L2*2+2, (xcut_1+3)*2*L2)
        int2 = tr.arange((xcut_2+2)*L2*2+2, (xcut_2+3)*2*L2)
    else:
        int1 = tr.arange((xcut_1)*L2*2, (xcut_1+1)*L2*2)
        int2 = tr.arange((L-1)*L2*2, L*L2*2)
    inter = tr.cat([int1, int2])

    
    source_x = 5*L2 + 8
    sink_x = (xcut_1+5+bw)*L2 + 8

 
    factorized_L, factorized_R, summed = sch.factorized_Propogator(q, xcut_1, xcut_2, bw, inter, overlap)
    # print(factorized_L.size())
    # print(factorized_R.size())

    #Accounts for shifting of the lattice indices in the factorizing process
    sink_adj = (xcut_1)*L2
    if overlap == True:
        source_adj = bw*L2
    else:
        source_adj = L2
    

    s_prop = summed[:, sink_x-sink_adj, source_x+source_adj, :,:]
    corr1 = tr.sum(tr.einsum("bij, bkj->bik", s_prop, s_prop.conj()), dim=(1,2))

    #Now try the projection approach

    #Generate projectors
    projs = tr.zeros((tr.numel(inter)), 2*L*L2, dtype=tr.complex64)
    ct=0
    if overlap:
        for x in tr.arange((xcut_1+2)*L2*2+2, (xcut_1+3)*L2*2):
            projs[ct, x] = 1.0
            ct += 1
        for x in tr.arange((xcut_2+2)*L2*2+2, (xcut_2+3)*L2*2):
            projs[ct, x] = 1.0
            ct += 1
    else:
        for x in tr.arange((xcut_1)*L2*2, (xcut_1+1)*L2*2):
            projs[ct, x] = 1.0
            ct += 1
        for x in tr.arange((L-1)*L2*2, (L)*L2*2):
            projs[ct, x] = 1.0
            ct += 1

    

    factorized_L2, factorized_R2, summed2 = sch.factorized_Propogator_Proj(q, xcut_1, xcut_2,
                                                                  bw, projs, overlap)
    

    # print(factorized_L2.size())
    # print(factorized_R2.size())


    #Accounts for shifting of the lattice indices in the factorizing process
    sink_adj = 2*(xcut_1)*L2
    if overlap == True:
        source_adj = 2*bw*L2
    else:
        source_adj = 2*L2

    prop = summed2[:, 2*sink_x-sink_adj:2*sink_x-sink_adj+2, 
            2*source_x+source_adj:2*source_x+source_adj + 2]
    
    corr2 = tr.sum(tr.einsum("bij, bkj->bik", prop, prop.conj()), dim=(1,2))

    d_inv = tr.inverse(sch.diracOperator(q[0]).to_dense())
    propogator = d_inv[:, 2*sink_x:2*sink_x+2, 2*source_x:2*source_x+2]
    correct = tr.sum(tr.einsum("bij, bkj->bik", propogator, propogator.conj()), dim=(1,2))
    
    
    print("Index result: ", tr.mean(corr1), tr.std(corr1))
    print("Projector Result: ", tr.mean(corr2), tr.std(corr2))
    print("Full Propogator: ", tr.mean(correct), tr.std(correct))



def plot_Factorized_Correlator_Data():
    data = np.loadtxt('factorized_data.csv',delimiter=',')
    n_configs=300

    fig = plt.figure()
    
    #fig, (ax1, ax2, ax3) = plt.subplots(3,1)

    gs = plt.GridSpec(3,1, figure= fig)

    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2,0])

    ax1.errorbar(data[0,:], data[1,:], data[2,:],
                marker= '.', ms=12, label='factorized')
    ax1.errorbar(data[0,:], data[3,:], data[4,:],
                marker='.', ms=12, label='Full Prop.')
    ax1.errorbar(data[0,:], data[5,:], data[6,:],
                marker= '.', ms=12, label="Bias")
    
    ax1.plot(data[0,:], data[2,:],
                marker= '.', ms=12, label='Factorized Err.')
    ax1.plot(data[0,:], data[4,:],
                marker='.', ms=12, label='Full Prop. Err.')
    ax1.plot(data[0,:], data[6,:],
                marker= '.', ms=12, label="Bias Err.")

    ax2.errorbar(data[0,:], tr.abs(data[7,:]), data[8,:], marker='.', lw=2.0, ms=12)

    ax1.set_yscale('log', nonpositive='clip')
    ax2.set_yscale('log', nonpositive='clip')

    ax1.legend(loc='lower right')
    ax1.set_title('Thin boundaries, n='+ str(300), fontsize=30)
    # if overlap==True:
    #     ax1.set_title('Thick overlap boundaries, n='+ str(batch_size*n0), fontsize=30)
    # else:
    #     ax1.set_title('Thin boundaries, n='+ str(batch_size*n0), fontsize=30)

    ax1.set_ylabel('Magnitude',fontsize=20)
    #ax1.set_xlabel("Timeslice in Subdomain 2", fontsize=30)

    #ax2.legend(loc='lower right')

    #ax2.set_ylabel("Error",
    #               fontsize=20)

    ax2.set_ylabel("Correlation",
                   fontsize=20)
    ax2.set_xlabel("Timeslice in Subdomain 2", fontsize=30)

    plt.show()

#Jacknife for pearson correlation since the correlation is unstable
#run to run
def jackknife_Correlation(x, y):

    N = x.shape[0]

    # Full-sample correlation
    cov = tr.mean(tr.real(x)*tr.real(y)) - tr.mean(tr.real(x))*tr.mean(tr.real(y))
    full_correlation = cov / (tr.std(tr.real(x))*tr.std(tr.real(y)))

    # Leave-one-out correlations
    r_jack = tr.empty(N, dtype=x.dtype, device=x.device)

    for i in range(N):
        mask = tr.ones(N, dtype=tr.bool)
        mask[i] = False
        cov = tr.mean(tr.real(x[mask])*tr.real(y[mask])) - tr.mean(tr.real(x[mask]))*tr.mean(tr.real(y[mask]))
        r_jack[i] = cov / (tr.std(tr.real(x[mask]))*tr.std(tr.real(y[mask])))

    # Jackknife mean
    r_jack_mean = r_jack.mean()

    # Jackknife standard error
    r_jack_se = tr.sqrt((N - 1) * tr.mean((r_jack - r_jack_mean) ** 2))

    return full_correlation, r_jack_mean, r_jack_se

#Tests measurement bias of the true/factorized 2 pt correlator measurement
def test_Factorized_Measurement_Scan():
    batch_size= 50
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.10*lam
    L = 32
    L2 = 16
    p_n = 0.0
    p = p_n*2*np.pi/L2
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 15
    xcut_2 = 31
    bw=1

    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    #Typical equilibration
    q = sim.evolve_f(q, 200)

    #Generate projectors
    #TODO: Needs work below
    #Lets try some different bases
    ov=1
    if False:
        title1 = 'Full boundary probe, n='+ str(batch_size) +', ' r'$m_\pi L = 5$'
        title2 = 'Full boundary 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        #Isolates the relevant spin index by boundary
        mult = 1.0
        projs = tr.zeros(2*L2, 2*L*L2, dtype=tr.complex64)
        ct=0
        #Only need second spin index on the first boundary
        for x in tr.arange(1, (1)*L2*2, 2):
                projs[ct, x] = 1.0
                ct += 1
        #Only need first spin index on the second boundary
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1)*L2*2, 2):
                projs[ct, x] = 1.0
                ct += 1

    #Try introducing an overlap
    if False:
        ov=2
        title1 = 'Full boundary probe, n='+ str(batch_size) +', ' r'$m_\pi L = 5$'
        title2 = 'Full boundary 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        mult = 1.0
        projs = tr.zeros(8*L2, 2*L*L2, dtype=tr.complex64)
        ct=0
        #Need points on both ends for the first boundary
        for x in tr.arange(0, (1)*L2*2):
                projs[ct, x] = 1.0
                ct += 1
        for x in tr.arange((L-1)*L2*2, L*L2*2):
                projs[ct, x] = 1.0
                ct += 1
     
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1+1)*L2*2):
                projs[ct, x] = 1.0
                ct += 1

    elif True:
        #Probing half the boundary
        #Need to include a multiplier for using fewer intermediates
        title1 = 'Half boundary probe, n='+ str(batch_size) +', ' r'$m_\pi L = 5$'
        title2 = 'Half boundary 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        mult = 2.0
        projs = tr.zeros(L2, 2*L*L2, dtype=tr.complex64)
        ct=0
        #Only need second spin index on the first boundary
        for x in tr.arange(1, (1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1
        #Only need first spin index on the second boundary
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1

    #Try introducing an overlap
    if False:
        ov=2
        title1 = 'Half boundary probe, n='+ str(batch_size) +', ' r'$m_\pi L = 5$'
        title2 = 'Half boundary 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        mult = 2.0
        projs = tr.zeros(4*L2, 2*L*L2, dtype=tr.complex64)
        ct=0
        #Need points on both ends for the first boundary
        for x in tr.arange(2, (1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1
                projs[ct, x+1] = 1.0
                ct +=1
        for x in tr.arange((L-1)*L2*2, L*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1
                projs[ct, x+1] = 1.0
                ct +=1
     
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1+1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1
                projs[ct, x+1] = 1.0
                ct +=1

    elif False:
        #Probing one quarter of the boundary
        #Need to include a multiplier for using fewer intermediates
        mult = 4.0
        projs = tr.zeros(int(L2/2), 2*L*L2, dtype=tr.complex64)
        ct=0
        #Only need second spin index on the first boundary
        for x in tr.arange(1, (1)*L2*2, 8):
                projs[ct, x] = 1.0
                ct += 1
        #Only need first spin index on the second boundary
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1)*L2*2, 8):
                projs[ct, x] = 1.0
                ct += 1

    elif False:
        #Deflation technique
        #Seek eigenmodes of the complement Dirac operator
        title1 = 'Deflation 2-pt probe, n='+ str(batch_size) +', ' r'$m_\pi L = 5$'
        title2 = 'Deflation probe 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        mult=1.0
        ov=3
        N_vec = 16
        projs = sch.complement_Deflation_Eigenvectors(q, xcut_1, N_vec, ov=ov)
    
    elif False:
        #Distillation technique
        #Seek distillation eigenvectors for a given batch of configurations
        title1 = 'Laplacian 2-pt probe, n='+ str(batch_size) +', ' r'$m_\pi L = 5$'
        title2 = 'Laplacian 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        mult=1.0
        N_vec = 32
        ov=3
        projs = sch.boundary_Distillation_Eigenvectors(q, xcut_1, N_vec, ov=ov)

        

    
    ensemble_l, ensemble_r, f_propogator = sch.factorized_Propogator_Proj(q, xcut_1, xcut_2,
                                                                          projs, ov)

    factorized_corr, corr = sch.measure_Factorized_Two_Point_Correlator(q, f_propogator, xcut_1, xcut_2, 
                                                                        bw, p=p, ov=ov)
    
    

    print(len(factorized_corr))
    c_avg = tr.zeros(len(factorized_corr))
    fc_avg = tr.zeros(len(factorized_corr))
    c_err = tr.zeros(len(factorized_corr))
    fc_err = tr.zeros(len(factorized_corr))

    bias = tr.zeros(len(factorized_corr))
    bias_err = tr.zeros(len(factorized_corr))

    correlation = tr.zeros(len(factorized_corr))
    correlation_err = tr.zeros(len(factorized_corr))

    #Scaling correlator by the correlator at a particular timeslice
    scaling = False
    if scaling == True:
        scale = tr.real(tr.mean(corr[7]))
        f_scale = mult*tr.real(tr.mean(factorized_corr[7]))

    for x in tr.arange(bw, len(factorized_corr)):
        if scaling == True:
            corr[x] = corr[x]/scale
            factorized_corr[x] = mult* factorized_corr[x]/f_scale
        else:
            factorized_corr[x] = mult *factorized_corr[x]
        c_corr_avg = tr.real(tr.mean(corr[x], dim=0))
        fc_corr_avg = tr.real(tr.mean(factorized_corr[x], dim=0))
        c_avg[x] = tr.mean(c_corr_avg)
        c_err[x] = tr.std(c_corr_avg)/np.sqrt(tr.numel(c_corr_avg)-1)
        fc_avg[x] = tr.real(tr.mean(fc_corr_avg))
        fc_err[x] = tr.std(tr.real(fc_corr_avg))/np.sqrt(tr.numel(fc_corr_avg)-1)
        bias[x] = c_avg[x] - fc_avg[x]
        bias_err[x] = tr.std(c_corr_avg-fc_corr_avg)/ np.sqrt(tr.numel(c_corr_avg)-1)
        #cov = tr.mean(tr.real(corr[x])*tr.real(factorized_corr[x])) - tr.mean(tr.real(corr[x]))*tr.mean(tr.real(factorized_corr[x]))
        # correlation[x] = cov / (tr.std(tr.real(corr[x]))*tr.std(tr.real(factorized_corr[x])))
        # correlation_err[x] = (1-tr.square(correlation[x]))/np.sqrt(tr.numel(corr[x]) - 3)
        #fcr, correlation[x], correlation_err[x] = jackknife_Correlation(corr[x], factorized_corr[x])
        fcr, correlation[x], correlation_err[x] = jackknife_Correlation(c_corr_avg, fc_corr_avg)


    #Save data
    data_write = tr.stack((tr.arange(1, int(L/2)+1), c_avg, c_err,
                           fc_avg, fc_err, bias, bias_err,
                            correlation, correlation_err), dim=0).numpy()
    np.savetxt('factorized_data.csv', data_write, delimiter=',')
    

    fig, (ax, ax2) = plt.subplots(2,1, sharex=True, constrained_layout=True)
    ax.set_yscale('log', nonpositive='clip')
    ax2.set_yscale('log', nonpositive='clip')


    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(fc_avg[bw:]), fc_err[bw:], label="Factorized signal",
                fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)
    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(c_avg[bw:]), c_err[bw:], label="True signal",
                fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)
    ax.plot(tr.arange(bw+1, int(L/2)+1), fc_err[bw:], label="Factorized Error")
    ax.plot(tr.arange(bw+1, int(L/2)+1), c_err[bw:], label="True Error")

    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(bias[bw:]), bias_err[bw:], label="Bias",
                fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)
    ax.plot(tr.arange(bw+1, int(L/2)+1), bias_err[bw:], label="Bias Error")

    ax.legend(loc='lower right',bbox_to_anchor=(1.02, 1),borderaxespad=0)
    ax.grid(which='major', linestyle='--', alpha=0.6)
    ax.set_title(title1)
    ax.set_ylabel('Magnitude')
    #ax.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)

    ax2.errorbar(tr.arange(bw+1, int(L/2)+1), correlation[bw:], correlation_err[bw:], 
                 fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)
    ax2.set_ylabel('Correlation')
    ax2.set_ylim(0,1)
    #ax2.grid(True, axis='y', ls='--')
    ax2.grid(which='major', linestyle='--', alpha=0.6)
    ax2.grid(which='minor', linestyle=':', alpha=0.6)
    ax2.set_xlabel(r'$|x_0 - y_0|$')

    plt.savefig(
    "figure.pdf",
    bbox_inches="tight"
    )


    #I'm gonna change up the StN measurment
    #Keep the number of samples computed the same
    measurements = tr.numel(corr[bw+1])
    for x in tr.arange(bw+1, len(factorized_corr)):
        bias[x] = tr.mean((tr.real(corr[x])-tr.real(factorized_corr[x])))
        bias_err[x] = tr.std(tr.real(corr[x])-tr.real(factorized_corr[x]))/ np.sqrt(tr.numel(corr[x])-1)
        # c_avg[x] = tr.real(tr.mean(corr[x][:measurements]))
        # c_err[x] = tr.std(tr.real(corr[x][:measurements]))/np.sqrt(measurements-1)
        # fc_avg[x] = tr.real(tr.mean(factorized_corr[x][:measurements]))
        # fc_err[x] = tr.std(tr.real(factorized_corr[x][:measurements]))/np.sqrt(measurements-1)


    fig2, ax = plt.subplots(1,1)
    ax.set_yscale('log', nonpositive='clip')

    #ax.plot(tr.arange(bw+1, int(L/2)+1-bw), tr.abs(bias[bw:-bw])/bias_err[bw:-bw], label="Bias")
    # ax.plot(tr.arange(bw+1, int(L/2)+1-bw), tr.abs(fc_avg[bw:-bw])/fc_err[bw:-bw], label="Factorized")
    # ax.plot(tr.arange(bw+1, int(L/2)+1-bw), tr.abs(c_avg[bw:-bw])/c_err[bw:-bw], label="True")


    ax.set_title(title2)
    ax.set_ylabel('StN Ratio')
    ax.set_xlabel(r'$|x_0 - y_0|$')
    ax.legend(loc='lower right',bbox_to_anchor=(1.02, 1),borderaxespad=0)


    plt.show()
    
def test_Exact_Schur_Correlator():
    batch_size= 30
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.10*lam
    L = 32
    L2 = 16
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 13
    xcut_2 = 29
    bw=3

    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    #Typical equilibration
    q = sim.evolve_f(q, 100)

    #Generate projectors

    #Just fill everything
    if True:
        overlap= True
        projs = tr.eye(2*L*L2, dtype=tr.complex64)
    elif True:
        overlap = True
        projs = tr.zeros(2*bw*2*L2, 2*L*L2, dtype=tr.complex64)
        ct=0
        if overlap:
            for x in tr.arange((xcut_1)*L2*2, (xcut_1+bw)*L2*2):
                projs[ct, x] = 1.0
                ct += 1
            for x in tr.arange((xcut_2)*L2*2, (xcut_2+bw)*L2*2):
                projs[ct, x] = 1.0
                ct += 1
        else:
            for x in tr.arange((xcut_1)*L2*2, (xcut_1+1)*L2*2):
                projs[ct, x] = 1.0
                ct += 1
            for x in tr.arange((L-1)*L2*2, (L)*L2*2):
                projs[ct, x] = 1.0
                ct += 1
    
    
    ensemble_l, ensemble_r, f_propogator = sch.exact_Factorized_Propogator(q, xcut_1, xcut_2,
                                                                          bw, projs)

    factorized_corr, corr = sch.measure_Factorized_Two_Point_Correlator(q, f_propogator, xcut_1, xcut_2,
                                                                        bw)
    
    print(len(factorized_corr))
    c_avg = tr.zeros(len(factorized_corr))
    fc_avg = tr.zeros(len(factorized_corr))
    c_err = tr.zeros(len(factorized_corr))
    fc_err = tr.zeros(len(factorized_corr))

    bias = tr.zeros(len(factorized_corr))
    bias_err = tr.zeros(len(factorized_corr))

    correlation = tr.zeros(len(factorized_corr))
    correlation_err = tr.zeros(len(factorized_corr))

    for x in tr.arange(len(factorized_corr)):
        c_avg[x] = tr.mean(corr[x])
        c_err[x] = tr.std(corr[x])/np.sqrt(tr.numel(corr[x])-1)
        fc_avg[x] = tr.mean(factorized_corr[x])
        fc_err[x] = tr.std(factorized_corr[x])/np.sqrt(tr.numel(factorized_corr[x])-1)
        bias[x] = tr.mean(tr.abs(corr[x]-factorized_corr[x]))
        bias_err[x] = tr.std(corr[x]-factorized_corr[x])/ np.sqrt(tr.numel(corr[x])-1)
        cov = tr.mean(corr[x]*factorized_corr[x]) - tr.mean(corr[x])*tr.mean(factorized_corr[x])
        correlation[x] = cov / (tr.std(corr[x])*tr.std(factorized_corr[x]))
        correlation_err[x] = tr.sqrt((1-tr.square(correlation[x]))/(tr.numel(corr[x]) - 2))


    #Save data
    data_write = tr.stack((tr.arange(bw+1, int(L/2)+1), c_avg, c_err,
                           fc_avg, fc_err, bias, bias_err,
                            correlation, correlation_err), dim=0).numpy()
    np.savetxt('factorized_data.csv', data_write, delimiter=',')
    

    fig, (ax, ax2) = plt.subplots(2,1)
    ax.set_yscale('log', nonpositive='clip')
    ax2.set_yscale('log', nonpositive='clip')


    ax.errorbar(tr.arange(bw+1, int(L/2)+1), fc_avg, fc_err, label="Factorized signal")
    ax.errorbar(tr.arange(bw+1, int(L/2)+1), c_avg, c_err, label="True signal")
    ax.plot(tr.arange(bw+1, int(L/2)+1), fc_err, label="Factorized Error")
    ax.plot(tr.arange(bw+1, int(L/2)+1), c_err, label="True Error")

    ax.errorbar(tr.arange(bw+1, int(L/2)+1), bias, bias_err, label="Bias")
    ax.plot(tr.arange(bw+1, int(L/2)+1), bias_err, label="Bias Error")

    ax.legend(loc='lower right')

    ax.set_title('Unapproximated factorization, n='+ str(batch_size) +', ' r'$m_\pi L = 5$', fontsize=30)
    ax.set_ylabel('Magnitude', fontsize=20)
    #ax.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)

    ax2.errorbar(tr.arange(bw+1, int(L/2)+1), correlation, correlation_err)
    ax2.set_ylabel('Correlation', fontsize=20)
    ax2.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)




    plt.show()


def test_Factorized_Pion_Correlator():
    batch_size= 100
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.10*lam
    L = 64
    L2 = 16
    p_n = 0.0
    p = p_n*2*np.pi/L2
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)


    #Boundary cut timeslices
    xcut_1 = 16
    xcut_2 = 48
    bw=16

    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    #Typical equilibration
    q = sim.evolve_f(q, 100)

    #Generate projectors
    #TODO: Needs work below
    #Lets try some different bases

    if False:
        #Isolates the relevant spin index by boundary
        mult = 1.0
        projs = tr.zeros(2*L2, 2*L*L2, dtype=tr.complex64)
        ct=0
        #Only need second spin index on the first boundary
        for x in tr.arange(1, (1)*L2*2, 2):
                projs[ct, x] = 1.0
                ct += 1
        #Only need first spin index on the second boundary
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1)*L2*2, 2):
                projs[ct, x] = 1.0
                ct += 1

    elif False:
        #Probing half the boundary
        #Need to include a multiplier for using fewer intermediates
        mult = 2.0
        projs = tr.zeros(L2, 2*L*L2, dtype=tr.complex64)
        ct=0
        #Only need second spin index on the first boundary
        for x in tr.arange(1, (1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1
        #Only need first spin index on the second boundary
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1

    elif False:
        #Probing one quarter of the boundary
        #Need to include a multiplier for using fewer intermediates
        mult = 4.0
        projs = tr.zeros(int(L2/2), 2*L*L2, dtype=tr.complex64)
        ct=0
        #Only need second spin index on the first boundary
        for x in tr.arange(1, (1)*L2*2, 8):
                projs[ct, x] = 1.0
                ct += 1
        #Only need first spin index on the second boundary
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1)*L2*2, 8):
                projs[ct, x] = 1.0
                ct += 1
    elif True:
        #Deflation technique
        #Seek eigenmodes of the complement Dirac operator
        title1 = 'Pion Deflation probe, n='+ str(batch_size) +', ' r'$m_\pi L = 5$'
        title2 = 'Deflation probe Pion signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        mult=1.0
        ov=16
        N_vec = 60
        projs = sch.complement_Deflation_Eigenvectors(q, xcut_1, N_vec, ov=ov)
    elif True:
        #Distillation technique
        #Seek distillation eigenvectors for a given batch of configurations
        #TODO:What should the signal boost multiplier be?
        title1 = 'Laplacian Pion probe, n='+ str(batch_size) +', ' r'$m_\pi L = 5$'
        title2 = 'Laplacian Pion signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        mult=1.0
        ov=16
        N_vec = 60
        projs = sch.boundary_Distillation_Eigenvectors(q, xcut_1, N_vec, ov=ov)



    
    
    ensemble_l, ensemble_r, f_propogator = sch.factorized_Propogator_Proj(q, xcut_1, xcut_2,
                                                                          projs, ov=ov)

    factorized_corr, corr = sch.measure_Factorized_Pion_Correlator(q, f_propogator, xcut_1, xcut_2,
                                                                        bw, p=p)
    
    print(len(factorized_corr))
    c_avg = tr.zeros(len(factorized_corr))
    fc_avg = tr.zeros(len(factorized_corr))
    c_err = tr.zeros(len(factorized_corr))
    fc_err = tr.zeros(len(factorized_corr))

    bias = tr.zeros(len(factorized_corr))
    bias_err = tr.zeros(len(factorized_corr))

    correlation = tr.zeros(len(factorized_corr))
    correlation_err = tr.zeros(len(factorized_corr))

    for x in tr.arange(bw, len(factorized_corr)):
        factorized_corr[x] = mult *factorized_corr[x]
        c_corr_avg = tr.real(tr.mean(corr[x], dim=0))
        fc_corr_avg = tr.real(tr.mean(factorized_corr[x], dim=0))
        c_avg[x] = tr.mean(c_corr_avg)
        c_err[x] = tr.std(c_corr_avg)/np.sqrt(tr.numel(c_corr_avg)-1)
        fc_avg[x] = tr.real(tr.mean(fc_corr_avg))
        fc_err[x] = tr.std(tr.real(fc_corr_avg))/np.sqrt(tr.numel(fc_corr_avg)-1)
        bias[x] = c_avg[x] - fc_avg[x]
        bias_err[x] = tr.std(c_corr_avg-fc_corr_avg)/ np.sqrt(tr.numel(c_corr_avg)-1)
        #cov = tr.mean(tr.real(corr[x])*tr.real(factorized_corr[x])) - tr.mean(tr.real(corr[x]))*tr.mean(tr.real(factorized_corr[x]))
        # correlation[x] = cov / (tr.std(tr.real(corr[x]))*tr.std(tr.real(factorized_corr[x])))
        # correlation_err[x] = (1-tr.square(correlation[x]))/np.sqrt(tr.numel(corr[x]) - 3)
        #fcr, correlation[x], correlation_err[x] = jackknife_Correlation(corr[x], factorized_corr[x])
        fcr, correlation[x], correlation_err[x] = jackknife_Correlation(c_corr_avg, fc_corr_avg)


    #Save data
    data_write = tr.stack((tr.arange(1, int(L/2)+1), c_avg, c_err,
                           fc_avg, fc_err, bias, bias_err,
                            correlation, correlation_err), dim=0).numpy()
    np.savetxt('factorized_data.csv', data_write, delimiter=',')
    

    fig, (ax, ax2) = plt.subplots(2,1)
    ax.set_yscale('log', nonpositive='clip')
    ax2.set_yscale('log', nonpositive='clip')


    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(fc_avg[bw:]), fc_err[bw:], label="Factorized signal")
    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(c_avg[bw:]), c_err[bw:], label="True signal")
    ax.plot(tr.arange(bw+1, int(L/2)+1), fc_err[bw:], label="Factorized Error")
    ax.plot(tr.arange(bw+1, int(L/2)+1), c_err[bw:], label="True Error")

    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(bias[bw:]), bias_err[bw:], label="Bias")
    ax.plot(tr.arange(bw+1, int(L/2)+1), bias_err[bw:], label="Bias Error")

    ax.legend(loc='lower right')

    ax.set_title(title1, fontsize=30)
    ax.set_ylabel('Magnitude', fontsize=20)
    #ax.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)

    ax2.errorbar(tr.arange(bw+1, int(L/2)+1), correlation[bw:], correlation_err[bw:])
    ax2.set_ylabel('Correlation', fontsize=20)
    ax2.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)

    plt.show()

    #I'm gonna change up the StN measurment
    #Keep the number of samples computed the same
    measurements = tr.numel(corr[bw+1])
    for x in tr.arange(len(factorized_corr)):
        bias[x] = tr.mean((tr.real(corr[x])-tr.real(factorized_corr[x])))
        bias_err[x] = tr.std(tr.real(corr[x])-tr.real(factorized_corr[x]))/ np.sqrt(tr.numel(corr[x])-1)
        # c_avg[x] = tr.real(tr.mean(corr[x][:measurements]))
        # c_err[x] = tr.std(tr.real(corr[x][:measurements]))/np.sqrt(measurements-1)
        # fc_avg[x] = tr.real(tr.mean(factorized_corr[x][:measurements]))
        # fc_err[x] = tr.std(tr.real(factorized_corr[x][:measurements]))/np.sqrt(measurements-1)


    fig2, ax = plt.subplots(1,1)
    ax.set_yscale('log', nonpositive='clip')

    ax.plot(tr.arange(bw+1, int(L/2)+1-bw), tr.abs(bias[bw:-bw])/bias_err[bw:-bw], label="Bias")
    # ax.plot(tr.arange(bw+1, int(L/2)+1-bw), tr.abs(fc_avg[bw:-bw])/fc_err[bw:-bw], label="Factorized")
    # ax.plot(tr.arange(bw+1, int(L/2)+1-bw), tr.abs(c_avg[bw:-bw])/c_err[bw:-bw], label="True")

    ax.set_title(title2, fontsize=30)
    ax.set_ylabel('StN Ratio', fontsize=20)
    ax.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)
    ax.legend(loc='lower right')

    plt.show()
    

#Outdated
def two_Level_Factorized_Pion():
    batch_size= 100
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.10*lam
    L = 32
    L2 = 16
    p_n = 1.0
    p = p_n*2*np.pi/L2
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)


    #Boundary cut timeslices
    xcut_1 = 13
    xcut_2 = 29
    bw=3

    #Number of level 1 configurations per level 0 config
    n1=2

    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)
    lvl2_im2 = i.minnorm2(sch.dd_Force,sch.evolveQ,20, 1.0)
    lvl2_sim = h.hmc(sch, lvl2_im2, False)

    #Typical equilibration
    q = sim.evolve_f(q, 200)

    #Try a simple even-odd projection
    #Probing half the boundary
    #Need to include a multiplier for using fewer intermediates
    if False:
        dynamic_projectors = False
        mult = 2.0
        ov=1
        projs = tr.zeros(L2, 2*L*L2, dtype=tr.complex64)
        ct=0
        #Only need second spin index on the first boundary
        for x in tr.arange(1, (1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1
        #Only need first spin index on the second boundary
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1

        N_vec= ct

        title1 = '2-level even-odd pion, n='+ str(batch_size) +', ' r'$m_\pi L = 5$ '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, '
        title2 = '2 level even-odd 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
    if True:
        dynamic_projectors = True
        #Deflation technique
        #TODO- needs to update each complement config
        #Seek eigenmodes of the complement Dirac operator
        title1 = 'Deflation probe, n='+ str(batch_size) +', ' r'$m_\pi L = 5$ '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, '
        title2 = 'Deflation probe 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        mult=1.0
        ov=3
        N_vec = 16
        projs = sch.complement_Deflation_Eigenvectors(q, xcut_1, N_vec, ov=ov)


    q1 = tuple(q)

    #Measure Bias and correlation in the level-0 config
    print("Measuring Bias")

    #Measure the level-0 config using true and factorized propogator
    e_l, e_r, f_propogator = sch.factorized_Propogator_Proj(q1, xcut_1, xcut_2,
                                                                        projs, ov)
    factorized_corr0, corr0 = sch.measure_Factorized_Pion_Correlator(q1, f_propogator, xcut_1, xcut_2, 
                                                                    bw, p=p)
    #Measure bias and correlation

    max_length = int((L)/2)

    c_avg0 = tr.zeros(len(factorized_corr0))
    fc_avg0 = tr.zeros(len(factorized_corr0))
    c_err0 = tr.zeros(len(factorized_corr0))
    fc_err0 = tr.zeros(len(factorized_corr0))
    bias0 = tr.zeros(max_length)
    bias_err0 = tr.zeros(max_length)
    correlation0 = tr.zeros(max_length)
    correlation_err0 = tr.zeros(max_length)


    for x in np.arange(bw, max_length):
        factorized_corr0[x] = mult *factorized_corr0[x]
        c_avg0[x] = tr.real(tr.mean(corr0[x]))
        c_err0[x] = tr.std(tr.real(corr0[x]))/np.sqrt(tr.numel(corr0[x])-1)
        fc_avg0[x] = tr.real(tr.mean(factorized_corr0[x]))
        fc_err0[x] = tr.std(tr.real(factorized_corr0[x]))/np.sqrt(tr.numel(factorized_corr0[x])-1)
        bias0[x] = tr.mean((tr.real(corr0[x])-tr.real(factorized_corr0[x])))
        bias_err0[x] = tr.std(tr.real(corr0[x])-tr.real(factorized_corr0[x]))/ np.sqrt(tr.numel(corr0[x])-1)
        cov = tr.mean(tr.real(corr0[x])*tr.real(factorized_corr0[x])) - tr.mean(tr.real(corr0[x]))*tr.mean(tr.real(factorized_corr0[x]))
        correlation0[x] = cov / (tr.std(tr.real(corr0[x]))*tr.std(tr.real(factorized_corr0[x])))
        correlation_err0[x] = tr.sqrt((1-tr.square(correlation0[x]))/(tr.numel(corr0[x]) - 2))
        



    print("Beginning 2-level integration")
    for m in np.arange(n1):
        #2nd level integration
        q = lvl2_sim.second_Level_Evolve(q, 50, xcut_1, xcut_2, bw)

        #Construct projectors here if not fixed
        if dynamic_projectors == True:
            projs = sch.complement_Deflation_Eigenvectors(q, xcut_1, N_vec, ov=ov)
            if m == 0:
                projs_ensemble = tr.zeros((batch_size, n1, N_vec+1, tr.numel(projs[0,0,:])), dtype=tr.complex64)
                projs_ensemble[:,m,:,:] = projs.clone()
            else:
                projs_ensemble[:,m,:,:] = projs.clone()
                

        if m == 0:
            bulk_product, s1_inv = sch.measure_Factorized_Subdomain_Inverses(q, xcut_1, ov)
            
            bulk_ensemble = tr.zeros((batch_size, n1, tr.numel(bulk_product[0,:,0]), tr.numel(bulk_product[0,0,:])), dtype=tr.complex64)
            s1_ensemble = tr.zeros((batch_size, n1, tr.numel(s1_inv[0,:,0]), tr.numel(s1_inv[0,0,:])), dtype=tr.complex64)

            bulk_ensemble[:, m, :,:] = bulk_product.clone()
            s1_ensemble[:,m,:,:] = s1_inv.clone()
            
        else:
            bulk_ensemble[:,m,:,:], s1_ensemble[:,m,:,:] = sch.measure_Factorized_Subdomain_Inverses(q, xcut_1, ov)
        print(m)

    factorized_corr = [None]* (max_length)
    
    print("Measuring Ensembles")    
    #Combine for n1^2 measurements
    for m1 in np.arange(n1):


        for m2 in np.arange(n1):
            #Make measurement
            if dynamic_projectors == True:
                ens_fc = sch.measure_Two_Lvl_Factorized_Pion_Correlator(bulk_ensemble[:,m1, :,:], s1_ensemble[:,m2,:,:], 
                                                                        xcut_1, projs_ensemble[:,m2,:,:], bw, ov, p)
            else:
                ens_fc = sch.measure_Two_Lvl_Factorized_Pion_Correlator(bulk_ensemble[:,m1, :,:], s1_ensemble[:,m2,:,:], 
                                                                        xcut_1, projs, bw, ov, p)
            
            #store measurement
            if m1 == 0 and m2 == 0:
                for t in np.arange(max_length):
                    factorized_corr[t] = ens_fc[t]
            else:
                for t in np.arange(bw, max_length):
                    temp = tr.cat((factorized_corr[t], ens_fc[t]), 0)
                    factorized_corr[t] = temp

        print(m1)




    fc_avg = tr.zeros(len(factorized_corr))
    fc_err = tr.zeros(len(factorized_corr))


    for x in tr.arange(bw, len(factorized_corr)):
        #factorized_corr[x] = mult *factorized_corr[x]
        factorized_corr[x] = mult *factorized_corr[x] + bias0[x]
        fc_avg[x] = tr.real(tr.mean(factorized_corr[x]))
        fc_err[x] = tr.std(tr.real(factorized_corr[x]))/np.sqrt(tr.numel(factorized_corr[x])-1)
        #fc_err[x] = tr.sqrt(tr.square(tr.std(tr.real(factorized_corr[x]))/np.sqrt(tr.numel(factorized_corr[x])-1)) 
        #                    + bias_err0[x]**2)
        # bias[x] = tr.mean((tr.real(corr[x])-tr.real(factorized_corr[x])))
        # bias_err[x] = tr.std(tr.real(corr[x])-tr.real(factorized_corr[x]))/ np.sqrt(tr.numel(corr[x])-1)




    #Save data
    data_write = tr.stack((tr.arange(1, int(L/2)+1), c_avg0, c_err0,
                           fc_avg, fc_err, bias0, bias_err0,
                            correlation0, correlation_err0), dim=0).numpy()
    np.savetxt('2_level_factorized_data.csv', data_write, delimiter=',')
    
    #First plot level 0

    fig, (ax, ax2) = plt.subplots(2,1)
    ax.set_yscale('log', nonpositive='clip')
    ax2.set_yscale('log', nonpositive='clip')


    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(fc_avg0[bw:]), fc_err0[bw:], label="Factorized signal")
    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(c_avg0[bw:]), c_err0[bw:], label="Level-0 true signal")
    ax.plot(tr.arange(bw+1, int(L/2)+1), fc_err0[bw:], label="Factorized Error")
    ax.plot(tr.arange(bw+1, int(L/2)+1), c_err0[bw:], label="True Error")

    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(bias0[bw:]), bias_err0[bw:], label="Bias")
    ax.plot(tr.arange(bw+1, int(L/2)+1), bias_err0[bw:], label="Bias Error")

    ax.legend(loc='lower right')

    ax.set_title(title1, fontsize=30)
    ax.set_ylabel('Magnitude', fontsize=20)
    #ax.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)

    ax2.errorbar(tr.arange(bw+1, int(L/2)+1), correlation0[bw:], correlation_err0[bw:])
    ax2.set_ylabel('Correlation', fontsize=20)
    ax2.set_ylim(0,1)
    #ax2.grid(True, axis='y', ls='--')
    ax2.grid(which='minor', color='0.9')
    ax2.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)

    #Now plot Level 1

    fig2, ax3 = plt.subplots(1,1)
    ax3.set_yscale('log', nonpositive='clip')


    ax3.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(fc_avg[bw:]), fc_err[bw:], label="2-level signal + Bias")
    ax3.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(c_avg0[bw:]), c_err0[bw:], label="Level-0 true signal")
    ax3.plot(tr.arange(bw+1, int(L/2)+1), fc_err[bw:], label="2-level Error")
    ax3.plot(tr.arange(bw+1, int(L/2)+1), c_err0[bw:], label="True Error")

    #ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(bias0[bw:]), bias_err0[bw:], label="Bias")
    #ax.plot(tr.arange(bw+1, int(L/2)+1), bias_err0[bw:], label="Bias Error")

    ax3.legend(loc='lower right')

    ax3.set_title("2nd-level integration, n = " + str(n1*n1*batch_size), fontsize=30)

    ax3.set_ylabel('Magnitude', fontsize=20)
    #ax.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)

    #I'm gonna change up the StN measurment
    #Keep the number of samples computed the same
    # measurements = tr.numel(corr[bw+1])
    # for x in tr.arange(bw+1, len(factorized_corr)):
    #     bias[x] = tr.mean((tr.real(corr[x])-tr.real(factorized_corr[x])))
    #     bias_err[x] = tr.std(tr.real(corr[x])-tr.real(factorized_corr[x]))/ np.sqrt(tr.numel(corr[x])-1)
    #     c_avg[x] = tr.real(tr.mean(corr[x][:measurements]))
    #     c_err[x] = tr.std(tr.real(corr[x][:measurements]))/np.sqrt(measurements-1)
    #     fc_avg[x] = tr.real(tr.mean(factorized_corr[x][:measurements]))
    #     fc_err[x] = tr.std(tr.real(factorized_corr[x][:measurements]))/np.sqrt(measurements-1)


    fig3, ax5 = plt.subplots(1,1)
    ax5.set_yscale('log', nonpositive='clip')

    ax5.plot(tr.arange(bw+1, int(L/2)+1-bw), tr.abs(bias0[bw:-bw])/bias_err0[bw:-bw] , label="Bias")
    ax5.plot(tr.arange(bw+1, int(L/2)+1-bw), tr.abs(fc_avg[bw:-bw])/fc_err[bw:-bw], label="2-level")
    ax5.plot(tr.arange(bw+1, int(L/2)+1-bw), tr.abs(c_avg0[bw:-bw])/c_err0[bw:-bw], label="1-level")


    ax5.set_title(title2, fontsize=30)
    ax5.set_ylabel('StN Ratio', fontsize=20)
    ax5.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)
    ax5.legend(loc='lower right')

    plt.show()


def two_Level_Stream_Pion_Bias():
    batch_size= 10
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.10*lam
    L = 32
    L2 = 16
    p_n = 0.0
    p = p_n*2*np.pi/L2
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)


    #Boundary cut timeslices
    xcut_1 = 13
    xcut_2 = 29
    bw=3

    #Number of level 1 configurations per level 0 config
    n1=20

    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)
    lvl2_im2 = i.minnorm2(sch.dd_Force,sch.evolveQ,20, 1.0)
    lvl2_sim = h.hmc(sch, lvl2_im2, False)

    #Typical equilibration
    q = sim.evolve_f(q, 200)

    #Try a simple even-odd projection
    #Probing half the boundary
    #Need to include a multiplier for using fewer intermediates
    if False:
        dynamic_projectors = False
        mult = 2.0
        ov=1
        projs = tr.zeros(L2, 2*L*L2, dtype=tr.complex64)
        ct=0
        #Only need second spin index on the first boundary
        for x in tr.arange(1, (1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1
        #Only need first spin index on the second boundary
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1

        N_vec= ct

        title1 = '2-level even-odd pion, n='+ str(batch_size) +', ' r'$m_\pi L = 5$ '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, '
        title2 = '2 level even-odd 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
    if True:
        dynamic_projectors = True
        #Deflation technique
        #TODO- needs to update each complement config
        #Seek eigenmodes of the complement Dirac operator
        title1 = 'Deflation probe, n0='+ str(batch_size) +', ' +'n1=' + str(n1) +', '  r'$m_\pi L = 5$ '+ ' p='+ str(p_n)+ \
                r'$*2\pi/L$, '
        title2 = 'Deflation probe 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        mult=1.0
        ov=3
        N_vec = 16
        projs = sch.complement_Deflation_Eigenvectors(q, xcut_1, N_vec, ov=ov)


    q1 = tuple(q)

    #Measure Bias and correlation in the level-0 config
    print("Measuring first level configs")

    #Measure the level-0 config using true and factorized propogator
    e_l, e_r, f_propogator = sch.factorized_Propogator_Proj(q1, xcut_1, xcut_2,
                                                                        projs, ov)
    factorized_corr0, corr0 = sch.measure_Factorized_Pion_Correlator(q1, f_propogator, xcut_1, xcut_2, 
                                                                    bw, p=p)


    #Measure the correlator and its error

    max_length = int((L)/2)

    #Take statistics on each configuration in the batch individually
    c_avg0 = tr.zeros(batch_size, len(corr0))
    c_err0 = tr.zeros(batch_size, len(corr0))


    for x in np.arange(bw, max_length):
        c_avg0[:, x] = tr.real(tr.mean(corr0[x], dim=0))
        c_err0[:, x] = tr.std(tr.real(corr0[x]), dim=0)/np.sqrt(tr.numel(corr0[x][:,0])-1)


    #Take a true correlator measurement on the unstitched two level configs
    c_avg1 = tr.zeros(batch_size, n1, len(corr0))


    print("Beginning 2-level integration")
    for m in np.arange(n1):
        #2nd level integration
        q = lvl2_sim.second_Level_Evolve(q, 50, xcut_1, xcut_2, bw)

        #Take true correlator measurement
        factorized_corr1, c1 = sch.measure_Factorized_Pion_Correlator(q, f_propogator, xcut_1, xcut_2, 
                                                                    bw, p=p)

        for x in np.arange(bw, max_length):
            c_avg1[:,m, x] = tr.real(tr.mean(c1[x], dim=0))

        #Construct projectors here if not fixed
        if dynamic_projectors == True:
            projs = sch.complement_Deflation_Eigenvectors(q, xcut_1, N_vec, ov=ov)
            if m == 0:
                projs_ensemble = tr.zeros((batch_size, n1, N_vec+1, tr.numel(projs[0,0,:])), dtype=tr.complex64)
                projs_ensemble[:,m,:,:] = projs.clone()
            else:
                projs_ensemble[:,m,:,:] = projs.clone()
                

        if m == 0:
            bulk_product, s1_inv = sch.measure_Factorized_Subdomain_Inverses(q, xcut_1, ov)
            
            bulk_ensemble = tr.zeros((batch_size, n1, tr.numel(bulk_product[0,:,0]), tr.numel(bulk_product[0,0,:])), dtype=tr.complex64)
            s1_ensemble = tr.zeros((batch_size, n1, tr.numel(s1_inv[0,:,0]), tr.numel(s1_inv[0,0,:])), dtype=tr.complex64)

            bulk_ensemble[:, m, :,:] = bulk_product.clone()
            s1_ensemble[:,m,:,:] = s1_inv.clone()
            
        else:
            bulk_ensemble[:,m,:,:], s1_ensemble[:,m,:,:] = sch.measure_Factorized_Subdomain_Inverses(q, xcut_1, ov)
        print(m)

    #Empty list which will contain tensors of variable size based on time separation
    factorized_corr = [None]* (max_length)
    
    print("Measuring Ensembles")    
    #Combine for n1^2 measurements
    for m1 in np.arange(n1):


        for m2 in np.arange(n1):
            #Make measurement
            if dynamic_projectors == True:
                ens_fc = sch.measure_Two_Lvl_Factorized_Pion_Correlator(bulk_ensemble[:,m1, :,:], s1_ensemble[:,m2,:,:], 
                                                                        xcut_1, projs_ensemble[:,m2,:,:], bw, ov, p)
            else:
                ens_fc = sch.measure_Two_Lvl_Factorized_Pion_Correlator(bulk_ensemble[:,m1, :,:], s1_ensemble[:,m2,:,:], 
                                                                        xcut_1, projs, bw, ov, p)
            
            #store measurement
            if m1 == 0 and m2 == 0:
                for t in np.arange(bw, max_length):
                    factorized_corr[t] = tr.zeros((batch_size, n1,n1, tr.numel(ens_fc[t][:,0])), dtype=tr.complex64)
                    factorized_corr[t][:,m1,m2,:] = tr.transpose(ens_fc[t],0,1)  
            else:
                for t in np.arange(bw, max_length):
                    factorized_corr[t][:,m1,m2, :] = tr.transpose(ens_fc[t],0,1)
        print(m1)




    fc_avg = tr.zeros(batch_size, len(factorized_corr))
    fc_err = tr.zeros(batch_size, len(factorized_corr))
    fc_splicing_bias = tr.zeros(batch_size, n1, len(factorized_corr))

    fc_splicing_bias_avg = tr.zeros(len(factorized_corr))
    fc_splicing_bias_err = tr.zeros(len(factorized_corr))

    instream_bias = tr.zeros(len(factorized_corr))
    instream_bias_err = tr.zeros(len(factorized_corr))

    #Take global measurements of each
    global_c_avg0 = tr.zeros(len(factorized_corr))
    global_c_err0 = tr.zeros(len(factorized_corr))
    global_fc_avg = tr.zeros(len(factorized_corr))
    global_fc_err = tr.zeros(len(factorized_corr))

    for x in np.arange(bw, len(factorized_corr)):
        fc_config_avg = tr.real(tr.mean(factorized_corr[x], dim=(1,2,3)))
        global_fc_avg[x] = tr.mean(fc_config_avg)
        global_fc_err[x] = tr.std(fc_config_avg) / np.sqrt(tr.numel(fc_config_avg) - 1)


    print("Measuring Bias") 
    for x in np.arange(bw, len(factorized_corr)):
        factorized_corr[x] = mult *factorized_corr[x]

        fc_avg[:,x] = tr.real(tr.mean(factorized_corr[x], dim=(1,2,3)))
        fc_err[:,x] = tr.std(tr.real(factorized_corr[x]), dim=(1,2,3))/np.sqrt(tr.numel(factorized_corr[x][0,:,:,:])-1)


        #Measure splicing bias
        for m1 in np.arange(n1):
            #unspliced_avg = tr.real(tr.mean(factorized_corr[x][:, m1,m1,:], dim=1))
            #unspliced_err = tr.std(tr.real(factorized_corr[x][:, m1,m1,:]), dim=1)/np.sqrt(tr.numel(factorized_corr[x][0,0,0,:])-1)
            if m1 ==0:
                spliced_measurements = tr.cat((tr.flatten(factorized_corr[x][:,m1,m1+1:,:],start_dim=1),
                                          tr.flatten(factorized_corr[x][:,m1+1:,m1,:],start_dim=1)), dim=1)
            elif m1 == n1-1:
                spliced_measurements = tr.cat((tr.flatten(factorized_corr[x][:,m1,0:m1,:],start_dim=1),
                                          tr.flatten(factorized_corr[x][:,0:m1,m1,:],start_dim=1)), dim=1)
            else:
                spliced_measurements = tr.cat((tr.flatten(factorized_corr[x][:,m1,0:m1,:],start_dim=1),
                                          tr.flatten(factorized_corr[x][:,m1,m1+1:,:],start_dim=1),
                                          tr.flatten(factorized_corr[x][:,0:m1,m1,:],start_dim=1),
                                          tr.flatten(factorized_corr[x][:,m1+1:,m1,:],start_dim=1)),dim=1)
 
            spliced_avg = tr.real(tr.mean(spliced_measurements, dim=1))

            #corr_level_bias = corr1[x][:, m1, :] - spliced_avg.unsqueeze(1).repeat(1, tr.numel(corr1[x][:, m1, :]))

            fc_splicing_bias[:, m1, x] = c_avg1[:,m1, x] - spliced_avg

        
        #Average the spliced bias measurement
        fc_splicing_bias_avg[x] = tr.real(tr.mean(fc_splicing_bias[:,:,x]))
        fc_splicing_bias_err[x] = tr.std(tr.real(fc_splicing_bias[:,:,x]))/np.sqrt(tr.numel(fc_splicing_bias[:,:,x])-1)

        #Now measure the stream bias
        #Need to repeat the 2-level expectation value to subtract from first level measurements

        corr_level_bias = corr0[x] - fc_avg[:,x]
        instream_bias[x] = tr.mean(tr.real(corr_level_bias))
        instream_bias_err[x] = tr.std(tr.real(corr_level_bias))/np.sqrt(batch_size-1)

        #Take global measurements
        config_avg = tr.real(tr.mean(corr0[x], dim=0))
        global_c_avg0[x] = tr.mean(config_avg)
        global_c_err0[x] = tr.std(config_avg)/np.sqrt(tr.numel(config_avg)-1)
        fc_config_avg = tr.real(tr.mean(factorized_corr[x], dim=(1,2,3)))
        global_fc_avg[x] = tr.mean(fc_config_avg)
        global_fc_err[x] = tr.std(fc_config_avg) / np.sqrt(tr.numel(fc_config_avg) - 1)
        # global_c_avg0[x] = tr.real(tr.mean(corr0[x], dim=(0,1)))
        # global_c_err0[x] = tr.std(tr.real(corr0[x]), dim=(0,1))/np.sqrt(tr.numel(corr0[x])-1)
        # global_fc_avg[x] = tr.real(tr.mean(factorized_corr[x], dim=(0,1,2,3)))
        # global_fc_err[x] = tr.std(tr.real(factorized_corr[x]), dim=(0,1,2,3))/np.sqrt(tr.numel(factorized_corr[x])-1)

        print(x)


    #Save data
    data_write = tr.stack((tr.arange(1, int(L/2)+1), global_c_avg0, global_c_err0,
                            global_fc_avg, global_fc_err, instream_bias, instream_bias_err, fc_splicing_bias_avg, fc_splicing_bias_err), dim=0).numpy()
    np.savetxt('2_level_factorized_data.csv', data_write, delimiter=',')
    
    #Plot Data

    fig, (ax, ax1, ax2) = plt.subplots(3,1, sharex=True, constrained_layout=True)
    ax.set_yscale('log', nonpositive='clip')
    ax1.set_yscale('log', nonpositive='clip')
    ax2.set_yscale('log', nonpositive='clip')


    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(global_c_avg0[bw:]), global_c_err0[bw:], label="First level measurement",
                fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)
    ax1.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(global_fc_avg[bw:]), global_fc_err[bw:], label="Two level Measurement",
                 fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)
    #ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(instream_bias[bw:]), instream_bias_err[bw:], label="Stream Bias")
    ax.errorbar(tr.arange(bw+1, int(L/2) + 1), tr.abs(fc_splicing_bias_avg[bw:]), fc_splicing_bias_err[bw:], label="Cross-stitch Bias",
                fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)
    ax1.errorbar(tr.arange(bw+1, int(L/2) + 1), tr.abs(fc_splicing_bias_avg[bw:]), fc_splicing_bias_err[bw:], label="Cross-stitch Bias",
                 fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)
    #ax.plot(tr.arange(bw+1, int(L/2)+1), instream_bias_err[bw:], label="Stream Bias Err")
    ax.plot(tr.arange(bw+1, int(L/2) + 1), global_c_err0[bw:], label="First level Error")
    ax1.plot(tr.arange(bw+1, int(L/2) + 1), global_fc_err[bw:], label="2nd level Error")
    ax.plot(tr.arange(bw+1, int(L/2) + 1), fc_splicing_bias_err[bw:], label="Cross-stitch Bias Err")
    ax1.plot(tr.arange(bw+1, int(L/2) + 1), fc_splicing_bias_err[bw:], label="Cross-stitch Bias Err")

    ax2.plot(tr.arange(bw+1, int(L/2)+1), tr.abs(global_c_avg0[bw:])/global_c_err0[bw:], label="1-level")
    ax2.plot(tr.arange(bw+1, int(L/2)+1), tr.abs(global_fc_avg[bw:])/global_fc_err[bw:], label="2-level")
    #ax2.plot(tr.arange(bw+1, int(L/2)+1-bw), tr.abs(instream_bias[bw:-bw])/instream_bias_err[bw:-bw] , label="Stream Bias")
    ax2.plot(tr.arange(bw+1, int(L/2)+1), tr.abs(fc_splicing_bias_avg[bw:])/fc_splicing_bias_err[bw:] , label="Cross-stitch Bias")

    # handles, labels = ax.gca().get_legend_handles_labels()

    # order = [2,3,4,5,0,1]
    # handles = [handles[ix] for ix in order]
    # labels = [labels[ix] for ix in order]

    # ax.legend(handles, labels, loc='lower right')
    ax.legend(
    loc="upper left",bbox_to_anchor=(1.02, 1),borderaxespad=0)       
    ax1.legend(
    loc="upper left",bbox_to_anchor=(1.02, 1),borderaxespad=0)
    ax2.legend(
    loc="upper left",bbox_to_anchor=(1.02, 1),borderaxespad=0)

    ax.set_title(title1)
    ax.set_ylabel('Magnitude')
    ax1.set_ylabel('Magnitude')
    #ax.set_xlabel(r'$|x_0 - y_0|$', fontsize=20)

    ax2.set_ylabel('StN')
    #ax2.grid(True, axis='y', ls='--')
    ax.grid(which='major', linestyle='--', alpha=0.6)
    ax1.grid(which='major', linestyle='--', alpha=0.6)
    ax2.grid(which='major', linestyle='--', alpha=0.6)
    #ax2.grid(which='minor', linestyle=':', alpha=0.3)
    ax2.set_xlabel(r'$|x_0 - y_0|$')

    plt.savefig(
    "figure.pdf",
    bbox_inches="tight"
    )
        
    plt.show()


#Fit function for pion triplet
def f_pi_decay(x, m, A):
    #Note hardcoded parameter for fit
    T = 32
    return A* (np.exp(-m *x) + np.exp(-m*(T-x)))

def two_Level_Quenched_Pion_Mass():
    batch_size= 20
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.10*lam
    L = 32
    L2 = 16
    p_n = 0.0
    p = p_n*2*np.pi/L2
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)


    #Boundary cut timeslices
    xcut_1 = 13
    xcut_2 = 29
    bw=3

    #Maximum possible length of cross-subdomain correlator
    max_length = int((L)/2)

    #Number of level 1 configurations per level 0 config
    n1=40

    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)
    lvl2_im2 = i.minnorm2(sch.dd_Force,sch.evolveQ,20, 1.0)
    lvl2_sim = h.hmc(sch, lvl2_im2, False)

    #Typical equilibration
    q = sim.evolve_f(q, 400)

    #Try a simple even-odd projection
    #Probing half the boundary
    #Need to include a multiplier for using fewer intermediates
    if False:
        dynamic_projectors = False
        mult = 2.0
        ov=1
        projs = tr.zeros(L2, 2*L*L2, dtype=tr.complex64)
        ct=0
        #Only need second spin index on the first boundary
        for x in tr.arange(1, (1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1
        #Only need first spin index on the second boundary
        for x in tr.arange((xcut_1-1)*2*L2, (xcut_1)*L2*2, 4):
                projs[ct, x] = 1.0
                ct += 1

        N_vec= ct

        title1 = '2-level even-odd pion, n='+ str(batch_size) +', ' r'$m_\pi L = 5$ '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, '
        title2 = '2 level even-odd 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
    if True:
        dynamic_projectors = True
        #Deflation technique
        #TODO- needs to update each complement config
        #Seek eigenmodes of the complement Dirac operator
        title1 = 'Deflation probe, n1='+ str(batch_size) +', n2=' + str(n1)+  r'$m_\pi L = 5$ '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, '
        title2 = 'Deflation probe 2-pt signal to noise: '+ 'n='+ str(batch_size) +', '+ 'p='+ str(p_n)+ \
                r'$*2\pi/L$, ' + r'$m_\pi L = 5$'
        mult=1.0
        ov=bw
        N_vec = 16
        projs = sch.complement_Deflation_Eigenvectors(q, xcut_1, N_vec, ov=ov)


    q1 = tuple(q)

    #Measure Bias and correlation in the level-0 config
    print("Measuring first level configs")

    # corr0 = [None] * max_length

    # for n in np.arange(n1):
    #     #Discard some in between
    #     q1= sim.evolve_f(q1, 50)

    #     e_l, e_r, f_propogator = sch.factorized_Propogator_Proj(q1, xcut_1, xcut_2,
    #                                                                     projs, ov)
    #     factorized_corr0, c = sch.measure_Factorized_Pion_Correlator(q1, f_propogator, xcut_1, xcut_2, 
    #                                                                 bw, p=p)
        
    #     if n == 0:
    #         for t in np.arange(bw, max_length):
    #             corr0[t] = tr.zeros((batch_size, n1, tr.numel(c[t][:,0])), dtype=tr.complex64)
    #             corr0[t][:,n,:] = tr.transpose(c[t],0,1)  
    #     else:
    #         for t in np.arange(bw, max_length):
    #             corr0[t][:,n,:] = tr.transpose(c[t],0,1)



    # #Average over all batches and configurations measured
    # c_avg0 = tr.zeros(max_length)
    # c_err0 = tr.zeros(max_length)
    # for x in np.arange(bw, len(corr0)):
    #     c_avg0[x] = tr.real(tr.mean(corr0[x]))
    #     c_err0[x] = tr.std(tr.real(corr0[x]))/np.sqrt(tr.numel(corr0[x])-1)
    

    #Measure the level-0 config using true and factorized propogator
    e_l, e_r, f_propogator = sch.factorized_Propogator_Proj(q1, xcut_1, xcut_2,
                                                                        projs, ov)
    factorized_corr0, corr0 = sch.measure_Factorized_Pion_Correlator(q1, f_propogator, xcut_1, xcut_2, 
                                                                    bw, p=p)
    

    #Measure the correlator and its error

    max_length = int((L)/2)

    #Take statistics on each configuration in the batch individually
    c_avg0 = tr.zeros(len(corr0))
    c_err0 = tr.zeros(len(corr0))


    for x in np.arange(bw, max_length):
        config_avg = tr.real(tr.mean(corr0[x], dim=0))
        c_avg0[x] = tr.mean(config_avg)
        c_err0[x] = tr.std(config_avg)/np.sqrt(tr.numel(config_avg)-1)
        # c_avg0[x] = tr.real(tr.mean(corr0[x], dim=(0,1)))
        # c_err0[x] = tr.std(tr.real(corr0[x]), dim=(0,1))/np.sqrt(tr.numel(corr0[x])-1)


    #Measure the correlator and its error

    #Take a true correlator measurement on the unstitched two level configs
    c_avg1 = tr.zeros(batch_size, n1, len(corr0))
    

    print("Beginning 2-level integration")
    for m in np.arange(n1):
        #2nd level integration
        q = lvl2_sim.second_Level_Evolve(q, 100, xcut_1, xcut_2, bw)

        #Take true correlator measurement
        factorized_corr1, c1 = sch.measure_Factorized_Pion_Correlator(q, f_propogator, xcut_1, xcut_2, 
                                                                    bw, p=p)
        for x in np.arange(bw, max_length):
            c_avg1[:,m, x] = tr.real(tr.mean(c1[x], dim=0))

        #Construct projectors here if not fixed
        if dynamic_projectors == True:
            projs = sch.complement_Deflation_Eigenvectors(q, xcut_1, N_vec, ov=ov)
            if m == 0:
                projs_ensemble = tr.zeros((batch_size, n1, N_vec+1, tr.numel(projs[0,0,:])), dtype=tr.complex64)
                projs_ensemble[:,m,:,:] = projs.clone()
            else:
                projs_ensemble[:,m,:,:] = projs.clone()
                

        if m == 0:
            bulk_product, s1_inv = sch.measure_Factorized_Subdomain_Inverses(q, xcut_1, ov)
            
            bulk_ensemble = tr.zeros((batch_size, n1, tr.numel(bulk_product[0,:,0]), tr.numel(bulk_product[0,0,:])), dtype=tr.complex64)
            s1_ensemble = tr.zeros((batch_size, n1, tr.numel(s1_inv[0,:,0]), tr.numel(s1_inv[0,0,:])), dtype=tr.complex64)

            bulk_ensemble[:, m, :,:] = bulk_product.clone()
            s1_ensemble[:,m,:,:] = s1_inv.clone()
            
        else:
            bulk_ensemble[:,m,:,:], s1_ensemble[:,m,:,:] = sch.measure_Factorized_Subdomain_Inverses(q, xcut_1, ov)
        print(m)

    #Empty list which will contain tensors of variable size based on time separation
    factorized_corr = [None]* (max_length)


    
    print("Measuring Ensembles")    
    #Combine for n1^2 measurements
    for m1 in np.arange(n1):


        for m2 in np.arange(n1):
            #Make measurement
            if dynamic_projectors == True:
                ens_fc = sch.measure_Two_Lvl_Factorized_Pion_Correlator(bulk_ensemble[:,m1, :,:], s1_ensemble[:,m2,:,:], 
                                                                        xcut_1, projs_ensemble[:,m2,:,:], bw, ov, p)
            else:
                ens_fc = sch.measure_Two_Lvl_Factorized_Pion_Correlator(bulk_ensemble[:,m1, :,:], s1_ensemble[:,m2,:,:], 
                                                                        xcut_1, projs, bw, ov, p)
            
            #store measurement
            if m1 == 0 and m2 == 0:
                for t in np.arange(bw, max_length):
                    factorized_corr[t] = tr.zeros((batch_size, n1,n1, tr.numel(ens_fc[t][:,0])), dtype=tr.complex64)
                    factorized_corr[t][:,m1,m2,:] = tr.transpose(ens_fc[t],0,1)  
            else:
                for t in np.arange(bw, max_length):
                    factorized_corr[t][:,m1,m2, :] = tr.transpose(ens_fc[t],0,1)
        print(m1)

    global_fc_avg = tr.zeros(len(factorized_corr))
    global_fc_err = tr.zeros(len(factorized_corr))

    for x in np.arange(bw, len(factorized_corr)):
        fc_config_avg = tr.real(tr.mean(factorized_corr[x], dim=(1,2,3)))
        global_fc_avg[x] = tr.mean(fc_config_avg)
        global_fc_err[x] = tr.std(fc_config_avg) / np.sqrt(tr.numel(fc_config_avg) - 1)
        # global_fc_avg[x] = tr.real(tr.mean(factorized_corr[x], dim=(0,1,2,3)))
        # global_fc_err[x] = tr.std(tr.real(factorized_corr[x]), dim=(0,1,2,3))/np.sqrt(tr.numel(factorized_corr[x])-1)


    #Bias measurement

    fc_avg = tr.zeros(batch_size, len(factorized_corr))
    fc_err = tr.zeros(batch_size, len(factorized_corr))
    fc_splicing_bias = tr.zeros(batch_size, n1, len(factorized_corr))

    fc_splicing_bias_avg = tr.zeros(len(factorized_corr))
    fc_splicing_bias_err = tr.zeros(len(factorized_corr))

    print("Measuring Bias") 
    for x in np.arange(bw, len(factorized_corr)):
        factorized_corr[x] = mult *factorized_corr[x]

        fc_avg[:,x] = tr.real(tr.mean(factorized_corr[x], dim=(1,2,3)))
        fc_err[:,x] = tr.std(tr.real(factorized_corr[x]), dim=(1,2,3))/np.sqrt(tr.numel(factorized_corr[x][0,:,:,:])-1)


        #Measure splicing bias
        for m1 in np.arange(n1):
            #unspliced_avg = tr.real(tr.mean(factorized_corr[x][:, m1,m1,:], dim=1))
            #unspliced_err = tr.std(tr.real(factorized_corr[x][:, m1,m1,:]), dim=1)/np.sqrt(tr.numel(factorized_corr[x][0,0,0,:])-1)
            if m1 ==0:
                spliced_measurements = tr.cat((tr.flatten(factorized_corr[x][:,m1,m1+1:,:],start_dim=1),
                                          tr.flatten(factorized_corr[x][:,m1+1:,m1,:],start_dim=1)), dim=1)
            elif m1 == n1-1:
                spliced_measurements = tr.cat((tr.flatten(factorized_corr[x][:,m1,0:m1,:],start_dim=1),
                                          tr.flatten(factorized_corr[x][:,0:m1,m1,:],start_dim=1)), dim=1)
            else:
                spliced_measurements = tr.cat((tr.flatten(factorized_corr[x][:,m1,0:m1,:],start_dim=1),
                                          tr.flatten(factorized_corr[x][:,m1,m1+1:,:],start_dim=1),
                                          tr.flatten(factorized_corr[x][:,0:m1,m1,:],start_dim=1),
                                          tr.flatten(factorized_corr[x][:,m1+1:,m1,:],start_dim=1)),dim=1)
 
            spliced_avg = tr.real(tr.mean(spliced_measurements, dim=1))

            #corr_level_bias = corr1[x][:, m1, :] - spliced_avg.unsqueeze(1).repeat(1, tr.numel(corr1[x][:, m1, :]))

            fc_splicing_bias[:, m1, x] = c_avg1[:,m1, x] - spliced_avg

        
        #Average the spliced bias measurement
        fc_splicing_bias_avg[x] = tr.real(tr.mean(fc_splicing_bias[:,:,x]))
        fc_splicing_bias_err[x] = tr.std(tr.real(fc_splicing_bias[:,:,x]))/np.sqrt(tr.numel(fc_splicing_bias[:,:,x])-1)

    #Adjust the global fc measurement by the bias

    adj_fc_avg = tr.zeros(len(factorized_corr))
    adj_fc_err = tr.zeros(len(factorized_corr))

    for x in np.arange(bw, len(factorized_corr)):
        adj_fc_avg[x] = global_fc_avg[x] + fc_splicing_bias_avg[x]
        #adj_fc_err[x] = np.sqrt(global_fc_err[x]**2 + fc_splicing_bias_err[x]**2)
        adj_fc_err[x] = global_fc_err[x]

    
    #Save data
    data_write = tr.stack((tr.arange(1, int(L/2)+1), c_avg0, c_err0, global_fc_avg, global_fc_err, fc_splicing_bias_avg, fc_splicing_bias_err), dim = 0).numpy()

    np.savetxt('2_level_factorized_data.csv', data_write, delimiter=',')
    
    #Plot Data

    fig, (ax, ax1, ax2) = plt.subplots(3,1, sharex=True, constrained_layout=True)
    ax.set_yscale('log', nonpositive='clip')
    ax1.set_yscale('log', nonpositive='clip')
    ax2.set_yscale('log', nonpositive = 'clip')
    ax.sharey(ax1)
    ax2.set_ylabel('StN')
    ax.set_title(title1)

 

    #Fit the effective mass curves
    #Note: adds one to the arranged timeslice length to account for indexing by correlator length
    popt, pcov = sp.optimize.curve_fit(f_pi_decay, np.arange(bw+3, len(factorized_corr)-3)+1, tr.abs(c_avg0[bw+3:-3]), sigma = tr.abs(c_err0[bw+3:-3]))
    print("First level fit")
    print(popt)
    print(pcov)

    popt2, pcov2 = sp.optimize.curve_fit(f_pi_decay, np.arange(bw+3, len(factorized_corr)-3)+1, tr.abs(adj_fc_avg[bw+3:-3]), sigma = tr.abs(adj_fc_err[bw+3:-3]))
    print("Second level fit")
    print(popt2)
    print(pcov2)    


    ax.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(c_avg0[bw:]), c_err0[bw:], ls='', marker='.', label="First level measurement",
                fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)
    ax1.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(fc_splicing_bias_avg[bw:]), fc_splicing_bias_err[bw:], ls='', marker='.', label="2-lvl Bias Correction",
                 fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)
    ax1.errorbar(tr.arange(bw+1, int(L/2)+1), tr.abs(adj_fc_avg[bw:]), adj_fc_err[bw:], ls='', marker='.', label="Second level Measurement",
                 fmt='o-', linewidth=1.5, elinewidth=1.2, capsize=3, markersize=4)

    ax.plot(np.linspace(bw+1, len(factorized_corr), 100), f_pi_decay(np.linspace(bw+1, len(factorized_corr), 100), *popt), label="First level fit")
    ax1.plot(np.linspace(bw+1, len(factorized_corr), 100), f_pi_decay(np.linspace(bw+1, len(factorized_corr), 100), *popt2), label="Second Level fit")

    ax2.plot(tr.arange(bw+1, int(L/2)+1), tr.abs(c_avg0[bw:])/c_err0[bw:], label="First level measurement")
    ax2.plot(tr.arange(bw+1, int(L/2)+1), tr.abs(fc_splicing_bias_avg[bw:])/fc_splicing_bias_err[bw:] , label="Bias Correction")
    ax2.plot(tr.arange(bw+1, int(L/2)+1), tr.abs(adj_fc_avg[bw:])/adj_fc_err[bw:], label="Second level measurement")
    ax.legend(loc="upper left",bbox_to_anchor=(1.02, 1),borderaxespad=0)
    ax1.legend(loc="upper left",bbox_to_anchor=(1.02, 1),borderaxespad=0)
    ax2.legend(loc="upper left",bbox_to_anchor=(1.02, 1),borderaxespad=0)

    ax.grid(which='major', linestyle='--', alpha=0.6)
    ax1.grid(which='major', linestyle='--', alpha=0.6)
    ax2.grid(which='major', linestyle='--', alpha=0.6)

    ax.set_ylabel('Magnitude')
    ax1.set_ylabel('Magnitude')
    ax2.set_ylabel('StN')

    
    ax2.set_xlabel(r'$|x_0 - y_0|$')

    plt.savefig(
    "figure.pdf",
    bbox_inches="tight"
    )

    plt.show()



    


    


    

def d_Ddag_Spectral_Radius():
    batch_size= 100
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 16
    L2 = 16
    pm = 2.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 6
    xcut_2 = 14
    bw=2

    s_range= (3,)


    u = sch.hotStart()

    q = (u,)

    d = sch.diracOperator(q[0]).to_dense()
    d_dag = tr.conj(d.transpose(dim0=1,dim1=2))
    dd = tr.einsum('bxy, byz->bxz', d, d_dag)*1.0/7.0
    eye = tr.eye(dd.size(dim=1))
    for b in np.arange(d.size(dim=0)):
        print(tr.max(tr.abs(tr.linalg.eigvals(eye -dd[b, :,:]))))



def test_Naive_Localized_Fermion_Action():
    batch_size= 5
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 10
    L2 = 10
    pm = 2.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 3
    xcut_2 = 8
    bw=2

    s_range= (1,)


    u = sch.hotStart()

    #q = (u,)

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    q = (u, f, d)

     #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    q = sim.evolve_f(q, 30)
    steps = 20
    for n in np.arange(0, steps):
        #Evolve, one HMC step
        q = sim.evolve_f(q, 1)
        sf = sch.fermionAction(q[2], q[1])
        sf_prime = sch.localized_Fermion_Action(q[2],q[1])
        print(sf)
        print(sf_prime)


#Testing to see if the dynamical two level integrator runs properly.
#Just checking config generation, no measurements yet
#TODO: Reweighting and measurements
def dynamical_Twolvl_Integrator():
    batch_size= 2
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 10
    L2 = 10
    pm = 1.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 3
    xcut_2 = 8
    bw=2

    s_range= (3,)


    u = sch.hotStart()
    d = sch.diracOperator(u)
    f = sch.generate_Pseudofermions(d)

    q = (u,f,d)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    lvl2_im2 = i.minnorm2(sch.dd_Force,sch.evolveQ,30, 1.0)
    sim = h.hmc(sch, im2, False)
    lvl2_sim = h.hmc(sch, lvl2_im2, True)

    #Typical equilibration
    q0 = sim.evolve_f(q, 50)
    q = tuple(q0)
    print("Thermalized")


    # Two level integration
    #Level 0 configurations per batch
    n0 = 10

    #Level 1 configurations per batch
    n1 = 10
    for n in np.arange(n0):
        #Thermalize several steps between 2nd level integration
        q = sim.evolve_f(q, 5)
        q1 = tuple(q)

        #To store 2nd-level gauge configs for ensemble averaging
        #batch, config no., lattice indexes
        #u_lvl2 = tr.zeros([batch_size, n1, 2, L, L2], dtype=tr.complex64)
        print(sch.action(q))
        for nx in np.arange(n1):
            #For each subdomain:
            # Evolve locally while maintaining boundaries
            # Update both subdomains for now
            #Check how many pass
            q = lvl2_sim.second_Level_Evolve(q, 5, xcut_1, xcut_2, bw)
        q = tuple(q1)
        print(n)
        

#Produces level. zero configurations generated and measures spectral radius
# of non-local behavior
def measure_Nonlocal_Spectral_Radius():
    batch_size= 5
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 10
    L2 = 10
    pm = 2.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 3
    xcut_2 = 8
    bw=2



    u = sch.hotStart()
    d = sch.diracOperator(u)
    f = sch.generate_Pseudofermions(d)

    q = (u,f,d)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    #Typical equilibration
    q = sim.evolve_f(q, 5)
    print("Thermalized")

    measurements=60
    
    for n in np.arange(measurements):
        q = sim.evolve_f(q,5)
        #Produce non-local matrix of interest
        bb_d = sch.bb_DiracOperator(q, xcut_1, xcut_2, bw).to_dense()
        d00 = bb_d[:,0:4*bw*L2, 0:4*bw*L2]
        d01 = bb_d[:, 0:4*bw*L2, 4*bw*L2:]
        d10 = bb_d[:, 4*bw*L2:, 0:4*bw*L2]
        d11 = bb_d[:, 4*bw*L2:, 4*bw*L2:]

        nlcl = tr.einsum("bij, bjk, bkl, blm->bim", tr.inverse(d11), d10, tr.inverse(d00),
                         d01)
        id = tr.eye(d11.size(dim=1)).reshape(1, d11.size(dim=1),d11.size(dim=1))
        bt_id = id.repeat(batch_size, 1, 1)

        #ms = bt_id - nlcl
        ms = nlcl
        L, V = tr.linalg.eig(ms)
        
        if n==0:
            eigenvalues = L.flatten()
        else:
            eigenvalues = tr.cat([eigenvalues, L.flatten()])

        print(n)

    df = pd.DataFrame(eigenvalues.detach().numpy())
    #TODO: Write more descriptive datafile name
    df.to_csv("nonlocal_matrix_eigenvalues.csv", index = False)

    fig, ax = plt.subplots(1,1)

    ax.scatter(eigenvalues.real, eigenvalues.imag, c='red', marker='.')
    
    theta = np.linspace(0, 2*np.pi, 10000)
    ax.plot(np.cos(theta), np.sin(theta), c='blue')

    ax.set_title("Eigenvalues of the Non-locality Matrix, n=300")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5,1.5)


    plt.show()

        
def test_tridiagonal():
    batch_size= 5
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= -0.08*lam
    L = 10
    L2 = 10
    pm = 2.0
    p= pm*(2.0*np.pi/L)
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 3
    xcut_2 = 8
    bw=2
    

    u = sch.hotStart()
    d = sch.diracOperator(u)
    f = sch.generate_Pseudofermions(d)

    q = (u,f,d)

    bb_d = sch.bb_DiracOperator(q, xcut_1, xcut_2, bw).to_dense()
    d00 = bb_d[:,0:4*bw*L2, 0:4*bw*L2]
    d01 = bb_d[:, 0:4*bw*L2, 4*bw*L2:]
    d10 = bb_d[:, 4*bw*L2:, 0:4*bw*L2]
    d11 = bb_d[:, 4*bw*L2:, 4*bw*L2:]

    correction = tr.einsum('bij, bjk, bkm->bim', d10, tr.inverse(d00), d01)
    tri_correction = sch.schur_Tridiagonal(correction)


    fig, ax = plt.subplots(1,1)

    ax.spy(d11.to_dense()[0,:,:])
    plt.show()
    




def main():
    #test_tridiagonal()
    #measure_Nonlocal_Spectral_Radius()
    #dynamical_Twolvl_Integrator()
    #d_Ddag_Spectral_Radius()
    #test_Naive_Localized_Fermion_Action()
    #block_Diagonal_Check()
    #propogator_Comparison()
    #dirac_Operator_Norm()
    #quenched_two_point_Comparison()
    #plot_Two_Point_Difference()
    #fit_Exact_and_Approx()
    #two_Point_Decay_Comparison()
    #approx_Propogator_Testing()
    #compute_Propogator_Correction()
    #compute_Pion_Mass_Correction()
    #import_correction_fit()
    #plot_correlator_difference()
    #dd_Integrator_dH()
    #dd_Integrated_Action()
    #plot_Signal_To_Noise()
    #config_Correlation()
    #autocorrelation_Comparison()
    #correlator_Dist()
    #factorized_Observable_Systematic_Error_Test()
    #factorized_observable()
    #quenched_Global_Twolvl_Observable()
    #test_Factorized_Propogator()
    #factorized_Pion_Comparison()
    #plot_Factorized_Correlator_Data()
    #compare_Factorizations()
    #test_Factorized_Measurement_Scan()
    #test_Factorized_Pion_Correlator()
    #test_Exact_Schur_Correlator()
    #two_Level_Factorized_Pion()
    two_Level_Stream_Pion_Bias()
    #two_Level_Quenched_Pion_Mass()



main()