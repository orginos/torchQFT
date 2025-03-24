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
    dd = sch.bb_DiracOperator(q, 2, 7)
    plt.spy(dd.to_dense()[0,:,:])

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

    s_inv = sch.dd_Factorized_Propogator(q, 3, 8)

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
    q = sim.evolve_f(q, 25, False)

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
    q = sim.evolve_f(q, 25, True)

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

    s_inv = sch.dd_Factorized_Propogator(q, 3, 8)

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
    L = 32
    L2 = 32
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 12
    xcut_2 = 28
    bw=4
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
    q = sim.evolve_f(q, 25, False)


    #Measurement process- Compare nm measurements on batches
    nm = 10
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10, False)
        d = sch.diracOperator(q[0])
        bb_d = sch.bb_DiracOperator(q, xcut_1, xcut_2, bw)
        d_inv = tr.linalg.inv(d.to_dense())

            
        #Vector of time slice correlations
        cl = sch.exact_Two_Point_Correlator(d_inv, np.concatenate((np.arange(4,8), np.arange(20,24))))
        cl2 = sch.dd_Two_Point_Correlator(bb_d, xcut_1, xcut_2, bw, r, np.concatenate((np.arange(4,8), np.arange(20,24))))
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

#Plot 2 pt functions, difference, and error differences to assess approximation
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
    #Approximation rank
    r=2


    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    q = (u, f, d)

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,5, 1.0)
    sim = h.hmc(sch, im2, False)
    
    #Does it run?
    fp = sch.dd_Approx_Propogator(q, xcut_1, xcut_2, r, 0)
    sch.dd_Two_Point_Correlator(fp, xcut_1, xcut_2, 0)

    #Equilibration
    q = sim.evolve_f(q, 25, True)



    #Measurement process- Compare nm measurements on batches
    nm = 1
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10, True)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())
        fp = sch.dd_Approx_Propogator(q, xcut_1, xcut_2, r, 0)

            
        #Vector of time slice correlations
        cl = sch.exact_Two_Point_correlator(d_inv)
        cl2 = sch.dd_Two_Point_Correlator(fp, xcut_1, xcut_2, 0)
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

    s_range = np.concatenate((np.arange(0,14), np.arange(16,30)))


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    print("Approx. calculation: ")
    #Equilibration
    q = sim.evolve_f(q, 25, False)

    #Measurement process- Compare nm measurements on batches
    nm = 10
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10, False)
        bb_d = sch.bb_DiracOperator(q, xcut_1, xcut_2, bw)

            
        #Vector of time slice correlations
        cl = sch.dd_Two_Point_Correlator(bb_d, xcut_1, xcut_2, bw, r, s_range)
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
    q = sim.evolve_f(q, 25, False)

    #Measurement process- Compare nm measurements on batches
    nm = 10
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10, False)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())
            
        #Vector of time slice correlations
        cl = sch.two_Point_correlator(q[2], s_range)

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
    #First we compute the pion mass observable using the approximation scheme:
    batch_size= 100
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.0
    L = 16
    L2 = 16
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 7
    xcut_2 = 14
    #Neumann Approximation rank
    r=1


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    print("Approx. calculation: ")
    #Equilibration
    q = sim.evolve_f(q, 25, False)

    #Measurement process- Compare nm measurements on batches
    nm = 10
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10, False)
        fp = sch.dd_Approx_Propogator(q, xcut_1, xcut_2, r, 0)

            
        #Vector of time slice correlations
        cl = sch.dd_Two_Point_Correlator(fp, xcut_1, xcut_2)
        if n ==0:
            c= cl
        else:
            c = tr.cat((c, cl),0)
        print(n)

    #Now Compute the Exact observable
    batch_size= 10
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.0
    L = 16
    L2 = 16
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    print("Exact calculation: ")
    #Equilibration
    q = sim.evolve_f(q, 25, False)

    #Measurement process- Compare nm measurements on batches
    nm = 10
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10, False)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())
            
        #Vector of time slice correlations
        cl = sch.exact_Two_Point_correlator(d_inv)

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


    #Now fit the effective mass curves and report measurement

    #Fit the effective mass curve for the exact computation and approximation
    #Select only the time slices near the center of the lattice
    approx_popt, approx_pcov = sp.optimize.curve_fit(f_pi_triplet, np.concatenate((np.arange(3,7),np.arange(9,12))), np.concatenate((np.abs(df.iloc[0, 3:7]), np.abs(df.iloc[0, 9:12]))), sigma = np.concatenate((np.abs(df.iloc[1, 3:7]), np.abs(df.iloc[1, 9:12]))))
    print("Approximation:")
    print(approx_popt)
    print(approx_pcov)

    exact_popt, exact_pcov = sp.optimize.curve_fit(f_pi_triplet, np.concatenate((np.arange(3,7),np.arange(9,12))), np.concatenate((np.abs(df.iloc[2, 3:7]), np.abs(df.iloc[2, 9:12]))), sigma = np.concatenate((np.abs(df.iloc[3, 3:7]), np.abs(df.iloc[3, 9:12]))))
    print("Exact")
    print(exact_popt)
    print(exact_pcov)

    #Plot 
    fig, ax1 = plt.subplots(1,1)

    ax1.plot(np.linspace(0, 16, 100), f_pi_triplet(np.linspace(0, 16, 100), *exact_popt), 'b')
    ax1.errorbar(np.arange(0, 16), np.abs(df.iloc[2, :]), np.abs(df.iloc[3, :]), ls="", fmt="b.", label="Exact")

    ax1.plot(np.linspace(0, 16, 100), f_pi_triplet(np.linspace(0, 16, 100), *approx_popt), 'r')
    ax1.errorbar(np.arange(0, 16), np.abs(df.iloc[0, :]), np.abs(df.iloc[1, :]), ls="", fmt="r.", label="Approx.")

    ax1.legend()

    plt.show()
    
#Testing implementation of multilevel integrator- quenched
#TODO: In Development
def multilevel_Integrator():
    batch_size= 100
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.0
    L = 16
    L2 = 16
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 7
    xcut_2 = 14
    #Neumann Approximation rank
    r=1


    u = sch.hotStart()

    q = (u,)

    im2 = i.minnorm2(sch.force,sch.evolveQ,20, 1.0)
    sim = h.hmc(sch, im2, False)

    #Typical equilibration
    q = sim.evolve_f(q, 25, False)

    #Level 0 configurations per batch
    n0 = 10

    for i in np.arange(n0):
        #seperate the subdomains- keep immediate edge of boundary regions
        u0 = tr.cat((q[0][:, xcut_2+1, :], q[0][:, 0:xcut_1+1, :]), dim=1)
        u1 = q[0][:, xcut_1 +2:xcut_2+1, :] 
        q0 = (u0,)
        q1 = (u1,)

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



def main():
    #block_Diagonal_Check()
    #propogator_Comparison()
    #dirac_Operator_Norm()
    #quenched_two_point_Comparison()
    #plot_Two_Point_Difference()
    #fit_Exact_and_Approx()
    #two_Point_Decay_Comparison()
    #approx_Propogator_Testing()
    #compute_Propogator_Correction()
    plot_correlator_difference()



main()