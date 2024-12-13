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
    dd = sch.bb_DiracOperator(q, 3, 8)
    plt.spy(dd.to_dense()[0,:,:])

    plt.show()

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

def pion_Decay_Comparison():
    batch_size=30
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.0
    L = 10
    L2 = 10
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 3
    xcut_2 = 8


    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    q = (u, f, d)

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,10, 1.0)
    sim = h.hmc(sch, im2, True)
    
    #Does it run?
    fp = sch.dd_Factorized_Propogator(q, xcut_1, xcut_2)
    sch.dd_Pi_Plus_Correlator(fp, xcut_1, xcut_2)

    #Equilibration
    q = sim.evolve_f(q, 25, True)



    #Measurement process- Compare nm measurements on batches
    nm = 10
    for n in np.arange(nm):
        #Discard some in between
        q= sim.evolve_f(q, 10, True)
        d = sch.diracOperator(q[0])
        d_inv = tr.linalg.inv(d.to_dense())
        fp = sch.dd_Factorized_Propogator(q, xcut_1, xcut_2)

            
        #Vector of time slice correlations
        cl = sch.pi_plus_correlator(d_inv)
        cl2 = sch.dd_Pi_Plus_Correlator(fp, xcut_1, xcut_2)
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

def approx_Propogator_Testing():
    batch_size=30
    lam = np.sqrt(1.0/10.0)
    #Below is bare mass
    mass= 0.0
    L = 10
    L2 = 10
    sch = s.schwinger([L,L2],lam,mass,batch_size=batch_size)

    #Boundary cut timeslices
    xcut_1 = 3
    xcut_2 = 8


    u = sch.hotStart()

    d = sch.diracOperator(u)

    f = sch.generate_Pseudofermions(d)

    q = (u, f, d)

    #Tune integrator to desired step size
    im2 = i.minnorm2(sch.force,sch.evolveQ,10, 1.0)
    sim = h.hmc(sch, im2, True)
    
    #Runs
    sch.dd_Approx_Propogator(q, 3, 8, 2, 0)



def main():
    #block_Diagonal_Check()
    #propogator_Comparison()
    #pion_Decay_Comparison()
    approx_Propogator_Testing()

main()