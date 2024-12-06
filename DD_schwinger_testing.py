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



def main():
    block_Diagonal_Check()
    propogator_Comparison()