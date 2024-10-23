#!/opt/python/torch/bin/python
import numpy as np
import torch as tr
import su2_chain as s
import cmath as c
from  scipy.linalg import expm

import matplotlib.pyplot as plt

lat = [8,8]
beta = 2.0
Nwarm = 400

su2f = s.field(lat,Nbatch=2)
su2c = s.SU2chain(beta=beta,field_type=su2f)

U = su2f.hot()

#F = su2.force(U)
F = su2c.refreshP()
print('Kinetic energy: ',su2c.kinetic(F))
E = su2f.expo(F)
print(E.shape)
#x = int(np.random.uniform(0,1,1)*G.Vol)
#mu =  int(np.random.uniform(0,1,1)*G.Nd)
#print("Checking exponentiation at point x=",x, " and direction mu=", mu)

tt = E[0,0,0]
print("E = \n",tt)
#htt = np.einsum('ij->ji',tt.conj())
#print("E' = \n",htt)

print(' unitarity test:\n  ', tr.einsum('ik,jk->ij',tt,tt.conj()))
print(' unitarity det test: ', tr.sqrt(tt[0,0]*tt[1,1] - tt[1,0]*tt[0,1]))

M = F[0,0,0].numpy()
EE = expm(M)
print("Compare with SciPy matrix exponentiation")

print("SciPy expm")
print(EE)
print("diff from SciPy expm")
print(EE - tt.numpy())
print("The norm of diff: ",np.linalg.norm(EE-tt.numpy()))

        
        
