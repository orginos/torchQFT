import torch as tr
version = "1.0"
author  = "Kostas Orginos"

# I am assuming batched tensor with the group indices at the end

#levi_civita tensor
import itertools

def levi_civita_tensor(N):
    perms = list(itertools.permutations(range(N)))
    # Create a tensor to store the Levi-Civita symbols
    levi_civita = tr.zeros(N*[N])

    for p in perms:
        sign = 1.0
        # Count the number of inversions to determine the sign
        for j in range(N):
            for k in range(j + 1, N):
                if p[j] > p[k]:
                    sign *= -1.0
        levi_civita[p] = sign

    return levi_civita



#adjoint action
def ad(x,y):
    return x@y- y@x

def LieProject(X):
    return 0.5*(X-X.transpose(-1,-2))
    
# not needed for SO(N) groups
# but useful for SU(N) groups
def LieProjectCmplx(X):
    N = X.shape[-1]
    Y =  0.5*(X-X.transpose(-1,-2).conj())
    T = (tr.einsum('...aa->...',Y)/N).unsqueeze(-1).unsqueeze(-1)
    
    return Y - tr.eye(N).expand(*(X.shape[:-2]),-1,-1)*T

#naitive torch matrix exponential (Taylor series)
def expo(X):
    #print("This is matrix_exp")
    return tr.matrix_exp(X)

#differential of the exponential map
def dexpo(x,y,Ntaylor=20):
    #horner scheme
    r = (-1)**(Ntaylor%2)/tr.math.factorial(Ntaylor+1)*y
    for k in range(Ntaylor-1,0,-1):
        #print(k,r)
        r = ad(x,r) + (-1)**(k%2)/tr.math.factorial(k+1)*y
    #print(k,r)
    r = ad(x,r) + y
    #print(k,r)
    return r


from . import so3
from . import su2

    
