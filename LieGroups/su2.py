import torch as tr
import numpy as np

from . import *

tau = 0.5j*tr.tensor([[[0,1],
                       [1,0]],
                      [[  0, 1j],
                       [-1j,  0]],
                      [[1, 0],
                       [0,-1]]])

levi_civita=levi_civita_tensor(3).to(tau.dtype)

def check_algebra():
    foo = tr.einsum('aik,bkj->abij',tau,tau)
    foo = foo - foo.transpose(0,1)
    boo = tr.einsum('abk,kij->abij',levi_civita,tau)
    print("If zero the su2 algebra checks: ",((foo-boo).norm()/boo.norm()).numpy())

def expo(X):
    nn = (tr.norm(X,dim=(-1,-2),keepdim=True)/np.sqrt(2)+tr.finfo(X.dtype).eps)
    return tr.cos(nn)*tr.eye(2, dtype=X.dtype, device=X.device).expand_as(X) + X*(tr.sin(nn)/nn) 

def dexpo(X,Y):
    nX = (X.norm(dim=(-1,-2),keepdim=True)/np.sqrt(2))+tr.finfo(X.dtype).eps
    adj = ad(X,Y)
    two_nX = 2.0*nX
    sin_two_nX_over_two_nX = tr.sin(two_nX)/two_nX 

    return Y - (0.5*(tr.sin(nX)/nX)**2)*adj + 0.25*(1.0-sin_two_nX_over_two_nX)/(nX*nX)*ad(X,adj)


def simpsons_rule(fx: tr.Tensor, h: tr.float) -> tr.Tensor:
    if fx.shape[-1] % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points.")
    
    S = fx[..., 0] + fx[..., -1]  # First and last terms
    S += 4 * fx[..., 1:-1:2].sum(dim=-1)  # Odd indices
    S += 2 * fx[..., 2:-2:2].sum(dim=-1)  # Even indices (excluding first and last)
    
    return (h / 3) * S

def dexpo_step_by_step(X,Y,N=100):
    def cos2(a):
        return tr.cos(a)**2
    def sin2(a):
        return tr.sin(a)**2
    def sincos(a):
        return tr.sin(a)*tr.cos(a)

    nX = (X.norm(dim=(-1,-2),keepdim=True)/np.sqrt(2))+tr.finfo(X.dtype).eps
    a = tr.linspace(0,1,2*N+1)# ensure odd number of points
    anX = a.view([1]*len(nX.shape)+list(a.shape))*nX.unsqueeze(-1)
    h = anX[...,1]-anX[...,0]
    #term1 = (simpsons_rule(tr.cos(anX)**2,h)/nX)*Y
    #term1 = 0.5*(1.0+tr.sin(nX)/nX*tr.cos(nX))*Y
    #term2 = (simpsons_rule(tr.cos(anX)*tr.sin(anX),h)/(nX*nX))*ad(X,Y)
    term2 = 0.5*(tr.sin(nX)/nX)**2*ad(X,Y)
    #term3 = (simpsons_rule(tr.sin(anX)**2,h)/(nX**3))*(-X@Y@X)
    #term3 = 0.5*(1.0-tr.sin(nX)/nX*tr.cos(nX))/(nX*nX)*(-X@Y@X)
    #minusXYX = 0.5*(ad(X,ad(X,Y)) -X@X@Y-Y@X@X)
    #print((X@X +tr.eye(2).expand_as(X)*nX*nX).norm()/(nX*nX).norm())
    #minusXYX = 0.5*ad(X,ad(X,Y)) + nX**2*Y 
    #term3 = 0.5*(1.0-tr.sin(nX)/nX*tr.cos(nX))/(nX*nX)*minusXYX
    #R = term1-term2+term3
    #rearange
    term1 = Y
    term3 =  0.25*(1.0-tr.sin(nX)/nX*tr.cos(nX))/(nX*nX)*ad(X,ad(X,Y))
    return term1-term2+term3
    

def check_expo_and_dexpo():
    from . import expo as taylor_expo
    from . import dexpo as taylor_dexpo
    X = LieProjectCmplx(tr.randn(4,2,2,2,2)+1j*tr.randn(4,2,2,2,2))
    Y = LieProjectCmplx(tr.randn(4,2,2,2,2)+1j*tr.randn(4,2,2,2,2))
    expoX = expo(X)
    t_expoX = taylor_expo(X)
    tt = ((expoX-t_expoX).norm()/expoX.norm())
    print("If zero expo works: ",tt.numpy())
    dexpoXY = dexpo(X,Y)
    dexpoXYiter = taylor_dexpo(X,Y)
    tt=(dexpoXYiter - dexpoXY).norm()/dexpoXYiter.norm()
    print("If zero dexpo works:",tt.numpy())

    
