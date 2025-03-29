import torch as tr
import numpy as np

from . import * 

L  = tr.tensor([[[  0.0,  0,  0],
                 [  0,  0, -1],
                 [  0,  1,  0]],
                [[  0,  0,  1],
                 [  0,  0,  0],
                 [ -1,  0,  0]],
                [[  0, -1,  0],
                 [  1,  0,  0],
                 [ 0,  0,  0]]])

levi_civita=levi_civita_tensor(3)

def check_algebra():
    foo = tr.einsum('aik,bkj->abij',L,L)
    foo = foo - foo.transpose(0,1)
    boo = tr.einsum('abk,kij->abij',levi_civita,L)
    print("If zero the so3 algebra checks: ",((foo-boo).norm()/boo.norm()).numpy())


#SO(3) specific exponentiation
def expo(X):
    #print("This is so3 expo:")
    nX = (X.norm(dim=(-1,-2))/np.sqrt(2)).unsqueeze(-1).unsqueeze(-1)+tr.finfo(X.dtype).eps
    return tr.eye(3).expand(*(X.shape[:-2]),-1,-1) + tr.sin(nX)/nX*X + (1-tr.cos(nX))/(nX*nX)*(X@X)
    

#this is non-smooth so it have issues with autodiff
def expo_old(X):
    #print("This is so3 expo:")
    nX = (X.norm(dim=(-1,-2))/np.sqrt(2)).unsqueeze(-1).unsqueeze(-1)
    R = tr.eye(3).expand(*(X.shape[:-2]),-1,-1) + tr.sin(nX)/nX*X + (1-tr.cos(nX))/(nX*nX)*(X@X)
    return  tr.where(nX<tr.finfo(X.dtype).eps,tr.eye(3).expand(*(X.shape[:-2]),-1,-1),R)

#SO(3) specific differential of the exponential map
def dexpo(X,Y):
    nX = (X.norm(dim=(-1,-2))/np.sqrt(2)).unsqueeze(-1).unsqueeze(-1)+tr.finfo(X.dtype).eps
    adj = ad(X,Y)
    nX2= nX**2
    return Y - (1-tr.cos(nX))/(nX2)*adj + (1 - tr.sin(nX)/nX)/(nX2) * ad(X,adj)

# this is non-smooth so it have issues with autodiff
def dexpo_old(X,Y):
    nX = (X.norm(dim=(-1,-2))/np.sqrt(2)).unsqueeze(-1).unsqueeze(-1)
    adj = ad(X,Y)
    nX2= nX**2
    return tr.where(nX<tr.finfo(X.dtype).eps,Y.clone(),Y - (1-tr.cos(nX))/(nX2)*adj + (1 - tr.sin(nX)/nX)/(nX2) * ad(X,adj))

def check_expo_and_dexpo():
    from . import expo as taylor_expo
    from . import dexpo as taylor_dexpo
    X = LieProject(tr.randn(4,2,2,3,3))
    Y = LieProject(tr.randn(4,2,2,3,3))
    expoX = expo(X)
    t_expoX = taylor_expo(X)
    tt = ((expoX-t_expoX).norm()/expoX.norm())
    print("If zero expo works: ",tt.numpy())
    dexpoXY = dexpo(X,Y)
    dexpoXYiter = taylor_dexpo(X,Y)
    tt=(dexpoXYiter - dexpoXY).norm()/dexpoXYiter.norm()
    print("If zero dexpo works:",tt.numpy())

    expoX_old = expo_old(X)
    ((expoX-expoX_old).norm()/expoX.norm())
    print("If check old vs new expo: ",tt.numpy())
    dexpoXY_old = dexpo_old(X,Y)
    tt=(dexpoXY_old - dexpoXY).norm()/dexpoXYiter.norm()
    print("If check old vs new dexpo: ",tt.numpy())
