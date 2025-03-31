import torch as tr
import numpy as np

from . import *

T   = 0.5j*tr.tensor([[[  0,  1,  0],
                       [  1,  0,  0],
                       [  0,  0,  0]],
                      [[  0, 1j,  0],
                       [-1j,  0,  0],
                       [  0,  0,  0]],
                      [[  1,  0,  0],
                       [  0, -1,  0],
                       [  0,  0,  0]],
                      [[  0,  0,  1],
                       [  0,  0,  0],
                       [  1,  0,  0]],
                      [[  0,  0,-1j],
                       [  0,  0,  0],
                       [ 1j,  0,  0]],
                      [[  0,  0,  0],
                       [  0,  0,  1],
                       [  0,  1,  0]],
                      [[  0,  0,  0],
                       [  0,  0,-1j],
                       [  0, 1j,  0]],
                      [[  1,  0,  0],
                       [  0,  1,  0],
                       [  0,  0, -2]]
                      ])

T[7] *= 1.0/np.sqrt(3.0)
def calc_structure():
    foo = tr.einsum('aik,bkj->abij',T,T)
    comm = foo - foo.transpose(0,1)
    acom = foo + foo.transpose(0,1)

    f = -2.0*tr.einsum('aik,bcki->abc',T,comm)
    d = (2.0*1j)*tr.einsum('aik,bcki->abc',T,acom)
    return f,d

f_struc,d_struc = calc_structure()

def check_algebra():
    foo = tr.einsum('aik,bkj->abij',T,T)
    comm = foo - foo.transpose(0,1)
    acom = foo + foo.transpose(0,1)
    
   
    print("Normalization checks if zero: ",(tr.einsum('aij,bji->ab',T,T)+0.5*tr.eye(8)).norm()/8.0)
    boo = tr.einsum('abc,cij->abij',f_struc,T)
    print("If zero the su3 algebra checks: ",((comm-boo).norm()/boo.norm()).numpy())
    boo = tr.einsum('abc,cij->abij',1j*d_struc,T)
    eye = tr.eye(8).unsqueeze(-1).unsqueeze(-1)*tr.eye(3).unsqueeze(0).unsqueeze(0)
    diff = acom-boo+(1./3.)*eye 
    print("If zero d-structure is correct: ",diff.norm().numpy()/8.0)

def simpsons_rule(fx: tr.Tensor, h: tr.float) -> tr.Tensor:
    if fx.shape[-1] % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points.")
    
    S = fx[..., 0] + fx[..., -1]  # First and last terms
    S += 4 * fx[..., 1:-1:2].sum(dim=-1)  # Odd indices
    S += 2 * fx[..., 2:-2:2].sum(dim=-1)  # Even indices (excluding first and last)
    
    return (h / 3) * S

def expo(X):
    nX = X.norm(dim=(-1,-2),keepdim=True)/ np.sqrt(2)
    Xhat =X/(nX+tr.finfo(X.dtype).eps)
    
    #foo  = tr.einsum('...ij,...ji->...',Xhat,Xhat)
    #print("Trace X*X: ",foo[0,0,0].numpy())

    eta = tr.linalg.det(Xhat).unsqueeze(-1).unsqueeze(-1)
    #S3 = tr.einsum('...ik,...kl,...lj->...ij',Xhat,Xhat,Xhat)
    #
    #Xhat^3 + Xhat - eta*1  = 0 is the characteristic polynomium
    #diff = S3 +Xhat - eta*tr.eye(3).expand_as(X)
    #print(diff.norm())
    psi = (tr.acos(-tr.imag(eta)*3*np.sqrt(3)/2.0)/3.0).squeeze()
    z=tr.zeros(X.shape[0:-1],dtype=X.dtype)
    #print(z.shape,psi.shape)
    
    z[...,0] = 2.0/np.sqrt(3.0)*tr.cos(psi)*1j
    z[...,1] = (-tr.sin(psi) - 1.0/np.sqrt(3)*tr.cos(psi)) *1j
    z[...,2] = ( tr.sin(psi) - 1.0/np.sqrt(3)*tr.cos(psi)) *1j
    diff = z**3+z - eta.squeeze(-1)
    #print("check if roots are correct: ",diff.norm().numpy())
    z = z.unsqueeze(-1).unsqueeze(-1)
    Xhat = Xhat.unsqueeze(-3)
    nX = nX.unsqueeze(-3)
    #print(z.shape,X.shape,nX.shape)
    R = tr.exp(z*nX)/(3*z*z+1 + tr.finfo(X.dtype).eps)*(tr.eye(3).expand_as(Xhat)*(z*z+1) + z*Xhat +Xhat@Xhat)
    
    return R.sum(dim=(-3))
    
      

def check_expo_and_dexpo():
    from . import expo as taylor_expo
    from . import dexpo as taylor_dexpo
    X = LieProjectCmplx(tr.randn(4,2,2,3,3)+1j*tr.randn(4,2,2,3,3))
    Y = LieProjectCmplx(tr.randn(4,2,2,3,3)+1j*tr.randn(4,2,2,3,3))
    
    expoX = expo(X)
    t_expoX = taylor_expo(X)
    tt = ((expoX-t_expoX).norm()/expoX.norm())
    print("If zero expo works: ",tt.numpy())
    #dexpoXY = dexpo(X,Y)
    #dexpoXYiter = taylor_dexpo(X,Y)
    #tt=(dexpoXYiter - dexpoXY).norm()/dexpoXYiter.norm()
    #print("If zero dexpo works:",tt.numpy())

    
