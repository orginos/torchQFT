import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm 

import phi4 as qft
import integrators as integ
import update as upd

import matplotlib.pyplot as plt

import os
import time

print("Torch Version: ",tr.__version__)
print("Numpy Version: ",np.__version__)

if tr.cuda.is_available():
    device = tr.device("cuda")
elif tr.backends.mps.is_available():
    device = tr.device("mps")
else:
    print ("CUDA/MPS device not found.")
    device = "cpu"
    
print("Using divice: ",device)

def C2pt(x, tau):
    """Connected two-point correlator.

    C_conn(tau) = ⟨φ(x)φ(x+τ)⟩ - ⟨φ⟩²

    Subtracting the batch mean (ensemble average) before the correlator
    is equivalent to the vacuum subtraction.
    """
    xm = x - tr.mean(x)  # subtract ensemble average
    return tr.mean(xm * tr.roll(xm, shifts=-tau, dims=2), dim=(1,2))

def symmetry_checker(x,model):
    out     = model(x)
    sout    = model(tr.roll(x,shifts=(2,3),dims=(1,2)))
    r90out  = model(tr.rot90(x,k=1,dims=(1,2)))
    r180out = model(tr.rot90(x,k=2,dims=(1,2)))
    f1out   = model(tr.flip(x,dims=[1]))
    f2out   = model(tr.flip(x,dims=[2]))
    pout    = model(-x)
    print("Parity: ",(tr.norm(out-pout)/tr.norm(out)).item())
    print("Translation: ",(tr.norm(out-sout)/tr.norm(out)).item())
    print("Rotation(90): ",(tr.norm(out-r90out)/tr.norm(out)).item())
    print("Rotation(180): ",(tr.norm(out-r180out)/tr.norm(out)).item())
    print("Flip(1): ",(tr.norm(out-f1out)/tr.norm(out)).item())
    print("Flip(2): ",(tr.norm(out-f2out)/tr.norm(out)).item())


def vector_equivariance_checker(x, model):
    """Check equivariance of local vector-field models g(phi)."""
    out = model(x).detach()
    denom = tr.norm(out).clamp_min(1.0e-12)

    sx = tr.roll(x, shifts=(2, 3), dims=(1, 2))
    sout_expected = tr.roll(out, shifts=(2, 3), dims=(1, 2))
    sout = model(sx).detach()

    r90x = tr.rot90(x, k=1, dims=(1, 2))
    r90_expected = tr.rot90(out, k=1, dims=(1, 2))
    r90out = model(r90x).detach()

    r180x = tr.rot90(x, k=2, dims=(1, 2))
    r180_expected = tr.rot90(out, k=2, dims=(1, 2))
    r180out = model(r180x).detach()

    f1x = tr.flip(x, dims=[1])
    f1_expected = tr.flip(out, dims=[1])
    f1out = model(f1x).detach()

    f2x = tr.flip(x, dims=[2])
    f2_expected = tr.flip(out, dims=[2])
    f2out = model(f2x).detach()

    px = -x
    pout = model(px).detach()

    print("g-Translation equivariance: ", (tr.norm(sout - sout_expected) / denom).item())
    print("g-Rotation(90) equivariance: ", (tr.norm(r90out - r90_expected) / denom).item())
    print("g-Rotation(180) equivariance: ", (tr.norm(r180out - r180_expected) / denom).item())
    print("g-Flip(1) equivariance: ", (tr.norm(f1out - f1_expected) / denom).item())
    print("g-Flip(2) equivariance: ", (tr.norm(f2out - f2_expected) / denom).item())
    print("g-Parity oddness: ", (tr.norm(pout + out) / denom).item())
    print("g-Parity evenness: ", (tr.norm(pout - out) / denom).item())


def control_model_F_symmetry_checker(x, control_model, shift=(2, 3), n_colors=None):
    """Check discrete symmetries of the control variate F produced by a control model.

    This is meant for first-order control variates where the base network g0 is not
    itself symmetric, but the constructed F should inherit translation invariance.
    """
    out     = control_model.F(x, n_colors=n_colors).detach()
    sout    = control_model.F(tr.roll(x, shifts=shift, dims=(1, 2)), n_colors=n_colors).detach()
    r90out  = control_model.F(tr.rot90(x, k=1, dims=(1, 2)), n_colors=n_colors).detach()
    r180out = control_model.F(tr.rot90(x, k=2, dims=(1, 2)), n_colors=n_colors).detach()
    f1out   = control_model.F(tr.flip(x, dims=[1]), n_colors=n_colors).detach()
    f2out   = control_model.F(tr.flip(x, dims=[2]), n_colors=n_colors).detach()
    pout    = control_model.F(-x, n_colors=n_colors).detach()

    denom = tr.norm(out).clamp_min(1.0e-12)
    if n_colors is not None:
        print("F n_colors: ", n_colors)
    print("F-Parity: ", (tr.norm(out - pout) / denom).item())
    print("F-Translation: ", (tr.norm(out - sout) / denom).item())
    print("F-Rotation(90): ", (tr.norm(out - r90out) / denom).item())
    print("F-Rotation(180): ", (tr.norm(out - r180out) / denom).item())
    print("F-Flip(1): ", (tr.norm(out - f1out) / denom).item())
    print("F-Flip(2): ", (tr.norm(out - f2out) / denom).item())


#WARNING: FieldTrans and Funct do not seem to work... they do not pass the derivative test
# I do not care to debug because the rest work beautifully
class FieldTrans(nn.Module):
    def __init__(self,L,depth=2,width=10,activation=nn.GELU(),dtype=tr.float): 
        super().__init__()

        self.depth = depth
        self.w = width
        # Define trainable parameters
        self.ai = nn.Parameter(tr.randn(self.w,1))
        self.ao = nn.Parameter(tr.randn(1,self.w))
        self.a  = nn.Parameter(tr.randn(self.depth,self.w,self.w))
        self.bi = nn.Parameter(tr.randn(self.w,1))
        self.bo = nn.Parameter(tr.randn(1,self.w))
        self.b  = nn.Parameter(tr.randn(self.depth,self.w,self.w))
        self.ci = nn.Parameter(tr.randn(self.w,1))
        self.co = nn.Parameter(tr.randn(1,self.w))
        self.c  = nn.Parameter(tr.randn(self.depth,self.w,self.w))
        self.bias_i = nn.Parameter(tr.randn(self.w))
        self.bias = nn.Parameter(tr.randn(self.depth,self.w))
        #self.pool = nn.AvgPool2d(kernel_size=block_size, stride=block_size) 
        self.activation = activation
        
    def func(self, x):
        """Applies the convolution by summing rolled versions of x, weighted by a, b, and c."""
        
        # Center term (c): No shift needed
        #print(self.ci.shape,x.shape)
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2])
        y = tr.einsum('oi,bixy->boxy',self.ci,x)
        y+= tr.einsum('oi,bixy->boxy',self.bi,(tr.roll(x, shifts=1, dims=2) + tr.roll(x, shifts=-1, dims=2)))  # Vertical shift
        y+= tr.einsum('oi,bixy->boxy',self.bi,(tr.roll(x, shifts=1, dims=3) + tr.roll(x, shifts=-1, dims=3)))  # Horizontal shift
        y+= tr.einsum('oi,bixy->boxy',self.ai,(tr.roll(x, shifts=(1, 1), dims=(2,3)) + tr.roll(x, shifts=(-1,-1), dims=(2,3))))
        y+= tr.einsum('oi,bixy->boxy',self.ai,(tr.roll(x, shifts=(1,-1), dims=(2,3)) + tr.roll(x, shifts=(-1, 1), dims=(2,3))))

        x = self.activation(y+self.bias_i.view(1,self.w,1,1))
        
        for k in range(self.depth):
            y = tr.einsum('oi,bixy->boxy',self.c[k],x)
            y+= tr.einsum('oi,bixy->boxy',self.b[k],(tr.roll(x, shifts=1, dims=2) + tr.roll(x, shifts=-1, dims=2)))  # Vertical shift
            y+= tr.einsum('oi,bixy->boxy',self.b[k],(tr.roll(x, shifts=1, dims=3) + tr.roll(x, shifts=-1, dims=3)))  # Horizontal shift
            y+= tr.einsum('oi,bixy->boxy',self.a[k],(tr.roll(x, shifts=(1,1), dims=(2,3)) + tr.roll(x, shifts=(-1,-1), dims=(2,3))))
            y+= tr.einsum('oi,bixy->boxy',self.a[k],(tr.roll(x, shifts=(1,-1), dims=(2,3)) + tr.roll(x, shifts=(-1,1), dims=(2,3))))
        
            x = self.activation(y+self.bias[k].view(1,self.w,1,1))

        y = tr.einsum('oi,bixy->boxy',self.co,x)
        y+= tr.einsum('oi,bixy->boxy',self.bo,(tr.roll(x, shifts=1, dims=2) + tr.roll(x, shifts=-1, dims=2)))  # Vertical shift
        y+= tr.einsum('oi,bixy->boxy',self.bo,(tr.roll(x, shifts=1, dims=3) + tr.roll(x, shifts=-1, dims=3)))  # Horizontal shift
        y+= tr.einsum('oi,bixy->boxy',self.ao,(tr.roll(x, shifts=(1, 1), dims=(2,3)) + tr.roll(x, shifts=(-1,-1), dims=(2,3))))
        y+= tr.einsum('oi,bixy->boxy',self.ao,(tr.roll(x, shifts=(1,-1), dims=(2,3)) + tr.roll(x, shifts=(-1, 1), dims=(2,3))))

        return y.squeeze()

    #enforce parity
    def forward(self,x):
        return 0.5*(self.func(x) - self.func(-x))
     
    
class Funct(nn.Module):
    def __init__(self,dim=2,y=0,FieldTrans=None,dtype=tr.float): 
        super().__init__()
        self.ft  = FieldTrans
        self.y   = y
        self.dim = dim
        self.b  = nn.Parameter(tr.randn(16))
        
    def func(self,x):
        y=self.ft(x)      
        sy = tr.roll(y,shifts=-self.y,dims=self.dim)
        sx = tr.roll(x,shifts=-self.y,dims=self.dim)
        c = 0
        out = tr.zeros(x.shape[0])
        for f in [x,y,sx,sy]:
            for ff in [x,y,sx,sy]:
                out+= self.b[c]*(f*ff).sum()
                c+=1
                
        return out
        
    def forward(self,x):
        fx = tr.flip(x,dims=[self.dim])
        
        return self.func(x)+self.func(fx)

    def grad_and_lapl(self,x):
        y = self.forward(x) 
        #print(y.shape,x.shape)
        grad = tr.autograd.grad(y,x,tr.ones_like(y),create_graph=True)[0]
        #print(grad.shape)
        lapl = tr.zeros_like(y)
        for i in range(x.shape[1]):  # Loop over dim 1
            for j in range(x.shape[2]):  # Loop over dim 2
                foo=tr.autograd.grad(grad[:,i,j],x,tr.ones_like(grad[:,i,j]),create_graph=True)[0]
                lapl+=foo[:,i,j]
        return grad,lapl



class Funct2(nn.Module):
    def __init__(self,dim=2,y=0,conv_layers=[32],dtype=tr.float64,activation=tr.nn.Tanh(),initializer=tr.nn.init.xavier_uniform_): 
        super().__init__()
        self.net  = nn.Sequential()
        self.y   = y
        self.dim = dim
        self.dtype = dtype
        in_dim =3
        k=0
        for l in conv_layers:
            #print(k,in_dim,l)
            layer = nn.Linear(in_dim,l,dtype=self.dtype)
            initializer(layer.weight)
            self.net.add_module('lin'+str(k),layer)
            self.net.add_module('act'+str(k),activation)

            in_dim=l
            k+=1
        layer = tr.nn.Linear(in_dim,1,dtype=self.dtype)
        initializer(layer.weight)
        self.net.add_module('lin'+str(k),layer)
        
        #print(self.net)
 
    def forward(self,x):
        x1  = tr.roll(x, shifts=1, dims=1) + tr.roll(x, shifts=-1, dims=1) 
        x1 += tr.roll(x, shifts=1, dims=2) + tr.roll(x, shifts=-1, dims=2)
        sx = tr.roll(x,shifts=self.y,dims=self.dim) +  tr.roll(x,shifts=-self.y,dims=self.dim)
        t1 = (x *x).unsqueeze(3)
        t2 = (x1*x).unsqueeze(3)
        t3 = (sx*x).unsqueeze(3)
        inp = tr.cat([t1,t2,t3],dim=3).view(np.prod(x.shape),3)
        #print(inp.shape)
        out = self.net(inp).view(x.shape).sum(dim=(1,2))
        
        return out

    def net_grad_and_hess(self,x):
        y = self.net(x)
        g = tr.autograd.grad(y,x,tr.ones_like(y),create_graph=True)[0]
        h=tr.zeros(x.shape[0],x.shape[1],x.shape[1],device=x.device,dtype=x.dtype)
        for j in range(g.shape[1]):
            g2 = tr.autograd.grad(g[:, j], x, tr.ones_like(g[:, j]), create_graph=True)[0] 
            #print(g2.shape)
            h[:,:,j] = g2
        return y,g,h
        
    def grad_and_lapl(self,x):
        x1  = tr.roll(x, shifts=1, dims=1) + tr.roll(x, shifts=-1, dims=1) 
        x1 += tr.roll(x, shifts=1, dims=2) + tr.roll(x, shifts=-1, dims=2)
        sx = tr.roll(x,shifts=self.y,dims=self.dim) +  tr.roll(x,shifts=-self.y,dims=self.dim)
        x_x  = (x *x)
        x1_x = (x1*x)
        sx_x = (sx*x)
        inp = tr.cat([x_x.unsqueeze(3),x1_x.unsqueeze(3),sx_x.unsqueeze(3)],dim=3).view(np.prod(x.shape),3)
        out,gg,h = self.net_grad_and_hess(inp)
        out = out.view(x.shape)
        h = h.view(list(x.shape) + [3,3]) # I hope this arranges things correctly....
        #print(gg2.shape)
        gg = gg.view(list(x.shape) + [3])
        # this is not right
        #grad = 2.0*(grad[...,0]*x + grad[...,1]*x1 + grad[...,2]*sx)
        grad = 2.0*gg[...,0]*x +  gg[...,1]*x1 + gg[...,2]*sx# this term 1 partial term 2 and 3
        g1x = gg[...,1]*x
        grad += tr.roll(g1x, shifts=1, dims=1) + tr.roll(g1x, shifts=-1, dims=1) 
        grad += tr.roll(g1x, shifts=1, dims=2) + tr.roll(g1x, shifts=-1, dims=2)
        g2x = gg[...,2]*x
        grad += tr.roll(g2x, shifts=self.y, dims=self.dim) + tr.roll(g2x, shifts=-self.y, dims=self.dim) 

        # CHECK correctness of the rest ... and optimize
    
        lapl  = 2.0*(gg[...,0]        ).sum(dim=(1,2)) 
        lapl += 4.0*(h[...,0,0]*(x_x)).sum(dim=(1,2)) 
       
        lapl += 2.0*((h[...,1,0]+h[...,0,1])*(x1_x )).sum(dim=(1,2))
        
        lapl += 2.0*((h[...,2,0]+h[...,0,2])*(sx_x)).sum(dim=(1,2))
                
        lapl += ((h[...,2,1]+h[...,1,2])*(sx*x1 )).sum(dim=(1,2)) 
     
        if (self.y == 1):
            lapl += 2.0*((h[...,2,1]+h[...,1,2])*x_x).sum(dim=(1,2)) 
        
        lapl += (h[...,1,1]*(x1*x1 + 4.0*x_x  )).sum(dim=(1,2))  
         
        lapl += (h[...,2,2]*(sx*sx + 2.0*x_x  )).sum(dim=(1,2))
        
        #print(grad.shape)
        #grad = grad.view(x.shape)
        return grad,lapl


class Funct3(nn.Module):
    def __init__(self,L,dim=2,y=0,conv_layers=[4,4],dtype=tr.float32, pool=nn.AvgPool2d(kernel_size=2,stride=2),activation=tr.nn.Tanh(),initializer=tr.nn.init.xavier_uniform_): 
        super().__init__()

        self.net = nn.Sequential()
        
        self.dtype = dtype
        self.Npool = int(np.log(L)/np.log(2))
        #self.pool = pool
        self.y   = y
        self.dim = dim
        in_dim =1
        k=0
        for p in range(self.Npool):           
            for l in conv_layers:
                #print(k,in_dim,l)
                layer = nn.Conv2d(in_dim,l, kernel_size=3, stride=1, padding=1, padding_mode='circular', dtype=self.dtype)
                initializer(layer.weight)
                self.net.add_module('lin'+str(k),layer)
                self.net.add_module('act'+str(k),activation)
                in_dim=l
                k+=1
            self.net.add_module('poo'+str(p),pool)
        self.net.add_module('fla0',nn.Flatten(start_dim=1) )
        layer = tr.nn.Linear(in_dim,1,dtype=self.dtype,bias=False) # the constant is a zero mode
        initializer(layer.weight)
        self.net.add_module('lin'+str(k),layer)
        self.net.add_module('fla1',nn.Flatten(start_dim=0) )

        
    def par(self,x,f):
        return 0.5*(f(x)+f(-x))
    def rot180(self,x,f): 
        return 0.5*(f(x)+f(tr.rot90(x,k=2,dims=(2,3))))
    def flip(self,x,f,dims=[2]):
        return 0.5*(f(x)+f(tr.flip(x,dims=dims)))
       
    def forward(self,x):  
        x = x.unsqueeze(1)
        #flip along the other axis comes out automatically because
        # it is equivalent to 180 rotation followed by a flip around the axis (already symmetrized)
        return self.par(x,lambda x: self.rot180(x,lambda x: self.flip(x,self.net)))
        

    def grad_and_lapl(self,x):
        y = self.forward(x) 
        #print(y.shape,x.shape)
        grad = tr.autograd.grad(y,x,tr.ones_like(y),create_graph=True)[0]
        #print(grad.shape)
        lapl = tr.zeros_like(y)
        for i in range(x.shape[1]):  # Loop over dim 1
            for j in range(x.shape[2]):  # Loop over dim 2
                foo=tr.autograd.grad(grad[:,i,j],x,tr.ones_like(grad[:,i,j]),create_graph=True)[0]
                lapl+=foo[:,i,j]
        return grad,lapl 

class Funct3no(nn.Module):
    def __init__(self,L,dim=2,y=0,conv_layers=[4,4],dtype=tr.float32, pool=nn.AvgPool2d(kernel_size=2,stride=2),activation=tr.nn.Tanh(),initializer=tr.nn.init.xavier_uniform_): 
        super().__init__()

        self.net = nn.Sequential()
        
        self.dtype = dtype
        self.Npool = int(np.log(L)/np.log(2))
        #self.pool = pool
        self.y   = y
        self.dim = dim
        in_dim =1
        k=0
        for p in range(self.Npool):           
            for l in conv_layers:
                #print(k,in_dim,l)
                layer = nn.Conv2d(in_dim,l, kernel_size=3, stride=1, padding=1, padding_mode='circular', dtype=self.dtype)
                initializer(layer.weight)
                self.net.add_module('lin'+str(k),layer)
                self.net.add_module('act'+str(k),activation)
                in_dim=l
                k+=1
            self.net.add_module('poo'+str(p),pool)
        self.net.add_module('fla0',nn.Flatten(start_dim=1) )
        layer = tr.nn.Linear(in_dim,1,dtype=self.dtype,bias=False) # the constant is a zero mode
        initializer(layer.weight)
        self.net.add_module('lin'+str(k),layer)
        self.net.add_module('fla1',nn.Flatten(start_dim=0) )

        
    def par(self,x,f):
        return 0.5*(f(x)+f(-x))
    def rot180(self,x,f): 
        return 0.5*(f(x)+f(tr.rot90(x,k=2,dims=(2,3))))
    def flip(self,x,f,dims=[2]):
        return 0.5*(f(x)+f(tr.flip(x,dims=dims)))
       
    def forward(self,x):  
        x = x.unsqueeze(1)
        #flip along the other axis comes out automatically because
        # it is equivalent to 180 rotation followed by a flip around the axis (already symmetrized)
        #return self.par(x,lambda x: self.rot180(x,lambda x: self.flip(x,self.net)))
        return self.net(x) # impose only parity via x**2
        

    def grad_and_lapl(self,x):
        y = self.forward(x) 
        #print(y.shape,x.shape)
        grad = tr.autograd.grad(y,x,tr.ones_like(y),create_graph=True)[0]
        #print(grad.shape)
        lapl = tr.zeros_like(y)
        for i in range(x.shape[1]):  # Loop over dim 1
            for j in range(x.shape[2]):  # Loop over dim 2
                foo=tr.autograd.grad(grad[:,i,j],x,tr.ones_like(grad[:,i,j]),create_graph=True)[0]
                lapl+=foo[:,i,j]
        return grad,lapl 

class Funct3Tinv(nn.Module):
    def __init__(self,L,dim=2,y=0,conv_layers=[4,4],dtype=tr.float32,activation=tr.nn.Tanh(),initializer=tr.nn.init.xavier_uniform_): 
        super().__init__()

        self.net = nn.Sequential()
     
        self.dtype = dtype
        self.Npool = int(np.log(L)/np.log(2))
        #self.pool = pool
        self.y   = y
        self.dim = dim
        in_dim =1
        k=0          
        for l in conv_layers:
            #print(k,in_dim,l)
            layer = nn.Conv2d(in_dim,l, kernel_size=3, stride=1, padding=1, padding_mode='circular', dtype=self.dtype)
            initializer(layer.weight)
            self.net.add_module('lin'+str(k),layer)
            self.net.add_module('act'+str(k),activation)
            in_dim=l
            k+=1

        self.net.add_module('global_pool',nn.AvgPool2d(kernel_size=L,stride=L))
        self.net.add_module('fla0',nn.Flatten(start_dim=1) )
        layer = tr.nn.Linear(in_dim,1,dtype=self.dtype)
        initializer(layer.weight)
        self.net.add_module('lin'+str(k),layer)
        self.net.add_module('fla1',nn.Flatten(start_dim=0) )

        
    def par(self,x,f):
        return 0.5*(f(x)+f(-x))
    def rot180(self,x,f): 
        return 0.5*(f(x)+f(tr.rot90(x,k=2,dims=(2,3))))
    def flip(self,x,f,dims=[2]):
        return 0.5*(f(x)+f(tr.flip(x,dims=dims)))
    

    def forward(self,x):
        x = x.unsqueeze(1)
        #flip along the other axis comes out automatically because
        # it is equivalent to 180 rotation followed by a flip around the axis (already symmetrized)
        return self.par(x,lambda x: self.rot180(x,lambda x: self.flip(x,self.net)))    

    def grad_and_lapl(self,x):
        y = self.forward(x) 
        #print(y.shape,x.shape)
        grad = tr.autograd.grad(y,x,tr.ones_like(y),create_graph=True)[0]
        #print(grad.shape)
        lapl = tr.zeros_like(y)
        for i in range(x.shape[1]):  # Loop over dim 1
            for j in range(x.shape[2]):  # Loop over dim 2
                foo=tr.autograd.grad(grad[:,i,j],x,tr.ones_like(grad[:,i,j]),create_graph=True)[0]
                lapl+=foo[:,i,j]
        return grad,lapl 

class Funct3T(nn.Module):
    def __init__(self,L,dim=2,y=0,conv_layers=[4,4],dtype=tr.float32,activation=tr.nn.Tanh(),initializer=tr.nn.init.xavier_uniform_): 
        super().__init__()

        self.net = nn.Sequential()
     
        self.dtype = dtype
        self.Npool = int(np.log(L)/np.log(2))
        #self.pool = pool
        self.y   = y
        self.dim = dim
        in_dim =1
        k=0          
        for l in conv_layers:
            #print(k,in_dim,l)
            layer = nn.Conv2d(in_dim,l, kernel_size=3, stride=1, padding=1, padding_mode='circular', dtype=self.dtype)
            initializer(layer.weight)
            self.net.add_module('lin'+str(k),layer)
            self.net.add_module('act'+str(k),activation)
            in_dim=l
            k+=1

        self.net.add_module('global_pool',nn.AvgPool2d(kernel_size=L,stride=L))
        self.net.add_module('fla0',nn.Flatten(start_dim=1) )
        layer = tr.nn.Linear(in_dim,1,dtype=self.dtype)
        initializer(layer.weight)
        self.net.add_module('lin'+str(k),layer)
        self.net.add_module('fla1',nn.Flatten(start_dim=0) )

        
    def par(self,x,f):
        return 0.5*(f(x)+f(-x))
    def rot180(self,x,f): 
        return 0.5*(f(x)+f(tr.rot90(x,k=2,dims=(2,3))))
    def flip(self,x,f,dims=[2]):
        return 0.5*(f(x)+f(tr.flip(x,dims=dims)))
    

    def forward(self,x):
        x = x.unsqueeze(1)
        #flip along the other axis comes out automatically because
        # it is equivalent to 180 rotation followed by a flip around the axis (already symmetrized)
        #return self.par(x,lambda x: self.rot180(x,lambda x: self.flip(x,self.net)))
        return self.net(x)

    def grad_and_lapl(self,x):
        y = self.forward(x) 
        #print(y.shape,x.shape)
        grad = tr.autograd.grad(y,x,tr.ones_like(y),create_graph=True)[0]
        #print(grad.shape)
        lapl = tr.zeros_like(y)
        for i in range(x.shape[1]):  # Loop over dim 1
            for j in range(x.shape[2]):  # Loop over dim 2
                foo=tr.autograd.grad(grad[:,i,j],x,tr.ones_like(grad[:,i,j]),create_graph=True)[0]
                lapl+=foo[:,i,j]
        return grad,lapl 


class gscalar(nn.Module):
    """Linear scalar model from LAP26 adapted for variates framework.
    
    Simple fully-connected network that produces scalar output from field configurations.
    Compatible with ControlModel_g for first-order control variates.
    
    Architecture:
    - Flattens input field (batch, L, L) -> (batch, L*L)
    - Multiple linear layers with Tanh activation
    - Final layer initialized to zero for stable training
    - Outputs scalar values (batch,)
    """
    
    def __init__(self, L, hidden_dim=[4,4,4], dtype=tr.float32, y=0, dim=2,
                 activation=tr.nn.Tanh(), initializer=tr.nn.init.xavier_uniform_):
        super().__init__()
        
        self.L = L
        self.y = y  # Required for ControlModel_g compatibility
        self.dim = dim  # Required for ControlModel_g compatibility
        self.dtype = dtype
        
        # Calculate volume
        vol = L * L
        
        # Build network architecture
        sizes = [vol] + hidden_dim + [1]
        layers = []
        
        for i in range(len(sizes) - 1):
            layer = nn.Linear(sizes[i], sizes[i+1], bias=False, dtype=dtype)
            if i < len(sizes) - 2:
                # Initialize hidden layers
                initializer(layer.weight)
            else:
                # Initialize final layer to zero (like LAP26)
                tr.nn.init.zeros_(layer.weight)
            
            layers.append(layer)
            
            # Add activation except for final layer
            if i < len(sizes) - 2:
                layers.append(activation)
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass: (batch, L, L) -> (batch,)"""
        # Flatten the field configuration
        x_flat = x.reshape(x.shape[0], -1)
        return self.net(x_flat).squeeze(-1)
    
    def grad(self, x):
        """Compute gradient dg/dphi for compatibility with Gfunc1 API."""
        y = self.forward(x)
        return tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]


class g_trans(nn.Module):
    """Translation-covariant local vector field g_x(phi) from a circular CNN."""

    def __init__(self, L, conv_layers=[4, 4, 4], dtype=tr.float32, y=0, dim=2,
                 activation=tr.nn.Tanh(), initializer=tr.nn.init.xavier_uniform_):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.dtype = dtype

        layers = []
        in_channels = 1
        for width in conv_layers:
            layer = nn.Conv2d(
                in_channels,
                width,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='circular',
                bias=False,
                dtype=dtype,
            )
            initializer(layer.weight)
            layers.append(layer)
            layers.append(activation)
            in_channels = width

        self.trunk = nn.Sequential(*layers)
        self.local_readout = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False, dtype=dtype)
        tr.nn.init.zeros_(self.local_readout.weight)
        self.equivariant_vector_g = True

    def local_vector(self, x):
        """Unprojected local vector field from the circular CNN."""
        h = self.trunk(x.unsqueeze(1))
        return self.local_readout(h).squeeze(1)

    def forward(self, x):
        """Forward pass: (batch, L, L) -> local vector field (batch, L, L)."""
        return self.local_vector(x)

    def scalar(self, x):
        """Optional invariant scalar summary of the vector field."""
        return self.forward(x).mean(dim=(1, 2))

    def grad(self, x):
        y = self.scalar(x)
        return tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]


# Backward-compatible alias for older run configs/checkpoints.
g_scalar_translational = g_trans


class g_trans_rot(g_trans):
    """Translation- and C4 rotation-equivariant local vector field.

    Circular convolutions give translation equivariance. The group average
    projects the local vector field onto the C4-rotation equivariant subspace:
        g(phi) = 1/4 sum_k R^{-k} h(R^k phi), k=0,1,2,3.
    """

    def forward(self, x):
        total = tr.zeros_like(x)
        for k in range(4):
            x_rot = tr.rot90(x, k=k, dims=(1, 2))
            g_rot = self.local_vector(x_rot)
            total = total + tr.rot90(g_rot, k=-k, dims=(1, 2))
        return total / 4.0


class g_escnn_c4(nn.Module):
    """C4/D4 steerable CNN local vector field using escnn.

    The input and output are scalar fields on the lattice. Hidden layers use
    regular group representations, so the network is equivariant under the
    selected lattice point group. Kernel-3 convolutions use explicit circular
    padding to respect periodic boundary conditions.
    """

    def __init__(self, L, conv_layers=[4, 4, 4], dtype=tr.float64, y=0, dim=2,
                 activation=tr.nn.Tanh(), initializer=tr.nn.init.xavier_uniform_,
                 point_group='c4'):
        super().__init__()

        try:
            from escnn import gspaces as escnn_gspaces
            from escnn import nn as escnn_nn
            self._configure_escnn_cache()
        except ImportError as exc:
            raise ImportError(
                "g_escnn_c4 requires escnn. Install it with: pip install escnn"
            ) from exc

        self.L = L
        self.y = y
        self.dim = dim
        self.dtype = dtype
        self.equivariant_vector_g = True
        self.escnn_nn = escnn_nn
        self.point_group = point_group

        if point_group == 'c4':
            self.r2_act = escnn_gspaces.rot2dOnR2(N=4)
        elif point_group == 'd4':
            self.r2_act = escnn_gspaces.flipRot2dOnR2(N=4)
        else:
            raise ValueError(f"Unknown point_group={point_group}. Use 'c4' or 'd4'.")
        self.in_type = escnn_nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        class CircularPadR2Conv(escnn_nn.EquivariantModule):
            """R2Conv with circular padding on the raw tensor."""

            def __init__(self, in_type, out_type, escnn_nn, kernel_size=3, bias=False):
                super().__init__()
                self.in_type = in_type
                self.out_type = out_type
                self.pad = kernel_size // 2
                self.escnn_nn = escnn_nn
                self.conv = escnn_nn.R2Conv(
                    in_type,
                    out_type,
                    kernel_size=kernel_size,
                    padding=0,
                    bias=bias,
                )

            def forward(self, x):
                tensor = F.pad(x.tensor, (self.pad, self.pad, self.pad, self.pad), mode='circular')
                x_pad = self.escnn_nn.GeometricTensor(tensor, self.in_type)
                return self.conv(x_pad)

            def evaluate_output_shape(self, input_shape):
                return (input_shape[0], self.out_type.size, *input_shape[2:])

        modules = []
        in_type = self.in_type
        for width in conv_layers:
            out_type = escnn_nn.FieldType(self.r2_act, width * [self.r2_act.regular_repr])
            modules.append(CircularPadR2Conv(in_type, out_type, escnn_nn, kernel_size=3, bias=False))
            modules.append(escnn_nn.ReLU(out_type, inplace=True))
            in_type = out_type

        self.out_type = escnn_nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        modules.append(escnn_nn.R2Conv(in_type, self.out_type, kernel_size=1, padding=0, bias=False))

        self.net = escnn_nn.SequentialModule(*modules)
        self.net.to(dtype=dtype)

    def forward(self, x):
        """Forward pass: (batch, L, L) -> local scalar field (batch, L, L)."""
        geo_x = self.escnn_nn.GeometricTensor(x.unsqueeze(1), self.in_type)
        geo_y = self.net(geo_x)
        return geo_y.tensor.squeeze(1)

    @staticmethod
    def _configure_escnn_cache():
        """Redirect escnn's joblib cache away from read-only site-packages."""
        cache_dir = os.environ.get('ESCNN_CACHE_DIR', '/tmp/escnn_cache')
        os.makedirs(cache_dir, exist_ok=True)

        from joblib import Memory
        import escnn.group as escnn_group
        import escnn.group._clebsh_gordan as cg
        import escnn.group.irrep as irrep

        escnn_group.__cache_path__ = cache_dir
        memory = Memory(cache_dir, verbose=0)

        # These functions are decorated at import time, so rewrap their original
        # callables with a writable cache location.
        for module, names in (
            (cg, ('_clebsh_gordan_tensor', '_find_tensor_decomposition')),
            (irrep, ('_restrict_irrep',)),
        ):
            module.cache = memory
            for name in names:
                func = getattr(module, name)
                original = getattr(func, 'func', func)
                setattr(module, name, memory.cache(original))

    def scalar(self, x):
        """Optional invariant scalar summary of the vector field."""
        return self.forward(x).mean(dim=(1, 2))

    def grad(self, x):
        y = self.scalar(x)
        return tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]


class g_escnn_d4(g_escnn_c4):
    """D4 steerable CNN: translations, 90-degree rotations, and flips."""

    def __init__(self, *args, **kwargs):
        kwargs['point_group'] = 'd4'
        super().__init__(*args, **kwargs)


class gscalar_Unified(nn.Module):
    """Unified version of gscalar with tau conditioning.

    Keeps the same flattened-MLP structure as gscalar, but concatenates a
    learnable tau embedding so one model can serve all tau in [0, L/2].
    """

    def __init__(self, L, hidden_dim=[4, 4, 4], dtype=tr.float32, y=0, dim=2,
                 activation=tr.nn.Tanh(), initializer=tr.nn.init.xavier_uniform_,
                 tau_embed_dim=8):
        super().__init__()

        self.L = L
        self.y = y
        self.dim = dim
        self.dtype = dtype
        self.tau_max = L // 2 + 1
        self.tau_embed_dim = tau_embed_dim

        vol = L * L
        self.tau_embedding = nn.Embedding(self.tau_max, tau_embed_dim)

        sizes = [vol + tau_embed_dim] + hidden_dim + [1]
        layers = []

        for i in range(len(sizes) - 1):
            layer = nn.Linear(sizes[i], sizes[i + 1], bias=False, dtype=dtype)
            if i < len(sizes) - 2:
                initializer(layer.weight)
            else:
                tr.nn.init.zeros_(layer.weight)

            layers.append(layer)
            if i < len(sizes) - 2:
                layers.append(activation)

        self.net = nn.Sequential(*layers)

    def _fold_tau(self, tau):
        if tau > self.L // 2:
            return self.L - tau
        return tau

    def set_tau(self, tau):
        self.y = self._fold_tau(tau)

    def forward(self, x, tau=None):
        if tau is None:
            tau = self.y
        else:
            tau = self._fold_tau(tau)

        x_flat = x.reshape(x.shape[0], -1)
        tau_tensor = tr.full((x.shape[0],), tau, dtype=tr.long, device=x.device)
        tau_embed = self.tau_embedding(tau_tensor)
        inp = tr.cat([x_flat, tau_embed], dim=1)
        return self.net(inp).squeeze(-1)

    def grad(self, x, tau=None):
        y = self.forward(x, tau=tau)
        return tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]


class Gfunc1(nn.Module):
    """Scalar g(phi) model with discrete symmetries except translation.

    Symmetries enforced:
      - parity: phi -> -phi
      - 180-degree rotation
      - single-axis flip (the other axis follows from rot180+flip)
    No translation averaging is applied here.
    """

    def __init__(self, L, dim=2, y=0, conv_layers=[4,4], dtype=tr.float32,
                 activation=tr.nn.Tanh(), initializer=tr.nn.init.xavier_uniform_):
        super().__init__()

        self.trunk = nn.Sequential()
        self.dtype = dtype
        self.y = y
        self.dim = dim
        self.L = L

        in_dim = 1
        k = 0
        for l in conv_layers:
            layer = nn.Conv2d(in_dim, l, kernel_size=3, stride=1, padding=1, padding_mode='circular', dtype=self.dtype)
            initializer(layer.weight)
            self.trunk.add_module('lin'+str(k), layer)
            self.trunk.add_module('act'+str(k), activation)
            in_dim = l
            k += 1

        # Local readout (no global pooling): translation covariance is preserved.
        # We pick one anchor site after symmetrization to define scalar g0.
        self.readout = nn.Conv2d(in_dim, 1, kernel_size=1, stride=1, padding=0, dtype=self.dtype)
        initializer(self.readout.weight)
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)

    def par(self, x, f):
        return 0.5 * (f(x) + f(-x))
    def rot180(self, x, f):
        return 0.5 * (f(x) + f(tr.rot90(x, k=2, dims=(2, 3))))
    def flip(self, x, f, dims=[2]):
        return 0.5 * (f(x) + f(tr.flip(x, dims=dims)))

    def scalar_from_map(self, x):
        h = self.trunk(x)
        m = self.readout(h)  # (B,1,L,L)
        # Anchor-site scalar: intentionally not translation invariant.
        return m[:, 0, 0, 0]

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.par(x, lambda z: self.rot180(z, lambda zz: self.flip(zz, self.scalar_from_map)))

    def grad(self, x):
        """Return first derivative d g / d phi."""
        y = self.forward(x)
        return tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]


class Gfunc_sym(nn.Module):
    """Translation-symmetrized wrapper for scalar g(phi).

    Enforces periodic translation invariance by averaging G over all lattice
    shifts. Intended to wrap Gfunc1 (or any scalar model with same API).
    """

    def __init__(self, G):
        super().__init__()
        self.G = G
        self.y = G.y
        self.dim = G.dim

    def forward(self, x):
        Lx, Ly = x.shape[1], x.shape[2]
        out = tr.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        norm = float(Lx * Ly)
        for sx in range(Lx):
            for sy in range(Ly):
                out += self.G(tr.roll(x, shifts=(sx, sy), dims=(1, 2)))
        return out / norm

    def grad(self, x):
        """Return first derivative d g_sym / d phi."""
        y = self.forward(x)
        return tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]


class gscalar_sym(nn.Module):
    """Symmetrized wrapper for gscalar.

    Enforces parity, 180-degree rotation, single-axis flip, and
    periodic translation invariance by averaging over all shifts.
    """

    def __init__(self, G):
        super().__init__()
        self.G = G
        self.y = G.y
        self.dim = G.dim

    def par(self, x, f):
        return 0.5 * (f(x) + f(-x))

    def rot180(self, x, f):
        return 0.5 * (f(x) + f(tr.rot90(x, k=2, dims=(1, 2))))

    def flip(self, x, f, dims=[1]):
        return 0.5 * (f(x) + f(tr.flip(x, dims=dims)))

    def _discrete_sym(self, x):
        return self.par(x, lambda z: self.rot180(z, lambda zz: self.flip(zz, self.G)))

    def forward(self, x):
        Lx, Ly = x.shape[1], x.shape[2]
        out = tr.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        norm = float(Lx * Ly)
        for sx in range(Lx):
            for sy in range(Ly):
                out += self._discrete_sym(tr.roll(x, shifts=(sx, sy), dims=(1, 2)))
        return out / norm

    def grad(self, x):
        """Return first derivative d gscalar_sym / d phi."""
        y = self.forward(x)
        return tr.autograd.grad(y, x, tr.ones_like(y), create_graph=True)[0]


#symmetrizer wrapper class
#I need to eliminate the symmetric modules as those are much slower
#i.e. train assymetric --- symmetrize at the end gives major improvement
class FunctSym(nn.Module):
    def __init__(self,F=F): 
        super().__init__()

        self.F = F
        self.y = F.y
        self.dim = F.dim
        
    def par(self,x,f):
        return 0.5*(f(x)+f(-x))
    def rot180(self,x,f): 
        return 0.5*(f(x)+f(tr.rot90(x,k=2,dims=(1,2))))
    def flip(self,x,f,dims=[1]):
        return 0.5*(f(x)+f(tr.flip(x,dims=dims)))
    

    def apply_symmetries(self, x):
        px = -x
        rx = tr.rot90(x, k=2, dims=(1, 2))
        fx = tr.flip(x, dims=[1])
        prx = -rx
        pfx = -fx
        frx = tr.flip(rx, dims=[1])
        prfx = -frx
        return 0.125*(self.F(x) + self.F(px) + self.F(rx) +self.F(fx) + self.F(prx) +self.F(pfx) + self.F(frx) + self.F(prfx))
                
    def forward(self,x):
        #x = x.unsqueeze(1)
        #flip along the other axis comes out automatically because
        # it is equivalent to 180 rotation followed by a flip around the axis (already symmetrized)
        return self.par(x,lambda x: self.rot180(x,lambda x: self.flip(x,self.F)))
        #return self.F(x)
        #return self.apply_symmetries(x)

    
    def grad_and_lapl(self,x):
        y = self.forward(x) 
        #print(y.shape,x.shape)
        grad = tr.autograd.grad(y,x,tr.ones_like(y),create_graph=True)[0]
        #print(grad.shape)
        lapl = tr.zeros_like(y)
        for i in range(x.shape[1]):  # Loop over dim 1
            for j in range(x.shape[2]):  # Loop over dim 2
                foo=tr.autograd.grad(grad[:,i,j],x,tr.ones_like(grad[:,i,j]),create_graph=True)[0]
                lapl+=foo[:,i,j]
        return grad,lapl 


def _module_dtype_device(module):
    """Return the dtype/device used by a module's tensors."""
    for tensor in module.parameters():
        return tensor.dtype, tensor.device
    for tensor in module.buffers():
        return tensor.dtype, tensor.device
    return tr.get_default_dtype(), tr.device("cpu")


class ControlModel(nn.Module):
    def __init__(self,muO=1.0,force=None,c2p_net=None):
        super(ControlModel, self).__init__()
        self.y = c2p_net.y
        if(c2p_net.dim==1):
            self.d = 2
        else:
            self.d =1
        
        self.force = force
        self.c2p_net = c2p_net
        dtype, device = _module_dtype_device(c2p_net)
        self.muO = nn.Parameter(tr.tensor([muO], dtype=dtype, device=device),requires_grad=True)
        
    def computeO(self, x):
        """Connected wall-to-wall correlator.

        C_conn(tau) = ⟨φ_0(t)φ_0(t+τ)⟩ - ⟨φ_0⟩²
        where φ_0(t) = (1/L) Σ_x φ(x,t)

        Subtracting the batch mean (ensemble average) before the correlator
        is equivalent to the vacuum subtraction.
        """
        x0 = tr.mean(x, dim=self.d).squeeze()  # wall average: (batch, L)
        mean_x0 = tr.mean(x0, dim=0, keepdim=True)  # ensemble average: (1, L)
        x0 = x0 - mean_x0  # subtract ensemble average
        xx = (x0 * tr.roll(x0, dims=1, shifts=-self.y)).mean(dim=1)
        return xx

    def F(self, x, n_colors=None):
        """Compute control variate F(x).

        Args:
            x: Input field
            n_colors: Override n_colors for probing (use more for evaluation)
        """
        if n_colors is not None and hasattr(self.c2p_net, 'grad_and_lapl'):
            # Try to pass n_colors if the model supports it
            try:
                g, l = self.c2p_net.grad_and_lapl(x, n_colors=n_colors)
            except TypeError:
                # Model doesn't support n_colors argument
                g, l = self.c2p_net.grad_and_lapl(x)
        else:
            g, l = self.c2p_net.grad_and_lapl(x)
        return (l + (self.force(x)*g).sum(dim=(1,2)))

    def Delta(self, x, n_colors=None):
        """Compute improved estimator Delta = O - F.

        Args:
            x: Input field
            n_colors: Override n_colors for probing (use more for evaluation)
        """
        return self.computeO(x) - self.F(x, n_colors=n_colors)

    def loss(self, x, n_colors=None):
        """Compute loss = Var(Delta - muO).

        Args:
            x: Input field
            n_colors: Override n_colors for probing (use more for evaluation)
        """
        return ((self.Delta(x, n_colors=n_colors) - self.muO)**2).mean()


class ControlModel_g(nn.Module):
    """First-order control variate model based on g(phi), no second derivatives.

    Uses the Schwinger-Dyson / integration-by-parts identity componentwise:
        <d_i g> = <g d_i S>
    and builds a scalar zero-mean control variate by summing over lattice dof:
        F_g(phi) = sum_i d_i g(phi) + g(phi) * sum_i force_i(phi)
    where force = -dS/dphi in this codebase convention.
    """

    def __init__(self, muO=1.0, force=None, g_net=None):
        super(ControlModel_g, self).__init__()
        self.y = g_net.y
        if(g_net.dim == 1):
            self.d = 2
        else:
            self.d = 1

        self.force = force
        self.g_net = g_net
        dtype, device = _module_dtype_device(g_net)
        self.muO = nn.Parameter(tr.tensor([muO], dtype=dtype, device=device), requires_grad=True)#activa parameter for minimization

    def computeO(self, x):
        """Connected wall-to-wall correlator."""
        x0 = tr.mean(x, dim=self.d).squeeze()
        mean_x0 = tr.mean(x0, dim=0, keepdim=True)
        x0 = x0 - mean_x0
        xx = (x0 * tr.roll(x0, dims=1, shifts=-self.y)).mean(dim=1)
        return xx

    def g(self, x):
        """Scalar network output g(phi), expected shape (batch,). g: Real^{volume}->Real."""

        return self.g_net(x)

    def F_no(self, x, n_colors=None):
        """Unsymmetrized first-order CV with cv_lin_inv structure.

        Implements (site-wise translated scalar field):
            F(phi) = sum_{i,j} [ d/dphi_{ij} g(T_{-(i,j)}phi)
                                 + g(T_{-(i,j)}phi) * force_{ij}(phi) ]
        where force = -dS/dphi in this codebase convention.

        n_colors is accepted for API compatibility and ignored.
        """
        Lx, Ly = x.shape[1], x.shape[2]
        force_x = self.force(x)
        f = tr.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for i in range(Lx):
            for j in range(Ly):
                # Translation-equivalent local scalar
                x_shift = tr.roll(x, shifts=(-i, -j), dims=(1, 2))

                g_shift = self.g(x_shift)  # (batch,)
                #parity of g
                #g_shift = 0.5 * (self.g(x_shift) + self.g(-x_shift))
                # Diagonal Jacobian component: d g_shift / d x_{ij}
                dg = tr.autograd.grad(outputs=g_shift.sum(),inputs=x,create_graph=True,retain_graph=True)[0]

                f += dg[:, i, j] + g_shift * force_x[:, i, j]

        return f

    def F(self, x, n_colors=None):
        """Symmetrized first-order CV without translation averaging.

        Builds a scalar control variate from F_no by averaging over the
        square-lattice point-group symmetries and parity:
            phi -> -phi
            phi -> R^k phi, k = 0,1,2,3
            phi -> flip(R^k phi), k = 0,1,2,3

        This enforces the same symmetries checked by
        control_model_F_symmetry_checker, except translations.
        """
        transforms = []
        for sign in (1.0, -1.0):
            sx = sign * x
            for k in range(4):
                rx = tr.rot90(sx, k=k, dims=(1, 2))
                transforms.append(rx)
                transforms.append(tr.flip(rx, dims=(1,)))

        total = tr.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for tx in transforms:
            total = total + self.F_no(tx, n_colors=n_colors)

        return total / len(transforms)

    def Delta(self, x, n_colors=None):
        """Improved estimator Delta = O - F_g."""
        return self.computeO(x) - self.F(x, n_colors=n_colors)

    def loss(self, x, n_colors=None):
        """Loss = Var(Delta - muO)."""
        return ((self.Delta(x, n_colors=n_colors) - self.muO)**2).mean()


class ControlModel_g_equiv(nn.Module):
    """First-order CV wrapper for equivariant/local vector-field g models.

    This class is intentionally independent from ControlModel_g. It assumes the
    model itself handles any desired equivariance/symmetry and applies no extra
    symmetrization.
    """

    def __init__(self, muO=1.0, force=None, g_net=None):
        super(ControlModel_g_equiv, self).__init__()
        self.y = g_net.y
        if(g_net.dim == 1):
            self.d = 2
        else:
            self.d = 1

        self.force = force
        self.g_net = g_net
        dtype, device = _module_dtype_device(g_net)
        self.muO = nn.Parameter(tr.tensor([muO], dtype=dtype, device=device), requires_grad=True)

    def computeO(self, x):
        """Connected wall-to-wall correlator."""
        x0 = tr.mean(x, dim=self.d).squeeze()
        mean_x0 = tr.mean(x0, dim=0, keepdim=True)
        x0 = x0 - mean_x0
        xx = (x0 * tr.roll(x0, dims=1, shifts=-self.y)).mean(dim=1)
        return xx

    def g_vector(self, x):
        """Local vector-field output g_x(phi), expected shape (batch, L, L)."""
        g_vec = self.g_net(x)
        if g_vec.shape != x.shape:
            raise ValueError(f"Equivariant g_net must return shape {tuple(x.shape)}, got {tuple(g_vec.shape)}")
        return g_vec

    def F(self, x, n_colors=None):
        """First-order CV from a translation-covariant local vector field."""
        force_x = self.force(x)
        g_vec = self.g_vector(x)

        n_probes = n_colors if n_colors is not None else 4
        div_g = tr.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for _ in range(n_probes):
            z = 2 * tr.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) - 1
            grad_z = tr.autograd.grad(
                outputs=(g_vec * z).sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True
            )[0]
            div_g = div_g + (grad_z * z).sum(dim=(1, 2))
        div_g = div_g / n_probes

        return div_g + (g_vec * force_x).sum(dim=(1, 2))

    def Delta(self, x, n_colors=None):
        """Improved estimator Delta = O - F_g."""
        return self.computeO(x) - self.F(x, n_colors=n_colors)

    def loss(self, x, n_colors=None):
        """Loss = Var(Delta - muO)."""
        return ((self.Delta(x, n_colors=n_colors) - self.muO)**2).mean()


def train_control_model(CM,phi,learning_rate=1e-3,epochs=100,super_batch=1,update=lambda phi: tr.randn(phi.shape)):
    c=0
    for tt in CM.parameters():
        if tt.requires_grad==True :
            c+=tt.numel()
    print("parameter count: ",c)

    params = [p for p in CM.parameters() if p.requires_grad==True]
    optimizer = tr.optim.Adam(params, lr=learning_rate)
    pbar = tqdm.tqdm(range(epochs))
    loss = tr.zeros(1,device=device)
    
    loss_history = []
    #muO_history = []
    for t in pbar: 
        optimizer.zero_grad()
        loss = tr.zeros(1,device=device)
        for b in range(super_batch):
            phi = update(phi)
            x = phi.clone()
            x.requires_grad = True
            tloss = CM.loss(x)/super_batch
            tloss.backward()
        loss += tloss
        optimizer.step()
        loss_history.append(loss.detach().to("cpu").numpy())
        #loss_history.append(CM.muO.detach().to("cpu").numpy())
        pbar.set_postfix({'loss': loss.detach().to("cpu").numpy()})

    return loss_history,phi



#model_factory_v0 = {'Funct2'      : lambda L,y,ll,type : Funct2(y=y,conv_layers=ll,dtype=type),
#                 'Funct3'      : lambda L,y,ll,type : Funct3(L,dim=2,y=y,conv_layers=ll,dtype=type),
#                 'Funct3Tinv'  : lambda L,y,ll,type : Funct3Tinv(L,dim=2,y=y,conv_layers=ll,dtype=type),
#                 'Funct3T'     : lambda L,y,ll,type : Funct3T(L,dim=2,y=y,conv_layers=ll,dtype=type),
#                 'Funct3no'    : lambda L,y,ll,type : Funct3no(L,dim=2,y=y,conv_layers=ll,dtype=type)}


# Factory Functions
def activation_factory(name, **kwargs):
    """Factory function to create activation functions dynamically."""
    class ArcSinh(nn.Module):
        def forward(self, x):
            return tr.asinh(x)

    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "softplus": nn.Softplus,
        "arcsinh": ArcSinh,
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    
    return activations[name](**kwargs)  # Pass any extra arguments dynamically


def model_factory(class_name, sym_flag=False, **kwargs):
    classes = {
        'Funct2'      : Funct2,
        'Funct3'      : Funct3,
        'Funct3Tinv'  : Funct3Tinv,
        'Funct3T'     : Funct3T,
        'Funct3no'    : Funct3no,
        'Gfunc1'      : Gfunc1,
        'gscalar'     : gscalar,
        'g_trans'      : g_trans,
        'g_trans_rot'  : g_trans_rot,
        'g_escnn_c4'   : g_escnn_c4,
        'g_escnn_d4'   : g_escnn_d4,
        'g_scalar_translational': g_trans,
        'LinScalar'   : gscalar,     # backward-compatible alias
    }
    
    if class_name not in classes:
        raise ValueError(f"Unknown class: {class_name}")

    foo=classes[class_name](**kwargs)  # Dynamically pass arguments
    if class_name == 'Gfunc1' and sym_flag:
        return Gfunc_sym(foo)
    if class_name in ('gscalar', 'g_trans', 'g_trans_rot', 'g_escnn_c4', 'g_escnn_d4', 'g_scalar_translational', 'LinScalar') and sym_flag:
        return gscalar_sym(foo)
    if(sym_flag):
        return FunctSym(F=foo)
    else:
        return foo

def test_symmetries(model,flag):
    L=8
    bs=3
    model = model_factory(model,sym_flag=flag,L=L,y=2,conv_layers=[4,4],dtype=tr.double)
    model.to(tr.double)
    x = tr.randn(bs,L,L,dtype=tr.double)
    symmetry_checker(x,model)
    
def test_grad_and_lap(model,flag):
    L=8
    bs = 3
    model = model_factory(model,sym_flag=flag,L=L,y=2,conv_layers=[4,4],dtype=tr.double)
    model.to(tr.double)
    #check grad
    x = tr.randn(bs,L,L,dtype=tr.double)
    eps=1e-3
    e = tr.zeros_like(x)
    e[:,1,1] = eps
    f = model(x)
    xx = x.clone()
    xx.requires_grad = True
    g,h=model.grad_and_lapl(xx)
    #print(g.shape)
    g_diff = tr.zeros_like(g)
    g_diff2 = tr.zeros_like(g)
    lap_diff = tr.zeros_like(f)
    lap_diff2 = tr.zeros_like(f)
    
    #u_x2 = (u(xm2) + 8*u(xp) - 8*u(xm) - u(xp2))/(12.*eps)
    #u_xx2 =(-u(xm2) + 16.0*u(xp) -30.0*u(x) + 16.0*u(xm) - u(xp2))/(12.*eps**2)
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            e = tr.zeros_like(x)
            e[:,i,j] = eps
            xp = x + e
            xm = x - e
            xp2 = x + 2*e
            xm2 = x - 2*e
            g_diff[:,i,j] = 0.5*(model(xp)-model(xm))/eps
            g_diff2[:,i,j] = (model(xm2)+ 8.*model(xp)- 8.*model(xm) - model(xp2))/(12.0*eps)
            lap_diff += (model(xp)+model(xm) - 2.0*model(x))/(eps**2)
            lap_diff2 += ( -model(xm2) - model(xp2) + 16.*model(xp)+ 16.0*model(xm) - 30.0*model(x))/(12.*eps**2)

       
    print("Grad diff   (num-auto)/auto :",(tr.norm(g_diff-g )/tr.norm(g)).detach().numpy())
    print("Grad diff2  (num-auto)/auto :",(tr.norm(g_diff2-g)/tr.norm(g)).detach().numpy())
    print("Lapl diff   (num-auto)/auto :",(tr.norm(h-lap_diff)/tr.norm(h)).detach().numpy())
    print("Lapl diff2  (num-auto)/auto :",(tr.norm(h-lap_diff2)/tr.norm(h)).detach().numpy())

def test_train(model_name,epochs,learning_rate,load_flag=False,ff='control_model_file.dict'):
    L=8
    bs = 256
    model = model_factory(model_name,L=L,y=2,conv_layers=[4,4,4,4],dtype=tr.float)
    if(load_flag):
        model.load_state_dict(tr.load(ff))
        model.eval()
    model.to(device)
    lat=[L,L]

    lam = 2.4
    mas = -0.50

    Nwarm = 2000
    Nmeas = 1000
    Nskip = 5
    hmc_batch_size = bs

    Vol = np.prod(lat)


    sg = qft.phi4(lat,lam,mas,batch_size=hmc_batch_size,device=device)

    phi = sg.hotStart()
    mn2 = integ.minnorm2(sg.force,sg.evolveQ,3,1.0)
 
    print("HMC Initial field characteristics: ",phi.shape,Vol,tr.mean(phi),tr.std(phi))

    hmc = upd.hmc(T=sg,I=mn2,verbose=False)

    tic=time.perf_counter()
    phi = hmc.evolve(phi,Nwarm)
    toc=time.perf_counter()
    print(f"time {(toc - tic)*1.0e6/Nwarm:0.4f} micro-seconds per HMC trajecrory")
    print("Acceptance: ",hmc.calc_Acceptance())

    CM = ControlModel(muO=0.3,force = sg.force,c2p_net=model)
    CM.to(device)
    hmc.AcceptReject = []
    ll_hist, phi= train_control_model(CM,phi,learning_rate=learning_rate,epochs=epochs,super_batch=1,update=lambda x : hmc.evolve(x,Nskip))
    print('HMC acceptance: ',hmc.calc_Acceptance())
    print('HMC history length: ',len(hmc.AcceptReject))

    x = phi.clone()
    x.requires_grad = True # because some of the tests may require grads

    print("Variance of O: ", CM.computeO(x).var().to("cpu").detach().numpy())
    print("Variance of imp(O): ", CM.Delta(x).var().to("cpu").detach().numpy())
    gain = CM.computeO(x).var()/CM.Delta(x).var()
    print("Variance improvement: ",gain.to("cpu").detach().numpy())

    print("Value of muO: ",CM.muO.to("cpu").detach().numpy())
    print("Mean(O): ",CM.computeO(x).mean().to("cpu").detach().numpy())
    print("Mean(impO): ",CM.Delta(x).mean().to("cpu").detach().numpy())
    tF = CM.F(x).to("cpu").detach().numpy()
    tO = CM.computeO(x).to("cpu").detach().numpy()

    corr = np.corrcoef(tF,tO)
    print("Correlation: ", corr[0,1])

    print("\n======")
    
    symmetry_checker(x,model)
    
    plt.plot(ll_hist)
    plt.yscale('log')
    plt.show()

    
def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-l'     , default='no-load')
    parser.add_argument('-e'     , type=int,default=1000)
    parser.add_argument('-lr'    , type=float,default=1.0e-3)
    parser.add_argument('-t'     ,  default='test_ders')
    parser.add_argument('-model' , default='Funct2')
    parser.add_argument('-symm'  , action='store_true')
    
    
    args = parser.parse_args()
    if(args.t=='grads'):
        print("Testing derivatives for "+args.model)
        if(args.symm):
            print("Testing symmetrized model")
        test_grad_and_lap(args.model,args.symm)

    elif(args.t=='symms'):
        print("Testing symmetries for "+args.model)
        if(args.symm):
            print("Testing symmetrized model")
        test_symmetries(args.model,args.symm)
        
    elif(args.t=="train"):
        print("Testing training of model "+args.model)
        if(not args.l=="no-load"):
            test_train(args.model,args.e,args.lr,True,args.l)
        else:
            test_train(args.model,args.e,args.lr)
            
    else:
        print("Nothing to test")
        print("Production training on the way....")

        production_train()
        
        print("...Done!")

if __name__ == "__main__":
    main()
    
