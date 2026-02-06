import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm 

import phi4 as qft
import integrators as integ
import update as upd

import matplotlib.pyplot as plt

import time

print("Torch Version: ",tr.__version__)
print("Numpy Version: ",np.__version__)

if tr.backends.mps.is_available():
    device = tr.device("mps")
else:
    print ("MPS device not found.")
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
                layer = nn.Conv2d(in_dim,l, kernel_size=3, stride=1, padding=1, padding_mode='circular')
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
                layer = nn.Conv2d(in_dim,l, kernel_size=3, stride=1, padding=1, padding_mode='circular')
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
            layer = nn.Conv2d(in_dim,l, kernel_size=3, stride=1, padding=1, padding_mode='circular')
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
            layer = nn.Conv2d(in_dim,l, kernel_size=3, stride=1, padding=1, padding_mode='circular')
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
        self.muO = nn.Parameter(tr.tensor([muO]),requires_grad=True)
        
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
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "softplus": nn.Softplus
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
    }
    
    if class_name not in classes:
        raise ValueError(f"Unknown class: {class_name}")

    foo=classes[class_name](**kwargs)  # Dynamically pass arguments
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
    
