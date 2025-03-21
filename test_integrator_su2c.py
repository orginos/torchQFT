import numpy as np
import torch as tr
import su2_chain as s
import integrators as integ
import matplotlib.pyplot as plt

lat = [8,8]
beta = 1.85
Nwarm = 400

su2f = s.field(lat,Nbatch=1)
su2c = s.SU2chain(beta=beta,field_type=su2f)

U =su2c.f.hot()
print(U.shape)

P  = su2c.refreshP()
K  = su2c.kinetic(P)
V  = su2c.action(U)
Hi = K + V


print("The total initial energy is: ",Hi)
x=[]
y=[]
y2=[]
for rk in np.logspace(1,3,50):
    k=int(rk)
    dt = 1.0/k
    print("Using dt= ",dt)
    l = integ.leapfrog(su2c.force,su2f.evolveQ,k,1.0)
    l2 = integ.minnorm2(su2c.force,su2f.evolveQ,k,1.0)
    PP,QQ = l.integrate(P,U)
    PP2,QQ2 = l2.integrate(P,U)
    Hf = su2c.kinetic(PP)+ su2c.action(QQ)
    Hf2 = su2c.kinetic(PP2)+ su2c.action(QQ2)
    print("The total final energy is: ", Hf)
    DH = tr.abs(Hf - Hi)
    DH2 = tr.abs(Hf2 - Hi)
    x.append(dt**2) # esp^2 integrator
    y.append(DH)
    y2.append(DH2)
    

# plotting the points  
plt.plot(x, y,x,y2) 
# naming the x axis 
plt.xlabel('$\\epsilon^2$') 
# naming the y axis 
plt.ylabel('$\\Delta H$') 

# function to show the plot 
plt.show() 
