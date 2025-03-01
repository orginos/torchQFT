from control_variates import *
import argparse
import sys
    
parser = argparse.ArgumentParser()
parser.add_argument('-tau'   , type=int,   default=2)
parser.add_argument('-f'     ,             default='no-load')
parser.add_argument('-e'     , type=int,   default=1000)
parser.add_argument('-lr'    , type=float, default=1.0e-3)
parser.add_argument('-t'     ,             default='test_ders')
parser.add_argument('-model' ,             default='Funct2')
parser.add_argument('-Nwarm' , type=int,   default=1000 )
parser.add_argument('-Nskip' , type=int,   default=5    )
parser.add_argument('-m'     , type=float, default=-0.5)
parser.add_argument('-g'     , type=float, default=1.0 )
parser.add_argument('-hmc_bs', type=int,   default=128  )
parser.add_argument('-L'     , type=int,   default=8, help="Lattice size")
parser.add_argument('-Nmd'   , type=int,   default=2, help="md steps")
parser.add_argument('-activ' ,             default="gelu", help="activation function")
parser.add_argument('-conv_l', type=int, default=[4,4,4,4], nargs='+', help="Convolutional layer widths")
    
args = parser.parse_args()

L=args.L

lat=[L,L]

lam = args.g
mas = args.m

Nwarm = args.Nwarm
Nskip = args.Nskip
hmc_batch_size = args.hmc_bs

Vol = np.prod(lat)


sg = qft.phi4(lat,lam,mas,batch_size=hmc_batch_size,device=device)

phi = sg.hotStart()
mn2 = integ.minnorm2(sg.force,sg.evolveQ,args.Nmd,1.0)
 
print("HMC Initial field characteristics: ",phi.shape,Vol,tr.mean(phi),tr.std(phi))

hmc = upd.hmc(T=sg,I=mn2,verbose=False)

tic=time.perf_counter()
phi = hmc.evolve(phi,Nwarm)
toc=time.perf_counter()
print(f"time {(toc - tic)*1.0e3/Nwarm:0.4f} ms per HMC trajecrory")
print("Acceptance: ",hmc.calc_Acceptance())


load_flag=True
if(args.f=="no-load"):
   load_flag=False

activ=activation_factory(args.activ)

funct=model_factory(args.model,L=L,
                    y=args.tau,
                    conv_layers=args.conv_l,
                    activation=activ,dtype=tr.float)

if(load_flag):
    funct.load_state_dict(tr.load(args.f))
    funct.eval()
funct.to(device)

muO = C2pt(phi,args.tau).mean().to("cpu").numpy().item()

CM = ControlModel(muO=muO,force = sg.force,c2p_net=funct)
CM.to(device)
print(CM)

hmc.AcceptReject = []
ll_hist, phi= train_control_model(CM,phi,learning_rate=args.lr,epochs=args.e,super_batch=1,update=lambda x : hmc.evolve(x,Nskip))
print('HMC acceptance: ',hmc.calc_Acceptance())
print('HMC history length: ',len(hmc.AcceptReject))


ff = args.f
if(not load_flag):
    ws = "_".join(str(l) for l in args.conv_l)
    ff = "cv_"+args.model+"_act_"+args.activ+"_cl_"+ws+".dict"
tr.save(funct.state_dict(), ff)


phi = hmc.evolve(phi,Nskip)

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

symmetry_checker(x,funct)

plt.plot(ll_hist)
plt.yscale('log')
plt.show()


