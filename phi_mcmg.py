import torch as tr


class phi4_c1:
    def action(self,phi_c):
        rphis=[]
        rphis.append(phi_c)
        iii=0
        logdet_total=tr.zeros(phi_c.shape[0],dtype=self.dtype,device=self.device)
        for pi in reversed(self.pics):
            #print(pi.shape)
            rphi = self.rg.refine(rphis[iii],pi)
            if self.mode=="rnvp":
                #flowback through the network
                rphi,logdet = self.mgf.cflow[self.level-1-iii].backward(rphi)
                logdet_total+=logdet
            #print("log_det",logdet,"level",self.level-1-iii)
            rphis.append(rphi)
            iii+=1
        #phi_f = rphis[-1]
        #evaluate coarse field in action of rg
        #print(phi_f.shape,"shape of fine field")
        return self.sg.action(rphi)-logdet_total
        #if I dont add the .sum() I got a grad for the batch system, it seems to me that we include that in the force property the batch is summed?

    def force(self, phi_c):
        x_tensor = phi_c.clone().requires_grad_()

        S = -self.action(x_tensor)
        grad = tr.autograd.grad(S.sum(), x_tensor, retain_graph=True)[0]

        if grad is None:
            print("[ERROR] Gradient is None.")
            raise RuntimeError("autograd.grad returned None.")

        return grad
    
    def refreshP(self):
        P = tr.normal(0.0,1.0,self.phis[-1].shape).to(self.device).to(self.dtype)#only difference with fine level
        return P

    def evolveQ(self,dt,P,Q):
        return Q + dt*P
    
    def kinetic(self,P):
        return tr.einsum('bxy,bxy->b',P,P)/2.0

    def generate_cfg_levels(self,phi11):#run every time we need to contruct deeper or superficial levels
        #run a configuration
        self.level = self.mgf.depth
        phis=[]
        pis=[]
        phicopy=phi11.clone().to(self.device).to(self.dtype)
        
        #print("shape of the original field",phicopy.shape)
        phis.append(phicopy)
        for step in range(self.level):
            if self.mode=="rnvp":
                phicopy= self.mgf.cflow[step].forward(phicopy)
                #print("device phicopy ",phicopy.device," dtype ",phicopy.dtype)
            #print("coarsening level ",_," field shape ",phicopy.shape)
            phic,pic = self.rg.coarsen(phicopy)

            phis.append(phic)
            pis.append(pic)
            phicopy=phic
        self.phis=phis
        self.pics=pis

        #reversed
        rphis=[]
        
        rphis.append(phis[-1])
        #self.mgf.cflow[step].backward(rphis[0])
        sss=0
        for phics,pis in zip(reversed(phis),reversed(pis)):
            
            rphi = self.rg.refine(phics,pis)
            #flowback through the network
            if self.mode=="rnvp":
                rphi,logdet = self.mgf.cflow[self.level-1-sss].backward(rphi)
                #print("log_det",logdet,"level",self.level-1-sss)
            sss+=1
            
            rphis.append(rphi)
        self.rphis=rphis

    def __init__(self,sgg,mgf,device="cpu",dtype=tr.float64,mode="rnvp"):
        self.sg = sgg #theory? in the finest level
        self.mgf = mgf #neural net
        self.rg = mgf.rg #projector to coarse level
        self.mode = mode
        print("multigrid is done by: ",self.mode)
        self.device = device
        self.dtype = dtype
