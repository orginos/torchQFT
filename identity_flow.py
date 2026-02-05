
class IdentityFlow(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def g(self, z):
        # Forward: z -> x. Returns x, log_det
        # Identity: x = z, log_det = 0
        return z, z.new_zeros(z.shape[0])
    
    def f(self, x):
        # Backward: x -> z. Returns z, log_det_inv
        # Identity: z = x, log_det = 0
        return x, x.new_zeros(x.shape[0])
