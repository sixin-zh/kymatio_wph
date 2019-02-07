
import torch
from torch.autograd import gradcheck

nb=1
nc=1
M=4
N=4
X = torch.randn((nb,nc,M,N,2),dtype=torch.float32)
shift1 = 2
shift2 = 0

from kymatio.phaseharmonics2d.backend import PeriodicShift2D
pershift = PeriodicShift2D(M,N,shift1,shift2)

Y = pershift(X)
print(X[0,0,:,:,0],Y[0,0,:,:,0])

X.requires_grad_(True)
result = gradcheck(pershift, X, eps=1e-4, atol=1e-2, rtol=1e-3)
if result:
    print('pass')
else:
    print('not pass')
    
