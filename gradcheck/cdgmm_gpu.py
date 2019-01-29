
import torch
from torch.autograd import gradcheck

nb=1
nc=1
M=2
N=2
A = torch.randn((nb,nc,M,N,2),dtype=torch.float32).cuda()
B = torch.randn((M,N,2),dtype=torch.float32).cuda()

from kymatio.phaseharmonics2d.backend import cdgmm

#C = cdgmm(A,B)
A.requires_grad_(True)
B.requires_grad_(True)
result = gradcheck(cdgmm, (A,B))
if result:
    print('pass')
else:
    print('not pass')
    
