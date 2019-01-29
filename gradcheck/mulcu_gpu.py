
import torch
from torch.autograd import gradcheck

nb=1
nc=1
M=2
N=2
A = torch.zeros((nb,nc,M,N,2),dtype=torch.float32).cuda()
B = torch.zeros((M,N,2),dtype=torch.float32).cuda()

from kymatio.phaseharmonics2d.backend import mulcu

A.requires_grad_(True)
B.requires_grad_(True)
result = gradcheck(mulcu, (A,B))
if result:
    print('pass')
else:
    print('not pass')
    
