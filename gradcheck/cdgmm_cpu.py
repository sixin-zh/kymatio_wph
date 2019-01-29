
import torch
from torch.autograd import grad

nb=1
nc=1
M=2
N=2
A = torch.zeros((nb,nc,M,N,2),dtype=torch.float32)
B = torch.zeros((M,N,2),dtype=torch.float32)
print(A.shape,B.shape)

from kymatio.phaseharmonics2d.backend import cdgmm

C = cdgmm(A,B)
