
import torch
from torch.autograd import grad

nb=1
nc=1
M=2
N=2
A = torch.FloatTensor((nb,nc,M,N,2))
B = torch.FloatTensor((M,N,2))

from kymatio.phaseharmonics2d.backend import cdgmm

C = cdgmm(A,B)
