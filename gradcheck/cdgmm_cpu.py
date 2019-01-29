
import torch
from torch.autograd import gradcheck

nb=1
nc=1
M=2
N=2
A = torch.zeros((nb,nc,M,N,2),dtype=torch.float32)
B = torch.zeros((M,N,2),dtype=torch.float32)

from kymatio.phaseharmonics2d.backend import cdgmm

#C = cdgmm(A,B)
A.requires_grad_(True)
gradcheck(cdgmm, (A,B))
