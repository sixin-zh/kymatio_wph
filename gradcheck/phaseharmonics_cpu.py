
import torch
from torch.autograd import gradcheck

Pc=2
M=2
N=2
A = torch.randn((1,Pc,M,N,2),dtype=torch.float32)
k = torch.ones((1,Pc,1,1),dtype=torch.float32)*1

from kymatio.phaseharmonics2d.backend import PhaseHarmonic
phase_harmonics = PhaseHarmonic(check_for_nan=False)

A.requires_grad_(True)
result = gradcheck(phase_harmonics, (A,k), eps=1e-4)
if result:
    print('pass')
else:
    print('not pass')
    
