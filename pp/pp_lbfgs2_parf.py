import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import grad
import scipy.optimize as opt
import torch.nn.functional as F

import sys
from utils_gpu import pos_to_im_fourier3, get_hf_om, pos_to_im3
from lbfgs2_routine_parf import call_lbfgs2_routine

size = 128 # 256 # 128
res = size # 128
sigma = 4

filename = './poisson_vor_150_100.txt'
pos = size*np.loadtxt(fname=filename, delimiter=',', skiprows=1, usecols=(1,2))
nb_points = pos.shape[0]

x_ = torch.from_numpy(pos).type(torch.float).cuda()
res_ = torch.tensor(res).type(torch.float).cuda()
Mx_ =  torch.arange(0, res).type(torch.float).cuda()
My_ = torch.arange(0, res).type(torch.float).cuda()
pi_ = torch.from_numpy(np.array([np.pi])).cuda()

from kymatio.phaseharmonics2d.utils import ifft2_c2r
im = pos_to_im3(x_, res_, Mx_, My_, pi_, sigma)

index = Mx_.unsqueeze(0)
hf, om = get_hf_om(index,res,sigma,pi_) # hf: (1,res)
imf = pos_to_im_fourier3(x_, hf, om) # res_, Mx_, My_, pi_, sigma)

plt.figure()
plt.imshow(im[0,0,:,:].cpu())

plt.figure()
im_ = ifft2_c2r(imf)
plt.imshow(im_[0,0,:,:].cpu())

print('im diff',torch.norm(im_ - im))

plt.show()

print('imf',imf.shape)
print('nb points',nb_points)

# Parameters for transforms
J = 4 # 5 # 4
L =4 # 8 # 4
M, N = imf.shape[2], imf.shape[3]
delta_j = 0
delta_l = L/2
delta_k = 0
nb_chunks = 2
nb_restarts = 1
nGPU = 2
    
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_simplephase \
    import PhaseHarmonics2d

wph_streams = []
for devid in range(nGPU):
    with torch.cuda.device(devid):
        s = torch.cuda.Stream()
        wph_streams.append(s)
        
Sims = []
factr = 1e7
wph_ops = dict()
nCov = 0
opid = 0
for chunk_id in range(nb_chunks+1):
    devid = opid % nGPU
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, devid, fourier_input=1)
    wph_op = wph_op.cuda()
    wph_ops[chunk_id] = wph_op
    imf_ = imf.to(devid)
    with torch.cuda.device(devid):
        torch.cuda.stream(wph_streams[devid])
        Sim_ = wph_op(imf_) # TO *factr it internally
        nCov += Sim_.shape[2]
        opid += 1
        Sims.append(Sim_)

torch.cuda.synchronize()
    
x0 = torch.torch.Tensor(nb_points, 2).uniform_(0,size)
maxite = 30 # 0
x_fin = call_lbfgs2_routine(x0,sigma,res,wph_ops,wph_streams,Sims,nb_restarts,maxite,factr,nGPU)
