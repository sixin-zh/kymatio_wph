import numpy as np
import torch
#import matplotlib.pyplot as plt
from torch.autograd import grad
import scipy.optimize as opt
import torch.nn.functional as F

import sys
from utils_gpu import pos_to_im3
from lbfgs2_routine_par import call_lbfgs2_routine

size = 256 # 128
res = size # 128
sigma = 8

filename = './poisson_vor_150_100.txt'
pos = size*np.loadtxt(fname=filename, delimiter=',', skiprows=1, usecols=(1,2))
nb_points = pos.shape[0]

x_ = torch.from_numpy(pos).type(torch.float).cuda()
res_ = torch.tensor(res).type(torch.float).cuda()
Mx_ =  torch.arange(0, res).type(torch.float).cuda()
My_ = torch.arange(0, res).type(torch.float).cuda()
pi_ = torch.from_numpy(np.array([np.pi])).float().cuda()
im = pos_to_im3(x_, res_, Mx_, My_, pi_, sigma)

print('im',im.shape)
print('nb points',nb_points)

# Parameters for transforms
J = 5 # 4
L = 8 # 4
M, N = im.shape[-2], im.shape[-1]
delta_j = 0
delta_l = L/2
delta_k = 0
nb_chunks = 4
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
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, devid)
    wph_op = wph_op.cuda()
    wph_ops[chunk_id] = wph_op
    im_ = im.to(devid)
    with torch.cuda.device(devid):
        torch.cuda.stream(wph_streams[devid])
        Sim_ = wph_op(im_) # TO *factr it internally
        nCov += Sim_.shape[2]
        opid += 1
        Sims.append(Sim_)

torch.cuda.synchronize()
    
x0 = torch.torch.Tensor(nb_points, 2).uniform_(0,size)
maxite = 30 # 0
x_fin = call_lbfgs2_routine(x0,sigma,res,wph_ops,wph_streams,Sims,nb_restarts,maxite,factr,nGPU)
