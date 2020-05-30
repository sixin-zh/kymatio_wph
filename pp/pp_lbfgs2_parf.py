import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import grad
import scipy.optimize as opt
import torch.nn.functional as F

import sys
from utils_gpu import pos_to_im_fourier3, pos_to_im_fourier3b, get_hf_om
from lbfgs2_routine_parf import call_lbfgs2_routine

size = 256 # 128
res = size
sigma = 4

#filename = './poisson_vor_150_100.txt'
#pos = size*np.loadtxt(fname=filename, delimiter=',', skiprows=1, usecols=(1,2))

filename = './turb_zoom_cluster.txt' # N=256
pos = np.loadtxt(fname=filename, delimiter=' ', skiprows=1, usecols=(1,2))

nb_points = pos.shape[0]

x = torch.from_numpy(pos.T).type(torch.float).cuda() # (2,nbp)
index = torch.arange(0, res).type(torch.float).cuda().unsqueeze(0)
hf, om = get_hf_om(index,res,sigma,np.pi) # hf: (1,res)
imf = pos_to_im_fourier3b(x, hf, om)
print('imf',imf.shape)
print('nb points',nb_points)

from kymatio.phaseharmonics2d.utils import ifft2_c2r
plt.figure()
im_ = ifft2_c2r(imf)
plt.imshow(im_[0,0,:,:].cpu())
plt.show()

# Parameters for transforms
J = 5 # 4
L = 8 # 4
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
factr = 1e3 # 7
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

x0 = torch.torch.Tensor(2, nb_points).uniform_(0,size)
maxite = 10 # 30 # 0
x_fin = call_lbfgs2_routine(x0,hf,om,wph_ops,wph_streams,Sims,nb_restarts,maxite,factr,nGPU)
