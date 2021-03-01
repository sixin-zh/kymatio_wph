# RUN ON 1 GPU
import os,sys
from time import time

import numpy as np
import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import Variable, grad
sys.path.append(os.path.abspath(os.getcwd()))
from lbfgs_routine import *

ks = int(sys.argv[1]) # 0,1,2,3,...
name = str(sys.argv[2]) # tur2a, ...

#print(name)

size = 256 # image size
Krec = 1 # 10 # number of synthesis 

# --- load training dataset---#
imid = ks # image id
if name == 'tur2a':   
    data = sio.loadmat('./data/ns_randn4_train_N' + str(size) + '.mat')
elif name == 'anisotur2a':
    data = sio.loadmat('./data/ns_randn4_aniso_train_N' + str(size) + '.mat')
elif name == 'bubbles':
    data = sio.loadmat('./data/demo_brDuD111_N' + str(size) + '.mat')
elif name == 'mrw2dd':
    data = sio.loadmat('./data/demo_mrw2dd_train_N' + str(size) + '.mat')
else:
    assert(0)

# pick bar x whose image id = imid
im = data['imgs'][:,:,imid]
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

# Parameters for wph transforms
labelname = 'modelB'
J = 5
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 0
delta_l = 4
delta_n = 2
delta_k = 0
maxk_shift = 1
nb_chunks = 4
nb_restarts = 10
factr = 10
maxite = 500
maxcor = 20
init = 'normalstdbarx'

# Output path
FOLOUT = '../results/' + name + '/bump_lbfgs_gpu_N' + str(N) + 'J' + str(J) + 'L' + str(L) + 'dj' +\
         str(delta_j) + 'dl' + str(delta_l) + 'dk' + str(delta_k) + 'dn' + str(delta_n) +\
         '_maxkshift' + str(maxk_shift) +\
         '_factr' + str(int(factr)) + 'maxite' + str(maxite) +\
         'maxcor' + str(maxcor) + '_init' + init +\
         '_ks' + str(ks) + 'ns' + str(nb_restarts)
os.system('mkdir -p ' + FOLOUT)

# load kymatio wph
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_non_isotropic \
    import PhaseHarmonics2d
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_fftshift2d \
    import PhaseHarmonics2d as wphshift2d

Sims = []
wph_ops = []
factr_ops = []
nCov = 0
total_nbcov = 0
for chunk_id in range(J+1):
    wph_op = wphshift2d(M,N,J,L,delta_n,maxk_shift,J+1,chunk_id,submean=1,stdnorm=1)
    if chunk_id ==0:
        total_nbcov += wph_op.nbcov
    wph_op = wph_op.cuda()
    wph_ops.append(wph_op)
    Sim_ = wph_op(im) # (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    print('wph coefficients',Sim_.shape[2])
    Sims.append(Sim_)
    factr_ops.append(factr)

for chunk_id in range(nb_chunks):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k,
                              nb_chunks, chunk_id, submean=1, stdnorm=1)
    if chunk_id ==0:
        total_nbcov += wph_op.nbcov
    wph_op = wph_op.cuda()
    wph_ops.append(wph_op)
    Sim_ = wph_op(im) # output size: (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    print('wph coefficients',Sim_.shape[2])
    Sims.append(Sim_)
    factr_ops.append(factr)

print('total nbcov is',total_nbcov)

# synthesis
call_lbfgs_routine(FOLOUT,labelname,im,wph_ops,Sims,N,Krec,
                   nb_restarts,maxite,factr,factr_ops,init=init) # ,toskip=False)
