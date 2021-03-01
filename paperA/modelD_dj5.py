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

size = 256 # image size
Krec = 1 # 0 # number of synthesis

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

# Parameters for transforms
labelname = 'modelD_dj5'
J = 5
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 5
delta_l = L
delta_k = 0
nb_chunks = 1
nb_restarts = 10
factr = 10
maxite = 500
maxcor = 20
init = 'normalstdbarx'
kmax = 4 # for cov eval, to cut off large k,k'

FOLOUT = '../results/' + name + '/bump_lbfgs_gpu_N' + str(N) + 'J' + str(J) + 'L' + str(L) + 'dj' +\
         str(delta_j) + 'dl' + 'dk' + str(delta_k) +\
         '_factr' + str(int(factr)) + 'maxite' + str(maxite) +\
         'maxcor' + str(maxcor) + '_init' + init +\
         '_ks' + str(ks) + 'ns' + str(nb_restarts)
os.system('mkdir -p ' + FOLOUT)

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_isotropic_noreflect_norml \
    import PhaseHarmonics2d

Sims = []
wph_ops = []
factr_ops = []
nCov = 0
total_nbcov = 0
for chunk_id in range(nb_chunks):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, stdnorm=1, kmax=kmax)
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
