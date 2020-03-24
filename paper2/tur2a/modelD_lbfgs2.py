# TEST ON GPU
import os,sys
from time import time

import numpy as np
import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import Variable, grad

sys.path.append(os.path.abspath(os.getcwd()))
from lbfgs2_routine import *

ks=int(sys.argv[1]) # id of input image

size = 256
Krec = 1 # 10

# --- Dirac example---#
imid = ks
data = sio.loadmat('./data/ns_randn4_train_N' + str(size) + '.mat')
im = data['imgs'][:,:,imid]
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

# Parameters for transforms
J = 5
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 1
delta_l = L
delta_k = 0
nb_chunks = 1
nb_restarts = 10
factr = 10
maxite = 500
maxcor = 20
init = 'normalstdbarx'
stdn = 1
kmax = None

FOLOUT = '../results/tur2a/bump_lbfgs2_gpu_N' + str(N) + 'J' + str(J) + 'L' + str(L) + 'dj' +\
         str(delta_j) + 'dl' + 'dk' + str(delta_k) +\
         '_factr' + str(int(factr)) + 'maxite' + str(maxite) +\
         'maxcor' + str(maxcor) + '_init' + init +\
         '_ks' + str(ks) + 'ns' + str(nb_restarts)
os.system('mkdir -p ' + FOLOUT)
labelname = 'eval_wph2_isonoref_norml_stdnorm' + str(stdn)

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_isotropic_noreflect_norml \
    import PhaseHarmonics2d

Sims = []
wph_ops = []
factr_ops = []
nCov = 0
total_nbcov = 0
for chunk_id in range(nb_chunks):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, stdnorm=stdn, kmax=kmax)
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
call_lbfgs2_routine(FOLOUT,labelname,im,wph_ops,Sims,N,Krec,nb_restarts,maxite,factr,factr_ops,init=init) # ,toskip=False)
