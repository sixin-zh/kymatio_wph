# TEST ON GPU
import os,sys

#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import Variable, grad

from time import time
import gc

sys.path.append(os.path.abspath(os.getcwd()))
from lbfgs_routine import *
#---- create image without/with marks----#

ks=int(sys.argv[1]) # ks

size = 256
Krec = 10

# --- Dirac example---#
imid = ks
data = sio.loadmat('./data/ns_randn4_train_N' + str(size) + '.mat')
im = data['imgs'][:,:,imid]
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

# Parameters for transforms
J = 5
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = int(sys.argv[2])
delta_l = L # int(sys.argv[3])
delta_k = 0 # 1
nb_chunks = 1
nb_restarts = 10 # 5
factr = 10 # 00
maxite = 500
maxcor = 20
#init = 'normal'
init = 'normalstdbarx'
stdn = 1

FOLOUT = '../results/tur2a/bump_lbfgs_gpu_N' + str(N) + 'J' + str(J) + 'L' + str(L) + 'dj' +\
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
for chunk_id in range(nb_chunks):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, stdnorm=stdn)
    wph_op = wph_op.cuda()
    wph_ops.append(wph_op)
    Sim_ = wph_op(im) # (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    print('wph coefficients',Sim_.shape[2])
    Sims.append(Sim_)
    factr_ops.append(factr)

call_lbfgs_routine(FOLOUT,labelname,im,wph_ops,Sims,N,Krec,nb_restarts,maxite,factr,factr_ops,init=init) # ,toskip=False)
