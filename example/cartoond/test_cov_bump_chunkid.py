# TEST ON GPU

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import Variable, grad

from time import time
import gc

#---- create image without/with marks----#

size=256

# --- Dirac example---#
data = sio.loadmat('./example/cartoond/demo_toy7d_N' + str(size) + '.mat')
im = data['imgs']
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0) # .cuda()

recim = torch.load('./results/test_rec_bump_chunkid_lbfgs_gpu_N256_dj1_restart.pt')

# Parameters for transforms
J = 8
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 1
delta_l = L/2
delta_k = 1
nb_chunks = 10

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid \
    import PhaseHarmonics2d

for chunk_id in range(nb_chunks):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
    Sim = wph_op.forward(im)
    Srec = wph_op.forward(recim)
    plt.figure()
    S1 = Sim[0,0,:,:,:,0].squeeze().numpy()
    S2 = Srec[0,0,:,:,:,0].squeeze().numpy()
    plt.plot(S1)
    plt.plot((S2-S1))
    plt.show()

