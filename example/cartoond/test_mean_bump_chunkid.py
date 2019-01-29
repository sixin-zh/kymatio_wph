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

chunk_id = nb_chunks
wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
Sout = wph_op.compute_mean(im)
print('chunk mean', chunk_id, Sout)
print('im mean', im.mean())
