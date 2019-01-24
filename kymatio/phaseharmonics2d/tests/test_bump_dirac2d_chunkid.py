# TEST ON CPU

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt

import torch
from torch.autograd import Variable, grad

from time import time

#---- create image without/with marks----#

size=32

# --- Dirac example---#

im = np.zeros((size,size))
im[15,15] = 1
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0)


# Parameters for transforms

J = 3
L = 4
M, N = im.shape[-2], im.shape[-1]
delta_j = 1
delta_l = L
delta_k = 1

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid \
    import PhaseHarmonics2d

nb_chunks = 10
chunk_id = 0
wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)

Sim = wph_op(im) # (nb,nc,nb_channels,1,1,2)
print('sum', Sim.sum())
