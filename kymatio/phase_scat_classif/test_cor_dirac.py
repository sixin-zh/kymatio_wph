import torch
import torchvision
from utils import rgb2yuv

from representation_complex import compute_phase_harmonic_cor

from complex_utils import complex_log
from utils import mean_std, standardize_feature

import numpy as np

# dirac

size=32

# --- Dirac example---#

im = np.zeros((size,size))
im[15,15] = 1
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0)

J = 3
L = 4
batch_size = 1
delta = 1
l_max = L

Sim = compute_phase_harmonic_cor(im, J, L, delta, l_max, batch_size)
print (Sim.shape)

