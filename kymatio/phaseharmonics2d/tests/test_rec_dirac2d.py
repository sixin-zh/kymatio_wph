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

J = 5
L = 4
M, N = im.shape[-2], im.shape[-1]
j_max = 1
l_max = L

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_simple \
    import PhaseHarmonics2d

wph_op = PhaseHarmonics2d(M, N, J, L, j_max, l_max)

Sim,Smeta = wph_op(im)
for key,val in Smeta.items():
    print (key, "=>", val, ":", Sim[0,0,key,0,0,0], "+i ", Sim[0,0,key,0,0,1])
print (Sim.shape)

# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#
'''

#---- Optimisation with torch----#
# recontruct x by matching || Sx - Sx0 ||^2
x = torch.zeros(1,1,N,N)
x[0,0,0,0]=2
x = Variable(x, requires_grad=True)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam([x], lr=0.01)
nb_steps = 1000
for step in range(0, nb_steps + 1):
    optimizer.zero_grad()
    SJx = scattering_op(x)
    loss = criterion(SJx, SJx0)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print('step',step,'loss',loss)

# plot x
plt.imshow(x.detach().cpu().numpy().reshape((M,N)))
plt.show()
'''
