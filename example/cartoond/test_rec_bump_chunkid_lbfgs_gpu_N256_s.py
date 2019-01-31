# TEST ON GPU
import os,sys
FOLOUT = sys.argv[1] # store the result in output folder
os.execute('mkdir -p ' + FOLOUT)

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
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

# Parameters for transforms
J = 8
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 1
delta_l = L/2
delta_k = 1
nb_chunks = 10
nb_restarts = 10

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_simplephase \
    import PhaseHarmonics2d

Sims = []
factr = 1e5
wph_ops = dict()
nCov = 0
for chunk_id in range(nb_chunks+1):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
    wph_op = wph_op.cuda()
    wph_ops[chunk_id] = wph_op
    Sim_ = wph_op(im)*factr # (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    Sims.append(Sim_)
    
# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#
#---- Trying scipy L-BFGS ----#
def obj_fun(x,chunk_id):
    if x.grad is not None:
        x.grad.data.zero_()
    global wph_ops
    wph_op = wph_ops[chunk_id]
    p = wph_op(x)*factr
    diff = p-Sims[chunk_id]
    loss = torch.mul(diff,diff).sum()/nCov
    return loss

grad_err = im.clone()

def grad_obj_fun(x_gpu):
    loss = 0
    global grad_err
    grad_err[:] = 0
    for chunk_id in range(nb_chunks+1):
        x_t = x_gpu.clone().requires_grad_(True)
        loss_t = obj_fun(x_t,chunk_id)
        grad_err_t, = grad([loss_t],[x_t], retain_graph=False)
        loss = loss + loss_t
        grad_err = grad_err + grad_err_t
  
    return loss, grad_err

count = 0
from time import time
time0 = time()
def fun_and_grad_conv(x):
    x_float = torch.reshape(torch.tensor(x,dtype=torch.float),(1,1,size,size))
    x_gpu = x_float.cuda()
    loss, grad_err = grad_obj_fun(x_gpu)
    global count
    global time0
    count += 1
    if count%10 == 1:
        print(count, loss, 'using time (sec):' , time()-time0)
        time0 = time()
    return loss.cpu().item(), np.asarray(grad_err.reshape(size**2).cpu().numpy(), dtype=np.float64)

def callback_print(x):
    return

x = torch.Tensor(1, 1, N, N).normal_(std=0.01)+0.5
x0 = x.reshape(size**2).numpy()
x0 = np.asarray(x0, dtype=np.float64)

for start in range(nb_restarts):
    if start==0:
        x_opt = x0
    #res = opt.minimize(fun_and_grad_conv, x_opt, method='L-BFGS-B', jac=True, tol=None,
    #                   callback=callback_print,
    #                   options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 20})
    #final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']
    #print('OPT fini avec:', final_loss,niter,msg)

im_opt = np.reshape(x_opt, (size,size))
tensor_opt = torch.tensor(im_opt, dtype=torch.float).unsqueeze(0).unsqueeze(0)

ret = dict()
ret.tensor_opt = tensor_opt
ret.normalized_loss = final_loss/(factr**2)
torch.save(ret, FOLOUT + '/' + 'test_rec_bump_chunkid_lbfgs_gpu_N256_dj1_simplephase.pt')
