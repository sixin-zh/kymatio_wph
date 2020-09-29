
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Jan 30 11:33:42 2019
@author: antoinebrochard
"""

import numpy as np
import torch
#import matplotlib.pyplot as plt
from torch.autograd import grad
import scipy.optimize as opt
import torch.nn.functional as F

import sys
from utils import weight, pos_to_im3
from lbfgs2_routine import call_lbfgs2_routine

#sys.path.append('/users/trec/brochard/kymatio_wph')
# load image

size = 128 # 256 # 128
res = size # 128
gpu = True
sigma = 4.0

torch.manual_seed(999)
torch.cuda.manual_seed_all(999)
 
#filename = './poisson_vor_150_100.txt'
#pos = size*np.loadtxt(fname=filename, delimiter=',', skiprows=1, usecols=(1,2))
filename = './turb_zoom_cluster.txt' # N=256
pos = np.loadtxt(fname=filename, delimiter=' ', skiprows=1, usecols=(1,2))

nb_points = pos.shape[0]
pos = torch.from_numpy(pos).type(torch.float)
im = pos_to_im3(pos, res, gpu, sigma)

print('nb points',nb_points)

# Parameters for transforms
J = 4
L = 4
M, N = im.shape[-2], im.shape[-1]
delta_j = 0
delta_l = L/2
delta_k = 0
nb_chunks = 4
nb_restarts = 1

from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_simplephase \
    import PhaseHarmonics2d

Sims = []
factr = 1e3 # 7
wph_ops = dict()
nCov = 0
for chunk_id in range(nb_chunks+1):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
    wph_op = wph_op.cuda()
    wph_ops[chunk_id] = wph_op
    Sim_ = wph_op(im) # TO do it internally *factr # (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    Sims.append(Sim_)
#print(1)

#reg = 0
#grad_err = torch.zeros(nb_points, 2).cuda()
x0 = torch.torch.Tensor(nb_points, 2).uniform_(0,size)
#x_orig = x0.clone().requires_grad_(False)
#if gpu:
#    x_orig = x_orig.cuda()
maxite = 10 # 300
x_fin = call_lbfgs2_routine(x0,sigma,res,wph_ops,Sims,nb_restarts,maxite,factr)

'''
def obj_fun(x,chunk_id):
    if x.grad is not None:
        x.grad.data.zero_()
    global wph_ops
    wph_op = wph_ops[chunk_id]
    im = pos_to_im3(x, size, gpu, sigma)
    p = wph_op(im)*factr
    diff = p-Sims[chunk_id]
    loss = torch.mul(diff,diff).sum()/nCov
    return loss

def grad_obj_fun(x_gpu):
    loss = 0
    global grad_err
    grad_err[:] = 0
    #global wph_ops
    for chunk_id in range(nb_chunks+1):
        x_t = x_gpu.clone().requires_grad_(True)
        x_t = x_t.cuda()
        loss_t = obj_fun(x_t,chunk_id)
        grad_err_t, = grad([loss_t],[x_t], retain_graph=False)
        loss = loss + loss_t
        grad_err = grad_err + grad_err_t
    x_t = x_t.cuda()
    loss_reg = reg*torch.norm(x_t-x_orig)**2
    grad_reg, = grad([loss_reg], [x_t], retain_graph=False)
    loss = loss + loss_reg
    grad_err = grad_err + grad_reg
    return loss, grad_err

count = 0
from time import time
time0 = time()
def fun_and_grad_conv(x):
    x_float = torch.reshape(torch.tensor(x, requires_grad=True,dtype=torch.float),
                            (x.shape[0]//2,2))
    loss, grad_err = grad_obj_fun(x_float)
    global count
    global time0
    if count%5 == 0:
        x_t = torch.reshape(torch.tensor(x, requires_grad=True,dtype=torch.float),
                            (x.shape[0]//2,2))
        im_t = pos_to_im3(x_t, res, gpu, sigma)
    if count%50 == 0:
        print(loss)
    count += 1
    return  loss.cpu().item(), np.asarray(grad_err.reshape(2*x_float.size(0)).cpu().numpy(), dtype=np.float64)

#float(loss)
def callback_print(x):
    return
'''

#x0 = x.reshape(2*x.size(0)).numpy()
#x0 = np.asarray(x0, dtype=np.float64)

#for start in range(nb_restarts + 1):
#    if start == 0:
#        x_opt = x0
#    result = opt.minimize(fun_and_grad_conv, x_opt, method='L-BFGS-B', jac=True, tol=None,
#                       callback=callback_print,
#                       options={'maxiter': 300, 'gtol': 1e-14, 'ftol': 1e-15, 'maxcor': 50#
#                                })
#    final_loss, x_opt, niter, msg = result['fun'], result['x'], result['nit'], result['message']
#    print('OPT fini avec:', final_loss,niter,msg)

#x_fin = (torch.reshape(torch.tensor(x_opt,dtype=torch.float),
#                       (x_opt.shape[0]//2,2))%res)/res
#torch.save(x_fin, './results/test_rot_1.pt')
