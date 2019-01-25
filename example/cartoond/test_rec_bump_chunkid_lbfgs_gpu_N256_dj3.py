# TEST ON GPU

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import Variable, grad

#device_id = 0
#torch.cuda.set_device(device_id)

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
delta_j = 3
delta_l = L/2
delta_k = 1
nb_chunks = 30

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid \
    import PhaseHarmonics2d

Sims = []
factr = 1e3
wph_ops = dict()
for chunk_id in range(nb_chunks+1):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
    wph_op = wph_op.cuda()
    wph_ops[chunk_id] = wph_op
    Sim_ = wph_op(im)*factr # (nb,nc,nb_channels,1,1,2)
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
    loss = torch.mul(diff,diff).mean()
    return loss

grad_err = im.clone()

def grad_obj_fun(x_gpu):
    loss = 0
    global grad_err
    grad_err[:] = 0
    #global wph_ops
    for chunk_id in range(nb_chunks+1):
        x_t = x_gpu.clone().requires_grad_(True)
        #print('chunk_id in grad', chunk_id)
        #if chunk_id not in wph_ops.keys():
        #    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
        #    wph_op = wph_op.cuda()
        #    wph_ops[chunk_id] = wph_op
        loss = loss + obj_fun(x_t,chunk_id)
        grad_err_, = grad([loss],[x_t], retain_graph=False)
        grad_err = grad_err + grad_err_
        #x_t.detach()
        #del x_t
        #del grad_err_
        #del wph_ops[chunk_id]
        #gc.collect()
        
    return loss, grad_err

count = 0
def fun_and_grad_conv(x):
    x_float = torch.reshape(torch.tensor(x,dtype=torch.float),(1,1,size,size))
    x_gpu = x_float.cuda()#.requires_grad_(True)
    loss, grad_err = grad_obj_fun(x_gpu)
    #del x_gpu
    #gc.collect()
    global count
    count += 1
    if count%40 == 1:
        print(loss)
    return  loss.cpu().item(), np.asarray(grad_err.reshape(size**2).cpu().numpy(), dtype=np.float64)

#float(loss)
def callback_print(x):
    return

x = torch.Tensor(1, 1, N, N).normal_(std=0.1)
#x[0,0,0,0] = 2
#x = x.clone().detach().requires_grad_(True) # torch.tensor(x, requires_grad=True)
x0 = x.reshape(size**2).detach().numpy()
x0 = np.asarray(x0, dtype=np.float64)

res = opt.minimize(fun_and_grad_conv, x0, method='L-BFGS-B', jac=True, tol=None,
                   callback=callback_print,
                   options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 100})
final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']

im_opt = np.reshape(x_opt, (size,size))
tensor_opt = torch.tensor(im_opt, dtype=torch.float).unsqueeze(0).unsqueeze(0)

torch.save(tensor_opt, 'test_rec_bump_chunkid_lbfgs_gpu_N256_dj3.pt')

#tensor_opt = torch.tensor(im_opt, dtype=torch.float).unsqueeze(0).unsqueeze(0)
#plt.figure()
#im_opt = np.reshape(x_opt, (size,size))
#plt.imshow(im_opt)
#plt.show()
