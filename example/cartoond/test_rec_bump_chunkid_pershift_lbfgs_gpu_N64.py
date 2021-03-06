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

size=64

# --- Dirac example---#
data = sio.loadmat('./example/cartoond/demo_toy7d_N' + str(size) + '.mat')
im = data['imgs']
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

# Parameters for transforms

J = 6
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 1
delta_l = L/2
delta_k = 1
delta_n = 1
nb_chunks = 10
factr = 1e3

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid \
    import PhaseHarmonics2d
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_pershift \
    import PHkPerShift2d

Sims = []
wph_ops = []
opid = 0
for dn1 in range(-delta_n,delta_n+1):
    for dn2 in range(-delta_n,delta_n+1):
        if dn1**2+dn2**2 <= delta_n**2:
            for chunk_id in range(J):
                wph_op = PHkPerShift2d(M, N, J, L, dn1, dn2, delta_l, J, chunk_id) 
                wph_op = wph_op.cuda()
                wph_ops.append(wph_op)
                assert(wph_ops[opid]==wph_op)
                opid += 1
                Sim_ = wph_op(im)*factr # (nb,nc,nb_channels,1,1,2)
                Sims.append(Sim_)

# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#
#---- Trying scipy L-BFGS ----#

def obj_fun(x,opid):
    if x.grad is not None:
        x.grad.data.zero_()
    global wph_ops
    wph_op = wph_ops[opid]
    p = wph_op(x)*factr
    diff = p-Sims[opid]
    loss = torch.mul(diff,diff).mean()
    return loss

grad_err = im.clone()

def grad_obj_fun(x_gpu):
    loss = 0
    global grad_err
    grad_err[:] = 0
    for opid in range(len(Sims)):
        x_t = x_gpu.clone().requires_grad_(True)
        loss_t = obj_fun(x_t,opid)
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
    if count%10 == 0:
        print(count, loss, 'using time (sec):' , time()-time0)
        time0 = time()
    return loss.cpu().item(), np.asarray(grad_err.reshape(size**2).cpu().numpy(), dtype=np.float64)

#float(loss)
def callback_print(x):
    return

x = torch.Tensor(1, 1, N, N).normal_(std=0.01)+0.5
x0 = x.reshape(size**2).numpy()
x0 = np.asarray(x0, dtype=np.float64)

res = opt.minimize(fun_and_grad_conv, x0, method='L-BFGS-B', jac=True, tol=None,
                   callback=callback_print,
                   options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 100})
final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']
print('OPT fini avec:', final_loss,niter,msg)

im_opt = np.reshape(x_opt, (size,size))
tensor_opt = torch.tensor(im_opt, dtype=torch.float).unsqueeze(0).unsqueeze(0)

ret = dict()
ret['tensor_opt'] = tensor_opt
ret['normalized_loss'] = final_loss/(factr**2)

torch.save(ret, 'test_rec_bump_chunkid_pershift_lbfgs_gpu_N64.pt')

