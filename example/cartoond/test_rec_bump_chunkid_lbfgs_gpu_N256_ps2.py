# TEST ON GPU
import os,sys
FOLOUT = sys.argv[1] # store the result in output folder
os.system('mkdir -p ' + FOLOUT)

#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import grad # Variable, grad

from time import time

#---- create image without/with marks----#

size=256

# --- Dirac example---#
data = sio.loadmat('./data/demo_toy7d_N' + str(size) + '.mat')
im = data['imgs']
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

# Parameters for transforms
J = 8
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 3 # int(sys.argv[2])
delta_l = L/2
delta_k = 1
delta_n = 0 # int(sys.argv[3])
nb_chunks = 40 # int(sys.argv[4])
nb_restarts = 10
factr = 1e5
factr2 = factr**2

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_pershift \
    import PHkPerShift2d
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_scaleinter \
    import PhkScaleInter2d

Sims = []
wph_ops = []
opid = 0
nCov = 0
devid = 0
for dn1 in range(-delta_n,delta_n+1):
    for dn2 in range(-delta_n,delta_n+1):
        if dn1**2+dn2**2 <= delta_n**2:
            for chunk_id in range(J):
                if dn1==0 and dn2==0:
                    wph_op = PHkPerShift2d(M, N, J, L, dn1, dn2, delta_l, J, chunk_id)
                else:
                    wph_op = PHkPerShift2d(M, N, J, L, dn1, dn2, 0, J, chunk_id) 
                wph_op = wph_op.cuda()
                wph_ops.append(wph_op)
                assert(wph_ops[opid]==wph_op)
                opid += 1
                im_ = im.cuda()
                Sim_ = wph_op(im_) # (nb,nc,nb_channels,1,1,2)
                nCov += Sim_.shape[2]
                Sims.append(Sim_)

for chunk_id in range(nb_chunks+1):
    wph_op = PhkScaleInter2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, devid)
    wph_op = wph_op.cuda()
    wph_ops.append(wph_op)
    assert(wph_ops[opid]==wph_op)
    opid += 1
    im_ = im.cuda()
    Sim_ = wph_op(im_) # (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    Sims.append(Sim_)

print('total ops is', len(wph_ops))
print('total cov is', nCov)

# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#
#---- Trying scipy L-BFGS ----#

def obj_fun(x,opid):
    #if x.grad is not None:
    #    x.grad.data.zero_()
    global wph_ops
    wph_op = wph_ops[opid]
    p = wph_op(x)
    diff = p-Sims[opid]
    loss = factr2*torch.mul(diff,diff).sum()/nCov
    return loss

grad_err = im.cuda()

def grad_obj_fun(x_t):
    loss_a = []
    grad_err_a = []

    loss = 0
    global grad_err
    grad_err[:] = 0
    
    for opid in range(len(wph_ops)):
        loss_ = obj_fun(x_t,opid)
        loss_a.append(loss_)
        grad_err_, = grad([loss_],[x_t], retain_graph=False)
        grad_err = grad_err + grad_err_
        loss = loss + loss_a[opid]
         
    return loss, grad_err

count = 0
from time import time
time0 = time()
def fun_and_grad_conv(x):
    x_float = torch.reshape(torch.tensor(x,dtype=torch.float),(1,1,size,size))
    x_gpu = x_float.cuda().requires_grad_(True)
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

seed = 2018
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

x = torch.Tensor(1, 1, N, N).normal_(std=0.01)+0.5
x0 = x.reshape(size**2).numpy()
x0 = np.asarray(x0, dtype=np.float64)

for start in range(nb_restarts):
    if start==0:
        x_opt = x0
    res = opt.minimize(fun_and_grad_conv, x_opt, method='L-BFGS-B', jac=True, tol=None,
                       callback=callback_print,
                       options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 20})
    final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']
    print('OPT fini avec:', final_loss,niter,msg)

    im_opt = np.reshape(x_opt, (size,size))
    tensor_opt = torch.tensor(im_opt, dtype=torch.float).unsqueeze(0).unsqueeze(0)

    ret = dict()
    ret['tensor_opt'] = tensor_opt
    ret['normalized_loss'] = final_loss/(factr2)
    torch.save(ret, FOLOUT + '/' + 'test_rec_bump_chunkid_lbfgs_gpu_N256_ps2' + '_dn' + str(delta_n) + '_dj' + str(delta_j) + '.pt')
