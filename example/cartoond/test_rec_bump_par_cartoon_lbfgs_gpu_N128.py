# TEST ON GPU

#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import gc

import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import Variable, grad

from time import time

#---- create image without/with marks----#

size=128

# --- Dirac example---#

data = sio.loadmat('./example/cartoond/demo_toy7d_N128.mat')
im = data['imgs']
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

# Parameters for transforms

J = 7
L = 8
M, N = im.shape[-2], im.shape[-1]
j_max = 1
l_max = L/2
delta_k = 1
nGPU = 4
nChunksPerGPU=1

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_par \
    import PhaseHarmonics2d

wph_op = PhaseHarmonics2d(M, N, J, L, j_max, l_max, delta_k, nGPU, nChunksPerGPU)
wph_op = wph_op.cuda()

factr = 1e3
Sim = wph_op(im)*factr
#for key,val in Smeta.items():
#    print (key, "=>", val, ":", Sim[0,0,key,0,0,0], "+i ", Sim[0,0,key,0,0,1])
#print (Sim.shape)

# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#


#---- Trying scipy L-BFGS ----#

def f(a, b):
    p = wph_op(a)*factr
    diff = p-b
    return (torch.mul(diff,diff).mean(), p)

def closure():
    loss, Sx = f(x, Sim)
    loss_vals.append(loss.item())
    loss.backward()
    return loss


def obj_fun(x):
    if x.grad is not None:
        x.grad.data.zero_()
    loss, Sx = f(x, Sim)
    return loss


def grad_obj_fun(x):
    loss = obj_fun(x)
    grad_err, = grad([loss],[x], retain_graph=True)
    return loss, grad_err


count = 0
from time import time
time0 = time()
def fun_and_grad_conv(x):
    x_t = torch.reshape(torch.tensor(x, requires_grad=True,dtype=torch.float),
                        (1,1,size,size)).cuda()
    loss, grad_err = grad_obj_fun(x_t)

    global count
    count += 1
    global time0
    if count%10 == 1:
        print(count, loss, 'using time (sec):' , time()-time0)
        print('grad norm is', grad_err:norm())
        time0 = time()
        
    return  loss.cpu().item(), np.asarray(grad_err.reshape(size**2).cpu().numpy(), dtype=np.float64)


def callback_print(x):
    return

#    global count
#    count +=1
#    if count%40 == 1:
#        x_t = torch.reshape(torch.tensor(x, requires_grad=True,dtype=torch.float),
#                            (1,1,size,size)).cuda()
#        p = wph_op(x_t)*factr
#        diff = p-Sim
#        loss = torch.mul(diff,diff).mean()
#        print(loss)

        #nelement = 0
        #for obj in gc.get_objects():
        #    if torch.is_tensor(obj) or ((hasattr(obj, 'data') and torch.is_tensor(obj.data))):
        #        print(type(obj), obj.size())
        #        nelement += obj.nelement()

        #print('nelement floats', nelement, ', =', nelement*4/1024/1024, 'MB')
        
#        return loss

x = torch.Tensor(1, 1, N, N).normal_(std=0.01) + 0.5
#x[0,0,0,0] = 2
#x = torch.tensor(x, requires_grad=True)
x0 = x.reshape(size**2).numpy()
x0 = np.asarray(x0, dtype=np.float64)

res = opt.minimize(fun_and_grad_conv, x0, method='L-BFGS-B', jac=True, tol=None,
                   callback=callback_print,
                   options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 100})
final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']
print('OPT fini avec:', final_loss,niter,msg)

im_opt = np.reshape(x_opt, (size,size))
tensor_opt = torch.tensor(im_opt, dtype=torch.float).unsqueeze(0).unsqueeze(0)

torch.save(tensor_opt, './results/test_rec_bump_par_cartoon_lbfgs_gpu_N128_run2.pt')

#plt.figure()
#im_opt = np.reshape(x_opt, (size,size))
#plt.imshow(im_opt)
#plt.show()
