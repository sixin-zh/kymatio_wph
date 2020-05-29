import os
from time import time
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim

from utils_gpu import pos_to_im_fourier3, pos_to_im_fourier3b

def obj_func_id(x_t,wph_ops,wph_streams,Sims,factr2,hfs,oms,op_id,nGPU):
    # convert points to im on devid, using a loop
    devid = op_id % nGPU
    hf = hfs[devid]
    om = oms[devid]
    with torch.cuda.device(devid):
        torch.cuda.stream(wph_streams[devid])
        imf_t = pos_to_im_fourier3(x_t, hf, om)
        # compute wph grad on devid
        wph_op = wph_ops[op_id]
        p = wph_op(imf_t)
        diff = p-Sims[op_id]
        loss = torch.mul(diff,diff).sum()
        loss = loss*factr2
        
    return loss

def obj_func(x,wph_ops,wph_streams,Sims,factr2,hfs,oms,nGPU):
    loss = 0
    loss_a = []

    # copy all the points x to multiple gpus
    x_a = []
    for devid in range(nGPU):
        x_t = x.to(devid)
        x_a.append(x_t)
    
    # compute gradients with respect to x_a
    for op_id in range(len(wph_ops)):
        devid = op_id % nGPU
        x_t = x_a[devid]
        loss_t = obj_func_id(x_t,wph_ops,wph_streams,Sims,factr2,hfs,oms,op_id,nGPU)
        loss_t.backward(retain_graph=False) # accumulate grad into x.grad
        loss_a.append(loss_t)
        
    torch.cuda.synchronize() # wait all the gradients computation finish
    
    # sum the loss
    for op_id in range(len(wph_ops)):
        loss = loss + loss_a[op_id]
        
    return loss

def call_lbfgs2_routine(x0,hf,om,wph_ops,wph_streams,Sims,nb_restarts,maxite,factr,\
                        nGPU=2,maxcor=20,gtol=1e-14,ftol=1e-14):
    # x0 init points (no need to be on GPU)
    # hf,om: gaussian filter in Fourier and om
    # return x: optimal points
    assert(nGPU >= 2)

    # copy filter and om to each GPu
    hfs = []
    oms = []
    K = len(hf)
    for devid in range(nGPU):
        hfs.append([hf[k].to(devid) for k in range(K)])
        oms.append([om[k].to(devid) for k in range(K)])
    for start in range(nb_restarts+1):
        if start==0:
            x = x0.cuda()
            x.requires_grad_(True)
        time0 = time()
        optimizer = optim.LBFGS({x}, max_iter=maxite, line_search_fn='strong_wolfe',\
                                tolerance_grad = gtol, tolerance_change = ftol,\
                                history_size = maxcor)
        
        def closure():
            optimizer.zero_grad()
            loss = obj_func(x,wph_ops,wph_streams,Sims,factr**2,hfs,oms,nGPU)
            return loss

        optimizer.step(closure)

        opt_state = optimizer.state[optimizer._params[0]]
        niter = opt_state['n_iter']
        final_loss = opt_state['prev_loss']
        print('At restart',start,'OPT fini avec:', final_loss,niter,'in',time()-time0,'sec')
        
    return x
