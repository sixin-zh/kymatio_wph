import os
from time import time
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from utils import pos_to_im3

gpu = True

def obj_func_id(x,wph_ops,Sims,factr2,sigma,res,op_id):
    wph_op = wph_ops[op_id]
    im = pos_to_im3(x, res, gpu, sigma)
    p = wph_op(im)
    diff = p-Sims[op_id]
    loss = torch.mul(diff,diff).sum()
    loss = loss*factr2    
    return loss

def obj_func(x,wph_ops,Sims,factr2,sigma,res):
    loss = 0
    if x.grad is not None:
        x.grad.data.zero_()
    for op_id in range(len(wph_ops)):
        loss_t = obj_func_id(x,wph_ops,Sims,factr2,sigma,res,op_id)
        loss_t.backward() # accumulate grad into x.grad
        loss = loss + loss_t
    return loss

def call_lbfgs2_routine(x0,sigma,res,wph_ops,Sims,nb_restarts,maxite,factr,\
                        maxcor=20,gtol=1e-14,ftol=1e-14):
    # x0 init points (no need to be on GPU)
    # sigma: gaussian width
    # return x: optimal points
    for start in range(nb_restarts+1):
        if start==0:
            x = x0.cuda()
            x.requires_grad_(True)
        time0 = time()
        optimizer = optim.LBFGS({x}, max_iter=maxite, max_eval=10*maxite, line_search_fn='strong_wolfe',\
                                tolerance_grad = gtol, tolerance_change = ftol,\
                                history_size = maxcor)
            
        def closure():
            optimizer.zero_grad()
            loss = obj_func(x,wph_ops,Sims,factr**2,sigma,res)
            return loss

        final_loss = optimizer.step(closure)

        opt_state = optimizer.state[optimizer._params[0]]
        niter = opt_state['n_iter']
        #final_loss = opt_state['prev_loss']
        print('At restart',start,'OPT fini avec:', final_loss,'niter',niter,'in',time()-time0,'sec')
               
    return x
    
