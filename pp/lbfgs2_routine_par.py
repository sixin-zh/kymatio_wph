import os
from time import time
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from utils_gpu import pos_to_im3

def obj_func_id(x_t,wph_ops,wph_streams,Sims,factr2,sigma,ress,Mxs,Mys,pis,op_id,nGPU):
    # convert points to im on devid, using a loop
    devid = op_id % nGPU
    res_t = ress[devid]
    Mx_t = Mxs[devid]
    My_t = Mys[devid]
    pi_t = pis[devid]
    with torch.cuda.device(devid):
        torch.cuda.stream(wph_streams[devid])
        im_t = pos_to_im3(x_t, res_t, Mx_t, My_t, pi_t, sigma)
        # compute wph grad on devid
        wph_op = wph_ops[op_id]
        p = wph_op(im_t)
        diff = p-Sims[op_id]
        loss = torch.mul(diff,diff).sum()
        loss = loss*factr2
        
    return loss

'''    
def obj_func_id_par2(x,wph_ops,wph_streams,Sims,factr2,sigma,ress,Mxs,Mys,pis,op_id,nGPU):
    # cut points to nGPU, then compute each pieced images
    xlen = x.shape[0]
    avglen = int(xlen/nGPU)
    offset = 0
    im_a = []
    for devid in range(nGPU):
        res_t = ress[devid]
        Mx_t = Mxs[devid]
        My_t = Mys[devid]
        pi_t = pis[devid]
        if devid < nGPU-1:
            x_t = x.narrow(0,offset,avglen).to(devid)
        else:
            x_t = x.narrow(0,offset,xlen-offset).to(devid)
        with torch.cuda.device(devid):
            torch.cuda.stream(wph_streams[devid])
            im_t = pos_to_im3(x_t, res_t, Mx_t, My_t, pi_t, sigma)
            im_a.append(im_t)
        offset += avglen
    #torch.cuda.synchronize()
    
    # sum to an im on gpu 0
    for devid in range(nGPU):        
        if devid == 0:
            im = im_a[devid].to(0)
        else:
            im += im_a[devid].to(0)

    # compute wph grad on devid
    devid = op_id % nGPU
    im_ = im.to(devid)
    wph_op = wph_ops[op_id]
    with torch.cuda.device(devid):
        torch.cuda.stream(wph_streams[devid])
        p = wph_op(im_)
        diff = p-Sims[op_id]
        loss = torch.mul(diff,diff).sum()
        loss = loss*factr2
    
    return loss
'''
def obj_func(x,wph_ops,wph_streams,Sims,factr2,sigma,ress,Mxs,Mys,pis,nGPU):
    loss = 0
    loss_a = []

    # copy x to multiple gpus
    x_a = []
    for devid in range(nGPU):
        x_t = x.to(devid)
        x_a.append(x_t)
        
    # compute gradients with respect to x_a
    for op_id in range(len(wph_ops)):
        devid = op_id % nGPU
        x_t = x_a[devid]
        loss_t = obj_func_id(x_t,wph_ops,wph_streams,Sims,factr2,sigma,ress,Mxs,Mys,pis,op_id,nGPU)
        loss_t.backward(retain_graph=False) # accumulate grad into x.grad
        loss_a.append(loss_t)
        
    torch.cuda.synchronize()
    
    # sum them to grad of x   
    for op_id in range(len(wph_ops)):
        loss = loss + loss_a[op_id]
        
    return loss

def call_lbfgs2_routine(x0,sigma,res,wph_ops,wph_streams,Sims,nb_restarts,maxite,factr,\
                        nGPU=2,maxcor=20,gtol=1e-14,ftol=1e-14):
    # x0 init points (no need to be on GPU)
    # sigma: gaussian width
    # return x: optimal points
    assert(nGPU >= 2)
    ress = []
    Mxs = []
    Mys = []
    pis = []
    x_a = []
    for devid in range(nGPU):
        res_ = torch.tensor(res).type(torch.float).cuda().to(devid)
        Mx_ =  torch.arange(0, res).type(torch.float).cuda().to(devid)
        My_ = torch.arange(0, res).type(torch.float).cuda().to(devid)
        pi_ = torch.from_numpy(np.array([np.pi])).float().cuda().to(devid)
        ress.append(res_)
        Mxs.append(Mx_)
        Mys.append(My_)
        pis.append(pi_)
    
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
            loss = obj_func(x,wph_ops,wph_streams,Sims,factr**2,sigma,ress,Mxs,Mys,pis,nGPU)
            return loss

        optimizer.step(closure)

        opt_state = optimizer.state[optimizer._params[0]]
        niter = opt_state['n_iter']
        final_loss = opt_state['prev_loss']
        print('At restart',start,'OPT fini avec:', final_loss,niter,'in',time()-time0,'sec')
        
    return x
