import os
from time import time
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from torch.autograd import grad
from utils_gpu import pos_to_im3

def cut_x(x,nGPU):
    xs = []
    nbp = x.shape[0]
    avgnbp = nbp//nGPU
    offset = 0
    for devid in range(nGPU):
        if devid < nGPU-1:
            x_t = x.narrow(0,offset,avgnbp).to(devid)
        else:
            x_t = x.narrow(0,offset,nbp-offset).to(devid)
        x_t.requires_grad_(True)
        xs.append(x_t)
        offset += avgnbp
    return xs

def cat_gradxs(x,gradxs,nGPU):
    nbp = x.shape[0]
    avgnbp = nbp//nGPU
    offset = 0
    for devid in range(nGPU):
        if devid < nGPU-1:
            x.grad.narrow(0,offset,avgnbp).copy_(gradxs[devid].detach())
        else:
            x.grad.narrow(0,offset,nbp-offset).copy_(gradxs[devid].detach())
        offset += avgnbp

def obj_func(x,wph_ops,wph_streams,Sims,factr2,sigma,ress,Mxs,Mys,pis,nGPU):
    # all points x: (nbp,2)
    # cut x to all gpus
    xs = cut_x(x,nGPU)
    
    # compute im from each xs, in parallel
    ims = []
    for devid in range(nGPU):
        with torch.cuda.device(devid):
            torch.cuda.stream(wph_streams[devid])
            x_t = xs[devid]
            res_t = ress[devid]
            Mx_t = Mxs[devid]
            My_t = Mys[devid]
            pi_t = pis[devid]
            im_t = pos_to_im3(x_t, res_t, Mx_t, My_t, pi_t, sigma)
            ims.append(im_t)
    torch.cuda.synchronize()
    
    # then sum ims into one im, then copy it to each chunk
    for devid in range(nGPU):
        im_t = ims[devid]
        if devid == 0:
            im = im_t.detach().to(0)
        else:
            im += im_t.detach().to(0)
    im_a = []
    for op_id in range(len(wph_ops)):
        devid = op_id % nGPU
        im_a.append(im.to(devid).requires_grad_(True))
        
    # compute gradients with respect to im, in parallel
    grad_a = []
    loss_a = []
    for op_id in range(len(wph_ops)):
        devid = op_id % nGPU
        with torch.cuda.device(devid):
            torch.cuda.stream(wph_streams[devid])
            # compute wph grad on devid
            im_t = im_a[op_id]
            wph_op_t = wph_ops[op_id]
            p_t = wph_op_t(im_t)
            diff_t = p_t-Sims[op_id]
            loss_t = torch.mul(diff_t,diff_t).sum()
            loss_t = loss_t*factr2
            grad_err_t, = grad([loss_t],[im_t], retain_graph=False)
            grad_a.append(grad_err_t)
            loss_a.append(loss_t)
    torch.cuda.synchronize()

    # sum the loss and grads
    # accumulate gradims into gradim
    loss = 0
    for op_id in range(len(wph_ops)):
        loss = loss + loss_a[op_id]
        if op_id == 0:
            gradim = grad_a[op_id].to(0)
        else:
            gradim += grad_a[op_id].to(0)

    # then copy gradim into all gpus
    # to compute gradient with respect to each group of points, in parallel
    gradims = []
    for devid in range(nGPU):
        gradims.append(gradim.to(devid))        
        
    gradxs = []
    for devid in range(nGPU):
        with torch.cuda.device(devid):
            torch.cuda.stream(wph_streams[devid])
            im_t = ims[devid]
            x_t = xs[devid]
            gradim_t = gradims[devid]
            gradx_t, = grad([im_t],[x_t],[gradim_t], retain_graph=False) # directional gradient
            gradxs.append(gradx_t)
    torch.cuda.synchronize()
    
    # and finally cat them in x.grad
    cat_gradxs(x,gradxs,nGPU)
    
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
            x.grad = x.clone().fill_(0)
        time0 = time()
        optimizer = optim.LBFGS({x}, max_iter=maxite, max_eval=100*maxite, line_search_fn='strong_wolfe',\
                                tolerance_grad = gtol, tolerance_change = ftol,\
                                history_size = maxcor)
        #optimizer = optim.SGD({x}, lr=0.1, momentum=0.9)

        def closure():
            optimizer.zero_grad()
            loss = obj_func(x,wph_ops,wph_streams,Sims,factr**2,sigma,ress,Mxs,Mys,pis,nGPU)
            return loss

        final_loss = optimizer.step(closure)

        #opt_state = optimizer.state[optimizer._params[0]]
        #niter = opt_state['n_iter']
        #final_loss = opt_state['prev_loss']

        
        print('At restart',start,'OPT fini avec:', final_loss,'in',time()-time0,'sec')
        #print('At restart',start,'OPT fini avec:', final_loss,niter,'in',time()-time0,'sec')
    return x
