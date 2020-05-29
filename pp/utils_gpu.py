import numpy as np
import torch
import matplotlib.pyplot as plt

def weight(index, pos, res, pi, sigma):
    # index: (1,res)
    # pos: (nbp,1)
    # res, pi, sigma: number
    v = torch.min(torch.min(torch.abs(pos-index), torch.abs(pos+res-index)),
                  torch.abs(pos-res-index))
    w = torch.exp(-torch.mul(v,v)/(2*(sigma**2)))/torch.sqrt(2*pi*sigma**2)
    return w

def pos_to_im3(x, res, Mx, My, pi, sigma):
    # x: (nbp, 2)
    # Mx: (res), My: (res)
    im_x = weight(Mx.unsqueeze(0), x[:, 0].unsqueeze(1), res, pi, sigma).unsqueeze(2)
    im_y = weight(My.unsqueeze(0), x[:, 1].unsqueeze(1), res, pi, sigma).unsqueeze(1)
    M = torch.matmul(im_x, im_y).sum(0)
    return M.unsqueeze(0).unsqueeze(0)

def get_hf_om(index,res,sigma,pi,K=1):
    # index: (1,res)
    # return hf[k]: (1,res), om[k]: (1,res)
    if K==1:
        assert sigma>0.49, 'too small sigma for K=1'
    hf = []
    om = []
    for k in range(K+1):
        om.append( (2*pi/res)*(index-k*res) ) # (1,res)
        hf.append( torch.exp(-0.5*(om[k]**2)*sigma**2) )
    return hf, om

def pos_to_im_fourier3(x, hf, om, K=1):
    # x: (nbp, 2)
    # hf[k]: (1,res)
    # om[k]: (1,res)
    # return a image in Fourier domain
    pos_x = x[:, 0].unsqueeze(1) # (nbs,1)
    pos_y = x[:, 1].unsqueeze(1)    
    for k in range(K+1): # avoid aliasing
        if k == 0:
            wf_real_x = torch.cos(pos_x*om[k])*hf[k] # (nbp,res)
            wf_imag_x = torch.sin(-pos_x*om[k])*hf[k]
            wf_real_y = torch.cos(pos_y*om[k])*hf[k]
            wf_imag_y = torch.sin(-pos_y*om[k])*hf[k]
        else:
            wf_real_x += torch.cos(pos_x*om[k])*hf[k]
            wf_imag_x += torch.sin(-pos_x*om[k])*hf[k]
            wf_real_y += torch.cos(pos_y*om[k])*hf[k]
            wf_imag_y += torch.sin(-pos_y*om[k])*hf[k]

    M_real = torch.matmul(wf_real_x.unsqueeze(2),wf_real_y.unsqueeze(1)) -\
             torch.matmul(wf_imag_x.unsqueeze(2),wf_imag_y.unsqueeze(1)) # (nbp,res,res)
    M_real = torch.sum(M_real,axis=0) # (res,res)
    
    M_imag = torch.matmul(wf_real_x.unsqueeze(2),wf_imag_y.unsqueeze(1)) +\
             torch.matmul(wf_imag_x.unsqueeze(2),wf_real_y.unsqueeze(1))
    M_imag = torch.sum(M_imag,axis=0)
    
    M = torch.stack([M_real,M_imag],axis=2) # -> (res,res,2)
    return M.unsqueeze(0).unsqueeze(0) # -> (1,1,res,res,2)
