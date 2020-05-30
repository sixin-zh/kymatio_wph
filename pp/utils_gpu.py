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

# Fourier domain method, not very efficient. 
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

# not faster than Pixel domain
def pos_to_im_fourier3(xt, hf, om, K=1):
    # xt: (2, nbp)
    # hf[k]: (1,res)
    # om[k]: (1,res)
    # return a image in Fourier domain
    pos_x = xt[0,:].unsqueeze(1) # (nbs,1)
    pos_y = xt[1,:].unsqueeze(1) 
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

    M_real = torch.matmul(wf_real_x.unsqueeze(2),wf_real_y.unsqueeze(1)).sum(0) -\
             torch.matmul(wf_imag_x.unsqueeze(2),wf_imag_y.unsqueeze(1)).sum(0) # (nbp,res,res) -> (res,res)    
    M_imag = torch.matmul(wf_real_x.unsqueeze(2),wf_imag_y.unsqueeze(1)).sum(0) +\
             torch.matmul(wf_imag_x.unsqueeze(2),wf_real_y.unsqueeze(1)).sum(0)    
    M = torch.stack([M_real,M_imag],axis=2) # -> (res,res,2)
    
    return M.unsqueeze(0).unsqueeze(0) # -> (1,1,res,res,2)

# faster version of pos_to_im_fourier3? too big mem. usage!!
def pos_to_im_fourier3b(pt, hf, om, K=1, loop=10):
    # pt: (2, nbp)
    # hf[k]: (1,res)
    # om[k]: (1,res)
    # return a image in Fourier domain
    pos_x = pt[0,:].unsqueeze(1).unsqueeze(1) # (nbs,1,1)
    pos_y = pt[1,:].unsqueeze(1).unsqueeze(1)
    nbs = pt.shape[1]
    perloop = nbs//loop
    res = hf[0].shape[1]
    for k1 in range(K+1):
        for k2 in range(K+1):
            om1 = om[k1].unsqueeze(2) # (1,res,1)
            om2 = om[k2].unsqueeze(1) # (1,1,res)
            offset = 0
            for lid in range(loop):
                if lid < loop-1:
                    l_pos_x = pos_x[offset:offset+perloop,:,:]
                    l_pos_y = pos_y[offset:offset+perloop,:,:]
                else:
                    l_pos_x = pos_x[offset:-1,:,:]
                    l_pos_y = pos_y[offset:-1,:,:]
                offset += perloop
                pom = l_pos_x*om1+l_pos_y*om2 # (perloop,res,res)
                if lid == 0: #  and k1 ==0 and k2 == 0:
#                    cospom = hf[k1].t().expand((res,res)) * torch.cos(pom).sum(0) * hf[k2].expand((res,res))
#                    sinpom = hf[k1].t().expand((res,res)) * torch.sin(-pom).sum(0) * hf[k2].expand((res,res))
                    cospom = torch.cos(pom).sum(0)
                    sinpom = torch.sin(-pom).sum(0)                                  
                else:
#                    cospom += hf[k1].t().expand((res,res)) * torch.cos(pom).sum(0) * hf[k2].expand((res,res))
#                    sinpom += hf[k1].t().expand((res,res)) * torch.sin(-pom).sum(0) * hf[k2].expand((res,res))
                    cospom += torch.cos(pom).sum(0)
                    sinpom += torch.sin(-pom).sum(0)                                  
                #print(k1,k2,lid,pom.shape)
                
            if k1==0 and k2==0:
                M_real = hf[k1].t().expand((res,res)) * cospom * hf[k2].expand((res,res))
                M_imag = hf[k1].t().expand((res,res)) * sinpom * hf[k2].expand((res,res))
            else:
                M_real += hf[k1].t().expand((res,res)) * cospom * hf[k2].expand((res,res))
                M_imag += hf[k1].t().expand((res,res)) * sinpom * hf[k2].expand((res,res))

    #M = torch.stack([cospom,sinpom],axis=2) # -> (res,res,2)
    M = torch.stack([M_real,M_imag],axis=2) # -> (res,res,2)
    return M.unsqueeze(0).unsqueeze(0) # -> (1,1,res,res,2)
