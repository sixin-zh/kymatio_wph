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

# return a image in Fourier domain
def weight_fourier2(pos, hf, om):
    # pos: (nbp,1)
    # hf: (1,res)
    # om: (1,res)
    # return
    # w_fourier: (nbp,res), (nbp,res)
#    plt.plot(hf[0,:].cpu(),'o')
#    plt.show()
    wf_real = torch.cos(pos*om)*hf
#    hf = hf.expand_as(wf_real)
#    wf_real = wf_real * hf
    wf_imag = torch.sin(-pos*om)*hf
    return wf_real, wf_imag
    
def pos_to_im3_fourier2(x, hf, om):
    # x: (nbp, 2)
    # hf: (1,res)
    # om: (1,res)
    im_xf_real, im_xf_imag = weight_fourier2(x[:, 0].unsqueeze(1), hf, om)
    im_yf_real, im_yf_imag = weight_fourier2(x[:, 1].unsqueeze(1), hf, om)
    # sum over points
#    plt.plot(im_xf_real[0,:].cpu(),'o')
#    plt.show()
    #print(ab.shape)
    M_real = torch.matmul(im_xf_real.unsqueeze(2),im_yf_real.unsqueeze(1)) -\
             torch.matmul(im_xf_imag.unsqueeze(2),im_yf_imag.unsqueeze(1)) # (nbp,res,res)
    M_real = torch.sum(M_real,axis=0)
#    plt.imshow(M_real.cpu())
#    plt.show()
    M_imag = torch.matmul(im_xf_real.unsqueeze(2),im_yf_imag.unsqueeze(1)) +\
             torch.matmul(im_xf_imag.unsqueeze(2),im_yf_real.unsqueeze(1))
    M_imag = torch.sum(M_imag,axis=0)
#    plt.imshow(M_real.cpu()**2+M_imag.cpu()**2)
#    plt.show()
    M = torch.stack([M_real,M_imag],axis=2) # (res,res,2)
    return M.unsqueeze(0).unsqueeze(0) # (1,1,res,res,2)

def get_hf_om(index,res,sigma,pi,K=1):
    # index: (1,res)
    # return hf[k]: (1,res), om[k]: (1,res)
    hf = []
    om = []
    for k in range(K+1):
        om.append( (2*pi/res)*(index-k*res) ) # (1,res)
        hf.append( torch.exp(-0.5*(om[k]**2)*sigma**2) )
    '''
    h = torch.exp(-0.5*(index**2)/(sigma**2))
    for shift in range(-2,3): # make it period
        if shift != 0:
            #print('shift',shift)
            index1 = index + shift*res
            h += torch.exp(-0.5*(index1**2)/(sigma**2))
    h = h / torch.sqrt(2*pi*sigma**2)
    #plt.plot(np.log10(h[0,:].cpu().numpy()))
    #plt.show()
    h0 = h.clone().fill_(0)
    h = torch.stack([h,h0],axis=2)
    hf =  torch.fft(h,signal_ndim=1) # (1,res): hat of Gaussian emvelope function in 1d
    hf = hf[:,:,0] # only keep real part
    '''
    #print(index)

    #print(pi)
    #print(om)
    return hf, om

# return a image in Fourier domain   
def pos_to_im_fourier3(x, hf, om, K=1):
    # x: (nbp, 2)
    # hf[k]: (1,res)
    # om[k]: (1,res)
    
    # om1*x[:,0:1]: (nbp,res) -> (nbp,res,1)
    # om2*x[:,1:2]: (nbp,res) -> (nbp,1,res)

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
    
    '''
    nbp = x.shape[0]
    res = hf2.shape[0]
    wf_real = torch.zeros((nbp,res,res)) # .cuda()
    wf_imag = torch.zeros((nbp,res,res)) # .cuda()
    for p in range(nbp):
        for rx in range(res):
            for ry in range(res):
                wf_real[p,rx,ry] = torch.cos( x[p,0]*om1[0,rx] + x[p,1]*om2[0,ry])
                wf_imag[p,rx,ry] = torch.sin( x[p,0]*om1[0,rx] + x[p,1]*om2[0,ry])                
#    wf_real = torch.cos((x[:,0:1]*om1).unsqueeze(2)+(x[:,1:2]*om2).unsqueeze(1)) # -> (nbp,res,res)
#    wf_imag = torch.sin((x[:,0:1]*om1).unsqueeze(2)+(x[:,1:2]*om2).unsqueeze(1)) # -> (nbp,res,res)
    # sum over points
    M_real = torch.sum(wf_real,axis=0) # -> (res,res)
    M_real = M_real * hf2 # * hf.t().expand((res,res))
#    plt.imshow(M_real.cpu())
#    plt.show()
    M_imag = torch.sum(wf_imag,axis=0)
    M_imag = M_imag * hf2 # .t().expand((res,res))
#    plt.imshow(M_real.cpu()**2+M_imag.cpu()**2)
#    plt.show()
    '''
    M = torch.stack([M_real,M_imag],axis=2) # -> (res,res,2)
    return M.unsqueeze(0).unsqueeze(0) # -> (1,1,res,res,2)
