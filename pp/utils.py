import numpy as np
import torch

pi = torch.from_numpy(np.array([np.pi])).float().cuda()

def weight(index, pos, res, gpu, sigma):
    if gpu:
        index = torch.tensor(index).cuda()
        pos = pos.cuda()
        res=torch.tensor(res).type(torch.float).cuda()
    #w = F.relu(1 - torch.abs(-index + pos))
    v = torch.min(torch.min(torch.abs(pos-index), torch.abs(pos+res-index)),
                  torch.abs(pos-res-index))
    w = torch.exp(-torch.mul(v,v)/(2*(sigma**2)))/torch.sqrt(2*pi*sigma**2)
    return w

def pos_to_im2(x, res, gpu):
    M = torch.zeros(res, res)
    Mx = torch.arange(0,res).type(torch.float)
    My = torch.arange(0,res).type(torch.float)
    if gpu:
        M = M.cuda(); Mx = Mx.cuda(); My = My.cuda()
    for k in range(x.shape[0]):
         im_x = weight(Mx,x[k,0], res, gpu).unsqueeze(1)
         im_y = weight(My,x[k,1], res, gpu).unsqueeze(0)
         M += torch.mm(im_x,im_y)
    return M.unsqueeze(0).unsqueeze(0)

def pos_to_im3(x, res, gpu, sigma):
    Mx = torch.arange(0, res).type(torch.float)
    My = torch.arange(0, res).type(torch.float)
    if gpu:
        Mx = Mx.cuda(); My = My.cuda()
    im_x = weight(Mx.unsqueeze(0), x[:, 0].unsqueeze(1), res, gpu, sigma).unsqueeze(2)
    im_y = weight(My.unsqueeze(0), x[:, 1].unsqueeze(1), res, gpu, sigma).unsqueeze(1)
    #print(im_x.shape, im_y.shape)
    M = torch.matmul(im_x, im_y).sum(0)
    #print(M.shape)
    return M.unsqueeze(0).unsqueeze(0)
