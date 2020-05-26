import numpy as np
import torch

def weight(index, pos, res, pi, sigma):
    v = torch.min(torch.min(torch.abs(pos-index), torch.abs(pos+res-index)),
                  torch.abs(pos-res-index))
    w = torch.exp(-torch.mul(v,v)/(2*(sigma**2)))/torch.sqrt(2*pi*sigma**2)
    return w

def pos_to_im3(x, res, Mx, My, pi, sigma):
    im_x = weight(Mx.unsqueeze(0), x[:, 0].unsqueeze(1), res, pi, sigma).unsqueeze(2)
    im_y = weight(My.unsqueeze(0), x[:, 1].unsqueeze(1), res, pi, sigma).unsqueeze(1)
    M = torch.matmul(im_x, im_y).sum(0)
    return M.unsqueeze(0).unsqueeze(0)
