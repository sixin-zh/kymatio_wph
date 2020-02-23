
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import scipy.io as sio
"""
import sys
sys.path.append('/users/trec/brochard/kymatio_wph')
"""

res = 256
J = 6
L = 8
#M, N = im.shape[-2], im.shape[-1]
M, N = res, res
delta_j = 0
delta_l = 0
delta_k = 0
nb_chunks = 1
nb_restarts = 0
size = 256
gpu = 0
sigma = 1

matfilters = sio.loadmat('./matlab/filters/bumpsteerableg1_fft2d_N' + str(N) + '_J' + str(J) + '_L' + str(L) + '.mat')

fftphi = matfilters['filt_fftphi'].astype(np.complex_)
hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)

fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)
hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)

hatpsi = torch.FloatTensor(hatpsi).cuda() # (J,L2,M,N,2)
hatphi = torch.FloatTensor(hatphi).cuda() # (M,N,2)

j = 2
psi_test = hatpsi[j, 2, ...]; psi_test2 = hatpsi[j, 6, ...]


pi = np.pi

def weight(index, pos, res, gpu, sigma):
    if gpu:
        index = torch.tensor(index).cuda()
        pos = pos.cuda()
        res=torch.tensor(res).type(torch.float).cuda()
    #w = F.relu(1 - torch.abs(-index + pos))
    v = torch.min(torch.min(torch.abs(pos-index), torch.abs(pos+res-index)),
                  torch.abs(pos-res-index))
    w = torch.exp(-torch.mul(v,v)/(2*(sigma**2)))/torch.sqrt(torch.tensor([2*pi*sigma**2]))
    return w


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

def cdgmm(A, B, inplace=False):
    """
        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : tensor
            input tensor with size (B, C, M, N, 2)
        B : tensor
            B is a complex tensor of size (M, N, 2)
        inplace : boolean, optional
            if set to True, all the operations are performed inplace

        Returns
        -------
        C : tensor
            output tensor of size (B, C, M, N, 2) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :]
    """
    A, B = A.contiguous(), B.contiguous()


    C = A.new(A.size())

    A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3))
    A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3))

    B_r = B[...,0].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_i)
    B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_r)

    C[..., 0].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_r - A_i * B_i
    C[..., 1].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_i + A_i * B_r

    return C if not inplace else A.copy_(C)



def complex_mul(a, b):
    ar = a[..., 0]
    br = b[..., 0]
    ai = a[..., 1]
    bi = b[..., 1]
    real = ar*br - ai*bi
    imag = ar*bi + ai*br

    return torch.stack((real, imag), dim=-1)

"""
filename = '/users/trec/brochard/kymatio_wph/data/poissonvor_150_100.txt'
pos = size*np.loadtxt(fname=filename, delimiter=',', skiprows=1, usecols=(1,2))
nb_points = pos.shape[0]
pos = torch.from_numpy(pos).type(torch.float)
imm = pos_to_im3(pos, res, gpu, sigma).type(torch.float)
"""


def shift2(im_, u):
    """
    Takes as input a batch of images and a tensor of coordinates for shift,
    and returns a tensor of size (1, P_c, m, N, N, 2) of the images in the
    batch shifted.

    """

    # u: (1, P_c ,m, 2)
    # im_: (1, P_c, N, N)
    size = im_.size(-2)
    u = u.type(torch.float).cuda()
    im = torch.stack((im_, torch.zeros(im_.size()).type(torch.cuda.FloatTensor)), dim=-1)  # (1, P_c, N, N, 2)
    im_fft = torch.fft(im, 2)  # (1, P_c, N, N, 2)
    map = torch.arange(size).repeat(tuple(u.size()[:3])+(1,)).type(torch.float).cuda()  # (1, P_c, m, N)
    z = torch.matmul(map.unsqueeze(-1), u.unsqueeze(-2)).type(torch.float).cuda()  # (1, P_c, m, N, 1), (1, P_c, m, 1, 2)->(1, P_c, m, N, 2)
    sp = z[..., 0].unsqueeze(-1).repeat(1,1,1,1,size) + z[..., 1].unsqueeze(-2).repeat(1,1,1,size,1)  # (1, P_c, m, N, N)
    del(z)
    fft_shift = torch.stack((torch.cos(2*np.pi*sp/size), torch.sin(2*np.pi*sp/size)), dim=-1)  # (1, P_c, m, N, N, 2)
    im_shift_fft = complex_mul(fft_shift, im_fft.unsqueeze(2).repeat(1,1,u.size(2), 1, 1, 1))  # (1, P_c, m, N, N, 2)
    del(fft_shift); del(sp); del(im_fft)
    im_shift = torch.ifft(im_shift_fft, 2)  # (1, P_c, m, N, N, 2)
    return im_shift


def unshift2(ims, u):
    """
    Same as shift2, but the images are already of size (1, P_c, m, N, N, 2),
    and we shift by the opposite coordinates.
    """

    size = ims.size(-2)
    u = u.type(torch.float)
    ims_fft = torch.fft(ims, 2)  # (1, P_c, m, N, N, 2)
    map = torch.arange(size).repeat(tuple(u.size()[:3])+(1,)).type(torch.cuda.FloatTensor)  # (1, P_c, m, N)
    u_back = -u  # (1, P_c, m, 2)
    z = torch.matmul(map.unsqueeze(-1), u_back.unsqueeze(-2))  # (1, P_c, m, N, 1), (1, P_c, m, 1, 2)->(1, P_c, m, N, 2)
    sp = z[..., 0].unsqueeze(-1).repeat(1,1,1,1,size) + z[..., 1].unsqueeze(-2).repeat(1,1,1,size,1)  # (1, P_c, m, N, N)
    # compute e^(-iw.u0)
    fft_shift = torch.stack((torch.cos(2*np.pi*sp/size), torch.sin(2*np.pi*sp/size)), dim=-1)  # (1, P_c, m, N, N, 2)
    im_shift_fft = complex_mul(ims_fft, fft_shift)  # (1, P_c, m, N, N, 2)
    del(fft_shift); del(sp); del(ims_fft)
    im_shift = torch.ifft(im_shift_fft, 2)[..., 0]  # (1, P_c, m, N, N)
    return im_shift


def indices_unpad(im, indices, pad):
    """
    The indices given by the maxpool function are the ones of the periodicly padded image,
    so this function gets back the actual indices of the image, according to the padding size.
    """

    size = im.size(-2)
    indices = indices.view(tuple(im.size()[:2])+(-1,))
    indices_col = (indices % (size+2*pad)).type(torch.LongTensor).cuda()
    indices_col = (indices_col - pad)%size
    indices_row = ((indices )//(size+2*pad)).type(torch.LongTensor).cuda()
    indices_row = (indices_row - pad)%size

    indices = size*indices_row + indices_col
    return indices, indices_row, indices_col


def local_max_indices(im, number_of_centers):
    """
    Given a batch of (modulus) images, and a fixed integer 'number_of_centers', this function
    returns the 'number_of_centers' first local maxima, sorted by amplitude
    of the maxima.
    """
    #  im: (1, P_c, N, N)
    size = im.size(-1)
    mp1 = torch.nn.MaxPool2d(5, stride=torch.Size([1,1]), return_indices=True)

    im_pad1 = F.pad(im, (2, 2, 2, 2), mode='circular')

    maxed1 = mp1(im_pad1)

    maxed1_ = indices_unpad(im, maxed1[1], 2)

    z = torch.arange(size**2).unsqueeze(0).unsqueeze(0)
    z = z.repeat(im.size(0), im.size(1), 1).type(torch.cuda.FloatTensor)

    eq = torch.eq(maxed1_[0], z).type(torch.cuda.FloatTensor)  # (1, P_c, N*N)

    imx = maxed1[0].view(im.size(0), im.size(1), -1)*eq  # (1, P_c, N*N)
    zero_count = (imx == 0).sum(dim=-1).max()
    top = torch.topk(imx, min(number_of_centers, size*size-zero_count))[1]  # (1, P_c, number_of_centers)


    del(eq); del(maxed1); #del(maxed2)

    return top, torch.stack((top//size, top%size), dim=-1)


def orth_phase(im2, loc):
    """
    Given a batch of images and a tensor of local maxima, this function returns
    a tuple consisting of the phase and the orthogonal phase centered at the local
    minima.
    """
    #im2 (1, P_c, N, N, 2)
    phase = torch.atan2(im2[...,1], im2[...,0])  # (1, P_c, N, N)
    shifted_phase = shift2(phase, loc)  # (1, P_c, m, N, N, 2)
    t_s_phase = torch.transpose(shifted_phase, -2, -3)  # (1, P_c, m, N, N, 2)
    del(shifted_phase)
    t_s_phase = torch.flip(t_s_phase.unsqueeze(-3), [-3,-4]).squeeze()  # (1, P_c, m, N, N, 2)
    orth_ph = unshift2(t_s_phase, loc)  # (1, P_c, m, N, N)
    del(t_s_phase)
    return phase, orth_ph



def shifted_phase(im2, loc, theta):
    """
    theta: tensor of size (P_c)
    Given a batch of images, a tensor of local maxima, and a tensor of rotation angles,
    this funtion returns the rotated phases for every angle and local maximum.
    """
    size = im2.size(-2)
    theta_ = theta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    theta_ = theta_.repeat(1, 1, loc.size(2), size, size)  # (1, P_c, m, N, N)
    phase, orth_ph = orth_phase(im2, loc)  # (1, P_c, N, N), (1, P_c, m, N, N)
    return torch.cos(theta_)*phase.unsqueeze(2).repeat(1, 1, loc.size(2), 1, 1) - torch.sin(theta_)*orth_ph  # (1, P_c, m, N, N)



def periodic_distance(x, y, N):
    return torch.min(torch.abs(x-y), torch.abs(x-y+N)).min(torch.abs(x-y-N))


def dist_to_max(u, size):
    """
    Given a tensor of local maxima indices and a map of size 'size',
    this function returns a tensor of maps indicators of the Voronoi cells
    of the maxima.
    """
    z = torch.arange(size).repeat(tuple(u.size()[:3])+(1,)).type(torch.cuda.FloatTensor)  # (1, P_c, m, N)
    z1 = periodic_distance(z, u[..., 0].unsqueeze(3).repeat(1, 1, 1, size), size).unsqueeze(-1).repeat(1, 1, 1, 1, size)  # (1, P_c, m, N, N)
    z2 = periodic_distance(z, u[..., 1].unsqueeze(3).repeat(1, 1, 1, size), size).unsqueeze(-2).repeat(1, 1, 1, size, 1)  # (1, P_c, m, N, N)
    z1 = z1**2
    z2 = z2**2
    z = torch.sqrt(z1 + z2)  # (1, P_c, m, N, N)
    z_min = torch.min(z, dim=2)[0].unsqueeze(2).repeat(1, 1, u.size(2), 1, 1)  # (1, P_c, m, N, N)
    vors = torch.eq(z, z_min).type(torch.float).cuda()
    del(z); del(z_min); del(z1); del(z2)
    return vors

def new_phase(im1, im2, theta, nb_centers, k1, k2):
    """
    This function returns the new phase of im2.
    """
    # im1 and im2 (1, P_c, N, N, 2)
    k1 = torch.tensor(k1)
    k2 = torch.tensor(k2)
    k = (1-k1.eq(0).type(torch.cuda.FloatTensor))*(1-k2.eq(0).type(torch.cuda.FloatTensor))  # (1, P_c, 1, 1)
    k = k.unsqueeze(0).unsqueeze(-1).repeat(1, 1, im1.size(-2), im1.size(-2))  # (1, P_c, N, N)
    im = im1.norm(p=2, dim=-1)*im2.norm(p=2, dim=-1)  # (1, P_c, N, N)
    loc = local_max_indices(im, nb_centers)[1]  # (1, P_c, m, 2)
    del(im)
    shifted_ph = shifted_phase(im2, loc, theta)  # (1, P_c, m, N, N)
    vor = dist_to_max(loc, im1.size(-2)).type(torch.float).cuda()  # (1, P_c, m, N, N)
    n_ph = (shifted_ph*vor).sum(dim=2) * k + torch.atan2(im2[...,1], im2[...,0]) * (1-k)
    return n_ph


def phase_rot(im1, im2, theta, nb_centers, k1, k2):
    """
    This function returns the phase rotated version of im2.
    """
    z = im2.norm(p=2, dim=-1)
    ph_rot = new_phase(im1, im2, theta, nb_centers, k1, k2)
    return torch.stack((z*torch.cos(ph_rot), z*torch.sin(ph_rot)), dim=-1)



# Test on an image consisting of two Diracs

im_test = torch.zeros(256, 256)
i, j = 104, 130
im_test[i, j] = 1
im_test[200,76] = 1
im_test = im_test.unsqueeze(0).unsqueeze(0).cuda()
im_test = torch.stack((im_test, torch.zeros(im_test.size()).cuda()), dim=-1)
im_fft = torch.fft(im_test, 2)
#im_fft = torch.fft(torch.stack((imm, torch.zeros(imm.size()).cuda()), -1), 2)
fft_prod1 = cdgmm(im_fft, psi_test)
fft_prod2 = cdgmm(im_fft, psi_test2)
conv1 = torch.ifft(fft_prod1, 2)
conv2 = torch.ifft(fft_prod2, 2)


M = phase_rot(conv1, conv2, torch.tensor([-np.pi/4]).cuda(), 2)
plt.imshow(M[0,0,...,1].cpu()); plt.show()


def conjugate(z):
    return torch.stack((z[..., 0], -z[..., 1]), dim=-1)


corr1 = complex_mul(conv1-conv1.mean((2,3)), conjugate(conv2-conv2.mean((2,3)))).mean((2,3))[0,0,:]
corr2 = complex_mul(conv1-conv1.mean((2,3)), conjugate(M - M.mean((2,3)))).mean((2,3))[0,0,:]
corr3 = complex_mul(conv2-conv2.mean((2,3)), conjugate(conv2-conv2.mean((2,3)))).mean((2,3))[0,0,:]

print(corr1, corr2, corr3)

"""
filename6 = './results/poissonvor_400_100_pos_s256_J5dj3_4ms.pt'
pos6 = res*torch.load(filename6)
im6 = pos_to_im3(pos6, res, gpu, sigma).cuda()
im6 = torch.stack((im6,  torch.zeros(im6.size()).cuda()),dim=-1)
im6_fft = torch.fft(im6, 2)
fft_prod61 = cdgmm(im6_fft, psi_test)
fft_prod62 = cdgmm(im6_fft, psi_test2)
conv61 = torch.ifft(fft_prod61, 2)
conv62 = torch.ifft(fft_prod62, 2)
M6 = phase_rot(conv61, conv62, torch.tensor([-np.pi/2]).cuda(), 40)

print(conv61.size())
corr61 = complex_mul(conv61-conv61.mean((2,3)), conjugate(conv62-conv62.mean((2,3)))).mean((2,3))
corr62 = complex_mul(conv61-conv61.mean((2,3)), conjugate(M6 - M6.mean((2,3)))).mean((2,3))
corr63 = complex_mul(conv62-conv62.mean((2,3)), conjugate(conv62-conv62.mean((2,3)))).mean((2,3))
print(corr61, corr62, corr63)

"""
