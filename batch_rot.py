import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import scipy.io as sio
import sys
sys.path.append('/users/trec/brochard/kymatio_wph')


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

hatpsi = torch.FloatTensor(hatpsi) # (J,L2,M,N,2)
hatphi = torch.FloatTensor(hatphi) # (M,N,2)

j = 2
psi_test = hatpsi[j, 2, ...]; psi_test2 = hatpsi[j, 6, ...]  # filter ###################


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


filename = '/users/trec/brochard/kymatio_wph/data/poissonvor_150_100.txt'
pos = size*np.loadtxt(fname=filename, delimiter=',', skiprows=1, usecols=(1,2))
nb_points = pos.shape[0]
pos = torch.from_numpy(pos).type(torch.float)
imm = pos_to_im3(pos, res, gpu, sigma).type(torch.float)



def shift2(im_, u):
    # u: (1, P_c ,m, 2)
    # im: (1, P_c, N, N)
    u = u.type(torch.float)
    im = torch.stack((im_, torch.zeros(im_.size())), dim=-1)  # (1, P_c, N, N, 2)
    im_fft = torch.fft(im, 2)  # (1, P_c, N, N, 2)
    map = torch.arange(size).repeat(tuple(u.size()[:3])+(1,)).type(torch.float)  # (1, P_c, m, N)
    z = torch.matmul(map.unsqueeze(-1), u.unsqueeze(-2)).type(torch.float)  # (1, P_c, m, N, 1), (1, P_c, m, 1, 2)->(1, P_c, m, N, 2)
    sp = z[..., 0].unsqueeze(-1).repeat(1,1,1,1,size) + z[..., 1].unsqueeze(-2).repeat(1,1,1,size,1)  # (1, P_c, m, N, N)
    # compute e^(-iw.u0)
    fft_shift = torch.stack((torch.cos(2*np.pi*sp/size), torch.sin(2*np.pi*sp/size)), dim=-1)  # (1, P_c, m, N, N, 2)
    im_shift_fft = complex_mul(fft_shift, im_fft.unsqueeze(2).repeat(1,1,u.size(2), 1, 1, 1))  # (1, P_c, m, N, N, 2)
    im_shift = torch.ifft(im_shift_fft, 2)  # (1, P_c, m, N, N, 2)
#    plt.imshow(im_shift[0, ..., 0]); plt.show()
    return im_shift


def unshift2(ims, u):
    u = u.type(torch.float)
    ims_fft = torch.fft(ims, 2)  # (1, P_c, m, N, N, 2)
    map = torch.arange(size).repeat(tuple(u.size()[:3])+(1,)).type(torch.float)  # (1, P_c, m, N)
    u_back = -u  # (1, P_c, m, 2)
    z = torch.matmul(map.unsqueeze(-1), u_back.unsqueeze(-2))  # (1, P_c, m, N, 1), (1, P_c, m, 1, 2)->(1, P_c, m, N, 2)
    sp = z[..., 0].unsqueeze(-1).repeat(1,1,1,1,size) + z[..., 1].unsqueeze(-2).repeat(1,1,1,size,1)  # (1, P_c, m, N, N)
    # compute e^(-iw.u0)
    fft_shift = torch.stack((torch.cos(2*np.pi*sp/size), torch.sin(2*np.pi*sp/size)), dim=-1)  # (1, P_c, m, N, N, 2)
    im_shift_fft = complex_mul(ims_fft, fft_shift)  # (1, P_c, m, N, N, 2)
    im_shift = torch.ifft(im_shift_fft, 2)[..., 0]  # (1, P_c, m, N, N)
    return im_shift


def indices_unpad(im, indices, pad):
    indices = indices.view(tuple(im.size()[:2])+(-1,))
    indices_col = (indices % (size+2*pad)).type(torch.LongTensor)
    indices_col = (indices_col - pad)%size
    indices_row = ((indices )//(size+2*pad)).type(torch.LongTensor)
    indices_row = (indices_row - pad)%size

    indices = size*indices_row + indices_col
    return indices, indices_row, indices_col


def local_max_indices(im):
    #  im: (1, P_c, N, N)
    mp1 = torch.nn.MaxPool2d(5, stride=torch.Size([1,1]), return_indices=True)
    mp2 = torch.nn.MaxPool2d(7, stride=torch.Size([1,1]), return_indices=True)

    im_pad1 = F.pad(im, (2, 2, 2, 2), mode='circular')
    im_pad2 = F.pad(im, (3, 3, 3, 3), mode='circular')

    maxed1 = mp1(im_pad1)[1]
    maxed2 = mp2(im_pad2)[1]

    maxed1 = indices_unpad(im, maxed1, 2)
    maxed2 = indices_unpad(im, maxed2, 3)
    eq = torch.eq(maxed1[0], maxed2[0]).type(torch.LongTensor)  # (1, P_c, N*N)
    T = -(1 - eq)*3*size*size + eq * maxed1[0]  # (1, P_c, N*N)

    which = torch.arange(im.size(1)).unsqueeze(0).unsqueeze(-1).repeat(1, 1, size*size)  # (1, P_c, N*N)
    T = torch.stack((T, which), dim=1)  # (1, 2, P_c, N*N)
    T = T.view(2, -1)  # (2, P_c*N*N)
    T = T.unique(sorted=False, dim=-1)  # (2, m_sum)
    T = T.unsqueeze(0).repeat(im.size(1), 1, 1)  # (P_c, 2, m_sum)
    z = torch.arange(im.size(1)).unsqueeze(1).repeat(1, T.size(-1))  # (P_c, m_sum)
    affect = torch.eq(T[:, 1, :], z).type(torch.float)  # (P_c, m_sum)
    T_ = (T[:, 0, :]*affect - 3*size*size*(1-affect)).unsqueeze(0)  # (1, P_c, m_sum)

    maxima = torch.max(T_, dim=-1)[0].unsqueeze(-1).repeat(1, 1, z.size(-1))  # (1, P_c, m_sum)
    threshold = torch.gt(T_,  maxima/10).type(torch.float)
    T_ = T_*threshold + 3*size*size*(1-threshold)  # (1, P_c, m_sum)

    return T_, torch.stack((T_//size, T_%size), dim=-1)


def orth_phase(im2, loc):
    #im2 (1, P_c, N, N, 2)
    phase = torch.atan2(im2[...,1], im2[...,0])  # (1, P_c, N, N)
    shifted_phase = shift2(phase, loc)  # (1, P_c, m, N, N, 2)
    t_s_phase = torch.transpose(shifted_phase, -2, -3)  # (1, P_c, m, N, N, 2)
    t_s_phase = torch.flip(t_s_phase.unsqueeze(-3), [-3,-4]).squeeze()  # (1, P_c, m, N, N, 2)
    orth_ph = unshift2(t_s_phase, loc)  # (1, P_c, m, N, N)
    return phase, orth_ph


def shifted_phase(im2, loc, theta):
    """
    theta: list of length P_c
    """
    theta_ = torch.Tensor(theta)
    theta_ = theta_.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    print(theta_.size())
    print(loc.size())
    theta_ = theta_.repeat(1, 1, loc.size(2), size, size)  # (1, P_c, m, N, N)
    phase, orth_ph = orth_phase(im2, loc)  # (1, P_c, N, N), (1, P_c, m, N, N)
    print(theta_.size(), phase.unsqueeze(2).repeat(1, 1, loc.size(2), 1, 1).size())
    return torch.cos(theta_)*phase.unsqueeze(2).repeat(1, 1, loc.size(2), 1, 1) - torch.sin(theta_)*orth_ph  # (1, P_c, m, N, N)



def periodic_distance(x, y, N):
    return torch.min(torch.abs(x-y), torch.abs(x-y+N)).min(torch.abs(x-y-N))


def dist_to_max(u):
    z = torch.arange(size).repeat(tuple(u.size()[:3])+(1,)).type(torch.float)  # (1, P_c, m, N)
    z1 = periodic_distance(z, u[..., 0].unsqueeze(3).repeat(1, 1, 1, size), size).unsqueeze(-1).repeat(1, 1, 1, 1, size)  # (1, P_c, m, N, N)
    z2 = periodic_distance(z, u[..., 1].unsqueeze(3).repeat(1, 1, 1, size), size).unsqueeze(-2).repeat(1, 1, 1, size, 1)  # (1, P_c, m, N, N)
    z1 = z1**2
    z2 = z2**2
    z = torch.sqrt(z1 + z2)  # (1, P_c, m, N, N)
    z_min = torch.min(z, dim=2)[0].unsqueeze(2).repeat(1, 1, u.size(2), 1, 1)  # (1, P_c, m, N, N)
    return torch.eq(z, z_min).type(torch.float)


def new_phase(im1, im2, theta):
    # im1 and im2 (1, P_c, N, N, 2)
    im = im1.norm(p=2, dim=-1)*im2.norm(p=2, dim=-1)  # (1, P_c, N, N)
    loc = local_max_indices(im)[1]  # (1, P_c, m, 2)
    shifted_ph = shifted_phase(im2, loc, theta)  # (1, P_c, m, N, N)
    vor = dist_to_max(loc).type(torch.float)  # (1, P_c, m, N, N)
    return (shifted_ph*vor).sum(dim=2)


def phase_rot(im1, im2, theta):
    z = im2.norm(p=2, dim=-1)
    ph_rot = new_phase(im1, im2, theta)
    return torch.stack((z*torch.cos(ph_rot), z*torch.sin(ph_rot)), dim=-1)


im_test = torch.zeros(256, 256)

i, j = 104, 130

im_test[i, j] = 1

im_test[200,76] = 1

im_test = im_test.unsqueeze(0).unsqueeze(0)
im_test = torch.stack((im_test, torch.zeros(im_test.size())), dim=-1)
im_fft = torch.fft(im_test, 2)
im_fft = torch.fft(torch.stack((imm, torch.zeros(imm.size())), -1), 2)
fft_prod1 = cdgmm(im_fft, psi_test)
fft_prod2 = cdgmm(im_fft, psi_test2)
conv1 = torch.ifft(fft_prod1, 2)
#plt.imshow(conv1.squeeze()[...,1]); plt.show()
conv2 = torch.ifft(fft_prod2, 2)
#plt.imshow(conv2.squeeze()[...,1]); plt.show()
T = local_max_indices(conv1.norm(p=2, dim=-1)**2)
#print(T[1].size())


M = phase_rot(conv1, conv2, [-np.pi/2])
print(M.size())
plt.imshow(M[0,0,...,0]); plt.show()


def conjugate(z):
    return torch.stack((z[..., 0], -z[..., 1]), dim=-1)


corr1 = complex_mul(conv1-conv1.mean((2,3)), conjugate(conv2-conv2.mean((2,3)))).mean((2,3))
corr2 = complex_mul(conv1-conv1.mean((2,3)), conjugate(M - M.mean((2,3)))).mean((2,3))
corr3 = complex_mul(conv2-conv2.mean((2,3)), conjugate(conv2-conv2.mean((2,3)))).mean((2,3))

#print(corr1, corr2, corr3)


filename6 = './results/poissonvor_400_100_pos_s256_J5dj3_4ms.pt'
pos6 = res*torch.load(filename6)
im6 = pos_to_im3(pos6, res, gpu, sigma)
im6 = torch.stack((im6,  torch.zeros(im6.size())),dim=-1)
im6_fft = torch.fft(im6, 2)
fft_prod61 = cdgmm(im6_fft, psi_test)
fft_prod62 = cdgmm(im6_fft, psi_test2)
conv61 = torch.ifft(fft_prod61, 2)
conv62 = torch.ifft(fft_prod62, 2)
M6 = phase_rot(conv61, conv62, [-np.pi/2])

print(conv61.size())
corr61 = complex_mul(conv61-conv61.mean((2,3)), conjugate(conv62-conv62.mean((2,3)))).mean((2,3))
corr62 = complex_mul(conv61-conv61.mean((2,3)), conjugate(M6 - M6.mean((2,3)))).mean((2,3))
corr63 = complex_mul(conv62-conv62.mean((2,3)), conjugate(conv62-conv62.mean((2,3)))).mean((2,3))
print(corr61, corr62, corr63)


