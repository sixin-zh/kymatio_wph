# Authors: Edouard Oyallon, Sergey Zagoruyko

import torch
import torch.nn as nn
from torch.nn import ReflectionPad2d
from torch.autograd import Function
import numpy as np

NAME = 'torch'

from .backend_utils import is_long_tensor
from .backend_utils import HookDetectNan, masked_fill_zero



def iscomplex(input):
    return input.size(-1) == 2

class SubInitMean(object):
    def __init__(self, dim):
        self.dim = dim # use the last "dim" dimensions to compute the mean
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            #print('subinitmean:input',input.shape)
            for d in range(self.dim):
                minput = torch.mean(minput, -1)
            for d in range(self.dim):
                minput = minput.unsqueeze(-1)
            #print('subinitmean:minput',minput.shape)
            minput.expand_as(input)
            self.minput = minput

        #print('subinitmean:minput sum',self.minput.sum())
        output = input - self.minput
        return output

class Pad(object):
    def __init__(self, pad_size, pre_pad=False):
        """
            Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.

            Parameters
            ----------
            pad_size : int
                size of padding to apply.
            pre_pad : boolean
                if set to true, then there is no padding, one simply adds the imaginarty part.
        """
        self.pre_pad = pre_pad
        self.padding_module = ReflectionPad2d(pad_size)

    def __call__(self, input):
        if(self.pre_pad):
            output = input.new_zeros(input.size(0), input.size(1), input.size(2), input.size(3), 2)
            output.narrow(output.ndimension()-1, 0, 1)[:] = input
        else:
            out_ = self.padding_module(input)
            output = input.new_zeros(*(out_.size() + (2,)))
            output.select(4, 0)[:] = out_
        return output

def unpad(in_):
    """
        Slices the input tensor at indices between 1::-1

        Parameters
        ----------
        in_ : tensor_like
            input tensor

        Returns
        -------
        in_[..., 1:-1, 1:-1]
    """
    return in_[..., 1:-1, 1:-1]

class SubsampleFourier(object):
    """
        Subsampling of a 2D image performed in the Fourier domain
        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.

        Parameters
        ----------
        x : tensor_like
            input tensor with at least 5 dimensions, the last being the real
             and imaginary parts.
            Ideally, the last dimension should be a power of 2 to avoid errors.
        k : int
            integer such that x is subsampled by 2**k along the spatial variables.

        Returns
        -------
        res : tensor_like
            tensor such that its fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            FFT^{-1}(res)[u1, u2] = FFT^{-1}(x)[u1 * (2**k), u2 * (2**k)]
    """
    def __call__(self, input, k):
        out = input.new(input.size(0), input.size(1), input.size(2) // k, input.size(3) // k, 2)


        y = input.view(input.size(0), input.size(1),
                       input.size(2)//out.size(2), out.size(2),
                       input.size(3)//out.size(3), out.size(3),
                       2)

        out = y.mean(4, keepdim=False).mean(2, keepdim=False)
        return out


class Modulus(object):
    """
        This class implements a modulus transform for complex numbers.

        Usage
        -----
        modulus = Modulus()
        x_mod = modulus(x)

        Parameters
        ---------
        x: input tensor, with last dimension = 2 for complex numbers

        Returns
        -------
        output: a tensor with imaginary part set to 0, real part set equal to
        the modulus of x.
    """
    def __call__(self, input):

        norm = input.norm(p=2, dim=-1, keepdim=True)
        return torch.cat([norm, torch.zeros_like(norm)], -1)


def modulus(z):
    z_mod = z.norm(p=2, dim=-1)

    # if z.requires_grad:
    #     # z_mod_mask.register_hook(HookDetectNan("z_mod_mask in modulus"))
    #     z_mod.register_hook(HookDetectNan("z_mod in modulus"))
    #     z.register_hook(HookDetectNan("z in modulus"))

    return z_mod



def fft(input, direction='C2C', inverse=False):
    """
        Interface with torch FFT routines for 2D signals.

        Example
        -------
        x = torch.randn(128, 32, 32, 2)
        x_fft = fft(x, inverse=True)

        Parameters
        ----------
        input : tensor
            complex input for the FFT
        direction : string
            'C2R' for complex to real, 'C2C' for complex to complex
        inverse : bool
            True for computing the inverse FFT.
            NB : if direction is equal to 'C2R', then the transform
            is automatically inverse.
    """
    if direction == 'C2R':
        inverse = True

    if not iscomplex(input):
        raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

    if (not input.is_contiguous()):
        raise (RuntimeError('Tensors must be contiguous!'))

    if direction == 'C2R':
        output = torch.irfft(input, 2, normalized=False, onesided=False)*input.size(-2)*input.size(-3)
    elif direction == 'C2C':
        if inverse:
            output = torch.ifft(input, 2, normalized=False)*input.size(-2)*input.size(-3)
        else:
            output = torch.fft(input, 2, normalized=False)

    return output




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
    if A.size()[-3:] != B.size():
        raise RuntimeError('The filters are not compatible for multiplication!')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    if B.ndimension() != 3:
        raise RuntimeError('The filters must be simply a complex array!')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')


    C = A.new(A.size())

    A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3))
    A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3))

    B_r = B[...,0].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_i)
    B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_r)

    C[..., 0].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_r - A_i * B_i
    C[..., 1].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_i + A_i * B_r

    return C if not inplace else A.copy_(C)


class StablePhaseExp(Function):
    @staticmethod
    def forward(ctx, z, r):
        eitheta = z / r
        eitheta.masked_fill_(r == 0, 0)

        ctx.save_for_backward(eitheta, r)
        return eitheta

    @staticmethod
    def backward(ctx, grad_output):
        eitheta, r = ctx.saved_tensors

        dldz = grad_output / r
        dldz.masked_fill_(r == 0, 0)

        dldr = - eitheta * grad_output / r
        dldr.masked_fill_(r == 0, 0)
        dldr = dldr.sum(dim=-1).unsqueeze(-1)

        return dldz, dldr


phaseexp = StablePhaseExp.apply


def ones_like(z):
    re = torch.ones_like(z[..., 0])
    im = torch.zeros_like(z[..., 1])
    return torch.stack((re, im), dim=-1)


def real(z):
    return z[..., 0]


def imag(z):
    return z[..., 1]


def conjugate(z):
    z_copy = z.clone()
    z_copy[..., 1] = -z_copy[..., 1]
    return z_copy

def pows(z, max_k, dim=0):
    z_pows = [ones_like(z)]
    if max_k > 0:
        z_pows.append(z)
        z_acc = z
        for k in range(2, max_k + 1):
            z_acc = mul(z_acc, z)
            z_pows.append(z_acc)
    z_pows = torch.stack(z_pows, dim=dim)
    return z_pows


def log2_pows(z, max_pow_k, dim=0):
    z_pows = [ones_like(z)]
    if max_pow_k > 0:
        z_pows.append(z)
        z_acc = z
        for k in range(2, max_pow_k + 1):
            z_acc = mul(z_acc, z_acc)
            z_pows.append(z_acc)
    assert len(z_pows) == max_pow_k + 1
    z_pows = torch.stack(z_pows, dim=dim)
    return z_pows

def mul(z1, z2):
    zr = real(z1) * real(z2) - imag(z1) * imag(z2)
    zi = real(z1) * imag(z2) + imag(z1) * real(z2)
    z = torch.stack((zr, zi), dim=-1)
    return z

# k in Linear or Log
class PhaseExpLL(nn.Module):
    def __init__(self, K, k_type='linear', keep_k_dim=False, check_for_nan=False):
        super(PhaseExpLL, self).__init__()
        self.K = K
        self.keep_k_dim = keep_k_dim
        self.check_for_nan = check_for_nan
        assert k_type in ['linear', 'log2']
        self.k_type = k_type

    def forward(self, z):
        s = z.size()
        z_mod = z.norm(p=2, dim=-1)  # modulus

        eitheta = phaseexp(z, z_mod.unsqueeze(-1))  # phase

        # compute phase exponent : |z| * exp(i k theta)
        if self.k_type == 'linear':
            eiktheta = pows(eitheta, self.K - 1, dim=1)
        elif self.k_type == 'log2':
            eiktheta = log2_pows(eitheta, self.K - 1, dim=1)
        z_pe = z_mod.unsqueeze(-1) * eiktheta

        if not self.keep_k_dim:
            z_pe = z_pe.view(s[0], -1, *s[2:])

#        if z.requires_grad and self.check_for_nan:
#            z.register_hook(HookDetectNan("z in PhaseExp"))
#            if self.K > 1:
#                z_mod.register_hook(HookDetectNan("z_mod in PhaseExp"))
#            eitheta.register_hook(HookDetectNan("eitheta in PhaseExp"))
#            eiktheta.register_hook(HookDetectNan("eiktheta in PhaseExp"))
#            z_pe.register_hook(HookDetectNan("z_pe in PhaseExp"))

        return z_pe

# comptue only a single phase exponent k, k >= 0 integer
class PhaseExpSk(nn.Module):
    def __init__(self, keep_k_dim=False, check_for_nan=False):
        super(PhaseExpSk, self).__init__()
        self.keep_k_dim = keep_k_dim
        self.check_for_nan = check_for_nan

    def forward(self, z, k):
        s = z.size()
        z_mod = z.norm(p=2, dim=-1)  # modulus

        eitheta = phaseexp(z, z_mod.unsqueeze(-1))  # phase

        if k == 0:
            eiktheta = [ones_like(z)]
        elif k == 1:
            eiktheta = eitheta
        elif k > 1:
            z_acc = z
            for kb in range(2,k+1):
                z_acc = mul(z_acc,z)
            eiktheta = z_acc
        else:
            assert k>=0, 'need postive k exponent'

        z_pe = z_mod.unsqueeze(-1) * eiktheta

        if not self.keep_k_dim:
            z_pe = z_pe.view(s[0], -1, *s[2:])

    #        if z.requires_grad and self.check_for_nan:
#            z.register_hook(HookDetectNan("z in PhaseExp"))
#            if self.K > 1:
#                z_mod.register_hook(HookDetectNan("z_mod in PhaseExp"))
#            eitheta.register_hook(HookDetectNan("eitheta in PhaseExp"))
#            eiktheta.register_hook(HookDetectNan("eiktheta in PhaseExp"))
#            z_pe.register_hook(HookDetectNan("z_pe in PhaseExp"))
        return z_pe

class PhaseExp_par(nn.Module):
    def __init__(self, K, k_type='linear', keep_k_dim=False, check_for_nan=False):
        super(PhaseExp_par, self).__init__()
        self.K = K
        self.keep_k_dim = keep_k_dim
        self.check_for_nan = check_for_nan
        assert k_type in ['linear', 'log2']
        self.k_type = k_type

    def forward(self, z):
        s = z.size()
        z_mod = z.norm(p=2, dim=-1)  # modulus

        eitheta = phaseexp(z, z_mod.unsqueeze(-1))  # phase
        #print(eitheta.size())

        # compute phase exponent : |z| * exp(i k theta)
        if self.k_type == 'linear':
            eiktheta = pows(eitheta, self.K - 1, dim=1)
            #print(eiktheta.size())
        elif self.k_type == 'log2':
            eiktheta = log2_pows(eitheta, self.K - 1, dim=1)
        #print(z_mod.unsqueeze(1).size())
        z_pe = z_mod.unsqueeze(1).unsqueeze(-1) * eiktheta

        if not self.keep_k_dim:
            z_pe = z_pe.view(s[0], -1, *s[2:])

#        if z.requires_grad and self.check_for_nan:
#            z.register_hook(HookDetectNan("z in PhaseExp"))
#            if self.K > 1:
#                z_mod.register_hook(HookDetectNan("z_mod in PhaseExp"))
#            eitheta.register_hook(HookDetectNan("eitheta in PhaseExp"))
#            eiktheta.register_hook(HookDetectNan("eiktheta in PhaseExp"))
#            z_pe.register_hook(HookDetectNan("z_pe in PhaseExp"))

        return z_pe



class StablePhase(Function):
    @staticmethod
    def forward(ctx, z):
        z = z.detach()
        x, y = real(z), imag(z)
        r = z.norm(p=2, dim=-1)

        # NaN positions
        eps = 1e-32
        mask_real_neg = (torch.abs(y) <= eps) * (x <= 0)
        mask_zero = r <= eps

        x_tilde = r + x
        # theta = torch.atan(y / x_tilde) * 2
        theta = torch.atan2(y, x)

        # relace NaNs
        theta.masked_fill_(mask_real_neg, np.pi)
        theta.masked_fill_(mask_zero, 0.)

        # ctx.save_for_backward(x.detach(), y.detach(), r.detach())
        ctx.save_for_backward(x, y, r, mask_real_neg, mask_zero)
        return theta

    @staticmethod
    def backward(ctx, grad_output):
        x, y, r, mask_real_neg, mask_zero = ctx.saved_tensors

        # some intermediate variables
        x_tilde = r + x
        e = x_tilde ** 2 + y ** 2

        # derivative with respect to the real part
        dtdx = - y * x_tilde * 2 / (r * e)
        mask_real_neg_bis = (torch.abs(y) == 0) * (x <= 0)
        dtdx.masked_fill_(mask_real_neg, 0)
        dtdx.masked_fill_(mask_zero, 0)

        # derivative with respect to the imaginary part
        dtdy = x * x_tilde * 2 / (r * e)
        dtdy[mask_real_neg] = -1 / r[mask_real_neg]
        # dtdy.masked_fill_(mask, 0)
        dtdy.masked_fill_(mask_zero, 0)

        dtdz = grad_output.unsqueeze(-1) * torch.stack((dtdx, dtdy), dim=-1)
        return dtdz

phase = StablePhase.apply


class PhaseHarmonic(nn.Module):
    def __init__(self, check_for_nan=False):
        super(PhaseHarmonic, self).__init__()
        self.check_for_nan = check_for_nan

    def forward(self, z, k):
        # check type ok k, move to float
        #if not is_long_tensor(k):
        #    raise TypeError("Expected torch(.cuda).LongTensor but got {}".format(k.type()))
        #if is_double_tensor(z):
        #    k = k.double()
        #else:
        #    k = k.float()

        #s = z.size()
    
        print('z shape',z.shape,z.size())
        z_mod = modulus(z)  # modulus
        print('z_mod shape',z_mod.shape)
        
        # compute phase
        
        theta = phase(z)  # phase
        print('theta shape',theta.shape)
        k = k.unsqueeze(0).unsqueeze(1)
        print('k shape',k.shape)
        for spatial_dim in theta.size()[2:-1]:
            k = k.unsqueeze(-1)
        print('k shape',k.shape)
        ktheta = k * theta
        
        eiktheta = torch.stack((torch.cos(ktheta), torch.sin(ktheta)), dim=-1)

        # compute phase exponent : |z| * exp(i k theta)
        z_pe = z_mod.unsqueeze(-1) * eiktheta

        if z.requires_grad and self.check_for_nan:
            z.register_hook(HookDetectNan("z in PhaseExp"))
                # , torch.stack((z_mod, z_mod), dim=-1), torch.stack((theta, theta), dim=-1)))
            z_mod.register_hook(HookDetectNan("z_mod in PhaseExp"))
            eiktheta.register_hook(HookDetectNan("eiktheta in PhaseExp"))
            z_pe.register_hook(HookDetectNan("z_pe in PhaseExp"))

        return z_pe
