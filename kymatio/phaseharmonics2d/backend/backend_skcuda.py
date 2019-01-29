# Authors: Edouard Oyallon, Sergey Zagoruyko

from collections import defaultdict, namedtuple
import torch
from skcuda import cublas
import cupy
from string import Template

import torch.nn as nn
from torch.nn import ReflectionPad2d
from torch.autograd import Function
import numpy as np

NAME = 'skcuda'

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


Stream = namedtuple('Stream', ['ptr'])


def getDtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


def iscomplex(input):
    return input.size(-1) == 2


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

def real(z):
    return z[..., 0]


def imag(z):
    return z[..., 1]


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


class cdgmmMul(Function):
    @staticmethod
    def forward(ctx, A, B):
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

        if not A.is_cuda:
            raise RuntimeError('Use the torch backend for cpu tensors!')

        conjA = A.clone()
        conjB = B.clone()
        conjA[:,:,:,:,1] = -A[:,:,:,:,1]
        conjB[:,:,1] = -B[:,:,1]
        ctx.save_for_backward(conjA, conjB)
        
        C = A.new(A.size())
        m, n = B.nelement() // 2, A.nelement() // B.nelement()
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
        return C
    
    @staticmethod
    def backward(ctx, grad_output):
        conjA, conjB =  ctx.saved_tensors
        m, n = conjB.nelement() // 2, conjA.nelement() // conjB.nelement()
        # n is the B*C
        # m is the M*N
        gradA = conjA.new(conjA.size()) # (n,m), col-major
        #gradB = conjB.new(conjB.size()) # (m)
        gradC = grad_output # (n,m), col-major
        # grad_A = grad_C * conj(B)
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, gradC.data_ptr(), lda, conjB.data_ptr(), incx, gradA.data_ptr(), ldc)
        
        # grad_B = sum_n grad_C * conj(A)
        # view grad_C and conjA as one vector of size n*m
        gradB_ = gradC.new(gradC.size()) # mul(gradC,conjA) # (B,C,M,N,2)
        lda = m*n
        ldc = m*n
        incx = 1
        #handle = torch.cuda.current_blas_handle()
        #stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m*n, 1, gradC.data_ptr(), lda, conjA.data_ptr(), incx, gradB_.data_ptr(), ldc)

       # gradB_ = mul(gradC,conjA) # (B,C,M,N,2)
        gradB = torch.sum(torch.sum(gradB_,0),0) # 
        
        return gradA, gradB


cdgmm = cdgmmMul.apply

class SubInitSpatialMeanC(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            minput = torch.mean(minput, -2, True)
            minput = torch.mean(minput, -3, True)
            self.minput = minput
            print('sum of minput',self.minput.sum())
            
        output = input - self.minput
        return output



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

        #print('z shape',z.shape,z.size())
        z_mod = modulus(z)  # modulus
        #print('z_mod shape',z_mod.shape)

        # compute phase
        theta = phase(z)  # phase
        #print('theta shape',theta.shape)
        #k = k.unsqueeze(0) #.unsqueeze(1)
        #print('k shape',k.shape)
        #k = k.unsqueeze(-1).unsqueeze(-1)
        #print('k shape',k.shape)
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


def mul(z1, z2):
    zr = real(z1) * real(z2) - imag(z1) * imag(z2)
    zi = real(z1) * imag(z2) + imag(z1) * real(z2)
    z = torch.stack((zr, zi), dim=-1)
    return z

class cdgmmMulcu(Function):
    @staticmethod
    def forward(ctx, A, B):
        # assume A and B has the same size , with last dim = 2
        A, B = A.contiguous(), B.contiguous()
                
        if not iscomplex(A) or not iscomplex(B):
            raise TypeError('The input, filter and output should be complex')

        if A.nelement() != B.nelement():
            raise TypeError('The input and filter should have same size')

        if type(A) is not type(B):
            raise RuntimeError('A and B should be same type!')

        if not A.is_cuda:
            raise RuntimeError('Use the torch backend for cpu tensors!')

        conjA = A.clone()
        conjB = B.clone()
        conjA[...,1] = -A[...,1]
        conjB[...,1] = -B[...,1]
        ctx.save_for_backward(conjA, conjB)
        
        C = A.new(A.size())
        m, n = B.nelement() // 2, A.nelement() // B.nelement()
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
        return C
    
    @staticmethod
    def backward(ctx, grad_output):
        conjA, conjB =  ctx.saved_tensors
        m, n = conjB.nelement() // 2, conjA.nelement() // conjB.nelement()
        # n is the B*C
        # m is the M*N
        gradA = conjA.new(conjA.size()) # (n,m), col-major
        #gradB = conjB.new(conjB.size()) # (m)
        gradC = grad_output # (n,m), col-major
        # grad_A = grad_C * conj(B)
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, gradC.data_ptr(), lda, conjB.data_ptr(), incx, gradA.data_ptr(), ldc)
        
        # grad_B = sum_n grad_C * conj(A)
        # view grad_C and conjA as one vector of size n*m
        gradB = gradC.new(gradC.size()) # mul(gradC,conjA) # (...,2)
        lda = m*n
        ldc = m*n
        incx = 1
        #handle = torch.cuda.current_blas_handle()
        #stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m*n, 1, gradC.data_ptr(), lda, conjA.data_ptr(), incx, gradB.data_ptr(), ldc)

       # gradB_ = mul(gradC,conjA) # (B,C,M,N,2)
        #gradB = torch.sum(torch.sum(gradB_,0),0) # 
        
        return gradA, gradB

    
mulcu = cdgmmMulcu.apply
