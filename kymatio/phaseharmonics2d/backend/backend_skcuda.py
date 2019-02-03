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

        ctx.save_for_backward(A,B)
        
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
        A, B = ctx.saved_tensors
        conjA = A.clone()
        conjB = B.clone()
        conjA[:,:,:,:,1] = -A[:,:,:,:,1]
        conjB[:,:,1] = -B[:,:,1]
        m, n = conjB.nelement() // 2, conjA.nelement() // conjB.nelement()
        # n is the B*C
        # m is the M*N
        gradA = conjA.new(conjA.size()) # (n,m), col-major
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
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m*n, 1, gradC.data_ptr(), lda, conjA.data_ptr(), incx, gradB_.data_ptr(), ldc)
        gradB = torch.sum(torch.sum(gradB_,0),0) # (m)
        
        return gradA, gradB


cdgmm = cdgmmMul.apply



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

