import scipy.fftpack
import warnings

def compute_padding(M, N, J):
    """
         Precomputes the future padded size.

         Parameters
         ----------
         M, N : int
             input size

         Returns
         -------
         M, N : int
             padded size
    """
    M_padded = ((M + 2 ** J) // 2 ** J + 1) * 2 ** J
    N_padded = ((N + 2 ** J) // 2 ** J + 1) * 2 ** J
    return M_padded, N_padded

def fft2(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return scipy.fftpack.fft2(x)

from .backend import fft

def ifft2_c2r(x):
    return fft(x,'C2R')/(x.size(-2)*x.size(-3))

def fft2_c2c(x):
    return fft(x,'C2C') # *(x.size(-2)*x.size(-3))

def ifft2_c2c(x):
    return fft(x,'C2C',inverse=True)/(x.size(-2)*x.size(-3))

def periodic_dis(i1, i2, per):
    if i2 > i1:
        return min(i2-i1, i1-i2+per)
    else:
        return min(i1-i2, i2-i1+per)

def periodic_signed_dis(i1, i2, per):
    if i2 < i1:
        return i2 - i1 + per
    else:
        return i2 - i1


def is_long_tensor(z):
    return (isinstance(z, torch.LongTensor) or isinstance(z, torch.cuda.LongTensor))


def count_nans(z):
    print('\nNumber of NaNs:', ((z != z).sum().detach().cpu().numpy()), 'out of', z.size())
    raise SystemExit

class NanError(Exception):
    pass

class HookDetectNan(object):
    def __init__(self, message, *tensors):
        super(HookDetectNan, self).__init__()
        self.message = message
        self.tensors = tensors

    def __call__(self, grad):
        if (grad != grad).any():
            mask = grad != grad
            nan_source = '\n\n'.join([str(tensor[mask]) for tensor in self.tensors])
            print(nan_source)
            raise NanError("NaN detected in gradient: " + self.message)
            # print("NaN detected in gradient at time {}: {}".format(time.time(), self.message))
            # print((grad != grad).nonzero())


class HookPrintName(object):
    def __init__(self, message):
        super(HookPrintName, self).__init__()
        self.message = message

    def __call__(self, grad):
        print(self.message)


class MaskedFillZero(Function):
    @staticmethod
    def forward(ctx, input, mask):
        output = input.clone()
        output.masked_fill_(mask, 0)
        ctx.save_for_backward(mask)
        return output

    def backward(ctx, grad):
        mask, = ctx.saved_variables
        grad.masked_fill(mask, 0)
        return grad, None


masked_fill_zero = MaskedFillZero.apply
