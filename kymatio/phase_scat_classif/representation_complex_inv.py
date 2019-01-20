from utils import periodize, unpad, cdgmm, add_imaginary_part, prepare_padding_size, \
    pad, modulus, cast, periodic_dis, periodic_signed_dis
from complex_utils import conjugate, phase_exp, mul
from torch.autograd import Variable
from FFT import fft_c2c, ifft_c2r, ifft_c2c
import torch
import torch.nn.functional as F
from filters_banks import filters_bank, phase_filters_bank
import numpy as np
import torch
from tqdm import tqdm

from representation_complex import phase_harmonic_cor

# compute spatial averaging
def compute_phase_harmonic_cor_inv(X, J, L, delta, l_max, batch_size):

    M, N = X.shape[-2], X.shape[-1]
    M_padded, N_padded = prepare_padding_size(M, N, J)

    filters = filters_bank(M_padded, N_padded, J, L)

    psi = filters['psi']
    phi = [filters['phi'][j] for j in range(J)]

    psi, phi = cast(psi, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    phase_harmonics = phase_harmonic_cor(X[0:batch_size].cuda(), phi, psi, J, L, delta, l_max).cpu() # (nb,nc,nch,Mj,Nj,2)
    # remove .cuda() to keep on CPU

    nb_batches = X.shape[0] // batch_size

    for idx_batch in tqdm(range(1, nb_batches)):
        phase_harmonics_tmp = phase_harmonic_cor(X[idx_batch * batch_size: (idx_batch + 1) * batch_size].cuda(), phi,
                                                 psi, J, L, delta, l_max).cpu()
        # remove .cuda() to keep on CPU
        phase_harmonics = torch.cat([phase_harmonics, phase_harmonics_tmp], dim=0)

    phase_harmonics_inv = torch.mean(torch.mean(phase_harmonics,-2,True),-3,True)

    return phase_harmonics_inv
