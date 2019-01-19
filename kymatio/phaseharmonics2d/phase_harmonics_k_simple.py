# implement basic phase harmonics based on John's code to check correctness

__all__ = ['PhaseHarmonics2d']

import warnings
import torch
import numpy as np
import torch.nn.functional as F
from .backend import cdgmm, Modulus, SubsampleFourier, fft, Pad, unpad, SubInitMean, StablePhaseExp, PhaseExp
from .filter_bank import filter_bank
from .utils import compute_padding, fft2_c2c, ifft2_c2r, ifft2_c2c

class PhaseHarmonics2d(object):
    def __init__(self, M, N, J, L, j_max, l_max, k_max, k_type='linear', addhaar=False, gpu=False):
        self.M, self.N, self.J, self.L = M, N, J, L # size of image, max scale, number of angles
        self.pre_pad = False # no padding
        self.j_max = j_max # max scale interactions
        self.l_max = l_max # max angular interactions
        self.k_max = k_max # max k used in phase harmonics
        self.k_type = k_type
        self.addhaar = addhaar # filter bank with haar filters
        self.cache = False # cache filter bank
        self.gpu = gpu # if to use gpu
        if self.l_max > self.L:
            raise (ValueError('l_max must be <= L'))
        self.build()

    def build(self):
        self.meta = None
        self.modulus = Modulus()
        self.pad = Pad(0, pre_pad = self.pre_pad)
        self.subsample_fourier = SubsampleFourier()
        self.phaseexp = StablePhaseExp.apply
        self.subinitmean = SubInitMean(2)
        self.phase_exp = PhaseExp(K=self.k_max,k_type=self.k_type,keep_k_dim=True,check_for_nan=False)
        self.M_padded, self.N_padded = self.M, self.N
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L, self.addhaar, self.cache)           
        self.Psi = filters['psi']
        self.Phi = [filters['phi'][j] for j in range(self.J)]
        if self.addhaar:
            self.Psi0 = filters['psi0']
    
    def cuda(self):
        """
            Moves tensors to the GPU
        """
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        """
            Moves tensors to the CPU
        """
        return self._type(torch.FloatTensor)

    def forward(self, input):
        J = self.J
        M = self.M
        N = self.N
        phi = self.Phi # low pass
        psi = self.Psi # high pass
        n = 0

        pad = self.pad
        modulus = self.modulus

        # denote
        # nb=batch number
        # nc=number of color channels
        # input: (nb,nc,M,N)
        U_r = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        U_0_c = fft2_c2c(U_r) # fft2 -> (nb,nc,M,N,2)

        set_meta = False
        if self.meta is None:
            set_meta = True
            self.meta = dict()

        # out coefficients: S
        S = None
        
        # compute x * psi_{j,theta}, no subsampling
        for n1 in range(len(psi)):
            j1 = psi[n1]['j']
            U_1_c = cdgmm(U_0_c, psi[n1][0])
            U_1_c = ifft2_c2c(U_1_c)
            if set_meta:
                self.meta[n1] = (psi[n1]['j'],psi[n1]['theta'])

        return S, self.meta

    def __call__(self, input):
        return self.forward(input)
