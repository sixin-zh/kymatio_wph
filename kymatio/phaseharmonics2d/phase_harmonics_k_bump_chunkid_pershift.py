# add periodic shift cov coefficients

__all__ = ['PhaseHarmonics2d']

import warnings
import math
import torch
import numpy as np
import scipy.io as sio
#import torch.nn.functional as F
from .backend import cdgmm, Modulus, fft, \
    Pad, SubInitSpatialMeanC, PhaseHarmonics2, mulcu
from .filter_bank import filter_bank
from .utils import fft2_c2c, ifft2_c2c, periodic_dis

class PhaseHarmonics2d(object):
    # nb_chunks = J, so that each dn can be applied to each chunk with the same shift,
    # chunk_id is the scale parameter j
    def __init__(self, M, N, J, L, dn, delta_l, nb_chunks, chunk_id, devid=0):
        self.M, self.N, self.J, self.L = M, N, J, L # size of image, max scale, number of angles [0,pi]
        self.dn = dn # shift 2^j * dn at scale j
        self.dl = delta_l # max angular interactions
        assert(nb_chunks == J)
        self.nb_chunks = nb_chunks # number of chunks to cut whp cov
        self.chunk_id = chunk_id
        self.devid = devid # gpu id
        assert( self.chunk_id < self.nb_chunks ) # chunk_id = 0..nb_chunks-1, are the wph cov
        if self.dl > self.L:
            raise (ValueError('delta_l must be <= L'))

        self.pre_pad = False # no padding
        self.cache = False # cache filter bank
        self.build()

    def build(self):
        check_for_nan = False # True
        self.modulus = Modulus()
        self.pad = Pad(0, pre_pad = self.pre_pad)
        self.phase_harmonics = PhaseHarmonics2.apply
        
        self.M_padded, self.N_padded = self.M, self.N
        self.filters_tensor()
        self.idx_wph = self.compute_idx()
        self.this_wph = self.get_this_chunk(self.nb_chunks, self.chunk_id)
        self.subinitmean1 = SubInitSpatialMeanC()
        self.subinitmean2 = SubInitSpatialMeanC()
        
    def filters_tensor(self):
        J = self.J
        L = self.L
        L2 = L*2

        assert(self.M == self.N)
        matfilters = sio.loadmat('./matlab/filters/bumpsteerableg1_fft2d_N' + str(self.N) + '_J' + str(self.J) + '_L' + str(self.L) + '.mat')

        fftphi = matfilters['filt_fftphi'].astype(np.complex_)
        hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)

        fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)
        hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)

        self.hatpsi = torch.FloatTensor(hatpsi) # (J,L2,M,N,2)
        self.hatphi = torch.FloatTensor(hatphi) # (M,N,2)

        #print('filter shapes')
        #print(self.hatpsi.shape)
        #print(self.hatphi.shape)

    def get_this_chunk(self, nb_chunks, chunk_id):
        # cut self.idx_wph into smaller pieces

        nb_cov = len(self.idx_wph['la1'])
        print('nb cov is', nb_cov)
        max_chunk = nb_cov // nb_chunks
        nb_cov_chunk = np.zeros(nb_chunks,dtype=np.int32)
        for idxc in range(nb_chunks):
            if idxc < nb_chunks-1:
                nb_cov_chunk[idxc] = int(max_chunk)
            else:
                nb_cov_chunk[idxc] = int(nb_cov - max_chunk*(nb_chunks-1))
                assert(nb_cov_chunk[idxc] > 0)

        this_wph = dict()
        offset = int(0)
        for idxc in range(nb_chunks):
            if idxc == chunk_id:
                this_wph['la1'] = self.idx_wph['la1'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['la2'] = self.idx_wph['la2'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['k1'] = self.idx_wph['k1'][:,offset:offset+nb_cov_chunk[idxc],:,:]
                this_wph['k2'] = self.idx_wph['k2'][:,offset:offset+nb_cov_chunk[idxc],:,:]
            offset = offset + nb_cov_chunk[idxc]

        print('this chunk', self.chunk_id, 'la1',self.this_wph['la1'])
        
        return this_wph

    def compute_ncoeff(self):
        # return number of mean (nb1) and cov (nb2) of all idx
        L = self.L
        L2 = L*2
        J = self.J
        dj = self.dj
        dl = self.dl
        dk = self.dk
        
        hit_nb1 = dict() # hash table
        hit_nb2 = dict() # value counts either real or complex numbers
        
        # TODO
        
        #print('hit nb1 values',list(hit_nb1.values()))
        nb1 = np.array(list(hit_nb1.values()), dtype=int).sum() + 1 # plus last phiJ channel
        nb2 = np.array(list(hit_nb2.values()), dtype=int).sum() + 1 # plus last phiJ channel

        return nb1, nb2
    
    def compute_idx(self):
        L = self.L
        L2 = L*2
        J = self.J
        dl = self.dl

        idx_la1 = []
        idx_la2 = []
        idx_k1 = []
        idx_k2 = []

        # j1=j2, k1=1, k2=0 or 1
        #      , k1=0, k2=0
        for j1 in range(J):
            j2 = j1
            for ell1 in range(L2):
                for ell2 in range(L2):
                    if periodic_dis(ell1, ell2, L2) <= dl:
                        k1 = 1
                        k2 = 0
                        idx_la1.append(L2*j1+ell1)
                        idx_la2.append(L2*j2+ell2)
                        idx_k1.append(k1)
                        idx_k2.append(k2)
                        k1 = 1
                        k2 = 1
                        idx_la1.append(L2*j1+ell1)
                        idx_la2.append(L2*j2+ell2)
                        idx_k1.append(k1)
                        idx_k2.append(k2)
                        k1 = 0
                        k2 = 0
                        idx_la1.append(L2*j1+ell1)
                        idx_la2.append(L2*j2+ell2)
                        idx_k1.append(k1)
                        idx_k2.append(k2)

        idx_wph = dict()
        idx_wph['la1'] = torch.tensor(idx_la1).type(torch.long)
        idx_wph['k1'] = torch.tensor(idx_k1).type(torch.long).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        idx_wph['la2'] = torch.tensor(idx_la2).type(torch.long)
        idx_wph['k2'] = torch.tensor(idx_k2).type(torch.long).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return idx_wph

    def _type(self, _type, devid=None):
        self.hatpsi = self.hatpsi.type(_type)
        self.hatphi = self.hatphi.type(_type)
        if devid is not None:
            self.hatpsi = self.hatpsi.to(devid)
            self.hatphi = self.hatphi.to(devid)
        #print('in _type',type(self.hatpsi))
        self.pad.padding_module.type(_type)
        return self

    def cuda(self):
        """
            Moves tensors to the GPU
        """
        devid = self.devid
        print('call cuda with devid=', devid)
        assert(devid>=0)
        if self.chunk_id < self.nb_chunks:
            self.this_wph['la1'] = self.this_wph['la1'].type(torch.cuda.LongTensor).to(devid)
            self.this_wph['la2'] = self.this_wph['la2'].type(torch.cuda.LongTensor).to(devid)
            self.this_wph['k1'] = self.this_wph['k1'].type(torch.cuda.FloatTensor).to(devid)
            self.this_wph['k2'] = self.this_wph['k2'].type(torch.cuda.FloatTensor).to(devid)

        return self._type(torch.cuda.FloatTensor, devid)

    def cpu(self):
        """
            Moves tensors to the CPU
        """
        print('call cpu')
        return self._type(torch.FloatTensor)

    def forward(self, input):
        J = self.J
        M = self.M
        N = self.N
        L2 = self.L*2
        dn = self.dn
        dl = self.dl
        pad = self.pad
        
        # denote
        # nb=batch number
        # nc=number of color channels
        # input: (nb,nc,M,N)
        x_c = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        hatx_c = fft2_c2c(x_c) # fft2 -> (nb,nc,M,N,2)
        #print('nbchannels',nb_channels)
        #if self.chunk_id < self.nb_chunks:
        nb = hatx_c.shape[0]
        nc = hatx_c.shape[1]
        hatpsi_la = self.hatpsi # (J,L2,M,N,2)
        assert(nb==1 and nc==1) # for submeanC
        nb_channels = self.this_wph['la1'].shape[0]
        Sout = input.new(nb, nc, nb_channels, \
                         1, 1, 2) # (nb,nc,nb_channels,1,1,2)
        for idxb in range(nb):
            for idxc in range(nc):
                hatx_bc = hatx_c[idxb,idxc,:,:,:] # (M,N,2)
                hatxpsi_bc = cdgmm(hatpsi_la, hatx_bc) # (J,L2,M,N,2)
                xpsi_bc = ifft2_c2c(hatxpsi_bc)
                # reshape to (1,J*L,M,N,2)
                xpsi_bc = xpsi_bc.view(1,J*L2,M,N,2)
                # select la1, et la2, P_c = number of |la1| in this chunk
                xpsi_bc_la1 = torch.index_select(xpsi_bc, 1, self.this_wph['la1']) # (1,P_c,M,N,2)
                xpsi_bc_la2 = torch.index_select(xpsi_bc, 1, self.this_wph['la2']) # (1,P_c,M,N,2)
                # shift xpsi_bc_la2 by 2^j*dn
                #print('xpsi la1 shape', xpsi_bc_la1.shape)
                #print('xpsi la2 shape', xpsi_bc_la2.shape)
                k1 = self.this_wph['k1']
                k2 = self.this_wph['k2']
                xpsi_bc_la1k1 = self.phase_harmonics(xpsi_bc_la1, k1) # (1,P_c,M,N,2)
                xpsi_bc_la2k2 = self.phase_harmonics(xpsi_bc_la2, -k2) # (1,P_c,M,N,2)
                # sub spatial mean along M and N
                xpsi0_bc_la1k1 = self.subinitmean1(xpsi_bc_la1k1) # (1,P_c,M,N,2)
                xpsi0_bc_la2k2 = self.subinitmean2(xpsi_bc_la2k2) # (1,P_c,M,N,2)
                # compute mean spatial
                corr_xpsi_bc = mulcu(xpsi0_bc_la1k1,xpsi0_bc_la2k2) # (1,P_c,M,N,2)
                corr_bc = torch.mean(torch.mean(corr_xpsi_bc,-2,True),-3,True) # (1,P_c,1,1,2)
                Sout[idxb,idxc,:,:,:,:] = corr_bc[0,:,:,:,:]

        return Sout
        
    def __call__(self, input):
        return self.forward(input)
