clear all
close all

addpath ../scatnet-0.2a
addpath_scatnet;

%% get image and filter
J=3;L=4;
dj = 1;
dk = 1;
dl = L;
N=32;
imgs = zeros(N,N,1);
S=1;
imgs(N/2,N/2,1)=1;

filtopts = struct();
filtopts.J=J;
filtopts.L=L;
filtopts.full2pi=1;
filtopts.fcenter=0.425; % om in [0,1], unit 2pi
filtopts.gamma1=1;
[filnew,lpal]=bumpsteerableg_wavelet_filter_bank_2d([N N], filtopts);

%% compute correlation at j1,ell1,k1,j2,ell2,k2

L2=L*2;
im = imgs(:,:,1);
fftimg = fft2(im);
j1=0; ell1=0;k1=1;
j2=0; ell2=0; k2=0;
corr = compute_corr(fftimg,filnew,J,L2,j1,ell1,k1,j2,ell2,k2)
    