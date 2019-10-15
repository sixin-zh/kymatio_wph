function Ktau = estimate_Cov_kjq(imgs,J,L,k,j,q)

% Compute K(la,k,la',k') where k=k', la=(j,ell,0), la'=(j,ell,a e_theta)
% estimate K at all ell and a 

if size(imgs,2)==size(imgs,3)
    M=size(imgs,1); % number of samples
    N=size(imgs,2); % size of image
    assert(M>=1 && N==size(imgs,3));
elseif size(imgs,2)==size(imgs,1)
    N=size(imgs,1); % number of samples
    M=size(imgs,3); % size of image
    imgs = permute(imgs,[3,1,2]);
else
    assert(0)
end


% get filters
filtopts = struct();
filtopts.J=J;
filtopts.L=L;
filtopts.full2pi=0; % no need for 2pi angles
filtopts.fcenter=0.425; % om in [0,1], unit 2pi
filtopts.gamma1=1;
% cache filters
cachename = sprintf('cdfBump_N%d_J%d_L%d_full2pi0.mat',N,J,L);
if exist(cachename)
    load(cachename)
else
    [filnew,~]=bumpsteerableg_wavelet_filter_bank_2d([N N], filtopts);
    save(cachename,'filnew')
end

% reshape filters
fftfil_la= cell(J,L);
%fftfil_phi = filnew.phi.filter.coefft{1};

filid=1;
for j1=0:J-1
    for q1=0:L-1
        fftfil_la{j1+1,q1+1} = filnew.psi.filter{filid}.coefft{1} * 2^j1; % to make it frame
        filid = filid +1;
    end
end

% no low pass filter here

fftim = zeros(N,N,M);
for m=1:M
    im = reshape(imgs(m,:,:),N,N);
    fftim(:,:,m) = fft2(im);
end

corr_tau = zeros(N);
mean_a = 0;
meansq_a = 0;

for m=1:M
    xpsi_a = ifft2(fftim(:,:,m) .* fftfil_la{j,q});
    xpsik_a = abs(xpsi_a) .* exp(1i * k * angle(xpsi_a));
    mean_a = mean_a + mean(xpsik_a(:));
    meansq_a = meansq_a + mean(abs(xpsik_a(:)).^2);
    corr_tau = corr_tau + ifft2(abs(fft2(xpsik_a)).^2) / (N^2);
end

mean_a = mean_a / M;
meansq_a = meansq_a / M;
corr_tau = corr_tau / M;

Ktau = corr_tau - abs(mean_a)^2;

%assert( abs ( Ktau(1,1) - (meansq_a-abs(mean_a)^2) ) < eps);

end