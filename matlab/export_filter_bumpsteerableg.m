clear all
close all

addpath ../scatnet-0.2a
addpath_scatnet;

%% get data and estimate spectral
N=64;
J=6;
L=4;
filtopts = struct();
filtopts.J=J;
filtopts.L=L;
filtopts.full2pi=1;

filter_id=1;
filtopts.fcenter=0.425; % om in [0,1], unit 2pi
filtopts.gamma1=1;
[filnew,lpal]=bumpsteerableg_wavelet_filter_bank_2d([N N], filtopts);

%% plot lpal
figure; imagesc(fftshift(lpal)); colormap gray
title(sprintf('Littlewood-Paley: xi0=%g*2pi',filtopts.fcenter))
colorbar

%% compute maps
filid=1;
% figure;
savelist = [];
for j=1:J
    for q = 1:2*L
        fil=filnew.psi.filter{filid};
        filid=filid+1;
        
%         if N==256 && j==5 && q==1
%             psi_la = ifft2(fil.coefft{1});
%             subplot(131)
%             %plot(ifftshift(imag(psi_la(30,:))));
%             imagesc(real(fftshift(psi_la)))
%             colorbar
%             axis square
%             title('real \psi','FontSize',20)
%             subplot(132)
%             %imagesc(angle(fftshift(ifft2(fil.coefft{1}))))
%             imagesc(imag(fftshift(psi_la)))
%             title('imag \psi','FontSize',20)
%             colorbar
%             axis square
%             subplot(133)
%             imagesc(real(fftshift(fil.coefft{1})))
%             title('hat \psi','FontSize',20) % ,'Interpreter','latex')
%             colorbar
%             axis square
%             colormap gray
%         end
        
        varname1 = sprintf('bumpg_fftpsi_j%d_q%d_re',j,q);
        eval(sprintf('%s = real(fil.coefft{1});',varname1));
        varname2 = sprintf('bumpg_fftpsi_j%d_q%d_im',j,q);
        eval(sprintf('%s = imag(fil.coefft{1});',varname2));
        savelist = [savelist, {varname1}];
        savelist = [savelist, {varname2}];
    end
end
assert(length(filnew.psi.filter)==filid-1);

bumpg_fftphi_re = real(filnew.phi.filter.coefft{1});
savelist = [savelist, {'bumpg_fftphi_re'}];

bumpg_fftphi_im = imag(filnew.phi.filter.coefft{1});
savelist = [savelist, {'bumpg_fftphi_im'}];

%path = './filters_local/';
path = './filters/';
filename = sprintf('bumpsteerableg%d_fft2d_N%d_J%d_L%d.mat',filter_id,N,J,L);
disp(savelist)
if exist(sprintf('%s/%s',path,filename)) > 0
    error(sprintf('file %s existed, can not export',filename))
end
save(sprintf('%s/%s',path,filename),savelist{:})
disp(filename)
