
function data_out = fi_2d_fft(data_in,alpha);

taille = size(data_in);taille = taille(1);

%% Fourier Transform
X=fft2(data_in);
kx=-taille/2:taille/2-1;
[Kx,Ky]=meshgrid(kx,kx);

%% Filtre
%H=0.5;
H = alpha - 1;
Kmod=sqrt(Kx.^2+Ky.^2);
listnull=find(Kmod==0);
Kmod(listnull)=min(Kmod(setdiff(1:length(Kmod(:)),listnull)))*ones(1,length(listnull));
Hfilt=1./(Kmod.^(1+H));

%% Filtrage et FFT inverse
Y=Hfilt.*fftshift(X);
Y=ifftshift(Y);
data_out=ifft2(Y);

end