% Tutorial 1: How to get scattering coefficients?
cd /Users/zsx/Coding/scatlearn/src/scatnet-0.2/ScatNetLight/scatnet_light
addpath_scatnet

x = imread('/Users/zsx/Coding/scatlearn/src/scatnet-0.2/demo/lena.ppm');
lena = mean(x,3)./255;
imagesc(x);
colormap gray

% Let us get a toy example
N=512; % 256;
M=512; % 256;
my_lena=lena;
%my_lena=rgb2yuv(imresize(my_lena,[N,M]));
%my_lena=my_lena(:,:,1);


% First define the type of scattering representation you want to use
% If you are not sure of the representation you want to use, just type the 
% following, it will  build the operators to get translation scattering.

% Number of layers
scat_opt.M=2;
% First layer
filt_opt.layer{1}.translation.J=3; % 4
filt_opt.layer{1}.translation.L=8;
scat_opt.layer{1}.translation.oversampling=0;
filt_opt.layer{1}.translation.n_wavelet_per_octave=1;
filt_opt.layer{1}.translation.min_margin = 0;

% Second layer
filt_opt.layer{2}.translation.J=3; % 4
filt_opt.layer{2}.translation.L=8;
scat_opt.layer{2}.translation.oversampling=0;
scat_opt.layer{2}.translation.type='t';
filt_opt.layer{2}.translation.min_margin = 0;

% Third layer(copy of the previous one)
filt_opt.layer{3}=filt_opt.layer{2};
scat_opt.layer{3}=scat_opt.layer{2};

% Wop is a cell within the operators for scattering computations are.
% Filters simply correspond to the weights of the filters used in this
% network.
[Wop,filters]=wavelet_factory_2d([N,M],filt_opt,scat_opt);

% Let us get scattering coefficients!
S=scat(my_lena,Wop);% Should take about 2s
max(S{1}.signal{1}(:)) 
% S is a structure that contains scattering coefficients
% - meta contains meta informations
% - signal contains the signals as a cell
S{3}.meta.j % scales of the coefficients

figure, % for S{1}.signal, the order of the coordinate is (u1,u2)
imagesc(S{1}.signal{1}) % Averaged image, S_0
figure, % for S{2}.signal, the order of the coordinate is {j1}(u1,u2,theta1)
imagesc(S{2}.signal{1}(:,:,1)) % Theta1=((1-1)/8)*2pi, j1=0
figure, % for S{3}.signal, the order of the coordinate is {j2}(u1,u2,theta1,theta2,j1)
imagesc(S{3}.signal{2}(:,:,3,2,1)) % Theta1=((3-1)/8)*2pi, Theta2=(2-1)/8, j1=0, j2=1

% Now let us get a vector that we can use for classification
% out=scat_to_tensor(S);% concatenation of all the scattering coefficients
% out_space=scat_to_tensor(S,1);% concatenation that does preserve space
% 
% size(out)
% size(out_space)
% figure,
% vol3d('Cdata',out_space(:,:,34:end)) % 1 S0, 2:33 S1, 34:end S2, let us visualize S2!