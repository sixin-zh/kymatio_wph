% estimate S(j,q), then S^model bar{x} (j,q), 1 <= q <= Q
% output eps_st(j,q) in mean and std
j = 1; % 1 or 2
Q = 5;
name = 'mrw2dd';
switch name
    case 'tur2a'
        droot = './out2';
        imfn = './data/kymatio_wph_data/ns_randn4_test_N256.mat';
        modelA = './maxent/synthesis/maxent_synthesis_tur2a_kb';
        modelD = './mats/tur2a_dj1_isonoref_J5_L8_export_ks';
    case 'mrw2dd'
        droot = './out2';
        imfn = './data/kymatio_wph_data/demo_mrw2dd_test_N256.mat';
        modelA = './maxent/synthesis/maxent_synthesis_mrw2dd_kb';
        modelD = './mats/mrw2dd_dj1_isonoref_J5_L8_export_ks';
end

%% compute S(j,q)
load(imfn);
siz = size(imgs);
N = siz(1);
% M = siz(3);
assert(siz(1)==siz(2))

% for each img in imgs, shift it with tau such that 2^(j-1) <= |tau| < 2^j
[y,x] = meshgrid(1:N,1:N);
x=x-(siz(1)/2+1);
y=y-(siz(2)/2+1);
modx=fftshift(sqrt(x.^2 + y.^2));
mask = ((modx>=2^(j-1))&(modx<(2^(j-1)+1)));
[taux,tauy]=find(mask==1);

S_ori = zeros(length(taux),Q);

% check white noise OK
% imgs = randn(N,N,M);
% imgs = imgs(:,:,1:M);
Sjq_ori = compute_Sjq(imgs,taux,tauy,Q);

%% compute for model A, mean and std
M = 10;

Sjq_A = zeros(M,Q);
err_A = zeros(M,Q);

for ks = 1:M
    load(sprintf('%s%d.mat',modelA,ks))
    Sjq_A(ks,:) = compute_Sjq(imgs,taux,tauy,Q);
    err_A(ks,:) = abs(Sjq_A(ks,:)-Sjq_ori)./Sjq_ori;
end
err_A_mean = mean(err_A,1);
err_A_std = std(err_A,1);

for q=1:Q
    fprintf('& %.2f (%.2f)',err_A_mean(q),err_A_std(q))
end
fprintf('\n')

%% compute for model D, mean and std
M = 10;

Sjq_D = zeros(M,Q);
err_D = zeros(M,Q);

for ks = 1:M
    load(sprintf('%s%d.mat',modelD,ks-1))
    imgs = permute(imgs,[2,3,1]);
    Sjq_D(ks,:) = compute_Sjq(imgs,taux,tauy,Q);
    err_D(ks,:) = abs(Sjq_D(ks,:)-Sjq_ori)./Sjq_ori;
end
err_D_mean = mean(err_D,1);
err_D_std = std(err_D,1);

for q=1:Q
    fprintf('& %.2f (%.2f)',err_D_mean(q),err_D_std(q))
end
fprintf('\n')
