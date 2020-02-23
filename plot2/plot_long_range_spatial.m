addpath ../scatnet-0.2a
addpath_scatnet
addpath ./altmany-export_fig-4703a84

for j = 1:3
    for k = 0:1

J=5; L=8; N=256;
name = 'mrw2dd';
switch name
    case 'tur2a'
        datamat = './data/ns_randn4_test_N256.mat';
        modelAmat = './cerfeuil_synthesis/maxent_synthesis_tur2a';
        modelDmat = './cerfeuil_synthesis/tur2a_modelD_synthesis';
    case 'mrw2dd'
        datamat = './data/demo_mrw2dd_test_N256.mat';
        modelAmat = './cerfeuil_synthesis/maxent_synthesis_mrw2dd';
        modelDmat = './cerfeuil_synthesis/mrw2dd_modelD_synthesis';
end

load (datamat)

sigma2 = zeros(L,1);
Ra_ori = zeros(1,N/2+1);
for q=1:L
    Ktau = estimate_Cov_kjq(imgs,J,L,k,j,q);
    sigma2(q) = Ktau(1,1); 
    % this is a normalization constant which is the same for all the rest
    Rtau = abs(Ktau/sigma2(q));
    [Ra,a]=mySpectre2Dmax(Rtau);
    Ra_ori = max(Ra, Ra_ori);
end

suba = 0:2^(j-1):2^(J+(j-1));
plot(suba/2^(j-1),Ra_ori(suba+1),'Color',[0.8500, 0.3250, 0.0980])

%% Model A
KS = 10;
Ra_A = zeros(1,N/2+1);
for q=1:L
    for ks = 0:KS-1
        load (sprintf('%s_kb%d.mat',modelAmat,ks+1))
        Ktau = estimate_Cov_kjq(imgs,J,L,k,j,q);
        if ks==0
            Rtau = Ktau/sigma2(q);
        else
            Rtau = Rtau + Ktau/sigma2(q);
        end
    end
    Rtau = abs(Rtau / KS);
    [Ra,a]=mySpectre2Dmax(Rtau);
    Ra_A = max(Ra, Ra_A);
end

hold on
plot(suba/2^(j-1),Ra_A(suba+1),'o','Color',[0.4940, 0.1840, 0.5560])
hold off

%% Model D
KS = 10;
Ra_D = zeros(1,N/2+1);
for q=1:L
    for ks = 0:KS-1
        load (sprintf('%s_ks%d.mat',modelDmat,ks))
        Ktau = estimate_Cov_kjq(imgs,J,L,k,j,q);
        if ks==0
            Rtau = Ktau/sigma2(q);
        else
            Rtau = Rtau + Ktau/sigma2(q);
        end
    end
    Rtau = abs(Rtau / KS);
    [Ra,a]=mySpectre2Dmax(Rtau);
    Ra_D = max(Ra, Ra_D);
end

hold on
plot(suba/2^(j-1),Ra_D(suba+1),'--','Color',[0, 0.4470, 0.7410])
hold off

set(gca, 'FontSize',24)
title(sprintf('k=%d,j=%d',k,j), 'FontSize',26,'FontWeight','bold')
axis tight
if strcmp(name,'tur2a')==1
    axis([0 12 0 1.1])
else
    axis([0 20 0 1.1])
end
proot = 'curves';
export_fig(sprintf('%s/%s_Ca_j%d_k%d.pdf',proot,name,j,k),'-pdf','-transparent')

    end
end