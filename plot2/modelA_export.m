name = 'bubbles';

switch name
    case 'anisotur2a'
        droot = './out2';
        tkt = 'pwregress_maxent_bumps2d_dj0_nor_Delta2_anisotur2a_J5_L8_K10_m1';
    case 'tur2a'
        droot = './out2';
        tkt = 'pwregress_maxent_bumps2d_dj0_nor_Delta2_tur2a_J5_L8_K10_m1';
%         tkt =
%         'pwregress_maxent_bumps2d_dj0_nor_Delta0_tur2a_J5_L8_K1_m1'; %
%         synthesis0
    case 'mrw2dd'
        droot = './out2';
        tkt = 'pwregress_maxent_bumps2d_dj0_nor_Delta2_mrw2dd_J5_L8_K10_m1';
        % TODO export kb=2:10 for high-order moments and long range spatial
    case 'bubbles'
        droot = './out2';
        tkt = 'pwregress_maxent_bumps2d_dj0_nor_Delta2_bubbles_J5_L8_K1_m1';
end

% load(sprintf('%s/%s.mat',droot,tkt));
N = 256; % size(imgs,1);
M = 10;
for kb = 1:1 % 10
    load(sprintf('%s/%s_kb%d.mat',droot,tkt,kb));
    % get M samples from hXrec
    imgs = zeros(N,N,M);
    for ks=1:M
        Xrec = randnpsd(hXrec);
        imgs(:,:,ks) = Xrec;
    end
    save(sprintf('./synthesis/maxent_synthesis_%s_kb%d.mat',name,kb),'imgs') % 0 for Delta=0
end