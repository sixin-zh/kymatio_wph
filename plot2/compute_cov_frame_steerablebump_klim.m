function [covre,covim] = compute_cov_frame_steerablebump_klim(imgs,J,L,Delta,kmin,kmax,nbcores)
    assert( size(imgs,2) == size(imgs,3) )
    M = size(imgs,1);
    N = size(imgs,2);
    J=double(J);
    L=double(L);
    kmin=double(kmin);
    kmax=double(kmax);
    
    nbcores=double(nbcores);
    parpool('local',nbcores);
    
    [wcoeffs] = compute_coeff_frame_steerablebump_klim_par(imgs,N,J,L,Delta,kmin,kmax);
    nbw2 = size(wcoeffs,1); NJ = size(wcoeffs,2);
    wcoeffs = reshape(wcoeffs,[nbw2,NJ*NJ*M]);
    meanr2 = mean(wcoeffs,2);
    corrr2 = zeros(nbw2,nbw2);
    parfor id=1:size(wcoeffs,2)
        corrr2 = corrr2 + wcoeffs(:,id)*wcoeffs(:,id)';
    end
    corrr2 = corrr2 / size(wcoeffs,2);
    covr2 = corrr2 - meanr2*meanr2';
    
    covre = real(covr2);
    covim = imag(covr2);
    
    delete(gcp('nocreate'))
end