function [wcoeffs] = compute_coeff_frame_steerablebump_klim_par(imgs,N,J,L,Delta,kmin,kmax)
    % build filters
    addpath ./scatnet-0.2a
    addpath_scatnet;

    filtopts = struct();
    filtopts.J=J;
    filtopts.L=L;
    filtopts.full2pi=1;
    
    filtopts.fcenter=0.425; % om in [0,1], unit 2pi
    filtopts.gamma1=1;
    [filnew,lpal]=bumpsteerableg_wavelet_filter_bank_2d([N N], filtopts);
    
    fftpsi = zeros(N,N,J,2*L);
    filid = 1;
    for j=1:J
        for q = 1:2*L
            fil=filnew.psi.filter{filid};
            fftpsi(:,:,j,q) = fil.coefft{1}* 2^j;
            filid=filid+1;
        end
    end
    assert(length(filnew.psi.filter)==filid-1);
    
    fftphi = filnew.phi.filter.coefft{1} *2^(J);
    
    % compute frame coefficients
    M = size(imgs,1);
    assert(size(imgs,2)==N)
    assert(size(imgs,3)==N)
    
    ds = 2^(J-1);
    NJ=  N/ds;
    % count number of channels
    idw = 1;
    for j=1:J
        for q = 1:2*L
            for k=kmin:min(kmax,2^(j-1))
                for dx = -Delta:Delta
                    for dy = -Delta:Delta
                        idw = idw + 1;
                    end
                end
            end
        end
    end
    for dx = -Delta:Delta
        for dy = -Delta:Delta
            idw = idw + 1;
        end
    end
    % compute coeffs
    nbw = idw - 1; % (J*2*L*(kmax-kmin+1)+1)*(2*Delta+1)^2;
    wcoeffs = zeros(nbw,NJ,NJ,M);
    parfor m=1:M
        im = reshape(imgs(m,:,:),N,N);
        fftim = fft2(im);
        wcs = zeros(nbw,NJ,NJ);
        idw = 1;
        for j=1:J
            for q = 1:2*L
                impsi = ifft2(fftim .* fftpsi(:,:,j,q));
                impsim = abs(impsi);
                impsip = angle(impsi);
                for k=kmin:min(kmax,2^(j-1))
                    impsik = impsim .* exp( 1i * k * impsip);
                    for dx = -Delta:Delta
                        for dy = -Delta:Delta
                            subgridx = 2^(j-1)*dx:ds:N+2^(j-1)*dx-1;
                            subgridy = 2^(j-1)*dy:ds:N+2^(j-1)*dy-1;
                            wcs(idw,:,:) = impsik(mod(subgridx,N)+1,mod(subgridy,N)+1);
                            idw = idw + 1;
                        end
                    end
                end
            end
        end
        imphi = ifft2(fftim .* fftphi);
        for dx = -Delta:Delta
            for dy = -Delta:Delta
                subgridx = 2^(j-1)*dx:ds:N+2^(j-1)*dx-1;
                subgridy = 2^(j-1)*dy:ds:N+2^(j-1)*dy-1;
                wcs(idw,:,:) = imphi(mod(subgridx,N)+1,mod(subgridy,N)+1);
                idw = idw + 1;
            end
        end
        assert(idw == nbw + 1);
        wcoeffs(:,:,:,m) = wcs;
    end
end
