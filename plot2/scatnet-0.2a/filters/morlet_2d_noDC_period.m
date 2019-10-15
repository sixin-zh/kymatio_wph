% MORLET_2D_NODC computes the 2-D elliptic Morlet filter given a set of 
%    parameters in Fourier domain
%
% Usage
%    gab = MORLET_2D_NODC(N, M, sigma, slant, xi, theta, offset)
%
% Input
%    N (numeric): Width of the filter.
%    M (numeric): Height of the filter.
%    sigma (numeric): Standard deviation of the envelope.
%    slant (numeric): Eccentricity of the elliptic envelope.
%       (the smaller slant, the larger angular resolution).
%    xi (numeric): The frequency peak.
%    theta (numeric): Orientation in radians of the filter.
%    offset (numeric, optional): 2-D vector reprensting the offset location 
%       (default [0 0]).
% 
% Output
%    gab (numeric): N-by-M matrix representing the gabor filter in spatial
%       domain.
%
% Description
%    Compute a Morlet wavelet in Fourier domain. 
%
%    Morlet wavelets have a 0 DC component.
%
% See also
%    GABOR_2D, MORLET_2D_PYRAMID

function gab = morlet_2d_noDC_period(N, M, sigma, slant, xi, theta, offset, extent)
	
	if ~exist('offset','var')
		offset = [0, 0];
	end
	[x0 , y0] = meshgrid(1:M, 1:N);
	oscilating_part = zeros(size(x0));
	gaussian_part = zeros(size(x0));
	for ex=-extent:extent
        for ey=-extent:extent
            x = x0 - ceil(M/2) - 1 - offset(1) + ex*M;
            y = y0 - ceil(N/2) - 1 - offset(2) + ey*N;

            Rth = rotation_matrix_2d(theta);
            A = Rth\ [1/sigma^2, 0 ; 0 slant^2/sigma^2] * Rth ;
            s = x.* ( A(1,1)*x + A(1,2)*y) + y.*(A(2,1)*x + A(2,2)*y ) ;

            %normalize such that the maximum of fourier modulus is 1
            gaussian_envelope = exp( - s/2);
            oscilating_part = oscilating_part + gaussian_envelope .* exp(1i*(x*xi*cos(theta) + y*xi*sin(theta)));
            gaussian_part = gaussian_part + gaussian_envelope;
        end
	end
% 	gaussian_envelope = abs(oscilating_part);
	K = sum(oscilating_part(:)) ./ sum(gaussian_part(:));
	gabc = oscilating_part - K.*gaussian_part;
	
    gab=1/(2*pi*sigma^2/slant)*fftshift(gabc);
end
