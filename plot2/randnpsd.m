function X=randnpsd(powerspectral)

sqrtpsd=sqrt(powerspectral);
M = size(powerspectral,1);
N = size(powerspectral,2);
assert(mod(M,2)==0 && mod(N,2)==0)

V = zeros(M,N);

V(1,1)=sqrtpsd(1,1)*randn();
V(1,N/2+1)=sqrtpsd(1,N/2+1)*randn();
V(M/2+1,1)=sqrtpsd(M/2+1,1)*randn();
V(M/2+1,N/2+1)=sqrtpsd(M/2+1,N/2+1)*randn();

V(1,[2:N/2])=sqrtpsd(1,[2:N/2]).*(randn(1,N/2-1)+1i*randn(1,N/2-1))/sqrt(2);
V(1,N+2-[2:N/2])=conj(V(1,[2:N/2]));
V(M/2+1,[2:N/2])=sqrtpsd(M/2+1,[2:N/2]).*(randn(1,N/2-1)+1i*randn(1,N/2-1))/sqrt(2);
V(M/2+1,N+2-[2:N/2])=conj(V(M/2+1,[2:N/2]));

V([2:M/2],1)=sqrtpsd([2:M/2],1).*(randn(M/2-1,1)+1i*randn(M/2-1,1))/sqrt(2);
V(M+2-[2:M/2],1)=conj(V([2:M/2],1));
V([2:M/2],N/2+1)=sqrtpsd([2:M/2],N/2+1).*(randn(M/2-1,1)+1i*randn(M/2-1,1))/sqrt(2);
V(M+2-[2:M/2],N/2+1)=conj(V([2:M/2],N/2+1));

V(2:M/2,2:N/2)=sqrtpsd(2:M/2,2:N/2).*(randn(M/2-1,N/2-1)+1i*randn(M/2-1,N/2-1))/sqrt(2);
V(M+2-[2:M/2],N+2-[2:N/2])=conj(V(2:M/2,2:N/2));
V(2:M/2,N+2-[2:N/2])=sqrtpsd(2:M/2,N+2-[2:N/2]).*(randn(M/2-1,N/2-1)+1i*randn(M/2-1,N/2-1))/sqrt(2);
V(M+2-[2:M/2],[2:N/2])=conj(V(2:M/2,N+2-[2:N/2]));

X = ifft2(V)*sqrt(M*N);

end

