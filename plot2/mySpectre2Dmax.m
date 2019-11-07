function [S,K]=mySpectre2Dmax(E)

siz=size(E);
[y,x] = meshgrid(1:siz(2),1:siz(1));
x=x-(siz(1)/2+1);
y=y-(siz(2)/2+1);
modx=fftshift(sqrt(x.^2 + y.^2));

K=0:(min(siz(1),siz(2))/2);
S=0.*K;
V=0.*K;

for i=K
    mask = ((modx>=i)&(modx<(i+1)));
    Emask = E.*mask;
    S(i+1) = max(max(Emask));
end

end