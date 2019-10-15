function Sjq = compute_Sjq(imgs,taux,tauy,Q)
siz = size(imgs);
N = siz(1);
assert(siz(1)==siz(2))
M = siz(3);

S = zeros(length(taux),Q);
for tid = 1:length(taux)
    for m=1:M
        img = imgs(:,:,m);
        img_ = circshift(img,[taux(tid),tauy(tid)]);
%         if m==1
%             figure(1)
%             subplot(121)
%             imagesc(img)
%             subplot(122)
%             imagesc(img_)
%         end
        diff = abs(img-img_);
        for q=1:Q
            S(tid,q) = S(tid,q) + mean(mean(diff.^q))/M;
        end
    end
end
Sjq = max(S,[],1);

end