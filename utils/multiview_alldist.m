function [Dists] = multiview_alldist(x1, x2, nViews)
if size(x1,1) ~= size(x2,1)
    error('feature dimension not match');
end
nShapes1 = size(x1,2) / nViews;
nShapes2 = size(x2,2) / nViews;
Dists = zeros(nShapes1, nShapes2);
%fprintf('Computing distances between shapes ...\n');
if isequal(x1,x2)
    for j=1:nShapes1,
        %fprintf('Phase:%d/%d\n', j, nShapes1);
        Dists(j,j) = 0;
        for k=(j+1):nShapes2,
            Dists(j,k) = multiview_dist(x1(:,(j-1)*nViews+1:j*nViews), x2(:,(k-1)*nViews+1:k*nViews));
            Dists(k,j) = Dists(j,k);
        end
    end
else
    for j=1:nShapes1,
        %fprintf('Phase:%d/%d\n', j, nShapes1);
        for k=1:nShapes2,
            Dists(j,k) = multiview_dist(x1(:,(j-1)*nViews+1:j*nViews), x2(:,(k-1)*nViews+1:k*nViews));
        end
    end
end

function [Dist] = multiview_dist(x1, x2)
if isequal(x1, x2)
    Dist = 0;
    return;
end
if size(x1,1) ~= size(x2,1)
    error('feature dimension not match');
end
nViews1 = size(x1, 2);
nViews2 = size(x2, 2);
Dv = sqrt(vl_alldist2(x1,x2));
%{
Dv = zeros(nViews1, nViews2);
for i=1:nViews1,
    for j=1:nViews2,
        f1 = x1(i,:);
        f2 = x2(j,:);
        Dv(i,j) = norm(f1-f2);
    end
end
%}
jmean = 0;
kmean = 0;
for i=1:nViews1, jmean = jmean + min(Dv(i,:)); end
for j=1:nViews2, kmean = kmean + min(Dv(:,j)); end
Dist = jmean / nViews1 / 2 + kmean / nViews2 / 2;