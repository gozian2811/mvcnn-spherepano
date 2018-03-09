function [results] = run_classification(prob_name, imdb_name)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

imdb = load(imdb_name);
prob = load(prob_name);
if ~isfield(prob, 'prob')
    error('Feature vector unexist!');
end
prob = prob.prob;

nFeatures = size(prob,1);
nViews = length(imdb.images.id) / length(unique(imdb.images.sid));
nClass = size(prob,2);
class = imdb.images.class(1:nViews:end);
for s = imdb.meta.sets
    fprintf('\n%s classification\n', s{1});
    for c = 1:length(imdb.meta.classes)
        classes = zeros(1, length(imdb.meta.classes));
        cshapes = find(arrayfun(@(cls) cls==c, class));
        for i = 1:length(cshapes)
            if strcmp(imdb.meta.sets{imdb.images.set(cshapes(i)*nViews-1)}, s{1})
                [score, pred] = max(prob(cshapes(i),:));
                classes(pred) = classes(pred)+1;
            end
        end
        fprintf('class_%d:', c);
        for i = 1:length(imdb.meta.classes)
            fprintf('%d ', classes(i));
        end
        fprintf('\n');
    end
    
    errnum = 0;
    corrnum = 0;
    testnum = 0;
    for i = 1:nFeatures
        if strcmp(imdb.meta.sets{imdb.images.set(i*nViews-1)}, s{1})
            testnum = testnum + 1;
            [score, pred] = max(prob(i,:));
            if pred ~= class(i)
                errnum = errnum + 1;
            else
                corrnum = corrnum + 1;
            end
        end
    end
    inst_acc = corrnum / testnum;
    fprintf('instance accuracy:%f%%\n', inst_acc*100);
    
    cls_acc = 0;
    for i = 1:nClass
        corrnum = 0;
        totalnum = 0;
        for j = 1:nFeatures
            if strcmp(imdb.meta.sets{imdb.images.set(j*nViews-1)}, s{1})
                if class(j)==i
                    totalnum = totalnum + 1;
                    [score, pred] = max(prob(j,:));
                    if pred==i
                        corrnum = corrnum + 1;
                    end
                end
            end
        end
        cls_acc = cls_acc + corrnum / totalnum;
        %{
    corrnum1 = 0;
    corrnum2 = 0;
    for j = 1:nFeatures
        if strcmp(imdb.meta.sets{imdb.images.set(j*nViews-1)}, 'test')
            [score, pred] = max(prob(j,:));
            if i==class(j) & i==pred
                corrnum1 = corrnum1 + 1;
            elseif i~=class(j) & i~=pred
                corrnum2 = corrnum2 + 1;
            end
        end
    end
    cls_acc = cls_acc + (corrnum1 + corrnum2) / testnum;
        %}
    end
    cls_acc = cls_acc / nClass;
    fprintf('class accuracy %f%%\n', cls_acc*100);
end