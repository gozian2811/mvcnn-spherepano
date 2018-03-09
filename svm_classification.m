function [acc] = svm_classification(mvfeat_name, imdb_name, varargin)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
opts.outPath = 'data/ModelNet40v1_relu7_svm_models';
opts.lbegin = 1;
opts.lend = 1000;
%opts.volume = 20;
[opts,~] = vl_argparse(opts, varargin);

if ~exist(opts.outPath, 'dir'), mkdir(opts.outPath); end

mvfeat = load(mvfeat_name);
if isfield(mvfeat, 'feat'), mvfeat = mvfeat.feat;
elseif isfield(mvfeat, 'mvfeat') mvfeat = mvfeat.mvfeat;
elseif isfield(mvfeat, 'prob') mvfeat = mvfeat.prob;
else
    error('Feature field error');
end

imdb = load(imdb_name);
trainid = find(strcmp(imdb.meta.sets, 'train'));
valid = find(strcmp(imdb.meta.sets, 'val'));
testid = find(strcmp(imdb.meta.sets, 'test'));

nViews = length(imdb.images.id) / length(unique(imdb.images.sid));
nShapeFeat = length(imdb.images.id) / size(mvfeat,1);
%trainfeat = mvfeat(imdb.images.set==trainid|imdb.images.set==valid,:);
%trainclass = imdb.images.class(imdb.images.set==trainid|imdb.images.set==valid)';
trainfeat = mvfeat(imdb.images.set(1:nShapeFeat:end)==trainid|imdb.images.set(1:nShapeFeat:end)==valid,:);
trainclass = imdb.images.class(imdb.images.set==trainid|imdb.images.set==valid)';
trainclass = trainclass(1:nShapeFeat:end);
testfeat = mvfeat(imdb.images.set(1:nShapeFeat:end)==testid,:);
testclass = imdb.images.class(imdb.images.set==testid)';
testclass = testclass(1:nShapeFeat:end);

nTest =  size(testfeat,1);
if nShapeFeat==1, nTest = nTest / nViews; end
nLabels = length(imdb.meta.classes);
models = cell(1, nLabels);

fprintf('Training ...\n');

for i=max(opts.lbegin,1):min(opts.lend,nLabels)
    filename = sprintf('%s/model-%d.mat',opts.outPath, i);
    if exist(filename, 'file')
        models{i} = load(filename);
        fprintf('Phase %d/%d loaded!\n', i, nLabels);
    else
        models{i} = svmtrain(double(trainclass==i), trainfeat, '-h 0 -b 1');
        model = models{i};
        save(sprintf('%s/model-%d.mat',opts.outPath, i), '-struct', 'model');
        fprintf('Phase %d/%d complete!\n', i, nLabels);
    end
end
%{
volume = opts.volume;
for i=1:nLabels
    filename = sprintf('%s/model-%d.mat',opts.outPath, i);
    if exist(filename, 'file')
        models{i} = load(filename);
        fprintf('Phase %d/%d loaded!\n', i, nLabels);
    else
        lbegin = i;
        break;
    end
end
if exist('lbegin', 'var')
    parnum = ceil((nLabels-lbegin+1)/volume);
    parclasses = cell(1,parnum);
    parfeats = cell(1,parnum);
    for i=1:parnum,
        parclasses{i} = trainclass;
        parfeats{i} = trainfeat;
    end
    for v=1:volume,
        parmodels = cell(1,parnum);
        parfor_progress(parnum);
        parfor i=1:parnum,
            lindex = lbegin-1+(i-1)*volume+v;
            if lindex<=nLabels
                parmodels{i} = svmtrain(double(parclasses{i}==lindex), parfeats{i}, '-h 0 -b 1');
            end
        end
        parfor_progress();
        for i=1:parnum,
            lindex = lbegin-1+(i-1)*volume+v;
            if lindex<=nLabels
                models{lindex} = parmodels{i};
            end
        end
    end
    parfor_progress(0);
    for i=lbegin:nLabels,
        model = models{i};
        save(sprintf('%s/model-%d.mat',opts.outPath, i), '-struct', 'model');
    end
end
%}
fprintf('Training complete!\n');

mvprobs = zeros(size(testfeat,1), nLabels);
probs = zeros(nTest, nLabels);
fprintf('Predicting ...\n');
for i=1:nLabels
    [~, ~, prob_estimates] = svmpredict(double(testclass==i), testfeat, models{i}, '-b 1');
    mvprobs(:,i) = prob_estimates(:,models{i}.Label==1);
    fprintf('Phase_%d/%d complete!\n', i, nLabels);
end

if nShapeFeat>1,
    probs = mvprobs;
else
    for i=1:nTest
        probs(i,:) = sum(mvprobs((i-1)*nViews+1:i*nViews,:),1);
    end
    testclass = testclass(1:nViews:end);
end
fprintf('Prediction complete!\n');

[~,pred] = max(probs,[],2);
inst_acc = sum(pred == testclass) / nTest;
%cmat = confusionmat(testclass, pred)
fprintf('instance accuracy:%f%%\n', inst_acc*100);

cls_acc = 0;
for i = 1:nLabels
    corrnum = 0;
    totalnum = 0;
    for j = 1:nTest
        if testclass(j)==i
            totalnum = totalnum + 1;
            if pred(j)==i
                corrnum = corrnum + 1;
            end
        end
    end
    cls_acc = cls_acc + corrnum / totalnum;
end
cls_acc = cls_acc / nLabels;
fprintf('class accuracy:%f%%\n', cls_acc*100);

save(sprintf('%s/accuracy.txt',opts.outPath), 'inst_acc', 'cls_acc', '-ascii');

%[svm_struct, svIndex] = svmtrain(mvfeat(imdb.images.set==trainid|imdb.images.set==valid,:), imdb.images.class(imdb.images.set==trainid|imdb.images.set==valid)');
%outclass = svmclassify(svm_struct, mvfeat(imdb.images.set==testid,:));
%result = 1 - length(find(outclass - imdb.images.class(imdb.images.set==3)')) / length(outclass);
%fprintf('classification precision:%d%%\n', result*100);
end