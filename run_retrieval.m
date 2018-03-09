function [results] = run_retrieval(feat, imdb, varargin)
% e.g.:  r = run_retrieval('prob.mat','shapenet55v2','savePath','prob', ...
%                          'distFn',@(x1,x2) par_alldist(x1,[],'numWorkers',36,'maxParts',500));


opts.distFn = @(x1,x2,nViews) par_alldist(x1,x2,'numWorkers',4,'maxParts',100,'numViews',nViews);
opts.topK = 1000;
opts.sets = {'train', 'val', 'test'}; 
opts.savePath = []; 
opts.saveDist = true; 
opts.resultType = 'fixedLength'; % 'fixedLength' | 'sameClass'
opts.multiView = false;
opts = vl_argparse(opts, varargin); 

if ischar(feat), 
  feat = load(feat);
  if isfield(feat, 'feat')
    feat = feat.feat;
  elseif isfield(feat, 'mvfeat')
    feat = feat.mvfeat;
  elseif isfield(feat, 'prob')
    feat = feat.prob;
  else
    error('Struct feature not exist!');
  end
end

if ischar(imdb), 
  imdb = get_imdb(imdb);
end

nViews = numel(imdb.images.id) / numel(unique(imdb.images.sid));

results = cell(2,numel(opts.sets)); 
TotalMAP = zeros(1, numel(opts.sets));
for i = 1:numel(opts.sets), 
  fprintf('%s set retrieval\n', opts.sets{i});
  setId = find(cellfun(@(s) strcmp(opts.sets{i},s),imdb.meta.sets));
  
  sid = imdb.images.sid(imdb.images.set==setId);
  sid = sid(1:nViews:end);
  class = imdb.images.class(imdb.images.set==setId);
  class = class(1:nViews:end);
 
  if ~opts.multiView
      f = feat(imdb.images.set(1:nViews:end)==setId,:);
      nShapes = size(f,1);
      D = opts.distFn(f',f',1);
  else
      f = feat(imdb.images.set==setId,:);
      nShapes = size(f,1) / nViews;
      D = multiview_alldist(f',f',nViews);
      %D = opts.distFn(f',f',nViews);
  end
  if strcmpi(opts.resultType,'sameClass'), 
    [~,I] = max(f,[],2);
    sameLabelMask = arrayfun(@(l) (I'==l), I,'UniformOutput', false);
    results{1,i} = cellfun(@(c) sid(c), sameLabelMask, 'UniformOutput', false);
    results{2,i} = cell(nShapes,1);
    for j=1:nShapes, 
      [results{2,i}{j},I] = sort(D(j,sameLabelMask{j}),'ascend');
      topK = min(opts.topK, numel(I));
      results{2,i}{j} = results{2,i}{j}(1:topK);
      results{1,i}{j} = results{1,i}{j}(I(1:topK));
    end
  elseif strcmpi(opts.resultType,'fixedLength')
    [Y,I] = sort(D,2,'ascend');
    topK = min(opts.topK, numel(unique(sid)));
    index_mat = I(:,1:topK); 
    dist_mat = Y(:,1:topK);
    result_mat = sid(index_mat); 
    results{1,i} = cell(nShapes, 1);
    results{2,i} = cell(nShapes, 1); 	
    for j=1:nShapes, 
      results{1,i}{j} = result_mat(j,:);
      results{2,i}{j} = dist_mat(j,:); 
      map = 0;
      retrel = 0;
      for k=1:topK,
        if class(I(j, k))==class(j)
            retrel = retrel+1;
            map = map+retrel/k;
        end
      end
      map = map/retrel;
      %fprintf('shape_%d retrieval mAP:%02f%%\n', j, map*100);
      TotalMAP(i) = TotalMAP(i)+map;
    end
    TotalMAP(i) = TotalMAP(i)/nShapes;
    fprintf('%s retrieval mAP:%02f%%\n', opts.sets{i}, TotalMAP(i)*100);
  else
    error('Unknown option: %s', opts.resultType);
  end
  
  % write to file 
  if ~isempty(opts.savePath), 
    fprintf('Saving retrieval results to %s ...', fullfile(opts.savePath,opts.sets{i}));
    vl_xmkdir(fullfile(opts.savePath,opts.sets{i})); 
    for k=1:numel(sid), 
      fid = fopen(fullfile(opts.savePath,opts.sets{i},sprintf('%06d',sid(k))),'w+');
      r = results{1,i}{k};
      if opts.saveDist, 
        r = [r ; results{2,i}{k}]; 
        fprintf(fid,'%06d %f\n',r);
      else
        fprintf(fid,'%06d\n',r);
      end
      fclose(fid);
    end
    fprintf(' done!\n'); 
  end
end