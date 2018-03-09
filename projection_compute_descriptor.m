function descr = projection_compute_descriptor( path_to_shape, varargin )
% SHAPE_COMPUTE_DESCRIPTOR Compute and save the CNN descriptor for a single
% input obj/off shape, or for each shape in an input folder
% Quick examples of use:
% shape_compute_descriptor('bunny.off');
% shape_compute_descriptor('my_bunnies_folder/');
%
%   path_to_shape:: (default) 'data/'
%        can be either a filename for a mesh in OBJ/OFF format
%        or a name of folder containing multiple OBJ/OFF meshes
%   `cnnModel`:: (default) ''
%       this is a matlab file with the saved CNN parameters
%       if the default file is not found, it will be downloaded from our
%       server.
%       Note: The default mat file assumes that shapes that are
%       upright oriented according to +Z axis!
%       if you want to use the CNN trained *without upright assumption*, use
%       'cnn-vggm-relu7-v2'
%   `applyMetric`:: (default) false
%       set to true to disable transforming descriptor based on specified
%       distance metric
%   `metricModel`:: (default:) ''
%       this is a matlab file with the saved metric parameters
%       if the default file is not found, it will attempt to download from our
%       server
%   `gpus`:: (default) []
%       set to use GPU

setup;

if nargin<1 || isempty(path_to_shape),
    path_to_shape = 'data/';
end

% default options
opts.features = {'relu6', 'viewpool', 'relu7', 'prob'};
opts.useUprightAssumption = true; 
opts.applyMetric = false;
opts.viewPool = true;

opts.gpus = [];
[opts, varargin] = vl_argparse(opts,varargin);

opts.metricModel = '';
opts.cnnModel = 'net-deployed';
opts.outputPath = fullfile(path_to_shape, 'Descriptors');
opts = vl_argparse(opts,varargin);

% locate network file
default_viewpool_loc = 'relu7'; 
vl_xmkdir(fullfile('data','models')) ;
%{
if isempty(opts.cnnModel), 
    if opts.useUprightAssumption, 
        opts.cnnModel = sprintf('cnn-vggm-%s-v1',default_viewpool_loc);
        nViews = 12;
    else
        opts.cnnModel = sprintf('cnn-vggm-%s-v2',default_viewpool_loc);
        nViews = 80;
    end
    baseModel = 'imagenet-matconvnet-vgg-m';
    cnn = cnn_shape_init({},'base',baseModel,'viewpoolPos',default_viewpool_loc,'nViews',nViews);
    netFilePath = fullfile('data','models',[opts.cnnModel '.mat']);
    save(netFilePath,'-struct','cnn');
end
%}

if ~ischar(opts.cnnModel), 
    cnn = opts.cnnModel;
else
    if ~strcmp(opts.cnnModel(end-3:end), '.mat')
        netFilePath = fullfile('data','models',[opts.cnnModel '.mat']);
    else
        netFilePath = opts.cnnModel;
    end
    if ~exist(netFilePath,'file'),
        fprintf('Downloading model (%s) ...', opts.cnnModel) ;
        urlwrite(fullfile('http://maxwell.cs.umass.edu/mvcnn/models/', ...
            [opts.cnnModel '.mat']), netFilePath) ;
        fprintf(' done!\n');
    end
    cnn = load(netFilePath);
end

% locate metric file
if isempty(opts.metricModel) && opts.applyMetric, 
    opts.applyMetric = false; 
    warning('No metric file specified. Post-processing turned off');
end
if opts.applyMetric
    metricFilePath = fullfile('data','models',[opts.metricModel '.mat']);
    if ~exist(metricFilePath,'file'),
        fprintf('Downloading model (%s) ...', opts.metricModel) ;
        urlwrite(fullfile('http://maxwell.cs.umass.edu/mvcnn/models/', ...
            [opts.metricModel '.mat']), metricFilePath) ;
        fprintf(' done!\n');
    end
    modelDimRedFV = load(metricFilePath);
end

% see if it's a multivew net
if opts.viewPool
    viewpoolIdx = find(cellfun(@(x)strcmp(x.name, 'viewpool'), cnn.layers));
    if ~isempty(viewpoolIdx),
        if numel(viewpoolIdx)>1,
            error('More than one viewpool layers found!');
        end
        if ~isfield(cnn.layers{viewpoolIdx},'vstride'),
            num_views = cnn.layers{viewpoolIdx}.stride; % old format
        else
            num_views = cnn.layers{viewpoolIdx}.vstride;
        end
        fprintf('CNN model is based on %d views. Will process %d views per mesh.\n', num_views, num_views);
    else
        error('Computing a descriptor per shape requires a multi-view CNN.');
    end
else
    opts.features = setdiff(opts.features, {'viewpool'});
    existingViewpoolLayer = find(cellfun(@(l) strcmp(l.name, 'viewpool'), cnn.layers));
    if ~isempty(existingViewpoolLayer),
        cnn = modify_net(cnn, existingViewpoolLayer, 'mode', 'rm_layer', 'loc', 'viewpool');
    end
end

% work on mesh (or meshes)

imdb_name  = fullfile(path_to_shape, 'imdb.mat' );
if isempty(imdb_name)
    error('No imdb found in the specified folder!');
end

fprintf('Loading images\n');

imdb = load(imdb_name);
nShapes = length(unique(imdb.images.sid));
nViews = length(imdb.images.id)/nShapes;
descr = struct;
descr.class = cell(1, nShapes);
for feature = opts.features
    descr.(feature{1}) = cell(1, nShapes);
end
for i=1:nShapes
    fprintf('load shape %d/%d\n', i, nShapes);
    ims = cell(1, nViews);
    batch = find(arrayfun(@(isid) isid==i, imdb.images.sid));
    descr.class{i} = imdb.meta.classes{imdb.images.class(batch(1))};
    if length(batch) ~= nViews
        error('Wrong number of Views\n');
    end
    for k=1:length(batch)
        ims(k) = vl_imreadjpeg(fullfile(imdb.imageDir, imdb.images.name(batch(k))));
    end
    if isempty(ims)
        error('Could not load images');
    else
        fprintf('done!\n');
    end
    outs = cnn_shape_get_features(ims, cnn, opts.features, 'gpus', opts.gpus);
    
    for feature = opts.features
        out = outs.(feature{1});
        if opts.applyMetric
            out = single(modelDimRedFV.W*out);
        end
        
        descr.(feature{1}){i} = out;
        out = double(out);
        
        %{
        pathname = fullfile(path_to_shape, strcat('Descriptor_', feature{1}));
        [subpath, imagename, ext] = fileparts(imdb.images.name{(i-1)*nViews+1});
        splitname = strsplit(imagename, '_');
        shapename = strjoin(splitname(1:end-1), '_');
        fullpathname = fullfile(pathname, subpath);
        if ~isdir(fullpathname)
            mkdir(fullpathname);
        end
        savename = fullfile(fullpathname, sprintf('%s_%s.txt', shapename, feature{1}));
        save(savename, 'out', '-ascii');
        %}
    end
end

savepath = opts.outputPath;
if ~exist(savepath, 'dir'), mkdir(savepath); end
fprintf('Saving ...\n');

save(fullfile(savepath, 'descriptors.mat'), '-struct', 'descr', '-v7.3');
fprintf('descriptor saved!\n');

if isfield(descr, 'relu6'),
    descrLength = size(descr.relu6{1}, 3);
    feat = zeros(nShapes*size(descr.relu6{1}, 4), descrLength);
    for i = 1:nShapes,
        for j = 1:nViews,
            for k = 1:descrLength,
                feat((i-1)*nViews+j, k) = descr.relu6{i}(:,:,k,j);
            end
        end
        priors = zeros(1,nViews);
        for k = 1:descrLength,
            [~, pred] = max(descr.relu6{i}(:,:,k,:));
            priors(pred) = priors(pred) + 1;
        end
        save(fullfile(path_to_shape, 'view_priors.txt'), 'priors', '-ascii');
    end
    if size(feat,1)>20000, save(fullfile(savepath, 'rl6feat.mat'), 'feat', '-v7.3');
    else save(fullfile(savepath, 'rl6feat.mat'), 'feat');
    end
    
    fprintf('relu6 feat saved!\n');
end
if isfield(descr, 'viewpool'),
    descrLength = length(descr.viewpool{1});
    feat = zeros(nShapes, descrLength);
    for i = 1:nShapes,
        for j = 1:length(descr.viewpool{i}),
            feat(i,j) = descr.viewpool{i}(j);
        end
    end
    if size(feat,1)>20000, save(fullfile(savepath, 'vpfeat.mat'), 'feat', '-v7.3');
    else save(fullfile(savepath, 'vpfeat.mat'), 'feat');
    end
    
    fprintf('viewpool feat saved!\n');
end
if isfield(descr, 'relu7'),
    descrLength = size(descr.relu7{1}, 3);
    if opts.viewPool
        feat = zeros(nShapes, descrLength);
        for i = 1:nShapes,
            for j = 1:length(descr.relu7{1}),
                feat(i,j) = descr.relu7{i}(j);
            end
        end
    else
        feat = zeros(nShapes*size(descr.relu7{1}, 4), descrLength);
        for i = 1:nShapes,
            for j = 1:nViews
                for k = 1:descrLength,
                    feat((i-1)*nViews+j, k) = descr.relu7{i}(:,:,k,j);
                end
            end
        end
    end
    if size(feat,1)>20000, save(fullfile(savepath, 'rl7feat.mat'), 'feat', '-v7.3');
    else save(fullfile(savepath, 'rl7feat.mat'), 'feat');
    end
    fprintf('relu7 feat saved!\n');
end
if isfield(descr, 'prob'),
    descrLength = size(descr.prob{1}, 3);
    if opts.viewPool
        prob = zeros(nShapes, descrLength);
        for i = 1:nShapes,
            for j = 1:length(descr.prob{1}),
                prob(i,j) = descr.prob{i}(j);
            end
        end
    else
        prob = zeros(nShapes*size(descr.prob{1}, 4), descrLength);
        for i = 1:nShapes,
            for j = 1:nViews
                for k = 1:descrLength,
                    prob((i-1)*nViews+j, k) = descr.prob{i}(:,:,k,j);
                end
            end
        end
    end
    save(fullfile(savepath, 'prob.mat'), 'prob');
    fprintf('prob vector saved!\n');
end