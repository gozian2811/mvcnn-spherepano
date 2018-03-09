function model_renders( path_to_shape )
% SHAPE_COMPUTE_DESCRIPTOR Compute and save the CNN descriptor for a single
% input obj/off shape, or for each shape in an input folder
% Quick examples of use:
% shape_compute_descriptor('bunny.off');
% shape_compute_descriptor('my_bunnies_folder/');
%
%   path_to_shape:: (default) 'data/'
%        can be either a filename for a mesh in OBJ/OFF format
%        or a name of folder containing multiple OBJ/OFF meshes
%   `cnn_model`:: (default) 'cnn-modelnet40-v1.mat'
%       this is a matlab file with the saved CNN parameters
%       if the default file is not found, it will be downloaded from our
%       server.
%       Note: The default mat file assumes that shapes that are
%       upright oriented according to +Z axis!
%       if you want to use the CNN trained *without upright assumption*, use
%       'cnn-modelnet40-v2.mat'
%   `post_process_desriptor_metric`:: (default) true
%       set to false to disable transforming descriptor based on our
%       learned distance metric
%   `metric_model`:: (default:) 'metric-relu7-v1.mat'
%       this is a matlab file with the saved metric parameters
%       if the default file is not found, it will be downloaded from our
%       server
%       if you want to use the model trained *without upright assumption*, use
%       'metric-relu7-v2.mat'
%   `gpus`:: (default) []
%       set to use GPU

setup;

if nargin<1 || isempty(path_to_shape),
    path_to_shape = 'data/ModelNet40';
end

% work on mesh (or meshes)
mesh_filenames  = [dir( strcat(path_to_shape, '/*.off' ) ); dir( strcat(path_to_shape, '/*.obj') )];
for i=1:length(mesh_filenames)
    mesh_filenames(i).name = [path_to_shape '/' mesh_filenames(i).name];
end
if isempty(mesh_filenames)
    error('No obj/off meshes found in the specified folder!');
end

descr = cell( 1, length(mesh_filenames));
fig = figure('Visible','off');
for i=1:length(mesh_filenames)
    fprintf('Loading input shape %s...', mesh_filenames(i).name);
    mesh = loadMesh( mesh_filenames(i).name );
    if isempty(mesh.F)
        error('Could not load mesh from file');
    else
        fprintf('Done.\n');
    end
    if num_views == 12
        ims = render_views(mesh, 'figHandle', fig);
    else
        ims = render_views(mesh, 'use_dodecahedron_views', true, 'figHandle', fig);
    end
    
    save( sprintf('%s_descriptor.txt', mesh_filenames(i).name(1:end-4)), 'out', '-ascii');
end
close(fig);
