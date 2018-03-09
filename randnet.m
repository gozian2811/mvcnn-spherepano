function [net] = randnet(net, rate)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for i=1:length(net.layers),
    if isfield(net.layers{i}, 'weights'),
        for j=1:length(net.layers{i}.weights),
            wsize = size(net.layers{i}.weights{j});
            sc = sqrt(2/(wsize(1)*wsize(2)*wsize(4))) ;
            net.layers{i}.weights{j} = rate*rand(wsize, 'single')*sc;
        end
    end
end