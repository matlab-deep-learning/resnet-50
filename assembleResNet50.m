function net = assembleResNet50()
% assembleResNet50   Assemble ResNet-50 network
%
% net = assembleResNet50 creates a ResNet-50 network with weights trained
% on ImageNet. You can load the same ResNet-50 network by installing the
% Deep Learning Toolbox Model for ResNet-50 Network support package from
% the Add-On Explorer and then using the resnet50 function.

%   Copyright 2019 The MathWorks, Inc.

% Download the network parameters. If these have already been downloaded,
% this step will be skipped.
% 
% The files will be downloaded to a file "resnet50Params.mat", in a
% directory "ResNet50" located in the system's temporary directory.
dataDir = fullfile(tempdir, "ResNet50");
paramFile = fullfile(dataDir, "resnet50Params.mat");
downloadUrl = "http://www.mathworks.com/supportfiles/nnet/data/networks/resnet50Params.mat";

if ~exist(dataDir, "dir")
    mkdir(dataDir);
end

if ~exist(paramFile, "file")
    disp("Downloading pretrained parameters file (94 MB).")
    disp("This may take several minutes...");
    websave(paramFile, downloadUrl);
    disp("Download finished.");
else
    disp("Skipping download, parameter file already exists.");
end

% Load the network parameters from the file resNet50Params.mat.
s = load(paramFile);
params = s.params;

% Create a layer graph with the network architecture of ResNet-50.
lgraph = resnet50Layers;

% Create a cell array containing the layer names.
layerNames = {lgraph.Layers(:).Name}';

% Loop over layers and add parameters.
for i = 1:numel(layerNames)
    name = layerNames{i};
    idx = strcmp(layerNames,name);
    layer = lgraph.Layers(idx);
    
    % Assign layer parameters.
    layerParams = params.(name);
    if ~isempty(layerParams)
        paramNames = fields(layerParams);
        for j = 1:numel(paramNames)
            layer.(paramNames{j}) = layerParams.(paramNames{j});
        end
    end
    
    % Add layer into layer graph.
    lgraph = replaceLayer(lgraph,name,layer);
end

% Assemble the network.
net = assembleNetwork(lgraph);

end