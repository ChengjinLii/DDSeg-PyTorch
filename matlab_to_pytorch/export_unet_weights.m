function export_unet_weights(model_mat_path, output_dir)
% Export MATLAB DAGNetwork structure + weights for PyTorch rebuild.
% Usage: export_unet_weights('UnetModel.mat', 'output_dir')

if nargin < 2
    error('Usage: export_unet_weights(model_mat_path, output_dir)');
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

s = load(model_mat_path);
if ~isfield(s, 'net1')
    error('net1 not found in %s', model_mat_path);
end
net = s.net1;

layers = net.Layers;
conns = net.Connections;

layer_info = struct();
layer_names = cell(numel(layers), 1);
for i = 1:numel(layers)
    L = layers(i);
    name = L.Name;
    layer_names{i} = name;
    info = struct();
    info.class = class(L);

    % Capture common properties
    props = properties(L);
    for p = 1:numel(props)
        key = props{p};
        try
            val = L.(key);
            if isnumeric(val) || islogical(val) || ischar(val) || isstring(val)
                info.(key) = val;
            end
        catch
        end
    end

    % Extract weights if present
    if isprop(L, 'Weights')
        w = L.Weights;
        save(fullfile(output_dir, [name '_Weights.mat']), 'w');
        info.HasWeights = true;
        info.WeightsFile = [name '_Weights.mat'];
        info.WeightsSize = size(w);
    else
        info.HasWeights = false;
    end

    if isprop(L, 'Bias')
        b = L.Bias;
        save(fullfile(output_dir, [name '_Bias.mat']), 'b');
        info.HasBias = true;
        info.BiasFile = [name '_Bias.mat'];
        info.BiasSize = size(b);
    else
        info.HasBias = false;
    end

    % BatchNorm params
    if isprop(L, 'TrainedMean')
        tm = L.TrainedMean; save(fullfile(output_dir, [name '_TrainedMean.mat']), 'tm');
        info.TrainedMeanFile = [name '_TrainedMean.mat'];
    end
    if isprop(L, 'TrainedVariance')
        tv = L.TrainedVariance; save(fullfile(output_dir, [name '_TrainedVariance.mat']), 'tv');
        info.TrainedVarianceFile = [name '_TrainedVariance.mat'];
    end
    if isprop(L, 'Scale')
        sc = L.Scale; save(fullfile(output_dir, [name '_Scale.mat']), 'sc');
        info.ScaleFile = [name '_Scale.mat'];
    end
    if isprop(L, 'Offset')
        off = L.Offset; save(fullfile(output_dir, [name '_Offset.mat']), 'off');
        info.OffsetFile = [name '_Offset.mat'];
    end

    layer_info.(matlab.lang.makeValidName(name)) = info;
end

meta.layers = layer_info;
meta.layer_order = layer_names;
meta.connections = conns;

save(fullfile(output_dir, 'network_meta.mat'), 'meta');

% Also write JSON summary for easier parsing in Python
json_text = jsonencode(meta);
json_path = fullfile(output_dir, 'network_meta.json');
fid = fopen(json_path, 'w');
fprintf(fid, '%s', json_text);
fclose(fid);

fprintf('Exported meta and weights to %s\n', output_dir);
end
