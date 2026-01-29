function export_unet_to_onnx(model_mat_path, output_onnx_path, input_height, input_width, input_channels)
% Export MATLAB DAGNetwork to ONNX
% Usage example:
% export_unet_to_onnx('UnetModel.mat', 'dti_axial.onnx', 144, 144, 5);

s = load(model_mat_path);
if ~isfield(s, 'net1')
    error('net1 not found in %s', model_mat_path);
end
net = s.net1;

input_name = net.Layers(1).Name;
input_size = [input_height, input_width, input_channels];

% Some MATLAB versions do not support InputDataFormats/OutputDataFormats.
try
    % Ensure support package bin is on PATH for onnxmex dependencies.
    try
        nnet.internal.cnn.onnx.util.addSpkgBinPath;
    catch
        % No-op if not available.
    end
    exportONNXNetwork(net, output_onnx_path, 'InputDataFormats', 'BSSC', 'OutputDataFormats', 'BSSC');
catch
    try
        nnet.internal.cnn.onnx.util.addSpkgBinPath;
    catch
    end
    exportONNXNetwork(net, output_onnx_path);
end

fprintf('Exported %s -> %s\n', model_mat_path, output_onnx_path);
end
