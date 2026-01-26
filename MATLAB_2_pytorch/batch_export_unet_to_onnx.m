function batch_export_unet_to_onnx(models_root, output_root)
% Export all DDSeg UnetModel.mat files to ONNX.
% Example:
% batch_export_unet_to_onnx('/path/to/DDSeg-master/CNN-models', '/path/to/weights');

if nargin < 2
    error('Usage: batch_export_unet_to_onnx(models_root, output_root)');
end

model_sets = {'Unet-DTI-n5-ATloss-GMWMCSF', 'Unet-MKCurve-n10-ATloss-GMWMCSF'};
views = {'axial', 'coronal', 'sagittal'};

if ~exist(output_root, 'dir')
    mkdir(output_root);
end

for s = 1:numel(model_sets)
    for v = 1:numel(views)
        model_dir = fullfile(models_root, model_sets{s}, views{v});
        model_mat = fullfile(model_dir, 'UnetModel.mat');
        if ~exist(model_mat, 'file')
            warning('Missing: %s', model_mat);
            continue;
        end

        if strcmp(model_sets{s}, 'Unet-DTI-n5-ATloss-GMWMCSF')
            input_channels = 5;
            prefix = 'dti';
        else
            input_channels = 10;
            prefix = 'mkcurve';
        end

        out_name = sprintf('%s_%s.onnx', prefix, views{v});
        out_path = fullfile(output_root, out_name);

        export_unet_to_onnx(model_mat, out_path, 144, 144, input_channels);
    end
end
end
