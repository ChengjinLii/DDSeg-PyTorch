# MATLAB to PyTorch Conversion Workspace

This folder contains the scripts needed to export MATLAB DAGNetwork weights and convert them to TorchScript for DDSeg.

## Why MATLAB is required for export

The `UnetModel.mat` files in DDSeg are MATLAB `DAGNetwork` objects, not plain numeric arrays. Python cannot directly deserialize these objects. The reliable route is:

1. Export weights from MATLAB to `.mat` files.
2. Convert the exported weights to TorchScript (`.pt`).

## Export weights in MATLAB

Use `export_unet_weights.m` for each model:

```
export_unet_weights('UnetModel.mat', 'output_dir')
```

This produces:
- `*_Weights.mat`
- `*_Bias.mat`
- `network_meta.json`

## Convert to TorchScript (Python)

Run the converter:

```
python matlab_to_pytorch/convert_matlab_weights_to_torchscript.py \
  --weights_export /path/to/weights_export \
  --out_dir /path/to/weights \
  --prefix dti
```

This writes:
- `weights/dti_axial.pt`
- `weights/dti_sagittal.pt`
- `weights/dti_coronal.pt`

## Notes

- The converter rebuilds the DDSeg U-Net in PyTorch and loads MATLAB weights with the correct layout.
- For strict parity with MATLAB output, you can also use the `--matlab_prediction_dir` option in `scripts/run_ddseg.py`.
