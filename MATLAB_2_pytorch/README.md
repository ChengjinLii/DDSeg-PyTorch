# MATLAB to PyTorch Conversion Workspace

This folder contains the conversion scripts and notes to export MATLAB `DAGNetwork` models into a PyTorch-friendly format.

## Why MATLAB is required for export

The `UnetModel.mat` files in DDSeg are MATLAB `DAGNetwork` objects, not plain numeric arrays. Python cannot directly deserialize these objects. The reliable route is:

1. Export `UnetModel.mat` to ONNX using MATLAB.
2. Validate the ONNX model in Python.
3. Optionally convert ONNX to a PyTorch model and save `.pt` weights.

## Export with MATLAB

Use one of these scripts:

- `export_unet_to_onnx.m` for a single model.
- `batch_export_unet_to_onnx.m` to export all axial/coronal/sagittal models.

Example (MATLAB):

```
export_unet_to_onnx('UnetModel.mat', 'dti_axial.onnx', 144, 144, 5);
```

## Validate and convert in Python

Use `convert_onnx_to_pytorch.py` to validate and optionally convert ONNX.

Example:

```
python convert_onnx_to_pytorch.py \
  --onnx /path/to/dti_axial.onnx \
  --out /path/to/dti_axial_state_dict.pt \
  --input-shape 1 144 144 5
```

Notes:
- The exported ONNX uses `BSSC` layout (batch, height, width, channels).
- `onnx2pytorch` is required for conversion to PyTorch. If it is missing, the script will stop after validation.

## Expected inputs

You should export the following models per parameter set:

- DTI: axial / sagittal / coronal
- MKCurve: axial / sagittal / coronal

## Output naming (suggested)

```
weights/
  dti_axial.onnx
  dti_sagittal.onnx
  dti_coronal.onnx
  mkcurve_axial.onnx
  mkcurve_sagittal.onnx
  mkcurve_coronal.onnx
```

If you provide PyTorch `.pt` weights instead, update the loader in `src/ddseg/model_loader.py`.
