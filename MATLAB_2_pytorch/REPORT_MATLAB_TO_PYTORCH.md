# MATLAB-to-PyTorch Conversion Report (DDSeg)

## Goal

Make the PyTorch implementation behavior identical to the MATLAB implementation. The only difference should be the programming language.

## Source and Target

- MATLAB project: `/data04/chengjin/DDsurfer/DDSeg/DDSeg-master`
- PyTorch project: `/data04/chengjin/DDsurfer/DDSeg-pytorch-version`

## Core MATLAB Pipeline (Reference)

1. Masking and normalizing input parameters
2. Split 3D features into axial/coronal/sagittal slices
3. CNN prediction per view
4. Combine 3-view predictions
5. Output segmentation and probabilistic maps

## One-to-One Behavior Checks

### 1) Input loading and orientation

MATLAB uses `load_nii` which applies `xform_nii` (sform/qform reorientation).

Python now applies the closest canonical orientation when loading `.nii/.nii.gz`:

- MATLAB: `load_nii` -> `xform_nii`
- Python: `nib.as_closest_canonical(...)`

This matches the MATLAB reorientation behavior.

### 2) NRRD loading

MATLAB reads NRRD and applies `fliplr` and `flipud`.

Python matches this exactly:

- `data = np.fliplr(data)`
- `data = np.flipud(data)`

### 3) Masking and truncation

Both versions:

- mask outside the brain -> `NaN`
- truncate to per-feature range
- normalize only inside mask

### 4) Normalization formula

MATLAB `normalize` uses sample standard deviation (N-1).

Python now uses `np.nanstd(..., ddof=1)` and the same mean/std logic.

### 5) Padding and unpadding

MATLAB uses `padding_unpadding` with predefined target sizes.

Python matches the same target list and padding rules.

### 6) NaN fill value

Both versions set background to `-100` before inference.

### 7) View slicing order

Axial, coronal, sagittal slicing order is identical.

### 8) Combine views

Both versions:

- sum three views
- divide by 3
- `argmax` over class dimension
- label range fixed to 0..3

Python now applies the `-1` offset to match MATLAB.

Sagittal view reordering matches MATLAB `permute([3, 1, 2, 4])`, so
the reconstructed volume is aligned as `(X, Y, Z, C)`.

### 9) Output maps and labels

Both versions:

- output labels as float32
- output GM/WM/CSF probabilistic maps as float32
- apply input mask to zero outside brain

## MATLAB to ONNX Export

These `UnetModel.mat` files store a MATLAB `DAGNetwork` object. Python cannot deserialize it directly. The reliable route is:

1. Export the MATLAB model to ONNX:
   - `export_unet_to_onnx.m`
   - `batch_export_unet_to_onnx.m`

2. Validate ONNX in Python:
   - `convert_onnx_to_pytorch.py`

3. Optionally convert ONNX to PyTorch state_dict

### MATLAB Export Notes

- Export format uses `BSSC` (NHWC) layout.
- Python inference now respects NHWC for ONNX.

## What Is Still Required for Full Numerical Identity

The logic is now aligned. Exact numerical identity still requires:

- MATLAB-exported ONNX files
- PyTorch inference on the same inputs
- A direct output comparison (voxel-wise diff)

These checks can only be done after the weights are available in ONNX format.

## Numerical Alignment Script

Use the script below after you have MATLAB reference outputs:

```
python validate_alignment.py \
  --feature-dir /path/to/features \
  --mask-nii /path/to/mask.nii.gz \
  --parameter-type DTI \
  --weights-dir /path/to/weights \
  --output-dir /path/to/pytorch_out \
  --ref-dir /path/to/matlab_out
```

This script runs the PyTorch pipeline and compares the outputs against the MATLAB output maps.

## Conversion Assets

Files located in:

`/data04/chengjin/DDsurfer/DDSeg-pytorch-version/MATLAB_2_pytorch`

- `export_unet_to_onnx.m`
- `batch_export_unet_to_onnx.m`
- `convert_onnx_to_pytorch.py`

## Summary

All non-weight logic is aligned between MATLAB and Python. With ONNX-exported weights, the pipeline should be numerically identical given the same inputs.
