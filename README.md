# DDSeg-PyTorch

This repository provides a PyTorch implementation of DDSeg. It mirrors the MATLAB workflow and supports TorchScript weights converted from MATLAB.

This code implements deep learning tissue segmentation method using diffusion MRI data, as described in the following paper:

    Fan Zhang, Anna Breger, Kang Ik Kevin Cho, Lipeng Ning, Carl-Fredrik Westin, Lauren O'Donnell, and Ofer Pasternak.
    Deep Learning Based Segmentation of Brain Tissue from Diffusion MRI
    NeuroImage 2021.

MATLAB version (original): https://github.com/zhangfanmark/DDSeg

## Release Status

- End-to-end DTI pipeline implemented (masking, normalization, padding, slicing, recombination).
- Inference for axial/sagittal/coronal models (TorchScript).
- MATLAB export + conversion workspace under `matlab_to_pytorch/`.
- DTI TorchScript weights included under `weights/`.

## Key Paths

- Conversion workspace: `matlab_to_pytorch/`
- PyTorch code: `src/ddseg/`
- Expected weights: `weights/`

## Installation

```
python -m pip install -r requirements.txt
```

## Conversion Steps (MATLAB -> TorchScript)

1. Export MATLAB `DAGNetwork` models to `.mat` weight dumps using:
   `matlab_to_pytorch/export_unet_weights.m`.
2. Convert dumps to TorchScript using:
   `matlab_to_pytorch/convert_matlab_weights_to_torchscript.py`.
3. Place generated weights under `weights/` with names like `dti_axial.pt`.
4. Run the pipeline with `scripts/run_ddseg.py`.

## Output channel order

The MATLAB pipeline expects the class scores ordered as: background, WM, GM, CSF.
Ensure the exported models preserve this channel order.

## Softmax handling

MATLAB `semanticseg` returns `allScores`. If your exported model already outputs probabilities, keep `--apply_softmax` disabled.
If your exported model outputs logits, enable `--apply_softmax` to match MATLAB behavior.

See `matlab_to_pytorch/README.md` for the export plan.

## Quick Start

```
python scripts/run_ddseg.py \
  --input_mask_nii /path/to/mask.nii.gz \
  --parameter_type DTI \
  --dwi_nii /path/to/dwi.nii.gz \
  --bval /path/to/dwi.bval \
  --bvec /path/to/dwi.bvec \
  --slicer_base /path/to/Slicer \
  --slicer_ext /path/to/SlicerDMRI \
  --weights_dir /path/to/weights \
  --output_folder /path/to/output \
  --device cuda
```

## Paths You Must Set

Before running, replace these with your local paths:

- `--input_mask_nii`
- `--dwi_nii`
- `--bval`
- `--bvec`
- `--slicer_base`
- `--slicer_ext`
- `--weights_dir`
- `--output_folder`

Notes:
- DTI parameters are generated from raw DWI using 3D Slicer CLI and written to `output_folder/DTI`.

## License

License is not specified yet. Add a LICENSE file if you intend to publish with explicit terms.
