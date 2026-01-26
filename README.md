# DDSeg-PyTorch

This repository provides a PyTorch implementation of DDSeg. It mirrors the MATLAB workflow and supports ONNX or TorchScript weights.

This code implements deep learning tissue segmentation method using diffusion MRI data, as described in the following paper:

    Fan Zhang, Anna Breger, Kang Ik Kevin Cho, Lipeng Ning, Carl-Fredrik Westin, Lauren O'Donnell, and Ofer Pasternak.
    Deep Learning Based Segmentation of Brain Tissue from Diffusion MRI
    NeuroImage 2021.

## Release Status

- ✅ End-to-end data pipeline implemented (masking, normalization, padding, slicing, recombination).
- ✅ Inference for axial/sagittal/coronal models (ONNX or TorchScript).
- ✅ Conversion workspace prepared under `matlab_to_pytorch/`.
- ⛔ MATLAB weights are not included. You must export or supply ONNX/TorchScript weights.

## Key Paths

- Conversion workspace: `matlab_to_pytorch/`
- PyTorch code: `src/ddseg/`
- Expected weights (placeholders): `weights/`

## Installation

```
python -m pip install -r requirements.txt
```

Optional (GPU ONNXRuntime):

```
python -m pip install onnxruntime-gpu
```

## Conversion Steps

1. Export MATLAB `DAGNetwork` models to ONNX.
2. Place converted weights under `weights/` with names like `dti_axial.onnx` or `dti_axial.pt`.
3. Run the pipeline with `scripts/run_ddseg.py`.

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
  --input_feature_folder /path/to/features \
  --input_mask_nii /path/to/mask.nii.gz \
  --parameter_type DTI \
  --weights_dir /path/to/weights \
  --output_folder /path/to/output \
  --device cuda
```

## Release Notes

See `RELEASE_NOTES.md` for release details and known limitations.

## License

License is not specified yet. Add a LICENSE file if you intend to publish with explicit terms.
