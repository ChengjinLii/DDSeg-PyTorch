# DDSeg PyTorch Version (Scaffold)

This folder is a PyTorch-ready scaffold for DDSeg. It mirrors the MATLAB workflow but does **not** include converted weights yet.

## Status

- ✅ Data preparation pipeline implemented (masking, normalization, padding, slicing, recombination).
- ✅ Inference API stubs for axial/sagittal/coronal models.
- ✅ Conversion workspace prepared under `MATLAB_2_pytorch/`.
- ⛔ MATLAB weights are not yet converted. You must export or supply PyTorch/ONNX weights.

## Key paths

- Conversion workspace: `MATLAB_2_pytorch/`
- PyTorch code: `src/ddseg/`
- Expected weights (placeholders): `weights/`

## Next steps to complete conversion

1. Export MATLAB `DAGNetwork` models to ONNX or TorchScript.
2. Place converted weights under `weights/` with names like `dti_axial.onnx` or `dti_axial.pt`.
3. Run the pipeline with `scripts/run_ddseg.py`.

## Output channel order

The MATLAB pipeline expects the class scores ordered as: background, WM, GM, CSF.
Ensure the exported models preserve this channel order.

## Softmax handling

MATLAB `semanticseg` returns `allScores`. If your exported model already outputs probabilities, keep `--apply_softmax` disabled.
If your exported model outputs logits, enable `--apply_softmax` to match MATLAB behavior.

See `MATLAB_2_pytorch/README.md` for the export plan.
