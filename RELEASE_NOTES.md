# Release Notes

## v1.0.0

### Highlights
- End-to-end DDSeg pipeline in PyTorch matching the MATLAB workflow.
- ONNX and TorchScript inference support for axial/sagittal/coronal models.
- MATLAB-to-ONNX conversion utilities with validation scripts.

### Known Limitations
- Pretrained weights are not included. You must export or supply ONNX/TorchScript weights.
- MATLAB is required to export `DAGNetwork` models to ONNX.
- Exact numerical identity must be validated using `matlab_to_pytorch/validate_alignment.py` once weights are available.

### Requirements
- Python 3.10+
- `onnxruntime` (CPU) or `onnxruntime-gpu` (GPU)

### Notes
- ONNX models are expected to use NHWC (BSSC) layout as exported by MATLAB.
- Output channel order must be background, WM, GM, CSF.
