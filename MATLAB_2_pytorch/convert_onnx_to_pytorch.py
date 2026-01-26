#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate ONNX and optionally convert to PyTorch state_dict."
    )
    parser.add_argument("--onnx", required=True, help="Path to ONNX model.")
    parser.add_argument(
        "--out",
        default="",
        help="Output .pt path for state_dict. If omitted, only validation runs.",
    )
    parser.add_argument(
        "--input-shape",
        nargs=4,
        type=int,
        required=True,
        metavar=("N", "H", "W", "C"),
        help="Input shape in BSSC format (batch, height, width, channels).",
    )
    return parser.parse_args()


def validate_onnx(onnx_path: Path, input_shape):
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    x = np.random.rand(*input_shape).astype(np.float32)
    _ = session.run(None, {input_name: x})


def convert_to_pytorch(onnx_path: Path, out_path: Path, input_shape):
    try:
        from onnx2pytorch import ConvertModel
    except Exception as exc:
        raise RuntimeError(
            "onnx2pytorch is required for conversion. "
            "Install it and rerun with --out."
        ) from exc

    onnx_model = onnx.load(str(onnx_path))
    torch_model = ConvertModel(onnx_model, experimental=True)
    torch_model.eval()

    x = torch.from_numpy(np.random.rand(*input_shape).astype(np.float32))
    with torch.no_grad():
        _ = torch_model(x)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_model.state_dict(), str(out_path))


def main():
    args = parse_args()
    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        print(f"ONNX not found: {onnx_path}", file=sys.stderr)
        return 2

    input_shape = tuple(args.input_shape)
    validate_onnx(onnx_path, input_shape)
    print(f"ONNX validation OK: {onnx_path}")

    if args.out:
        out_path = Path(args.out).resolve()
        convert_to_pytorch(onnx_path, out_path, input_shape)
        print(f"Saved state_dict: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
