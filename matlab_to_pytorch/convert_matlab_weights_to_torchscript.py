#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.insert(0, str(SRC))

from ddseg.unet import MatlabUNet


def load_mean(meta_path: Path) -> np.ndarray:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    mean = np.array(data["layers"]["ImageInputLayer"]["Mean"], dtype=np.float32)
    if mean.ndim != 3:
        raise ValueError(f"Unexpected mean shape in {meta_path}: {mean.shape}")
    return mean


def _load_mat_array(path: Path, key: str) -> np.ndarray:
    mat = sio.loadmat(path.as_posix())
    if key not in mat:
        raise KeyError(f"Missing key '{key}' in {path}")
    return mat[key]


def _assign_conv(module: torch.nn.Module, w_path: Path, b_path: Path, transpose: bool = True) -> None:
    w = _load_mat_array(w_path, "w").astype(np.float32)
    b = _load_mat_array(b_path, "b").astype(np.float32)

    if transpose:
        w = np.transpose(w, (3, 2, 0, 1))
    b = b.reshape(-1)

    module.weight.data = torch.from_numpy(w)
    module.bias.data = torch.from_numpy(b)


def build_and_load(weights_dir: Path, meta_path: Path) -> MatlabUNet:
    mean = load_mean(meta_path)
    model = MatlabUNet(mean=mean)

    # Map MATLAB layer names to PyTorch modules.
    layer_map: Dict[str, torch.nn.Module] = {
        "Encoder-Stage-1-Conv-1": model.enc1_conv1,
        "Encoder-Stage-1-Conv-2": model.enc1_conv2,
        "Encoder-Stage-2-Conv-1": model.enc2_conv1,
        "Encoder-Stage-2-Conv-2": model.enc2_conv2,
        "Encoder-Stage-3-Conv-1": model.enc3_conv1,
        "Encoder-Stage-3-Conv-2": model.enc3_conv2,
        "Encoder-Stage-4-Conv-1": model.enc4_conv1,
        "Encoder-Stage-4-Conv-2": model.enc4_conv2,
        "Bridge-Conv-1": model.bridge_conv1,
        "Bridge-Conv-2": model.bridge_conv2,
        "Decoder-Stage-1-UpConv": model.up1,
        "Decoder-Stage-1-Conv-1": model.dec1_conv1,
        "Decoder-Stage-1-Conv-2": model.dec1_conv2,
        "Decoder-Stage-2-UpConv": model.up2,
        "Decoder-Stage-2-Conv-1": model.dec2_conv1,
        "Decoder-Stage-2-Conv-2": model.dec2_conv2,
        "Decoder-Stage-3-UpConv": model.up3,
        "Decoder-Stage-3-Conv-1": model.dec3_conv1,
        "Decoder-Stage-3-Conv-2": model.dec3_conv2,
        "Decoder-Stage-4-UpConv": model.up4,
        "Decoder-Stage-4-Conv-1": model.dec4_conv1,
        "Decoder-Stage-4-Conv-2": model.dec4_conv2,
        "Final-ConvolutionLayer": model.final_conv,
    }

    for matlab_name, module in layer_map.items():
        w_path = weights_dir / f"{matlab_name}_Weights.mat"
        b_path = weights_dir / f"{matlab_name}_Bias.mat"
        if not w_path.exists():
            raise FileNotFoundError(f"Missing weights: {w_path}")
        if not b_path.exists():
            raise FileNotFoundError(f"Missing bias: {b_path}")
        _assign_conv(module, w_path, b_path)

    model.eval()
    return model


def save_torchscript(model: torch.nn.Module, out_path: Path, input_shape) -> None:
    ts = torch.jit.script(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts.save(out_path.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MATLAB-exported weights to TorchScript UNet.")
    parser.add_argument("--weights_export", required=True, help="Path to weights_export folder.")
    parser.add_argument("--out_dir", required=True, help="Output folder for .pt files.")
    parser.add_argument("--prefix", default="dti", help="Model name prefix (default: dti).")
    args = parser.parse_args()

    weights_export = Path(args.weights_export).resolve()
    out_dir = Path(args.out_dir).resolve()
    prefix = args.prefix

    for view in ["axial", "sagittal", "coronal"]:
        view_dir = weights_export / f"{prefix}_{view}"
        meta_path = view_dir / "network_meta.json"
        if not view_dir.exists():
            raise FileNotFoundError(f"Missing weights export dir: {view_dir}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta: {meta_path}")

        model = build_and_load(view_dir, meta_path)
        mean = load_mean(meta_path)
        h, w, c = mean.shape
        input_shape = (1, c, h, w)
        out_path = out_dir / f"{prefix}_{view}.pt"
        save_torchscript(model, out_path, input_shape)
        print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
