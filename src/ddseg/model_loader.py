from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class InferenceModel(Protocol):
    def forward(self, x: np.ndarray) -> np.ndarray:  # x: N,C,H,W
        ...
    input_layout: str
    output_layout: str


@dataclass
class TorchScriptModel:
    module: torch.nn.Module
    device: str
    input_layout: str = "NCHW"
    output_layout: str = "NCHW"

    def forward(self, x: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            out = self.module(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.cpu().numpy()


@dataclass
class OnnxRuntimeModel:
    session: "onnxruntime.InferenceSession"
    input_name: str
    input_layout: str = "NHWC"
    output_layout: str = "NHWC"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]


@dataclass
class ModelBundle:
    axial: InferenceModel
    sagittal: InferenceModel
    coronal: InferenceModel


def load_torch_model(path: Path, device: str) -> TorchScriptModel:
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    module = torch.jit.load(path.as_posix(), map_location=device)
    module.eval()
    return TorchScriptModel(module=module, device=device)


def load_onnx_model(path: Path) -> OnnxRuntimeModel:
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    import onnxruntime  # lazy import

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(
        path.as_posix(),
        providers=[p for p in providers if p in available],
    )
    input_name = session.get_inputs()[0].name
    return OnnxRuntimeModel(session=session, input_name=input_name)


def _pick_model(weights_dir: Path, base: str, device: str) -> InferenceModel:
    onnx_path = weights_dir / f"{base}.onnx"
    pt_path = weights_dir / f"{base}.pt"
    if onnx_path.exists():
        return load_onnx_model(onnx_path)
    return load_torch_model(pt_path, device)


def load_models(weights_dir: Path, prefix: str, device: str) -> ModelBundle:
    axial = _pick_model(weights_dir, f"{prefix}_axial", device)
    sagittal = _pick_model(weights_dir, f"{prefix}_sagittal", device)
    coronal = _pick_model(weights_dir, f"{prefix}_coronal", device)
    return ModelBundle(axial=axial, sagittal=sagittal, coronal=coronal)
