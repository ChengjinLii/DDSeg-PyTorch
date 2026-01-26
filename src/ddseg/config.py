from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ParameterType = Literal["DTI", "MKCurve"]


@dataclass(frozen=True)
class DDSegConfig:
    input_feature_folder: Path
    input_mask_nii: Path
    parameter_type: ParameterType
    output_folder: Path
    weights_dir: Path
    device: str = "cpu"
    apply_softmax: bool = False
