#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ddseg.config import DDSegConfig
from ddseg.pipeline import run_ddseg


def main() -> None:
    parser = argparse.ArgumentParser(description="DDSeg PyTorch pipeline")
    parser.add_argument("--input_feature_folder", required=True, help="Folder with DTI/MKCurve parameter maps")
    parser.add_argument("--input_mask_nii", required=True, help="Mask NIfTI")
    parser.add_argument("--parameter_type", choices=["DTI", "MKCurve"], required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--apply_softmax", action="store_true", help="Apply softmax to model outputs")
    args = parser.parse_args()

    cfg = DDSegConfig(
        input_feature_folder=Path(args.input_feature_folder),
        input_mask_nii=Path(args.input_mask_nii),
        parameter_type=args.parameter_type,
        output_folder=Path(args.output_folder),
        weights_dir=Path(args.weights_dir),
        device=args.device,
        apply_softmax=args.apply_softmax,
    )
    run_ddseg(cfg)


if __name__ == "__main__":
    main()
