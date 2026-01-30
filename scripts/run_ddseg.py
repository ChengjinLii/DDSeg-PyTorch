#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ddseg.config import DDSegConfig
from ddseg.pipeline import run_ddseg


def main() -> None:
    parser = argparse.ArgumentParser(description="DDSeg PyTorch pipeline")
    parser.add_argument("--input_mask_nii", required=True, help="Mask NIfTI")
    parser.add_argument("--parameter_type", choices=["DTI"], required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--apply_softmax", action="store_true", help="Apply softmax to model outputs")
    parser.add_argument("--dwi_nii", default="", help="Raw DWI NIfTI (for DTI generation via Slicer)")
    parser.add_argument("--bval", default="", help="bval file (for DTI generation via Slicer)")
    parser.add_argument("--bvec", default="", help="bvec file (for DTI generation via Slicer)")
    parser.add_argument("--slicer_base", default="", help="3D Slicer base path")
    parser.add_argument("--slicer_ext", default="", help="SlicerDMRI extension path")
    args = parser.parse_args()

    cfg = DDSegConfig(
        input_mask_nii=Path(args.input_mask_nii),
        parameter_type=args.parameter_type,
        output_folder=Path(args.output_folder),
        weights_dir=Path(args.weights_dir),
        device=args.device,
        apply_softmax=args.apply_softmax,
        dwi_nii=Path(args.dwi_nii) if args.dwi_nii else None,
        bval=Path(args.bval) if args.bval else None,
        bvec=Path(args.bvec) if args.bvec else None,
        slicer_base=Path(args.slicer_base) if args.slicer_base else None,
        slicer_ext=Path(args.slicer_ext) if args.slicer_ext else None,
    )
    run_ddseg(cfg)


if __name__ == "__main__":
    main()
