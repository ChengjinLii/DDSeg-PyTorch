#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PyTorch DDSeg outputs against MATLAB reference outputs."
    )
    parser.add_argument("--feature-dir", required=True, help="Input feature folder.")
    parser.add_argument("--mask-nii", required=True, help="Input brain mask NIfTI.")
    parser.add_argument("--parameter-type", required=True, choices=["DTI", "MKCurve"])
    parser.add_argument("--weights-dir", required=True, help="ONNX/PT weights folder.")
    parser.add_argument("--output-dir", required=True, help="PyTorch output folder.")
    parser.add_argument("--ref-dir", required=True, help="MATLAB output folder.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda.")
    parser.add_argument("--apply-softmax", action="store_true")
    parser.add_argument("--tol-prob", type=float, default=1e-4)
    parser.add_argument("--tol-label", type=float, default=0.0)
    return parser.parse_args()


def load_nii(path: Path) -> np.ndarray:
    return nib.load(path.as_posix()).get_fdata()


def compare_arrays(name: str, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    if a.shape != b.shape:
        print(f"{name}: shape mismatch {a.shape} vs {b.shape}")
        return False
    diff = np.abs(a - b)
    max_diff = float(np.nanmax(diff))
    mean_diff = float(np.nanmean(diff))
    print(f"{name}: max_abs_diff={max_diff:.6g}, mean_abs_diff={mean_diff:.6g}")
    return max_diff <= tol


def compare_labels(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    if a.shape != b.shape:
        print(f"labels: shape mismatch {a.shape} vs {b.shape}")
        return False
    mismatch = (a != b)
    mismatch_rate = float(np.mean(mismatch))
    print(f"labels: mismatch_rate={mismatch_rate:.6g}")
    return mismatch_rate <= tol


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, (root / "src").as_posix())

    from ddseg.config import DDSegConfig
    from ddseg.pipeline import run_ddseg

    cfg = DDSegConfig(
        input_feature_folder=Path(args.feature_dir),
        input_mask_nii=Path(args.mask_nii),
        parameter_type=args.parameter_type,
        output_folder=Path(args.output_dir),
        weights_dir=Path(args.weights_dir),
        device=args.device,
        apply_softmax=args.apply_softmax,
    )
    run_ddseg(cfg)

    out_seg = Path(args.output_dir) / "SegmentationMap_GMWMCSF.nii.gz"
    out_gm = Path(args.output_dir) / "ProbabilisticMap_GM.nii.gz"
    out_wm = Path(args.output_dir) / "ProbabilisticMap_WM.nii.gz"
    out_csf = Path(args.output_dir) / "ProbabilisticMap_CSF.nii.gz"

    ref_seg = Path(args.ref_dir) / "SegmentationMap_GMWMCSF.nii.gz"
    ref_gm = Path(args.ref_dir) / "ProbabilisticMap_GM.nii.gz"
    ref_wm = Path(args.ref_dir) / "ProbabilisticMap_WM.nii.gz"
    ref_csf = Path(args.ref_dir) / "ProbabilisticMap_CSF.nii.gz"

    ok = True
    ok &= compare_labels(load_nii(out_seg), load_nii(ref_seg), args.tol_label)
    ok &= compare_arrays("prob_gm", load_nii(out_gm), load_nii(ref_gm), args.tol_prob)
    ok &= compare_arrays("prob_wm", load_nii(out_wm), load_nii(ref_wm), args.tol_prob)
    ok &= compare_arrays("prob_csf", load_nii(out_csf), load_nii(ref_csf), args.tol_prob)

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
