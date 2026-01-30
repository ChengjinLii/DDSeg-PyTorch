from __future__ import annotations

from pathlib import Path
import shutil
from typing import Optional

import numpy as np
import nibabel as nib

from .config import DDSegConfig
from .preprocess import masking_and_normalizing, stack_features, split_views
from .model_loader import load_models
from .inference import predict_view, combine_views, load_matlab_predictions
from .utils import padding_unpadding, load_nii_matlab_like
from .dti_slicer import run_dti_feature_extraction


def run_ddseg(cfg: DDSegConfig, matlab_prediction_dir: Optional[Path] = None) -> None:
    cfg.output_folder.mkdir(parents=True, exist_ok=True)
    feature_folder = cfg.input_feature_folder
    if cfg.parameter_type == "DTI":
        # Generate DTI features from raw DWI using Slicer CLI if provided.
        if cfg.dwi_nii and cfg.bval and cfg.bvec:
            dti_out = cfg.output_folder / "DTI"
            run_dti_feature_extraction(
                cfg.dwi_nii, cfg.bval, cfg.bvec, dti_out, cfg.slicer_base, cfg.slicer_ext
            )
            feature_folder = dti_out
        elif cfg.input_feature_folder is not None:
            # Mirror MATLAB-style DTI folder in output for alignment.
            dti_out = cfg.output_folder / "DTI"
            if not dti_out.exists():
                shutil.copytree(cfg.input_feature_folder, dti_out, dirs_exist_ok=True)
            feature_folder = cfg.input_feature_folder

    feature_maps = masking_and_normalizing(
        feature_folder,
        cfg.input_mask_nii,
        cfg.parameter_type,
    )
    stacked = stack_features(feature_maps, cfg.parameter_type)
    views = split_views(stacked)

    prefix = "dti" if cfg.parameter_type == "DTI" else "mkcurve"
    if matlab_prediction_dir is not None:
        preds = load_matlab_predictions(matlab_prediction_dir.as_posix())
        pred_axial = preds["axial"]
        pred_sagittal = preds["sagittal"]
        pred_coronal = preds["coronal"]
    else:
        models = load_models(cfg.weights_dir, prefix, cfg.device)
        pred_axial = predict_view(models.axial, views["axial"], cfg.apply_softmax)
        pred_sagittal = predict_view(models.sagittal, views["sagittal"], cfg.apply_softmax)
        pred_coronal = predict_view(models.coronal, views["coronal"], cfg.apply_softmax)

    combined = combine_views(pred_axial, pred_sagittal, pred_coronal)
    prob_maps = combined["prob_maps"]
    labels = combined["labels"]

    mask_img = load_nii_matlab_like(cfg.input_mask_nii)
    mask = mask_img.get_fdata() > 0
    img_size = mask.shape

    tmp = np.concatenate([labels[..., None], prob_maps], axis=3)
    tmp = padding_unpadding(tmp, img_size, "unpadding")
    labels = tmp[..., 0]
    prob_maps = tmp[..., 1:]

    labels[~mask] = 0
    for idx in range(prob_maps.shape[3]):
        prob_maps[..., idx][~mask] = 0

    out_seg = cfg.output_folder / "SegmentationMap_GMWMCSF.nii.gz"
    out_gm = cfg.output_folder / "ProbabilisticMap_GM.nii.gz"
    out_wm = cfg.output_folder / "ProbabilisticMap_WM.nii.gz"
    out_csf = cfg.output_folder / "ProbabilisticMap_CSF.nii.gz"

    seg_img = nib.Nifti1Image(labels.astype(np.float32), mask_img.affine, mask_img.header)
    nib.save(seg_img, out_seg.as_posix())

    header = mask_img.header.copy()
    header.set_data_dtype(np.float32)

    gm = nib.Nifti1Image(prob_maps[..., 2].astype(np.float32), mask_img.affine, header)
    wm = nib.Nifti1Image(prob_maps[..., 1].astype(np.float32), mask_img.affine, header)
    csf = nib.Nifti1Image(prob_maps[..., 3].astype(np.float32), mask_img.affine, header)

    nib.save(gm, out_gm.as_posix())
    nib.save(wm, out_wm.as_posix())
    nib.save(csf, out_csf.as_posix())
