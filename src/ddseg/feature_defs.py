from __future__ import annotations

FEATURES = {
    "DTI": {
        "list": ["DTI_FA", "DTI_MD", "DTI_E1", "DTI_E2", "DTI_E3"],
        "range": [
            (0.0, 1.0),
            (-0.001, 0.004),
            (-0.001, 0.004),
            (-0.001, 0.004),
            (-0.001, 0.004),
        ],
    },
    "MKCurve": {
        "list": [
            "corrected_FA",
            "corrected_E1",
            "corrected_E2",
            "corrected_E3",
            "corrected_MK",
            "corrected_AK",
            "corrected_RK",
            "MKCurve_MaxMK-b0",
            "MKCurve_ZeroMK-b0",
            "MKCurve_MaxMK",
        ],
        "range": [
            (0.0, 1.0),
            (-0.001, 0.004),
            (-0.001, 0.004),
            (-0.001, 0.004),
            (-1.0, 4.0),
            (-1.0, 4.0),
            (-1.0, 4.0),
            (-1000.0, 100000.0),
            (-1000.0, 100000.0),
            (-1.0, 4.0),
        ],
    },
}
