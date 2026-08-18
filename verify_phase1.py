"""
verify_phase1.py
Comprehensive mathematical and physical validation of Phase 1 synthetic dataset.
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.synthetic_dram import DRAMLayoutGenerator, extract_reference_and_search
from src.sem_noise import apply_sem_degradation


def run_phase1_validation():
    print("==================================================")
    print("PHASE 1 RIGOROUS MATHEMATICAL & PHYSICAL VALIDATION")
    print("==================================================")

    data_dir = "data"
    manifest_path = os.path.join(data_dir, "dataset_manifest.csv")

    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"
    df = pd.read_csv(manifest_path)
    print(f"[1] Manifest found with {len(df)} entry/entries.")

    for idx, row in df.iterrows():
        ref_path = os.path.join(data_dir, row["reference_path"])
        search_path = os.path.join(data_dir, row["search_path"])

        print(f"\nChecking sample {int(row['sample_id'])}:")
        print(f"  Reference path: {ref_path}")
        print(f"  Search path:    {search_path}")

        assert os.path.exists(ref_path), f"Reference file missing: {ref_path}"
        assert os.path.exists(search_path), f"Search file missing: {search_path}"

        # 1. Load images using OpenCV in grayscale
        ref_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
        search_img = cv2.imread(search_path, cv2.IMREAD_UNCHANGED)

        # 2. Check dimensions and grayscale format
        assert ref_img is not None, "Failed to load reference image"
        assert search_img is not None, "Failed to load search image"
        assert ref_img.ndim == 2, f"Reference is not grayscale (ndim={ref_img.ndim})"
        assert search_img.ndim == 2, f"Search is not grayscale (ndim={search_img.ndim})"
        assert ref_img.shape == (1000, 1000), f"Reference shape {ref_img.shape} != (1000, 1000)"
        assert search_img.shape == (1000, 1000), f"Search shape {search_img.shape} != (1000, 1000)"
        assert ref_img.dtype == np.uint8, f"Reference dtype {ref_img.dtype} != uint8"
        assert search_img.dtype == np.uint8, f"Search dtype {search_img.dtype} != uint8"

        print("  [PASS] Image dimensions (1000x1000) and grayscale uint8 format verified.")

        # 3. Check ground truth coordinates
        gt_x = float(row["ground_truth_x"])
        gt_y = float(row["ground_truth_y"])
        scale_ratio = float(row["scale_ratio"])
        print(f"  GT coordinates: ({gt_x:.4f}, {gt_y:.4f}) | Scale: {scale_ratio}")

        assert 0.0 <= gt_x < 1000.0, f"Ground truth X out of bounds: {gt_x}"
        assert 0.0 <= gt_y < 1000.0, f"Ground truth Y out of bounds: {gt_y}"

        # Target bounding box in search coordinates
        target_size_px = 1000.0 / scale_ratio
        half_size = target_size_px / 2.0
        x_min = gt_x - half_size
        x_max = gt_x + half_size
        y_min = gt_y - half_size
        y_max = gt_y + half_size

        assert x_min >= 0.0 and x_max <= 1000.0, f"Target bbox X [{x_min}, {x_max}] exceeds search image"
        assert y_min >= 0.0 and y_max <= 1000.0, f"Target bbox Y [{y_min}, {y_max}] exceeds search image"
        print(f"  [PASS] Ground truth coordinates and target bounding box [{x_min:.1f}, {x_max:.1f}, {y_min:.1f}, {y_max:.1f}] strictly within bounds.")
        print(f"  [PASS] Target width in search image = {target_size_px:.1f} pixels (nominal 100x100).")

        # 4. Mathematical Ground Truth Alignment Proof
        generator = DRAMLayoutGenerator()
        sample_seed = int(row["random_seed"])
        canvas = generator.generate_canvas(width=12000, height=12000, seed=sample_seed)
        ref_clean, search_clean, calc_gt_x, calc_gt_y, meta = extract_reference_and_search(
            canvas=canvas,
            scale_ratio=scale_ratio,
            ref_size=(1000, 1000),
            search_size=(1000, 1000),
            margin=600,
            seed=sample_seed + 1000
        )

        assert abs(calc_gt_x - gt_x) < 1e-3, f"Calculated GT X {calc_gt_x} mismatch with manifest {gt_x}"
        assert abs(calc_gt_y - gt_y) < 1e-3, f"Calculated GT Y {calc_gt_y} mismatch with manifest {gt_y}"
        print("  [PASS] Ground-truth derivation matches physical coordinate geometry exactly.")

        # 5. Exact 1:1 High-Resolution Underlying Physical Patch Verification
        # The reference crop MUST be pixel-for-pixel identical to the high-res canvas at its location
        search_field_w = int(round(1000 * scale_ratio))
        search_field_h = int(round(1000 * scale_ratio))
        search_field_highres = canvas[
            meta["search_start_y"]:meta["search_start_y"] + search_field_h,
            meta["search_start_x"]:meta["search_start_x"] + search_field_w
        ]
        highres_target_patch = search_field_highres[
            meta["ref_rel_y"]:meta["ref_rel_y"] + 1000,
            meta["ref_rel_x"]:meta["ref_rel_x"] + 1000
        ]

        patch_diff = np.max(np.abs(ref_clean - highres_target_patch))
        print(f"  Exact High-Res Patch Difference Max: {patch_diff:.8f}")
        assert patch_diff == 0.0, "Reference crop does not match the exact high-res layout patch!"
        print("  [PASS] High-resolution underlying physical structure is 100.000% identical.")

        # 6. Check statistics of generated files
        print(f"  Reference stats: min={ref_img.min()}, max={ref_img.max()}, mean={ref_img.mean():.2f}, std={ref_img.std():.2f}")
        print(f"  Search stats:    min={search_img.min()}, max={search_img.max()}, mean={search_img.mean():.2f}, std={search_img.std():.2f}")

    print("\n==================================================")
    print("ALL PHASE 1 VALIDATION CHECKS PASSED PERFECTLY!")
    print("==================================================")


if __name__ == "__main__":
    run_phase1_validation()
