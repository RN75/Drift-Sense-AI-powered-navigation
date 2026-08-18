"""
test_multi_sample.py
Generate 5 samples and test candidate generation, scoring, and accuracy across them.
"""

import os
import sys
import time
import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.synthetic_dram import DRAMLayoutGenerator, extract_reference_and_search
from src.sem_noise import apply_sem_degradation


def generate_test_samples(num_samples: int = 5, base_seed: int = 42):
    os.makedirs("test_data/reference", exist_ok=True)
    os.makedirs("test_data/search", exist_ok=True)
    
    generator = DRAMLayoutGenerator()
    records = []

    for i in range(num_samples):
        sample_seed = base_seed + i
        canvas = generator.generate_canvas(width=12000, height=12000, seed=sample_seed)
        ref_raw, search_clean, gt_x, gt_y, meta = extract_reference_and_search(
            canvas=canvas,
            scale_ratio=10.0,
            ref_size=(1000, 1000),
            search_size=(1000, 1000),
            margin=600,
            seed=sample_seed + 1000
        )

        ref_img = np.clip(ref_raw, 0, 255).astype(np.uint8)
        search_img, noise_meta = apply_sem_degradation(
            search_clean,
            poisson_peak=130.0,
            gaussian_noise_std=5.5,
            blur_sigma=0.8,
            edge_charge_strength=0.35,
            salt_pepper_prob=0.0008,
            contrast_factor=1.05,
            brightness_offset=-4.0,
            gamma=1.04,
            streak_strength=2.5,
            seed=sample_seed + 2000
        )

        ref_path = f"test_data/reference/sample_{i:04d}_ref.png"
        search_path = f"test_data/search/sample_{i:04d}_search.png"
        cv2.imwrite(ref_path, ref_img)
        cv2.imwrite(search_path, search_img)

        records.append({
            "sample_id": i,
            "ref_path": ref_path,
            "search_path": search_path,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "scale_ratio": 10.0
        })

    return pd.DataFrame(records)


def evaluate_matching_strategy(samples_df):
    scales = np.linspace(9.0, 11.0, 21) # 21 scales
    rotations = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) # 5 rotations

    print("Evaluating matching on generated samples...")
    results = []

    for _, row in samples_df.iterrows():
        t0 = time.time()
        ref = cv2.imread(row["ref_path"], cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
        gt_x = row["gt_x"]
        gt_y = row["gt_y"]

        # Preprocessing: Gaussian filter to reduce SEM shot noise
        ref_smooth = cv2.GaussianBlur(ref, (3, 3), 0.7)
        search_smooth = cv2.GaussianBlur(search, (3, 3), 0.7)

        # Gradient
        ref_gx = cv2.Sobel(ref_smooth, cv2.CV_32F, 1, 0, ksize=3)
        ref_gy = cv2.Sobel(ref_smooth, cv2.CV_32F, 0, 1, ksize=3)
        ref_g = cv2.magnitude(ref_gx, ref_gy)
        ref_g_norm = cv2.normalize(ref_g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        search_gx = cv2.Sobel(search_smooth, cv2.CV_32F, 1, 0, ksize=3)
        search_gy = cv2.Sobel(search_smooth, cv2.CV_32F, 0, 1, ksize=3)
        search_g = cv2.magnitude(search_gx, search_gy)
        search_g_norm = cv2.normalize(search_g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Multi-scale & multi-rotation search
        best_cand = None
        all_candidates = []

        for s in scales:
            tmpl_size = int(round(1000.0 / s))
            tmpl_int = cv2.resize(ref_smooth, (tmpl_size, tmpl_size), interpolation=cv2.INTER_AREA)
            tmpl_g = cv2.resize(ref_g_norm, (tmpl_size, tmpl_size), interpolation=cv2.INTER_AREA)

            for rot in rotations:
                if abs(rot) > 1e-4:
                    center = (tmpl_size / 2.0, tmpl_size / 2.0)
                    rot_mat = cv2.getRotationMatrix2D(center, rot, 1.0)
                    tmpl_int_r = cv2.warpAffine(tmpl_int, rot_mat, (tmpl_size, tmpl_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    tmpl_g_r = cv2.warpAffine(tmpl_g, rot_mat, (tmpl_size, tmpl_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                else:
                    tmpl_int_r = tmpl_int
                    tmpl_g_r = tmpl_g

                res_int = cv2.matchTemplate(search_smooth, tmpl_int_r, cv2.TM_CCOEFF_NORMED)
                res_g = cv2.matchTemplate(search_g_norm, tmpl_g_r, cv2.TM_CCOEFF_NORMED)

                # Combined correlation map
                res_comb = 0.65 * res_int + 0.35 * res_g

                min_v, max_v, min_l, max_l = cv2.minMaxLoc(res_comb)
                center_x = max_l[0] + (tmpl_size / 2.0)
                center_y = max_l[1] + (tmpl_size / 2.0)

                all_candidates.append({
                    "x": center_x,
                    "y": center_y,
                    "top_left_x": max_l[0],
                    "top_left_y": max_l[1],
                    "score": max_v,
                    "scale": s,
                    "rotation": rot,
                    "tmpl_size": tmpl_size,
                    "corr_map": res_comb
                })

        # Sort candidates
        all_candidates.sort(key=lambda c: c["score"], reverse=True)
        top = all_candidates[0]

        # Parabolic subpixel refinement
        c_map = top["corr_map"]
        tx, ty = top["top_left_x"], top["top_left_y"]
        h_map, w_map = c_map.shape
        dx, dy = 0.0, 0.0
        if 1 <= tx < w_map - 1 and 1 <= ty < h_map - 1:
            c00 = c_map[ty, tx]
            cx_m = c_map[ty, tx - 1]
            cx_p = c_map[ty, tx + 1]
            cy_m = c_map[ty - 1, tx]
            cy_p = c_map[ty + 1, tx]

            den_x = 2.0 * (cx_m - 2.0 * c00 + cx_p)
            den_y = 2.0 * (cy_m - 2.0 * c00 + cy_p)
            if abs(den_x) > 1e-6:
                dx = np.clip((cx_m - cx_p) / den_x, -0.5, 0.5)
            if abs(den_y) > 1e-6:
                dy = np.clip((cy_m - cy_p) / den_y, -0.5, 0.5)

        pred_x = tx + dx + (top["tmpl_size"] / 2.0)
        pred_y = ty + dy + (top["tmpl_size"] / 2.0)

        err_x = abs(pred_x - gt_x)
        err_y = abs(pred_y - gt_y)
        euc_err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        elapsed = time.time() - t0

        print(f"Sample {row['sample_id']}: GT=({gt_x:.2f}, {gt_y:.2f}) -> Pred=({pred_x:.2f}, {pred_y:.2f}) | Error={euc_err:.4f}px | TopScore={top['score']:.4f} | Scale={top['scale']:.2f} | Rot={top['rotation']:.1f}° | Time={elapsed:.2f}s")
        results.append({
            "sample_id": row["sample_id"],
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error_x": err_x,
            "error_y": err_y,
            "euc_error": euc_err,
            "runtime": elapsed
        })

    res_df = pd.DataFrame(results)
    print("\nSummary Results:")
    print(f"Mean Error: {res_df['euc_error'].mean():.4f} px")
    print(f"Median Error: {res_df['euc_error'].median():.4f} px")
    print(f"Max Error: {res_df['euc_error'].max():.4f} px")
    print(f"Percentage < 1.0 px: {(res_df['euc_error'] < 1.0).mean() * 100:.1f}%")
    print(f"Average Runtime: {res_df['runtime'].mean():.2f}s")


if __name__ == "__main__":
    df = generate_test_samples(5, base_seed=42)
    evaluate_matching_strategy(df)
