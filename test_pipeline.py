"""
test_pipeline.py
Test script for the localization pipeline on sample_0000.
"""

import os
import sys
import time
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def preprocess_image(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns normalized intensity image and normalized gradient magnitude image."""
    # 1. Intensity representation (mild Gaussian blur to suppress shot noise + CLAHE)
    denoised = cv2.GaussianBlur(img, (3, 3), 0.8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_int = clahe.apply(denoised)

    # 2. Gradient representation (Sobel magnitude)
    gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    norm_grad = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return norm_int, norm_grad


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotates an image around its center."""
    if abs(angle) < 1e-4:
        return image
    h, w = image.shape
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def extract_local_maxima(corr_map: np.ndarray, threshold: float = 0.35, min_dist: int = 15) -> list[tuple[int, int, float]]:
    """Finds local maxima in a 2D correlation map."""
    # Peak detection via dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_dist, min_dist))
    dilated = cv2.dilate(corr_map, kernel)
    peaks_mask = (corr_map == dilated) & (corr_map >= threshold)
    
    y_indices, x_indices = np.where(peaks_mask)
    peaks = []
    for y, x in zip(y_indices, x_indices):
        peaks.append((int(x), int(y), float(corr_map[y, x])))
    return peaks


def subpixel_refine(corr_map: np.ndarray, x: int, y: int) -> tuple[float, float]:
    """2D parabolic subpixel peak fitting."""
    h, w = corr_map.shape
    if 1 <= x < w - 1 and 1 <= y < h - 1:
        c00 = corr_map[y, x]
        cx_m = corr_map[y, x - 1]
        cx_p = corr_map[y, x + 1]
        cy_m = corr_map[y - 1, x]
        cy_p = corr_map[y + 1, x]

        denom_x = 2.0 * (cx_m - 2.0 * c00 + cx_p)
        denom_y = 2.0 * (cy_m - 2.0 * c00 + cy_p)

        dx = (cx_m - cx_p) / denom_x if abs(denom_x) > 1e-6 else 0.0
        dy = (cy_m - cy_p) / denom_y if abs(denom_y) > 1e-6 else 0.0

        dx = np.clip(dx, -0.5, 0.5)
        dy = np.clip(dy, -0.5, 0.5)
        return float(x + dx), float(y + dy)
    return float(x), float(y)


def test_sample_0():
    t0 = time.time()
    ref_path = "data/reference/sample_0000_ref.png"
    search_path = "data/search/sample_0000_search.png"
    manifest_path = "data/dataset_manifest.csv"

    df = pd.read_csv(manifest_path)
    gt_row = df.iloc[0]
    gt_x = float(gt_row["ground_truth_x"])
    gt_y = float(gt_row["ground_truth_y"])
    print(f"Ground Truth for sample 0: ({gt_x}, {gt_y})")

    ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    ref_int, ref_grad = preprocess_image(ref_raw)
    search_int, search_grad = preprocess_image(search_raw)

    # Scale search range 9.0 to 11.0
    scales = np.linspace(9.0, 11.0, 21) # 0.1 step
    rotations = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) # Coarse rotation

    candidates = []

    for s in scales:
        tmpl_size = int(round(1000.0 / s))
        tmpl_int_scaled = cv2.resize(ref_int, (tmpl_size, tmpl_size), interpolation=cv2.INTER_AREA)
        tmpl_grad_scaled = cv2.resize(ref_grad, (tmpl_size, tmpl_size), interpolation=cv2.INTER_AREA)

        for rot in rotations:
            tmpl_int = rotate_image(tmpl_int_scaled, rot)
            tmpl_grad = rotate_image(tmpl_grad_scaled, rot)

            # Match on intensity
            corr_int = cv2.matchTemplate(search_int, tmpl_int, cv2.TM_CCOEFF_NORMED)
            corr_grad = cv2.matchTemplate(search_grad, tmpl_grad, cv2.TM_CCOEFF_NORMED)
            
            # Combined score
            corr_comb = 0.6 * corr_int + 0.4 * corr_grad

            peaks = extract_local_maxima(corr_comb, threshold=0.30, min_dist=20)
            for px, py, score in peaks:
                center_x = px + (tmpl_size / 2.0)
                center_y = py + (tmpl_size / 2.0)
                candidates.append({
                    "x": center_x,
                    "y": center_y,
                    "top_left_x": px,
                    "top_left_y": py,
                    "score": score,
                    "scale": s,
                    "rotation": rot,
                    "tmpl_size": tmpl_size,
                    "corr_map": corr_comb
                })

    print(f"Extracted {len(candidates)} total candidates across scales/rotations in {time.time() - t0:.3f}s.")

    # Spatial NMS
    candidates.sort(key=lambda c: c["score"], reverse=True)
    
    unique_candidates = []
    nms_radius = 25.0

    for cand in candidates:
        is_dup = False
        for u in unique_candidates:
            dist = np.sqrt((cand["x"] - u["x"])**2 + (cand["y"] - u["y"])**2)
            if dist < nms_radius:
                is_dup = True
                u["support_count"] = u.get("support_count", 1) + 1
                break
        if not is_dup:
            cand["support_count"] = 1
            unique_candidates.append(cand)

    print(f"Retained {len(unique_candidates)} distinct candidates after NMS.")
    
    print("\nTop 10 candidates after NMS:")
    for i, c in enumerate(unique_candidates[:10]):
        dist_center = np.sqrt((c['x'] - 500)**2 + (c['y'] - 500)**2)
        dist_gt = np.sqrt((c['x'] - gt_x)**2 + (c['y'] - gt_y)**2)
        print(f"  #{i+1}: Center=({c['x']:.2f}, {c['y']:.2f}), Score={c['score']:.4f}, Scale={c['scale']:.2f}, Rot={c['rotation']:.1f}°, Support={c['support_count']}, DistCenter={dist_center:.1f}px, ErrorVsGT={dist_gt:.2f}px")

    top_cand = unique_candidates[0]
    
    # Subpixel refinement
    sub_x, sub_y = subpixel_refine(top_cand["corr_map"], top_cand["top_left_x"], top_cand["top_left_y"])
    pred_x = sub_x + (top_cand["tmpl_size"] / 2.0)
    pred_y = sub_y + (top_cand["tmpl_size"] / 2.0)
    
    final_err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    print(f"\nFinal Predicted Coordinates: ({pred_x:.4f}, {pred_y:.4f})")
    print(f"Ground Truth Coordinates  : ({gt_x:.4f}, {gt_y:.4f})")
    print(f"Final Subpixel Error      : {final_err:.4f} pixels")
    print(f"Total Pipeline Runtime    : {time.time() - t0:.3f}s")


if __name__ == "__main__":
    test_sample_0()
