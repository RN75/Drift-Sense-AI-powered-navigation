"""
explore_matching.py
Exploratory script to analyze NCC maps, gradient correlation, scale response, and peak distributions on sample_0000.
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def analyze_sample_0():
    ref_path = "data/reference/sample_0000_ref.png"
    search_path = "data/search/sample_0000_search.png"
    manifest_path = "data/dataset_manifest.csv"

    df = pd.read_csv(manifest_path)
    gt_row = df.iloc[0]
    gt_x = float(gt_row["ground_truth_x"])
    gt_y = float(gt_row["ground_truth_y"])
    gt_scale = float(gt_row["scale_ratio"])
    print(f"Ground Truth: ({gt_x}, {gt_y}), Scale: {gt_scale}")

    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    # 1. Downsample reference to nominal scale (scale 10.0 -> 100x100)
    target_size = int(round(1000 / gt_scale))
    ref_scaled = cv2.resize(ref, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # 2. Compute NCC map
    res_ncc = cv2.matchTemplate(search, ref_scaled, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_ncc)
    
    # max_loc in matchTemplate corresponds to the top-left corner of the template
    # Center in search image coordinates:
    max_center_x = max_loc[0] + (target_size / 2.0)
    max_center_y = max_loc[1] + (target_size / 2.0)

    print(f"Global NCC Max: {max_val:.4f} at top-left {max_loc}, center ({max_center_x:.2f}, {max_center_y:.2f})")
    error = np.sqrt((max_center_x - gt_x)**2 + (max_center_y - gt_y)**2)
    print(f"Error of Global Max vs GT: {error:.4f} pixels")

    # 3. Gradient correlation
    # Sobel gradients
    def get_gradient_mag(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    grad_ref = get_gradient_mag(ref_scaled)
    grad_search = get_gradient_mag(search)
    res_grad = cv2.matchTemplate(grad_search, grad_ref, cv2.TM_CCOEFF_NORMED)
    
    min_v_g, max_v_g, min_l_g, max_l_g = cv2.minMaxLoc(res_grad)
    grad_center_x = max_l_g[0] + (target_size / 2.0)
    grad_center_y = max_l_g[1] + (target_size / 2.0)
    print(f"Gradient NCC Max: {max_v_g:.4f} at center ({grad_center_x:.2f}, {grad_center_y:.2f})")
    error_grad = np.sqrt((grad_center_x - gt_x)**2 + (grad_center_y - gt_y)**2)
    print(f"Error of Gradient Max vs GT: {error_grad:.4f} pixels")


if __name__ == "__main__":
    analyze_sample_0()
