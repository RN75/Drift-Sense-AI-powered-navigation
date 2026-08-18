"""
check_sample1_gt.py
Check correlation map values around the ground truth in Sample 1.
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def check_gt():
    ref = cv2.imread("test_data/reference/sample_0001_ref.png", cv2.IMREAD_GRAYSCALE)
    search = cv2.imread("test_data/search/sample_0001_search.png", cv2.IMREAD_GRAYSCALE)
    gt_x, gt_y = 879.3, 758.2

    ref_smooth = cv2.GaussianBlur(ref, (3, 3), 0.7)
    search_smooth = cv2.GaussianBlur(search, (3, 3), 0.7)

    tmpl = cv2.resize(ref_smooth, (100, 100), interpolation=cv2.INTER_AREA)
    corr = cv2.matchTemplate(search_smooth, tmpl, cv2.TM_CCOEFF_NORMED)

    # Value at GT
    gt_tl_x = int(round(gt_x - 50)) # 829
    gt_tl_y = int(round(gt_y - 50)) # 708
    val_at_gt = corr[gt_tl_y, gt_tl_x]
    print(f"Correlation at GT (tl={gt_tl_x}, {gt_tl_y}): {val_at_gt:.4f}")

    # Value at periodic block (tl = 829 - 258 = 571, 708)
    cand1_tl_x = int(round(624.0 - 50)) # 574
    val_at_cand1 = corr[gt_tl_y, cand1_tl_x]
    print(f"Correlation at Cand 1 (tl={cand1_tl_x}, {gt_tl_y}): {val_at_cand1:.4f}")

    # Let's inspect the entire row around Y = 708
    row_corr = corr[gt_tl_y, :]
    peaks_x = np.where((row_corr[1:-1] > row_corr[:-2]) & (row_corr[1:-1] > row_corr[2:]) & (row_corr[1:-1] > 0.65))[0] + 1
    print("Peaks along row Y=708:")
    for px in peaks_x:
        print(f"  tl_x = {px} (center x = {px + 50}) : corr = {row_corr[px]:.4f}")


if __name__ == "__main__":
    check_gt()
