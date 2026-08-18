"""
test_gradient_order.py
Compare gradient-first vs downsample-first matching on Sample 1.
"""

import cv2
import numpy as np


def test():
    ref = cv2.imread("test_data/reference/sample_0001_ref.png", cv2.IMREAD_GRAYSCALE)
    search = cv2.imread("test_data/search/sample_0001_search.png", cv2.IMREAD_GRAYSCALE)
    gt_x, gt_y = 879.3, 758.2

    ref_smooth = cv2.GaussianBlur(ref, (3, 3), 0.7)
    search_smooth = cv2.GaussianBlur(search, (3, 3), 0.7)

    # 1. Downsample reference FIRST
    tmpl_int = cv2.resize(ref_smooth, (100, 100), interpolation=cv2.INTER_AREA)

    # Gradient of downsampled template
    gx_t = cv2.Sobel(tmpl_int, cv2.CV_32F, 1, 0, ksize=3)
    gy_t = cv2.Sobel(tmpl_int, cv2.CV_32F, 0, 1, ksize=3)
    mag_t = cv2.magnitude(gx_t, gy_t)
    mag_t_norm = cv2.normalize(mag_t, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Gradient of search
    gx_s = cv2.Sobel(search_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy_s = cv2.Sobel(search_smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag_s = cv2.magnitude(gx_s, gy_s)
    mag_s_norm = cv2.normalize(mag_s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    res_int = cv2.matchTemplate(search_smooth, tmpl_int, cv2.TM_CCOEFF_NORMED)
    res_grad = cv2.matchTemplate(mag_s_norm, mag_t_norm, cv2.TM_CCOEFF_NORMED)

    res_comb = 0.60 * res_int + 0.40 * res_grad

    # Check top peak
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(res_comb)
    cx = max_l[0] + 50.0
    cy = max_l[1] + 50.0

    print(f"Top Combined Peak: {max_v:.4f} at center ({cx:.2f}, {cy:.2f})")
    print(f"Ground Truth: ({gt_x:.2f}, {gt_y:.2f})")
    print(f"Error: {np.sqrt((cx - gt_x)**2 + (cy - gt_y)**2):.4f} pixels")


if __name__ == "__main__":
    test()
