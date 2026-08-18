"""
inspect_sample_candidates.py

Detailed inspection of candidate peaks, scale/rotation consistency,
and score landscape for a selected sample.
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


DATASET_DIR = "data_new_combined"


def inspect_sample(sample_id: int):

    # ---------------------------------------------------------
    # File paths
    # ---------------------------------------------------------
    ref_path = os.path.join(
        DATASET_DIR,
        "reference",
        f"sample_{sample_id:04d}_ref.png"
    )

    search_path = os.path.join(
        DATASET_DIR,
        "search",
        f"sample_{sample_id:04d}_search.png"
    )

    manifest_path = os.path.join(
        DATASET_DIR,
        "dataset_manifest.csv"
    )

    # ---------------------------------------------------------
    # Check files
    # ---------------------------------------------------------
    if not os.path.exists(ref_path):
        print(f"[ERROR] Reference image not found: {ref_path}")
        return

    if not os.path.exists(search_path):
        print(f"[ERROR] Search image not found: {search_path}")
        return

    # ---------------------------------------------------------
    # Load images
    # ---------------------------------------------------------
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    if ref is None:
        print(f"[ERROR] Could not read reference image: {ref_path}")
        return

    if search is None:
        print(f"[ERROR] Could not read search image: {search_path}")
        return

    # ---------------------------------------------------------
    # Read ground truth
    # ---------------------------------------------------------
    gt_x = None
    gt_y = None

    if os.path.exists(manifest_path):

        df = pd.read_csv(manifest_path)

        # Try to find the appropriate row
        row = df.iloc[sample_id]

        possible_x = [
            "gt_x",
            "ground_truth_x",
            "center_x",
            "true_x"
        ]

        possible_y = [
            "gt_y",
            "ground_truth_y",
            "center_y",
            "true_y"
        ]

        for col in possible_x:
            if col in df.columns:
                gt_x = float(row[col])
                break

        for col in possible_y:
            if col in df.columns:
                gt_y = float(row[col])
                break

    print("\n" + "=" * 70)
    print(f"INSPECTING SAMPLE {sample_id}")
    print("=" * 70)

    print(f"Reference : {ref_path}")
    print(f"Search    : {search_path}")

    if gt_x is not None and gt_y is not None:
        print(
            f"Ground Truth Center : "
            f"({gt_x:.3f}, {gt_y:.3f})"
        )
    else:
        print("Ground Truth Center : NOT FOUND IN MANIFEST")

    print(f"Reference shape : {ref.shape}")
    print(f"Search shape    : {search.shape}")

    # ---------------------------------------------------------
    # Search parameters
    # ---------------------------------------------------------
    scales = np.linspace(9.0, 11.0, 21)

    rotations = np.array([
        -2.0,
        -1.0,
         0.0,
         1.0,
         2.0
    ])

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    ref_smooth = cv2.GaussianBlur(
        ref,
        (3, 3),
        0.7
    )

    search_smooth = cv2.GaussianBlur(
        search,
        (3, 3),
        0.7
    )

    # ---------------------------------------------------------
    # Reference gradient
    # ---------------------------------------------------------
    ref_gx = cv2.Sobel(
        ref_smooth,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    ref_gy = cv2.Sobel(
        ref_smooth,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    ref_g = cv2.normalize(
        cv2.magnitude(ref_gx, ref_gy),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # ---------------------------------------------------------
    # Search gradient
    # ---------------------------------------------------------
    search_gx = cv2.Sobel(
        search_smooth,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    search_gy = cv2.Sobel(
        search_smooth,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    search_g = cv2.normalize(
        cv2.magnitude(search_gx, search_gy),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # ---------------------------------------------------------
    # Candidate list
    # ---------------------------------------------------------
    candidates = []

    # ---------------------------------------------------------
    # Scale + rotation search
    # ---------------------------------------------------------
    for s in scales:

        tmpl_size = int(round(1000.0 / s))

        tmpl_int = cv2.resize(
            ref_smooth,
            (tmpl_size, tmpl_size),
            interpolation=cv2.INTER_AREA
        )

        tmpl_g = cv2.resize(
            ref_g,
            (tmpl_size, tmpl_size),
            interpolation=cv2.INTER_AREA
        )

        for rot in rotations:

            if abs(rot) > 1e-4:

                center = (
                    tmpl_size / 2.0,
                    tmpl_size / 2.0
                )

                rot_mat = cv2.getRotationMatrix2D(
                    center,
                    rot,
                    1.0
                )

                tmpl_int_r = cv2.warpAffine(
                    tmpl_int,
                    rot_mat,
                    (tmpl_size, tmpl_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )

                tmpl_g_r = cv2.warpAffine(
                    tmpl_g,
                    rot_mat,
                    (tmpl_size, tmpl_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )

            else:

                tmpl_int_r = tmpl_int
                tmpl_g_r = tmpl_g

            # -------------------------------------------------
            # Intensity matching
            # -------------------------------------------------
            res_int = cv2.matchTemplate(
                search_smooth,
                tmpl_int_r,
                cv2.TM_CCOEFF_NORMED
            )

            # -------------------------------------------------
            # Gradient matching
            # -------------------------------------------------
            res_g = cv2.matchTemplate(
                search_g,
                tmpl_g_r,
                cv2.TM_CCOEFF_NORMED
            )

            # -------------------------------------------------
            # Combined score
            # -------------------------------------------------
            res_comb = (
                0.60 * res_int +
                0.40 * res_g
            )

            # -------------------------------------------------
            # Find local maxima
            # -------------------------------------------------
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (15, 15)
            )

            dilated = cv2.dilate(
                res_comb,
                kernel
            )

            peaks_mask = (
                (res_comb == dilated) &
                (res_comb >= 0.40)
            )

            y_idx, x_idx = np.where(peaks_mask)

            for py, px in zip(y_idx, x_idx):

                score = float(
                    res_comb[py, px]
                )

                score_int = float(
                    res_int[py, px]
                )

                score_g = float(
                    res_g[py, px]
                )

                cx = px + tmpl_size / 2.0
                cy = py + tmpl_size / 2.0

                candidates.append({
                    "cx": cx,
                    "cy": cy,
                    "tl_x": px,
                    "tl_y": py,
                    "score": score,
                    "score_int": score_int,
                    "score_g": score_g,
                    "scale": s,
                    "rot": rot,
                    "tmpl_size": tmpl_size
                })

    # ---------------------------------------------------------
    # Spatial clustering / NMS
    # ---------------------------------------------------------
    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    clusters = []

    cluster_radius = 20.0

    for cand in candidates:

        found_cluster = False

        for cl in clusters:

            dist = np.sqrt(
                (cand["cx"] - cl["cx"]) ** 2 +
                (cand["cy"] - cl["cy"]) ** 2
            )

            if dist < cluster_radius:

                found_cluster = True

                cl["members"].append(cand)

                if cand["score"] > cl["max_score"]:

                    cl["max_score"] = cand["score"]

                    cl["best_cand"] = cand

                break

        if not found_cluster:

            clusters.append({
                "cx": cand["cx"],
                "cy": cand["cy"],
                "max_score": cand["score"],
                "best_cand": cand,
                "members": [cand]
            })

    # ---------------------------------------------------------
    # Cluster scoring
    # ---------------------------------------------------------
    for cl in clusters:

        members = cl["members"]

        scales_in_cluster = set(
            m["scale"] for m in members
        )

        rots_in_cluster = set(
            m["rot"] for m in members
        )

        cl["num_scales"] = len(
            scales_in_cluster
        )

        cl["num_rots"] = len(
            rots_in_cluster
        )

        cl["scale_support"] = (
            len(scales_in_cluster) /
            len(scales)
        )

        cl["rot_support"] = (
            len(rots_in_cluster) /
            len(rotations)
        )

        cl["total_support"] = len(
            members
        )

        cl["composite_score"] = (
            cl["max_score"] *
            (
                0.8 +
                0.1 * cl["scale_support"] +
                0.1 * cl["rot_support"]
            )
        )

    clusters.sort(
        key=lambda cl: cl["composite_score"],
        reverse=True
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------
    print("\n--- TOP 10 CANDIDATE CLUSTERS ---")

    for i, cl in enumerate(clusters[:10]):

        best = cl["best_cand"]

        if gt_x is not None and gt_y is not None:

            dist_gt = np.sqrt(
                (best["cx"] - gt_x) ** 2 +
                (best["cy"] - gt_y) ** 2
            )

        else:

            dist_gt = float("nan")

        print(
            f"\n#{i + 1}"
        )

        print(
            f"Center       = "
            f"({best['cx']:.2f}, {best['cy']:.2f})"
        )

        print(
            f"Score        = "
            f"{cl['max_score']:.4f}"
        )

        print(
            f"Intensity    = "
            f"{best['score_int']:.4f}"
        )

        print(
            f"Gradient     = "
            f"{best['score_g']:.4f}"
        )

        print(
            f"Composite    = "
            f"{cl['composite_score']:.4f}"
        )

        print(
            f"Scale        = "
            f"{best['scale']:.4f}"
        )

        print(
            f"Rotation     = "
            f"{best['rot']:+.2f} deg"
        )

        print(
            f"Scale support = "
            f"{cl['num_scales']}/{len(scales)}"
        )

        print(
            f"Rot support   = "
            f"{cl['num_rots']}/{len(rotations)}"
        )

        if not np.isnan(dist_gt):

            print(
                f"Distance from GT = "
                f"{dist_gt:.2f} px"
            )


if __name__ == "__main__":
    # Improved samples
    inspect_sample(2)
    inspect_sample(3)
    inspect_sample(4)
    # Regressed samples
    inspect_sample(0)
    inspect_sample(6)
    inspect_sample(7)