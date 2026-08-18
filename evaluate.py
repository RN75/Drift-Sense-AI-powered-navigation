"""
Applied Materials Drift-Sense Hackathon 2026
Automated Evaluation & Benchmarking Suite (evaluate.py)
"""

import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from localize import locate_pattern


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Drift-Sense Localization Pipeline")
    parser.add_argument("--manifest", type=str, default="data_robust/dataset_manifest.csv", help="Path to dataset manifest")
    parser.add_argument("--visualize", action="store_true", help="Save diagnostic comparison images")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory to save evaluation results")
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, candidates: list) -> str:
    """Finds the matching column name regardless of format."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in manifest columns: {list(df.columns)}")


def resolve_image_path(path_str: str, manifest_dir: Path) -> Path:
    """Resolves image paths whether relative to workspace or manifest directory."""
    p = Path(path_str)
    if p.exists():
        return p
    # Check relative to manifest directory
    alt_p = manifest_dir / p
    if alt_p.exists():
        return alt_p
    # Check if folder name is doubled or missing
    alt_p2 = manifest_dir.parent / p
    if alt_p2.exists():
        return alt_p2
    return p


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest_dir = manifest_path.parent
    df = pd.read_csv(manifest_path)
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Resolve manifest column names
    ref_col = resolve_column(df, ["reference_path", "ref_path", "reference"])
    search_col = resolve_column(df, ["search_path", "search"])
    gt_x_col = resolve_column(df, ["ground_truth_x", "gt_x", "target_x", "center_x"])
    gt_y_col = resolve_column(df, ["ground_truth_y", "gt_y", "target_y", "center_y"])
    id_col = resolve_column(df, ["sample_id", "id", "index"]) if any(c in df.columns for c in ["sample_id", "id", "index"]) else None

    results = []
    print(f"[*] Benchmarking {len(df)} samples from '{manifest_path}'...\n")

    for idx, row in df.iterrows():
        sample_id = str(row[id_col]) if id_col else f"{idx:04d}"
        
        # Safely resolve file paths
        ref_path = resolve_image_path(str(row[ref_col]), manifest_dir)
        search_path = resolve_image_path(str(row[search_col]), manifest_dir)
        
        gt_x = float(row[gt_x_col])
        gt_y = float(row[gt_y_col])

        t0 = time.perf_counter()
        pred_x, pred_y = locate_pattern(str(search_path), str(ref_path))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        err_x = pred_x - gt_x
        err_y = pred_y - gt_y
        euclidean_err = float(np.sqrt(err_x**2 + err_y**2))

        scale_val = float(row.get("scale_ratio", row.get("scale", 10.0)))
        rot_val = float(row.get("rotation_degrees", row.get("rotation", 0.0)))

        results.append({
            "sample_id": sample_id,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error_x": err_x,
            "error_y": err_y,
            "euclidean_error_px": euclidean_err,
            "scale_ratio": scale_val,
            "rotation_degrees": rot_val,
            "latency_ms": latency_ms,
        })

        print(f"Sample {sample_id:>4} | Scale: {scale_val:5.2f}x | Rot: {rot_val:+5.2f}° | Error: {euclidean_err:6.3f} px | {latency_ms:5.1f} ms")

        if args.visualize:
            search_bgr = cv2.imread(str(search_path))
            if search_bgr is not None:
                # Green circle = Ground Truth
                cv2.circle(search_bgr, (int(round(gt_x)), int(round(gt_y))), 9, (0, 255, 0), 2)
                # Red cross = Prediction
                cv2.drawMarker(
                    search_bgr,
                    (int(round(pred_x)), int(round(pred_y))),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=22,
                    thickness=2
                )
                cv2.putText(
                    search_bgr,
                    f"Err: {euclidean_err:.2f}px ({latency_ms:.0f}ms)",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                vis_file = vis_dir / f"eval_{sample_id}.png"
                cv2.imwrite(str(vis_file), search_bgr)

    res_df = pd.DataFrame(results)
    res_csv = out_dir / "localization_results.csv"
    res_df.to_csv(res_csv, index=False)

    # Summary Statistics
    errors = res_df["euclidean_error_px"].to_numpy()
    mean_err = np.mean(errors)
    med_err = np.median(errors)
    std_err = np.std(errors)
    max_err = np.max(errors)
    p90_err = np.percentile(errors, 90)
    pct_1px = np.mean(errors < 1.0) * 100.0
    pct_2px = np.mean(errors < 2.0) * 100.0
    pct_5px = np.mean(errors < 5.0) * 100.0
    avg_latency = res_df["latency_ms"].mean()

    print("\n" + "=" * 65)
    print("               DRIFT-SENSE BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Samples Evaluated: {len(res_df)}")
    print(f"Mean Euclidean Error:    {mean_err:.3f} px")
    print(f"Median Euclidean Error:  {med_err:.3f} px")
    print(f"Std Deviation:           {std_err:.3f} px")
    print(f"90th Percentile Error:   {p90_err:.3f} px")
    print(f"Max Error:               {max_err:.3f} px")
    print("-" * 65)
    print(f"Accuracy < 1.0 px:       {pct_1px:.1f}%")
    print(f"Accuracy < 2.0 px:       {pct_2px:.1f}%")
    print(f"Accuracy < 5.0 px:       {pct_5px:.1f}%")
    print(f"Mean Latency per Image:  {avg_latency:.2f} ms")
    print("=" * 65)
    print(f"Metrics saved to: {res_csv.resolve()}\n")


if __name__ == "__main__":
    main()