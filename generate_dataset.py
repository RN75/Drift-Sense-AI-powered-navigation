"""
generate_dataset.py
Drift-Sense synthetic semiconductor dataset generator.

Generates:
    - High-resolution synthetic DRAM layout
    - Clean reference image
    - SEM-degraded search image
    - Controlled scale variation
    - REAL rotation applied to the search image
    - Exact transformed ground-truth center
    - Diagnostic visualization
    - CSV dataset manifest
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Allow imports from src/
# ------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.synthetic_dram import (
    DRAMLayoutGenerator,
    extract_reference_and_search
)

from src.sem_noise import apply_sem_degradation


# ============================================================
# ROTATE IMAGE + TRANSFORM GROUND TRUTH
# ============================================================

def rotate_search_and_gt(
    image: np.ndarray,
    gt_x: float,
    gt_y: float,
    angle_degrees: float
):
    """
    Rotates the complete search image around its center and
    transforms the ground-truth target center accordingly.

    This makes --rotation a REAL physical image transformation
    instead of only storing the angle in the CSV.
    """

    h, w = image.shape[:2]

    center = (w / 2.0, h / 2.0)

    # OpenCV rotation matrix
    M = cv2.getRotationMatrix2D(
        center,
        angle_degrees,
        1.0
    )

    # Rotate image
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # Transform ground-truth center
    point = np.array(
        [gt_x, gt_y, 1.0],
        dtype=np.float64
    )

    transformed = M @ point

    new_gt_x = float(transformed[0])
    new_gt_y = float(transformed[1])

    return rotated, new_gt_x, new_gt_y


# ============================================================
# GENERATE ONE SAMPLE
# ============================================================

def generate_sample(
    generator: DRAMLayoutGenerator,
    sample_idx: int,
    output_dir: str,
    scale_ratio: float = 10.0,
    rotation_degrees: float = 0.0,
    seed: int = 42,
    save_diagnostic: bool = True,
    canvas_size: int = 12000
) -> dict:

    """
    Generate one Reference/Search image pair.

    Reference:
        1000 x 1000 pixels
        1 nm/pixel

    Search:
        1000 x 1000 pixels
        scale-dependent field
        SEM degradation applied
    """

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    ref_dir = os.path.join(
        output_dir,
        "reference"
    )

    search_dir = os.path.join(
        output_dir,
        "search"
    )

    diag_dir = os.path.join(
        output_dir,
        "diagnostics"
    )

    os.makedirs(
        ref_dir,
        exist_ok=True
    )

    os.makedirs(
        search_dir,
        exist_ok=True
    )

    if save_diagnostic:
        os.makedirs(
            diag_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Independent deterministic seed per sample
    # --------------------------------------------------------

    sample_seed = seed + sample_idx

    # --------------------------------------------------------
    # 1. Generate synthetic DRAM wafer canvas
    # --------------------------------------------------------

    canvas = generator.generate_canvas(
        width=canvas_size,
        height=canvas_size,
        seed=sample_seed
    )

    # --------------------------------------------------------
    # 2. Extract reference + search field
    # --------------------------------------------------------

    (
        ref_raw,
        search_clean,
        gt_x,
        gt_y,
        metadata
    ) = extract_reference_and_search(

        canvas=canvas,

        scale_ratio=scale_ratio,

        ref_size=(1000, 1000),

        search_size=(1000, 1000),

        # Keep target comfortably away from image boundary
        margin=600,

        seed=sample_seed + 1000
    )

    # --------------------------------------------------------
    # 3. Reference image
    # --------------------------------------------------------

    ref_img = np.clip(
        ref_raw,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 4. SEM degradation on search image
    # --------------------------------------------------------

    search_img, noise_meta = apply_sem_degradation(

        search_clean,

        # Electron shot noise
        poisson_peak=130.0,

        # Detector readout noise
        gaussian_noise_std=5.5,

        # Beam PSF
        blur_sigma=0.8,

        # Edge charging
        edge_charge_strength=0.35,

        # Rare detector spikes
        salt_pepper_prob=0.0008,

        # Contrast
        contrast_factor=1.05,

        # Brightness
        brightness_offset=-4.0,

        # Detector gamma
        gamma=1.04,

        # Scan-line drift
        streak_strength=2.5,

        seed=sample_seed + 2000
    )

    # --------------------------------------------------------
    # 5. APPLY REAL ROTATION
    # --------------------------------------------------------

    if abs(rotation_degrees) > 1e-6:

        (
            search_img,
            gt_x,
            gt_y
        ) = rotate_search_and_gt(
            search_img,
            gt_x,
            gt_y,
            rotation_degrees
        )

    # --------------------------------------------------------
    # Safety clipping after rotation
    # --------------------------------------------------------

    search_img = np.clip(
        search_img,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 6. Keep GT inside image
    # --------------------------------------------------------

    gt_x = float(
        np.clip(
            gt_x,
            0.0,
            search_img.shape[1] - 1.0
        )
    )

    gt_y = float(
        np.clip(
            gt_y,
            0.0,
            search_img.shape[0] - 1.0
        )
    )

    # --------------------------------------------------------
    # 7. File names
    # --------------------------------------------------------

    ref_filename = (
        f"sample_{sample_idx:04d}_ref.png"
    )

    search_filename = (
        f"sample_{sample_idx:04d}_search.png"
    )

    diag_filename = (
        f"sample_{sample_idx:04d}_diagnostic.png"
    )

    ref_path = os.path.join(
        ref_dir,
        ref_filename
    )

    search_path = os.path.join(
        search_dir,
        search_filename
    )

    diag_path = os.path.join(
        diag_dir,
        diag_filename
    )

    # --------------------------------------------------------
    # 8. Save images
    # --------------------------------------------------------

    cv2.imwrite(
        ref_path,
        ref_img
    )

    cv2.imwrite(
        search_path,
        search_img
    )

    # ========================================================
    # 9. Diagnostic visualization
    # ========================================================

    if save_diagnostic:

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6),
            dpi=150
        )

        # ----------------------------------------------------
        # Reference
        # ----------------------------------------------------

        axes[0].imshow(
            ref_img,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[0].set_title(
            "Reference Image\n"
            "1000 x 1000 | 1 nm/pixel",
            fontsize=11,
            fontweight="bold"
        )

        axes[0].set_xlabel(
            "X (pixels / nm)"
        )

        axes[0].set_ylabel(
            "Y (pixels / nm)"
        )

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        axes[1].imshow(
            search_img,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[1].set_title(
            f"SEM Search Image\n"
            f"Scale = {scale_ratio:.2f}x | "
            f"Rotation = {rotation_degrees:+.2f}°",
            fontsize=11,
            fontweight="bold"
        )

        axes[1].set_xlabel(
            "X (search pixels)"
        )

        axes[1].set_ylabel(
            "Y (search pixels)"
        )

        # ----------------------------------------------------
        # Search + GT
        # ----------------------------------------------------

        axes[2].imshow(
            search_img,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        # Target dimensions in search coordinates
        target_w = 1000.0 / scale_ratio
        target_h = 1000.0 / scale_ratio

        box_x = gt_x - target_w / 2.0
        box_y = gt_y - target_h / 2.0

        rect = plt.Rectangle(
            (box_x, box_y),
            target_w,
            target_h,
            linewidth=2.0,
            edgecolor="cyan",
            facecolor="none",
            linestyle="--"
        )

        axes[2].add_patch(rect)

        # GT center
        axes[2].plot(
            gt_x,
            gt_y,
            marker="+",
            markersize=16,
            markeredgewidth=2.5,
            color="red"
        )

        axes[2].plot(
            gt_x,
            gt_y,
            marker="o",
            markersize=7,
            markeredgewidth=1.5,
            color="yellow",
            fillstyle="none"
        )

        axes[2].set_title(
            f"Ground Truth\n"
            f"Center = ({gt_x:.2f}, {gt_y:.2f})",
            fontsize=11,
            fontweight="bold"
        )

        axes[2].set_xlabel(
            "X (search pixels)"
        )

        axes[2].set_ylabel(
            "Y (search pixels)"
        )

        plt.tight_layout()

        plt.savefig(
            diag_path,
            bbox_inches="tight"
        )

        plt.close(fig)

    # ========================================================
    # 10. Dataset manifest record
    # ========================================================

    record = {

        "sample_id":
            sample_idx,

        "reference_path":
            os.path.relpath(
                ref_path,
                output_dir
            ).replace("\\", "/"),

        "search_path":
            os.path.relpath(
                search_path,
                output_dir
            ).replace("\\", "/"),

        "ground_truth_x":
            round(float(gt_x), 4),

        "ground_truth_y":
            round(float(gt_y), 4),

        "scale_ratio":
            float(scale_ratio),

        "rotation_degrees":
            float(rotation_degrees),

        "random_seed":
            sample_seed,

        "ref_width":
            ref_img.shape[1],

        "ref_height":
            ref_img.shape[0],

        "search_width":
            search_img.shape[1],

        "search_height":
            search_img.shape[0],

        **noise_meta
    }

    # ========================================================
    # 11. Console report
    # ========================================================

    print(
        "=================================================="
    )

    print(
        f"SAMPLE {sample_idx:04d} GENERATION REPORT"
    )

    print(
        "=================================================="
    )

    print(
        f"Reference shape         : "
        f"{ref_img.shape} "
        f"(dtype: {ref_img.dtype})"
    )

    print(
        f"Search shape            : "
        f"{search_img.shape} "
        f"(dtype: {search_img.dtype})"
    )

    print(
        f"Scale ratio             : "
        f"{scale_ratio:.4f}"
    )

    print(
        f"Rotation (degrees)      : "
        f"{rotation_degrees:+.2f} deg"
    )

    print(
        f"Ground-truth center     : "
        f"(x={gt_x:.4f}, y={gt_y:.4f})"
    )

    print(
        f"Reference min/max/mean/std : "
        f"min={ref_img.min()}, "
        f"max={ref_img.max()}, "
        f"mean={ref_img.mean():.2f}, "
        f"std={ref_img.std():.2f}"
    )

    print(
        f"Search min/max/mean/std    : "
        f"min={search_img.min()}, "
        f"max={search_img.max()}, "
        f"mean={search_img.mean():.2f}, "
        f"std={search_img.std():.2f}"
    )

    print(
        f"Reference path          : "
        f"{ref_path}"
    )

    print(
        f"Search path             : "
        f"{search_path}"
    )

    if save_diagnostic:

        print(
            f"Diagnostic image saved  : "
            f"{diag_path}"
        )

    print(
        "==================================================\n"
    )

    return record


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic semiconductor "
            "dataset for DriftSense."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed"
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Fixed scale ratio"
    )

    parser.add_argument(
        "--rotation",
        type=float,
        default=0.0,
        help="Fixed rotation in degrees"
    )

    # --------------------------------------------------------
    # NEW: random scale
    # --------------------------------------------------------

    parser.add_argument(
        "--random-scale",
        action="store_true",
        help="Random scale uniformly between 9x and 11x"
    )

    # --------------------------------------------------------
    # NEW: random rotation
    # --------------------------------------------------------

    parser.add_argument(
        "--random-rotation",
        action="store_true",
        help="Random rotation uniformly between -2 and +2 degrees"
    )

    parser.add_argument(
        "--canvas-size",
        type=int,
        default=12000,
        help="Underlying wafer canvas dimension"
    )

    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable diagnostic images"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        args.output,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DRAM generator
    # --------------------------------------------------------

    generator = DRAMLayoutGenerator()

    # Use independent RNG for scale/rotation
    rng = np.random.default_rng(
        args.seed
    )

    records = []

    print(
        f"\n[DriftSense] Generating "
        f"{args.samples} synthetic dataset sample(s) "
        f"with seed {args.seed}..."
    )

    # ========================================================
    # Generate samples
    # ========================================================

    for i in range(args.samples):

        # ----------------------------------------------------
        # Determine scale
        # ----------------------------------------------------

        if args.random_scale:

            sample_scale = float(
                rng.uniform(
                    9.0,
                    11.0
                )
            )

        else:

            sample_scale = float(
                args.scale
            )

        # ----------------------------------------------------
        # Determine rotation
        # ----------------------------------------------------

        if args.random_rotation:

            sample_rotation = float(
                rng.uniform(
                    -2.0,
                    2.0
                )
            )

        else:

            sample_rotation = float(
                args.rotation
            )

        # ----------------------------------------------------
        # Generate sample
        # ----------------------------------------------------

        record = generate_sample(

            generator=generator,

            sample_idx=i,

            output_dir=args.output,

            scale_ratio=sample_scale,

            rotation_degrees=sample_rotation,

            seed=args.seed,

            save_diagnostic=(
                not args.no_visualize
            ),

            canvas_size=args.canvas_size
        )

        records.append(record)

    # ========================================================
    # Save manifest
    # ========================================================

    manifest_df = pd.DataFrame(
        records
    )

    manifest_path = os.path.join(
        args.output,
        "dataset_manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False
    )

    print(
        f"\n[DriftSense] Dataset manifest "
        f"successfully saved to: "
        f"{manifest_path}"
    )

    print(
        "[DriftSense] Generation completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()