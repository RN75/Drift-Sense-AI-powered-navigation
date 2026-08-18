import cv2
import numpy as np

REF = "data_robust/reference/sample_0000_ref.png"
SEARCH = "data_robust/search/sample_0000_search.png"

# Ground truth from dataset
GT_X = 640.4
GT_Y = 481.0

ref = cv2.imread(REF, cv2.IMREAD_GRAYSCALE)
search = cv2.imread(SEARCH, cv2.IMREAD_GRAYSCALE)

print("Reference:", ref.shape)
print("Search:", search.shape)

# Reference is 1000x1000 at high resolution.
# At 10x scale it should become ~100x100 in search image.
ref_small = cv2.resize(
    ref,
    (100, 100),
    interpolation=cv2.INTER_AREA
)

# Ground-truth top-left position
x0 = int(round(GT_X - 50))
y0 = int(round(GT_Y - 50))

gt_patch = search[y0:y0+100, x0:x0+100]

print("GT patch:", gt_patch.shape)
print("GT location:", x0, y0)

# Correlation at true location
score_gt = cv2.matchTemplate(
    gt_patch,
    ref_small,
    cv2.TM_CCOEFF_NORMED
)[0, 0]

print(f"\nCorrelation at TRUE location: {score_gt:.6f}")

# Search entire image using same template
corr = cv2.matchTemplate(
    search,
    ref_small,
    cv2.TM_CCOEFF_NORMED
)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)

print(f"Best correlation anywhere: {max_val:.6f}")
print(f"Best location top-left: {max_loc}")

pred_x = max_loc[0] + 50
pred_y = max_loc[1] + 50

print(f"Best predicted center: ({pred_x:.2f}, {pred_y:.2f})")

# Difference between actual GT patch and reference
diff = cv2.absdiff(gt_patch, ref_small)

print("\nGT patch statistics:")
print("Reference mean:", np.mean(ref_small))
print("GT patch mean:", np.mean(gt_patch))
print("Mean absolute difference:", np.mean(diff))