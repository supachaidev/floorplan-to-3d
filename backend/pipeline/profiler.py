"""Image profiler for categorizing floorplan input quality.

Classifies images into categories (clean_digital, low_contrast, hand_drawn,
photo) using OpenCV heuristics to help diagnose where detection fails.
"""

import cv2
import numpy as np


def profile_image(image: np.ndarray) -> dict:
    """Profile a floorplan image and classify its quality category.

    Args:
        image: BGR image (raw, before deskew).

    Returns:
        {
          "category": "clean_digital" | "low_contrast" | "hand_drawn" | "photo",
          "contrast_score": float,      # std dev of grayscale pixels (0-255)
          "line_straightness": float,   # 0.0-1.0, higher = straighter
          "edge_regularity": float,     # 0.0-1.0, higher = more uniform edges
          "skew_angle_deg": float       # residual skew after deskew()
        }
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contrast_score = _compute_contrast(gray)
    line_straightness = _compute_line_straightness(gray)
    edge_regularity = _compute_edge_regularity(gray)
    skew_angle_deg = _compute_skew_angle(gray)

    # Category assignment (checked in priority order)
    # Thresholds calibrated on CubiCasa5K: edge_regularity ranges ~0.006-0.13
    if contrast_score < 40:
        category = "low_contrast"
    elif edge_regularity < 0.02:
        category = "hand_drawn"
    elif skew_angle_deg > 3.0:
        category = "photo"
    else:
        category = "clean_digital"

    return {
        "category": category,
        "contrast_score": round(float(contrast_score), 2),
        "line_straightness": round(float(line_straightness), 4),
        "edge_regularity": round(float(edge_regularity), 4),
        "skew_angle_deg": round(float(skew_angle_deg), 2),
    }


def _compute_contrast(gray: np.ndarray) -> float:
    """Standard deviation of grayscale pixel values."""
    return float(np.std(gray))


def _compute_line_straightness(gray: np.ndarray) -> float:
    """Ratio of HoughLinesP-detected line pixels to total Canny edge pixels."""
    edges = cv2.Canny(gray, 50, 150)
    total_edge_pixels = np.count_nonzero(edges)
    if total_edge_pixels == 0:
        return 0.0

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=30, maxLineGap=10,
    )
    if lines is None:
        return 0.0

    # Count pixels on detected straight lines
    line_mask = np.zeros_like(edges)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

    line_pixels = np.count_nonzero(line_mask)
    ratio = line_pixels / total_edge_pixels
    return min(ratio, 1.0)


def _compute_edge_regularity(gray: np.ndarray) -> float:
    """1 minus normalized variance of local edge gradient orientations.

    High regularity = edges have consistent directions (typical of CAD/digital).
    Low regularity = noisy, varied edge directions (typical of hand-drawn).
    """
    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute orientations only where edges are significant
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    mag_threshold = np.percentile(magnitude, 90)
    mask = magnitude > mag_threshold

    if mask.sum() < 100:
        return 0.0

    orientations = np.arctan2(gy[mask], gx[mask])

    # Quantize orientations into a histogram (36 bins, 10 degrees each)
    hist, _ = np.histogram(orientations, bins=36, range=(-np.pi, np.pi))
    hist = hist.astype(np.float64)

    # Normalize histogram
    hist_sum = hist.sum()
    if hist_sum == 0:
        return 0.0
    hist_norm = hist / hist_sum

    # Regularity: inverse of entropy-like variance
    # Uniform distribution (all random) → high variance → low regularity
    # Peaked distribution (few dominant angles) → low variance → high regularity
    variance = np.var(hist_norm)
    # Normalize: max possible variance of a 36-bin distribution is when
    # all mass is in one bin: var = (1/36)*((1 - 1/36)^2 + 35*(0 - 1/36)^2)
    max_var = (1.0 / 36) * ((1 - 1.0 / 36) ** 2 + 35 * (0 - 1.0 / 36) ** 2)

    regularity = variance / max_var if max_var > 0 else 0.0
    return min(regularity, 1.0)


def _compute_skew_angle(gray: np.ndarray) -> float:
    """Residual rotation angle detected via minAreaRect on thresholded content."""
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 50:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle to [-45, 45] range
    if angle > 45:
        angle = angle - 90
    elif angle < -45:
        angle = angle + 90

    return abs(angle)
