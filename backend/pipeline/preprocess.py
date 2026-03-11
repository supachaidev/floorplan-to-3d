import cv2
import numpy as np


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct perspective distortion and deskew the floor plan image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize contrast for varying image quality
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    # If we found a quadrilateral, do perspective correction
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        rect = order_points(pts)
        w = max(
            np.linalg.norm(rect[0] - rect[1]),
            np.linalg.norm(rect[2] - rect[3]),
        )
        h = max(
            np.linalg.norm(rect[0] - rect[3]),
            np.linalg.norm(rect[1] - rect[2]),
        )
        # Guard against degenerate quadrilaterals
        if w >= 2 and h >= 2:
            dst = np.array(
                [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
            )
            M = cv2.getPerspectiveTransform(rect, dst)
            image = cv2.warpPerspective(image, M, (int(w), int(h)))

    # Deskew via minimum area rectangle angle
    gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(thresh2 > 0))
    if len(coords) > 50:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # OpenCV 4.5.1+ returns angles in [0, 90).
        # Older versions return in [-90, 0).
        # Normalize: we want the small rotation needed to straighten.
        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90

        angle = -angle  # negate for correction direction

        if abs(angle) > 0.5:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    return image


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect
