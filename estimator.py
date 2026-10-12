"""Reusable pH estimation pipeline (no plotting, no file paths)."""
import cv2
import numpy as np
import detection as detect
from prediction import predict_ph_value_with_distance

NUMBER_CLUSTERS = 6


def _bar(color, size=120):
    bar = np.zeros((size, size, 3), np.uint8)
    bar[:] = color
    return bar


def estimate_ph(img_bgr, n_neighbors=3):
    roi = detect.roi_from_bgr(img_bgr)
    h, w, _ = roi.shape
    if h * w < NUMBER_CLUSTERS:
        raise ValueError(
            "The detected area is too small to analyze. Use a clear photo with the "
            "entire test strip visible, then try again."
        )
    data = np.float32(roi.reshape(h * w, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(data, NUMBER_CLUSTERS, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Drop the 2 least colourful clusters (background / white / grey)
    spread = centers.max(axis=1) - centers.min(axis=1)
    drop = set(np.argpartition(spread, 2)[:2])
    kept = [c for i, c in enumerate(centers) if i not in drop]

    # centers are already RGB (ROI was converted to RGB)
    rgb_values = [[int(v) for v in c] for c in kept]
    ph, dist, sequence = predict_ph_value_with_distance(rgb_values, n_neighbors=n_neighbors)

    return {
        "ph": float(ph),
        "distance": float(dist),
        "roi": roi,
        "all_colors": np.hstack([_bar(c) for c in centers.astype(np.uint8)]),
        "dominant_colors": np.hstack([_bar(c) for c in np.array(kept).astype(np.uint8)]),
        "matched_sequence": np.hstack([_bar(np.array(c, dtype=np.uint8)) for c in sequence]),
    }
