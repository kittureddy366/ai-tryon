import cv2
import numpy as np

from utils.math_utils import clamp


def gaussian_feather_alpha(alpha_map, ksize=15, sigma=4.0):
    ksize = max(3, int(ksize) | 1)
    blurred = cv2.GaussianBlur(alpha_map.astype(np.float32), (ksize, ksize), sigma)
    return np.clip(blurred, 0.0, 1.0)


def adapt_garment_lighting(garment_rgba, frame_bgr, torso_quad):
    output = garment_rgba.copy()
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, torso_quad.astype(np.int32), 255)

    frame_l = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    region = frame_l[mask > 0]
    if region.size == 0:
        return output

    target_luma = float(np.mean(region)) / 255.0
    gain = clamp(0.85 + target_luma * 0.4, 0.72, 1.20)

    color = output[:, :, :3].astype(np.float32)
    color = np.clip(color * gain, 0, 255)

    # Slight contrast normalization to avoid flat pasted look under varied lighting.
    mean = np.mean(color, axis=(0, 1), keepdims=True)
    color = np.clip((color - mean) * 1.03 + mean, 0, 255)
    output[:, :, :3] = color.astype(np.uint8)
    return output
