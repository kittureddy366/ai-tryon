import io
import zipfile

import cv2
import numpy as np


def load_garment_zip_texture(zip_path):
    """
    Load garment textures from a zip archive.
    Prefers diffuse/albedo maps. Returns BGRA texture or None.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            if not names:
                return None

            preferred = []
            fallback = []
            for name in names:
                lname = name.lower()
                if lname.endswith((".png", ".jpg", ".jpeg")):
                    if any(k in lname for k in ("diffuse", "albedo", "basecolor", "base_color", "color")):
                        preferred.append(name)
                    elif "normal" in lname or "roughness" in lname or "metallic" in lname:
                        continue
                    else:
                        fallback.append(name)

            pick = preferred[0] if preferred else (fallback[0] if fallback else None)
            if pick is None:
                return None

            data = zf.read(pick)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                alpha = _estimate_alpha_from_background(img)
                # UV diffuse textures often have no explicit background; keep visible if mask collapses.
                if int(np.count_nonzero(alpha)) < int(alpha.size * 0.02):
                    alpha = np.full(alpha.shape, 255, dtype=np.uint8)
                img = np.dstack([img, alpha])
            elif img.shape[2] == 4:
                img = img.copy()

            return _crop_to_alpha(img)
    except Exception:
        return None


def _estimate_alpha_from_background(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    not_white = gray < 244
    saturated = hsv[:, :, 1] > 18
    fg = ((not_white & (gray < 212)) | (saturated & (gray < 252))).astype(np.uint8) * 255

    flood = fg.copy()
    h, w = fg.shape
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 128)
    cv2.floodFill(flood, mask, (w - 1, 0), 128)
    cv2.floodFill(flood, mask, (0, h - 1), 128)
    cv2.floodFill(flood, mask, (w - 1, h - 1), 128)
    fg[flood == 128] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    alpha = cv2.GaussianBlur(fg, (7, 7), 0)
    alpha[alpha < 20] = 0
    return alpha.astype(np.uint8)


def _crop_to_alpha(rgba):
    alpha = rgba[:, :, 3]
    nz = cv2.findNonZero(alpha)
    if nz is None:
        return rgba
    x, y, w, h = cv2.boundingRect(nz)
    pad_x = max(2, int(w * 0.02))
    pad_y = max(2, int(h * 0.02))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(rgba.shape[1], x + w + pad_x)
    y1 = min(rgba.shape[0], y + h + pad_y)
    return rgba[y0:y1, x0:x1].copy()
