import os
import sys

import cv2
import numpy as np


class OutputRefiner:
    """
    Optional visual refinement stage.
    - Uses Real-ESRGAN if runtime deps + weights exist.
    - Falls back to fast local enhancement around torso ROI.
    """

    def __init__(self, project_root, enable_esrgan=False):
        self.mode = "fast"
        self.upsampler = None
        self.frame_count = 0
        self._cached_roi = None
        self._cached_bounds = None
        if bool(enable_esrgan):
            self._try_init_esrgan(project_root)

    def _try_init_esrgan(self, project_root):
        esrgan_root = os.path.join(project_root, "Real-ESRGAN")
        basicsr_root = os.path.join(project_root, "BasicSR")
        weights_dir = os.path.join(esrgan_root, "weights")
        if not os.path.isdir(esrgan_root):
            return

        candidate_names = (
            "RealESRGAN_x4plus.pth",
            "RealESRNet_x4plus.pth",
            "realesr-general-x4v3.pth",
        )
        weight_path = None
        search_dirs = [project_root]
        if os.path.isdir(weights_dir):
            search_dirs.insert(0, weights_dir)
        for folder in search_dirs:
            for name in candidate_names:
                path = os.path.join(folder, name)
                if os.path.exists(path):
                    weight_path = path
                    break
            if weight_path is not None:
                break
        if weight_path is None:
            return

        try:
            if os.path.isdir(basicsr_root):
                sys.path.insert(0, basicsr_root)
            sys.path.insert(0, esrgan_root)
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=weight_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
            )
            self.mode = "esrgan"
        except Exception:
            self.upsampler = None
            self.mode = "fast"
        finally:
            if os.path.isdir(basicsr_root) and basicsr_root in sys.path:
                sys.path.remove(basicsr_root)
            if esrgan_root in sys.path:
                sys.path.remove(esrgan_root)

    def refine(self, frame_bgr, points=None):
        if points is None:
            return frame_bgr

        x0, y0, x1, y1 = self._torso_roi(points, frame_bgr.shape)
        if x1 <= x0 or y1 <= y0:
            return frame_bgr

        out = frame_bgr.copy()
        roi = out[y0:y1, x0:x1]
        if roi.size == 0:
            return out

        self.frame_count += 1
        if self.mode == "esrgan" and self.upsampler is not None:
            if (
                self._cached_roi is not None
                and self._cached_bounds == (x0, y0, x1, y1)
                and (self.frame_count % 3 != 0)
            ):
                refined = self._cached_roi
            else:
                refined = self._esrgan_refine_roi(roi)
                self._cached_roi = refined
                self._cached_bounds = (x0, y0, x1, y1)
        else:
            refined = self._fast_refine_roi(roi)
        out[y0:y1, x0:x1] = refined
        return out

    @staticmethod
    def _torso_roi(points, frame_shape):
        h, w = frame_shape[:2]
        xs = [
            points["left_shoulder"]["x"],
            points["right_shoulder"]["x"],
            points["left_hip"]["x"],
            points["right_hip"]["x"],
            points["chest_midpoint"]["x"],
        ]
        ys = [
            points["left_shoulder"]["y"],
            points["right_shoulder"]["y"],
            points["left_hip"]["y"],
            points["right_hip"]["y"],
            points["chest_midpoint"]["y"],
        ]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        torso_w = max_x - min_x
        torso_h = max_y - min_y
        pad_x = int(max(18, torso_w * 0.35))
        pad_y_top = int(max(18, torso_h * 0.28))
        pad_y_bottom = int(max(28, torso_h * 0.44))

        x0 = max(0, int(min_x - pad_x))
        y0 = max(0, int(min_y - pad_y_top))
        x1 = min(w, int(max_x + pad_x))
        y1 = min(h, int(max_y + pad_y_bottom))
        return x0, y0, x1, y1

    def _esrgan_refine_roi(self, roi):
        try:
            # Realtime-safe ESRGAN: downscale ROI, run SR, then resize back.
            h, w = roi.shape[:2]
            max_side = max(h, w)
            scale = 1.0
            if max_side > 220:
                scale = 220.0 / float(max_side)
            in_w = max(32, int(w * scale))
            in_h = max(32, int(h * scale))
            small = cv2.resize(roi, (in_w, in_h), interpolation=cv2.INTER_AREA)
            enhanced, _ = self.upsampler.enhance(small, outscale=1.0)
            refined = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LINEAR)
            return refined
        except Exception:
            self.mode = "fast"
            self.upsampler = None
            return self._fast_refine_roi(roi)

    @staticmethod
    def _fast_refine_roi(roi):
        # Fast edge-aware enhancement fallback when ESRGAN isn't available.
        blur = cv2.GaussianBlur(roi, (0, 0), 1.1)
        sharp = cv2.addWeighted(roi, 1.32, blur, -0.32, 0)
        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        return np.clip(out, 0, 255).astype(np.uint8)
