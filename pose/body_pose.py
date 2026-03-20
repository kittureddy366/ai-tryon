import cv2
import mediapipe as mp
import os
from pathlib import Path
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class BodyPoseDetector:
    def __init__(
        self,
        model_asset_path: str | None = None,
        running_mode: str = "video",
        output_segmentation_masks: bool = False,
    ):
        if not model_asset_path:
            model_asset_path = os.environ.get("POSE_LANDMARKER_MODEL", "")
        if not model_asset_path:
            repo_root = Path(__file__).resolve().parents[1]
            model_asset_path = str(repo_root / "pose_landmarker_lite.task")

        running_mode = (running_mode or "video").strip().lower()
        if running_mode not in {"video", "image"}:
            raise ValueError("running_mode must be 'video' or 'image'")
        self.running_mode = running_mode

        base_options = python.BaseOptions(
            model_asset_path=model_asset_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO if self.running_mode == "video" else vision.RunningMode.IMAGE,
            output_segmentation_masks=bool(output_segmentation_masks),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect_pose(self, frame, timestamp=None):

        if frame is None:
            raise ValueError("frame is None")
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        if self.running_mode == "image":
            result = self.detector.detect(mp_image)
        else:
            if timestamp is None:
                timestamp = 0
            result = self.detector.detect_for_video(mp_image, int(timestamp))

        return result
