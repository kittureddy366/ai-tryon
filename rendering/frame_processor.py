import cv2
import mediapipe as mp
import numpy as np

from utils.math_utils import angle_degrees, center_point, clamp, distance


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

try:
    _SELFIE_SEGMENTER = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
except Exception:
    _SELFIE_SEGMENTER = None


def initialize_landmark_smoother():
    return {"points": None, "torso_center": None}


def _pixel_landmark(landmarks, idx, width, height):
    landmark = landmarks[idx]
    return {
        "x": int(landmark.x * width),
        "y": int(landmark.y * height),
        "z": float(landmark.z),
        "visibility": float(getattr(landmark, "visibility", 1.0)),
    }


def _copy_points(points):
    copied = {}
    for key, value in points.items():
        if isinstance(value, dict):
            copied[key] = {
                "x": value["x"],
                "y": value["y"],
                "z": value["z"],
                "visibility": value["visibility"],
            }
        else:
            copied[key] = value
    return copied


def smooth_landmarks(points, smoother_state, alpha=0.32):
    if smoother_state["points"] is None:
        smoother_state["points"] = _copy_points(points)
        smoother_state["torso_center"] = points["torso_center"]
        return points

    prev = smoother_state["points"]
    # Adaptive smoothing: stabilize jitter when still, react faster during motion.
    motion = distance(
        (prev["left_shoulder"]["x"], prev["left_shoulder"]["y"]),
        (points["left_shoulder"]["x"], points["left_shoulder"]["y"]),
    ) + distance(
        (prev["right_shoulder"]["x"], prev["right_shoulder"]["y"]),
        (points["right_shoulder"]["x"], points["right_shoulder"]["y"]),
    )
    alpha = clamp(0.20 + motion * 0.012, 0.16, 0.72)

    smoothed = {}
    for key, point in points.items():
        if key == "torso_center":
            continue
        smoothed[key] = {
            "x": int(prev[key]["x"] + (point["x"] - prev[key]["x"]) * alpha),
            "y": int(prev[key]["y"] + (point["y"] - prev[key]["y"]) * alpha),
            "z": prev[key]["z"] + (point["z"] - prev[key]["z"]) * alpha,
            "visibility": prev[key]["visibility"]
            + (point["visibility"] - prev[key]["visibility"]) * alpha,
        }

    prev_center = smoother_state.get("torso_center")
    current_center = points["torso_center"]
    if prev_center is None:
        torso_center = current_center
    else:
        torso_center = (
            int(alpha * current_center[0] + (1.0 - alpha) * prev_center[0]),
            int(alpha * current_center[1] + (1.0 - alpha) * prev_center[1]),
        )
    smoothed["torso_center"] = torso_center
    smoother_state["torso_center"] = torso_center
    smoother_state["points"] = _copy_points(smoothed)
    return smoothed


def extract_pose_landmarks(result, frame_shape, smoother_state=None):
    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]
    height, width = frame_shape[:2]
    points = {
        "left_shoulder": _pixel_landmark(landmarks, LEFT_SHOULDER, width, height),
        "right_shoulder": _pixel_landmark(landmarks, RIGHT_SHOULDER, width, height),
        "left_hip": _pixel_landmark(landmarks, LEFT_HIP, width, height),
        "right_hip": _pixel_landmark(landmarks, RIGHT_HIP, width, height),
        "left_elbow": _pixel_landmark(landmarks, LEFT_ELBOW, width, height),
        "right_elbow": _pixel_landmark(landmarks, RIGHT_ELBOW, width, height),
        "left_wrist": _pixel_landmark(landmarks, LEFT_WRIST, width, height),
        "right_wrist": _pixel_landmark(landmarks, RIGHT_WRIST, width, height),
    }
    shoulder_center = center_point(
        (points["left_shoulder"]["x"], points["left_shoulder"]["y"]),
        (points["right_shoulder"]["x"], points["right_shoulder"]["y"]),
    )
    hip_center = center_point(
        (points["left_hip"]["x"], points["left_hip"]["y"]),
        (points["right_hip"]["x"], points["right_hip"]["y"]),
    )
    chest_midpoint = (
        int(0.65 * shoulder_center[0] + 0.35 * hip_center[0]),
        int(0.65 * shoulder_center[1] + 0.35 * hip_center[1]),
    )
    points["chest_midpoint"] = {
        "x": chest_midpoint[0],
        "y": chest_midpoint[1],
        "z": float(
            (
                points["left_shoulder"]["z"]
                + points["right_shoulder"]["z"]
                + points["left_hip"]["z"]
                + points["right_hip"]["z"]
            )
            * 0.25
        ),
        "visibility": min(
            points["left_shoulder"]["visibility"],
            points["right_shoulder"]["visibility"],
            points["left_hip"]["visibility"],
            points["right_hip"]["visibility"],
        ),
    }
    points["torso_center"] = (
        int((shoulder_center[0] + hip_center[0]) * 0.5),
        int((shoulder_center[1] + hip_center[1]) * 0.5),
    )

    required = (
        points["left_shoulder"],
        points["right_shoulder"],
        points["left_hip"],
        points["right_hip"],
    )
    if any(point["visibility"] < 0.35 for point in required):
        return None

    if smoother_state is not None:
        return smooth_landmarks(points, smoother_state)
    return points


def get_person_segmentation_mask(frame_bgr, frame_shape=None, threshold=0.45):
    if _SELFIE_SEGMENTER is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    output = _SELFIE_SEGMENTER.process(rgb)
    if output.segmentation_mask is None:
        return None
    soft = cv2.GaussianBlur(output.segmentation_mask.astype(np.float32), (7, 7), 0)
    return (soft > threshold).astype(np.uint8)


def get_result_segmentation_mask(result, frame_shape, threshold=0.30):
    if result is None:
        return None
    masks = getattr(result, "segmentation_masks", None)
    if not masks:
        return None
    try:
        mask_image = masks[0]
        if hasattr(mask_image, "numpy_view"):
            mask = mask_image.numpy_view().astype(np.float32)
        elif hasattr(mask_image, "data"):
            mask = np.array(mask_image.data, dtype=np.float32)
        else:
            return None
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        h, w = frame_shape[:2]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return (mask > threshold).astype(np.uint8)
    except Exception:
        return None


def calculate_body_measurements(points, smoother_state=None, center_alpha=0.35):
    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    left_hip = points["left_hip"]
    right_hip = points["right_hip"]

    shoulder_center = center_point(
        (left_shoulder["x"], left_shoulder["y"]),
        (right_shoulder["x"], right_shoulder["y"]),
    )
    hip_center = center_point(
        (left_hip["x"], left_hip["y"]),
        (right_hip["x"], right_hip["y"]),
    )
    torso_center = points.get("torso_center", center_point(shoulder_center, hip_center))
    if smoother_state is not None:
        prev_center = smoother_state.get("torso_center")
        if prev_center is not None:
            torso_center = (
                center_alpha * torso_center[0] + (1.0 - center_alpha) * prev_center[0],
                center_alpha * torso_center[1] + (1.0 - center_alpha) * prev_center[1],
            )
        smoother_state["torso_center"] = torso_center

    shoulder_width = distance(
        (left_shoulder["x"], left_shoulder["y"]),
        (right_shoulder["x"], right_shoulder["y"]),
    )
    hip_width = distance((left_hip["x"], left_hip["y"]), (right_hip["x"], right_hip["y"]))
    torso_width = shoulder_width * 0.68 + hip_width * 0.32
    torso_height = distance(shoulder_center, hip_center)
    shoulder_angle = angle_degrees(
        (left_shoulder["x"], left_shoulder["y"]),
        (right_shoulder["x"], right_shoulder["y"]),
    )

    shoulder_depth = (left_shoulder["z"] + right_shoulder["z"]) / 2.0
    hip_depth = (left_hip["z"] + right_hip["z"]) / 2.0
    torso_depth = (shoulder_depth + hip_depth) / 2.0
    depth_delta = hip_depth - shoulder_depth
    chest_ratio = torso_width / max(torso_height, 1.0)

    return {
        "shoulder_center": shoulder_center,
        "hip_center": hip_center,
        "torso_center": torso_center,
        "chest_midpoint": (points["chest_midpoint"]["x"], points["chest_midpoint"]["y"]),
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_width": torso_width,
        "torso_height": torso_height,
        "shoulder_angle": shoulder_angle,
        "shoulder_depth": shoulder_depth,
        "hip_depth": hip_depth,
        "torso_depth": torso_depth,
        "depth_delta": depth_delta,
        "chest_ratio": chest_ratio,
    }


def generate_torso_mesh(points, measurements, shoulder_expand=0.14, hip_expand=0.08):
    ls = np.array([points["left_shoulder"]["x"], points["left_shoulder"]["y"]], dtype=np.float32)
    rs = np.array([points["right_shoulder"]["x"], points["right_shoulder"]["y"]], dtype=np.float32)
    lh = np.array([points["left_hip"]["x"], points["left_hip"]["y"]], dtype=np.float32)
    rh = np.array([points["right_hip"]["x"], points["right_hip"]["y"]], dtype=np.float32)

    shoulder_vec = rs - ls
    shoulder_len = max(np.linalg.norm(shoulder_vec), 1.0)
    shoulder_dir = shoulder_vec / shoulder_len
    shoulder_normal = np.array([-shoulder_dir[1], shoulder_dir[0]], dtype=np.float32)

    hip_vec = rh - lh
    hip_len = max(np.linalg.norm(hip_vec), 1.0)
    hip_dir = hip_vec / hip_len
    hip_normal = np.array([-hip_dir[1], hip_dir[0]], dtype=np.float32)

    top_expand = shoulder_len * shoulder_expand
    bottom_expand = hip_len * hip_expand

    ls_top = ls - shoulder_dir * top_expand + shoulder_normal * 0.02 * shoulder_len
    rs_top = rs + shoulder_dir * top_expand + shoulder_normal * 0.02 * shoulder_len
    rh_bottom = rh + hip_dir * bottom_expand - hip_normal * 0.02 * hip_len
    lh_bottom = lh - hip_dir * bottom_expand - hip_normal * 0.02 * hip_len

    mid_top = (ls_top + rs_top) * 0.5
    mid_bottom = (lh_bottom + rh_bottom) * 0.5
    mid_torso = (mid_top + mid_bottom) * 0.5
    chest_mid = np.array(
        [points["chest_midpoint"]["x"], points["chest_midpoint"]["y"]],
        dtype=np.float32,
    )

    return {
        "quad": np.array([ls_top, rs_top, rh_bottom, lh_bottom], dtype=np.float32),
        "mid_top": tuple(mid_top.tolist()),
        "mid_bottom": tuple(mid_bottom.tolist()),
        "mid_torso": tuple(mid_torso.tolist()),
        "chest_midpoint": tuple(chest_mid.tolist()),
        "left_shoulder": tuple(ls.tolist()),
        "right_shoulder": tuple(rs.tolist()),
        "right_hip": tuple(rh.tolist()),
        "left_hip": tuple(lh.tolist()),
        "neck_axis": (tuple(ls_top.tolist()), tuple(rs_top.tolist())),
        "shoulder_angle": measurements["shoulder_angle"],
    }


def estimate_camera_distance_meters(shoulder_width_px):
    reference_shoulder_m = 0.40
    focal_like_constant = 720.0
    shoulder_width_px = max(1.0, shoulder_width_px)
    return (reference_shoulder_m * focal_like_constant) / shoulder_width_px
