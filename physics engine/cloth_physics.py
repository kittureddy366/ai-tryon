import numpy as np

from utils.math_utils import clamp


SHIRT_SIZE_THRESHOLD_RATIO = 0.37
GRAVITY_DROP_RATIO = 0.09
MAX_SHIRT_STRETCH = 1.14
MIN_SHIRT_STRETCH = 0.86
MAX_ROTATION = 38.0

# XL regular fit profile (inches).
XL_SHOULDER_IN = 20.0
XL_CHEST_CIRC_IN = 43.0
XL_LENGTH_IN = 29.75
XL_SLEEVE_LENGTH_IN = 9.0
XL_FABRIC_GSM = 180
XL_MATERIAL = "100% combed cotton"
XL_NECK_STYLE = "crew neck"
XL_COLOR = "jet black"
XL_CHEST_FLAT_IN = XL_CHEST_CIRC_IN * 0.5
XL_WIDTH_TO_SHOULDER = XL_CHEST_FLAT_IN / XL_SHOULDER_IN
XL_LENGTH_TO_SHOULDER = XL_LENGTH_IN / XL_SHOULDER_IN


def initialize_physics_state():
    return {
        "center": None,
        "target_center": None,
        "width": 0.0,
        "height": 0.0,
        "width_vel": 0.0,
        "height_vel": 0.0,
        "angle": 0.0,
        "mesh_warp": 0.0,
        "neckline_drop": 0.0,
    }


def simulate_shirt_physics(
    measurements,
    torso_mesh,
    frame_shape,
    shirt_base_size,
    state,
    dt,
    distance_scale=1.0,
):
    frame_h, frame_w = frame_shape[:2]
    shirt_w_px, shirt_h_px = shirt_base_size

    neck_left = np.array(torso_mesh["neck_axis"][0], dtype=np.float32)
    neck_right = np.array(torso_mesh["neck_axis"][1], dtype=np.float32)
    neck_center = (neck_left + neck_right) * 0.5
    torso_center = np.array(torso_mesh["mid_torso"], dtype=np.float32)
    torso_vertical = torso_center - neck_center
    torso_len = max(np.linalg.norm(torso_vertical), 1.0)
    torso_dir = torso_vertical / torso_len

    gravity_drop = measurements["torso_height"] * GRAVITY_DROP_RATIO
    anchor_center = neck_center + torso_dir * (torso_len * 0.60 + gravity_drop)

    perspective_scale = clamp(1.0 + measurements["depth_delta"] * 0.52, 0.88, 1.14)
    distance_scale = clamp(distance_scale, 0.84, 1.20)

    # Dynamic scaling based on torso dimensions + XL regular-fit references.
    shoulder_width = max(measurements["shoulder_width"], 1.0)
    torso_height = max(measurements["torso_height"], 1.0)
    fit_ease = 1.06  # Regular fit: comfortable, not slim.
    target_width = shoulder_width * XL_WIDTH_TO_SHOULDER * fit_ease * perspective_scale * distance_scale
    target_height = shoulder_width * XL_LENGTH_TO_SHOULDER * perspective_scale * distance_scale
    target_height = max(target_height, torso_height * 1.28 * perspective_scale * distance_scale)
    target_width = max(target_width, 80.0)
    target_height = max(target_height, 120.0)

    base_ratio = shirt_h_px / float(shirt_w_px)
    target_height = max(target_height, target_width * base_ratio * 0.92)
    shirt_size_limit = frame_w * SHIRT_SIZE_THRESHOLD_RATIO
    too_tight = measurements["shoulder_width"] > shirt_size_limit
    if too_tight:
        target_width = min(target_width, shirt_size_limit * 1.74)
        target_height = min(target_height, target_width * base_ratio * 1.13)

    if state["center"] is None:
        state["center"] = anchor_center.astype(np.float32)
        state["target_center"] = anchor_center.astype(np.float32)
        state["width"] = target_width
        state["height"] = target_height
        state["angle"] = measurements["shoulder_angle"]
        state["mesh_warp"] = measurements["depth_delta"] * 16.0
        state["neckline_drop"] = gravity_drop
    else:
        body_speed = np.linalg.norm(anchor_center - state["target_center"]) / max(dt, 1e-3)
        state["target_center"] = anchor_center.astype(np.float32)

        lag_alpha = clamp(0.44 - body_speed * 0.0011, 0.10, 0.44)
        state["center"] += (state["target_center"] - state["center"]) * lag_alpha

        stiffness = 10.5
        damping = 0.73
        state["width_vel"] += (target_width - state["width"]) * stiffness * dt
        state["height_vel"] += (target_height - state["height"]) * stiffness * dt
        state["width_vel"] *= damping
        state["height_vel"] *= damping
        state["width"] += state["width_vel"] * dt * 60.0
        state["height"] += state["height_vel"] * dt * 60.0

        min_w = target_width * MIN_SHIRT_STRETCH
        max_w = target_width * MAX_SHIRT_STRETCH
        min_h = target_height * MIN_SHIRT_STRETCH
        max_h = target_height * MAX_SHIRT_STRETCH
        state["width"] = float(np.clip(state["width"], min_w, max_w))
        state["height"] = float(np.clip(state["height"], min_h, max_h))

        target_angle = clamp(measurements["shoulder_angle"], -MAX_ROTATION, MAX_ROTATION)
        state["angle"] += (target_angle - state["angle"]) * 0.22

        warp_target = clamp(
            (measurements["hip_width"] - measurements["shoulder_width"]) * 0.16
            + measurements["depth_delta"] * 42.0,
            -16.0,
            16.0,
        )
        state["mesh_warp"] += (warp_target - state["mesh_warp"]) * 0.20
        state["neckline_drop"] += (gravity_drop - state["neckline_drop"]) * 0.20

    state["center"][0] = clamp(state["center"][0], 0, frame_w - 1)
    state["center"][1] = clamp(state["center"][1], 0, frame_h - 1)

    return {
        "center": (float(state["center"][0]), float(state["center"][1])),
        "width": max(1, int(state["width"])),
        "height": max(1, int(state["height"])),
        "angle": float(state["angle"]),
        "mesh_warp": float(state["mesh_warp"]),
        "neckline_drop": float(state["neckline_drop"]),
        "too_tight": too_tight,
        "max_supported_width": shirt_size_limit,
    }
