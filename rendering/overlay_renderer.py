import cv2
import numpy as np

from utils.image_utils import adapt_garment_lighting, gaussian_feather_alpha

MESH_GRID_X = 10
MESH_GRID_Y = 10


def _ensure_rgba(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([image, alpha], axis=2)
    return image


def _clean_transparency_artifacts(rgba):
    color = rgba[:, :, :3]
    alpha = rgba[:, :, 3].astype(np.float32)

    alpha[alpha < 10] = 0
    alpha[alpha > 245] = 255

    # Remove common checkerboard-like background artifacts connected to borders.
    h, w = alpha.shape
    corner_pixels = np.array(
        [
            color[2, 2],
            color[2, max(w - 3, 0)],
            color[max(h - 3, 0), 2],
            color[max(h - 3, 0), max(w - 3, 0)],
        ],
        dtype=np.float32,
    )
    corner_mean = corner_pixels.mean(axis=0)
    diff = np.linalg.norm(color.astype(np.float32) - corner_mean[None, None, :], axis=2)
    gray_like = (
        (np.abs(color[:, :, 0].astype(np.int16) - color[:, :, 1].astype(np.int16)) < 18)
        & (np.abs(color[:, :, 1].astype(np.int16) - color[:, :, 2].astype(np.int16)) < 18)
    )
    bg_like = (diff < 44.0) & gray_like

    border_seed = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood = np.zeros((h, w), dtype=np.uint8)
    flood[bg_like] = 255
    cv2.floodFill(flood, border_seed, (0, 0), 128)
    cv2.floodFill(flood, border_seed, (w - 1, 0), 128)
    cv2.floodFill(flood, border_seed, (0, h - 1), 128)
    cv2.floodFill(flood, border_seed, (w - 1, h - 1), 128)
    border_bg = flood == 128
    alpha[border_bg] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.morphologyEx(alpha.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    rgba[:, :, 3] = alpha
    return rgba


def load_shirt_rgba(path):
    shirt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if shirt is None:
        raise RuntimeError(f"Could not load shirt image: {path}")
    shirt = _ensure_rgba(shirt)
    shirt = _clean_transparency_artifacts(shirt)
    return shirt


def create_foreground_arm_mask(frame_shape, points, torso_depth, person_mask=None):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    arm_pairs = (
        ("left_shoulder", "left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow", "right_wrist"),
    )
    for shoulder_name, elbow_name, wrist_name in arm_pairs:
        shoulder = points[shoulder_name]
        elbow = points[elbow_name]
        wrist = points[wrist_name]
        arm_depth = (elbow["z"] + wrist["z"]) * 0.5
        if arm_depth < torso_depth - 0.04:
            cv2.line(mask, (shoulder["x"], shoulder["y"]), (elbow["x"], elbow["y"]), 255, 28, cv2.LINE_AA)
            cv2.line(mask, (elbow["x"], elbow["y"]), (wrist["x"], wrist["y"]), 255, 24, cv2.LINE_AA)
            cv2.circle(mask, (wrist["x"], wrist["y"]), 14, 255, -1, cv2.LINE_AA)
    if person_mask is not None:
        mask = cv2.bitwise_and(mask, (person_mask * 255).astype(np.uint8))
    return mask


def _build_torso_mask(frame_shape, torso_mesh, person_mask=None):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    quad = torso_mesh["quad"].astype(np.int32)
    cv2.fillConvexPoly(mask, quad, 255)
    if person_mask is not None:
        mask = cv2.bitwise_and(mask, (person_mask * 255).astype(np.uint8))
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask


def _non_rigid_deform(garment_rgba, mesh_warp):
    h, w = garment_rgba.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    warp_strength = float(mesh_warp)
    x_offset = warp_strength * np.sin((y / max(h, 1)) * np.pi) * 0.8
    y_offset = warp_strength * 0.25 * np.sin((x / max(w, 1)) * np.pi * 1.5)
    map_x = x + x_offset
    map_y = y + y_offset
    return cv2.remap(
        garment_rgba,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _quad_lerp(quad, u, v):
    top = quad[0] * (1.0 - u) + quad[1] * u
    bottom = quad[3] * (1.0 - u) + quad[2] * u
    return top * (1.0 - v) + bottom * v


def _build_mesh(quad, grid_x=12, grid_y=12):
    vertices = []
    for j in range(grid_y + 1):
        v = j / float(max(grid_y, 1))
        for i in range(grid_x + 1):
            u = i / float(max(grid_x, 1))
            vertices.append(_quad_lerp(quad, u, v))

    triangles = []
    stride = grid_x + 1
    for j in range(grid_y):
        for i in range(grid_x):
            i0 = j * stride + i
            i1 = i0 + 1
            i2 = i0 + stride
            i3 = i2 + 1
            triangles.append((i0, i2, i1))
            triangles.append((i1, i2, i3))
    return np.array(vertices, dtype=np.float32), triangles


def _warp_triangle(src_rgba, dst_rgba, src_tri, dst_tri):
    frame_h, frame_w = dst_rgba.shape[:2]
    src_rect = cv2.boundingRect(src_tri.astype(np.float32))
    dst_rect = cv2.boundingRect(dst_tri.astype(np.float32))
    sx, sy, sw, sh = src_rect
    dx, dy, dw, dh = dst_rect
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return

    src_patch = src_rgba[sy : sy + sh, sx : sx + sw]
    if src_patch.size == 0:
        return

    src_local = src_tri - np.array([sx, sy], dtype=np.float32)
    dst_local = dst_tri - np.array([dx, dy], dtype=np.float32)

    affine = cv2.getAffineTransform(src_local.astype(np.float32), dst_local.astype(np.float32))
    warped = cv2.warpAffine(
        src_patch,
        affine,
        (dw, dh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst_local).astype(np.int32), 255)

    x0 = max(dx, 0)
    y0 = max(dy, 0)
    x1 = min(dx + dw, frame_w)
    y1 = min(dy + dh, frame_h)
    if x0 >= x1 or y0 >= y1:
        return

    cx0 = x0 - dx
    cy0 = y0 - dy
    cx1 = cx0 + (x1 - x0)
    cy1 = cy0 + (y1 - y0)

    warped_roi = warped[cy0:cy1, cx0:cx1].astype(np.float32)
    mask_roi = (mask[cy0:cy1, cx0:cx1].astype(np.float32) / 255.0)[:, :, None]

    dst_roi = dst_rgba[y0:y1, x0:x1].astype(np.float32)
    alpha = (warped_roi[:, :, 3:4] / 255.0) * mask_roi
    inv_alpha = 1.0 - alpha

    dst_roi[:, :, :3] = warped_roi[:, :, :3] * alpha + dst_roi[:, :, :3] * inv_alpha
    dst_roi[:, :, 3:4] = np.maximum(dst_roi[:, :, 3:4], warped_roi[:, :, 3:4] * mask_roi)
    dst_rgba[y0:y1, x0:x1] = np.clip(dst_roi, 0, 255).astype(np.uint8)


def _mesh_warp_to_torso(garment_rgba, torso_mesh, frame_shape, grid_x=MESH_GRID_X, grid_y=MESH_GRID_Y):
    frame_h, frame_w = frame_shape[:2]
    gh, gw = garment_rgba.shape[:2]
    output = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)

    src_quad = np.float32(
        [
            [gw * 0.18, gh * 0.16],
            [gw * 0.82, gh * 0.16],
            [gw * 0.73, gh * 0.97],
            [gw * 0.27, gh * 0.97],
        ]
    )
    dst_quad = torso_mesh["quad"].astype(np.float32)

    src_vertices, triangles = _build_mesh(src_quad, grid_x=grid_x, grid_y=grid_y)
    dst_vertices, _ = _build_mesh(dst_quad, grid_x=grid_x, grid_y=grid_y)

    for idx0, idx1, idx2 in triangles:
        src_tri = np.array([src_vertices[idx0], src_vertices[idx1], src_vertices[idx2]], dtype=np.float32)
        dst_tri = np.array([dst_vertices[idx0], dst_vertices[idx1], dst_vertices[idx2]], dtype=np.float32)
        _warp_triangle(garment_rgba, output, src_tri, dst_tri)

    return output


def render_shirt_with_occlusion(
    frame_bgr,
    base_frame_bgr,
    shirt_rgba,
    shirt_state,
    points,
    torso_mesh,
    torso_depth,
    person_mask=None,
):
    garment = cv2.resize(
        shirt_rgba,
        (shirt_state["width"], shirt_state["height"]),
        interpolation=cv2.INTER_AREA,
    )
    garment = _non_rigid_deform(garment, shirt_state["mesh_warp"])

    if abs(shirt_state["angle"]) > 0.2:
        gh, gw = garment.shape[:2]
        matrix = cv2.getRotationMatrix2D((gw * 0.5, gh * 0.18), shirt_state["angle"], 1.0)
        garment = cv2.warpAffine(
            garment,
            matrix,
            (gw, gh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    projected = _mesh_warp_to_torso(garment, torso_mesh, frame_bgr.shape)
    projected = adapt_garment_lighting(projected, frame_bgr, torso_mesh["quad"])

    torso_mask = _build_torso_mask(frame_bgr.shape, torso_mesh, person_mask=person_mask)
    edge_feather = gaussian_feather_alpha(torso_mask.astype(np.float32) / 255.0, ksize=19, sigma=6.0)

    alpha = projected[:, :, 3].astype(np.float32) / 255.0
    alpha *= edge_feather
    color = projected[:, :, :3].astype(np.float32)
    base = frame_bgr.astype(np.float32)
    blended = color * alpha[:, :, None] + base * (1.0 - alpha[:, :, None])
    frame_bgr[:] = blended.astype(np.uint8)

    arm_mask = create_foreground_arm_mask(
        frame_bgr.shape,
        points,
        torso_depth=torso_depth,
        person_mask=person_mask,
    )
    frame_bgr[arm_mask > 0] = base_frame_bgr[arm_mask > 0]
