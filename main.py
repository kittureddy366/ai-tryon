import time
import os
import argparse

import cv2
import numpy as np

from garment.garment_library import get_preset, list_preset_keys
from garment.cloth_mesh import create_shirt_cloth_mesh
from garment.obj_garment_loader import load_obj_garment_mesh, planar_uvs_from_vertices
from garment.triposr_obj_generator import TripoSRObjGenerator, find_2d_garment_image
from garment.zip_garment_loader import load_garment_zip_texture
from physics_engine.mass_spring_cloth import MassSpringClothSimulator
from pose.body_pose import BodyPoseDetector
from pose.body_mesh_generator import PreciseBodyMeshGenerator
from rendering.frame_processor import (
    extract_pose_landmarks,
    get_person_segmentation_mask,
    get_result_segmentation_mask,
    initialize_landmark_smoother,
)
from rendering.output_refiner import OutputRefiner
from rendering.realtime_mesh_renderer import OPENGL_AVAILABLE, RealtimeMeshRenderer


def _build_anchor_targets(body_mesh, points, anchor_state=None):
    torso_z = float(body_mesh.joints.get("torso_center", np.array([0.0, 0.0, -8.0], dtype=np.float32))[2])
    chest_z = float(body_mesh.joints.get("chest", np.array([0.0, 0.0, torso_z - 1.0], dtype=np.float32))[2])
    shoulder_width = float(np.hypot(
        points["right_shoulder"]["x"] - points["left_shoulder"]["x"],
        points["right_shoulder"]["y"] - points["left_shoulder"]["y"],
    ))
    torso_height = float(np.hypot(
        ((points["left_hip"]["x"] + points["right_hip"]["x"]) * 0.5) - ((points["left_shoulder"]["x"] + points["right_shoulder"]["x"]) * 0.5),
        ((points["left_hip"]["y"] + points["right_hip"]["y"]) * 0.5) - ((points["left_shoulder"]["y"] + points["right_shoulder"]["y"]) * 0.5),
    ))
    neck_xy = (
        int((points["left_shoulder"]["x"] + points["right_shoulder"]["x"]) * 0.5),
        int((points["left_shoulder"]["y"] + points["right_shoulder"]["y"]) * 0.5 - 0.07 * shoulder_width),
    )
    shoulder_angle = float(np.degrees(np.arctan2(
        points["right_shoulder"]["y"] - points["left_shoulder"]["y"],
        points["right_shoulder"]["x"] - points["left_shoulder"]["x"],
    )))
    garment_scale = float(np.clip(shoulder_width / 165.0, 0.82, 1.24))
    garment_length_scale = float(np.clip(torso_height / 195.0, 0.84, 1.20))

    shoulder_pad = shoulder_width * 0.06
    left_shoulder_xy = np.array([points["left_shoulder"]["x"] - shoulder_pad, points["left_shoulder"]["y"]], dtype=np.float32)
    right_shoulder_xy = np.array([points["right_shoulder"]["x"] + shoulder_pad, points["right_shoulder"]["y"]], dtype=np.float32)
    hip_inset = shoulder_width * 0.03
    left_hip_xy = np.array([points["left_hip"]["x"] - hip_inset, points["left_hip"]["y"]], dtype=np.float32)
    right_hip_xy = np.array([points["right_hip"]["x"] + hip_inset, points["right_hip"]["y"]], dtype=np.float32)

    anchors = {
        "left_shoulder": np.array([left_shoulder_xy[0], left_shoulder_xy[1], torso_z - 1.0], dtype=np.float32),
        "right_shoulder": np.array([right_shoulder_xy[0], right_shoulder_xy[1], torso_z - 1.0], dtype=np.float32),
        "left_hip": np.array([left_hip_xy[0], left_hip_xy[1], torso_z], dtype=np.float32),
        "right_hip": np.array([right_hip_xy[0], right_hip_xy[1], torso_z], dtype=np.float32),
        "left_elbow": np.array([points["left_elbow"]["x"], points["left_elbow"]["y"], torso_z - 0.5], dtype=np.float32),
        "right_elbow": np.array([points["right_elbow"]["x"], points["right_elbow"]["y"], torso_z - 0.5], dtype=np.float32),
    }

    anchors["chest"] = np.array(
        [points["chest_midpoint"]["x"], points["chest_midpoint"]["y"], chest_z - 0.8],
        dtype=np.float32,
    )
    anchors["neck"] = np.array([neck_xy[0], neck_xy[1], chest_z - 1.0], dtype=np.float32)
    anchors["garment_scale"] = garment_scale
    anchors["garment_length_scale"] = garment_length_scale
    anchors["shoulder_width"] = shoulder_width
    anchors["torso_height"] = torso_height
    anchors["shoulder_angle"] = shoulder_angle

    if anchor_state is not None:
        prev = anchor_state.get("anchors")
        if prev is not None:
            smoothed = {}
            for key, value in anchors.items():
                if isinstance(value, np.ndarray) and key in prev:
                    smoothed[key] = prev[key] * 0.62 + value * 0.38
                else:
                    smoothed[key] = value
            anchors = smoothed
        anchor_state["anchors"] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in anchors.items()}
    return anchors


def _create_garment_runtime(project_root, preset_key):
    preset = get_preset(preset_key)
    mesh = None
    obj_path = ""
    if isinstance(preset.get("generated_obj_path", ""), str):
        obj_path = preset.get("generated_obj_path", "")
    if not obj_path and isinstance(preset.get("garment_obj_path", ""), str):
        obj_path = preset.get("garment_obj_path", "")
    if obj_path:
        if not os.path.isabs(obj_path):
            obj_path = os.path.join(project_root, obj_path)
        mesh = load_obj_garment_mesh(obj_path)
    if mesh is None:
        mesh = create_shirt_cloth_mesh(rows=preset["mesh_rows"], cols=preset["mesh_cols"])
    else:
        print(
            f"Loaded OBJ garment mesh: {mesh['vertices'].shape[0]} verts, "
            f"{mesh['faces'].shape[0]} faces (simplified for realtime)."
        )
    sim = MassSpringClothSimulator(mesh, **preset["physics"])
    texture = None
    zip_path = preset.get("garment_zip_path", "")
    if zip_path:
        if not os.path.isabs(zip_path):
            zip_path = os.path.join(project_root, zip_path)
        texture = load_garment_zip_texture(zip_path)
    if texture is None:
        tex_path = preset.get("texture_path", "")
        if tex_path and (not os.path.isabs(tex_path)):
            tex_path = os.path.join(project_root, tex_path)
        texture = _prepare_garment_texture(tex_path)
    return preset, mesh, sim, texture


def _prepare_garment_texture(path):
    texture = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if texture is None:
        return None
    if len(texture.shape) == 2:
        texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGRA)
    elif texture.shape[2] == 3:
        bgr = texture
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Catalog images often contain white/checkerboard backgrounds with no alpha.
        not_white = gray < 242
        has_color = hsv[:, :, 1] > 20
        dark_object = gray < 205
        fg = (not_white & dark_object) | (has_color & (gray < 250))
        fg = fg.astype(np.uint8) * 255

        # Remove border-connected background residues.
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

        num, labels, stats, _ = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), connectivity=8)
        if num > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            fg = np.where(labels == largest, 255, 0).astype(np.uint8)

        alpha = cv2.GaussianBlur(fg, (7, 7), 0)
        alpha[alpha < 20] = 0
        texture = np.dstack([bgr, alpha.astype(np.uint8)])
    elif texture.shape[2] == 4:
        texture = texture.copy()

    alpha = texture[:, :, 3]
    nz = cv2.findNonZero(alpha)
    if nz is not None:
        x, y, w, h = cv2.boundingRect(nz)
        pad_x = max(2, int(w * 0.02))
        pad_y = max(2, int(h * 0.02))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(texture.shape[1], x + w + pad_x)
        y1 = min(texture.shape[0], y + h + pad_y)
        texture = texture[y0:y1, x0:x1]
    return texture


def _format_triposr_status(snapshot):
    status = snapshot.get("status", "idle")
    if status == "running":
        return f"TripoSR: generating OBJ... {snapshot.get('elapsed', 0.0):.1f}s", (60, 220, 245)
    if status == "ready":
        return "TripoSR: OBJ ready (using generated mesh)", (55, 210, 70)
    if status == "failed":
        return "TripoSR: failed, using fallback cloth mesh", (0, 0, 255)
    return "TripoSR: idle", (180, 180, 180)


def run_virtual_tryon_3d(*, triposr_mode="auto", refine=True, esrgan=False, camera_index=0):
    project_root = os.path.dirname(os.path.abspath(__file__))
    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    pose_detector = BodyPoseDetector()
    body_model = PreciseBodyMeshGenerator(limb_sides=12, limb_rings=9, torso_rings=14, torso_sides=20)
    preset_keys = list_preset_keys()
    preset_idx = 0
    current_key = preset_keys[preset_idx]
    initial_preset = get_preset(current_key)
    triposr_image = find_2d_garment_image(project_root, [initial_preset.get("texture_path", "")])

    triposr_mode = (triposr_mode or "require").strip().lower()
    if triposr_mode not in {"off", "auto", "require"}:
        raise ValueError("triposr_mode must be one of: off, auto, require")

    triposr_gen = None
    if triposr_mode != "off":
        triposr_gen = TripoSRObjGenerator(project_root, triposr_image)
        triposr_gen.start_async(force=False)

    active_preset, cloth_mesh, cloth_sim, cloth_texture = _create_garment_runtime(project_root, current_key)
    triposr_applied = False
    renderer = RealtimeMeshRenderer()
    refiner = OutputRefiner(project_root=project_root, enable_esrgan=bool(esrgan)) if refine else None
    refine_enabled = bool(refine)
    smoother_state = initialize_landmark_smoother()

    cloth_initialized = False
    last_t = time.time()
    last_fps_at = last_t
    frames_since_fps = 0
    fps_smooth = 0.0
    frame_idx = 0
    last_points = None
    last_result = None
    anchor_state = {"anchors": None}
    paused = False
    screenshot_idx = 0
    texture_enabled = True
    texture_mode = "quad"

    while True:
        # Low-latency capture: drop buffered frames when CPU can't keep up.
        for _ in range(2):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt = max(1.0 / 120.0, now - last_t)
        last_t = now
        frames_since_fps += 1
        if now - last_fps_at >= 0.5:
            inst_fps = frames_since_fps / max(now - last_fps_at, 1e-6)
            fps_smooth = inst_fps if fps_smooth <= 0 else (fps_smooth * 0.7 + inst_fps * 0.3)
            last_fps_at = now
            frames_since_fps = 0
        triposr_state = triposr_gen.snapshot() if triposr_gen is not None else {"status": "idle"}

        if triposr_state["status"] == "ready" and not triposr_applied:
            preset = get_preset(current_key).copy()
            preset["generated_obj_path"] = triposr_state["obj_path"]
            active_preset = preset
            mesh = load_obj_garment_mesh(triposr_state["obj_path"], max_faces=6500)
            if mesh is not None:
                # TripoSR UVs are not guaranteed to align with the 2D garment image.
                # Use a stable planar UV projection so the 2D texture maps sensibly.
                planar_uvs = planar_uvs_from_vertices(mesh.get("vertices"))
                if planar_uvs is not None:
                    mesh["uvs"] = planar_uvs
                cloth_mesh = mesh
                cloth_sim = MassSpringClothSimulator(cloth_mesh, **preset["physics"])
                tex_path = preset.get("texture_path", "")
                if tex_path and (not os.path.isabs(tex_path)):
                    tex_path = os.path.join(project_root, tex_path)
                cloth_texture = _prepare_garment_texture(tex_path)
                cloth_initialized = False
                triposr_applied = True

        if (triposr_mode == "require") and (triposr_gen is not None) and (not triposr_applied):
            output = frame
            triposr_text, triposr_color = _format_triposr_status(triposr_state)
            cv2.putText(
                output,
                "Waiting for TripoSR garment mesh (required)",
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                triposr_text,
                (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                triposr_color,
                2,
                cv2.LINE_AA,
            )
            if triposr_state.get("status") == "failed":
                err = (triposr_state.get("error") or "").replace("\n", " ").strip()
                cv2.putText(
                    output,
                    ("TripoSR failed: " + err)[-110:],
                    (14, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    "Press Q/ESC to quit",
                    (14, 106),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow("Real-time 3D Virtual Try-On", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                triposr_mode = "auto"
            continue

        # Run pose every other frame for smoother overall realtime throughput.
        pose_stride = 4 if triposr_state["status"] == "running" else 2
        # Adaptive: when FPS drops, increase stride to keep UI responsive.
        if fps_smooth and fps_smooth < 12.0:
            pose_stride = max(pose_stride, 4)
        elif fps_smooth and fps_smooth < 18.0:
            pose_stride = max(pose_stride, 3)
        if frame_idx % pose_stride == 0 or last_points is None:
            result = pose_detector.detect_pose(frame, int(now * 1000))
            points = extract_pose_landmarks(result, frame.shape, smoother_state=smoother_state)
            last_points = points
            last_result = result
        else:
            points = last_points
            result = last_result
        frame_idx += 1

        output = frame
        if paused:
            cv2.putText(
                output,
                "PAUSED (space to resume, C to capture)",
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Real-time 3D Virtual Try-On", output)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" ") or key == ord("p"):
                paused = False
            elif key == ord("c"):
                os.makedirs(os.path.join(project_root, "generated", "captures"), exist_ok=True)
                out_path = os.path.join(project_root, "generated", "captures", f"capture_{screenshot_idx:04d}.png")
                screenshot_idx += 1
                cv2.imwrite(out_path, output)
                print(f"Saved capture: {out_path}")
            elif key == ord("q") or key == 27:
                break
            continue
        if points is not None:
            body_mesh = body_model.generate(result, frame.shape)
            if body_mesh is None:
                cloth_initialized = False
                last_points = None
                cv2.putText(
                    output,
                    "Body mesh generation unavailable",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Real-time 3D Virtual Try-On", output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue
            anchor_targets = _build_anchor_targets(body_mesh, points, anchor_state=anchor_state)
            person_mask = get_result_segmentation_mask(result, frame.shape, threshold=0.30)
            if person_mask is None:
                person_mask = get_person_segmentation_mask(frame, frame.shape, threshold=0.45)

            if not cloth_initialized:
                cloth_sim.initialize_from_anchors(anchor_targets)
                cloth_initialized = True

            cloth_vertices = cloth_sim.step(dt, anchor_targets, body_mesh=body_mesh)
            render_ok = True
            try:
                if (not np.isfinite(cloth_vertices).all()) or (
                    float(np.max(np.abs(cloth_vertices[:, :2]))) > (float(max(frame.shape[:2])) * 8.0 + 1024.0)
                ):
                    raise ValueError("Cloth vertices out of bounds")

                output = renderer.render_overlay(
                    frame,
                    body_mesh.vertices,
                    body_mesh.faces,
                    cloth_vertices,
                    cloth_mesh["faces"],
                    cloth_uvs=cloth_mesh.get("uvs"),
                    cloth_texture_rgba=(cloth_texture if texture_enabled else None),
                    cloth_color_bgra=active_preset["cloth_color_bgra"],
                    pose_points=points,
                    anchor_targets=anchor_targets,
                    person_mask=person_mask,
                    texture_mode=texture_mode,
                )
            except (cv2.error, MemoryError, ValueError):
                # When tracking glitches or the cloth sim explodes, reset and keep the camera loop alive.
                render_ok = False
                cloth_initialized = False
                last_points = None
                anchor_state["anchors"] = None
                output = frame.copy()
                cv2.putText(
                    output,
                    "Tracking/renderer reset (pose lost)",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if (
                render_ok
                and refine_enabled
                and (refiner is not None)
                and (triposr_state["status"] != "running" or (frame_idx % 2 == 0))
            ):
                output = refiner.refine(output, points=points)

            cv2.putText(
                output,
                "Virtual Try-On Preview",
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (30, 220, 30),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                f"Garment: {active_preset['label']}  (keys 1-3)",
                (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                "Press Q or ESC to quit",
                (14, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                "Space: pause  C: capture  R: refine  T: texture  M: mode  S: skip TripoSR wait",
                (14, 156),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                (
                    f"Refine: {'on' if refine_enabled else 'off'}"
                    f"{(' (' + refiner.mode + ')') if (refiner is not None and refine_enabled) else ''}"
                    f"  Texture: {('off' if (not texture_enabled) else texture_mode)}"
                ),
                (14, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
            triposr_text, triposr_color = _format_triposr_status(triposr_state)
            cv2.putText(
                output,
                triposr_text,
                (14, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                triposr_color,
                1,
                cv2.LINE_AA,
            )
            if triposr_state["status"] == "failed":
                err = (triposr_state.get("error") or "").replace("\n", " ").strip()
                if err:
                    cv2.putText(
                        output,
                        err[-95:],
                        (14, 148),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
        else:
            cloth_initialized = False
            last_points = None
            body_model.reset()
            anchor_state["anchors"] = None
            cv2.putText(
                output,
                "Pose not detected - step back and keep torso visible",
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if fps_smooth:
            cv2.putText(
                output,
                f"{fps_smooth:4.1f} fps",
                (output.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Real-time 3D Virtual Try-On", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("r"):
            refine_enabled = not refine_enabled
            if refine_enabled and refiner is None:
                refiner = OutputRefiner(project_root=project_root, enable_esrgan=bool(esrgan))
        if key == ord("t"):
            texture_enabled = not texture_enabled
        if key == ord("m"):
            texture_mode = "mesh" if texture_mode == "quad" else "quad"
        if key == ord(" "):
            paused = True
        if key == ord("c"):
            os.makedirs(os.path.join(project_root, "generated", "captures"), exist_ok=True)
            out_path = os.path.join(project_root, "generated", "captures", f"capture_{screenshot_idx:04d}.png")
            screenshot_idx += 1
            cv2.imwrite(out_path, output)
            print(f"Saved capture: {out_path}")
        if key in (ord("1"), ord("2"), ord("3")):
            preset_idx = int(chr(key)) - 1
            if 0 <= preset_idx < len(preset_keys):
                current_key = preset_keys[preset_idx]
                active_preset, cloth_mesh, cloth_sim, cloth_texture = _create_garment_runtime(project_root, current_key)
                next_preset = get_preset(current_key)
                triposr_image = find_2d_garment_image(project_root, [next_preset.get("texture_path", "")])
                if triposr_mode != "off":
                    triposr_gen = TripoSRObjGenerator(project_root, triposr_image)
                    triposr_gen.start_async(force=False)
                    triposr_applied = False
                else:
                    triposr_gen = None
                    triposr_applied = True
                cloth_initialized = False

    renderer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if not OPENGL_AVAILABLE:
        print("PyOpenGL/glfw not available: using software fallback renderer.")
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--triposr", default="auto", choices=("off", "auto", "require"))
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--esrgan", action="store_true", help="Enable ESRGAN refinement (slow; optional).")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    args = parser.parse_args()
    run_virtual_tryon_3d(
        triposr_mode=args.triposr,
        refine=(not args.no_refine),
        esrgan=bool(args.esrgan),
        camera_index=args.camera,
    )
