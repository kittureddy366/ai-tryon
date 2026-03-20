import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from garment.cloth_mesh import create_shirt_cloth_mesh  # noqa: E402
from garment.garment_library import get_preset, list_preset_keys  # noqa: E402
from garment.obj_garment_loader import load_obj_garment_mesh, planar_uvs_from_vertices  # noqa: E402
from garment.triposr_obj_generator import TripoSRObjGenerator, find_2d_garment_image  # noqa: E402
from physics_engine.mass_spring_cloth import MassSpringClothSimulator  # noqa: E402
from pose.body_mesh_generator import PreciseBodyMeshGenerator  # noqa: E402
from pose.body_pose import BodyPoseDetector  # noqa: E402
from rendering.frame_processor import (  # noqa: E402
    extract_pose_landmarks,
    get_person_segmentation_mask,
    get_result_segmentation_mask,
)
from rendering.output_refiner import OutputRefiner  # noqa: E402
from rendering.realtime_mesh_renderer import RealtimeMeshRenderer  # noqa: E402


def _prepare_garment_texture(path: str | None):
    if not path:
        return None
    texture = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if texture is None:
        return None
    if len(texture.shape) == 2:
        texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGRA)
    elif texture.shape[2] == 3:
        bgr = texture
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        not_white = gray < 242
        has_color = hsv[:, :, 1] > 20
        dark_object = gray < 205
        fg = (not_white & dark_object) | (has_color & (gray < 250))
        fg = fg.astype(np.uint8) * 255

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


def _build_anchor_targets(body_mesh, points):
    torso_z = float(body_mesh.joints.get("torso_center", np.array([0.0, 0.0, -8.0], dtype=np.float32))[2])
    chest_z = float(body_mesh.joints.get("chest", np.array([0.0, 0.0, torso_z - 1.0], dtype=np.float32))[2])
    shoulder_width = float(
        np.hypot(
            points["right_shoulder"]["x"] - points["left_shoulder"]["x"],
            points["right_shoulder"]["y"] - points["left_shoulder"]["y"],
        )
    )
    torso_height = float(
        np.hypot(
            ((points["left_hip"]["x"] + points["right_hip"]["x"]) * 0.5)
            - ((points["left_shoulder"]["x"] + points["right_shoulder"]["x"]) * 0.5),
            ((points["left_hip"]["y"] + points["right_hip"]["y"]) * 0.5)
            - ((points["left_shoulder"]["y"] + points["right_shoulder"]["y"]) * 0.5),
        )
    )
    neck_xy = (
        int((points["left_shoulder"]["x"] + points["right_shoulder"]["x"]) * 0.5),
        int((points["left_shoulder"]["y"] + points["right_shoulder"]["y"]) * 0.5 - 0.07 * shoulder_width),
    )
    shoulder_angle = float(
        np.degrees(
            np.arctan2(
                points["right_shoulder"]["y"] - points["left_shoulder"]["y"],
                points["right_shoulder"]["x"] - points["left_shoulder"]["x"],
            )
        )
    )
    garment_scale = float(np.clip(shoulder_width / 165.0, 0.82, 1.24))
    garment_length_scale = float(np.clip(torso_height / 195.0, 0.84, 1.20))

    shoulder_pad = shoulder_width * 0.06
    left_shoulder_xy = np.array(
        [points["left_shoulder"]["x"] - shoulder_pad, points["left_shoulder"]["y"]], dtype=np.float32
    )
    right_shoulder_xy = np.array(
        [points["right_shoulder"]["x"] + shoulder_pad, points["right_shoulder"]["y"]], dtype=np.float32
    )
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
        [points["chest_midpoint"]["x"], points["chest_midpoint"]["y"], chest_z - 0.8], dtype=np.float32
    )
    anchors["neck"] = np.array([neck_xy[0], neck_xy[1], chest_z - 1.0], dtype=np.float32)
    anchors["garment_scale"] = garment_scale
    anchors["garment_length_scale"] = garment_length_scale
    anchors["shoulder_width"] = shoulder_width
    anchors["torso_height"] = torso_height
    anchors["shoulder_angle"] = shoulder_angle
    return anchors


def _create_garment_runtime(project_root: Path, preset_key: str, texture_override: str | None, obj_override: str | None):
    preset = get_preset(preset_key).copy()

    mesh = None
    if obj_override:
        mesh = load_obj_garment_mesh(obj_override)
    elif preset.get("garment_obj_path"):
        candidate = project_root / str(preset["garment_obj_path"])
        if candidate.is_file():
            mesh = load_obj_garment_mesh(str(candidate))

    if mesh is None:
        mesh = create_shirt_cloth_mesh(rows=preset["mesh_rows"], cols=preset["mesh_cols"])

    simulator = MassSpringClothSimulator(mesh, **preset["physics"])

    texture_path = texture_override or preset.get("texture_path", "")
    if texture_path:
        texture_candidate = project_root / str(texture_path)
        if texture_candidate.is_file():
            texture_path = str(texture_candidate)
    texture_rgba = _prepare_garment_texture(texture_path)

    return preset, mesh, simulator, texture_rgba


def _resolve_triposr_input_image(project_root: Path, explicit_path: str | None, fallback_texture: str | None):
    if explicit_path:
        p = Path(explicit_path).expanduser()
        return str(p) if p.is_file() else None
    if fallback_texture:
        t = Path(fallback_texture)
        if not t.is_absolute():
            t = project_root / t
        if t.is_file():
            return str(t)
    candidate = find_2d_garment_image(str(project_root), preferred_paths=[])
    return candidate if candidate and Path(candidate).is_file() else None


def _run_triposr_blocking(project_root: Path, image_path: str, force: bool):
    gen = TripoSRObjGenerator(str(project_root), image_path)
    gen.start_async(force=force)
    last = None
    while True:
        snap = gen.snapshot()
        if snap.get("status") != last:
            last = snap.get("status")
            if last == "running":
                print("TripoSR: generating garment OBJ...")
        if snap.get("status") in {"ready", "failed"}:
            return snap
        time.sleep(0.25)


def run_on_frame(
    frame_bgr: np.ndarray,
    pose_detector: BodyPoseDetector,
    body_model: PreciseBodyMeshGenerator,
    renderer: RealtimeMeshRenderer,
    refiner: OutputRefiner | None,
    preset,
    cloth_mesh,
    cloth_sim: MassSpringClothSimulator,
    cloth_texture_rgba,
    timestamp_ms: int | None,
    sim_steps: int,
    initialize_cloth: bool,
    texture_mode: str,
):
    result = pose_detector.detect_pose(frame_bgr, timestamp_ms)
    points = extract_pose_landmarks(result, frame_bgr.shape)
    if points is None:
        raise RuntimeError("Pose not detected. Make sure the full torso is visible in the image/video.")

    body_mesh = body_model.generate(result, frame_bgr.shape)
    if body_mesh is None:
        raise RuntimeError("Body mesh generation failed.")

    anchors = _build_anchor_targets(body_mesh, points)
    person_mask = get_result_segmentation_mask(result, frame_bgr.shape, threshold=0.30)
    if person_mask is None:
        person_mask = get_person_segmentation_mask(frame_bgr, frame_bgr.shape, threshold=0.45)

    if initialize_cloth:
        cloth_sim.initialize_from_anchors(anchors)
    dt = 1.0 / 60.0
    cloth_vertices = None
    for _ in range(max(1, int(sim_steps))):
        cloth_vertices = cloth_sim.step(dt, anchors, body_mesh=body_mesh)

    output = renderer.render_overlay(
        frame_bgr,
        body_mesh.vertices,
        body_mesh.faces,
        cloth_vertices,
        cloth_mesh["faces"],
        cloth_uvs=cloth_mesh.get("uvs"),
        cloth_texture_rgba=cloth_texture_rgba,
        cloth_color_bgra=preset["cloth_color_bgra"],
        pose_points=points,
        anchor_targets=anchors,
        person_mask=person_mask,
        texture_mode=texture_mode,
    )

    if refiner is not None:
        output = refiner.refine(output, points=points)
    return output


def _is_video_path(path: Path):
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main():
    parser = argparse.ArgumentParser(
        description="Headless (no webcam/GUI) 3D try-on runner (image/video in -> output file)."
    )
    parser.add_argument("--input", required=True, help="Input image or video path.")
    parser.add_argument("--output", required=True, help="Output image/video path.")
    parser.add_argument("--preset", default="tshirt_blue", help=f"Garment preset key. Options: {', '.join(list_preset_keys())}")
    parser.add_argument("--texture", default="", help="Optional garment texture override (PNG/JPG).")
    parser.add_argument("--obj", default="", help="Optional garment OBJ override.")
    parser.add_argument(
        "--triposr",
        default="auto",
        choices=("off", "auto", "require"),
        help="Generate garment OBJ from 2D image using TripoSR (auto=try then fallback).",
    )
    parser.add_argument("--triposr-image", default="", help="2D garment image to feed TripoSR (defaults to --texture/preset texture).")
    parser.add_argument("--triposr-force", action="store_true", help="Force regenerate TripoSR OBJ (ignore cache).")
    parser.add_argument("--sim-steps", type=int, default=14, help="Cloth sim steps per frame (higher = more drape, slower).")
    parser.add_argument("--no-refine", action="store_true", help="Disable output refinement.")
    parser.add_argument("--texture-mode", default="mesh", choices=("mesh", "quad"), help="How to apply garment texture.")
    parser.add_argument("--max-frames", type=int, default=0, help="For videos: stop after N frames (0 = all).")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    project_root = PROJECT_ROOT
    preset, cloth_mesh, cloth_sim, cloth_texture = _create_garment_runtime(
        project_root, args.preset, args.texture or None, args.obj or None
    )

    is_video = _is_video_path(input_path)
    pose_detector = BodyPoseDetector(running_mode="video" if is_video else "image")
    body_model = PreciseBodyMeshGenerator(limb_sides=12, limb_rings=9, torso_rings=14, torso_sides=20)
    renderer = RealtimeMeshRenderer()
    refiner = None if args.no_refine else OutputRefiner(project_root=str(project_root))

    triposr_mode = (args.triposr or "auto").strip().lower()
    if triposr_mode != "off":
        triposr_input = _resolve_triposr_input_image(
            project_root,
            explicit_path=args.triposr_image or None,
            fallback_texture=(args.texture or preset.get("texture_path") or None),
        )
        if not triposr_input:
            if triposr_mode == "require":
                raise SystemExit("TripoSR required, but no 2D garment image found (use --triposr-image or --texture).")
        else:
            snap = _run_triposr_blocking(project_root, triposr_input, force=bool(args.triposr_force))
            if snap.get("status") == "ready" and snap.get("obj_path"):
                mesh = load_obj_garment_mesh(snap["obj_path"], max_faces=6500)
                if mesh is not None:
                    planar_uvs = planar_uvs_from_vertices(mesh.get("vertices"))
                    if planar_uvs is not None:
                        mesh["uvs"] = planar_uvs
                    cloth_mesh = mesh
                    cloth_sim = MassSpringClothSimulator(cloth_mesh, **preset["physics"])
                    print(f"TripoSR: using generated OBJ: {snap['obj_path']}")
                elif triposr_mode == "require":
                    raise SystemExit("TripoSR required, but generated OBJ could not be loaded.")
            else:
                err = (snap.get("error") or "").strip()
                if triposr_mode == "require":
                    raise SystemExit(f"TripoSR required, but failed: {err or 'unknown error'}")
                if err:
                    print(f"TripoSR failed (fallback to cloth grid): {err[-300:]}")

    if is_video:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {input_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Empty video.")
            h, w = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise SystemExit(f"Could not create video writer: {output_path}")

        frame_idx = 0
        started = time.time()
        cloth_initialized = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int((frame_idx / max(fps, 1e-6)) * 1000.0)
            try:
                out = run_on_frame(
                    frame,
                    pose_detector,
                    body_model,
                    renderer,
                    refiner,
                    preset,
                    cloth_mesh,
                    cloth_sim,
                    cloth_texture,
                    timestamp_ms=timestamp_ms,
                    sim_steps=args.sim_steps,
                    initialize_cloth=not cloth_initialized,
                    texture_mode=args.texture_mode,
                )
                cloth_initialized = True
            except RuntimeError as exc:
                cloth_initialized = False
                out = frame
                cv2.putText(
                    out,
                    f"Try-on skipped: {str(exc)[:60]}",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(out)
            frame_idx += 1
            if args.max_frames and frame_idx >= int(args.max_frames):
                break
            if frame_idx % 30 == 0:
                elapsed = time.time() - started
                print(f"Processed {frame_idx} frames ({frame_idx / max(elapsed, 1e-6):.1f} fps)")

        writer.release()
        cap.release()
        renderer.close()
        print(f"Wrote: {output_path}")
        return

    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"Could not read image: {input_path}")

    out = run_on_frame(
        frame,
        pose_detector,
        body_model,
        renderer,
        refiner,
        preset,
        cloth_mesh,
        cloth_sim,
        cloth_texture,
        timestamp_ms=None,
        sim_steps=args.sim_steps,
        initialize_cloth=True,
        texture_mode=args.texture_mode,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), out):
        raise SystemExit(f"Could not write output: {output_path}")
    renderer.close()
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(str(exc))
