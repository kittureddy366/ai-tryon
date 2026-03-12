import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BodyMesh:
    vertices: np.ndarray
    faces: np.ndarray
    joints: dict


_POSE_IDS = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class PreciseBodyMeshGenerator:
    """
    Higher-accuracy body mesh fit from MediaPipe pose landmarks.
    - Uses world landmarks with similarity alignment when available.
    - Applies temporal joint smoothing for jitter reduction.
    - Builds torso as a lofted elliptical volume aligned to body axes.
    """

    def __init__(self, limb_sides=12, limb_rings=9, torso_rings=14, torso_sides=20, smooth_alpha=0.42):
        self.limb_sides = int(limb_sides)
        self.limb_rings = int(limb_rings)
        self.torso_rings = int(torso_rings)
        self.torso_sides = int(torso_sides)
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.1, 0.9))
        self._prev_joints = None

    def reset(self):
        self._prev_joints = None

    def generate(self, pose_result, frame_shape):
        h, w = frame_shape[:2]
        joints = self._extract_joints_3d(pose_result, w, h)
        if joints is None:
            self._prev_joints = None
            return None

        joints = self._smooth_joints(joints)

        ls = joints["left_shoulder"]
        rs = joints["right_shoulder"]
        lh = joints["left_hip"]
        rh = joints["right_hip"]
        le = joints["left_elbow"]
        re = joints["right_elbow"]
        nose = joints["nose"]
        lear = joints["left_ear"]
        rear = joints["right_ear"]

        shoulder_center = (ls + rs) * 0.5
        hip_center = (lh + rh) * 0.5
        torso_center = (shoulder_center + hip_center) * 0.5

        shoulder_w = max(float(np.linalg.norm(rs - ls)), 24.0)
        hip_w = max(float(np.linalg.norm(rh - lh)), 20.0)
        torso_h = max(float(np.linalg.norm(hip_center - shoulder_center)), 45.0)

        x_axis = (rs - ls) + (rh - lh) * 0.4
        x_axis /= np.linalg.norm(x_axis) + 1e-6
        y_axis = hip_center - shoulder_center
        y_axis /= np.linalg.norm(y_axis) + 1e-6
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-6

        chest_center = shoulder_center * 0.62 + hip_center * 0.38
        waist_center = shoulder_center * 0.34 + hip_center * 0.66

        torso_v, torso_f = self._torso_loft(
            shoulder_center,
            hip_center,
            x_axis,
            y_axis,
            z_axis,
            shoulder_w,
            hip_w,
            torso_h,
        )

        pelvis_v, pelvis_f = self._ellipsoid_axes(
            hip_center + y_axis * (torso_h * 0.09),
            x_axis,
            y_axis,
            z_axis,
            hip_w * 0.30,
            torso_h * 0.20,
            hip_w * 0.20,
            8,
            16,
        )

        neck = shoulder_center - y_axis * (torso_h * 0.14)
        ear_mid = (lear + rear) * 0.5
        head_center = (nose * 0.42 + ear_mid * 0.30 + neck * 0.28)
        head_r = shoulder_w * 0.19
        head_v, head_f = self._ellipsoid_axes(
            head_center,
            x_axis,
            y_axis,
            z_axis,
            head_r,
            head_r * 1.22,
            head_r * 0.96,
            11,
            16,
        )

        upper_arm_r = shoulder_w * 0.10
        lower_arm_r = shoulder_w * 0.082
        thigh_r = hip_w * 0.112
        shin_r = hip_w * 0.088

        limb_defs = [
            ("left_shoulder", "left_elbow", upper_arm_r),
            ("left_elbow", "left_wrist", lower_arm_r),
            ("right_shoulder", "right_elbow", upper_arm_r),
            ("right_elbow", "right_wrist", lower_arm_r),
            ("left_hip", "left_knee", thigh_r),
            ("left_knee", "left_ankle", shin_r),
            ("right_hip", "right_knee", thigh_r),
            ("right_knee", "right_ankle", shin_r),
        ]

        meshes = [(torso_v, torso_f), (pelvis_v, pelvis_f), (head_v, head_f)]
        for a, b, r in limb_defs:
            cap_v, cap_f = self._capsule(joints[a], joints[b], max(float(r), 4.5))
            meshes.append((cap_v, cap_f))

        # Shoulder bridge for smoother garment anchoring area.
        shoulder_bridge_v, shoulder_bridge_f = self._capsule(ls, rs, shoulder_w * 0.12)
        meshes.append((shoulder_bridge_v, shoulder_bridge_f))

        vertices, faces = self._merge_meshes(meshes)

        joints["chest"] = chest_center.astype(np.float32)
        joints["waist"] = waist_center.astype(np.float32)
        joints["torso_center"] = torso_center.astype(np.float32)
        joints["neck"] = neck.astype(np.float32)
        joints["x_axis"] = x_axis.astype(np.float32)
        joints["y_axis"] = y_axis.astype(np.float32)
        joints["z_axis"] = z_axis.astype(np.float32)

        return BodyMesh(vertices=vertices, faces=faces, joints=joints)

    def _smooth_joints(self, joints):
        if self._prev_joints is None:
            self._prev_joints = {k: v.copy() for k, v in joints.items()}
            return joints

        smoothed = {}
        for k, cur in joints.items():
            prev = self._prev_joints.get(k, cur)
            motion = float(np.linalg.norm(cur[:2] - prev[:2]))
            alpha = float(np.clip(self.smooth_alpha + motion * 0.0016, 0.28, 0.82))
            smoothed[k] = prev * (1.0 - alpha) + cur * alpha

        self._prev_joints = {k: v.copy() for k, v in smoothed.items()}
        return smoothed

    def _extract_joints_3d(self, pose_result, frame_w, frame_h):
        if not pose_result or not pose_result.pose_landmarks:
            return None

        lm2d = pose_result.pose_landmarks[0]
        has_world = bool(getattr(pose_result, "pose_world_landmarks", None)) and bool(pose_result.pose_world_landmarks)

        if has_world:
            lm3d = pose_result.pose_world_landmarks[0]
            world_xy = []
            img_xy = []
            for _, idx in _POSE_IDS.items():
                p2 = lm2d[idx]
                p3 = lm3d[idx]
                vis = min(float(getattr(p2, "visibility", 1.0)), float(getattr(p3, "visibility", 1.0)))
                if vis < 0.35:
                    continue
                world_xy.append([float(p3.x), float(p3.y)])
                img_xy.append([float(p2.x * frame_w), float(p2.y * frame_h)])

            if len(world_xy) >= 5:
                world_xy = np.array(world_xy, dtype=np.float32)
                img_xy = np.array(img_xy, dtype=np.float32)
                s, r, t = self._fit_similarity_2d(world_xy, img_xy)

                z_ref = float((lm3d[_POSE_IDS["left_hip"]].z + lm3d[_POSE_IDS["right_hip"]].z) * 0.5)
                z_scale = max(float(s * 0.95), 40.0)

                joints = {}
                for name, idx in _POSE_IDS.items():
                    p3 = lm3d[idx]
                    xy = np.array([float(p3.x), float(p3.y)], dtype=np.float32)
                    out_xy = (s * (r @ xy)) + t
                    z = (float(p3.z) - z_ref) * z_scale
                    joints[name] = np.array([out_xy[0], out_xy[1], z], dtype=np.float32)
                return joints

        # Fallback to image-space landmarks with pseudo depth.
        joints = {}
        for name, idx in _POSE_IDS.items():
            p = lm2d[idx]
            x = float(p.x * frame_w)
            y = float(p.y * frame_h)
            z = float(p.z * frame_w * 0.26)
            joints[name] = np.array([x, y, z], dtype=np.float32)
        return joints

    @staticmethod
    def _fit_similarity_2d(src, dst):
        src_m = src.mean(axis=0)
        dst_m = dst.mean(axis=0)
        src_c = src - src_m
        dst_c = dst - dst_m

        cov = (dst_c.T @ src_c) / max(src.shape[0], 1)
        u, svals, vt = np.linalg.svd(cov)
        r = u @ vt
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1.0
            r = u @ vt

        var_src = float(np.mean(np.sum(src_c * src_c, axis=1))) + 1e-8
        scale = float(np.sum(svals) / var_src)
        t = dst_m - scale * (r @ src_m)
        return scale, r.astype(np.float32), t.astype(np.float32)

    @staticmethod
    def _orthonormal_basis(axis):
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(axis, up))) > 0.92:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = np.cross(axis, up)
        u = u / (np.linalg.norm(u) + 1e-8)
        v = np.cross(axis, u)
        v = v / (np.linalg.norm(v) + 1e-8)
        return u, v

    def _capsule(self, p0, p1, radius):
        axis = p1 - p0
        length = float(np.linalg.norm(axis))
        if length < 1e-5:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            length = 1.0
        n = axis / length
        u, v = self._orthonormal_basis(n)

        rings = []
        verts = []
        for i in range(self.limb_rings + 1):
            t = i / float(self.limb_rings)
            center = p0 * (1.0 - t) + p1 * t
            ring = []
            taper = 1.0 - 0.10 * abs(t - 0.5) * 2.0
            rr = radius * taper
            for j in range(self.limb_sides):
                ang = 2.0 * math.pi * (j / float(self.limb_sides))
                pt = center + (math.cos(ang) * u + math.sin(ang) * v) * rr
                ring.append(len(verts))
                verts.append(pt.tolist())
            rings.append(ring)

        p0_idx = len(verts)
        p1_idx = len(verts) + 1
        verts.append(p0.tolist())
        verts.append(p1.tolist())

        faces = []
        for i in range(self.limb_rings):
            ra = rings[i]
            rb = rings[i + 1]
            for j in range(self.limb_sides):
                jn = (j + 1) % self.limb_sides
                faces.append([ra[j], rb[j], ra[jn]])
                faces.append([ra[jn], rb[j], rb[jn]])

        first_ring = rings[0]
        last_ring = rings[-1]
        for j in range(self.limb_sides):
            jn = (j + 1) % self.limb_sides
            faces.append([p0_idx, first_ring[jn], first_ring[j]])
            faces.append([p1_idx, last_ring[j], last_ring[jn]])

        return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

    def _torso_loft(self, shoulder_center, hip_center, x_axis, y_axis, z_axis, shoulder_w, hip_w, torso_h):
        verts = []
        faces = []
        ring_count = self.torso_rings
        side_count = self.torso_sides

        for i in range(ring_count + 1):
            t = i / float(max(ring_count, 1))
            center = shoulder_center * (1.0 - t) + hip_center * t
            center = center + z_axis * (-0.04 * shoulder_w * np.exp(-((t - 0.35) ** 2) / 0.05))

            w = (shoulder_w * 0.54) * (1.0 - t) + (hip_w * 0.50) * t
            w *= (1.0 + 0.07 * np.exp(-((t - 0.32) ** 2) / 0.03))

            d = (shoulder_w * 0.22) * (1.0 - t) + (hip_w * 0.26) * t
            d *= (1.0 + 0.12 * np.exp(-((t - 0.36) ** 2) / 0.04))

            for j in range(side_count):
                u = j / float(side_count)
                ang = 2.0 * math.pi * u
                pt = center + x_axis * (math.cos(ang) * w) + z_axis * (math.sin(ang) * d)
                verts.append(pt.tolist())

        for i in range(ring_count):
            for j in range(side_count):
                jn = (j + 1) % side_count
                i0 = i * side_count + j
                i1 = i * side_count + jn
                i2 = (i + 1) * side_count + j
                i3 = (i + 1) * side_count + jn
                faces.append([i0, i2, i1])
                faces.append([i1, i2, i3])

        return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

    def _ellipsoid_axes(self, center, x_axis, y_axis, z_axis, rx, ry, rz, lat, lon):
        verts = []
        faces = []
        for i in range(lat + 1):
            v = i / float(max(lat, 1))
            theta = math.pi * v
            cy = math.cos(theta)
            ring = math.sin(theta)
            for j in range(lon):
                u = j / float(max(lon, 1))
                phi = 2.0 * math.pi * u
                cx = ring * math.cos(phi)
                cz = ring * math.sin(phi)
                pt = center + x_axis * (cx * rx) + y_axis * (cy * ry) + z_axis * (cz * rz)
                verts.append(pt.tolist())

        for i in range(lat):
            for j in range(lon):
                jn = (j + 1) % lon
                i0 = i * lon + j
                i1 = i * lon + jn
                i2 = (i + 1) * lon + j
                i3 = (i + 1) * lon + jn
                faces.append([i0, i2, i1])
                faces.append([i1, i2, i3])

        return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

    @staticmethod
    def _merge_meshes(meshes):
        vertices = []
        faces = []
        offset = 0
        for v, f in meshes:
            vertices.append(v)
            faces.append(f + offset)
            offset += v.shape[0]
        return np.vstack(vertices), np.vstack(faces).astype(np.int32)
