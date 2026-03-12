import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BodyMesh:
    vertices: np.ndarray
    faces: np.ndarray
    joints: dict


class SimpleSMPLBodyModel:
    """SMPL-like parametric torso model fitted from pose landmarks in screen space."""

    def __init__(self, lat_steps=16, lon_steps=22):
        self.lat_steps = lat_steps
        self.lon_steps = lon_steps
        self.template_vertices, self.template_faces = self._build_template()
        self.anchor_template = self._build_anchor_map(self.template_vertices)

    def _build_template(self):
        vertices = []
        faces = []

        for i in range(self.lat_steps + 1):
            v = i / float(self.lat_steps)
            theta = math.pi * v
            y = math.cos(theta)
            ring = math.sin(theta)
            for j in range(self.lon_steps):
                u = j / float(self.lon_steps)
                phi = 2.0 * math.pi * u
                x = ring * math.cos(phi)
                z = ring * math.sin(phi)
                # Slightly flatten front/back to resemble torso profile.
                z *= 0.66
                vertices.append([x, y, z])

        for i in range(self.lat_steps):
            for j in range(self.lon_steps):
                jn = (j + 1) % self.lon_steps
                i0 = i * self.lon_steps + j
                i1 = i * self.lon_steps + jn
                i2 = (i + 1) * self.lon_steps + j
                i3 = (i + 1) * self.lon_steps + jn
                faces.append([i0, i2, i1])
                faces.append([i1, i2, i3])

        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)

    def _closest(self, vertices, target):
        diffs = vertices - target[None, :]
        dists = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(dists))

    def _build_anchor_map(self, vertices):
        return {
            "left_shoulder": self._closest(vertices, np.array([-0.95, 0.45, 0.0], dtype=np.float32)),
            "right_shoulder": self._closest(vertices, np.array([0.95, 0.45, 0.0], dtype=np.float32)),
            "chest": self._closest(vertices, np.array([0.0, 0.15, -0.55], dtype=np.float32)),
            "left_hip": self._closest(vertices, np.array([-0.7, -0.7, 0.0], dtype=np.float32)),
            "right_hip": self._closest(vertices, np.array([0.7, -0.7, 0.0], dtype=np.float32)),
        }

    def fit_from_landmarks(self, points):
        ls = np.array([points["left_shoulder"]["x"], points["left_shoulder"]["y"]], dtype=np.float32)
        rs = np.array([points["right_shoulder"]["x"], points["right_shoulder"]["y"]], dtype=np.float32)
        lh = np.array([points["left_hip"]["x"], points["left_hip"]["y"]], dtype=np.float32)
        rh = np.array([points["right_hip"]["x"], points["right_hip"]["y"]], dtype=np.float32)

        shoulder_center = (ls + rs) * 0.5
        hip_center = (lh + rh) * 0.5
        torso_center_2d = (shoulder_center + hip_center) * 0.5

        shoulder_vec = rs - ls
        shoulder_width = max(float(np.linalg.norm(shoulder_vec)), 1.0)
        torso_height = max(float(np.linalg.norm(hip_center - shoulder_center)), 1.0)

        angle = math.atan2(float(shoulder_vec[1]), float(shoulder_vec[0]))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        z_vals = np.array(
            [
                points["left_shoulder"]["z"],
                points["right_shoulder"]["z"],
                points["left_hip"]["z"],
                points["right_hip"]["z"],
            ],
            dtype=np.float32,
        )
        depth_center = float(np.mean(z_vals))
        depth_range = float(np.max(z_vals) - np.min(z_vals))
        torso_depth = max(24.0, shoulder_width * (0.22 + depth_range * 0.6))

        v = self.template_vertices.copy()
        v[:, 0] *= shoulder_width * 0.52
        v[:, 1] *= torso_height * 0.72
        v[:, 2] *= torso_depth

        xy = np.matmul(v[:, :2], rotation.T)
        v[:, 0] = xy[:, 0] + torso_center_2d[0]
        v[:, 1] = xy[:, 1] + torso_center_2d[1]
        v[:, 2] = v[:, 2] + depth_center * 120.0

        joints = {
            name: v[idx].copy() for name, idx in self.anchor_template.items()
        }

        return BodyMesh(vertices=v, faces=self.template_faces, joints=joints)
