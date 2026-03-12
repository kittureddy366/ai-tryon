import numpy as np


class MassSpringClothSimulator:
    def __init__(self, mesh, stiffness=260.0, damping=0.985, gravity=980.0):
        self.faces = mesh["faces"]
        self.anchors = mesh["anchors"]
        self.rest_vertices = mesh["vertices"].copy()
        self.vertex_count = self.rest_vertices.shape[0]

        self.positions = self.rest_vertices.copy()
        self.velocities = np.zeros_like(self.positions)

        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.gravity = float(gravity)

        if "rows" in mesh and "cols" in mesh:
            self.springs = self._build_springs_grid(mesh["rows"], mesh["cols"])
        else:
            self.springs = self._build_springs_from_faces(mesh["faces"])
        self._u_coords, self._v_coords = self._build_uv_coords()
        self.follow_strength = 0.09
        self.global_scale = 1.0
        self.length_scale = 1.0
        self.anchor_affine = None

    def _build_uv_coords(self):
        v = self.rest_vertices
        min_x = float(np.min(v[:, 0]))
        max_x = float(np.max(v[:, 0]))
        min_y = float(np.min(v[:, 1]))
        max_y = float(np.max(v[:, 1]))
        u = (v[:, 0] - min_x) / max(max_x - min_x, 1e-6)
        vv = (max_y - v[:, 1]) / max(max_y - min_y, 1e-6)
        return u.astype(np.float32), vv.astype(np.float32)

    def initialize_from_anchors(self, anchor_targets):
        ls = anchor_targets["left_shoulder"]
        rs = anchor_targets["right_shoulder"]
        lh = anchor_targets["left_hip"]
        rh = anchor_targets["right_hip"]
        chest = anchor_targets["chest"]
        self.global_scale = float(anchor_targets.get("garment_scale", 1.0))
        self.length_scale = float(anchor_targets.get("garment_length_scale", 1.0))

        self.anchor_affine = self._solve_anchor_affine(anchor_targets)
        if self.anchor_affine is not None:
            self.positions = self._apply_affine(self.anchor_affine)
        else:
            # Fallback to previous panel mapping when anchor solve is unavailable.
            shoulder_center = (ls + rs) * 0.5
            hip_center = (lh + rh) * 0.5
            x_axis = rs - ls
            x_norm = np.linalg.norm(x_axis) + 1e-6
            x_axis = x_axis / x_norm
            for idx, v in enumerate(self.rest_vertices):
                x_local = float(v[0])
                y_local = float(v[1])
                y01 = np.clip((0.9 - y_local) * 0.5, 0.0, 1.0)
                y01_len = np.clip(y01 * self.length_scale, 0.0, 1.0)
                base = shoulder_center * (1.0 - y01_len) + hip_center * y01_len
                width_taper = 1.02 - 0.10 * y01
                offset_x = x_axis * x_local * x_norm * 0.48 * width_taper * self.global_scale
                depth = np.array([0.0, 0.0, -2.5 - (1.0 - abs(x_local)) * 1.2], dtype=np.float32)
                self.positions[idx] = base + offset_x + depth

        centerness = 1.0 - np.abs(self._u_coords - 0.5) * 2.0
        self.positions[:, 2] -= (2.0 + centerness * 1.0).astype(np.float32)
        if "chest" in self.anchors:
            self.positions[self.anchors["chest"]] = chest
        self._recompute_rest_lengths()
        self.velocities[:] = 0.0

    def _recompute_rest_lengths(self):
        pairs = self.springs["pairs"]
        delta = self.positions[pairs[:, 1]] - self.positions[pairs[:, 0]]
        lengths = np.linalg.norm(delta, axis=1)
        # Keep a little ease to avoid a rigid painted-on look.
        self.springs["rest"] = np.clip(lengths * 1.03, 1e-3, None).astype(np.float32)

    def _build_springs_grid(self, rows, cols):
        springs = []

        def idx(r, c):
            return r * cols + c

        for r in range(rows):
            for c in range(cols):
                a = idx(r, c)
                if c + 1 < cols:
                    b = idx(r, c + 1)
                    springs.append((a, b))
                if r + 1 < rows:
                    b = idx(r + 1, c)
                    springs.append((a, b))
                if r + 1 < rows and c + 1 < cols:
                    springs.append((a, idx(r + 1, c + 1)))
                if r + 1 < rows and c - 1 >= 0:
                    springs.append((a, idx(r + 1, c - 1)))

        rest_lengths = []
        for i, j in springs:
            d = self.positions[j] - self.positions[i]
            rest_lengths.append(float(np.linalg.norm(d)))

        return {
            "pairs": np.array(springs, dtype=np.int32),
            "rest": np.array(rest_lengths, dtype=np.float32),
        }

    def _build_springs_from_faces(self, faces):
        edges = set()
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            edges.add(tuple(sorted((a, b))))
            edges.add(tuple(sorted((b, c))))
            edges.add(tuple(sorted((c, a))))

        springs = sorted(edges)
        rest_lengths = []
        for i, j in springs:
            d = self.positions[j] - self.positions[i]
            rest_lengths.append(float(np.linalg.norm(d)))

        return {
            "pairs": np.array(springs, dtype=np.int32),
            "rest": np.array(rest_lengths, dtype=np.float32),
        }

    def reset_from_body(self, anchor_targets):
        for anchor_name, vertex_idx in self.anchors.items():
            if anchor_name in anchor_targets:
                self.positions[vertex_idx] = anchor_targets[anchor_name]
                self.velocities[vertex_idx] = 0.0

    def step(self, dt, anchor_targets, body_mesh=None):
        dt = max(float(dt), 1.0 / 240.0)
        self.global_scale = float(anchor_targets.get("garment_scale", self.global_scale))
        self.length_scale = float(anchor_targets.get("garment_length_scale", self.length_scale))

        forces = np.zeros_like(self.positions)
        forces[:, 1] += self.gravity

        pairs = self.springs["pairs"]
        rest = self.springs["rest"]

        p0 = self.positions[pairs[:, 0]]
        p1 = self.positions[pairs[:, 1]]
        delta = p1 - p0
        lengths = np.linalg.norm(delta, axis=1) + 1e-6
        direction = delta / lengths[:, None]
        stretch = lengths - rest
        force_mag = self.stiffness * stretch
        spring_force = direction * force_mag[:, None]

        np.add.at(forces, pairs[:, 0], spring_force)
        np.add.at(forces, pairs[:, 1], -spring_force)

        self.velocities += forces * dt * 0.0014
        self.velocities *= self.damping
        self.positions += self.velocities * dt

        target_surface = self._target_surface(anchor_targets)
        self.positions += (target_surface - self.positions) * self.follow_strength

        for anchor_name, vertex_idx in self.anchors.items():
            if anchor_name in anchor_targets:
                target = anchor_targets[anchor_name]
                current = self.positions[vertex_idx]
                correction = (target - current) * 0.92
                self.positions[vertex_idx] += correction
                self.velocities[vertex_idx] = 0.0

        if body_mesh is not None:
            self._resolve_body_collision(body_mesh)

        return self.positions.copy()

    def _target_surface(self, anchor_targets):
        ls = anchor_targets["left_shoulder"]
        rs = anchor_targets["right_shoulder"]
        lh = anchor_targets["left_hip"]
        rh = anchor_targets["right_hip"]
        chest = anchor_targets["chest"]
        le = anchor_targets.get("left_elbow", ls)
        re = anchor_targets.get("right_elbow", rs)
        neck = anchor_targets.get("neck", (ls + rs) * 0.5)
        shoulder_angle = float(anchor_targets.get("shoulder_angle", 0.0))

        self.anchor_affine = self._solve_anchor_affine(anchor_targets)
        if self.anchor_affine is not None:
            base = self._apply_affine(self.anchor_affine)
        else:
            u = self._u_coords[:, None]
            v = self._v_coords[:, None]
            one = np.ones_like(u)
            top = ls[None, :] * (one - u) + rs[None, :] * u
            raw_bottom = lh[None, :] * (one - u) + rh[None, :] * u
            bottom = raw_bottom * 0.84 + top * 0.16
            v_len = np.clip(v * self.length_scale, 0.0, 1.0)
            base = top * (one - v_len) + bottom * v_len

        u = self._u_coords[:, None]
        v = self._v_coords[:, None]
        one = np.ones_like(u)

        top_y = ((ls[1] + rs[1]) * 0.5)
        base[:, 1] = top_y + (base[:, 1] - top_y) * self.length_scale
        center_x = float((ls[0] + rs[0] + lh[0] + rh[0]) * 0.25)
        base[:, 0] = center_x + (base[:, 0] - center_x) * self.global_scale

        chest_pull = np.exp(-((v - 0.30) ** 2) / 0.02) * np.exp(-((u - 0.50) ** 2) / 0.06)
        base += (chest[None, :] - base) * chest_pull * 0.16
        neck_pull = np.exp(-((v - 0.03) ** 2) / 0.003) * np.exp(-((u - 0.50) ** 2) / 0.08)
        base += (neck[None, :] - base) * neck_pull * 0.22

        left_edge = np.clip((0.24 - u) / 0.24, 0.0, 1.0)
        right_edge = np.clip((u - 0.76) / 0.24, 0.0, 1.0)
        sleeve_zone = np.exp(-((v - 0.16) ** 2) / 0.02)
        base += (le[None, :] - base) * (left_edge * sleeve_zone * 0.12)
        base += (re[None, :] - base) * (right_edge * sleeve_zone * 0.12)

        tilt = np.clip(np.deg2rad(shoulder_angle), -0.55, 0.55)
        base[:, 0] += (v[:, 0] - 0.30) * np.sin(tilt) * 8.0 * self.global_scale

        centerness = 1.0 - np.abs(self._u_coords - 0.5) * 2.0
        base[:, 2] -= (2.0 + centerness * 1.0).astype(np.float32)
        return base

    def _solve_anchor_affine(self, anchor_targets):
        keys = [k for k in self.anchors.keys() if k in anchor_targets]
        if len(keys) < 3:
            return None
        src = np.array([self.rest_vertices[self.anchors[k]] for k in keys], dtype=np.float32)
        dst = np.array([anchor_targets[k] for k in keys], dtype=np.float32)
        src_h = np.concatenate([src, np.ones((src.shape[0], 1), dtype=np.float32)], axis=1)
        try:
            m, *_ = np.linalg.lstsq(src_h, dst, rcond=None)
            return m.astype(np.float32)  # 4x3
        except Exception:
            return None

    def _apply_affine(self, m):
        vh = np.concatenate(
            [self.rest_vertices, np.ones((self.rest_vertices.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        return (vh @ m).astype(np.float32)

    def _resolve_body_collision(self, body_mesh):
        verts = body_mesh.vertices
        center = np.mean(verts, axis=0)
        radii = np.std(verts, axis=0) * 1.85 + np.array([8.0, 8.0, 12.0], dtype=np.float32)

        rel = self.positions - center[None, :]
        ellipsoid_value = (
            (rel[:, 0] / radii[0]) ** 2
            + (rel[:, 1] / radii[1]) ** 2
            + (rel[:, 2] / radii[2]) ** 2
        )
        inside = ellipsoid_value < 1.0
        if not np.any(inside):
            return

        rel_inside = rel[inside]
        norms = np.linalg.norm(rel_inside, axis=1)[:, None] + 1e-6
        push_dir = rel_inside / norms
        penetration = (1.0 - ellipsoid_value[inside])[:, None]
        self.positions[inside] += push_dir * penetration * 4.2
        self.velocities[inside] *= 0.75
