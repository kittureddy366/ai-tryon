import os

import numpy as np


def load_obj_garment_mesh(path, max_faces=2200):
    if not path or not os.path.exists(path):
        return None

    positions = []
    texcoords = []
    faces_raw = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    texcoords.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for tok in parts:
                    vals = tok.split("/")
                    vi = int(vals[0]) if vals and vals[0] else 0
                    vti = int(vals[1]) if len(vals) > 1 and vals[1] else 0
                    face.append((vi, vti))
                if len(face) >= 3:
                    faces_raw.append(face)

    if not positions or not faces_raw:
        return None

    positions = np.array(positions, dtype=np.float32)
    texcoords = np.array(texcoords, dtype=np.float32) if texcoords else None

    key_to_idx = {}
    verts = []
    uvs = []
    faces = []

    def add_corner(vi, vti):
        key = (vi, vti)
        if key in key_to_idx:
            return key_to_idx[key]
        p = positions[vi - 1] if vi > 0 else np.zeros(3, dtype=np.float32)
        if texcoords is not None and vti > 0:
            uv = texcoords[vti - 1]
        else:
            uv = np.array([0.5, 0.5], dtype=np.float32)
        idx = len(verts)
        key_to_idx[key] = idx
        verts.append(p.tolist())
        uvs.append(uv.tolist())
        return idx

    for face in faces_raw:
        idxs = [add_corner(vi, vti) for (vi, vti) in face]
        for i in range(1, len(idxs) - 1):
            faces.append([idxs[0], idxs[i], idxs[i + 1]])

    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    uvs = np.array(uvs, dtype=np.float32)
    if faces.size == 0 or verts.size == 0:
        return None

    verts, faces, uvs = _simplify_mesh(verts, faces, uvs, max_faces=max_faces)

    # Normalize OBJ into cloth local space.
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    min_x, max_x = float(np.min(x)), float(np.max(x))
    min_y, max_y = float(np.min(y)), float(np.max(y))
    min_z, max_z = float(np.min(z)), float(np.max(z))

    w = max(max_x - min_x, 1e-6)
    h = max(max_y - min_y, 1e-6)
    d = max(max_z - min_z, 1e-6)

    xn = (x - (min_x + max_x) * 0.5) / (w * 0.5)
    yn01 = (max_y - y) / h
    zn = (z - (min_z + max_z) * 0.5) / d

    out = np.zeros_like(verts)
    out[:, 0] = xn
    out[:, 1] = 0.72 - yn01 * 1.32
    out[:, 2] = zn * 0.08

    uvs = _normalize_uvs(uvs)

    if np.allclose(uvs, 0.5):
        uu = (xn + 1.0) * 0.5
        vv = np.clip(yn01, 0.0, 1.0)
        uvs = np.column_stack([uu, vv]).astype(np.float32)

    anchors = _estimate_anchors(out)
    return {
        "vertices": out.astype(np.float32),
        "faces": faces.astype(np.int32),
        "uvs": uvs.astype(np.float32),
        "anchors": anchors,
    }


def _normalize_uvs(uvs):
    if uvs.size == 0:
        return uvs

    out = uvs.copy().astype(np.float32)
    u = out[:, 0]
    v = out[:, 1]

    u_min, u_max = float(np.min(u)), float(np.max(u))
    v_min, v_max = float(np.min(v)), float(np.max(v))

    u_span = u_max - u_min
    v_span = v_max - v_min

    # Many exported OBJ files store UVs in atlas/pixel-like coordinates.
    # Convert to [0, 1] robustly.
    out_of_unit = (u_min < -0.05) or (u_max > 1.05) or (v_min < -0.05) or (v_max > 1.05)
    if u_span > 2.0 or v_span > 2.0 or out_of_unit:
        if u_span > 1e-6:
            u = (u - u_min) / u_span
        else:
            u = np.full_like(u, 0.5)
        if v_span > 1e-6:
            v = (v - v_min) / v_span
        else:
            v = np.full_like(v, 0.5)
    else:
        # Keep UVs inside the texture without tiling/wrapping.
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

    out[:, 0] = np.clip(u, 0.0, 1.0)
    out[:, 1] = np.clip(v, 0.0, 1.0)
    return out


def planar_uvs_from_vertices(vertices):
    if vertices is None or vertices.size == 0:
        return None
    v = vertices.astype(np.float32)
    x = v[:, 0]
    y = v[:, 1]
    min_x, max_x = float(np.min(x)), float(np.max(x))
    min_y, max_y = float(np.min(y)), float(np.max(y))
    u = (x - min_x) / max(max_x - min_x, 1e-6)
    vv = (max_y - y) / max(max_y - min_y, 1e-6)
    return np.column_stack([np.clip(u, 0.0, 1.0), np.clip(vv, 0.0, 1.0)]).astype(np.float32)


def _simplify_mesh(vertices, faces, uvs, max_faces=2200):
    face_count = int(faces.shape[0])
    if face_count <= max_faces:
        return vertices, faces, uvs

    # Uniform stride sampling keeps broad garment silhouette while reducing cost.
    step = max(1, int(np.ceil(face_count / float(max_faces))))
    keep_idx = np.arange(0, face_count, step, dtype=np.int32)
    sampled_faces = faces[keep_idx]

    used = np.unique(sampled_faces.reshape(-1))
    remap = np.full(vertices.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    compact_faces = remap[sampled_faces]
    compact_vertices = vertices[used]
    compact_uvs = uvs[used]

    return compact_vertices, compact_faces, compact_uvs


def _estimate_anchors(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]

    def pick(tx, ty, wx=1.25, wy=1.0):
        score = ((x - tx) ** 2) * wx + ((y - ty) ** 2) * wy
        return int(np.argmin(score))

    return {
        "left_shoulder": pick(-0.56, 0.67),
        "right_shoulder": pick(0.56, 0.67),
        "chest": pick(0.0, 0.34, wx=1.0, wy=1.4),
        "left_hip": pick(-0.34, -0.55),
        "right_hip": pick(0.34, -0.55),
    }
