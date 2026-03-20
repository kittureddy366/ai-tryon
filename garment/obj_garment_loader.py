import os

import numpy as np


def load_obj_garment_mesh(path, max_faces=2200, clean=True):
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

    if clean:
        out, faces, uvs = _clean_and_simplify_mesh(out, faces, uvs, max_faces=max_faces)

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


def _clean_and_simplify_mesh(vertices, faces, uvs, max_faces=2200):
    v, f, t = _weld_vertices(vertices, faces, uvs, tol=0.004)
    v, f, t = _keep_largest_component(v, f, t)
    if int(f.shape[0]) <= int(max_faces):
        return v, f, t

    # Connectivity-preserving simplification: iterative vertex clustering in cloth-local space.
    # This avoids "random triangle shards" caused by face-stride sampling on dense meshes.
    for tol in (0.006, 0.010, 0.016, 0.024, 0.032, 0.045):
        vv, ff, tt = _weld_vertices(v, f, t, tol=float(tol))
        vv, ff, tt = _keep_largest_component(vv, ff, tt)
        if int(ff.shape[0]) <= int(max_faces):
            return vv, ff, tt

    # Final fallback: if still too dense, take a spatially-uniform subset of faces.
    return _spatial_face_sample(v, f, t, max_faces=max_faces)


def _weld_vertices(vertices, faces, uvs, tol=0.004):
    if vertices is None or faces is None or vertices.size == 0 or faces.size == 0:
        return vertices, faces, uvs

    tol = float(max(tol, 1e-6))
    v = vertices.astype(np.float32)
    q = np.round(v / tol).astype(np.int32)

    key_to_new = {}
    new_index = np.empty((v.shape[0],), dtype=np.int32)
    sums = []
    counts = []
    uv_sums = [] if uvs is not None and uvs.size else None

    for i in range(v.shape[0]):
        key = (int(q[i, 0]), int(q[i, 1]), int(q[i, 2]))
        idx = key_to_new.get(key)
        if idx is None:
            idx = len(sums)
            key_to_new[key] = idx
            sums.append(v[i].copy())
            counts.append(1)
            if uv_sums is not None:
                uv_sums.append(uvs[i].astype(np.float32).copy())
        else:
            sums[idx] += v[i]
            counts[idx] += 1
            if uv_sums is not None:
                uv_sums[idx] += uvs[i].astype(np.float32)
        new_index[i] = idx

    new_v = (np.stack(sums, axis=0) / np.array(counts, dtype=np.float32)[:, None]).astype(np.float32)
    new_uv = None
    if uv_sums is not None:
        new_uv = (np.stack(uv_sums, axis=0) / np.array(counts, dtype=np.float32)[:, None]).astype(np.float32)

    new_f = new_index[faces.reshape(-1)].reshape(faces.shape).astype(np.int32)
    a = new_f[:, 0]
    b = new_f[:, 1]
    c = new_f[:, 2]
    keep = (a != b) & (b != c) & (a != c)
    new_f = new_f[keep]
    if new_f.size == 0:
        return new_v, new_f, new_uv if new_uv is not None else uvs

    used = np.unique(new_f.reshape(-1))
    remap = np.full((new_v.shape[0],), -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    compact_f = remap[new_f]
    compact_v = new_v[used]
    compact_uv = (new_uv[used] if new_uv is not None else (uvs[used] if uvs is not None and uvs.size else uvs))
    return compact_v, compact_f.astype(np.int32), compact_uv


def _keep_largest_component(vertices, faces, uvs):
    if vertices is None or faces is None or vertices.size == 0 or faces.size == 0:
        return vertices, faces, uvs

    n = int(vertices.shape[0])
    adj = [[] for _ in range(n)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].append(b)
        adj[a].append(c)
        adj[b].append(a)
        adj[b].append(c)
        adj[c].append(a)
        adj[c].append(b)

    label = np.full((n,), -1, dtype=np.int32)
    comp_sizes = []
    comp_id = 0
    for i in range(n):
        if label[i] != -1:
            continue
        stack = [i]
        label[i] = comp_id
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in adj[u]:
                if label[v] == -1:
                    label[v] = comp_id
                    stack.append(v)
        comp_sizes.append(size)
        comp_id += 1

    if not comp_sizes:
        return vertices, faces, uvs
    largest = int(np.argmax(np.array(comp_sizes, dtype=np.int32)))
    keep_v_mask = label == largest

    keep_faces = keep_v_mask[faces[:, 0]] & keep_v_mask[faces[:, 1]] & keep_v_mask[faces[:, 2]]
    kept_faces = faces[keep_faces]
    if kept_faces.size == 0:
        return vertices, faces, uvs

    used = np.unique(kept_faces.reshape(-1))
    remap = np.full((n,), -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    new_faces = remap[kept_faces].astype(np.int32)
    new_vertices = vertices[used].astype(np.float32)
    new_uvs = (uvs[used].astype(np.float32) if uvs is not None and getattr(uvs, "size", 0) else uvs)
    return new_vertices, new_faces, new_uvs


def _spatial_face_sample(vertices, faces, uvs, max_faces=2200):
    face_count = int(faces.shape[0])
    if face_count <= int(max_faces):
        return vertices, faces, uvs

    tri = vertices[faces]
    cent = np.mean(tri[:, :, :2], axis=1)  # Nx2
    # Scanline ordering: mostly preserves local continuity.
    order = np.lexsort((cent[:, 0], cent[:, 1]))
    step = max(1, int(np.ceil(face_count / float(max_faces))))
    keep = order[::step][: int(max_faces)]
    sampled_faces = faces[keep].astype(np.int32)

    used = np.unique(sampled_faces.reshape(-1))
    remap = np.full((vertices.shape[0],), -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    compact_faces = remap[sampled_faces].astype(np.int32)
    compact_vertices = vertices[used].astype(np.float32)
    compact_uvs = (uvs[used].astype(np.float32) if uvs is not None and getattr(uvs, "size", 0) else uvs)
    return compact_vertices, compact_faces, compact_uvs
