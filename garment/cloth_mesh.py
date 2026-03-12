import numpy as np


def create_shirt_cloth_mesh(rows=18, cols=14):
    """Create a front torso cloth grid with anchor vertices and triangle faces."""
    vertices = []
    uvs = []
    for r in range(rows):
        v = r / float(max(rows - 1, 1))
        torso_width = 0.92 - 0.18 * v
        for c in range(cols):
            u = c / float(max(cols - 1, 1))
            x_base = (u - 0.5) * 2.0 * torso_width
            sleeve_zone = np.exp(-((v - 0.14) ** 2) / 0.018)
            edge = abs(u - 0.5) * 2.0
            sleeve_push = 0.45 * sleeve_zone * max(edge - 0.72, 0.0)
            x = x_base + np.sign(u - 0.5) * sleeve_push
            # Shorter torso profile (shirt hem near waist, not knee length).
            y = -v * 1.35 + 0.72
            z = -0.06 * (1.0 - edge)
            vertices.append([x, y, z])
            uvs.append([u, v])

    vertices = np.array(vertices, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)

    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i0 = r * cols + c
            i1 = i0 + 1
            i2 = i0 + cols
            i3 = i2 + 1
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])

    faces = np.array(faces, dtype=np.int32)

    anchors = {
        "left_shoulder": 0 * cols + 1,
        "right_shoulder": 0 * cols + (cols - 2),
        "chest": int((rows * 0.32)) * cols + cols // 2,
        "left_hip": (rows - 1) * cols + int(cols * 0.25),
        "right_hip": (rows - 1) * cols + int(cols * 0.75),
    }

    return {
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs,
        "anchors": anchors,
        "rows": rows,
        "cols": cols,
    }
