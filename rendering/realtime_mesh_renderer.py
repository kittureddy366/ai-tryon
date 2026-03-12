import numpy as np
import cv2

try:
    import glfw
    from OpenGL import GL

    OPENGL_AVAILABLE = True
except Exception:
    OPENGL_AVAILABLE = False


class RealtimeMeshRenderer:
    """Render body + cloth meshes and alpha-composite them over camera frames."""

    def __init__(self):
        self.use_opengl = False
        self.window = None
        self.frame_w = None
        self.frame_h = None

    def _init_gl(self, frame_w, frame_h):
        if not OPENGL_AVAILABLE:
            return False
        if not glfw.init():
            return False

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        self.window = glfw.create_window(int(frame_w), int(frame_h), "tryon", None, None)
        if self.window is None:
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self.use_opengl = True
        return True

    def close(self):
        if self.window is not None:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None

    def render_overlay(
        self,
        frame_bgr,
        body_vertices,
        body_faces,
        cloth_vertices,
        cloth_faces,
        cloth_uvs=None,
        cloth_texture_rgba=None,
        cloth_color_bgra=(45, 65, 235, 205),
        pose_points=None,
        anchor_targets=None,
        person_mask=None,
    ):
        h, w = frame_bgr.shape[:2]
        if self.frame_w is None:
            self._init_gl(w, h)

        if self.use_opengl:
            overlay = self._render_opengl(w, h, cloth_vertices, cloth_faces, cloth_color_bgra)
        else:
            if cloth_texture_rgba is not None and cloth_uvs is not None:
                overlay = self._render_software_textured(
                    w, h, cloth_vertices, cloth_faces, cloth_uvs, cloth_texture_rgba, cloth_color_bgra
                )
                if not self._has_sufficient_coverage(overlay[:, :, 3], min_pixels=12000, min_bbox_area=16000):
                    overlay = self._render_software(w, h, cloth_vertices, cloth_faces, cloth_color_bgra)
            else:
                overlay = self._render_software(w, h, cloth_vertices, cloth_faces, cloth_color_bgra)

        if (not self._has_sufficient_coverage(overlay[:, :, 3], min_pixels=9000, min_bbox_area=14000)) and anchor_targets is not None:
            overlay = self._fallback_anchor_overlay(
                frame_bgr.shape, anchor_targets, cloth_color_bgra, cloth_texture_rgba=cloth_texture_rgba
            )

        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        overlay = self._match_scene_lighting(overlay, frame_bgr, cloth_vertices)
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        if person_mask is not None:
            body_alpha = cv2.dilate((person_mask > 0).astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1)
            alpha *= body_alpha[:, :, None].astype(np.float32)
        out = frame_bgr.astype(np.float32)
        out = overlay[:, :, :3].astype(np.float32) * alpha + out * (1.0 - alpha)
        composed = np.clip(out, 0, 255).astype(np.uint8)
        if pose_points is not None:
            arm_mask = self._foreground_arm_mask(frame_bgr.shape, pose_points)
            composed[arm_mask > 0] = frame_bgr[arm_mask > 0]
        return composed

    @staticmethod
    def _has_sufficient_coverage(alpha, min_pixels, min_bbox_area):
        nz = cv2.findNonZero((alpha > 20).astype(np.uint8))
        if nz is None:
            return False
        count = int(nz.shape[0])
        x, y, w, h = cv2.boundingRect(nz)
        area = int(w * h)
        return count >= int(min_pixels) and area >= int(min_bbox_area)

    @staticmethod
    def _fallback_anchor_overlay(frame_shape, anchors, color, cloth_texture_rgba=None):
        h, w = frame_shape[:2]
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        required = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
        if not all(k in anchors for k in required):
            return canvas

        ls = anchors["left_shoulder"]
        rs = anchors["right_shoulder"]
        rh = anchors["right_hip"]
        lh = anchors["left_hip"]

        top_mid = (ls + rs) * 0.5
        hip_mid = (lh + rh) * 0.5
        torso_vec = hip_mid - top_mid
        shoulder_w = float(np.linalg.norm(rs - ls))
        torso_h = float(np.linalg.norm(torso_vec))
        if shoulder_w < 1.0 or torso_h < 1.0:
            return canvas

        x_dir = (rs - ls) / (shoulder_w + 1e-6)
        y_dir = torso_vec / (torso_h + 1e-6)

        # Better wearable fit: shoulder spread + shorter hem around upper hip.
        shoulder_pad = shoulder_w * 0.10
        chest_drop = torso_h * 0.02
        hem_height = top_mid + y_dir * (torso_h * 0.78)
        hem_half_w = shoulder_w * 0.32

        top_l = ls - x_dir * shoulder_pad + y_dir * chest_drop
        top_r = rs + x_dir * shoulder_pad + y_dir * chest_drop
        hem_l = hem_height - x_dir * hem_half_w
        hem_r = hem_height + x_dir * hem_half_w

        poly = np.array(
            [
                [top_l[0], top_l[1]],
                [top_r[0], top_r[1]],
                [hem_r[0], hem_r[1]],
                [hem_l[0], hem_l[1]],
            ],
            dtype=np.int32,
        )
        if cloth_texture_rgba is not None and cloth_texture_rgba.size > 0:
            return RealtimeMeshRenderer._warp_texture_to_quad(
                canvas,
                cloth_texture_rgba,
                poly.astype(np.float32),
                alpha_scale=float(color[3]) / 255.0,
            )

        fallback_color = (220, 120, 40, 205)
        cv2.fillConvexPoly(canvas, poly, fallback_color)
        canvas[:, :, 3] = cv2.GaussianBlur(canvas[:, :, 3], (9, 9), 0)
        return canvas

    @staticmethod
    def _warp_texture_to_quad(dst_canvas, texture_rgba, quad_dst, alpha_scale=1.0):
        h, w = dst_canvas.shape[:2]
        th, tw = texture_rgba.shape[:2]
        # Crop source region to avoid long tail / side artifacts from catalog textures.
        sx0 = int(tw * 0.14)
        sx1 = int(tw * 0.86)
        sy0 = int(th * 0.06)
        sy1 = int(th * 0.90)
        src = np.float32(
            [
                [sx0, sy0],
                [sx1, sy0],
                [sx1, sy1],
                [sx0, sy1],
            ]
        )

        persp = cv2.getPerspectiveTransform(src, quad_dst.astype(np.float32))
        warped = cv2.warpPerspective(
            texture_rgba,
            persp,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        ).astype(np.float32)

        # Neck cut-out for better realism only if texture does not already contain a neckline hole.
        ls, rs, rh, lh = quad_dst
        neck_center = ((ls + rs) * 0.5).astype(np.int32)
        shoulder_w = max(10.0, float(np.linalg.norm(rs - ls)))
        torso_h = max(20.0, float(np.linalg.norm(((lh + rh) * 0.5) - ((ls + rs) * 0.5))))
        neck_mask = np.zeros((h, w), dtype=np.uint8)
        if not RealtimeMeshRenderer._texture_has_top_center_hole(texture_rgba):
            cv2.ellipse(
                neck_mask,
                (int(neck_center[0]), int(neck_center[1] + shoulder_w * 0.02)),
                (int(shoulder_w * 0.10), int(torso_h * 0.055)),
                0,
                0,
                360,
                255,
                -1,
            )

        alpha = warped[:, :, 3:4] / 255.0
        alpha *= float(alpha_scale)
        alpha[neck_mask > 0] = 0.0

        # Slight center shading to avoid flat pasted look.
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        shade = (0.92 + 0.10 * (1.0 - np.abs(yy - 0.42) * 1.8))[:, :, None]
        warped[:, :, :3] *= shade

        dst = dst_canvas.astype(np.float32)
        dst[:, :, :3] = warped[:, :, :3] * alpha + dst[:, :, :3] * (1.0 - alpha)
        dst[:, :, 3:4] = np.maximum(dst[:, :, 3:4], alpha * 255.0)
        out = np.clip(dst, 0, 255).astype(np.uint8)
        out[:, :, 3] = cv2.GaussianBlur(out[:, :, 3], (7, 7), 0)
        return out

    @staticmethod
    def _texture_has_top_center_hole(texture_rgba):
        if texture_rgba is None or texture_rgba.size == 0 or texture_rgba.shape[2] < 4:
            return False
        alpha = texture_rgba[:, :, 3]
        h, w = alpha.shape[:2]
        y0, y1 = int(h * 0.03), int(h * 0.26)
        x0, x1 = int(w * 0.36), int(w * 0.64)
        roi = alpha[y0:y1, x0:x1]
        if roi.size == 0:
            return False
        hole_ratio = float(np.mean(roi < 40))
        return hole_ratio > 0.22

    def _render_opengl(self, w, h, cloth_vertices, cloth_faces, cloth_color_bgra):
        glfw.make_context_current(self.window)
        GL.glViewport(0, 0, w, h)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, w, h, 0, -1000.0, 1000.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Render only the simulated cloth mesh. Body mesh remains internal.
        b, g, r, a = cloth_color_bgra
        GL.glColor4f(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        GL.glBegin(GL.GL_TRIANGLES)
        for f in cloth_faces:
            for idx in f:
                v = cloth_vertices[idx]
                GL.glVertex3f(float(v[0]), float(v[1]), float(v[2] - 5.0))
        GL.glEnd()

        pixels = GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 4)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return img

    def _render_software(self, w, h, cloth_vertices, cloth_faces, cloth_color_bgra):
        color_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        alpha_canvas = np.zeros((h, w), dtype=np.uint8)

        draw_list = []
        for f in cloth_faces:
            tri = cloth_vertices[f]
            depth = float(np.mean(tri[:, 2]) - 4.0)
            draw_list.append((depth, tri, cloth_color_bgra))

        draw_list.sort(key=lambda item: item[0], reverse=True)

        for _, tri, color in draw_list:
            pts = np.round(tri[:, :2]).astype(np.int32)
            min_x = int(np.min(pts[:, 0]))
            max_x = int(np.max(pts[:, 0]))
            min_y = int(np.min(pts[:, 1]))
            max_y = int(np.max(pts[:, 1]))
            if max_x < 0 or min_x >= w:
                continue
            if max_y < 0 or min_y >= h:
                continue
            self._blend_triangle(color_canvas, alpha_canvas, pts, color)

        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        canvas[:, :, :3] = cv2.GaussianBlur(color_canvas, (3, 3), 0)
        canvas[:, :, 3] = alpha_canvas
        canvas[:, :, 3] = cv2.GaussianBlur(canvas[:, :, 3], (5, 5), 0)
        return canvas

    def _render_software_textured(
        self,
        w,
        h,
        cloth_vertices,
        cloth_faces,
        cloth_uvs,
        cloth_texture_rgba,
        cloth_color_bgra,
    ):
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        th, tw = cloth_texture_rgba.shape[:2]

        draw_order = []
        for f in cloth_faces:
            tri = cloth_vertices[f]
            depth = float(np.mean(tri[:, 2]))
            draw_order.append((depth, f))
        draw_order.sort(key=lambda x: x[0], reverse=True)

        tint = np.array(cloth_color_bgra[:3], dtype=np.float32) / 255.0
        alpha_scale = float(cloth_color_bgra[3]) / 255.0

        for _, face in draw_order:
            dst_tri = cloth_vertices[face][:, :2].astype(np.float32)
            src_uv = cloth_uvs[face].astype(np.float32)
            src_tri = np.column_stack([src_uv[:, 0] * (tw - 1), src_uv[:, 1] * (th - 1)]).astype(np.float32)
            self._warp_textured_triangle(canvas, cloth_texture_rgba, src_tri, dst_tri, tint, alpha_scale)

        canvas[:, :, :3] = cv2.GaussianBlur(canvas[:, :, :3], (3, 3), 0)
        canvas[:, :, 3] = cv2.GaussianBlur(canvas[:, :, 3], (5, 5), 0)
        canvas[:, :, 3][canvas[:, :, 3] < 24] = 0
        return canvas

    @staticmethod
    def _blend_triangle(color_canvas, alpha_canvas, pts, color):
        h, w = alpha_canvas.shape
        x, y, bw, bh = cv2.boundingRect(pts)
        if bw <= 0 or bh <= 0:
            return

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + bw)
        y1 = min(h, y + bh)
        if x0 >= x1 or y0 >= y1:
            return

        local_pts = pts - np.array([x0, y0], dtype=np.int32)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_pts, 255)

        tri_alpha = (mask.astype(np.float32) / 255.0) * (float(color[3]) / 255.0)
        tri_alpha_3 = tri_alpha[:, :, None]

        roi_color = color_canvas[y0:y1, x0:x1].astype(np.float32)
        fill_color = np.array(color[:3], dtype=np.float32)[None, None, :]
        blended = fill_color * tri_alpha_3 + roi_color * (1.0 - tri_alpha_3)
        color_canvas[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

        new_alpha = np.clip(tri_alpha * 255.0, 0, 255).astype(np.uint8)
        roi_alpha = alpha_canvas[y0:y1, x0:x1]
        alpha_canvas[y0:y1, x0:x1] = np.maximum(roi_alpha, new_alpha)

    @staticmethod
    def _warp_textured_triangle(dst_rgba, src_rgba, src_tri, dst_tri, tint, alpha_scale):
        h, w = dst_rgba.shape[:2]
        if src_tri is None or dst_tri is None:
            return
        if getattr(src_tri, "shape", None) != (3, 2) or getattr(dst_tri, "shape", None) != (3, 2):
            return
        if (not np.isfinite(src_tri).all()) or (not np.isfinite(dst_tri).all()):
            return

        # Guard against tracker/simulation explosions producing huge coordinates that would make
        # OpenCV try to allocate gigantic temporary images (and crash the app).
        frame_scale = float(max(w, h))
        if frame_scale <= 0:
            return
        max_coord = frame_scale * 8.0 + 1024.0
        if float(np.max(np.abs(dst_tri))) > max_coord:
            return

        sx, sy, sw, sh = cv2.boundingRect(src_tri)
        dx, dy, dw, dh = cv2.boundingRect(dst_tri)
        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            return

        # Clamp warps to a reasonable size relative to the destination frame.
        # This keeps per-triangle warping from OOM'ing when a single triangle bounds is enormous.
        max_side = int(max(1024, frame_scale * 2.5))
        max_side = min(max_side, 4096)
        max_area = int(w * h * 3.0)
        max_area = max(256 * 256, min(max_area, max_side * max_side))
        if dw > max_side or dh > max_side or (dw * dh) > max_area:
            return
        if dx >= w or dy >= h or (dx + dw) <= 0 or (dy + dh) <= 0:
            return

        src_patch = src_rgba[sy : sy + sh, sx : sx + sw]
        if src_patch.size == 0:
            return

        src_local = src_tri - np.array([sx, sy], dtype=np.float32)
        dst_local = dst_tri - np.array([dx, dy], dtype=np.float32)
        affine = cv2.getAffineTransform(src_local.astype(np.float32), dst_local.astype(np.float32))
        try:
            warped = cv2.warpAffine(
                src_patch,
                affine,
                (dw, dh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
        except cv2.error:
            return

        tri_mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, np.round(dst_local).astype(np.int32), 255)

        x0 = max(dx, 0)
        y0 = max(dy, 0)
        x1 = min(dx + dw, w)
        y1 = min(dy + dh, h)
        if x0 >= x1 or y0 >= y1:
            return

        cx0 = x0 - dx
        cy0 = y0 - dy
        cx1 = cx0 + (x1 - x0)
        cy1 = cy0 + (y1 - y0)

        wr = warped[cy0:cy1, cx0:cx1].astype(np.float32)
        if wr.shape[2] == 3:
            alpha = np.ones((wr.shape[0], wr.shape[1], 1), dtype=np.float32)
            wr = np.concatenate([wr, alpha * 255.0], axis=2)

        wr[:, :, :3] *= tint[None, None, :]
        mask = (tri_mask[cy0:cy1, cx0:cx1].astype(np.float32) / 255.0)[:, :, None]
        src_a = (wr[:, :, 3:4] / 255.0) * mask * alpha_scale
        src_a[src_a < 0.18] = 0.0

        dst = dst_rgba[y0:y1, x0:x1].astype(np.float32)
        dst[:, :, :3] = wr[:, :, :3] * src_a + dst[:, :, :3] * (1.0 - src_a)
        dst[:, :, 3:4] = np.maximum(dst[:, :, 3:4], src_a * 255.0)
        dst_rgba[y0:y1, x0:x1] = np.clip(dst, 0, 255).astype(np.uint8)

    @staticmethod
    def _match_scene_lighting(overlay_bgra, frame_bgr, cloth_vertices):
        overlay = overlay_bgra.copy()
        alpha = overlay[:, :, 3]
        if np.count_nonzero(alpha) < 50:
            return overlay

        pts = np.round(cloth_vertices[:, :2]).astype(np.int32)
        x0 = max(0, int(np.min(pts[:, 0])))
        y0 = max(0, int(np.min(pts[:, 1])))
        x1 = min(frame_bgr.shape[1], int(np.max(pts[:, 0])) + 1)
        y1 = min(frame_bgr.shape[0], int(np.max(pts[:, 1])) + 1)
        if x0 >= x1 or y0 >= y1:
            return overlay

        frame_roi = frame_bgr[y0:y1, x0:x1]
        ov_roi = overlay[y0:y1, x0:x1]
        mask = ov_roi[:, :, 3] > 25
        if not np.any(mask):
            return overlay

        frame_y = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ov_y = cv2.cvtColor(ov_roi[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)

        target_luma = float(np.mean(frame_y[mask]))
        cloth_luma = float(np.mean(ov_y[mask])) + 1e-6
        gain = np.clip((target_luma * 0.90) / cloth_luma, 0.72, 1.28)

        color = ov_roi[:, :, :3].astype(np.float32)
        color *= gain

        # Soft vertical shading for depth cue.
        h, w = color.shape[:2]
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        vignette = 0.96 + 0.08 * (1.0 - np.abs(yy - 0.45) * 1.8)
        color *= vignette[:, :, None]

        ov_roi[:, :, :3] = np.clip(color, 0, 255).astype(np.uint8)
        overlay[y0:y1, x0:x1] = ov_roi
        return overlay

    @staticmethod
    def _foreground_arm_mask(frame_shape, points):
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        ls = points["left_shoulder"]
        rs = points["right_shoulder"]
        lh = points["left_hip"]
        rh = points["right_hip"]
        torso_depth = 0.25 * (ls["z"] + rs["z"] + lh["z"] + rh["z"])

        arm_pairs = (
            ("left_shoulder", "left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow", "right_wrist"),
        )
        for s_name, e_name, w_name in arm_pairs:
            s = points[s_name]
            e = points[e_name]
            wr = points[w_name]
            arm_depth = 0.5 * (e["z"] + wr["z"])
            if arm_depth < torso_depth - 0.03:
                cv2.line(mask, (s["x"], s["y"]), (e["x"], e["y"]), 255, 24, cv2.LINE_AA)
                cv2.line(mask, (e["x"], e["y"]), (wr["x"], wr["y"]), 255, 20, cv2.LINE_AA)
                cv2.circle(mask, (wr["x"], wr["y"]), 10, 255, -1, cv2.LINE_AA)
        return mask
