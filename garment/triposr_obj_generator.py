import hashlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def find_2d_garment_image(project_root, preferred_paths=None):
    preferred_paths = preferred_paths or []
    for rel in preferred_paths:
        if not rel:
            continue
        candidate = Path(project_root) / rel
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return str(candidate)

    root = Path(project_root)
    score_words = ("shirt", "tshirt", "tee", "garment", "cloth", "top")
    best = None
    best_score = -1
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = path.name.lower()
        score = sum(1 for w in score_words if w in name)
        if score > best_score:
            best = path
            best_score = score
    return str(best) if best is not None else None


def find_local_triposr_model_dir(project_root):
    candidates = [
        os.environ.get("TRIPOSR_MODEL_PATH", ""),
        os.path.join(project_root, "TripoSR", "weights"),
        os.path.join(project_root, "TripoSR", "pretrained"),
        os.path.join(project_root, "TripoSR", "stabilityai", "TripoSR"),
        os.path.join(project_root, "stabilityai", "TripoSR"),
    ]
    for folder in candidates:
        if not folder or not os.path.isdir(folder):
            continue
        config_path = os.path.join(folder, "config.yaml")
        weight_path = os.path.join(folder, "model.ckpt")
        if os.path.exists(config_path) and os.path.exists(weight_path):
            return folder
    return None


class TripoSRObjGenerator:
    def __init__(self, project_root, image_path, output_root=None):
        self.project_root = os.path.abspath(project_root)
        self.image_path = os.path.abspath(image_path) if image_path else None
        self.output_root = output_root or os.path.join(self.project_root, "generated", "triposr")
        self.status = "idle"
        self.error = ""
        self.obj_path = None
        self._thread = None
        self._lock = threading.Lock()
        self._started_at = 0.0

    def elapsed_seconds(self):
        if self.status != "running" or self._started_at <= 0:
            return 0.0
        return time.time() - self._started_at

    def snapshot(self):
        with self._lock:
            return {
                "status": self.status,
                "error": self.error,
                "obj_path": self.obj_path,
                "elapsed": self.elapsed_seconds(),
            }

    def start_async(self, force=False):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if not self.image_path or not os.path.exists(self.image_path):
                self.status = "failed"
                self.error = "No valid 2D garment image found for TripoSR."
                return
            cached = self._cached_obj_path()
            if cached and os.path.exists(cached) and not force:
                self.status = "ready"
                self.error = ""
                self.obj_path = cached
                return
            self.status = "running"
            self.error = ""
            self.obj_path = None
            self._started_at = time.time()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _cache_dir(self):
        if not self.image_path:
            return self.output_root
        stat = os.stat(self.image_path)
        key = f"{self.image_path}|{int(stat.st_mtime)}|{stat.st_size}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return os.path.join(self.output_root, digest)

    def _cached_obj_path(self):
        return os.path.join(self._cache_dir(), "0", "mesh.obj")

    def _worker(self):
        os.makedirs(self._cache_dir(), exist_ok=True)
        hf_home = os.path.join(self.project_root, ".cache", "huggingface")
        hf_hub_cache = os.path.join(hf_home, "hub")
        transformers_cache = os.path.join(hf_home, "transformers")
        os.makedirs(hf_hub_cache, exist_ok=True)
        os.makedirs(transformers_cache, exist_ok=True)
        cmd = [
            sys.executable,
            os.path.join("TripoSR", "run.py"),
            self.image_path,
            "--output-dir",
            self._cache_dir(),
            "--model-save-format",
            "obj",
            "--mc-resolution",
            "48",
            "--chunk-size",
            "2048",
            "--no-remove-bg",
        ]
        local_model = find_local_triposr_model_dir(self.project_root)
        if local_model:
            cmd.extend(["--pretrained-model-name-or-path", local_model])
        env = os.environ.copy()
        env["HF_HOME"] = hf_home
        env["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache
        env["TRANSFORMERS_CACHE"] = transformers_cache
        env["TRIPOSR_HF_LOCAL_DIR"] = os.path.join(self.project_root, ".cache", "triposr_hf")

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1200,
                env=env,
            )
            obj_path = self._cached_obj_path()
            if proc.returncode == 0 and os.path.exists(obj_path):
                with self._lock:
                    self.status = "ready"
                    self.obj_path = obj_path
                    self.error = ""
                return
            err_text = (proc.stderr or proc.stdout or "").strip()
            if len(err_text) > 700:
                err_text = err_text[-700:]
            with self._lock:
                self.status = "failed"
                self.error = err_text or "TripoSR failed to generate mesh.obj."
        except subprocess.TimeoutExpired:
            with self._lock:
                self.status = "failed"
                self.error = "TripoSR timed out while generating mesh."
        except Exception as exc:
            with self._lock:
                self.status = "failed"
                self.error = str(exc)
