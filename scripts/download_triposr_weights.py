import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download TripoSR pretrained weights into the local cache directory.")
    parser.add_argument(
        "--repo",
        default="stabilityai/TripoSR",
        help="Hugging Face repo id to download from (default: stabilityai/TripoSR).",
    )
    parser.add_argument(
        "--dest",
        default="",
        help="Destination folder (default: .cache/triposr_hf/stabilityai__TripoSR under repo root).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dest = Path(args.dest).expanduser() if args.dest else (repo_root / ".cache" / "triposr_hf" / "stabilityai__TripoSR")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub.\n"
            "Install it with: pip install -U huggingface_hub\n"
            f"Original error: {exc}"
        )

    allow = {"config.yaml", "model.ckpt"}
    print(f"Downloading {args.repo} -> {dest}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        allow_patterns=list(allow),
        token=os.environ.get("HF_TOKEN", None),
    )

    config_path = dest / "config.yaml"
    weight_path = dest / "model.ckpt"
    if not config_path.exists() or not weight_path.exists():
        raise SystemExit(f"Download incomplete: expected {config_path} and {weight_path}")
    print("Done.")
    print(f"Config: {config_path}")
    print(f"Weights: {weight_path}")


if __name__ == "__main__":
    main()

