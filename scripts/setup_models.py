from __future__ import annotations

import argparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
BGE_DIR = MODEL_DIR / "bge-m3"
HF_REPO_ID = "BAAI/bge-m3"


def is_model_ready() -> bool:
    required_files = [
        BGE_DIR / "config.json",
        BGE_DIR / "pytorch_model.bin",
        BGE_DIR / "tokenizer.json",
    ]
    return all(path.exists() for path in required_files)


def download_bge_m3(*, force: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download model assets.\n"
            "Install it with: pip install huggingface_hub"
        ) from exc

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(BGE_DIR),
        local_dir_use_symlinks=False,
        resume_download=not force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download required local model assets into ./model")
    parser.add_argument("--force", action="store_true", help="Download even if the target model already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if is_model_ready() and not args.force:
        print(f"Model already exists: {BGE_DIR}")
        return 0

    print(f"Downloading model assets into: {BGE_DIR}")
    print(f"Source: https://huggingface.co/{HF_REPO_ID}")
    download_bge_m3(force=bool(args.force))

    if not is_model_ready():
        raise RuntimeError(
            "Model download completed, but required files were not found under "
            f"{BGE_DIR}. Please inspect the downloaded folder structure."
        )

    print(f"Model setup completed: {BGE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
