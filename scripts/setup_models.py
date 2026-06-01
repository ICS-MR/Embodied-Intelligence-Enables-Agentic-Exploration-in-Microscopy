from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
BGE_DIR = MODEL_DIR / "bge-m3"
HF_REPO_ID = "BAAI/bge-m3"

MODEL_BINARY_CANDIDATES = (
    BGE_DIR / "model.safetensors",
    BGE_DIR / "pytorch_model.bin",
)
REQUIRED_TEXT_FILES = (
    BGE_DIR / "config.json",
    BGE_DIR / "tokenizer.json",
)
IGNORE_PATTERNS = [
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.onnx_data",
    "*.xml",
    "*.msgpack",
    "*.h5",
    "tf_model.h5",
    "flax_model.msgpack",
    "rust_model.ot",
]


def is_model_ready() -> bool:
    return _missing_required_files() == []


def _missing_required_files() -> list[str]:
    missing: list[str] = []
    for required_path in REQUIRED_TEXT_FILES:
        if not required_path.exists():
            missing.append(str(required_path))
    if not any(path.exists() for path in MODEL_BINARY_CANDIDATES):
        missing.append("one of: " + ", ".join(str(path) for path in MODEL_BINARY_CANDIDATES))
    return missing


def _iter_existing_model_binaries() -> Iterable[Path]:
    for candidate in MODEL_BINARY_CANDIDATES:
        if candidate.exists():
            yield candidate


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
        ignore_patterns=IGNORE_PATTERNS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download required local model assets into ./model")
    parser.add_argument("--force", action="store_true", help="Download even if the target model already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if is_model_ready() and not args.force:
        print(f"Model already exists: {BGE_DIR}")
        for model_path in _iter_existing_model_binaries():
            print(f"Using model weights: {model_path.name}")
        return 0

    print(f"Downloading model assets into: {BGE_DIR}")
    print(f"Source: https://huggingface.co/{HF_REPO_ID}")
    print("Skipping optional ONNX/OpenVINO artifacts to reduce download size and timeout risk.")
    download_bge_m3(force=bool(args.force))

    missing_files = _missing_required_files()
    if missing_files:
        raise RuntimeError(
            "Model download completed, but required files were not found under "
            f"{BGE_DIR}: {missing_files}. Please inspect the downloaded folder structure."
        )

    for model_path in _iter_existing_model_binaries():
        print(f"Using model weights: {model_path.name}")
    print(f"Model setup completed: {BGE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
