"""Capture FRAP startup reference images on the cellSens machine.

Run this on the machine where cellSens is installed, after cellSens has fully
loaded. The post_click reference must be captured with the FRAP tab open.

Examples:
    python docs_public/frap/capture_frap_startup_reference.py --check pre_click
    python docs_public/frap/capture_frap_startup_reference.py --check post_click
    python docs_public/frap/capture_frap_startup_reference.py --check both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parent / "frap_ui_profile.json"
DEFAULT_OUT_DIR = PROFILE_PATH.parent / "references"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        choices=("pre_click", "post_click", "both"),
        default="both",
        help="Which startup reference image to capture.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for reference images.",
    )
    args = parser.parse_args()

    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    checks = payload.get("options", {}).get("startup_reference_checks", {})
    if not isinstance(checks, dict):
        raise SystemExit(f"Invalid startup_reference_checks in {PROFILE_PATH}")

    names = ["pre_click", "post_click"] if args.check == "both" else [args.check]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import pyautogui

    for name in names:
        region = checks.get(name, {}).get("region")
        if not region:
            raise SystemExit(
                f"startup_reference_checks.{name}.region is not configured in {PROFILE_PATH}."
            )
        left = int(region["left"])
        top = int(region["top"])
        width = int(region["right"]) - left + 1
        height = int(region["bottom"]) - top + 1
        if width <= 0 or height <= 0:
            raise SystemExit(
                f"startup_reference_checks.{name}.region must have positive width and height."
            )
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        out_path = args.out_dir / f"{name}.png"
        screenshot.save(out_path)
        print(f"Captured {name}: {out_path} ({screenshot.width}x{screenshot.height})")
        print(f"Profile image path (relative to the profile file): references/{out_path.name}")


if __name__ == "__main__":
    main()
