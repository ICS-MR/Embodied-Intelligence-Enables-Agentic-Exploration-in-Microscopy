"""Score screenshots against a FRAP startup reference image to pick a threshold.

Candidate images may be region-sized captures or full screenshots; full
screenshots are cropped to the configured region automatically.

Example:
    python docs_public/frap/calibrate_frap_startup_reference.py --check post_click ^
        --ready loaded_1.png loaded_2.png --not-ready splash_1.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROFILE_PATH = Path(__file__).resolve().parent / "frap_ui_profile.json"


def resolve_profile_relative(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROFILE_PATH.parent / path
    return path.resolve()


def region_size(region: dict) -> tuple[int, int]:
    width = int(region["right"]) - int(region["left"]) + 1
    height = int(region["bottom"]) - int(region["top"]) + 1
    return width, height


def load_candidate_gray(path: Path, region: dict | None) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    if region is not None:
        expected_width, expected_height = region_size(region)
        height, width = image.shape[:2]
        if (width, height) == (expected_width, expected_height):
            pass
        elif width >= expected_width and height >= expected_height:
            left = int(region["left"])
            top = int(region["top"])
            image = image[
                top : top + expected_height,
                left : left + expected_width,
            ]
        else:
            raise SystemExit(
                f"Image {path.name} is {width}x{height}; expected at least "
                f"{expected_width}x{expected_height} for region cropping."
            )
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def score_against(reference: np.ndarray, candidate: np.ndarray, label: str) -> float:
    if candidate.shape != reference.shape:
        raise SystemExit(
            f"{label}: candidate size {candidate.shape[1]}x{candidate.shape[0]} does not "
            f"match reference size {reference.shape[1]}x{reference.shape[0]}."
        )
    match = cv2.matchTemplate(candidate, reference, cv2.TM_CCOEFF_NORMED)
    return float(np.max(match))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, choices=("pre_click", "post_click"))
    parser.add_argument(
        "--reference",
        help="Reference image path; defaults to the one configured in the profile.",
    )
    parser.add_argument("--ready", nargs="+", required=True, help="Screenshots in the loaded state.")
    parser.add_argument(
        "--not-ready", nargs="+", required=True, help="Screenshots in the not-loaded state."
    )
    args = parser.parse_args()

    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    check = payload.get("options", {}).get("startup_reference_checks", {}).get(args.check, {})
    region = check.get("region")

    reference_text = args.reference or check.get("image", "")
    if not reference_text:
        raise SystemExit(
            f"No reference available: pass --reference or configure "
            f"startup_reference_checks.{args.check}.image in {PROFILE_PATH}."
        )
    reference_path = resolve_profile_relative(reference_text)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise SystemExit(f"Could not read reference image: {reference_path}")

    results: list[tuple[str, Path, float]] = []
    for label, paths in (("READY", args.ready), ("NOT_READY", args.not_ready)):
        for path_text in paths:
            path = resolve_profile_relative(path_text)
            candidate = load_candidate_gray(path, region)
            score = score_against(reference, candidate, label)
            results.append((label, path, score))
            print(f"{label:9s} {path.name:40s} score={score:.4f}")

    ready_scores = [score for label, _, score in results if label == "READY"]
    not_ready_scores = [score for label, _, score in results if label == "NOT_READY"]
    min_ready = min(ready_scores)
    max_not_ready = max(not_ready_scores)
    if min_ready > max_not_ready:
        suggested = round((min_ready + max_not_ready) / 2.0, 3)
        print(
            f"Separable. min(ready)={min_ready:.4f}, max(not_ready)={max_not_ready:.4f}, "
            f"suggested startup_match_threshold={suggested}"
        )
    else:
        print(
            f"Not separable: min(ready)={min_ready:.4f} <= max(not_ready)={max_not_ready:.4f}. "
            "Recapture the references or choose a region that better distinguishes the states."
        )


if __name__ == "__main__":
    main()
