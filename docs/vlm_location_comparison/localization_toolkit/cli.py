"""Command line entry point for unified localization experiments."""

from __future__ import annotations

import argparse

from .pipeline import (
    LocalizationConfig,
    compare_localizations,
    run_model_localization,
    run_vlm_localization,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run model-based localization, VLM localization, or both."
    )
    parser.add_argument("--mode", choices=["vlm", "model", "compare"], required=True)
    parser.add_argument("--image", default="", help="Input image path.")
    parser.add_argument("--output-dir", default="localization_output")
    parser.add_argument("--image-id", type=int, default=1)
    parser.add_argument("--category-id", type=int, default=0)

    parser.add_argument("--config", help="MMDetection config file.")
    parser.add_argument("--checkpoint", help="MMDetection checkpoint file.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--nms-thr", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)

    parser.add_argument("--queries", nargs="+", default=["cell"], help="VLM target texts.")
    parser.add_argument("--vlm-thr", type=float, default=0.3)

    parser.add_argument("--gt", help="COCO ground-truth annotation file for comparison.")
    parser.add_argument("--model-pred", help="Model prediction JSON for compare mode.")
    parser.add_argument("--vlm-pred", help="VLM COCO prediction JSON for compare mode.")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--confidence-thr", type=float, default=0.3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode in {"vlm", "model"} and not args.image:
        raise SystemExit("--image is required for vlm and model modes")

    cfg = LocalizationConfig(
        image_path=args.image,
        output_dir=args.output_dir,
        image_id=args.image_id,
        category_id=args.category_id,
        config_file=args.config,
        checkpoint_file=args.checkpoint,
        device=args.device,
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        tile_size=args.tile_size,
        overlap=args.overlap,
        detection_threshold=args.vlm_thr,
        query_texts=args.queries,
        gt_annotation_file=args.gt,
        iou_threshold=args.iou_thr,
        confidence_threshold=args.confidence_thr,
    )

    if args.mode == "model":
        run_model_localization(cfg)
    elif args.mode == "vlm":
        run_vlm_localization(cfg)
    else:
        pred_files = None
        if args.model_pred and args.vlm_pred:
            pred_files = [args.model_pred, args.vlm_pred]
        compare_localizations(cfg, pred_files=pred_files)


if __name__ == "__main__":
    main()
