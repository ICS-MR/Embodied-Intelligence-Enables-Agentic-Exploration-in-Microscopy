import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import localization_toolkit.vlm_inference as vlm_inference_module
from localization_toolkit.pipeline import _vlm_json_to_coco
from localization_toolkit.vlm_inference import _load_api_config, parse_detection_results


class ApiConfigurationTests(unittest.TestCase):
    def test_all_api_configuration_placeholders_must_be_replaced(self):
        configured_values = {
            "API_KEY": "test-api-key",
            "API_URL": "test-api-endpoint",
            "MODEL_NAME": "test-model-name",
        }
        for placeholder_name in configured_values:
            with self.subTest(placeholder_name=placeholder_name):
                values = configured_values | {placeholder_name: f"<your-{placeholder_name.lower()}>"}
                with patch.multiple(vlm_inference_module, **values):
                    with self.assertRaisesRegex(RuntimeError, placeholder_name):
                        _load_api_config()

    def test_api_configuration_uses_module_values(self):
        values = {
            "API_KEY": "test-api-key",
            "API_URL": "test-api-endpoint",
            "MODEL_NAME": "test-model-name",
        }
        with patch.multiple(vlm_inference_module, **values):
            self.assertEqual(
                _load_api_config(),
                (values["API_KEY"], values["API_URL"], values["MODEL_NAME"]),
            )


class VlmResponseTests(unittest.TestCase):
    def test_parses_object_response_and_applies_threshold(self):
        response = json.dumps({
            "detections": [
                {"label": "cell", "bbox": [100, 200, 500, 600], "confidence": 0.9},
                {"label": "cell", "bbox": [0, 0, 100, 100], "confidence": 0.2},
            ]
        })

        boxes, scores, labels = parse_detection_results(response, 1000, 500, 0.3)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(scores, [0.9])
        self.assertEqual(labels, ["cell"])
        self.assertAlmostEqual(boxes[0][0], 100.1001, places=3)
        self.assertAlmostEqual(boxes[0][1], 100.1001, places=3)

    def test_parses_legacy_array_response(self):
        response = '[{"label":"cell","bbox":[0,0,999,999]}]'
        boxes, scores, labels = parse_detection_results(response, 20, 10, 0.3)
        self.assertEqual(boxes, [[0, 0, 20.0, 10.0]])
        self.assertEqual(scores, [1.0])
        self.assertEqual(labels, ["cell"])


class CocoConversionTests(unittest.TestCase):
    def test_converts_xyxy_to_xywh(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "coco.json"
            input_path.write_text(json.dumps([{
                "label": "cell",
                "confidence": 0.75,
                "bbox": {"x_min": 10, "y_min": 20, "x_max": 40, "y_max": 70},
            }]), encoding="utf-8")

            count = _vlm_json_to_coco(str(input_path), str(output_path), image_id=7, category_id=1)

            self.assertEqual(count, 1)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), [{
                "image_id": 7,
                "category_id": 1,
                "bbox": [10.0, 20.0, 30.0, 50.0],
                "score": 0.75,
            }])


class ConfigurationValidationTests(unittest.TestCase):
    def test_rejects_invalid_tiling(self):
        from localization_toolkit import LocalizationConfig

        with self.assertRaisesRegex(ValueError, "overlap"):
            LocalizationConfig(image_path="image.jpg", tile_size=128, overlap=128)

    def test_rejects_invalid_threshold(self):
        from localization_toolkit import LocalizationConfig

        with self.assertRaisesRegex(ValueError, "detection_threshold"):
            LocalizationConfig(image_path="image.jpg", detection_threshold=1.1)


class ComparisonTests(unittest.TestCase):
    def test_comparison_writes_metrics_and_plot(self):
        from localization_toolkit import LocalizationConfig, compare_localizations

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_path = root / "ground_truth.json"
            model_path = root / "model.json"
            vlm_path = root / "vlm.json"
            gt_path.write_text(json.dumps({
                "images": [{"id": 1, "file_name": "image.jpg", "width": 100, "height": 100}],
                "annotations": [{
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 20, 30, 40],
                    "area": 1200,
                    "iscrowd": 0,
                }],
                "categories": [{"id": 1, "name": "cell"}],
            }), encoding="utf-8")
            prediction = [{
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, 30, 40],
                "score": 0.9,
            }]
            model_path.write_text(json.dumps(prediction), encoding="utf-8")
            vlm_path.write_text(json.dumps(prediction), encoding="utf-8")

            cfg = LocalizationConfig(
                image_path="",
                output_dir=str(root / "output"),
                gt_annotation_file=str(gt_path),
            )
            results = compare_localizations(cfg, pred_files=[str(model_path), str(vlm_path)])

            self.assertEqual(results["Model"]["matched_count"], 1)
            self.assertEqual(results["VLM"]["mean_center_distance"], 0.0)
            self.assertEqual(results["Model"]["true_positive"], 1)
            self.assertEqual(results["Model"]["false_positive"], 0)
            self.assertEqual(results["Model"]["false_negative"], 0)
            self.assertEqual(results["Model"]["precision"], 1.0)
            self.assertEqual(results["Model"]["recall"], 1.0)
            self.assertEqual(results["Model"]["f1"], 1.0)
            self.assertTrue((root / "output" / "error_results.json").is_file())
            self.assertTrue((root / "output" / "error_analysis.png").is_file())

    def test_comparison_counts_false_positives_false_negatives_and_categories(self):
        from localization_toolkit import LocalizationConfig, compare_localizations

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_path = root / "ground_truth.json"
            pred_path = root / "predictions.json"
            gt_path.write_text(json.dumps({
                "images": [{"id": 1, "file_name": "image.jpg", "width": 100, "height": 100}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},
                    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
                ],
                "categories": [{"id": 1, "name": "cell"}],
            }), encoding="utf-8")
            pred_path.write_text(json.dumps([
                {"image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "score": 0.9},
                {"image_id": 1, "category_id": 1, "bbox": [80, 80, 10, 10], "score": 0.9},
                {"image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10], "score": 0.9},
            ]), encoding="utf-8")

            cfg = LocalizationConfig(
                image_path="",
                output_dir=str(root / "output"),
                gt_annotation_file=str(gt_path),
            )
            results = compare_localizations(
                cfg,
                pred_files=[str(pred_path)],
                method_names=("Model",),
            )["Model"]

            self.assertEqual(results["gt_count"], 2)
            self.assertEqual(results["prediction_count"], 3)
            self.assertEqual(results["true_positive"], 1)
            self.assertEqual(results["false_positive"], 2)
            self.assertEqual(results["false_negative"], 1)
            self.assertAlmostEqual(results["precision"], 1 / 3)
            self.assertAlmostEqual(results["recall"], 1 / 2)
            self.assertAlmostEqual(results["f1"], 0.4)


if __name__ == "__main__":
    unittest.main()
