import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import localization_toolkit.vlm_inference as vlm_inference_module
from localization_toolkit.pipeline import _vlm_json_to_coco
from localization_toolkit.vlm_inference import _load_api_config, call_qwen_vl_api, parse_detection_results


class ApiConfigurationTests(unittest.TestCase):
    def _write_config(self, directory: str, **fields) -> Path:
        path = Path(directory) / "vlm_api_config.json"
        path.write_text(json.dumps(fields), encoding="utf-8")
        return path

    def test_config_file_placeholder_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._write_config(
                directory,
                api_key="test-api-key",
                api_url="test-api-endpoint",
                model_name="<your-vlm-model-name>",
            )
            with self.assertRaisesRegex(RuntimeError, "MODEL_NAME"):
                _load_api_config(config_path=str(cfg))

    def test_config_file_values_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._write_config(
                directory,
                api_key="test-api-key",
                api_url="test-api-endpoint",
                model_name="test-model-name",
            )
            self.assertEqual(
                _load_api_config(config_path=str(cfg)),
                ("test-api-key", "test-api-endpoint", "test-model-name"),
            )

    def test_missing_config_file_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                _load_api_config(config_path=str(Path(directory) / "missing.json"))

    def test_module_constants_fallback(self):
        values = {
            "API_KEY": "test-api-key",
            "API_URL": "test-api-endpoint",
            "MODEL_NAME": "test-model-name",
        }
        missing_path = Path(tempfile.mkdtemp()) / "no_such_config.json"
        with patch.multiple(vlm_inference_module, **values, DEFAULT_API_CONFIG_PATH=missing_path):
            self.assertEqual(
                _load_api_config(),
                (values["API_KEY"], values["API_URL"], values["MODEL_NAME"]),
            )

    def test_call_api_can_bypass_environment_proxy(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        session = Mock()
        session.post.return_value = response

        with patch.object(vlm_inference_module, "_load_api_config", return_value=("key", "https://example.test/v1", "model")):
            with patch.object(vlm_inference_module.requests, "Session", return_value=session):
                self.assertEqual(call_qwen_vl_api("abc", ["cell"], use_env_proxy=False), "{}")

        self.assertFalse(session.trust_env)


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

    def test_empty_response_raises(self):
        with self.assertRaisesRegex(ValueError, "VLM response is empty"):
            parse_detection_results("", 20, 10, 0.3)


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


class PresetResolutionTests(unittest.TestCase):
    def test_resolves_detector_example_preset(self):
        from localization_toolkit.cli import _list_preset_targets, _resolve_preset

        self.assertIn("2Dcell", _list_preset_targets())
        preset = _resolve_preset(
            "2Dcell",
            image_name="Image_12106.jpg",
            image_path=None,
            manifest_path=None,
            require_model_files=False,
        )

        self.assertEqual(preset["image_id"], 1)
        self.assertEqual(preset["category_id"], 1)
        self.assertEqual(preset["query_texts"], ["2D_cell"])
        self.assertEqual(preset["score_thr"], 0.2)
        self.assertTrue(Path(preset["image_path"]).is_file())
        self.assertTrue(Path(preset["gt_annotation_file"]).is_file())

    def test_missing_preset_image_raises(self):
        from localization_toolkit.cli import _resolve_preset

        with self.assertRaisesRegex(RuntimeError, "missing.jpg"):
            _resolve_preset(
                "2Dcell",
                image_name="missing.jpg",
                image_path=None,
                manifest_path=None,
                require_model_files=False,
            )

    def test_rejects_mismatched_image_name_and_path(self):
        from localization_toolkit.cli import _resolve_preset

        with self.assertRaisesRegex(ValueError, "must match"):
            _resolve_preset(
                "2Dcell",
                image_name="Image_12106.jpg",
                image_path="docs_public/detector_model_examples/testset/2Dcell/images/Image_12107.jpg",
                manifest_path=None,
                require_model_files=False,
            )


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
