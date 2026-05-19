import base64
import json
import io
import logging
import os
import re
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from adapters.llm_clients import create_chat_completion
from bootstrap.config import load_model_config

try:
    from aicsimageio import AICSImage
except ImportError:  # pragma: no cover - optional in mock-only environments
    AICSImage = None

from PIL import Image

from agent.utils import convert_to_list
from utils.cli_logging import get_cli_logger


logger = get_cli_logger("CHECKER")

@dataclass
class ImageDefect:
    """Data class for single-channel/overall image defects"""
    no_target: bool = False  # No target defect
    out_of_focus: bool = False  # Out-of-focus defect
    over_exposed: bool = False  # Overexposure defect
    reason: str = ""  # Defect description


@dataclass
class CheckResult:
    """Data class for complete detection results of a single image (image_id removed)"""
    defects: ImageDefect
    raw_vlm_response: str = ""
    file_info: Optional[Dict] = None
    channel_defects: Optional[List[Dict]] = None  # Channel-level detailed errors

    def to_dict(self) -> Dict:
        """Convert to dictionary, supporting JSON serialization (image_id removed)"""
        return {
            "defects": {
                "no_target": self.defects.no_target,
                "out_of_focus": self.defects.out_of_focus,
                "over_exposed": self.defects.over_exposed,
                "reason": self.defects.reason
            },
            "channel_defects": self.channel_defects,
            "raw_vlm_response": self.raw_vlm_response,
            "file_info": self.file_info
        }

class ExperimentCheckAgent:
    _MOCK_MICROSCOPE_HEADER = "mock microscope acquisition"

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        llm_client: Optional[OpenAI] = None,
        vlm_client: Optional[OpenAI] = None,
        output_path=None,
        history_manager=None,
    ):
        self._cfg = cfg or {}
        self.results: List[CheckResult] = []
        self._history_manager = history_manager
        self.color_channel_mapping = {
            (255, 0, 0): "TRITC",
            (0, 255, 0): "FITC",
            (0, 0, 255): "DAPI",  # Note: Original code maps (0,0,255) -> DAPI, which is blue in RGB, corrected to common combination here
            (128, 120, 128): "brighted"  # Example, adjust according to your actual data
        }
        self.output_directory: str = output_path
        # Initialize OpenAI compatible clients
        if llm_client is not None and vlm_client is not None:
            self._llm_client = llm_client
            self._vlm_client = vlm_client
        else:
            model_config = load_model_config()
            self._llm_client = OpenAI(base_url=model_config.base_url, api_key=model_config.openai_api_key)
            self._vlm_client = OpenAI(base_url=model_config.vlm_base_url, api_key=model_config.vlm_api_key)

    def _extract_json_object(self, content: str) -> Optional[Dict[str, Any]]:
        if not content:
            return None

        cleaned = content.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.IGNORECASE)
        candidates = []
        if fenced_match:
            candidates.append(fenced_match.group(1))
        if cleaned.startswith("{") and cleaned.endswith("}"):
            candidates.append(cleaned)

        brace_starts = [idx for idx, ch in enumerate(cleaned) if ch == "{"]
        for start in brace_starts:
            depth = 0
            for end in range(start, len(cleaned)):
                char = cleaned[end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(cleaned[start:end + 1])
                        break

        seen = set()
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            for normalized in (candidate, candidate.replace("'", '"')):
                try:
                    parsed = json.loads(normalized)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
        return None

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "1", "present", "detected"}:
                return True
            if normalized in {"false", "no", "n", "0", "absent", "not detected"}:
                return False
        return False

    def _pick_reason(self, parsed: Optional[Dict[str, Any]], fallback: str) -> str:
        if not parsed:
            return fallback
        for key in ("reason", "description", "analysis", "message", "detail"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return fallback

    def _pick_reason_for_defect(
        self,
        parsed: Optional[Dict[str, Any]],
        defect_key: str,
        fallback: str,
    ) -> str:
        if not parsed:
            return fallback
        reason_value = parsed.get("reason")
        if isinstance(reason_value, dict):
            defect_reason = reason_value.get(defect_key)
            if isinstance(defect_reason, str) and defect_reason.strip():
                return defect_reason.strip()
        return self._pick_reason(parsed, fallback)

    def _normalize_vlm_result(self, parsed: Optional[Dict[str, Any]], defect_key: str) -> Dict[str, Any]:
        if not parsed:
            return {defect_key: False, "reason": "Detection failed"}

        aliases = {
            "no_target": ("no_target", "missing_target", "target_missing", "has_target"),
            "over_exposed": ("over_exposed", "overexposed", "is_overexposed", "exposure_issue"),
            "out_of_focus": ("out_of_focus", "out_of_focus_blur", "is_blurry", "blurred", "focus_issue"),
        }
        keys = aliases.get(defect_key, (defect_key,))
        detected = False
        for key in keys:
            if key not in parsed:
                continue
            value = self._coerce_bool(parsed.get(key))
            if defect_key == "no_target" and key == "has_target":
                value = not value
            detected = value
            break

        return {
            defect_key: detected,
            "reason": self._pick_reason_for_defect(parsed, defect_key, "Detection completed"),
        }

    def _result_is_defect_free(self, result: CheckResult) -> bool:
        defects = result.defects
        return not (defects.no_target or defects.out_of_focus or defects.over_exposed)

    def has_any_no_target(self) -> bool:
        return any(result.defects.no_target for result in self.results)

    def all_results_defect_free(self) -> bool:
        return bool(self.results) and all(self._result_is_defect_free(result) for result in self.results)

    def clear_history_results(self):
        """Clear all historical detection records"""
        self.results.clear()


    def _parse_channel_names(self, description: str) -> List[str]:
        """Parse channel names (e.g., DAPI/FITC) from file description"""
        pattern = r"channel_names:\s*\[(.*?)\]"
        match = re.search(pattern, description)
        if not match:
            return []

        tuple_str = match.group(1)
        tuple_matches = re.findall(r"\((\d+),\s*(\d+),\s*(\d+)\)", tuple_str)
        channel_names = []
        for rgb in tuple_matches:
            rgb_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            channel_name = self.color_channel_mapping.get(rgb_tuple, f"Ch{len(channel_names)}")
            channel_names.append(channel_name)
        return channel_names

    def _read_file_prefix(self, image_path: str, byte_count: int = 256) -> bytes:
        with open(image_path, "rb") as handle:
            return handle.read(byte_count)

    def _is_mock_microscope_placeholder(self, image_path: str) -> bool:
        try:
            prefix = self._read_file_prefix(image_path)
        except OSError:
            return False
        return prefix.decode("utf-8", errors="ignore").startswith(self._MOCK_MICROSCOPE_HEADER)

    def _build_mock_placeholder_result(
        self,
        image_path: str,
        file_info: Optional[Dict] = None,
    ) -> CheckResult:
        channel_names: List[str] = []
        if file_info and "description" in file_info:
            channel_names = self._parse_channel_names(str(file_info["description"]))
        if not channel_names:
            channel_names = ["Ch0"]

        channel_defects_list = [
            {
                "channel_index": index,
                "channel_name": channel_name,
                "no_target": False,
                "out_of_focus": False,
                "over_exposed": False,
                "reason": "Normal",
            }
            for index, channel_name in enumerate(channel_names)
        ]

        final_result = CheckResult(
            defects=ImageDefect(reason="All channels are normal"),
            raw_vlm_response="Skipped VLM validation for deterministic mock acquisition placeholder.",
            file_info=file_info,
            channel_defects=channel_defects_list,
        )
        self.results.append(final_result)
        if self._history_manager:
            self._history_manager.record_interaction(
                agent_name="checker",
                event_type="checker_mock_image_skipped",
                message="Checker skipped image parsing for a deterministic mock acquisition placeholder.",
                payload={
                    "image_path": image_path,
                    "file_info": file_info or {},
                    "result": final_result.to_dict(),
                },
            )
        return final_result


    # ================== VLM/LLM Calls (Using OpenAI SDK) ==================
    def _call_vlm_custom(self, image_b64: str, prompt: str) -> Tuple[Optional[Dict], str]:
        """Call VLM model (using OpenAI compatible client)"""
        try:
            response = create_chat_completion(
                self._vlm_client,
                model=self._cfg.get('vlm_engine'),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=self._cfg.get('vlm_temperature'),
                seed=self._cfg.get('seed'),
                max_tokens=self._cfg.get('vlm_max_tokens'),
            )
            content = response.choices[0].message.content
            if not content:
                return None, "VLM returned empty content"
            parsed = self._extract_json_object(content)
            if parsed is None:
                logger.warning("VLM response is not valid JSON object: %s", content[:300])
                if self._history_manager:
                    self._history_manager.record_interaction(
                        agent_name="checker",
                        event_type="checker_vlm_parse_failed",
                        message="Checker received a non-JSON VLM response.",
                        payload={"prompt": prompt, "raw_response": content},
                    )
                return None, content
            if self._history_manager:
                self._history_manager.record_interaction(
                    agent_name="checker",
                    event_type="checker_vlm_response",
                    message="Checker completed one VLM inspection call.",
                    payload={"prompt": prompt, "parsed_response": parsed, "raw_response": content},
                )
            return parsed, content
        except Exception as e:
            if self._history_manager:
                self._history_manager.record_interaction(
                    agent_name="checker",
                    event_type="checker_vlm_failed",
                    message="Checker VLM call failed.",
                    payload={"prompt": prompt, "error": str(e)},
                )
            return None, f"VLM call failed: {str(e)}"

    def _call_llm_custom(self, prompt: str) -> Tuple[Optional[str], str]:
        """Call LLM (reuse VLM client, assuming text is supported)"""
        try:
            response = create_chat_completion(
                self._llm_client,
                model=self._cfg.get('engine'),  # Or specify another text model
                messages=[{"role": "user", "content": prompt}],
                temperature=self._cfg.get('temperature'),
                seed=self._cfg.get('seed'),
                max_tokens=self._cfg.get('max_tokens'),
            )
            content = response.choices[0].message.content.strip()
            if self._history_manager:
                self._history_manager.record_interaction(
                    agent_name="checker",
                    event_type="checker_llm_response",
                    message="Checker generated retry guidance.",
                    payload={"prompt": prompt, "raw_response": content},
                )
            return content, content
        except Exception as e:
            if self._history_manager:
                self._history_manager.record_interaction(
                    agent_name="checker",
                    event_type="checker_llm_failed",
                    message="Checker LLM call failed.",
                    payload={"prompt": prompt, "error": str(e)},
                )
            return None, f"LLM call failed: {str(e)}"

    # ================== Image Preprocessing ==================
    def _array_to_linear_uint8(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        elif arr.dtype == np.uint16:
            return (arr >> 8).astype(np.uint8)
        elif arr.dtype == np.uint32:
            return (arr >> 24).astype(np.uint8)
        elif arr.dtype.kind == 'f':
            max_val = arr.max()
            if max_val <= 0:
                return np.zeros_like(arr, dtype=np.uint8)
            scaled = arr / max_val
            return np.clip(scaled * 255, 0, 255).astype(np.uint8)
        else:
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max == arr_min:
                return np.zeros_like(arr, dtype=np.uint8)
            return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)

    def _array_to_contrast_uint8(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype != np.uint8:
            p_low, p_high = np.percentile(arr, [0.01, 99.9])
            if p_high == p_low:
                arr_norm = np.zeros_like(arr)
            else:
                arr_norm = np.clip((arr - p_low) / (p_high - p_low), 0, 1)
            arr_uint8 = (arr_norm * 255).astype(np.uint8)
        else:
            arr_uint8 = arr
        return arr_uint8

    def _uint8_to_base64_png(self, arr_uint8: np.ndarray) -> str:
        if arr_uint8.ndim == 2:
            arr_rgb = np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)
        else:
            arr_rgb = arr_uint8
        img = Image.fromarray(arr_rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ================== Image Detection ==================
    def check_ome_tiff_as_single(
        self,
        image_path: str,
        file_info: Optional[Dict] = None
    ) -> CheckResult:
        if not os.path.exists(image_path):
            error_result = CheckResult(
                defects=ImageDefect(reason=f"File does not exist: {image_path}"),
                file_info=file_info,
                channel_defects=[]
            )
            self.results.append(error_result)
            return error_result

        if self._is_mock_microscope_placeholder(image_path):
            return self._build_mock_placeholder_result(image_path, file_info=file_info)

        try:
            if AICSImage is None:
                raise ImportError("Please install aicsimageio: pip install aicsimageio[pillow]")
            img = AICSImage(image_path)
            dims = img.dims.order
            num_channels = img.shape[dims.index('C')]

            channel_names = []
            if file_info and "description" in file_info:
                channel_names = self._parse_channel_names(file_info["description"])
            while len(channel_names) < num_channels:
                channel_names.append(f"Ch{len(channel_names)}")

            channel_defects_list = []
            raw_responses = []

            for c in range(num_channels):
                indexer = {"C": c}
                if 'T' in dims:
                    indexer["T"] = 0
                if 'Z' in dims:
                    indexer["Z"] = 0
                plane = img.get_image_data("YX", **indexer)

                contrast_uint8 = self._array_to_contrast_uint8(plane)
                b64_contrast = self._uint8_to_base64_png(contrast_uint8)

                quality_res, quality_raw = self._call_vlm_custom(
                    b64_contrast,
                    self._cfg.get('prompt_quality_check'),
                )
                no_target_info = self._normalize_vlm_result(quality_res, "no_target")
                no_target = no_target_info["no_target"]
                over_exposed_info = self._normalize_vlm_result(quality_res, "over_exposed")
                over_exposed = over_exposed_info["over_exposed"]
                out_of_focus_info = self._normalize_vlm_result(quality_res, "out_of_focus")
                out_of_focus = out_of_focus_info["out_of_focus"]

                defects = []
                if no_target:
                    defects.append("No target")
                if out_of_focus:
                    defects.append("Out of focus")
                if over_exposed:
                    defects.append("Overexposed")
                if defects:
                    detail_reasons = [
                        info["reason"]
                        for info in (no_target_info, out_of_focus_info, over_exposed_info)
                        if info["reason"] and info["reason"] != "Detection completed"
                    ]
                    combined_reason = "; ".join(defects + detail_reasons)
                else:
                    combined_reason = "Normal"

                channel_defect = {
                    "channel_index": c,
                    "channel_name": channel_names[c],
                    "no_target": no_target,
                    "out_of_focus": out_of_focus,
                    "over_exposed": over_exposed,
                    "reason": combined_reason
                }
                channel_defects_list.append(channel_defect)
                raw_responses.append(quality_raw)

            # Overall defects
            final_no_target = any(cd["no_target"] for cd in channel_defects_list)
            final_out_of_focus = any(cd["out_of_focus"] for cd in channel_defects_list)
            final_over_exposed = any(cd["over_exposed"] for cd in channel_defects_list)

            reason_parts = []
            for cd in channel_defects_list:
                if cd["reason"] != "Normal":
                    reason_parts.append(f"{cd['channel_name']}: {cd['reason']}")
            final_reason = "; ".join(reason_parts) if reason_parts else "All channels are normal"

            final_result = CheckResult(
                defects=ImageDefect(
                    no_target=final_no_target,
                    out_of_focus=final_out_of_focus,
                    over_exposed=final_over_exposed,
                    reason=final_reason
                ),
                raw_vlm_response=" | ".join(raw_responses),
                file_info=file_info,
                channel_defects=channel_defects_list
            )
            self.results.append(final_result)
            if self._history_manager:
                self._history_manager.record_interaction(
                    agent_name="checker",
                    event_type="checker_image_result",
                    message="Checker completed image validation for one file.",
                    payload={
                        "image_path": image_path,
                        "file_info": file_info or {},
                        "result": final_result.to_dict(),
                    },
                )
            return final_result

        except Exception as e:
            error_result = CheckResult(
                defects=ImageDefect(reason=f"Image parsing failed: {str(e)[:50]}"),
                file_info=file_info,
                channel_defects=[]
            )
            self.results.append(error_result)
            if self._history_manager:
                self._history_manager.record_interaction(
                    agent_name="checker",
                    event_type="checker_image_failed",
                    message="Checker failed while parsing or validating an image file.",
                    payload={"image_path": image_path, "error": str(e)},
                )
            return error_result

    def batch_check_from_json(self, image_config: Dict) -> List[CheckResult]:
        results = []
        image_base_dir = self.output_directory

        for file_key, file_info in image_config.items():
            # Only process OME-TIFF files created by the microscope.
            if file_info.get("created_by") != "microscope":
                continue
            if file_info.get("file_type") != "ome-tiff":
                continue

            filename = file_info.get("filename", file_key)
            image_path = os.path.join(image_base_dir, filename) if image_base_dir else filename

            result = self.check_ome_tiff_as_single(
                image_path=image_path,
                file_info=file_info
            )
            results.append(result)

        if self._history_manager:
            self._history_manager.record_interaction(
                agent_name="checker",
                event_type="checker_batch_completed",
                message="Checker completed validation for the current batch.",
                payload={"checked_files": len(results), "results": [result.to_dict() for result in results]},
            )

        return results
    
    def summarize_task_defects(self) -> Dict[str, str]:
        if not self.results:
            raise ValueError("No detection results available, please perform image check first")

        task_defect_summary = {}
        for result in self.results:
            filename = result.file_info.get("filename", "Unknown file") if result.file_info else "Unknown file"

            defective_channels = []
            if result.channel_defects:
                for ch in result.channel_defects:
                    defects = []
                    if ch['no_target']:
                        defects.append("No target")
                    if ch['out_of_focus']:
                        defects.append("Out of focus")
                    if ch['over_exposed']:
                        defects.append("Overexposed")
                    if defects:
                        defective_channels.append(f"{ch['channel_name']}({','.join(defects)})")

            file_defect_desc = "; ".join(defective_channels) if defective_channels else "All channels are defect-free"
            task_defect_summary[filename] = file_defect_desc

        return task_defect_summary

    def generate_task_unified_instruction(self, original_x_y, original_instruction: str):
        task_defect_dict = self.summarize_task_defects()
        global_error_info_parts = ["Summary of channel errors for each file in this task:"]
        for filename, defect_desc in task_defect_dict.items():
            global_error_info_parts.append(f"- {filename}: {defect_desc}")
        global_error_info = "\n".join(global_error_info_parts)

        has_no_target_error = self.has_any_no_target()

        if has_no_target_error:
            selected_prompt = self._cfg.get('instruction_prompt_with_no_target').format(
                original_x_y = original_x_y,
                original_instruction=original_instruction,
                global_error_info=global_error_info
            )
        else:
            selected_prompt = self._cfg.get('instruction_prompt_without_no_target').format(
                original_instruction=original_instruction,
                global_error_info=global_error_info
            )

        unified_instruction, _ = self._call_llm_custom(selected_prompt)
        tasks = convert_to_list(unified_instruction)

        if self._history_manager:
            self._history_manager.record_interaction(
                agent_name="checker",
                event_type="checker_retry_plan",
                message="Checker proposed a revised task plan after validation.",
                payload={
                    "has_no_target_error": has_no_target_error,
                    "task_defect_summary": task_defect_dict,
                    "revised_tasks": tasks if tasks else unified_instruction,
                },
            )

        return tasks if tasks else "Generation failed"
    
