import base64
import json
import io
import os
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI

try:
    from aicsimageio import AICSImage
except ImportError:
    raise ImportError("Please install aicsimageio: pip install aicsimageio[pillow]")

from PIL import Image
from config.agent_config import vlm_api_key, vlm_base_url, vlm_model_name

from agent.utils import (
    _parse_json_response,
    merge_module_tasks,
    extract_task_steps,
    convert_to_list
)

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
    def __init__(self, cfg: Optional[Dict] = None, llm_client: Optional[OpenAI] = None, vlm_client: Optional[OpenAI] = None, output_path = None):
        self._cfg = cfg or {}
        self.results: List[CheckResult] = []
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
            client = OpenAI(base_url=vlm_base_url, api_key=vlm_api_key)
            self._llm_client = client
            self._vlm_client = client

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


    # ================== VLM/LLM Calls (Using OpenAI SDK) ==================
    def _call_vlm_custom(self, image_b64: str, prompt: str) -> Tuple[Optional[Dict], str]:
        """Call VLM model (using OpenAI compatible client)"""
        try:
            response = self._vlm_client.chat.completions.create(
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
                max_tokens=self._cfg.get('vlm_max_tokens')
            )
            content = response.choices[0].message.content
            if not content:
                return None, "VLM returned empty content"

            parsed = json.loads(content)
            return parsed, content
        except Exception as e:
            return None, f"VLM call failed: {str(e)}"

    def _call_llm_custom(self, prompt: str) -> Tuple[Optional[str], str]:
        """Call LLM (reuse VLM client, assuming text is supported)"""
        try:
            response = self._llm_client.chat.completions.create(
                model=self._cfg.get('engine'),  # Or specify another text model
                messages=[{"role": "user", "content": prompt}],
                temperature=self._cfg.get('temperature'),
                max_tokens=self._cfg.get('max_tokens')
            )
            content = response.choices[0].message.content.strip()
            return content, content
        except Exception as e:
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

        try:
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

                linear_uint8 = self._array_to_linear_uint8(plane)
                contrast_uint8 = self._array_to_contrast_uint8(plane)
                b64_linear = self._uint8_to_base64_png(linear_uint8)
                b64_contrast = self._uint8_to_base64_png(contrast_uint8)

                # 1. No target
                no_target_res, no_target_raw = self._call_vlm_custom(b64_contrast, self._cfg.get('prompt_no_target'))
                no_target = bool(no_target_res.get("no_target", False)) if no_target_res else False
                no_target_reason = no_target_res.get("reason", "") if no_target_res else "Detection failed"

                # 2. Overexposure
                over_exposed_res, over_exposed_raw = self._call_vlm_custom(b64_linear, self._cfg.get('prompt_over_exposed'))
                over_exposed = bool(over_exposed_res.get("over_exposed", False)) if over_exposed_res else False
                over_exposed_reason = over_exposed_res.get("reason", "") if over_exposed_res else "Detection failed"

                # 3. Out of focus
                out_of_focus_res, out_of_focus_raw = self._call_vlm_custom(b64_contrast, self._cfg.get('prompt_out_of_focus'))
                out_of_focus = bool(out_of_focus_res.get("out_of_focus", False)) if out_of_focus_res else False
                out_of_focus_reason = out_of_focus_res.get("reason", "") if out_of_focus_res else "Detection failed"

                defects = []
                if no_target:
                    defects.append("No target")
                if out_of_focus:
                    defects.append("Out of focus")
                if over_exposed:
                    defects.append("Overexposed")
                combined_reason = "; ".join(defects) if defects else "Normal"

                channel_defect = {
                    "channel_index": c,
                    "channel_name": channel_names[c],
                    "no_target": no_target,
                    "out_of_focus": out_of_focus,
                    "over_exposed": over_exposed,
                    "reason": combined_reason
                }
                channel_defects_list.append(channel_defect)
                raw_responses.extend([no_target_raw, over_exposed_raw, out_of_focus_raw])

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
            return final_result

        except Exception as e:
            error_result = CheckResult(
                defects=ImageDefect(reason=f"Image parsing failed: {str(e)[:50]}"),
                file_info=file_info,
                channel_defects=[]
            )
            self.results.append(error_result)
            return error_result

    def batch_check_from_json(self, image_config: Dict) -> List[CheckResult]:
        results = []
        image_base_dir = self.output_directory

        for file_key, file_info in image_config.items():
            # ✅ 只处理由 microscope 创建的 ome-tiff 文件
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

            file_defect_desc = "; ".join(defective_channels) if defective_channels else "All channels are defect-free (No target/Out of focus/Overexposed)"
            task_defect_summary[filename] = file_defect_desc

        return task_defect_summary

    def generate_task_unified_instruction(self, original_x_y, original_instruction: str):
        task_defect_dict = self.summarize_task_defects()
        global_error_info_parts = ["Summary of channel errors for each file in this task:"]
        for filename, defect_desc in task_defect_dict.items():
            global_error_info_parts.append(f"- {filename}: {defect_desc}")
        global_error_info = "\n".join(global_error_info_parts)

        has_no_target_error = False
        for defect_desc in task_defect_dict.values():
            if "All channels are defect-free" in defect_desc:
                continue
            if "No target" in defect_desc:
                has_no_target_error = True
                break

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

        return tasks if tasks else "Generation failed"
    