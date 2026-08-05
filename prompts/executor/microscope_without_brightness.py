from __future__ import annotations

import re

from prompts.executor.microscope_with_brightness import prompt_olympus as _full_prompt_olympus


def _remove_block(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise ValueError(f"Could not remove block between {start_marker!r} and {end_marker!r}")
    return text[:start] + text[end:]


def _build_no_brightness_prompt(full_prompt: str) -> str:
    text = full_prompt
    text = _remove_block(text, "def set_brightness(brightness: int):", "def set_objective(objective_label: str):")
    text = _remove_block(text, "def perform_autobrightness() -> int:", "# -------------------------- System Control --------------------------")

    replacements = {
        "# -------------------------- Auto Focus / Auto Brightness Adjustment --------------------------":
            "# -------------------------- Auto Focus --------------------------",
        "- Improve image quality by automatically focusing and adjusting brightness before capturing images.":
            "- Improve image quality by automatically focusing before capturing images.",
        "- In brightfield mode, use transmitted illumination with appropriate halogen brightness and relatively low exposure.":
            "- In brightfield mode, use the configured transmitted-illumination channel with relatively low exposure.",
        "- In fluorescence mode, set halogen brightness to 0 and use a relatively higher exposure than in brightfield, while avoiding saturation.":
            "- In fluorescence mode, use a relatively higher exposure than in brightfield while avoiding saturation.",
        "- Imaging Parameters: Brightness({{transmitted_light.min}}→{{transmitted_light.max}}), Exposure Time(0ms→1000ms)":
            "- Imaging Parameters: Exposure Time(0ms→1000ms)",
        "    - Turns off the halogen lamp (sets brightness to 0)\n": "",
        "current_brightness = get_brightness()\n": "",
        "say(f\"[INFO] Current brightness: {current_brightness}\")\n": "",
        "say(\"[INFO] Automatically adjusting brightness for brightfield imaging\")\n": "",
        "say(\"[INFO] Performing automatic brightness adjustment for brightfield\")\n": "",
        "Automatic Image Acquisition Parameter Setting: Set the filter set to brightfield, set the exposure parameter to the current exposure value; set the XY position parameter to the current position, with the size requirement to cover the 3 cm 脳 3 cm tumor section; do not set Z-axis stack parameters or time parameters":
            "Automatic Image Acquisition Parameter Setting: Set the filter set to brightfield, set the exposure parameter to the current exposure value; set the XY position parameter to the current position, with the size requirement to cover the 3 cm 脳 3 cm tumor section; do not set Z-axis stack parameters or time parameters",
        "Parameter Setting: Set the currently used objective lens to 4脳, set the filter set to brightfield mode, and adjust the halogen lamp brightness to a level suitable for brightfield imaging":
            "Parameter Setting: Set the currently used objective lens to 4脳 and set the filter set to brightfield mode",
        "Parameter Setting: Set the currently used objective lens to 20脳, set the filter set to blue fluorescence mode, configure the camera's exposure time to meet the requirements of blue fluorescence imaging, and set the halogen lamp brightness to 0 (in line with fluorescence imaging requirements)":
            "Parameter Setting: Set the currently used objective lens to 20脳, set the filter set to blue fluorescence mode, and configure the camera's exposure time to meet the requirements of blue fluorescence imaging",
        "Auxiliary operations: Automatically adjust the halogen lamp brightness; Perform autofocus on the representative well containing organoids  ":
            "Auxiliary operations: Perform autofocus on the representative well containing organoids  ",
        "Parameter Setting: Set the filter set to brightfield mode, configure the camera exposure time to a low value suitable for brightfield imaging, adjust the objective lens to a suitable magnification for organoid observation (e.g., 10脳), and enable automatic halogen lamp brightness adjustment;":
            "Parameter Setting: Set the filter set to brightfield mode, configure the camera exposure time to a low value suitable for brightfield imaging, and adjust the objective lens to a suitable magnification for organoid observation (e.g., 10脳);",
        "Parameter Setting: Set the filter set to the green fluorescence channel, configure the camera exposure time to meet the requirements of this fluorescent channel imaging, and set the halogen lamp brightness to 0;":
            "Parameter Setting: Set the filter set to the green fluorescence channel and configure the camera exposure time to meet the requirements of this fluorescent channel imaging;",
        "Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0, and finally perform automatic focusing on the current field of view containing organoids.":
            "Auxiliary Operation: First automatically configure the camera exposure time, and then perform automatic focusing on the current field of view containing organoids.",
        "Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0, and finally perform automatic focusing on the current field of view.":
            "Auxiliary Operation: First automatically configure the camera exposure time, and then perform automatic focusing on the current field of view.",
        "\"command\": \"Set the filter set to FITC fluorescence mode and configure the fluorescence imaging condition.\",\n    \"state\": {\n      \"objective\": \"3-LUCPLFLN20XRC\",\n      \"channel\": \"3-FITC\",\n      \"exposure\": 100,\n      \"brightness\": 0\n    },\n":
            "\"command\": \"Set the filter set to FITC fluorescence mode and configure the fluorescence imaging condition.\",\n    \"state\": {\n      \"objective\": \"3-LUCPLFLN20XRC\",\n      \"channel\": \"3-FITC\",\n      \"exposure\": 100\n    },\n",
        "\"command\": \"Parameter Setting: Set the filter set to FITC fluorescence mode; \\n#Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0.\"":
            "\"command\": \"Parameter Setting: Set the filter set to FITC fluorescence mode; \\n#Auxiliary Operation: First automatically configure the camera exposure time.\"",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r", brightness: \d+", "", text)
    text = re.sub(r'^\s*"brightness":\s*\d+,?\n', "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*perform_autobrightness.*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*set_brightness\(.*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*get_brightness\(.*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*halogen lamp brightness.*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*Auto-brightness.*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*automatic brightness adjustment.*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*Current brightness.*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    forbidden_tokens = ("set_brightness", "get_brightness", "perform_autobrightness")
    for token in forbidden_tokens:
        if token in text:
            raise ValueError(f"Brightness-capable token {token!r} still present in no-brightness prompt")
    return text


prompt_olympus = _build_no_brightness_prompt(_full_prompt_olympus)
