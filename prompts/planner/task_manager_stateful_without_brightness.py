from __future__ import annotations

import re

from prompts.planner.task_manager_stateful_with_brightness import prompt_manger as _BASE_PROMPT


def _remove_current_state_brightness_fields(text: str) -> str:
    text = re.sub(r'^\s*"brightness":\s*\d+\s*\n?', "", text, flags=re.MULTILINE)
    text = text.replace(',\n}\n</Current System State>', '\n}\n</Current System State>')
    return text


def _build_no_brightness_prompt() -> str:
    text = _BASE_PROMPT
    replacements = {
        "- Configure the camera's exposure time, the currently used objective lens (4×, 10×, 20×, 40×, 60×), the filter set, and the brightness of the halogen lamp for brightfield illumination.":
            "- Configure the camera's exposure time, the currently used objective lens (4×, 10×, 20×, 40×, 60×), and the filter set.",
        "- Obtain the camera's current exposure time, currently used objective lens, filter set, and the brightness of the halogen lamp for brightfield illumination.":
            "- Obtain the camera's current exposure time, currently used objective lens, and filter set.",
        "- Automatically adjust the halogen lamp brightness.\n":
            "- Automatically adjust the camera exposure.\n",
        "- Dynamically adjust brightness and focus to ensure images are clear. The current brightness and focus should not be assumed to be already appropriate.":
            "- Dynamically adjust exposure and focus to ensure images are clear. The current exposure and focus should not be assumed to be already appropriate.",
        "- After replacing the objective lens of a microscope, the target may be lost due to the difference in magnification. Therefore, it is necessary to move to the target position and recalibrate the brightness and focus.":
            "- After replacing the objective lens of a microscope, the target may be lost due to the difference in magnification. Therefore, it is necessary to move to the target position and recalibrate exposure and focus.",
        "- In microscope operation, exposure values should be adjusted first, followed by brightness adjustment, and finally focusing.":
            "- In microscope operation, exposure values should be adjusted before focusing.",
        "- When switching between different fluorescent channels, it is necessary to adjust brightness and exposure parameters. ":
            "- When switching between different fluorescent channels, it is necessary to adjust exposure parameters. ",
        "- In brightfield mode, the filter set should be set to brightfield mode, with low exposure parameters used and automatic brightness adjustment performed. ":
            "- In brightfield mode, the filter set should be set to brightfield mode, with exposure adjusted appropriately for transmitted-light imaging. ",
        "- In fluorescent channels, the filter set should be set to the corresponding fluorescent mode, brightness should be set to 0, and high exposure parameters used.":
            "- In fluorescent channels, the filter set should be set to the corresponding fluorescent mode, with exposure adjusted appropriately while avoiding saturation.",
        '"command": "Auxiliary Operation: Firstly, Perform automatic brightness adjustment ; Secondly, Perform auto-focus;"':
            '"command": "Auxiliary Operation: Firstly, Perform automatic exposure adjustment ; Secondly, Perform auto-focus;"',
        '"command": "Auxiliary Operation: Firstly, Perform automatic brightness adjustment ; Secondly, Perform auto-focus; "':
            '"command": "Auxiliary Operation: Firstly, Perform automatic exposure adjustment ; Secondly, Perform auto-focus; "',
        '"command": "Auxiliary operation: Firstly, Perform automatic brightness adjustment ; Secondly, Perform auto-focus;"':
            '"command": "Auxiliary operation: Firstly, Perform automatic exposure adjustment ; Secondly, Perform auto-focus;"',
        '"command": "Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0, and finally perform automatic focusing on the current field of view containing organoids."':
            '"command": "Auxiliary Operation: First automatically configure the camera exposure time, and then perform automatic focusing on the current field of view containing organoids."',
        '"command": "Parameter Setting: Set the filter set to FITC fluorescence mode; \\n#Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0."':
            '"command": "Parameter Setting: Set the filter set to FITC fluorescence mode; \\n#Auxiliary Operation: First automatically configure the camera exposure time."',
        '"command": "Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0, and finally perform automatic focusing on the current field of view."':
            '"command": "Auxiliary Operation: First automatically configure the camera exposure time, and then perform automatic focusing on the current field of view."',
        '"command": "Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0."':
            '"command": "Auxiliary Operation: First automatically configure the camera exposure time."',
        '"command": "Parameter Setting: Set the currently used objective lens to 40×; Set the filter set to FITC fluorescence mode; \\n#Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0, and finally perform automatic focusing on the current field of view containing the cell sections"':
            '"command": "Parameter Setting: Set the currently used objective lens to 40×; Set the filter set to FITC fluorescence mode; \\n#Auxiliary Operation: First automatically configure the camera exposure time, and then perform automatic focusing on the current field of view containing the cell sections"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = _remove_current_state_brightness_fields(text)
    text = re.sub(
        r'"\s*command":\s*"Auxiliary Operation: Firstly,\s*Perform automatic brightness adjustment\s*;\s*Secondly,\s*Perform auto-focus;\s*"',
        '"command": "Auxiliary Operation: Perform auto-focus; "',
        text,
    )
    text = re.sub(
        r'"\s*command":\s*"Auxiliary operation: Firstly,\s*Perform automatic brightness adjustment\s*;\s*Secondly,\s*Perform auto-focus;"',
        '"command": "Auxiliary operation: Perform auto-focus;"',
        text,
    )
    text = re.sub(
        r"then set the light source brightness to 0, and finally perform automatic focusing",
        "and then perform automatic focusing",
        text,
    )
    text = re.sub(
        r"then set the light source brightness to 0\.",
        ".",
        text,
    )
    text = re.sub(r'^\s*"brightness":\s*0\s*\n?', "", text, flags=re.MULTILINE)
    text = text.replace(',\n    },', '\n    },')
    text = text.replace(',\n}\n</Historical Tasks>', '\n}\n</Historical Tasks>')
    if "automatic brightness adjustment" in text.lower():
        raise ValueError("Brightness-specific planner text still present in no-brightness prompt")
    if "light source brightness to 0" in text.lower():
        raise ValueError("Fluorescence brightness reset text still present in no-brightness prompt")
    return text


prompt_manger = _build_no_brightness_prompt()
