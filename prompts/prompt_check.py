prompt_no_target = """
        You are analyzing a single-channel grayscale fluorescence microscopy image.
        Please determine whether the image has the "no target" defect: that is, there are no identifiable valid biological structures (such as cell nuclei, fluorescently labeled regions) in the image, only blank or noise.
        The output format must be strictly JSON, containing two fields: no_target (boolean value), reason (description no more than 30 words)
"""

prompt_over_exposed = """
        You are analyzing a single-channel grayscale fluorescence microscopy image.
        Please determine whether the image has the "over exposed" defect: that is, there are large areas of pure white regions in the image, without texture gradients, and details are lost.
        The output format must be strictly JSON, containing two fields: over_exposed (boolean value), reason (description no more than 30 words)
"""

prompt_out_of_focus = """
        You are analyzing a single-channel grayscale fluorescence microscopy image.
        Please determine whether the image has the "out of focus" defect: that is, the entire image is blurred, has no clear edges, and cannot distinguish the details of biological structures.
        The output format must be strictly JSON, containing two fields: out_of_focus (boolean value), reason (description no more than 30 words)
"""

prompt_quality_check = """
        You are analyzing a single-channel grayscale microscopy image for acquisition quality.
        Evaluate the image for these three defects at the same time:

        1. "no_target":
        Mark true when there is no identifiable valid biological structure in the image, only blank background, noise, debris, or meaningless texture.

        2. "over_exposed":
        Mark true when there are obvious saturated bright regions with lost intensity detail and missing texture gradients.

        3. "out_of_focus":
        Mark true for mild, moderate, or severe defocus. This includes cases where biological structures are still visible but their boundaries are noticeably soft, sharpness is reduced, or fine details are weakened compared with a properly focused image.

        Rules:
        - Mild defocus should still be marked as "out_of_focus" = true.
        - Do not require the whole image to be completely blurred before marking "out_of_focus" = true.
        - If visible structures remain crisp, low signal, low contrast, or sparse targets alone are not enough for "out_of_focus".
        - If the main problem is missing biological structure, prefer "no_target" over "out_of_focus".
        - If the main problem is saturation with lost highlight detail, prefer "over_exposed" over "out_of_focus".

        Return strictly valid JSON with exactly four top-level fields:
        - "no_target": boolean
        - "over_exposed": boolean
        - "out_of_focus": boolean
        - "reason": object

        The "reason" object must contain exactly three string fields:
        - "no_target": brief reason, no more than 30 words
        - "over_exposed": brief reason, no more than 30 words
        - "out_of_focus": brief reason, no more than 30 words

        Return JSON only. Do not return Markdown, code fences, or any extra text.
"""

instruction_prompt_with_no_target = """        
        Modification Requirements:
        1. Retain the core intent of the original instruction, prioritize handling the no-target error (re-focus before executing the original command);
        2. The language is concise and professional, with minimal modifications, only return the corrected instruction without additional explanations.
        3. The corrected instruction MUST have the same format as the original instruction: 
        - It must be a list of dictionaries;
        - Each dictionary MUST contain the three mandatory key fields: 'subtask_index' (integer type, keep original value), 'module' (string type), 'command' (string type);
        - Do NOT modify/delete/rename any of the three mandatory fields;
        - Do NOT add or delete any sub-task dictionary in the list;
        4. Move to the initial position and re-focus before re-executing the original instruction (only modify the 'command' field content of the corresponding sub-task, do not touch other fields)
        5. Return JSON only. Do not return explanations, Markdown, or prose before or after the JSON.
        6. Use a JSON array of objects with double quotes for all keys and string values.
        7. Do not return Python-style lists or dictionaries with single quotes.

        Example:
        Initial Position:
        original_x_y = 5000, 5000
        Original Instruction:
        [{{'subtask_index': 1, 'module': 'Microscope Operation Platform', 'command': 'Image Automatic Acquisition Parameter Setting: Configure the filter set for FITC fluorescence channel and set the corresponding exposure parameter to the current camera exposure time; configure the XY position parameter to the current position, with size requirement matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the monkey kidney cell section image with 40× objective'}}, {{'subtask_index': 2, 'module': 'Image Analysis Platform', 'command': 'Image Import: Import the acquired 40× magnified image of the monkey kidney cell sections; \n#Target Detection: Detect monkey kidney cell sections in the imported 40× magnified image and save the detection results as a JSON file'}}, {{'subtask_index': 3, 'module': 'Microscope Operation Platform', 'command': 'Parameter Setting: Set the currently used objective lens to 60×; \n#Target Position Loading: Load the target position bounding boxes of detected monkey kidney cell sections from the JSON file; \n#Position Control: Move to the location of the first detected cell section; \n#Parameter Setting: Set the filter set to FITC fluorescence mode; \n#Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0, and finally perform automatic focusing on the current field of view; \n#Image Automatic Acquisition Parameter Setting: Configure the filter sets for DAPI, FITC, and TRITC fluorescence channels and set their corresponding exposure parameters; configure the XY position parameter to the loaded positions of all detected monkey kidney cell sections, with size requirement matching each detected section; do not configure Z-axis stack parameters; do not configure time parameters; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture multi-channel images (DAPI, FITC, TRITC) of all detected monkey kidney cell sections with 60× objective'}}]
        Error Information:
        Summary of channel errors for each file in this task:
        - sample_A.ome.tif：DAPI(No Target); FITC(Over Exposed)
        - sample_B.ome.tif：All channels are defect-free (No Target/Out of Focus/Over Exposed)
        - sample_C.ome.tif：TRITC(Out of Focus)
        Corrected Instruction:
        [{{'subtask_index': 1, 'module': 'Microscope Operation Platform', 'command': 'Position Control: Move to initial position (5000, 5000); #Auxiliary Operation: Perform automatic focusing on the current field of view; #Image Automatic Acquisition Parameter Setting: Configure the filter set for FITC fluorescence channel and set the corresponding exposure parameter to the current camera exposure time; configure the XY position parameter to the current position, with size requirement matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the monkey kidney cell section image with 40× objective'}}, {{'subtask_index': 2, 'module': 'Image Analysis Platform', 'command': 'Image Import: Import the acquired 40× magnified image of the monkey kidney cell sections; \n#Target Detection: Detect monkey kidney cell sections in the imported 40× magnified image and save the detection results as a JSON file'}}, {{'subtask_index': 3, 'module': 'Microscope Operation Platform', 'command': 'Parameter Setting: Set the currently used objective lens to 60×; \n#Target Position Loading: Load the target position bounding boxes of detected monkey kidney cell sections from the JSON file; \n#Position Control: Move to the location of the first detected cell section; \n#Parameter Setting: Set the filter set to FITC fluorescence mode; \n#Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0, and finally perform automatic focusing on the current field of view; \n#Image Automatic Acquisition Parameter Setting: Configure the filter sets for DAPI, FITC, and TRITC fluorescence channels and set their corresponding exposure parameters; configure the XY position parameter to the loaded positions of all detected monkey kidney cell sections, with size requirement matching each detected section; do not configure Z-axis stack parameters; do not configure time parameters; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture multi-channel images (DAPI, FITC, TRITC) of all detected monkey kidney cell sections with 60× objective'}}]

        Initial Position (corresponding to the overall task of this time):
        {original_x_y}
        Original Instruction (corresponding to the overall task of this time):
        {original_instruction}

        Global Error Information of this Task:
        {global_error_info}
"""


instruction_prompt_without_no_target = """
        Modification Requirements:
        1. Retain the core intent of the original instruction, optimize for out-of-focus and over-exposed errors, and clearly reference the corresponding error file names in the revised command. Only retain subtasks related to error files and remove subtasks for normal files.
        2. The language is concise and professional, only return the corrected instruction without additional explanations.
        3. The corrected instruction MUST have the same format as the original instruction:
        - It must be a list of dictionaries;
        - Each dictionary MUST contain the three mandatory key fields: 'subtask_index' (integer type, keep original value), 'module' (string type), 'command' (string type);
        - Do NOT modify/delete/rename any of the three mandatory fields;
        - Clearly specify the error file names in the 'command' field.
        4. Return JSON only. Do not return explanations, Markdown, or prose before or after the JSON.
        5. Use a JSON array of objects with double quotes for all keys and string values.
        6. Do not return Python-style lists or dictionaries with single quotes.
        7. For any out-of-focus defects, always apply a conservative retry strategy limited to the affected files only.
        8. For each affected target, autofocus MUST be performed individually at its own position before reacquisition.
        9. Do NOT reuse a single autofocus result across multiple targets under any circumstances.

        Example:
        Original Instruction:
        [{{'subtask_index': 1, 'module': 'Microscope Operation Platform', 'command': 'Image Automatic Acquisition Parameter Setting: Configure the filter set for FITC fluorescence channel and set the corresponding exposure parameter to the current camera exposure time; configure the XY position parameter to the current position, with size requirement matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the monkey kidney cell section image with 40× objective'}}, {{'subtask_index': 2, 'module': 'Image Analysis Platform', 'command': 'Image Import: Import the acquired 40× magnified image of the monkey kidney cell sections; \n#Target Detection: Detect monkey kidney cell sections in the imported 40× magnified image and save the detection results as a JSON file'}}, {{'subtask_index': 3, 'module': 'Microscope Operation Platform', 'command': 'Parameter Setting: Set the currently used objective lens to 60×; \n#Target Position Loading: Load the target position bounding boxes of detected monkey kidney cell sections from the JSON file; \n#Position Control: Move to the location of the first detected cell section; \n#Parameter Setting: Set the filter set to FITC fluorescence mode; \n#Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0, and finally perform automatic focusing on the current field of view; \n#Image Automatic Acquisition Parameter Setting: Configure the filter sets for DAPI, FITC, and TRITC fluorescence channels and set their corresponding exposure parameters; configure the XY position parameter to the loaded positions of all detected monkey kidney cell sections, with size requirement matching each detected section; do not configure Z-axis stack parameters; do not configure time parameters; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture multi-channel images (DAPI, FITC, TRITC) of all detected monkey kidney cell sections with 60× objective'}}]
        Error Information:
        Summary of channel errors for each file in this task:
        - sample_1_40x.ome.tif:All channels are normal
        - sample_1_60x.ome.tif:All channels are normal
        - sample_2_60x.ome.tif:TRITC(Out of Focus)
        - sample_3_60x.ome.tif:TRITC(Out of Focus)
        Corrected Instruction:
        [{{'subtask_index': 1, 'module': 'Microscope Operation Platform', 'command': 'Parameter Setting: Set the currently used objective lens to 60×; \n#Target Position Loading: Load the target position bounding boxes of detected monkey kidney cell sections from the JSON file; \n#Position Control: Move to the location of the second detected cell section; \n#Parameter Setting: Set the filter set to FITC fluorescence mode; \n#Auxiliary Operation: First automatically configure the camera exposure time, then set the light source brightness to 0, and finally perform automatic focusing on the current field of view; \n#Image Automatic Acquisition Parameter Setting: Configure the filter sets for DAPI, FITC, and TRITC fluorescence channels and set their corresponding exposure parameters; configure the XY position parameter to the loaded positions of all detected monkey kidney cell sections, with size requirement matching each detected section; do not configure Z-axis stack parameters; do not configure time parameters; file naming as sample_2_60x.ome.tif, sample_3_60x.ome.tif \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture multi-channel images (DAPI, FITC, TRITC) of all detected monkey kidney cell sections with 60× objective'}}]

        Original Instruction (corresponding to the overall task of this time):
        {original_instruction}

        Global Error Information of this Task:
        {global_error_info}
"""
