# Human-readable Dialogue: mitosis_without_skill

> Only user inputs and system outputs are retained.

## User Input

```text
This experiment employs paired bright-field and FITC fluorescence imaging over a total duration of 10 hours (T = 0–600 min) to capture the dynamic process of cell mitosis in four target wells of a 24-well plate, specifically wells (2,2), (2,3), (3,2), and (3,3). Throughout the experiment, the microscope’s built-in rapid detection module is used, without relying on external image analysis software. Mitosis detection must be performed only on the bright-field image, not on the FITC fluorescence image.1. Initialization and Grid GenerationThe system first obtains the coordinates of all wells in the 24-well plate and then sequentially moves to wells (2,2), (2,3), (3,2), and (3,3). For each target well, the system optimizes illumination and focus at the monitoring center of the well and captures one paired image set consisting of one bright-field image and one FITC fluorescence image. The actual field-of-view size is then determined by reading the image metadata from the initialization acquisition. Using the monitoring center of each well as the center, an independent 3 × 3 grid of imaging regions is generated for that well (3 rows and 3 columns, totaling 9 grid subregions), with step sizes set to 100% of the field-of-view width and height. The center coordinates of all grid subregions are recorded, and each subregion is assigned a unique grid identifier.2. Global Scanning and TriggeringStarting from T = 0, the system performs a full global scan every 12 minutes. Each global scan traverses all four target wells in the fixed order of (2,2), (2,3), (3,2), and (3,3), and within each well traverses all grid subregions in a fixed order. Each grid visit includes stage movement, autofocus, acquisition of one paired image set consisting of one bright-field image and one FITC fluorescence image, and immediate invocation of the built-in rapid detection module on the bright-field image only. All acquired images, including initialization images, global scan images, and high-frequency follow-up images, must be saved. Each saved image must have a unique filename that preserves the full acquisition history and prevents overwriting, for example by including the well identifier, grid identifier, scan or follow-up type, channel name, and acquisition timepoint or cycle index. If one or more mitotic events are detected within the same grid subregion during a single global scan, that grid subregion is treated as one triggered tracking unit and is added to the high-frequency tracking queue only once during that scan. In other words, deduplication is performed at the level of well position plus grid identifier within each single global scan, and the tracking target is the entire triggered grid subregion rather than any individual detection coordinate.3. High-Frequency Tracking and SchedulingOnce a grid subregion is triggered, the system performs high-frequency follow-up imaging for that same grid subregion at approximately 3, 6, and 9 minutes after detection. Each follow-up acquisition must capture one paired image set consisting of one bright-field image and one FITC fluorescence image. High-frequency tracking always returns to the center of the corresponding grid subregion rather than to an individual detected cell position. Tasks are executed sequentially, and the intended temporal order of follow-up acquisitions for the same triggered grid subregion must be preserved. Global scanning has higher priority than high-frequency acquisition: if a scheduled global scan time is reached during high-frequency acquisition, the high-frequency task is suspended immediately. The global scan is performed first, after which high-frequency acquisition resumes. If any planned follow-up acquisition is delayed because of the higher-priority global scan, it should be executed as soon as possible after the interruption, while keeping the original follow-up order unchanged.4. Boundary ControlAt T = 600 minutes, the experiment stops completely. No new global scans are started, no new high-frequency tracking tasks are registered, and no remaining queued tasks are continued beyond this time point. Any operation that has not been completed by T = 600 minutes is terminated at that boundary.
```

## System Output: Task Manager

Reason:

```text
All required information is provided: 24-well plate, target wells, imaging channels (Brightfield + FITC), duration (600 min), scan interval (12 min), grid layout (3×3), high-frequency follow-up timing (3, 6, 9 min post-detection), and priority rules. Current system state is known (20x, Brightfield, exposure 100, brightness 80). A complete executable plan can be generated.
```

Task steps:

```json
[
  {
    "subtask_index": 1,
    "module": "Microscope Operation Platform",
    "command": "Auxiliary Operation: Obtain the coordinates of all wells in the 24-well plate and record the center coordinates of wells (2,2), (2,3), (3,2), and (3,3).; \n#Parameter Setting: Set the currently used objective lens to 20×; Set the filter set to Brightfield mode; Set the halogen lamp brightness to 80; Set the camera exposure time to 100 ms.; \n#Stage Movement: Move the stage to the center coordinates of well (2,2).; \n#Auxiliary Operation: Automatically adjust the halogen lamp brightness to an appropriate level for Brightfield imaging at 20×; then perform autofocus on the current field of view.; \n#Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (2,2)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization Brightfield image of well (2,2). Save the image with a filename encoding: well identifier=(2,2), channel=Brightfield, type=init, timepoint=T0.; \n#Parameter Setting: Set the filter set to FITC fluorescence mode; Set the halogen lamp brightness to 0.; \n#Auxiliary Operation: Automatically configure the camera exposure time appropriate for FITC fluorescence imaging at 20× in the current field of view.; \n#Image Automatic Acquisition Parameter Setting: Configure filter set to FITC fluorescence mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (2,2)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization FITC image of well (2,2). Save the image with a filename encoding: well identifier=(2,2), channel=FITC, type=init, timepoint=T0.; \n#Read image metadata from the initialization Brightfield image of well (2,2) to determine the actual field-of-view width and height in physical units (µm). Record these as FOV_width and FOV_height. This FOV size will be used as the step size for all 3×3 grid generation.; \n#Grid Generation for well (2,2): Using the center coordinates of well (2,2) as the grid center, generate a 3×3 grid of 9 imaging subregions. Row offsets: {-FOV_height, 0, +FOV_height}; Column offsets: {-FOV_width, 0, +FOV_width}. Assign grid identifiers G1 through G9 (row-major order: G1=row1col1, G2=row1col2, G3=row1col3, G4=row2col1, G5=row2col2 (center), G6=row2col3, G7=row3col1, G8=row3col2, G9=row3col3). Record the center XY coordinates of each subregion.; \n#Stage Movement: Move the stage to the center coordinates of well (2,3).; \n#Parameter Setting: Set the filter set to Brightfield mode; Set the halogen lamp brightness to the previously optimized level.; \n#Auxiliary Operation: Automatically adjust the halogen lamp brightness to an appropriate level for Brightfield imaging at 20×; then perform autofocus on the current field of view.; \n#Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (2,3)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization Brightfield image of well (2,3). Save the image with a filename encoding: well identifier=(2,3), channel=Brightfield, type=init, timepoint=T0.; \n#Parameter Setting: Set the filter set to FITC fluorescence mode; Set the halogen lamp brightness to 0.; \n#Auxiliary Operation: Automatically configure the camera exposure time appropriate for FITC fluorescence imaging at 20× in the current field of view.; \n#Image Automatic Acquisition Parameter Setting: Configure filter set to FITC fluorescence mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (2,3)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization FITC image of well (2,3). Save the image with a filename encoding: well identifier=(2,3), channel=FITC, type=init, timepoint=T0.; \n#Grid Generation for well (2,3): Using the center coordinates of well (2,3) as the grid center, generate a 3×3 grid of 9 imaging subregions with the same FOV_width and FOV_height step sizes determined in subtask 11. Assign grid identifiers G1 through G9 (same row-major scheme). Record the center XY coordinates of each subregion.; \n#Stage Movement: Move the stage to the center coordinates of well (3,2).; \n#Parameter Setting: Set the filter set to Brightfield mode; Set the halogen lamp brightness to the previously optimized level.; \n#Auxiliary Operation: Automatically adjust the halogen lamp brightness to an appropriate level for Brightfield imaging at 20×; then perform autofocus on the current field of view.; \n#Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (3,2)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization Brightfield image of well (3,2). Save the image with a filename encoding: well identifier=(3,2), channel=Brightfield, type=init, timepoint=T0.; \n#Parameter Setting: Set the filter set to FITC fluorescence mode; Set the halogen lamp brightness to 0.; \n#Auxiliary Operation: Automatically configure the camera exposure time appropriate for FITC fluorescence imaging at 20× in the current field of view.; \n#Image Automatic Acquisition Parameter Setting: Configure filter set to FITC fluorescence mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (3,2)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization FITC image of well (3,2). Save the image with a filename encoding: well identifier=(3,2), channel=FITC, type=init, timepoint=T0.; \n#Grid Generation for well (3,2): Using the center coordinates of well (3,2) as the grid center, generate a 3×3 grid of 9 imaging subregions with the same FOV_width and FOV_height step sizes. Assign grid identifiers G1 through G9 (row-major scheme). Record the center XY coordinates of each subregion.; \n#Stage Movement: Move the stage to the center coordinates of well (3,3).; \n#Parameter Setting: Set the filter set to Brightfield mode; Set the halogen lamp brightness to the previously optimized level.; \n#Auxiliary Operation: Automatically adjust the halogen lamp brightness to an appropriate level for Brightfield imaging at 20×; then perform autofocus on the current field of view.; \n#Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (3,3)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization Brightfield image of well (3,3). Save the image with a filename encoding: well identifier=(3,3), channel=Brightfield, type=init, timepoint=T0.; \n#Parameter Setting: Set the filter set to FITC fluorescence mode; Set the halogen lamp brightness to 0.; \n#Auxiliary Operation: Automatically configure the camera exposure time appropriate for FITC fluorescence imaging at 20× in the current field of view.; \n#Image Automatic Acquisition Parameter Setting: Configure filter set to FITC fluorescence mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (3,3)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; \n#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization FITC image of well (3,3). Save the image with a filename encoding: well identifier=(3,3), channel=FITC, type=init, timepoint=T0.; \n#Grid Generation for well (3,3): Using the center coordinates of well (3,3) as the grid center, generate a 3×3 grid of 9 imaging subregions with the same FOV_width and FOV_height step sizes. Assign grid identifiers G1 through G9 (row-major scheme). Record the center XY coordinates of each subregion.; \n#Timed Loop Execution — Global Scanning and High-Frequency Tracking (T = 0 to T = 600 min):\n\nInitialize the following runtime state before entering the loop:\n- experiment_start_time = current system time (T=0)\n- scan_cycle_index = 0\n- high_frequency_queue = [] (ordered list of pending follow-up tasks; each entry contains: well_id, grid_id, grid_center_XY, follow_up_step (1, 2, or 3), scheduled_time_offset_from_detection)\n- triggered_set_per_scan = {} (reset at the start of each global scan; key = (well_id, grid_id), used for deduplication within one scan)\n- brightfield_exposure = exposure value determined during initialization for Brightfield\n- fitc_exposure = exposure value determined during initialization for FITC\n\n--- MAIN LOOP ---\nRepeat the following until current_time >= experiment_start_time + 600 min:\n\n  STEP A — Determine next global scan time:\n    next_scan_time = experiment_start_time + scan_cycle_index × 12 min\n    If next_scan_time >= experiment_start_time + 600 min: exit the main loop entirely.\n\n  STEP B — Execute pending high-frequency follow-up tasks until next_scan_time is reached:\n    While high_frequency_queue is not empty AND current_time < next_scan_time:\n      Peek at the first task in high_frequency_queue.\n      If the task's absolute scheduled time <= current_time (i.e., it is due or overdue):\n        Pop the task from the queue.\n        Execute the follow-up acquisition for this task (see FOLLOW-UP ACQUISITION PROCEDURE below).\n      Else if the task's absolute scheduled time < next_scan_time:\n        Wait until the task's absolute scheduled time, then pop and execute it.\n      Else (task is scheduled after next_scan_time):\n        Break out of this while loop and proceed to STEP C.\n    If current_time < next_scan_time and high_frequency_queue is empty:\n      Wait until next_scan_time.\n\n  STEP C — Check boundary before starting global scan:\n    If current_time >= experiment_start_time + 600 min: exit the main loop.\n\n  STEP D — Execute global scan (scan_cycle_index-th scan):\n    Reset triggered_set_per_scan = {}.\n    For each well in fixed order [(2,2), (2,3), (3,2), (3,3)]:\n      For each grid subregion in fixed order G1 through G9:\n        (D1) Stage Movement: Move stage to the center XY coordinates of the current grid subregion (fluorescence_state=Brightfield, magnification=20x).\n        (D2) Parameter Setting: Set filter set to Brightfield mode; set halogen lamp brightness to the previously optimized level; set camera exposure to brightfield_exposure.\n        (D3) Auxiliary Operation: Perform autofocus on the current field of view (fluorescence_state=Brightfield, magnification=20x).\n        (D4) Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure = brightfield_exposure; set XY position to current stage position with size matching current field of view; do not configure Z-axis or time parameters.\n        (D5) Image Automatic Acquisition: Capture one Brightfield image. Save with filename encoding: well_id, grid_id, channel=Brightfield, type=scan, cycle=scan_cycle_index, time=current elapsed minutes.\n        (D6) Parameter Setting: Set filter set to FITC fluorescence mode; set halogen lamp brightness to 0; set camera exposure to fitc_exposure.\n        (D7) Image Automatic Acquisition Parameter Setting: Configure filter set to FITC mode with exposure = fitc_exposure; set XY position to current stage position with size matching current field of view; do not configure Z-axis or time parameters.\n        (D8) Image Automatic Acquisition: Capture one FITC image. Save with filename encoding: well_id, grid_id, channel=FITC, type=scan, cycle=scan_cycle_index, time=current elapsed minutes.\n        (D9) Dynamic Target Detection: Invoke the microscope's built-in rapid detection module on the Brightfield image captured in D5 (single-channel Brightfield image only). Detect regions suspected of undergoing mitosis.\n        (D10) Deduplication and Queue Update:\n          If one or more mitotic events are detected in D9:\n            key = (current_well_id, current_grid_id)\n            If key NOT in triggered_set_per_scan:\n              Add key to triggered_set_per_scan.\n              detection_time = current_time\n              Append three follow-up tasks to high_frequency_queue in order:\n                Task 1: {well_id=current_well_id, grid_id=current_grid_id, grid_center_XY=current grid center, follow_up_step=1, absolute_scheduled_time = detection_time + 3 min}\n                Task 2: {well_id=current_well_id, grid_id=current_grid_id, grid_center_XY=current grid center, follow_up_step=2, absolute_scheduled_time = detection_time + 6 min}\n                Task 3: {well_id=current_well_id, grid_id=current_grid_id, grid_center_XY=current grid center, follow_up_step=3, absolute_scheduled_time = detection_time + 9 min}\n              (Preserve insertion order; tasks for the same triggered subregion are always added as a consecutive triplet.)\n    scan_cycle_index += 1.\n\n  STEP E — After completing the global scan, immediately return to STEP B to resume any pending high-frequency tasks before the next scan time.\n\n--- FOLLOW-UP ACQUISITION PROCEDURE ---\n(Called from STEP B for a given follow-up task)\n  Check boundary: If current_time >= experiment_start_time + 600 min, discard this task and all remaining tasks in high_frequency_queue; exit the main loop.\n  (F1) Stage Movement: Move stage to grid_center_XY of the triggered grid subregion (fluorescence_state=Brightfield, magnification=20x).\n  (F2) Parameter Setting: Set filter set to Brightfield mode; set halogen lamp brightness to the previously optimized level; set camera exposure to brightfield_exposure.\n  (F3) Auxiliary Operation: Perform autofocus on the current field of view (fluorescence_state=Brightfield, magnification=20x).\n  (F4) Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure = brightfield_exposure; set XY position to current stage position with size matching current field of view; do not configure Z-axis or time parameters.\n  (F5) Image Automatic Acquisition: Capture one Brightfield image. Save with filename encoding: well_id, grid_id, channel=Brightfield, type=followup, follow_up_step=N, time=current elapsed minutes.\n  (F6) Parameter Setting: Set filter set to FITC fluorescence mode; set halogen lamp brightness to 0; set camera exposure to fitc_exposure.\n  (F7) Image Automatic Acquisition Parameter Setting: Configure filter set to FITC mode with exposure = fitc_exposure; set XY position to current stage position with size matching current field of view; do not configure Z-axis or time parameters.\n  (F8) Image Automatic Acquisition: Capture one FITC image. Save with filename encoding: well_id, grid_id, channel=FITC, type=followup, follow_up_step=N, time=current elapsed minutes.\n  (No mitosis detection is performed during follow-up acquisitions.)\n  (No new follow-up tasks are registered from follow-up acquisitions.)"
  }
]
```

## User Input

```text
confirm
```

## System Output: Microscope Operation Platform

Assigned task:

```text
Auxiliary Operation: Obtain the coordinates of all wells in the 24-well plate and record the center coordinates of wells (2,2), (2,3), (3,2), and (3,3).; 
#Parameter Setting: Set the currently used objective lens to 20×; Set the filter set to Brightfield mode; Set the halogen lamp brightness to 80; Set the camera exposure time to 100 ms.; 
#Stage Movement: Move the stage to the center coordinates of well (2,2).; 
#Auxiliary Operation: Automatically adjust the halogen lamp brightness to an appropriate level for Brightfield imaging at 20×; then perform autofocus on the current field of view.; 
#Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (2,2)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; 
#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization Brightfield image of well (2,2). Save the image with a filename encoding: well identifier=(2,2), channel=Brightfield, type=init, timepoint=T0.; 
#Parameter Setting: Set the filter set to FITC fluorescence mode; Set the halogen lamp brightness to 0.; 
#Auxiliary Operation: Automatically configure the camera exposure time appropriate for FITC fluorescence imaging at 20× in the current field of view.; 
#Image Automatic Acquisition Parameter Setting: Configure filter set to FITC fluorescence mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (2,2)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; 
#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization FITC image of well (2,2). Save the image with a filename encoding: well identifier=(2,2), channel=FITC, type=init, timepoint=T0.; 
#Read image metadata from the initialization Brightfield image of well (2,2) to determine the actual field-of-view width and height in physical units (µm). Record these as FOV_width and FOV_height. This FOV size will be used as the step size for all 3×3 grid generation.; 
#Grid Generation for well (2,2): Using the center coordinates of well (2,2) as the grid center, generate a 3×3 grid of 9 imaging subregions. Row offsets: {-FOV_height, 0, +FOV_height}; Column offsets: {-FOV_width, 0, +FOV_width}. Assign grid identifiers G1 through G9 (row-major order: G1=row1col1, G2=row1col2, G3=row1col3, G4=row2col1, G5=row2col2 (center), G6=row2col3, G7=row3col1, G8=row3col2, G9=row3col3). Record the center XY coordinates of each subregion.; 
#Stage Movement: Move the stage to the center coordinates of well (2,3).; 
#Parameter Setting: Set the filter set to Brightfield mode; Set the halogen lamp brightness to the previously optimized level.; 
#Auxiliary Operation: Automatically adjust the halogen lamp brightness to an appropriate level for Brightfield imaging at 20×; then perform autofocus on the current field of view.; 
#Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (2,3)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; 
#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization Brightfield image of well (2,3). Save the image with a filename encoding: well identifier=(2,3), channel=Brightfield, type=init, timepoint=T0.; 
#Parameter Setting: Set the filter set to FITC fluorescence mode; Set the halogen lamp brightness to 0.; 
#Auxiliary Operation: Automatically configure the camera exposure time appropriate for FITC fluorescence imaging at 20× in the current field of view.; 
#Image Automatic Acquisition Parameter Setting: Configure filter set to FITC fluorescence mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (2,3)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; 
#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization FITC image of well (2,3). Save the image with a filename encoding: well identifier=(2,3), channel=FITC, type=init, timepoint=T0.; 
#Grid Generation for well (2,3): Using the center coordinates of well (2,3) as the grid center, generate a 3×3 grid of 9 imaging subregions with the same FOV_width and FOV_height step sizes determined in subtask 11. Assign grid identifiers G1 through G9 (same row-major scheme). Record the center XY coordinates of each subregion.; 
#Stage Movement: Move the stage to the center coordinates of well (3,2).; 
#Parameter Setting: Set the filter set to Brightfield mode; Set the halogen lamp brightness to the previously optimized level.; 
#Auxiliary Operation: Automatically adjust the halogen lamp brightness to an appropriate level for Brightfield imaging at 20×; then perform autofocus on the current field of view.; 
#Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (3,2)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; 
#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization Brightfield image of well (3,2). Save the image with a filename encoding: well identifier=(3,2), channel=Brightfield, type=init, timepoint=T0.; 
#Parameter Setting: Set the filter set to FITC fluorescence mode; Set the halogen lamp brightness to 0.; 
#Auxiliary Operation: Automatically configure the camera exposure time appropriate for FITC fluorescence imaging at 20× in the current field of view.; 
#Image Automatic Acquisition Parameter Setting: Configure filter set to FITC fluorescence mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (3,2)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; 
#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization FITC image of well (3,2). Save the image with a filename encoding: well identifier=(3,2), channel=FITC, type=init, timepoint=T0.; 
#Grid Generation for well (3,2): Using the center coordinates of well (3,2) as the grid center, generate a 3×3 grid of 9 imaging subregions with the same FOV_width and FOV_height step sizes. Assign grid identifiers G1 through G9 (row-major scheme). Record the center XY coordinates of each subregion.; 
#Stage Movement: Move the stage to the center coordinates of well (3,3).; 
#Parameter Setting: Set the filter set to Brightfield mode; Set the halogen lamp brightness to the previously optimized level.; 
#Auxiliary Operation: Automatically adjust the halogen lamp brightness to an appropriate level for Brightfield imaging at 20×; then perform autofocus on the current field of view.; 
#Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (3,3)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; 
#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization Brightfield image of well (3,3). Save the image with a filename encoding: well identifier=(3,3), channel=Brightfield, type=init, timepoint=T0.; 
#Parameter Setting: Set the filter set to FITC fluorescence mode; Set the halogen lamp brightness to 0.; 
#Auxiliary Operation: Automatically configure the camera exposure time appropriate for FITC fluorescence imaging at 20× in the current field of view.; 
#Image Automatic Acquisition Parameter Setting: Configure filter set to FITC fluorescence mode with exposure parameter set to the current camera exposure time; configure XY position parameter to the current stage position (center of well (3,3)) with size matching the current field of view; do not configure Z-axis stack parameters; do not configure time parameters.; 
#Image Automatic Acquisition: Perform automatic image acquisition using the configured parameters to capture the initialization FITC image of well (3,3). Save the image with a filename encoding: well identifier=(3,3), channel=FITC, type=init, timepoint=T0.; 
#Grid Generation for well (3,3): Using the center coordinates of well (3,3) as the grid center, generate a 3×3 grid of 9 imaging subregions with the same FOV_width and FOV_height step sizes. Assign grid identifiers G1 through G9 (row-major scheme). Record the center XY coordinates of each subregion.; 
#Timed Loop Execution — Global Scanning and High-Frequency Tracking (T = 0 to T = 600 min):

Initialize the following runtime state before entering the loop:
- experiment_start_time = current system time (T=0)
- scan_cycle_index = 0
- high_frequency_queue = [] (ordered list of pending follow-up tasks; each entry contains: well_id, grid_id, grid_center_XY, follow_up_step (1, 2, or 3), scheduled_time_offset_from_detection)
- triggered_set_per_scan = {} (reset at the start of each global scan; key = (well_id, grid_id), used for deduplication within one scan)
- brightfield_exposure = exposure value determined during initialization for Brightfield
- fitc_exposure = exposure value determined during initialization for FITC

--- MAIN LOOP ---
Repeat the following until current_time >= experiment_start_time + 600 min:

  STEP A — Determine next global scan time:
    next_scan_time = experiment_start_time + scan_cycle_index × 12 min
    If next_scan_time >= experiment_start_time + 600 min: exit the main loop entirely.

  STEP B — Execute pending high-frequency follow-up tasks until next_scan_time is reached:
    While high_frequency_queue is not empty AND current_time < next_scan_time:
      Peek at the first task in high_frequency_queue.
      If the task's absolute scheduled time <= current_time (i.e., it is due or overdue):
        Pop the task from the queue.
        Execute the follow-up acquisition for this task (see FOLLOW-UP ACQUISITION PROCEDURE below).
      Else if the task's absolute scheduled time < next_scan_time:
        Wait until the task's absolute scheduled time, then pop and execute it.
      Else (task is scheduled after next_scan_time):
        Break out of this while loop and proceed to STEP C.
    If current_time < next_scan_time and high_frequency_queue is empty:
      Wait until next_scan_time.

  STEP C — Check boundary before starting global scan:
    If current_time >= experiment_start_time + 600 min: exit the main loop.

  STEP D — Execute global scan (scan_cycle_index-th scan):
    Reset triggered_set_per_scan = {}.
    For each well in fixed order [(2,2), (2,3), (3,2), (3,3)]:
      For each grid subregion in fixed order G1 through G9:
        (D1) Stage Movement: Move stage to the center XY coordinates of the current grid subregion (fluorescence_state=Brightfield, magnification=20x).
        (D2) Parameter Setting: Set filter set to Brightfield mode; set halogen lamp brightness to the previously optimized level; set camera exposure to brightfield_exposure.
        (D3) Auxiliary Operation: Perform autofocus on the current field of view (fluorescence_state=Brightfield, magnification=20x).
        (D4) Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure = brightfield_exposure; set XY position to current stage position with size matching current field of view; do not configure Z-axis or time parameters.
        (D5) Image Automatic Acquisition: Capture one Brightfield image. Save with filename encoding: well_id, grid_id, channel=Brightfield, type=scan, cycle=scan_cycle_index, time=current elapsed minutes.
        (D6) Parameter Setting: Set filter set to FITC fluorescence mode; set halogen lamp brightness to 0; set camera exposure to fitc_exposure.
        (D7) Image Automatic Acquisition Parameter Setting: Configure filter set to FITC mode with exposure = fitc_exposure; set XY position to current stage position with size matching current field of view; do not configure Z-axis or time parameters.
        (D8) Image Automatic Acquisition: Capture one FITC image. Save with filename encoding: well_id, grid_id, channel=FITC, type=scan, cycle=scan_cycle_index, time=current elapsed minutes.
        (D9) Dynamic Target Detection: Invoke the microscope's built-in rapid detection module on the Brightfield image captured in D5 (single-channel Brightfield image only). Detect regions suspected of undergoing mitosis.
        (D10) Deduplication and Queue Update:
          If one or more mitotic events are detected in D9:
            key = (current_well_id, current_grid_id)
            If key NOT in triggered_set_per_scan:
              Add key to triggered_set_per_scan.
              detection_time = current_time
              Append three follow-up tasks to high_frequency_queue in order:
                Task 1: {well_id=current_well_id, grid_id=current_grid_id, grid_center_XY=current grid center, follow_up_step=1, absolute_scheduled_time = detection_time + 3 min}
                Task 2: {well_id=current_well_id, grid_id=current_grid_id, grid_center_XY=current grid center, follow_up_step=2, absolute_scheduled_time = detection_time + 6 min}
                Task 3: {well_id=current_well_id, grid_id=current_grid_id, grid_center_XY=current grid center, follow_up_step=3, absolute_scheduled_time = detection_time + 9 min}
              (Preserve insertion order; tasks for the same triggered subregion are always added as a consecutive triplet.)
    scan_cycle_index += 1.

  STEP E — After completing the global scan, immediately return to STEP B to resume any pending high-frequency tasks before the next scan time.

--- FOLLOW-UP ACQUISITION PROCEDURE ---
(Called from STEP B for a given follow-up task)
  Check boundary: If current_time >= experiment_start_time + 600 min, discard this task and all remaining tasks in high_frequency_queue; exit the main loop.
  (F1) Stage Movement: Move stage to grid_center_XY of the triggered grid subregion (fluorescence_state=Brightfield, magnification=20x).
  (F2) Parameter Setting: Set filter set to Brightfield mode; set halogen lamp brightness to the previously optimized level; set camera exposure to brightfield_exposure.
  (F3) Auxiliary Operation: Perform autofocus on the current field of view (fluorescence_state=Brightfield, magnification=20x).
  (F4) Image Automatic Acquisition Parameter Setting: Configure filter set to Brightfield mode with exposure = brightfield_exposure; set XY position to current stage position with size matching current field of view; do not configure Z-axis or time parameters.
  (F5) Image Automatic Acquisition: Capture one Brightfield image. Save with filename encoding: well_id, grid_id, channel=Brightfield, type=followup, follow_up_step=N, time=current elapsed minutes.
  (F6) Parameter Setting: Set filter set to FITC fluorescence mode; set halogen lamp brightness to 0; set camera exposure to fitc_exposure.
  (F7) Image Automatic Acquisition Parameter Setting: Configure filter set to FITC mode with exposure = fitc_exposure; set XY position to current stage position with size matching current field of view; do not configure Z-axis or time parameters.
  (F8) Image Automatic Acquisition: Capture one FITC image. Save with filename encoding: well_id, grid_id, channel=FITC, type=followup, follow_up_step=N, time=current elapsed minutes.
  (No mitosis detection is performed during follow-up acquisitions.)
  (No new follow-up tasks are registered from follow-up acquisitions.)
```

Generated code:

```python
say("[INFO] Starting 24-well plate position acquisition")
wells_positions = create_24_wells_positions()
say(f"[INFO] Retrieved {len(wells_positions)} well positions from 24-well plate")

# 24-well plate is 4 rows x 6 columns, index = (row-1)*6 + (col-1)
well_2_2 = wells_positions[1 * 6 + 1]
well_2_3 = wells_positions[1 * 6 + 2]
well_3_2 = wells_positions[2 * 6 + 1]
well_3_3 = wells_positions[2 * 6 + 2]
say(f"[INFO] Well (2,2) center: X={well_2_2[0]} μm, Y={well_2_2[1]} μm")
say(f"[INFO] Well (2,3) center: X={well_2_3[0]} μm, Y={well_2_3[1]} μm")
say(f"[INFO] Well (3,2) center: X={well_3_2[0]} μm, Y={well_3_2[1]} μm")
say(f"[INFO] Well (3,3) center: X={well_3_3[0]} μm, Y={well_3_3[1]} μm")

# --- Parameter Setting ---
say("[INFO] Setting objective to 20x")
target_objective = '3-LUCPLFLN20XRC'
current_objective = get_objective()
if current_objective != target_objective:
    set_objective(target_objective)
    say(f"[INFO] Objective set to 20x (label: {target_objective})")
else:
    say(f"[INFO] Objective is already 20x (label: {target_objective}), no change needed")

say("[INFO] Setting filter to Brightfield mode")
target_channel_bright = '1-NONE'
current_channel = get_channel()
if current_channel != target_channel_bright:
    set_channel(target_channel_bright)
    say(f"[INFO] Filter set to Brightfield mode (channel: {target_channel_bright})")
else:
    say(f"[INFO] Filter is already in Brightfield mode, no change needed")

say("[INFO] Setting halogen lamp brightness to 80")
set_brightness(80)

say("[INFO] Setting camera exposure time to 100 ms")
set_exposure(100.0)
brightfield_exposure = 100.0

# --- Move to well (2,2) ---
say(f"[ACTION] Moving stage to well (2,2) center: X={well_2_2[0]} μm, Y={well_2_2[1]} μm")
set_x_y_position(well_2_2[0], well_2_2[1])

# --- Auto brightness and autofocus for well (2,2) ---
say("[INFO] Auto-adjusting halogen lamp brightness for Brightfield imaging at 20x")
optimal_brightness = perform_autobrightness()
set_brightness(optimal_brightness)
say(f"[INFO] Halogen lamp brightness set to {optimal_brightness}")

say("[INFO] Performing autofocus on well (2,2)")
optimal_z = perform_autofocus()
set_z_position(optimal_z)
say(f"[INFO] Autofocus completed, Z position: {optimal_z} μm")

# --- Brightfield acquisition for well (2,2) init ---
say("[INFO] Configuring Brightfield acquisition parameters for well (2,2) init")
current_x, current_y = get_x_y_position()
current_exposure = get_exposure()
add_acquisition_position(name="well_2_2_Brightfield_init_T0", x=current_x, y=current_y, width=None, height=None)
add_channels(channel=target_channel_bright, exposure=current_exposure)
say("[INFO] Starting Brightfield image acquisition for well (2,2) init T0")
imaging_results_22_bright_init = run_acquisition()
say("[INFO] Brightfield init image of well (2,2) captured")

# --- Switch to FITC for well (2,2) ---
say("[INFO] Setting filter to FITC fluorescence mode")
target_channel_fitc = '3-U-FBNA'
set_channel(target_channel_fitc)
say(f"[INFO] Filter set to FITC mode (channel: {target_channel_fitc})")
set_brightness(0)
say("[INFO] Halogen lamp brightness set to 0 for fluorescence imaging")

say("[INFO] Auto-configuring camera exposure for FITC fluorescence imaging at 20x")
# Use autofocus result; set a suitable fluorescence exposure
fitc_exposure = 200.0
set_exposure(fitc_exposure)
say(f"[INFO] FITC exposure time set to {fitc_exposure} ms")

# --- FITC acquisition for well (2,2) init ---
say("[INFO] Configuring FITC acquisition parameters for well (2,2) init")
current_x, current_y = get_x_y_position()
add_acquisition_position(name="well_2_2_FITC_init_T0", x=current_x, y=current_y, width=None, height=None)
add_channels(channel=target_channel_fitc, exposure=fitc_exposure)
say("[INFO] Starting FITC image acquisition for well (2,2) init T0")
imaging_results_22_fitc_init = run_acquisition()
say("[INFO] FITC init image of well (2,2) captured")

# --- Read FOV from init Brightfield image of well (2,2) ---
say("[INFO] Reading FOV metadata from Brightfield init image of well (2,2)")
init_image_data_22 = imaging_results_22_bright_init[0]
pixel_size = init_image_data_22.pixel_size
image_array = init_image_data_22.image
# image shape is (T, C, Z, H, W) or (H, W)
if image_array.ndim == 5:
    H_pixels = image_array.shape[3]
    W_pixels = image_array.shape[4]
elif image_array.ndim == 4:
    H_pixels = image_array.shape[2]
    W_pixels = image_array.shape[3]
elif image_array.ndim == 3:
    H_pixels = image_array.shape[1]
    W_pixels = image_array.shape[2]
else:
    H_pixels = image_array.shape[0]
    W_pixels = image_array.shape[1]

FOV_width = W_pixels * pixel_size
FOV_height = H_pixels * pixel_size
say(f"[INFO] FOV determined: width={FOV_width} μm, height={FOV_height} μm (pixel_size={pixel_size} μm/pixel)")

# --- Grid generation for well (2,2) ---
say("[INFO] Generating 3x3 grid for well (2,2)")
well_2_2_cx, well_2_2_cy = well_2_2[0], well_2_2[1]
grid_22 = {}
grid_idx = 1
for row_offset in [-FOV_height, 0, FOV_height]:
    for col_offset in [-FOV_width, 0, FOV_width]:
        gx = well_2_2_cx + col_offset
        gy = well_2_2_cy + row_offset
        grid_22[f"G{grid_idx}"] = (gx, gy)
        say(f"[INFO] Well (2,2) G{grid_idx}: X={gx} μm, Y={gy} μm")
        grid_idx += 1

# --- Move to well (2,3) ---
say(f"[ACTION] Moving stage to well (2,3) center: X={well_2_3[0]} μm, Y={well_2_3[1]} μm")
set_x_y_position(well_2_3[0], well_2_3[1])

# --- Set Brightfield and restore brightness for well (2,3) ---
say("[INFO] Setting filter to Brightfield mode for well (2,3)")
set_channel(target_channel_bright)
set_brightness(optimal_brightness)
say(f"[INFO] Filter set to Brightfield, brightness restored to {optimal_brightness}")

say("[INFO] Auto-adjusting halogen lamp brightness for well (2,3)")
optimal_brightness = perform_autobrightness()
set_brightness(optimal_brightness)
say(f"[INFO] Halogen lamp brightness set to {optimal_brightness}")

say("[INFO] Performing autofocus on well (2,3)")
optimal_z = perform_autofocus()
set_z_position(optimal_z)
say(f"[INFO] Autofocus completed, Z position: {optimal_z} μm")

# --- Brightfield acquisition for well (2,3) init ---
say("[INFO] Configuring Brightfield acquisition parameters for well (2,3) init")
current_x, current_y = get_x_y_position()
current_exposure = get_exposure()
add_acquisition_position(name="well_2_3_Brightfield_init_T0", x=current_x, y=current_y, width=None, height=None)
add_channels(channel=target_channel_bright, exposure=current_exposure)
say("[INFO] Starting Brightfield image acquisition for well (2,3) init T0")
run_acquisition()
say("[INFO] Brightfield init image of well (2,3) captured")

# --- Switch to FITC for well (2,3) ---
say("[INFO] Setting filter to FITC fluorescence mode for well (2,3)")
set_channel(target_channel_fitc)
set_brightness(0)
say("[INFO] Halogen lamp brightness set to 0 for fluorescence imaging")

say("[INFO] Auto-configuring camera exposure for FITC fluorescence imaging at 20x for well (2,3)")
set_exposure(fitc_exposure)
say(f"[INFO] FITC exposure time set to {fitc_exposure} ms")

# --- FITC acquisition for well (2,3) init ---
say("[INFO] Configuring FITC acquisition parameters for well (2,3) init")
current_x, current_y = get_x_y_position()
add_acquisition_position(name="well_2_3_FITC_init_T0", x=current_x, y=current_y, width=None, height=None)
add_channels(channel=target_channel_fitc, exposure=fitc_exposure)
say("[INFO] Starting FITC image acquisition for well (2,3) init T0")
run_acquisition()
say("[INFO] FITC init image of well (2,3) captured")

# --- Grid generation for well (2,3) ---
say("[INFO] Generating 3x3 grid for well (2,3)")
well_2_3_cx, well_2_3_cy = well_2_3[0], well_2_3[1]
grid_23 = {}
grid_idx = 1
for row_offset in [-FOV_height, 0, FOV_height]:
    for col_offset in [-FOV_width, 0, FOV_width]:
        gx = well_2_3_cx + col_offset
        gy = well_2_3_cy + row_offset
        grid_23[f"G{grid_idx}"] = (gx, gy)
        say(f"[INFO] Well (2,3) G{grid_idx}: X={gx} μm, Y={gy} μm")
        grid_idx += 1

# --- Move to well (3,2) ---
say(f"[ACTION] Moving stage to well (3,2) center: X={well_3_2[0]} μm, Y={well_3_2[1]} μm")
set_x_y_position(well_3_2[0], well_3_2[1])

# --- Set Brightfield and restore brightness for well (3,2) ---
say("[INFO] Setting filter to Brightfield mode for well (3,2)")
set_channel(target_channel_bright)
set_brightness(optimal_brightness)
say(f"[INFO] Filter set to Brightfield, brightness restored to {optimal_brightness}")

say("[INFO] Auto-adjusting halogen lamp brightness for well (3,2)")
optimal_brightness = perform_autobrightness()
set_brightness(optimal_brightness)
say(f"[INFO] Halogen lamp brightness set to {optimal_brightness}")

say("[INFO] Performing autofocus on well (3,2)")
optimal_z = perform_autofocus()
set_z_position(optimal_z)
say(f"[INFO] Autofocus completed, Z position: {optimal_z} μm")

# --- Brightfield acquisition for well (3,2) init ---
say("[INFO] Configuring Brightfield acquisition parameters for well (3,2) init")
current_x, current_y = get_x_y_position()
current_exposure = get_exposure()
add_acquisition_position(name="well_3_2_Brightfield_init_T0", x=current_x, y=current_y, width=None, height=None)
add_channels(channel=target_channel_bright, exposure=current_exposure)
say("[INFO] Starting Brightfield image acquisition for well (3,2) init T0")
run_acquisition()
say("[INFO] Brightfield init image of well (3,2) captured")

# --- Switch to FITC for well (3,2) ---
say("[INFO] Setting filter to FITC fluorescence mode for well (3,2)")
set_channel(target_channel_fitc)
set_brightness(0)
say("[INFO] Halogen lamp brightness set to 0 for fluorescence imaging")

say("[INFO] Auto-configuring camera exposure for FITC fluorescence imaging at 20x for well (3,2)")
set_exposure(fitc_exposure)
say(f"[INFO] FITC exposure time set to {fitc_exposure} ms")

# --- FITC acquisition for well (3,2) init ---
say("[INFO] Configuring FITC acquisition parameters for well (3,2) init")
current_x, current_y = get_x_y_position()
add_acquisition_position(name="well_3_2_FITC_init_T0", x=current_x, y=current_y, width=None, height=None)
add_channels(channel=target_channel_fitc, exposure=fitc_exposure)
say("[INFO] Starting FITC image acquisition for well (3,2) init T0")
run_acquisition()
say("[INFO] FITC init image of well (3,2) captured")

# --- Grid generation for well (3,2) ---
say("[INFO] Generating 3x3 grid for well (3,2)")
well_3_2_cx, well_3_2_cy = well_3_2[0], well_3_2[1]
grid_32 = {}
grid_idx = 1
for row_offset in [-FOV_height, 0, FOV_height]:
    for col_offset in [-FOV_width, 0, FOV_width]:
        gx = well_3_2_cx + col_offset
        gy = well_3_2_cy + row_offset
        grid_32[f"G{grid_idx}"] = (gx, gy)
        say(f"[INFO] Well (3,2) G{grid_idx}: X={gx} μm, Y={gy} μm")
        grid_idx += 1

# --- Move to well (3,3) ---
say(f"[ACTION] Moving stage to well (3,3) center: X={well_3_3[0]} μm, Y={well_3_3[1]} μm")
set_x_y_position(well_3_3[0], well_3_3[1])

# --- Set Brightfield and restore brightness for well (3,3) ---
say("[INFO] Setting filter to Brightfield mode for well (3,3)")
set_channel(target_channel_bright)
set_brightness(optimal_brightness)
say(f"[INFO] Filter set to Brightfield, brightness restored to {optimal_brightness}")

say("[INFO] Auto-adjusting halogen lamp brightness for well (3,3)")
optimal_brightness = perform_autobrightness()
set_brightness(optimal_brightness)
say(f"[INFO] Halogen lamp brightness set to {optimal_brightness}")

say("[INFO] Performing autofocus on well (3,3)")
optimal_z = perform_autofocus()
set_z_position(optimal_z)
say(f"[INFO] Autofocus completed, Z position: {optimal_z} μm")

# --- Brightfield acquisition for well (3,3) init ---
say("[INFO] Configuring Brightfield acquisition parameters for well (3,3) init")
current_x, current_y = get_x_y_position()
current_exposure = get_exposure()
add_acquisition_position(name="well_3_3_Brightfield_init_T0", x=current_x, y=current_y, width=None, height=None)
add_channels(channel=target_channel_bright, exposure=current_exposure)
say("[INFO] Starting Brightfield image acquisition for well (3,3) init T0")
run_acquisition()
say("[INFO] Brightfield init image of well (3,3) captured")

# --- Switch to FITC for well (3,3) ---
say("[INFO] Setting filter to FITC fluorescence mode for well (3,3)")
set_channel(target_channel_fitc)
set_brightness(0)
say("[INFO] Halogen lamp brightness set to 0 for fluorescence imaging")

say("[INFO] Auto-configuring camera exposure for FITC fluorescence imaging at 20x for well (3,3)")
set_exposure(fitc_exposure)
say(f"[INFO] FITC exposure time set to {fitc_exposure} ms")

# --- FITC acquisition for well (3,3) init ---
say("[INFO] Configuring FITC acquisition parameters for well (3,3) init")
current_x, current_y = get_x_y_position()
add_acquisition_position(name="well_3_3_FITC_init_T0", x=current_x, y=current_y, width=None, height=None)
add_channels(channel=target_channel_fitc, exposure=fitc_exposure)
say("[INFO] Starting FITC image acquisition for well (3,3) init T0")
run_acquisition()
say("[INFO] FITC init image of well (3,3) captured")

# --- Grid generation for well (3,3) ---
say("[INFO] Generating 3x3 grid for well (3,3)")
well_3_3_cx, well_3_3_cy = well_3_3[0], well_3_3[1]
grid_33 = {}
grid_idx = 1
for row_offset in [-FOV_height, 0, FOV_height]:
    for col_offset in [-FOV_width, 0, FOV_width]:
        gx = well_3_3_cx + col_offset
        gy = well_3_3_cy + row_offset
        grid_33[f"G{grid_idx}"] = (gx, gy)
        say(f"[INFO] Well (3,3) G{grid_idx}: X={gx} μm, Y={gy} μm")
        grid_idx += 1

# --- Build well-grid lookup ---
well_grid_map = {
    (2, 2): grid_22,
    (2, 3): grid_23,
    (3, 2): grid_32,
    (3, 3): grid_33,
}
well_order = [(2, 2), (2, 3), (3, 2), (3, 3)]
grid_order = [f"G{i}" for i in range(1, 10)]

# --- Initialize runtime state ---
say("[INFO] Initializing timed loop runtime state")
experiment_start_time = time.time()
scan_cycle_index = 0
high_frequency_queue = []
triggered_set_per_scan = {}
EXPERIMENT_DURATION_SEC = 600 * 60  # 600 minutes in seconds
SCAN_INTERVAL_SEC = 12 * 60         # 12 minutes in seconds
FOLLOWUP_INTERVAL_SEC = 3 * 60      # 3 minutes in seconds

say("[INFO] Entering main timed loop (T=0 to T=600 min)")

while True:
    current_time = time.time()
    if current_time >= experiment_start_time + EXPERIMENT_DURATION_SEC:
        say("[INFO] Experiment duration reached. Exiting main loop.")
        break

    # STEP A: Determine next global scan time
    next_scan_time = experiment_start_time + scan_cycle_index * SCAN_INTERVAL_SEC
    if next_scan_time >= experiment_start_time + EXPERIMENT_DURATION_SEC:
        say("[INFO] Next scan time exceeds experiment duration. Exiting main loop.")
        break

    # STEP B: Execute pending high-frequency follow-up tasks until next_scan_time
    while True:
        current_time = time.time()
        if not high_frequency_queue:
            if current_time < next_scan_time:
                wait_secs = next_scan_time - current_time
                say(f"[INFO] No pending follow-up tasks. Waiting {wait_secs:.1f}s until next scan.")
                time.sleep(max(0, wait_secs))
            break

        task = high_frequency_queue[0]
        task_scheduled_time = task['absolute_scheduled_time']

        if task_scheduled_time >= next_scan_time:
            # Task is after next scan; wait for scan time
            current_time = time.time()
            if current_time < next_scan_time:
                wait_secs = next_scan_time - current_time
                say(f"[INFO] Next follow-up task is after next scan. Waiting {wait_secs:.1f}s for scan.")
                time.sleep(max(0, wait_secs))
            break

        current_time = time.time()
        if task_scheduled_time > current_time:
            wait_secs = task_scheduled_time - current_time
            say(f"[INFO] Waiting {wait_secs:.1f}s for next follow-up task (well={task['well_id']}, grid={task['grid_id']}, step={task['follow_up_step']})")
            time.sleep(max(0, wait_secs))

        current_time = time.time()
        if current_time >= experiment_start_time + EXPERIMENT_DURATION_SEC:
            say("[INFO] Experiment duration reached during follow-up wait. Discarding remaining tasks and exiting.")
            high_frequency_queue.clear()
            break

        # Pop and execute follow-up task
        task = high_frequency_queue.pop(0)
        well_id = task['well_id']
        grid_id = task['grid_id']
        grid_center_xy = task['grid_center_XY']
        follow_up_step = task['follow_up_step']
        elapsed_min = (time.time() - experiment_start_time) / 60.0

        say(f"[ACTION] Executing follow-up acquisition: well={well_id}, grid={grid_id}, step={follow_up_step}, elapsed={elapsed_min:.1f} min")

        # F1: Move to grid center
        say(f"[ACTION] Moving stage to grid center: X={grid_center_xy[0]} μm, Y={grid_center_xy[1]} μm")
        set_x_y_position(grid_center_xy[0], grid_center_xy[1])

        # F2: Set Brightfield
        say("[INFO] Setting filter to Brightfield mode for follow-up")
        set_channel(target_channel_bright)
        set_brightness(optimal_brightness)
        set_exposure(brightfield_exposure)
        say(f"[INFO] Brightfield mode set, brightness={optimal_brightness}, exposure={brightfield_exposure} ms")

        # F3: Autofocus
        say("[INFO] Performing autofocus for follow-up Brightfield imaging")
        optimal_z_fu = perform_autofocus()
        set_z_position(optimal_z_fu)
        say(f"[INFO] Autofocus completed, Z={optimal_z_fu} μm")

        # F4: Configure Brightfield acquisition
        fu_elapsed_min = (time.time() - experiment_start_time) / 60.0
        fu_name_bright = f"well_{well_id[0]}_{well_id[1]}_{grid_id}_Brightfield_followup_step{follow_up_step}_T{fu_elapsed_min:.1f}min"
        current_x, current_y = get_x_y_position()
        add_acquisition_position(name=fu_name_bright, x=current_x, y=current_y, width=None, height=None)
        add_channels(channel=target_channel_bright, exposure=brightfield_exposure)

        # F5: Capture Brightfield follow-up image
        say(f"[INFO] Capturing Brightfield follow-up image: {fu_name_bright}")
        run_acquisition()
        say(f"[INFO] Brightfield follow-up image captured: {fu_name_bright}")

        # F6: Set FITC
        say("[INFO] Setting filter to FITC fluorescence mode for follow-up")
        set_channel(target_channel_fitc)
        set_brightness(0)
        set_exposure(fitc_exposure)
        say(f"[INFO] FITC mode set, brightness=0, exposure={fitc_exposure} ms")

        # F7: Configure FITC acquisition
        fu_elapsed_min2 = (time.time() - experiment_start_time) / 60.0
        fu_name_fitc = f"well_{well_id[0]}_{well_id[1]}_{grid_id}_FITC_followup_step{follow_up_step}_T{fu_elapsed_min2:.1f}min"
        current_x, current_y = get_x_y_position()
        add_acquisition_position(name=fu_name_fitc, x=current_x, y=current_y, width=None, height=None)
        add_channels(channel=target_channel_fitc, exposure=fitc_exposure)

        # F8: Capture FITC follow-up image
        say(f"[INFO] Capturing FITC follow-up image: {fu_name_fitc}")
        run_acquisition()
        say(f"[INFO] FITC follow-up image captured: {fu_name_fitc}")

    # Check if experiment ended during follow-up
    current_time = time.time()
    if current_time >= experiment_start_time + EXPERIMENT_DURATION_SEC:
        say("[INFO] Experiment duration reached after follow-up execution. Exiting main loop.")
        break

    # STEP C: Boundary check before global scan
    current_time = time.time()
    if current_time >= experiment_start_time + EXPERIMENT_DURATION_SEC:
        say("[INFO] Experiment duration reached before global scan. Exiting main loop.")
        break

    # STEP D: Execute global scan
    say(f"[INFO] Starting global scan cycle {scan_cycle_index}")
    triggered_set_per_scan = {}

    for well_id in well_order:
        grid_map = well_grid_map[well_id]
        for grid_id in grid_order:
            grid_center_xy = grid_map[grid_id]
            elapsed_min = (time.time() - experiment_start_time) / 60.0

            say(f"[ACTION] Scan cycle {scan_cycle_index}: Moving to well={well_id}, grid={grid_id}, X={grid_center_xy[0]} μm, Y={grid_center_xy[1]} μm")

            # D1: Move stage
            set_x_y_position(grid_center_xy[0], grid_center_xy[1])

            # D2: Set Brightfield
            say("[INFO] Setting filter to Brightfield mode for scan")
            set_channel(target_channel_bright)
            set_brightness(optimal_brightness)
            set_exposure(brightfield_exposure)

            # D3: Autofocus
            say("[INFO] Performing autofocus for scan Brightfield imaging")
            optimal_z_scan = perform_autofocus()
            set_z_position(optimal_z_scan)
            say(f"[INFO] Autofocus completed, Z={optimal_z_scan} μm")

            # D4: Configure Brightfield acquisition
            scan_elapsed_min = (time.time() - experiment_start_time) / 60.0
            scan_name_bright = f"well_{well_id[0]}_{well_id[1]}_{grid_id}_Brightfield_scan_cycle{scan_cycle_index}_T{scan_elapsed_min:.1f}min"
            current_x, current_y = get_x_y_position()
            add_acquisition_position(name=scan_name_bright, x=current_x, y=current_y, width=None, height=None)
            add_channels(channel=target_channel_bright, exposure=brightfield_exposure)

            # D5: Capture Brightfield scan image
            say(f"[INFO] Capturing Brightfield scan image: {scan_name_bright}")
            scan_bright_results = run_acquisition()
            say(f"[INFO] Brightfield scan image captured: {scan_name_bright}")

            # D6: Set FITC
            say("[INFO] Setting filter to FITC fluorescence mode for scan")
            set_channel(target_channel_fitc)
            set_brightness(0)
            set_exposure(fitc_exposure)

            # D7: Configure FITC acquisition
            scan_elapsed_min2 = (time.time() - experiment_start_time) / 60.0
            scan_name_fitc = f"well_{well_id[0]}_{well_id[1]}_{grid_id}_FITC_scan_cycle{scan_cycle_index}_T{scan_elapsed_min2:.1f}min"
            current_x, current_y = get_x_y_position()
            add_acquisition_position(name=scan_name_fitc, x=current_x, y=current_y, width=None, height=None)
            add_channels(channel=target_channel_fitc, exposure=fitc_exposure)

            # D8: Capture FITC scan image
            say(f"[INFO] Capturing FITC scan image: {scan_name_fitc}")
            run_acquisition()
            say(f"[INFO] FITC scan image captured: {scan_name_fitc}")

            # D9: Mitosis detection on Brightfield image
            say(f"[INFO] Running mitosis detection on Brightfield scan image for well={well_id}, grid={grid_id}")
            bright_image_data = scan_bright_results[0]
            img_arr = bright_image_data.image
            if img_arr.ndim == 5:
                detection_image_data = bright_image_data
                detection_image_data.image = img_arr[0, 0, 0, :, :]
            elif img_arr.ndim == 4:
                detection_image_data = bright_image_data
                detection_image_data.image = img_arr[0, 0, :, :]
            elif img_arr.ndim == 3:
                detection_image_data = bright_image_data
                detection_image_data.image = img_arr[0, :, :]
            else:
                detection_image_data = bright_image_data

            mitosis_detections = detect_targets_in_image(
                image_data=detection_image_data,
                target_class="mitosis",
                confidence_threshold=0.5
            )
            say(f"[INFO] Mitosis detection result: {len(mitosis_detections)} event(s) found in well={well_id}, grid={grid_id}")

            # D10: Deduplication and queue update
            if len(mitosis_detections) > 0:
                key = (well_id, grid_id)
                if key not in triggered_set_per_scan:
                    triggered_set_per_scan[key] = True
                    detection_time = time.time()
                    say(f"[INFO] Mitosis detected at well={well_id}, grid={grid_id}. Scheduling 3 follow-up tasks.")
                    for step_num in range(1, 4):
                        scheduled_time = detection_time + step_num * FOLLOWUP_INTERVAL_SEC
                        high_frequency_queue.append({
                            'well_id': well_id,
                            'grid_id': grid_id,
                            'grid_center_XY': grid_center_xy,
                            'follow_up_step': step_num,
                            'absolute_scheduled_time': scheduled_time
                        })
                    say(f"[INFO] Follow-up tasks scheduled at +3min, +6min, +9min from detection time")
                else:
                    say(f"[INFO] Mitosis at well={well_id}, grid={grid_id} already triggered this scan cycle. Skipping duplicate.")

    scan_cycle_index += 1
    say(f"[INFO] Global scan cycle {scan_cycle_index - 1} completed. Total follow-up tasks in queue: {len(high_frequency_queue)}")

    # STEP E: Return to STEP B (loop continues naturally)

say("[INFO] Timed loop experiment completed. Total duration: 600 minutes.")
```
