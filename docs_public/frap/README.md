# FRAP Runtime Dependency

This directory contains the FRAP (Fluorescence Recovery After Photobleaching)
tool's runtime configuration and calibration utility. The EIMS executor loads
`frap_ui_profile.json` at runtime to drive the Olympus cellSens FRAP interface
through simulated mouse clicks.

## Files

| File | Purpose |
| --- | --- |
| `frap_ui_profile.json` | UI coordinate profile for the cellSens FRAP interface. Loaded by `tool/frap.py` at runtime. |
| `record_frap_click_once.py` | Calibration script for recording screen click coordinates to update the UI profile. |

## frap_ui_profile.json

Defines the screen layout of the cellSens FRAP control panel:

- `window_title_keyword`: keyword used to locate the cellSens window.
- `launch_command` / `launch_workdir`: cellSens executable path and working directory.
- `image_region`: pixel coordinates of the microscope image area within the cellSens window.
- `controls`: absolute screen pixel coordinates for FRAP UI buttons (FRAP tab, single-click bleaching, start, stop).
- `options`: pixel size, Cellpose segmentation parameters, and click timing settings.

All coordinates are **machine-specific** and depend on the display resolution,
DPI scaling, and cellSens window layout. Users must recalibrate for their own
environment using `record_frap_click_once.py`.

## record_frap_click_once.py

A standalone Windows utility for recording absolute screen click positions.

Usage:

1. Launch cellSens and open the FRAP panel.
2. Run the script and keep the cellSens window focused.
3. Press the `y` key to arm recording mode.
4. Click a target UI button. The script records the absolute screen coordinates
   and appends them to `frap_click_points.json` in the same directory.

Diagnostic mode:

```powershell
python docs_public\frap\record_frap_click_once.py --diagnose
```

This prints current screen dimensions, cursor position, and the detected cellSens
window bounds without recording any clicks, useful for troubleshooting coordinate
mismatches.
