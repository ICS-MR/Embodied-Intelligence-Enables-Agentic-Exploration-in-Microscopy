# FRAP Runtime Dependency

This directory contains the FRAP (Fluorescence Recovery After Photobleaching)
tool's runtime configuration and calibration utility. The EIMS executor loads
`frap_ui_profile.json` at runtime to drive the Olympus cellSens FRAP interface
through simulated mouse clicks.

## Files

| File | Purpose |
| --- | --- |
| `frap_ui_profile.json` | UI coordinate profile for the cellSens FRAP interface. Loaded by `tool/frap.py` at runtime. |
| `frap_click_points.json` | Click coordinates recorded by `record_frap_click_once.py`, used to update the profile. |
| `record_frap_click_once.py` | Calibration script for recording screen click coordinates to update the UI profile. |
| `capture_frap_startup_reference.py` | Captures startup verification reference images from the configured regions. |
| `calibrate_frap_startup_reference.py` | Scores ready/not-ready screenshots against a reference and suggests `startup_match_threshold`. |
| `screenshots/` | Reference screenshots of the cellSens loaded and loading (splash) states. |

## frap_ui_profile.json

Defines the screen layout of the cellSens FRAP control panel:

- `window_title_keyword`: keyword used to locate the cellSens window.
- `launch_command` / `launch_workdir`: cellSens executable path and working directory.
- `image_region`: pixel coordinates of the microscope image area within the cellSens window.
- `controls`: absolute screen pixel coordinates for FRAP UI buttons (FRAP tab, single-click bleaching, start, stop).
- `options`: pixel size, Cellpose segmentation parameters, and click timing settings,
  startup waits (`startup_window_timeout_sec`, `startup_settle_seconds`), and startup
  visual verification settings (`startup_match_threshold`, `startup_reference_checks`).

All coordinates are **machine-specific** and depend on the display resolution,
DPI scaling, and cellSens window layout. Users must recalibrate for their own
environment using `record_frap_click_once.py`.

## Startup visual verification

Before each FRAP session the tool verifies that cellSens is actually ready
instead of trusting fixed sleeps alone. Two reference-image checks run in
`tool/frap.py`:

1. `pre_click`: after the settle wait, a stable region that does not depend on
   the selected tab (by default the top toolbar) must match its reference
   image. On mismatch, cellSens is considered not fully loaded and the FRAP
   tab click is refused.
2. `post_click`: after clicking the FRAP tab, the bottom tab strip (with the
   FRAP tab selected) must match its reference image. This confirms that the
   click actually opened the FRAP console.

Each check compares a grayscale screenshot of its configured region against
the reference image (`cv2.matchTemplate`, normalized correlation). A score
below `startup_match_threshold` raises an error and saves an evidence
screenshot under `logs/`. A check with an empty `image` path is skipped with a
warning.

Setup on the cellSens machine:

```powershell
# 1. Capture references after cellSens fully loads; keep the FRAP tab open
#    for the post_click reference.
python docs_public\frap\capture_frap_startup_reference.py --check pre_click
python docs_public\frap\capture_frap_startup_reference.py --check post_click

# 2. Optional: verify the threshold separates ready / not-ready states.
#    Full-screen screenshots are cropped to the configured region automatically.
python docs_public\frap\calibrate_frap_startup_reference.py --check post_click ^
    --ready loaded_1.png loaded_2.png --not-ready splash_1.png
```

The `screenshots/` directory documents the two states: `cellSens_loaded_...`
shows the ready interface, and `cellSens_loading_...` shows the startup splash
during which no verification can pass.

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
