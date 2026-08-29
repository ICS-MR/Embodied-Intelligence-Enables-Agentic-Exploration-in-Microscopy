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
| `capture_frap_startup_reference.py` | Captures startup verification reference images from the configured regions. |
| `calibrate_frap_startup_reference.py` | Scores ready/not-ready screenshots against a reference and suggests `startup_match_threshold`. |
| `references/` | Runtime verification reference images (`pre_click.png`, `post_click.png`, `laser_on.png`, `laser_off.png`) referenced by `frap_ui_profile.json`. |

## frap_ui_profile.json

Defines the screen layout of the cellSens FRAP control panel:

- `window_title_keyword`: keyword used to locate the cellSens window.
- `launch_command` / `launch_workdir`: cellSens executable path and working directory.
- `image_region`: pixel coordinates of the microscope image area within the cellSens window.
- `controls`: absolute screen pixel coordinates for FRAP UI buttons (FRAP tab,
  single-click bleaching, start, stop) and the cellSens close button
  (`cellsens_close_button`).
- `options`: pixel size, Cellpose segmentation parameters, and click timing settings;
  startup waits (`startup_window_timeout_sec`, `startup_poll_interval_sec`,
  `startup_settle_seconds`, `startup_ready_settle_seconds`); startup visual
  verification settings (`startup_match_threshold`, `startup_reference_checks`);
  start/stop state checks (`frap_start_state_check` / `frap_stop_state_check` with
  `frap_start_settle_seconds` / `frap_stop_settle_seconds`); and close verification
  settings (`close_settle_seconds`, `close_window_timeout_sec`,
  `close_process_timeout_sec`).

All coordinates are **machine-specific** and depend on the display resolution,
DPI scaling, and cellSens window layout. Users must recalibrate for their own
environment using `record_frap_click_once.py`.

## Startup visual verification

Before each FRAP session the tool verifies that cellSens is actually ready
instead of trusting a fixed one-shot sleep. Two reference-image checks run in
`tool/frap.py`:

1. `pre_click`: during readiness polling, a stable region that does not depend
   on the selected tab (by default the top toolbar) must match its reference
   image. The poll interval is controlled by `startup_poll_interval_sec`.
   After the match, the tool waits an additional ready-settle period
   before clicking the FRAP tab.
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

The `references/` images above are machine-specific captures: regenerate them
with `capture_frap_startup_reference.py` after recalibrating the profile on a
new display or cellSens layout.

## Lifecycle state verification

Beyond startup readiness, `tool/frap.py` verifies each lifecycle transition
against a reference image:

- **Start**: after clicking `frap_start_button`, the tool waits
  `frap_start_settle_seconds` and then polls the `frap_start_state_check` region
  until it matches `references/laser_on.png`.
- **Stop**: after clicking `frap_stop_button`, the tool waits
  `frap_stop_settle_seconds` and then polls the `frap_stop_state_check` region
  until it matches `references/laser_off.png`.

Both checks reuse `startup_poll_interval_sec` for polling and
`startup_window_timeout_sec` as the timeout. A check with an empty `image` path
is skipped with a warning.

Session close verifies that cellSens actually exits:

1. Click the configured `cellsens_close_button` and wait `close_settle_seconds`.
2. Wait up to `close_window_timeout_sec` for the cellSens window to disappear;
   if the window is still present, fall back to `Alt+F4`.
3. Wait for the `SisXV.exe` process to exit within `close_process_timeout_sec`.

## record_frap_click_once.py

A standalone Windows utility for recording absolute screen click positions.

Usage:

1. Launch cellSens and open the FRAP panel.
2. Run the script and keep the cellSens window focused.
3. Press the `y` key to arm recording mode.
4. Click a target UI button. The script records the absolute screen coordinates
   and appends them to `frap_click_points.json` (created on first use) in the
   same directory.

Diagnostic mode:

```powershell
python docs_public\frap\record_frap_click_once.py --diagnose
```

This prints current screen dimensions, cursor position, and the detected cellSens
window bounds without recording any clicks, useful for troubleshooting coordinate
mismatches.
