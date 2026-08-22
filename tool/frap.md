# FRAP Tool Notes

This file records FRAP settings that are specific to the current cellSens workstation. These values are not treated as universal defaults. For the new environment, an additional calibration process is required.

## Workstation-Specific UI Profile

The active profile is `docs_public/frap/frap_ui_profile.json`.

Current cellSens launch configuration:

```json
"launch_command": [
  "C:\\Program Files\\cellSens Dimension\\SisXV.exe"
],
"launch_workdir": "C:\\Program Files\\cellSens Dimension"
```

Current screen and UI assumptions:

- Screen coordinate system: absolute physical screen pixels.
- Expected screen size during calibration: `3840 x 2160`.
- cellSens window/layout is expected to remain unchanged.
- FRAP control and image-region points are calibrated for this workstation and this cellSens layout.

Current operation image region:

```json
"image_region": {
  "left": 1170,
  "top": 224,
  "right": 2982,
  "bottom": 2039,
  "source_width": 2048,
  "source_height": 2048
}
```

`left/top/right/bottom` are the screen absolute physical pixel bounds of the displayed field-of-view region in cellSens.

`source_width/source_height = 2048 x 2048` describe the source image coordinate system used for FRAP coordinate conversion. They are device/session-specific assumptions for the current camera/image setup, not general microscope constants.

Current FRAP control points:

```json
"bottom_frap_tab_button": {"x": 3684, "y": 2014},
"single_click_fade_multi_point_button": {"x": 3425, "y": 1055},
"frap_start_button": {"x": 3440, "y": 240},
"frap_stop_button": {"x": 3713, "y": 227}
```

These points are valid only while the cellSens window position, monitor scaling, and FRAP panel layout remain unchanged.

Current physical pixel scale:

```json
"pixel_size_x_um": 0.32468,
"pixel_size_y_um": 0.32468
```

These values are used by `laser_position(x, y)` to convert field-centered microns into source-image and screen coordinates.

## Lifetime Behavior

Instantiating `Frap` opens or focuses cellSens and prepares the FRAP panel.

Releasing the `Frap` instance closes cellSens through `Frap.close()` / `Frap.__del__()`. This is intentional for the current workflow so that a later `Frap` load starts from a clean cellSens process instead of inheriting stale UI state.

`laser_off()` does not close cellSens. It clicks the configured `frap_stop_button` and only stops the FRAP operation.

## Cell Detection Area Filter

`cellpose_min_area_px` is the minimum refined Cellpose mask area required for a detected object to be treated as a valid cell.

Current value:

```json
"cellpose_min_area_px": 3000.0
```

This threshold is used to remove small bright artifacts that Cellpose may segment as cells. It is applied after the fluorescence-intensity refinement step, so it filters the refined cell body area rather than only the raw Cellpose mask area.

Current validation scope:

- Small bright artifacts are filtered out.
- The expected target cells are retained.

Known limitation:

- `3000.0` is an empirical threshold for the current 60x FRAP image scale and current cellSens image-region setup.
- If magnification, display scaling, source image size, or cell morphology changes, this value may need to be revalidated.
- A value that is too high can remove real small cells; a value that is too low can keep bright artifacts.

For the current stage, this is an accepted fixed parameter rather than an adaptive threshold.
