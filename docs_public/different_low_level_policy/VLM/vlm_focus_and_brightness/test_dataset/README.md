# Focus and Brightness Test Dataset

This directory contains representative image sequences for public inspection of
focus selection and brightness selection behavior.

## Structure

| Path | Content |
| --- | --- |
| `focus/` | 101 brightfield Z-stack images for focus selection. Filenames encode the Z position. |
| `brightness/` | 101 brightfield images captured across brightness settings. Filenames encode the brightness value. |

## Intended Use

- Use `focus/` to inspect how a scoring method selects the sharpest image from a Z-stack.
- Use `brightness/` to inspect how a scoring method selects an appropriate brightfield illumination level.
- These images are representative public test images for qualitative review and method comparison, not a full statistical benchmark.

The benchmark scripts in the parent directory acquire fresh images from a connected
microscope when run online. This dataset provides static image examples so the
focus and brightness tasks can be reviewed without exposing any private path or
local acquisition environment.

The benchmark scripts also support offline evaluation over this dataset by setting
`"source": "testset"` in their `RUN_CONFIG`; that path scores these images without
connecting the microscope backend.
