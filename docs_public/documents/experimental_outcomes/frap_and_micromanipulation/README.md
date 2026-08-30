# FRAP and Micromanipulation (Fig. 5c, Fig. 5d, Fig. S2)

FRAP + micromanipulation driving tasks and tools.

## Structure

- `frap/frap_test/` - processed FRAP test session display MDs:
  - `frap/frap_test/01_cell_bleaching.md` - single-cell photobleaching.
  - `frap/frap_test/02_writing_letters.md` - letter drawing with FRAP laser.
  - `frap/frap_test/03_geometric_patterns.md` - geometric pattern drawing.
  - `frap/frap_test/04_spiral_drawing.md` - spiral pattern drawing.
  - `frap/frap_test/05_circle_36_points_installed.md` - 36-point circle with objective installed.
  - `frap/frap_test/06_circle_36_points_uninstalled.md` - 36-point circle with objective uninstalled.
  - `frap/prompts/` - archived FRAP prompt snapshots used for the published replay.
    - `frap/prompts/frap.executor_prompt.txt`
    - `frap/prompts/frap.planner_summary.txt`

- `micromanipulation_system/test/` - processed micromanipulation-system interaction records
  (movement, spheroid localization, pump aspiration/dispensing, and MP285-based
  spheroid-transfer / -release / -grasping execution records):
  - `micromanipulation_system/test/01_move_row_3_column_4.md` - stage movement to a specified grid position.
  - `micromanipulation_system/test/02_aspirate_and_dispense_10_ml.md` - pump aspiration and dispensing.
  - `micromanipulation_system/test/03_locate_microsphere.md` - microsphere localization.
  - `micromanipulation_system/test/04_spheroid_transfer.md` - end-to-end EIMS spheroid-transfer workflow (MP285).
  - `micromanipulation_system/test/05_release_three_balls_stage_alignment.md` - releasing three balls with stage alignment (MP285).
  - `micromanipulation_system/test/06_grasp_green_balls_pump_velocity_setup.md` - grasping green balls with pump velocity setup (MP285).
  - `micromanipulation_system/prompts/` - archived MP285 prompt snapshots used for the published replay.
    - `micromanipulation_system/prompts/mp285_controller.executor_prompt.txt`
    - `micromanipulation_system/prompts/mp285_controller.planner_summary.txt`

