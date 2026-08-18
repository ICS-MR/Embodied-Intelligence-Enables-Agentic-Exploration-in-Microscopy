---
name: mitosis-multichannel-tracking
description: Expert-guided multichannel mitosis observation and tracking driven by the microscope's built-in detection module. Use when the user asks for mitosis monitoring, cell division tracking, repeated global scans with revisit imaging, or brightfield-detected mitosis with fluorescence confirmation.
triggers:
- mitosis
- mitosis monitoring
- mitosis tracking
- cell division observation
- cell division tracking
- brightfield mitosis experiment
- brightfield fitc mitosis
- multichannel mitosis
- fitc mitosis tracking
- mitotic dynamics analysis
examples:
- Run a brightfield mitosis monitoring workflow
- I want to run a mitosis tracking experiment
- Plan a cell division observation workflow with global scans and revisit imaging
- Acquire Brightfield and FITC images but detect mitosis only from Brightfield
- Plan a multichannel mitosis workflow with Brightfield detection and fluorescence confirmation
priority: 90
template_goal: Design a scientifically grounded multichannel mitosis monitoring protocol, recommend concrete parameters with rationale, and confirm before planning.
required_inputs:
- specimen or biological target
- target wells or scan positions
output_strategy: recommend_then_confirm
---

Use for mitosis monitoring or tracking driven by the microscope's built-in detection module, especially when the detection channel and the confirmation channel differ.

## Design rationale (use to justify recommendations)
- Brightfield is the detection channel: label-free and safe for repeated high-frequency imaging; mitosis (rounding, furrow, separation) is visible without fluorescence.
- FITC is the confirmation channel: it verifies the event, but every fluorescence exposure bleaches and stresses cells. Default (when the user does not specify channel usage): capture FITC at init and at trigger/revisit only. If the user requests paired acquisition (e.g., "paired Brightfield and FITC"), capture both channels at every acquisition position; never narrow to init/revisit-only.
- A mitotic event lasts ~30-90 min; scan every 5-10 min (default 6 min) to sample it several times while limiting light exposure.
- Revisit +3/+6/+9 min tracks division progression; return to the triggered grid center, not a single cell.
- Global scans preempt revisits; delayed revisits keep order. At cutoff: no new scans/registrations; discard remaining queue.

## Recommended defaults (overridable)
- Objective: user value; default 20x. Channels: BF detection; FITC confirmation at init and trigger/revisit (default only; user-specified paired acquisition overrides it - both channels at every position).
- Grid: 3x3 per well, centers 100% FOV width/height apart, centered on the well.
- Scan interval: 6 min (range 5-10 min). Revisit offsets: +3/+6/+9 min. Cutoff: user-specified duration.
- Unique filenames encoding well, grid, channel, acquisition type, timepoint.

## Protocol skeleton
1. Init and grid: per well optimize illumination/focus at center, capture paired BF+FITC init set, read FOV from metadata, build grid.
2. Global scan: fixed well/grid order each interval; move, autofocus, capture BF (+FITC when the user requested paired acquisition at every position), run built-in mitosis detection on BF only (never FITC).
3. Revisit: on trigger register +3/+6/+9 min absolute offsets, dedup per subregion per cycle; each revisit captures paired BF+FITC at the subregion center; scans preempt revisits, delayed revisits keep order.
4. Boundary: at cutoff stop new scans/registrations, discard the remaining queue.

## Resolution policy
- Enough parameters: produce one complete protocol with short rationale per key choice, then ask confirm/adjust; do not ask again.
- Missing critical info (specimen, target wells/positions): first summarize the recommended protocol (defaults with one-line rationale for interval, channels, revisit), then ask one consolidated question for blocking items only; accept "use your recommended parameters".
- Never silently change user-provided values; label recommended values.
- If the user specifies paired/multichannel acquisition, every acquisition position (init, global scan, revisit) captures both channels; never narrow it to init/revisit-only.
- The resolved protocol is microscope-driven only: do not add image-analysis platform steps unless the user explicitly requests analysis.
- Use the user-confirmed parameters and recommended defaults; the example only illustrates output structure.
- Output a formal, concise protocol with the four sections above.

## Resolved example (structure reference only)
"Run a 6-hour (T=0-360 min) Brightfield-detected mitosis experiment at 40x on wells (1,1),(1,2),(2,1) of a 6-well plate with FITC confirmation, using only the built-in detection module. Scan BF every 5 min over a 2x2 grid (100% FOV spacing); detect mitosis on BF only. On trigger, capture paired BF+FITC at +2/+4/+6 min at the subregion center. Scans preempt revisits; stop at T=360. Save unique filenames encoding well, grid, channel, type, timepoint. Rationale: 5-min scans sample 30-90 min events; FITC limited to confirmation to reduce photobleaching; +2/+4/+6 min track division progression."
