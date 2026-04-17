---
name: mitosis-brightfield-tracking
description: Planning template for mitosis observation and tracking tasks in bright-field microscopy. Use when the user asks for mitosis monitoring, mitosis tracking, cell division observation, or other repeated global-scan workflows that rely on built-in microscope detection and high-frequency revisit imaging.
triggers:
- mitosis
- mitosis monitoring
- mitosis tracking
- cell division observation
- cell division tracking
- brightfield mitosis experiment
examples:
- Run a 2-hour brightfield mitosis monitoring workflow
- I want to run a mitosis tracking experiment
- Plan a cell division observation workflow with global scans and revisit imaging
priority: 90
skill_type: planning_template
template_goal: Plan a bright-field mitosis workflow that performs initialization, grid generation, repeated global scanning, built-in detection, duplicate-safe queueing, FIFO high-frequency revisit imaging, and strict boundary control.
planning_stages:
- initialization
- grid_generation
- global_scan
- detection_and_queueing
- high_frequency_tracking
- boundary_control
required_inputs:
- specimen or biological target
- objective or magnification requirement
- imaging mode
- total duration
- scan area
- grid spacing rule
- global scan interval
- high-frequency revisit schedule
output_strategy: single_question_then_plan
---

Use this template when the user wants a mitosis workflow and the microscope's built-in rapid detection module should drive the experiment.

Treat the workflow below as a reference structure, not as an automatic default parameter set:
- Initialization: optimize illumination and focus at the current position, then acquire one image
- Grid generation: read the first image metadata to obtain field-of-view width and height
- Repeated global scanning across a user-specified area centered on the current XY position
- Per-grid-position actions: move stage, autofocus, capture one image, invoke the built-in microscope detection module immediately
- Detection handling: add newly detected mitotic coordinates to a queue and avoid duplicate registrations
- High-frequency revisit imaging at user-specified absolute offsets after detection
- FIFO sequential execution for revisit tasks
- Priority rule: global scan always preempts high-frequency tracking; resume high-frequency tasks after the scan
- Boundary rule: stop registering new tasks at the user-specified cutoff time, then finish the queued tasks
- Analysis policy: do not rely on external image analysis software unless the user explicitly asks for it

Ask at most one blocking clarification question before producing the final plan. If critical information is missing, ask one consolidated question that:
- explicitly lists the workflow parameters that are still unspecified
- asks only for missing items that materially change execution
- does not silently fill in duration, scan area, interval, revisit offsets, or imaging mode unless the user has already provided them
- allows the user to answer either with concrete parameters or with an explicit statement such as "use your recommended parameters"

Use a consolidated clarification style like this when needed:
"To produce an executable mitosis experiment plan, I still need the workflow parameters that are not yet specified: imaging mode, objective or magnification, total duration, scan area, grid spacing rule, global scan interval, and the high-frequency revisit timepoints after detection. You may also reply with 'use your recommended parameters'."

When the user simply says they want a mitosis task and does not provide these workflow parameters, prefer asking the one consolidated clarification question instead of silently applying a default parameter set.

When producing the final executable plan:
- keep the workflow in microscope-oriented steps
- preserve the priority relationship between global scans and revisit tasks
- keep duplicate avoidance explicit
- keep the user-provided cutoff rule explicit
- preserve queued-task completion after the cutoff