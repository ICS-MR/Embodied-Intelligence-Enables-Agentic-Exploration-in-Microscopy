---
name: mitosis-brightfield-tracking
description: Planning template for bright-field mitosis observation workflows that combine repeated global scanning, built-in microscope detection, and revisit imaging of detected mitotic events.
skill_type: planning_template
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

Use this template when the user wants a bright-field mitosis workflow driven by the microscope's built-in detection capability.

Treat the workflow below as a planning reference, not as a default parameter set:
- Optimize bright-field illumination and focus at the current position, then acquire one initialization image
- Use the initialization image metadata to derive field-of-view dimensions and build the scan grid
- Repeatedly scan the user-specified area around the current XY position
- At each grid position: move, autofocus, capture one image, then run the built-in mitosis detection step immediately
- Add newly detected mitotic coordinates to a FIFO revisit queue and reject duplicates
- Revisit detected sites at the user-specified high-frequency schedule after detection
- Global scans take priority over revisit tasks; revisit work resumes between scans
- Stop registering new targets at the user-specified cutoff time, then finish already queued revisit tasks
- Do not rely on external image-analysis tools unless the user explicitly asks for them

Ask at most one blocking clarification question before producing the final plan. If critical information is missing, ask one consolidated question that:
- explicitly lists the workflow parameters that are still unspecified
- asks only for missing items that materially change execution
- does not silently fill in duration, scan area, interval, revisit offsets, or imaging mode unless the user has already provided them
- allows the user to answer either with concrete parameters or with an explicit statement such as "use your recommended parameters"

Use a clarification style like this when needed:
"To produce an executable mitosis experiment plan, I still need the workflow parameters that are not yet specified: imaging mode, objective or magnification, total duration, scan area, grid spacing rule, global scan interval, and the revisit timepoints after detection. You may also reply with 'use your recommended parameters'."

If the user only says they want a mitosis workflow and does not provide these parameters, prefer the single consolidated clarification question over silent defaults.

When producing the final executable plan:
- keep the workflow in microscope-oriented steps
- preserve the priority relationship between global scans and revisit tasks
- keep duplicate avoidance explicit
- keep the user-provided cutoff rule explicit
- preserve queued-task completion after the cutoff
