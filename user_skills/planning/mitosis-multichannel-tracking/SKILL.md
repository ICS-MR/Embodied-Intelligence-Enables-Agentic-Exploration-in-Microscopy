---
name: mitosis-multichannel-tracking
description: Skill package for multichannel mitosis observation and tracking workflows driven by the microscope's built-in detection module. Use when the user asks for mitosis monitoring, cell division tracking, repeated global scans with revisit imaging, or paired Brightfield plus fluorescence acquisition where one channel is used for detection and the other channel is saved alongside it.
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
- Run a 2-hour brightfield mitosis monitoring workflow
- I want to run a mitosis tracking experiment
- Plan a cell division observation workflow with global scans and revisit imaging
- Acquire Brightfield and FITC images but detect mitosis only from Brightfield
- Plan a multichannel mitosis workflow with Brightfield detection and fluorescence saving
priority: 90
template_goal: Resolve multichannel mitosis workflow requests into one complete natural-language task instruction for downstream planning.
required_inputs:
- specimen or biological target
- objective or magnification requirement
- acquisition channels
- detection channel
- total duration
- grid definition
- global scan interval
- high-frequency revisit schedule
- target wells or scan positions
output_strategy: single_question_then_plan
---

Use this skill when the user wants a mitosis workflow and the microscope's built-in rapid detection module should drive the experiment, especially when acquisition channels and detection channels are not the same.

Interpretation rules:
- Treat the workflow below as a reference structure, not as an automatic default parameter set.
- If paired acquisition is requested, capture both Brightfield and FITC images at each required position.
- When paired Brightfield and FITC imaging is requested, run mitosis detection on the Brightfield image only and do not use FITC for triggering unless the user explicitly requests otherwise.
- If the user says "3x3 grid", interpret this as 3 rows by 3 columns of imaging regions, totaling 9 subregions per well, not a 3 mm x 3 mm physical area unless the user explicitly says so.
- Build the grid using the user-specified spacing rule, or field-of-view-sized spacing when the user says adjacent grid centers should be separated by 100 percent of field-of-view width and height.
- Save all initialization, global-scan, and revisit images with unique filenames that preserve well id, grid id, channel, acquisition type, and timepoint or cycle index.
- At the user-specified cutoff time, stop starting new global scans, stop registering new revisit tasks, and stop executing any remaining queued revisit tasks beyond the cutoff.
- Do not rely on external image-analysis tools unless the user explicitly asks for them.

Scheduling semantics:
- Revisit timepoints are defined relative to the actual detection time of the triggering event, not relative to the next global-scan boundary or the next coarse scheduler tick.
- A revisit requested at detection_time + offset should be executed as close as possible to that absolute timepoint, subject only to the stated global-scan priority rule.
- Do not approximate revisit timing by rounding all revisit tasks to the next global scan cycle.
- Deduplication applies only within a single global scan cycle. The same well/subregion may trigger again in later global scan cycles and may then generate a new revisit schedule.
- A subregion should not remain permanently enrolled for the entire experiment after its first detection unless the user explicitly requests that behavior.
- Global scans have higher priority than revisit work. If a scheduled global scan becomes due while revisit work is pending, perform the global scan first, then resume delayed revisit work afterward only if it is still within the cutoff.
- Preserve the intended revisit order for the same triggered subregion even when revisit work is delayed by a higher-priority global scan.
- The cutoff rule is literal: after the cutoff, do not start new scans, do not register new revisit work, and do not drain the remaining revisit queue as cleanup work beyond the cutoff.

Workflow reference:
- Initialization: for each target well, optimize illumination and focus at the monitoring center, then acquire the requested initialization image set.
- Read the first initialization image metadata to obtain field-of-view width and height.
- Repeatedly scan the user-specified wells and grid subregions in fixed order.
- At each grid position: move stage, autofocus, capture the requested image set, then run the built-in mitosis detection step immediately on the Brightfield image only.
- Add newly detected grid subregions to a revisit queue and avoid duplicate registrations within the same global scan.
- Revisit triggered grid subregions at the user-specified high-frequency schedule after detection.
- Global scans take priority over revisit tasks; revisit work resumes between scans while preserving follow-up order for the same triggered grid subregion.

Resolver rules:
- If one or more critical workflow parameters are still missing, ask exactly one blocking consolidated clarification question.
- Ask only for missing items that materially change execution.
- Do not silently fill in acquisition channels, detection channel, duration, grid definition, interval, revisit offsets, or target wells unless the user has already provided them or explicitly says to use recommended parameters.
- If the request is sufficiently specified, do not ask again. Instead, rewrite the request into one complete natural-language experiment task instruction for the downstream planner.
- The final resolved task instruction must be written as a formal experiment protocol, not as fragmented notes, reminders, or task steps.
- The final resolved task instruction should use a continuous prose style with numbered sections when appropriate.
- When the workflow is fully specified, the resolved task instruction should explicitly include these sections in order:
  1. Initialization and Grid Generation
  2. Global Scanning and Triggering
  3. High-Frequency Tracking and Scheduling
  4. Boundary Control
- Preserve all user-confirmed parameters exactly, including wells, imaging channels, detection channel, duration, grid definition, spacing rule, scan interval, follow-up offsets, and cutoff behavior.
- The final task instruction must preserve target wells, traversal order, channel roles, detection channel, duplicate avoidance, revisit timing, image-saving requirements, and cutoff behavior.
- The final task instruction must state revisit timing as absolute offsets after detection and must not weaken this into scan-cycle-aligned timing.
- The final task instruction must make it explicit that deduplication is scan-local rather than experiment-lifetime enrollment.
- The final task instruction must make it explicit that queued revisit work is not drained after the cutoff.
- If implementation detail is described, prefer metadata-derived field-of-view dimensions over hard-coded image sizes.

Desired resolved instruction style:
- Start with one complete introductory paragraph describing the experiment goal, channels, duration, target wells, and detection policy.
- Then present the workflow as a formal protocol with the four numbered sections above.
- Use full sentences and explicit constraints, similar to a complete experiment specification that can be handed to a downstream planner verbatim.

Avoid these mistakes:
- Do not convert revisit scheduling into a simple fixed 12-minute polling loop when the requested revisit offsets are finer than the scan interval.
- Do not keep a triggered subregion permanently suppressed from future enrollment after its first detection.
- Do not continue queued revisit work after the cutoff as a final cleanup step.
- Do not hard-code field-of-view width or height when they can be read from image metadata.
- Do not replace the required formal section headings with arbitrary headings if the workflow has already been specified in a stricter section structure.

Resolved instruction example:
The following example is a style-and-structure reference only. Do not copy its parameter values unless the user request or clarification history explicitly provides the same values.

This experiment employs paired bright-field and FITC fluorescence imaging over a total duration of 10 hours (T = 0–600 min) to capture the dynamic process of cell mitosis in four target wells of a 24-well plate, specifically wells (2,2), (2,3), (3,2), and (3,3). Throughout the experiment, the microscope's built-in rapid detection module is used, without relying on external image analysis software. Mitosis detection must be performed only on the bright-field image, not on the FITC fluorescence image.

1. Initialization and Grid Generation
The system first obtains the coordinates of all wells in the 24-well plate and then sequentially moves to wells (2,2), (2,3), (3,2), and (3,3). For each target well, the system optimizes illumination and focus at the monitoring center of the well and captures one paired image set consisting of one bright-field image and one FITC fluorescence image. The actual field-of-view size is then determined by reading the image metadata from the initialization acquisition. Using the monitoring center of each well as the center, an independent 3 × 3 grid of imaging regions is generated for that well (3 rows and 3 columns, totaling 9 grid subregions), with step sizes set to 100% of the field-of-view width and height. The center coordinates of all grid subregions are recorded, and each subregion is assigned a unique grid identifier.

2. Global Scanning and Triggering
Starting from T = 0, the system performs a full global scan every 12 minutes. Each global scan traverses all four target wells in the fixed order of (2,2), (2,3), (3,2), and (3,3), and within each well traverses all grid subregions in a fixed order. Each grid visit includes stage movement, autofocus, acquisition of one paired image set consisting of one bright-field image and one FITC fluorescence image, and immediate invocation of the built-in rapid detection module on the bright-field image only. All acquired images, including initialization images, global scan images, and high-frequency follow-up images, must be saved. Each saved image must have a unique filename that preserves the full acquisition history and prevents overwriting, for example by including the well identifier, grid identifier, scan or follow-up type, channel name, and acquisition timepoint or cycle index. If one or more mitotic events are detected within the same grid subregion during a single global scan, that grid subregion is treated as one triggered tracking unit and is added to the high-frequency tracking queue only once during that scan. In other words, deduplication is performed at the level of well position plus grid identifier within each single global scan, and the tracking target is the entire triggered grid subregion rather than any individual detection coordinate.

3. High-Frequency Tracking and Scheduling
Once a grid subregion is triggered, the system performs high-frequency follow-up imaging for that same grid subregion at approximately 3, 6, and 9 minutes after detection. Each follow-up acquisition must capture one paired image set consisting of one bright-field image and one FITC fluorescence image. High-frequency tracking always returns to the center of the corresponding grid subregion rather than to an individual detected cell position. Tasks are executed sequentially, and the intended temporal order of follow-up acquisitions for the same triggered grid subregion must be preserved. Global scanning has higher priority than high-frequency acquisition: if a scheduled global scan time is reached during high-frequency acquisition, the high-frequency task is suspended immediately. The global scan is performed first, after which high-frequency acquisition resumes. If any planned follow-up acquisition is delayed because of the higher-priority global scan, it should be executed as soon as possible after the interruption, while keeping the original follow-up order unchanged.

4. Boundary Control
At T = 600 minutes, the experiment stops completely. No new global scans are started, no new high-frequency tracking tasks are registered, and no remaining queued tasks are continued beyond this time point. Any operation that has not been completed by T = 600 minutes is terminated at that boundary.

When critical information is missing, use a consolidated clarification style like this:
"To produce an executable mitosis experiment instruction, I still need the workflow parameters that are not yet specified: objective or magnification, acquisition channels, which channel should be used for mitosis detection, total duration, grid definition, global scan interval, revisit timepoints after detection, and target wells or scan positions. You may also reply with 'use your recommended parameters'."
