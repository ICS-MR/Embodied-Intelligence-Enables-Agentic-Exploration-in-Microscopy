# Custom Skill Guide

In this project, a `skill` is a file-driven planning aid rather than a code plugin.

You can think about skills in two categories:

- `guidance`: lightweight planning rules or heuristics
- `planning_template`: a full planning mode or workflow template

Supported locations:

- Single files: `user_skills/planning/*.md`
- Package directories: `user_skills/planning/<skill_name>/SKILL.md`

Supported file formats:

- `.md`
- `.txt`
- `.json`

## When To Use `guidance`

Use `guidance` when you want to tell the planner:

- how this kind of task is usually handled
- what precautions matter
- which steps should be prioritized

If `skill_type` is omitted, it defaults to `guidance`.

Example:

```md
---
name: Brightfield Focus Workflow
description: Recommended planning order for brightfield tasks
tags:
- brightfield
- focus
triggers:
- brightfield image
- overview scan
examples:
- capture a brightfield overview image
priority: 3
---

- Always focus before large-area brightfield scanning.
- Prefer low magnification for overview acquisition.
- Avoid unnecessary fluorescence switching.
```

## When To Use `planning_template`

Use `planning_template` when you want to express a planning mode rather than a single rule.

Typical cases:

- the workflow is fixed
- some inputs are mandatory before planning can continue
- the planner should ask a blocking question when information is missing
- the planner should only produce the final plan after clarification

Example scenarios:

- mitosis observation mode
- multi-channel fluorescence parameter confirmation
- low-magnification scan followed by high-magnification review

## Recommended `planning_template` Fields

Suggested front matter:

```md
---
name: Your Template Name
skill_type: planning_template
description: What planning problem this template solves
tags:
- planning
- workflow
triggers:
- user phrase 1
- user phrase 2
template_goal: what this template is meant to solve
required_inputs:
- key_input_1
- key_input_2
planning_stages:
- assess_missing_info
- ask_one_blocking_question
- integrate_user_answer
- produce_final_plan
output_strategy: single_question_then_plan
priority: 5
---
```

Suggested body content:

- applicable scenarios
- planning objective
- how to ask for missing information
- recommended workflow
- behavioral constraints

## Key Field Meanings

- `name`: concise and clear skill name
- `description`: one-line summary of the skill
- `tags`: topic labels that help routing
- `triggers`: realistic user phrases
- `examples`: representative user requests
- `priority`: routing weight; larger values are preferred when similar skills compete
- `template_goal`: the planning problem this template solves
- `required_inputs`: the inputs the template considers critical
- `planning_stages`: the preferred organization of the planning workflow
- `output_strategy`: how the planner should behave

Recommended `output_strategy` values:

- `direct_plan`: produce a plan directly whenever possible
- `single_question_then_plan`: ask exactly one blocking question before producing the final plan

## How To Encourage Clarification

If you want a skill to ask one blocking question before planning:

1. Set `skill_type` to `planning_template`
2. Set `output_strategy` to `single_question_then_plan`
3. State clearly in the body:
   - when information is insufficient
   - how many questions should be asked
   - which question has the highest priority

Good clarification rules:

- ask only one critical question at a time
- avoid vague prompts like “please provide more details”
- ask a truly blocking question

## How To Design A Good Skill

Recommended practices:

- keep each skill focused on one job
- make the name immediately understandable
- write `triggers` in realistic user language
- make `examples` representative
- keep the body short, clear, and actionable

Avoid:

- mixing too many workflows into one skill
- overly abstract triggers
- writing only high-level principles without behavior rules

## How To Test Whether A Skill Works

You can test a skill like this:

1. Place the skill file under `user_skills/planning/`
2. Submit a normal task request
3. Enter `debug_plan` during the interaction phase

You should then be able to inspect:

- which skills were matched for the round
- whether a planning template was activated
- what the raw planner output looked like

## Recommended Naming

Keep the filename and skill name semantically aligned, for example:

- `brightfield_focus_workflow.md`
- `clarify_missing_imaging_parameters.md`
- `mitosis_observation_template.md`

## Minimal `planning_template` Example

```md
---
name: Clarify Missing Imaging Parameters
skill_type: planning_template
description: Ask one blocking question before final planning when critical imaging parameters are missing
tags:
- planning
- clarification
- imaging
output_strategy: single_question_then_plan
priority: 5
---

Use this template when the request cannot become executable until one blocking imaging parameter is clarified.
Ask exactly one concise question.
After receiving the answer, integrate it and return a final executable plan.
```

## Recommended Rule Of Thumb

When adding a new experiment mode, ask:

- is this mainly a rule or preference?
- or is it a planning mode with clarification and stage control?

Use:

- `guidance` for rules, preferences, and cautions
- `planning_template` for question-driven workflows with stage control
