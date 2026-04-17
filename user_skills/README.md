# Planning Skills

Drop planning skills into this folder to guide the plan agent.

Supported layouts:
- One-file skills: `.md`, `.txt`, `.json`
- Package skills: `skill_name/SKILL.md`

How it works:
- The planner automatically loads skills from `user_skills/planning`.
- It ranks them against the current user command using trigger phrases, tags, example queries, keyword overlap, and priority.
- The selected skills are injected into the planning prompt as structured planning guidance.

Recommended structure:
- One workflow or rule set per skill.
- Keep the guidance concise and actionable.
- Use explicit trigger phrases when the skill should activate in a specific scenario.
- Add example queries if you want to bias selection toward representative requests.
- Use `priority` only when a skill should outrank otherwise similar skills.

Supported metadata:
- `name`
- `description`
- `tags`
- `triggers`
- `examples`
- `priority`
- `skill_type`
- `template_goal`
- `required_inputs`
- `planning_stages`
- `output_strategy`
- `content` / `guidance`

Skill types:
- `guidance`: default rule or workflow guidance
- `planning_template`: a reusable planning mode or workflow template

Template-specific metadata:
- `skill_type: planning_template`
- `template_goal`: what planning problem the template solves
- `required_inputs`: key slots the template checks before planning
- `planning_stages`: ordered planning workflow stages
- `output_strategy`: `direct_plan` or `single_question_then_plan`

Example package skill:

```md
---
name: Fluorescence First Pass
description: Conservative fluorescence planning workflow
tags:
- fluorescence
- preview
triggers:
- fluorescence imaging
- stain preview
examples:
- capture a fluorescence preview of the stained sample
priority: 5
---

- Start with a low-exposure preview before high-intensity acquisition.
- Confirm the main channel before running a large batch acquisition.
- Prefer a conservative scan plan before high-intensity capture.
```

Example JSON skill:

```json
{
  "name": "Brightfield Scan Rules",
  "description": "Preferred sequence for brightfield overview work",
  "tags": ["brightfield", "scan", "focus"],
  "triggers": ["brightfield image", "overview scan"],
  "examples": ["capture a brightfield overview image"],
  "priority": 3,
  "content": "Always focus before large-area scanning. Prefer low magnification for overview acquisition."
}
```

Example planning template skill:

```md
---
name: Clarify Missing Imaging Parameters
skill_type: planning_template
description: Ask one blocking question before final planning when key imaging inputs are missing
tags:
- planning
- clarification
template_goal: resolve underspecified imaging tasks
required_inputs:
- fluorescence_state
- magnification
planning_stages:
- assess_missing_info
- ask_one_blocking_question
- integrate_user_answer
- produce_final_plan
output_strategy: single_question_then_plan
---

Ask exactly one concise blocking question, then continue to a final executable plan.
```
