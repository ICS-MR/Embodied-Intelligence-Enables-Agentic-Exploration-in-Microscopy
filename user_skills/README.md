# Planning Skills

Drop planning skills into this folder to guide the plan agent.

Supported layouts:
- One-file skills: `.md`, `.txt`, `.json`
- Package skills: `skill_name/SKILL.md`

How it works:
- The skill resolver automatically loads skills from `user_skills/planning`.
- It routes the current user command by reading a concise semantic view of each available `SKILL.md` package, including metadata plus a content excerpt, and judging whether each skill's actual workflow semantics fit the request.
- Only the selected skills are then read in full during the resolution stage. This progressive disclosure keeps routing lightweight while still relying on model-based understanding instead of keyword matching alone.
- The selected skills are used to decide whether the system should ask for missing information or directly rewrite the request into one complete task instruction for the planner.
- The downstream planner then converts that resolved task instruction into executable task steps.

Recommended structure:
- Prefer package skills: `skill_name/SKILL.md`.
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
- `template_goal`
- `required_inputs`
- `planning_stages`
- `output_strategy`
- `content` / `guidance`

Optional metadata:
- `skill_type`: optional descriptive label only. It can be omitted. Routing and resolution should be driven by the skill content itself, not by this field.

Template-specific metadata:
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
