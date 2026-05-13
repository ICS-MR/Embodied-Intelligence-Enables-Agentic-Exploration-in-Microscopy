# Custom Skill Guide

In the current project architecture, a skill is a **directory-based knowledge package** that the skill resolver reads before planning.

Primary layout:

- `user_skills/planning/<skill_name>/SKILL.md`

The resolver now works in two stages:

1. Routing reads a concise semantic view of each `SKILL.md` package.
2. Resolution reads the selected skill packages in full, then either:
   - asks one blocking clarification question, or
   - produces one complete natural-language task instruction for the downstream planner.

## Recommended Structure

Use one workflow per package:

```text
user_skills/planning/
  your-skill-name/
    SKILL.md
```

`SKILL.md` should contain:

- frontmatter metadata for routing hints
- workflow interpretation rules
- clarification policy
- resolved task instruction style or example

## Recommended Metadata

Common frontmatter fields:

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

Optional metadata:

- `skill_type`
  This is optional and descriptive only. Do not rely on it as the main behavior switch.

Useful `output_strategy` values:

- `direct_plan`
- `single_question_then_plan`

## What Makes A Good Skill

Recommended practices:

- keep each skill focused on one workflow
- make `description` and `examples` realistic
- state clearly what counts as missing blocking information
- state clearly what the final resolved instruction should look like
- include a resolved instruction example if output style matters

Avoid:

- mixing multiple unrelated workflows into one skill
- relying only on trigger phrases
- leaving the final output style unspecified
- assuming default parameters without saying so explicitly

## Minimal Package Example

```md
---
name: Clarify Missing Imaging Parameters
description: Ask one blocking question before final planning when critical imaging parameters are missing
tags:
- planning
- clarification
- imaging
template_goal: resolve underspecified imaging tasks into a final executable instruction
required_inputs:
- fluorescence_state
- magnification
output_strategy: single_question_then_plan
priority: 5
---

Use this skill when the request cannot become executable until one blocking imaging parameter is clarified.

Resolver rules:
- Ask exactly one concise blocking question.
- Ask only for missing information that materially changes execution.
- If the request becomes sufficiently specified, rewrite it into one complete task instruction for the downstream planner.
```

## How To Test A Skill

1. Place the package under `user_skills/planning/`
2. Submit a task request that should match the workflow
3. Inspect `history/.../agent_interactions.json`

Look for:

- `skill_routing`
- `skill_resolution`
- the final planner prompt

## Naming

Keep the directory name and skill name semantically aligned, for example:

- `brightfield-focus-workflow`
- `multichannel-preview-scan`
- `mitosis-multichannel-tracking`
