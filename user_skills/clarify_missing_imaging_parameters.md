---
name: Clarify Missing Imaging Parameters
skill_type: planning_template
description: Ask one critical question first when an imaging task is blocked by missing parameters, then produce the final plan
tags:
- planning
- clarification
- imaging
triggers:
- unclear imaging request
- missing channel
- missing magnification
template_goal: resolve underspecified imaging tasks
required_inputs:
- fluorescence_state
- magnification
output_strategy: single_question_then_plan
planning_stages:
- assess_missing_info
- ask_one_blocking_question
- integrate_user_answer
- produce_final_plan
examples:
- capture fluorescence images of cells
- acquire images of the sample
priority: 5
---

Use this template when the request cannot become executable until one blocking imaging parameter is clarified.
Ask exactly one concise question.
After receiving the answer, integrate it and return a final executable plan.
