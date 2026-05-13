from typing import Any, Callable, Dict, List, Optional

from adapters.llm_clients import create_chat_completion, stream_chat_completion_text


def summarize_spoken_messages(client: Optional[Any], model_name: str, spoken_messages: List[str]) -> str:
    if not spoken_messages:
        return "(No spoken output)"
    if client is None:
        return " ".join(spoken_messages)

    messages_text = "\n".join(f"- {msg}" for msg in spoken_messages)
    prompt = (
        "Summarize the following robot spoken messages into one or two concise and coherent English sentences. "
        "Use third-person perspective and do not add any information beyond what is provided:\n"
        f"{messages_text}\n"
    )
    response = create_chat_completion(
        client,
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an objective observer tasked with summarizing the robot's verbal behavior."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def format_raw_planner_debug(plan: Any, *, prefers_zh: bool) -> str:
    planner_raw = str(getattr(plan, "planner_raw_response", "") or "").strip()
    skill_raw = str(getattr(plan, "skill_routing_raw_response", "") or "").strip()
    active_templates = list(getattr(plan, "active_templates", []) or [])

    if not planner_raw and not skill_raw and not active_templates:
        return "There is no raw planner output available yet."

    lines: List[str] = []
    lines.append("Here is the raw planning output for this round:")
    if active_templates:
        lines.append("[Active planning templates]")
        for item in active_templates:
            name = str(item.get("name") or "").strip()
            skill_type = str(item.get("skill_type") or "").strip()
            template_goal = str(item.get("template_goal") or "").strip()
            output_strategy = str(item.get("output_strategy") or "").strip()
            template_line = f"- {name or 'Unnamed template'}"
            if skill_type:
                template_line += f" | type={skill_type}"
            if output_strategy:
                template_line += f" | strategy={output_strategy}"
            if template_goal:
                template_line += f" | goal={template_goal}"
            lines.append(template_line)
    if skill_raw:
        lines.append("[Raw skill routing output]")
        lines.append(skill_raw)
    if planner_raw:
        lines.append("[Raw planner output]")
        lines.append(planner_raw)
    return "\n".join(lines)


def format_planner_failure_message(plan: Any, *, prefers_zh: bool) -> str:
    del prefers_zh
    planner_raw = str(getattr(plan, "planner_raw_response", "") or "").strip()
    skill_raw = str(getattr(plan, "skill_routing_raw_response", "") or "").strip()

    if planner_raw:
        return planner_raw
    if skill_raw:
        return skill_raw
    return "There is no raw planner output available yet."


def summarize_my_spoken_messages(client: Optional[Any], model_name: str, spoken_messages: List[str]) -> str:
    if not spoken_messages:
        return "(No spoken output)"
    if client is None:
        return " ".join(spoken_messages)

    messages_text = "\n".join(f"- {msg}" for msg in spoken_messages)
    prompt = (
        "Summarize the following spoken messages into one or two concise and coherent English sentences, "
        "as if you are the speaker describing your own actions or intentions. "
        "Use first-person perspective and do not add any information beyond what is provided:\n"
        f"{messages_text}\n"
    )
    response = create_chat_completion(
        client,
        model=model_name,
        messages=[
            {"role": "system", "content": "You are summarizing your own spoken messages from a first-person perspective."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def summarize_step_completion(
    client: Optional[Any],
    model_name: str,
    step: Dict[str, Any],
    spoken_messages: List[str],
) -> str:
    fallback = _build_step_summary_fallback(step, spoken_messages)
    if client is None:
        return fallback

    module_name = step.get("module", "Unknown")
    command = str(step.get("command", "")).strip() or "run this step"
    spoken_text = "\n".join(f"- {msg}" for msg in spoken_messages) if spoken_messages else "- No spoken output was captured."
    prefers_chinese = _prefers_chinese_response(command)

    prompt = (
        "You are a concise and reliable lab assistant. Summarize what this sub-agent just completed in one short sentence.\n"
        "Requirements:\n"
        "1. Use first-person voice.\n"
        "2. Do not invent results.\n"
        "3. Focus on the module used and the action completed.\n\n"
        f"Module: {module_name}\n"
        f"Step command: {command}\n"
        f"Spoken execution notes:\n{spoken_text}"
    )

    try:
        response = create_chat_completion(
            client,
            model=model_name,
            messages=[
                {"role": "system", "content": "You produce short execution updates for a microscopy workflow UI."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        content = response.choices[0].message.content.strip()
        return content or fallback
    except Exception:
        return fallback


def summarize_checker_success(client: Optional[Any], model_name: str, user_command: str) -> str:
    del client
    del model_name
    return _build_checker_success_fallback(user_command)


def summarize_checker_issue(
    client: Optional[Any],
    model_name: str,
    user_command: str,
    revised_steps: List[Dict[str, Any]],
    *,
    has_no_target_error: bool = False,
) -> str:
    fallback = _build_checker_warning_fallback(user_command, revised_steps, has_no_target_error=has_no_target_error)
    if client is None:
        return fallback

    serialized_steps = []
    for step in revised_steps:
        idx = step.get("subtask_index", "?")
        module = step.get("module", "Unknown")
        command = str(step.get("command", "")).strip() or "run this step"
        serialized_steps.append(f"{idx}. [{module}] {command}")

    prefers_chinese = _prefers_chinese_response(user_command)
    prompt = (
        "You are Scopebot in the frontend UI. Newly acquired images were checked and issues were found."
        " Write 1 or 2 short sentences telling the user that I found a problem, I will adjust the workflow, and I will retry.\n"
        "Requirements:\n"
        "1. Use first-person voice.\n"
        "2. Do not invent specific defect details.\n"
        "3. If revised steps are provided, briefly mention that I will retry with the updated plan.\n\n"
        f"Original command: {user_command}\n"
        f"Has no-target issue: {has_no_target_error}\n"
        f"Revised steps:\n{chr(10).join(serialized_steps) if serialized_steps else 'No revised steps were provided.'}"
    )

    try:
        response = create_chat_completion(
            client,
            model=model_name,
            messages=[
                {"role": "system", "content": "You explain execution retries clearly and calmly for users."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=160,
        )
        content = response.choices[0].message.content.strip()
        return content or fallback
    except Exception:
        return fallback


def summarize_task_execution(
    client: Optional[Any],
    model_name: str,
    user_command: str,
    lmp_steps: List[Dict[str, Any]],
) -> str:
    if not lmp_steps:
        return "No execution steps were generated."
    if client is None:
        return f"Executed {len(lmp_steps)} steps for command: {user_command}"

    serialized = []
    for step in lmp_steps:
        idx = step.get("subtask_index", "?")
        module = step.get("module", "Unknown")
        command = step.get("command", "").strip()
        serialized.append(f"{idx}. [{module}] {command}")

    prompt = (
        "You are an intelligent lab assistant. Generate a concise English summary of the task execution based on the following information:\n"
        f"User's original command:\n\"{user_command}\"\n"
        f"Actual execution steps:\n{chr(10).join(serialized)}\n"
        "Summarize the task's implementation process in one or two sentences using third-person, objective tone, "
        "and end with a gentle, guiding question about the next step."
    )

    try:
        response = create_chat_completion(
            client,
            model=model_name,
            messages=[
                {"role": "system", "content": "You excel at summarizing experimental workflows with concise and professional language."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"[Task summary failed: {exc}]"


def stream_task_execution_summary(
    client: Optional[Any],
    model_name: str,
    user_command: str,
    lmp_steps: List[Dict[str, Any]],
    on_delta: Callable[[str], None],
) -> str:
    if not lmp_steps:
        fallback = "No execution steps were generated."
        on_delta(fallback)
        return fallback

    messages, fallback = _build_task_execution_summary_prompt(user_command, lmp_steps)
    return _stream_text_via_callback(
        client,
        model_name=model_name,
        messages=messages,
        fallback_text=fallback,
        on_delta=on_delta,
        temperature=0.3,
        max_tokens=250,
    )


def stream_plan_preview_for_confirmation(
    client: Optional[Any],
    model_name: str,
    user_command: str,
    lmp_steps: List[Dict[str, Any]],
    on_delta: Callable[[str], None],
) -> str:
    prefers_chinese = _prefers_chinese_response(user_command)
    fallback_text = _build_plan_preview_fallback(user_command, lmp_steps)
    if not lmp_steps:
        on_delta(fallback_text)
        return fallback_text

    serialized_steps = []
    for step in lmp_steps:
        idx = step.get("subtask_index", "?")
        module = step.get("module", "Unknown")
        command = str(step.get("command", "")).strip() or "run this step"
        serialized_steps.append(f"{idx}. [{module}] {command}")

    prompt = (
        "You are Scopebot in the frontend UI. Rewrite the structured experiment plan below into a short, natural, user-facing preview before execution.\n"
        "Requirements:\n"
        "1. Preserve the key steps and their order.\n"
        "2. Do not invent extra parameters or outcomes.\n"
        "3. Keep it to 1 or 2 short sentences.\n"
        "4. Use first-person voice, such as 'My plan is to...'.\n"
        "5. Do not include the confirmation prompt itself.\n\n"
        f"User command: {user_command}\n"
        f"Structured plan:\n{chr(10).join(serialized_steps)}"
    )

    messages = [
        {"role": "system", "content": "You are Scopebot, a concise and reliable microscopy assistant."},
        {"role": "user", "content": prompt},
    ]
    return _stream_text_via_callback(
        client,
        model_name=model_name,
        messages=messages,
        fallback_text=fallback_text,
        on_delta=on_delta,
        temperature=0.2,
        max_tokens=220,
    )


def rewrite_task_plan_for_confirmation(
    client: Optional[Any],
    model_name: str,
    user_command: str,
    lmp_steps: List[Dict[str, Any]],
) -> str:
    prefers_chinese = _prefers_chinese_response(user_command)
    fallback_text = _build_plan_preview_fallback(user_command, lmp_steps)
    if not lmp_steps or client is None:
        return fallback_text

    serialized_steps = []
    for step in lmp_steps:
        idx = step.get("subtask_index", "?")
        module = step.get("module", "Unknown")
        command = str(step.get("command", "")).strip() or "run this step"
        serialized_steps.append(f"{idx}. [{module}] {command}")

    prompt = (
        "You are Scopebot in the frontend UI. Rewrite the structured experiment plan below into a short, natural, user-facing preview before execution.\n"
        "Requirements:\n"
        "1. Preserve the key steps and their order.\n"
        "2. Do not invent extra parameters or outcomes.\n"
        "3. Keep it to 1 or 2 short sentences.\n"
        "4. Use first-person voice, such as 'My plan is to...'.\n"
        "5. Do not include the confirmation prompt itself.\n\n"
        f"User command: {user_command}\n"
        f"Structured plan:\n{chr(10).join(serialized_steps)}"
    )

    try:
        response = create_chat_completion(
            client,
            model=model_name,
            messages=[
                {"role": "system", "content": "You are Scopebot, a concise and reliable microscopy assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        content = response.choices[0].message.content.strip()
        return content or fallback_text
    except Exception:
        return fallback_text


def _build_task_execution_summary_prompt(user_command: str, lmp_steps: List[Dict[str, Any]]) -> tuple[list[dict[str, str]], str]:
    serialized = []
    for step in lmp_steps:
        idx = step.get("subtask_index", "?")
        module = step.get("module", "Unknown")
        command = step.get("command", "").strip()
        serialized.append(f"{idx}. [{module}] {command}")

    prompt = (
        "You are an intelligent lab assistant. Generate a concise English summary of the task execution based on the following information:\n"
        f"User's original command:\n\"{user_command}\"\n"
        f"Actual execution steps:\n{chr(10).join(serialized)}\n"
        "Summarize the task's implementation process in one or two sentences using third-person, objective tone, "
        "and end with a gentle, guiding question about the next step."
    )
    messages = [
        {"role": "system", "content": "You excel at summarizing experimental workflows with concise and professional language."},
        {"role": "user", "content": prompt},
    ]
    fallback = f"Executed {len(lmp_steps)} steps for command: {user_command}"
    return messages, fallback


def _build_step_summary_fallback(step: Dict[str, Any], spoken_messages: List[str]) -> str:
    module_name = step.get("module", "Unknown")
    command = str(step.get("command", "")).strip()
    prefers_chinese = _prefers_chinese_response(command or module_name)
    if spoken_messages:
        joined_messages = " ".join(msg.strip() for msg in spoken_messages if str(msg).strip())
        if joined_messages:
            return joined_messages
    if prefers_chinese:
        return f"I completed the {module_name} step: {command or 'run this step'}."
    return f"I completed the {module_name} step: {command or 'run this step'}."


def _build_checker_success_fallback(user_command: str) -> str:
    prefers_chinese = _prefers_chinese_response(user_command)
    if prefers_chinese:
        return "I checked the newly acquired images and the current results look good, so I will continue the task."
    return "I checked the newly acquired images and the current results look good, so I will continue the task."


def _build_checker_warning_fallback(
    user_command: str,
    revised_steps: List[Dict[str, Any]],
    *,
    has_no_target_error: bool = False,
) -> str:
    prefers_chinese = _prefers_chinese_response(user_command)
    revised_step_count = len(revised_steps)
    if prefers_chinese:
        if has_no_target_error:
            return (
                "I checked the newly acquired images and found issues, including missing-target results. "
                f"I will adjust the workflow and retry with the updated {revised_step_count}-step plan."
            )
        return f"I checked the newly acquired images and found issues, so I will retry with the updated {revised_step_count}-step plan."
    if has_no_target_error:
        return (
            "I checked the newly acquired images and found issues, including missing-target results. "
            f"I will adjust the workflow and retry with the updated {revised_step_count}-step plan."
        )
    return f"I checked the newly acquired images and found issues, so I will retry with the updated {revised_step_count}-step plan."


def _stream_text_via_callback(
    client: Optional[Any],
    *,
    model_name: str,
    messages: list[dict[str, str]],
    fallback_text: str,
    on_delta: Callable[[str], None],
    temperature: float,
    max_tokens: int,
) -> str:
    if client is None:
        if fallback_text:
            on_delta(fallback_text)
        return fallback_text

    chunks: list[str] = []
    try:
        for delta in stream_chat_completion_text(
            client,
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            chunks.append(delta)
            on_delta(delta)
    except Exception:
        if fallback_text:
            on_delta(fallback_text)
        return fallback_text

    combined = "".join(chunks).strip()
    if combined:
        return combined
    if fallback_text:
        on_delta(fallback_text)
    return fallback_text


def _prefers_chinese_response(text: str) -> bool:
    del text
    return False


def _build_plan_preview_fallback(user_command: str, lmp_steps: List[Dict[str, Any]]) -> str:
    prefers_chinese = _prefers_chinese_response(user_command)
    if not lmp_steps:
        return "I do not have an executable plan to show yet."

    fragments = []
    for step in lmp_steps[:3]:
        module = step.get("module", "Unknown")
        command = str(step.get("command", "")).strip() or "run this step"
        fragments.append(f"[{module}] {command}")

    if prefers_chinese:
        preview = ", then ".join(fragments)
        if len(lmp_steps) > len(fragments):
            preview = f"{preview}, and then continue with the remaining {len(lmp_steps) - len(fragments)} step(s)"
        return f"My plan is to work through these steps in order: {preview}."

    preview = ", then ".join(fragments)
    if len(lmp_steps) > len(fragments):
        preview = f"{preview}, and then continue with the remaining {len(lmp_steps) - len(fragments)} step(s)"
    return f"My plan is to work through these steps in order: {preview}."
