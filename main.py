import logging
import os
import multiprocessing as mp
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable, Optional
import time

os.environ["PYMM_LOG_FILE"] = "0"
os.environ["BFIO_LOG_TO_FILE"] = "0"

from bootstrap.config import load_runtime_settings
from services.task_orchestrator import TaskPlan, TaskRequest
from utils.cli_logging import configure_cli_logging, get_cli_logger
from utils.interaction_flow import (
    combine_clarification_context,
    interpret_clarification_feedback,
    interpret_plan_feedback,
    is_debug_plan_request,
    pick_text,
    prefers_chinese,
)
from utils.preview_process import PreviewProcessManager
from utils.runtime_core import initialize_system_components, release_resources, setup_microscope
from utils.runtime_text import format_raw_planner_debug


NOISY_LOGGERS = [
    "pymmcore_plus",
    "bfio",
    "bfio.backends",
    "jpype",
    "scyjava",
    "openai",
    "httpx",
    "httpcore",
    "urllib3",
]

EXIT_KEYWORDS = {"exit", "quit", "bye"}
ROOT_DIR = Path(__file__).resolve().parent
CLI_LOG_PATH = ROOT_DIR / "logs" / "cli_runtime.log"
CLI_STDOUT = sys.stdout
CLI_STDIN = sys.stdin

class Color:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def silence_external_console_noise() -> None:
    """Keep third-party technical logs out of the interactive CLI prompt."""
    logging.captureWarnings(True)
    for name in NOISY_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.ERROR)


def show_eims_welcome_logo() -> None:
    """Display the static EIMS welcome banner."""
    # Optional: clear the terminal first (leave commented to preserve existing output)
    # print("\033c", end="")

    # EIMS ASCII banner (best viewed with a monospace font)
    eims_logo = f"""
{Color.BOLD}{Color.CYAN}
                ███████╗ ██╗ ███╗   ███╗ ███████╗
                ██╔════╝ ██║ ████╗ ████║ ██╔════╝
                █████╗   ██║ ██╔████╔██║ ███████╗
                ██╔══╝   ██║ ██║╚██╔╝██║ ╚════██║
                ███████╗ ██║ ██║ ╚═╝ ██║ ███████║
                ╚══════╝ ╚═╝ ╚═╝     ╚═╝ ╚══════╝
                                                                        
                          EIMS System
                     ======================{Color.RESET}
                      
{Color.GREEN}✅ EIMS initialized successfully!{Color.RESET}
{Color.WHITE}📌 Waiting for your commands (type 'exit' to quit){Color.RESET}
"""
    # Print the static banner directly
    cli_print(eims_logo)

configure_cli_logging(logging.INFO, log_path=CLI_LOG_PATH, console_level=None)
silence_external_console_noise()
system_logger = get_cli_logger("SYSTEM")
planner_logger = get_cli_logger("PLANNER")


def cli_print(text: str = "", *, end: str = "\n", flush: bool = True) -> None:
    CLI_STDOUT.write(f"{text}{end}")
    if flush:
        CLI_STDOUT.flush()


@contextmanager
def capture_technical_output():
    CLI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLI_LOG_PATH.open("a", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            yield


def cli_input(prompt: str) -> str:
    cli_print(prompt, end="")
    with capture_technical_output():
        return CLI_STDIN.readline().strip()


def print_divider() -> None:
    cli_print("-" * 72)


def print_scopebot_message(text: str) -> None:
    if text:
        cli_print(f"Scopebot: {text}")


def stream_scopebot_message(producer: Callable[[Callable[[str], None]], str]) -> str:
    chunks: list[str] = []
    CLI_STDOUT.write("Scopebot: ")
    CLI_STDOUT.flush()

    def on_delta(delta: str) -> None:
        if not delta:
            return
        chunks.append(delta)
        CLI_STDOUT.write(delta)
        CLI_STDOUT.flush()

    text = producer(on_delta) or ""
    if text and not chunks:
        CLI_STDOUT.write(text)
        CLI_STDOUT.flush()
    CLI_STDOUT.write("\n")
    CLI_STDOUT.flush()
    return text or "".join(chunks)


def emit_robot_action(text: str) -> None:
    if text:
        cli_print(f"Action: {text}")


def emit_step_summary(text: str) -> None:
    if text:
        cli_print(f"Scopebot: {text}")


def emit_checker_warning(text: str) -> None:
    if text:
        cli_print(f"Scopebot: {text}")


def show_cli_interaction_artifact(artifact: dict) -> None:
    if str(artifact.get("kind") or "") != "image":
        return

    artifact_path = str(artifact.get("path") or "").strip()
    if not artifact_path:
        return

    title = str(artifact.get("title") or "Fiji Detection Result")
    notice = str(artifact.get("text") or f"Displaying annotated image: {os.path.basename(artifact_path)}")
    display_seconds = max(0.0, float(artifact.get("display_seconds") or 3.0))
    print_scopebot_message(notice)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print_scopebot_message(f"Matplotlib GUI is unavailable, cannot display image window: {exc}")
        print_scopebot_message(f"Annotated image saved at: {artifact_path}")
        return

    try:
        image = plt.imread(artifact_path)
    except Exception as exc:
        print_scopebot_message(f"Failed to load annotated image: {artifact_path}. error={exc}")
        return

    figure = None
    try:
        figure, axis = plt.subplots(num=title)
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
        plt.show(block=False)
        started_at = time.monotonic()
        while plt.fignum_exists(figure.number):
            plt.pause(0.1)
            if display_seconds > 0 and (time.monotonic() - started_at) >= display_seconds:
                break
    except Exception as exc:
        print_scopebot_message(f"Failed to display annotated image window: {exc}")
        print_scopebot_message(f"Annotated image saved at: {artifact_path}")
    finally:
        if figure is not None:
            try:
                plt.close(figure)
            except Exception:
                pass


def print_cli_skill_routing_summary(runtime_context, plan: TaskPlan, *, prefers_zh: bool) -> None:
    task_manager = getattr(runtime_context, "task_manager", None)
    if task_manager is None or not getattr(task_manager, "_skill_enabled", False):
        return

    selected_skills = [str(item).strip() for item in getattr(plan, "selected_skills", []) if str(item).strip()]
    if not selected_skills:
        return

    reason = str(getattr(plan, "skill_reason", "") or "").strip()
    skill_text = ", ".join(selected_skills)
    if prefers_zh:
        message = f"This planning round will use these skills: {skill_text}."
        if reason:
            message += f" Reason: {reason}"
    else:
        message = f"This planning round will use these skills: {skill_text}."
        if reason:
            message += f" Reason: {reason}"
    print_scopebot_message(message)


def print_cli_raw_planner_debug(plan: TaskPlan, *, prefers_zh: bool) -> None:
    print_scopebot_message(format_raw_planner_debug(plan, prefers_zh=prefers_zh))


def _record_cli_user_input(
    runtime_context,
    text: str,
    *,
    input_kind: str,
    prompt_text: str = "",
    command_snapshot: str = "",
) -> None:
    runtime_context.history_manager.record_user_input(
        text,
        source="cli",
        input_kind=input_kind,
        prompt_text=prompt_text,
        prompt_mode="plan_confirmation" if prompt_text else "",
        command_snapshot=command_snapshot,
    )

def request_plan_confirmation(runtime_context, original_command: str) -> Optional[TaskPlan]:
    current_command = original_command.strip()
    revisions: list[str] = []
    clarification_entries: list[str] = []
    prefers_zh = prefers_chinese(original_command)

    while True:
        with capture_technical_output():
            plan = runtime_context.task_orchestrator.plan(
                TaskRequest(
                    user_command=current_command,
                    human_mode=True,
                    planner_context=combine_clarification_context(clarification_entries),
                )
            )
        if plan.tokens:
            planner_logger.info("Planner tokens: %s", plan.tokens)
        print_cli_skill_routing_summary(runtime_context, plan, prefers_zh=prefers_zh)

        if plan.ready and plan.steps:
            stream_scopebot_message(
                lambda on_delta: runtime_context.task_orchestrator.stream_plan_preview(plan, on_delta)
            )
            reply = cli_input(
                pick_text(
                    prefers_zh,
                    "Reply with 'confirm' or 'continue' to execute, 'cancel' to stop, type 'debug_plan' to inspect the raw planner output, or type a revision:\nYou: ",
                    "Reply with 'confirm' or 'continue' to execute, 'cancel' to stop, type 'debug_plan' to inspect the raw planner output, or type a revision:\nYou: ",
                )
            )
            _record_cli_user_input(
                runtime_context,
                reply,
                input_kind="plan_feedback",
                prompt_text="plan_ready_confirmation",
                command_snapshot=current_command,
            )
            if is_debug_plan_request(reply):
                print_cli_raw_planner_debug(plan, prefers_zh=prefers_zh)
                continue
            decision = interpret_plan_feedback(
                reply,
                plan_ready=True,
                original_command=original_command,
                revisions=revisions,
            )
            if decision.action == "confirm":
                return plan
            if decision.action == "cancel":
                print_scopebot_message(
                    pick_text(
                        prefers_zh,
                        "Okay, I will not execute this plan for now. You can revise it and ask me again later.",
                        "Okay, I will not execute this plan for now. You can revise it and ask me again later.",
                    )
                )
                return None

            if decision.action == "empty":
                print_scopebot_message(
                    pick_text(
                        prefers_zh,
                        "I have not received any revision yet. You can confirm, cancel, or type a revision directly.",
                        "I have not received any revision yet. You can confirm, cancel, or type a revision directly.",
                    )
                )
                continue

            revisions = decision.revisions
            current_command = decision.current_command
            print_scopebot_message(
                pick_text(
                    prefers_zh,
                    "Received. I will reorganize the plan based on your revision.",
                    "Received. I will reorganize the plan based on your revision.",
                )
            )
            print_divider()
            continue

        if getattr(plan, "status", "") == "ask_user":
            prompt_text = str(plan.question or "").strip() or pick_text(
                prefers_zh,
                "I need one key detail before I can continue planning.",
                "I need one key detail before I can continue planning.",
            )
            print_scopebot_message(prompt_text)
            reply = cli_input(
                pick_text(
                    prefers_zh,
                    "Answer the question, type 'debug_plan' to inspect the raw planner output, or type 'cancel':\nYou: ",
                    "Answer the question, type 'debug_plan' to inspect the raw planner output, or type 'cancel':\nYou: ",
                )
            )
            _record_cli_user_input(
                runtime_context,
                reply,
                input_kind="plan_feedback",
                prompt_text=prompt_text,
                command_snapshot=current_command,
            )
            if is_debug_plan_request(reply):
                print_cli_raw_planner_debug(plan, prefers_zh=prefers_zh)
                continue
            clarification_decision = interpret_clarification_feedback(
                reply,
                entries=clarification_entries,
                planner_question=prompt_text,
            )
            if clarification_decision.action == "cancel":
                print_scopebot_message(
                    pick_text(
                        prefers_zh,
                        "Okay, I will pause this task for now.",
                        "Okay, I will pause this task for now.",
                    )
                )
                return None
            if clarification_decision.action == "confirm_without_plan":
                print_scopebot_message(
                    pick_text(
                        prefers_zh,
                        "I still do not have an executable plan yet, so I cannot start. Please answer the question first.",
                        "I still do not have an executable plan yet, so I cannot start. Please answer the question first.",
                    )
                )
                continue
            if clarification_decision.action == "empty":
                print_scopebot_message(
                    pick_text(
                        prefers_zh,
                        "I have not received any new detail yet. Please answer the question first or type 'cancel'.",
                        "I have not received any new detail yet. Please answer the question first or type 'cancel'.",
                    )
                )
                continue

            clarification_entries = clarification_decision.entries
            print_scopebot_message(
                pick_text(
                    prefers_zh,
                    "Received. I will replan using that resolved workflow detail.",
                    "Received. I will replan with that new detail.",
                )
            )
            print_divider()
            continue

        if getattr(plan, "status", "") == "unsupported":
            print_scopebot_message(
                pick_text(
                    prefers_zh,
                    "The current system cannot execute this request. Here is the original planner output:",
                    "The current system cannot execute this request. Here is the original planner output:",
                )
            )
            print_scopebot_message(plan.planner_raw_response or plan.error or "Unsupported request.")
            return None

        print_scopebot_message(
            pick_text(
                prefers_zh,
                "I still cannot turn this into an executable plan. You can add more detail or type 'cancel'.",
                "I still cannot turn this into an executable plan. You can add more detail or type 'cancel'.",
            )
        )
        reply = cli_input(
            pick_text(
                prefers_zh,
                "Please add a revision, type 'debug_plan' to inspect the raw planner output, or type 'cancel':\nYou: ",
                "Please add a revision, type 'debug_plan' to inspect the raw planner output, or type 'cancel':\nYou: ",
            )
        )
        _record_cli_user_input(
            runtime_context,
            reply,
            input_kind="plan_feedback",
            prompt_text="plan_revision_request",
            command_snapshot=current_command,
        )
        if is_debug_plan_request(reply):
            print_cli_raw_planner_debug(plan, prefers_zh=prefers_zh)
            continue
        decision = interpret_plan_feedback(
            reply,
            plan_ready=False,
            original_command=original_command,
            revisions=revisions,
        )
        if decision.action == "cancel":
            print_scopebot_message(
                pick_text(
                    prefers_zh,
                    "Okay, I will pause this task for now.",
                    "Okay, I will pause this task for now.",
                )
            )
            return None
        if decision.action == "confirm_without_plan":
            print_scopebot_message(
                pick_text(
                    prefers_zh,
                    "I still do not have an executable updated plan, so I cannot start yet. Please keep revising it.",
                    "I still do not have an executable updated plan, so I cannot start yet. Please keep revising it.",
                )
            )
            continue
        if decision.action == "empty":
            continue
        revisions = decision.revisions
        current_command = decision.current_command
        print_scopebot_message(
            pick_text(
                prefers_zh,
                "Received. I will keep replanning.",
                "Received. I will keep replanning.",
            )
        )
        print_divider()


def main() -> None:
    show_eims_welcome_logo()
    cli_print(f"Technical logs: {CLI_LOG_PATH}")
    settings = load_runtime_settings()
    simulation_mode = bool(settings.model.Simulation_mode)
    with capture_technical_output():
        runtime_context = initialize_system_components(settings.model.Simulation_mode)
    preview_manager: Optional[PreviewProcessManager] = None

    try:
        if hasattr(runtime_context.env_imagej, "set_interaction_artifact_listener"):
            runtime_context.env_imagej.set_interaction_artifact_listener(show_cli_interaction_artifact)
        with capture_technical_output():
            setup_microscope(runtime_context.env_olympus, settings.startup)
        if simulation_mode:
            system_logger.info("Simulation mode detected; skipping local preview window startup.")
        else:
            if bool(getattr(settings.startup, "start_preview", True)) and hasattr(runtime_context.env_olympus, "start_preview"):
                try:
                    with capture_technical_output():
                        runtime_context.env_olympus.start_preview()
                except Exception as exc:
                    system_logger.warning("Failed to start microscope preview acquisition: %s", exc)
            try:
                preview_manager = PreviewProcessManager(
                    runtime_context.env_olympus.get_live_preview_image,
                    window_name=getattr(runtime_context.env_olympus, "preview_window_name", "micro live"),
                )
                with capture_technical_output():
                    preview_manager.start()
            except Exception as exc:
                preview_manager = None
                system_logger.warning("Local preview window is unavailable: %s", exc)
        system_logger.info("System initialized successfully")
        print_scopebot_message(
            "Microscope is ready. Tell me what you want to do, and I will show a plan before execution."
        )

        while True:
            print_divider()
            user_command = cli_input("You: ")
            if not user_command:
                continue
            _record_cli_user_input(
                runtime_context,
                user_command,
                input_kind="initial_command",
                command_snapshot=user_command.strip(),
            )
            if user_command.lower() in EXIT_KEYWORDS or user_command in EXIT_KEYWORDS:
                print_scopebot_message("Session closed. Come back anytime when you want to run another task.")
                break

            plan = request_plan_confirmation(runtime_context, user_command)
            if plan is None:
                continue

            print_scopebot_message(
                pick_text(
                    prefers_chinese(user_command),
                    "Confirmation received. I am starting execution now.",
                    "Confirmation received. I am starting execution now.",
                )
            )

            with capture_technical_output():
                result = runtime_context.task_orchestrator.execute(
                    plan,
                    on_robot_action=emit_robot_action,
                    on_step_summary=emit_step_summary,
                    on_checker_warning=emit_checker_warning,
                    summarize_completion=False,
                )
            if not result.success:
                system_logger.error("Task failed. retry_times=%s error=%s", result.retry_times, result.error)
                print_scopebot_message(result.summary or "Task execution failed.")
                continue

            stream_scopebot_message(
                lambda on_delta: runtime_context.task_orchestrator.stream_task_summary(plan, on_delta, steps=result.steps)
            )
    except KeyboardInterrupt:
        system_logger.info("User interrupted the session")
    finally:
        if preview_manager is not None:
            try:
                with capture_technical_output():
                    preview_manager.stop()
            except Exception as exc:
                system_logger.warning("Failed to stop preview process cleanly: %s", exc)
        if hasattr(runtime_context.env_imagej, "set_interaction_artifact_listener"):
            runtime_context.env_imagej.set_interaction_artifact_listener(None)
        with capture_technical_output():
            release_resources(runtime_context)


if __name__ == "__main__":
    mp.freeze_support()
    main()











