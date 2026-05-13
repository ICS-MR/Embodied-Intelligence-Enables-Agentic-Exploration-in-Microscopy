import logging
from typing import Dict, Optional, Tuple, Any

from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer
from openai import OpenAI
from agent.fiji_subprocess_executor import (
    FijiSubprocessError,
    run_generated_fiji_code_in_subprocess,
)
from agent.llm_fgen import LMPFGen
from agent.utils import (
    SAFE_BUILTIN_CALLS,
    call_openai_generic,
    exec_safe,
    extract_python_code,
    merge_dicts,
    remove_all_imports,
)
from utils.cli_logging import get_cli_logger


logger = get_cli_logger("EXECUTOR")

class ExperimentExecuteAgent:
    """Language Model Program (accepts external client instance)"""

    def __init__(
            self,
            name: str,
            cfg: Dict,
            cfg_fgen: Dict,
            fixed_vars: Dict,
            variable_vars: Dict,
            client: OpenAI,  # New: Accept global client instance passed from main program
            _historyManager=None,
            clarify_tag: bool = False,
            execution_context: Optional[Dict[str, Any]] = None,
    ):
        self._name = name
        self._cfg = cfg or {}
        self._base_prompt = self._cfg.get('prompt_text', "")
        self._historyManager = _historyManager
        self.clarify_tag = clarify_tag
        self.code_history = ''
        self._lmp_fgen = LMPFGen(cfg_fgen, fixed_vars, variable_vars, client)
        self._fixed_vars = fixed_vars or {}
        self._variable_vars = variable_vars or {}
        self._client = client  # Cache: Save global client instance
        self._execution_context = execution_context or {}

    def clear_exec_hist(self):
        """Clear code execution history"""
        self.code_history = ''

    def build_prompt(self, query: str, context: str = '') -> Tuple[str, str, str]:
        """Construct LLM prompt"""
        prompt_parts = []
        # Session history
        if self._cfg.get('maintain_session') and self.code_history.strip():
            prompt_parts.append(self.code_history)
        # Context
        if context.strip():
            prompt_parts.append(context)
        # Query
        query = f"{self._cfg.get('query_prefix', '')}{query}{self._cfg.get('query_suffix', '')}"
        prompt_parts.append(f"{query}\n# Generate runnable Python code without markdown")
        return self._base_prompt, "\n".join(prompt_parts), query

    def run(self, query: str, context: str = '', **kwargs) -> Optional[Dict[str, int]]:
        base_prompt, prompt, use_query = self.build_prompt(query, context)

        # Core change 1: disable streaming and request a complete response.
        response = call_openai_generic(
            client=self._client,  # Reuse the shared client instance.
            prompt=prompt,
            system_prompt=base_prompt,
            model=self._cfg.get('engine', 'gpt-3.5-turbo'),
            temperature=self._cfg.get('temperature', 0.7),
            stop_tokens=list(self._cfg.get('stop', [])),
            stream=False  # Disable streaming so we receive the full response at once.
        )

        # Core change 2: parse the complete response in one pass.
        reasoning_content, answer_content = "", ""
        usage_dict = None

        # Parse the full response; the non-streaming shape differs from streaming.
        if hasattr(response, 'choices') and response.choices:
            # Extract the full answer content.
            answer_content = response.choices[0].message.content or ""

        # Extract token usage; non-streaming responses expose usage at the root.
        if hasattr(response, 'usage') and response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        # Keep the original flow: extract code, log it, and execute it.
        # Extract and sanitize the generated code.
        code_str = extract_python_code(answer_content)
        code_str = remove_all_imports(code_str)
        to_log = f"{use_query}\n# Reasoning:\n{reasoning_content}\n# Code:\n{code_str}"

        # Log the generated code in one block instead of streaming it.
        logger.info("Generated code:\n%s", highlight(code_str, PythonLexer(), TerminalFormatter()).rstrip())

        if self._historyManager:
            self._historyManager.record_interaction(
                agent_name=self._name,
                event_type="executor_generated_code",
                message="Executor generated code for the current task step.",
                payload={
                    "query": query,
                    "formatted_query": use_query,
                    "context": context,
                    "prompt": prompt,
                    "raw_response": answer_content,
                    "generated_code": code_str,
                    "usage": usage_dict or {},
                },
            )

        if not self._cfg.get('debug_mode'):
            try:
                logger.info("Executing generated code")
                execution_payload: Dict[str, Any] = {}
                if self._should_use_fiji_subprocess():
                    execution_payload = self._run_fiji_subprocess(code_str)
                else:
                    self._run_in_process(code_str, kwargs)
                if self._cfg.get('maintain_session'):
                    self._variable_vars.update(kwargs)
                self.code_history += f"\n{code_str}"
                if self._historyManager:
                    self._historyManager.record_interaction(
                        agent_name=self._name,
                        event_type="executor_execution_succeeded",
                        message="Executor ran the generated code successfully.",
                        payload={"query": query, "generated_code": code_str, **execution_payload},
                    )
            except Exception as e:
                logger.error("Code execution failed: %s", e)
                if self._historyManager:
                    failure_payload = {"query": query, "generated_code": code_str, "error": str(e)}
                    if isinstance(e, FijiSubprocessError):
                        failure_payload.update(e.payload)
                    self._historyManager.record_interaction(
                        agent_name=self._name,
                        event_type="executor_execution_failed",
                        message="Executor failed while running generated code.",
                        payload=failure_payload,
                    )
                if isinstance(e, FijiSubprocessError):
                    raise

        return usage_dict

    def _should_use_fiji_subprocess(self) -> bool:
        return bool(self._execution_context.get("use_fiji_subprocess"))

    def _run_in_process(self, code_str: str, kwargs: Dict[str, Any]) -> None:
        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        allowed_call_names = {
            name for name, value in gvars.items()
            if callable(value)
        } | SAFE_BUILTIN_CALLS
        allowed_attribute_roots = set(gvars.keys()) | set(kwargs.keys())
        exec_safe(
            code_str,
            gvars,
            kwargs,
            allowed_call_names=allowed_call_names,
            allowed_attribute_roots=allowed_attribute_roots,
            timeout_seconds=float(self._execution_context.get("timeout_seconds") or 0) or None,
        )

    def _run_fiji_subprocess(self, code_str: str) -> Dict[str, Any]:
        timeout_seconds = float(self._execution_context.get("timeout_seconds") or 300.0)
        artifact_emitter_getter = self._execution_context.get("artifact_emitter_getter")
        artifact_emitter = artifact_emitter_getter() if callable(artifact_emitter_getter) else None
        result = run_generated_fiji_code_in_subprocess(
            code_str,
            session_dir=self._execution_context["session_dir"],
            output_dir=self._execution_context["output_dir"],
            timeout_seconds=timeout_seconds,
            storage_manager=self._execution_context.get("storage_manager"),
            say_capture=self._execution_context.get("say_capture"),
            artifact_emitter=artifact_emitter,
            workdir=self._execution_context.get("workdir") or self._execution_context.get("output_dir"),
            max_startup_retries=int(self._execution_context.get("startup_retry_times") or 2),
            startup_retry_backoff_seconds=float(
                self._execution_context.get("startup_retry_backoff_seconds") or 2.0
            ),
        )
        return {
            "execution_backend": "fiji_subprocess",
            "fiji_subprocess": result.payload(),
        }
    
