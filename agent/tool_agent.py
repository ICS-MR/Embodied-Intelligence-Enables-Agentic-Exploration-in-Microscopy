
from typing import Dict, Optional, Tuple, Any

from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer
from openai import OpenAI
from agent.llm_fgen import LMPFGen
from agent.utils import (
    extract_python_code,
    remove_all_imports,
    exec_safe,
    merge_dicts,
    call_openai_generic
)


class LMP:
    """Language Model Program (accepts external client instance)"""
    def __init__(
        self,
        name: str,
        cfg: Dict,
        lmp_fgen: "LMPFGen",
        fixed_vars: Dict,
        variable_vars: Dict,
        client: OpenAI,  # New: Accept global client instance passed from main program
        _historyManager=None,
        clarify_tag: bool = False
    ):
        self._name = name
        self._cfg = cfg or {}
        self._base_prompt = self._cfg.get('prompt_text', "")
        self._historyManager = _historyManager
        self.clarify_tag = clarify_tag
        self.code_history = ''
        self._lmp_fgen = lmp_fgen
        self._fixed_vars = fixed_vars or {}
        self._variable_vars = variable_vars or {}
        self._client = client  # Cache: Save global client instance

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

    def __call__(self, query: str, context: str = '', **kwargs) -> Optional[Dict[str, int]]:
        base_prompt, prompt, use_query = self.build_prompt(query, context)

        # ========== 核心修改1：关闭流式输出（stream=False） ==========
        response = call_openai_generic(
            client=self._client,  # 复用全局客户端
            prompt=prompt,
            system_prompt=base_prompt,
            model=self._cfg.get('engine', 'gpt-3.5-turbo'),
            temperature=self._cfg.get('temperature', 0.7),
            stop_tokens=list(self._cfg.get('stop', [])),
            stream=False  # 关闭流式，获取完整响应
        )

        # ========== 核心修改2：解析完整响应，一次性获取内容 ==========
        reasoning_content, answer_content = "", ""
        usage_dict = None

        # 解析完整响应（非流式的响应结构与流式不同）
        if hasattr(response, 'choices') and response.choices:
            # 获取完整的回答内容
            answer_content = response.choices[0].message.content or ""
        
        # 解析用量信息（非流式的 usage 直接在 response 根节点）
        if hasattr(response, 'usage') and response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        # ========== 原有逻辑保留（提取代码、输出、执行） ==========
        # 提取并清理代码
        code_str = extract_python_code(answer_content)
        code_str = remove_all_imports(code_str)
        to_log = f"{use_query}\n# Reasoning:\n{reasoning_content}\n# Code:\n{code_str}"
        
        # 一次性输出代码（替代原流式输出）
        print("\n===== Executing Code =====")
        print(highlight(code_str, PythonLexer(), TerminalFormatter()))

        # 日志记录
        if self._historyManager:
            self._historyManager.append(to_log, self._name)
        
        # 执行代码（非调试模式下）
        if not self._cfg.get('debug_mode'):
            try:
                gvars = merge_dicts([self._fixed_vars, self._variable_vars])
                print("\n===== Execution Process =====")
                exec_safe(code_str, gvars, kwargs)
                if self._cfg.get('maintain_session'):
                    self._variable_vars.update(kwargs)
                self.code_history += f"\n{code_str}"
            except Exception as e:
                print(f"Code execution failed: {e}")

        return usage_dict