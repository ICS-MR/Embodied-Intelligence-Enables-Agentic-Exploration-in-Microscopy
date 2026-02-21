
import ast
from typing import Dict, Optional, Any

from openai import OpenAI

from agent.utils import (
    extract_python_code,
    remove_all_imports,
    exec_safe,
    merge_dicts,
    var_exists,
    validate_param_type,
    call_openai_generic
)
from agent.ast_parser import FunctionParser

class LMPFGen:
    """LLM-based Function Generator (accepts external client instance)"""
    def __init__(
        self,
        cfg: Dict,
        fixed_vars: Dict,
        variable_vars: Dict,
        client: OpenAI  # New: Accept global client instance passed from main program
    ):
        if not validate_param_type(cfg, dict, "cfg"):
            raise TypeError("cfg must be a dictionary")
        self._cfg = cfg
        self._stop_tokens = list(cfg.get('stop', []))
        self._fixed_vars = fixed_vars or {}
        self._variable_vars = variable_vars or {}
        self._base_prompt = cfg.get('prompt_text', "")
        self._client = client  # Cache: Save global client instance for subsequent reuse

    def create_f_from_sig(
        self,
        f_name: str,
        f_sig: str,
        other_vars: Optional[Dict] = None,
        return_src: bool = False
    ) -> Any:
        # Construct prompt (unchanged)
        query = f"{self._cfg.get('query_prefix', '')}{f_sig}{self._cfg.get('query_suffix', '')}"
        prompt = f"{self._base_prompt}\n{query}"

        # Call generic function: Pass cached client instance (reuse)
        response = call_openai_generic(
            client=self._client,  # Reuse global client passed from main program
            prompt=prompt,
            model=self._cfg.get('engine', 'gpt-3.5-turbo'),
            temperature=self._cfg.get('temperature', 0.7),
            stop_tokens=self._stop_tokens,
            stream=True
        )

        # Subsequent logic (extract code, safe execution, etc. - unchanged)
        answer_content = "".join([chunk.choices[0].delta.content or "" for chunk in response])
        f_src = extract_python_code(answer_content)
        f_src = remove_all_imports(f_src)


        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars or {}])
        lvars = {}
        exec_safe(f_src, gvars, lvars)

        if f_name not in lvars:
            raise RuntimeError(f"Function {f_name} generation failed")

        return (lvars[f_name], f_src) if return_src else lvars[f_name]

    def create_new_fs_from_code(
        self,
        code_str: str,
        other_vars: Optional[Dict] = None,
        return_src: bool = False
    ) -> Any:
        """Extract function calls from code and recursively generate undefined functions"""
        if not code_str.strip():
            return ({}, {}) if return_src else {}

        # Extract function calls and assignments
        fs, f_assigns = {}, {}
        try:
            FunctionParser(fs, f_assigns).visit(ast.parse(code_str))
        except SyntaxError as e:

            return ({}, {}) if return_src else {}

        # Prioritize assignments as signatures (more complete)
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        new_fs, srcs = {}, {}
        all_vars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars or {}])

        # Generate undefined functions
        for f_name, f_sig in fs.items():
            if var_exists(f_name, merge_dicts([all_vars, new_fs])):
                continue

            # Generate function (executed and returns function object)
            f, f_src = self.create_f_from_sig(f_name, f_sig, new_fs, return_src=True)
            new_fs[f_name] = f
            srcs[f_name] = f_src

            # Recursively parse new function calls in function body
            try:
                tree = ast.parse(f_src)
                if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # ✅ Use standard library ast.unparse (Python 3.9+)
                    func_body = ast.unparse(tree.body[0].body).strip()
                    if func_body:  # Recurse only if function body is non-empty
                        child_fs, child_srcs = self.create_new_fs_from_code(
                            func_body,
                            merge_dicts([all_vars, new_fs]),
                            return_src=True
                        )
                        new_fs.update(child_fs)
                        srcs.update(child_srcs)
                        # Note: No repeated exec_safe, as child functions are executed in create_f_from_sig
            except Exception as e:
                print(f"Failed to recursively generate child functions: {e}")

        return (new_fs, srcs) if return_src else new_fs