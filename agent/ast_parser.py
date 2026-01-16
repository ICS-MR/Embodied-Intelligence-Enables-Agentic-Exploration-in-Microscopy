import ast
from typing import Dict

from agent.utils import validate_param_type


class FunctionParser(ast.NodeTransformer):
    """AST Traverser: Extract function calls and assignment statements"""
    def __init__(self, fs: Dict[str, str], f_assigns: Dict[str, str]):
        super().__init__()
        if not (validate_param_type(fs, dict, "fs") and validate_param_type(f_assigns, dict, "f_assigns")):
            raise TypeError("fs and f_assigns must be dictionaries")
        self._fs = fs
        self._f_assigns = f_assigns

    def visit_Call(self, node: ast.Call):
        """Extract function call signature"""
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            try:
                sig = ast.unparse(node).strip()
                func_name = ast.unparse(node.func).strip()
                self._fs[func_name] = sig
            except Exception as e:
                print(f"Failed to extract function signature: {e}")
        return node

    def visit_Assign(self, node: ast.Assign):
        """Extract function calls in assignment statements"""
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            try:
                assign_str = ast.unparse(node).strip()
                func_name = ast.unparse(node.value.func).strip()
                self._f_assigns[func_name] = assign_str
            except Exception as e:
                print(f"Failed to extract assignment statement: {e}")
        return node