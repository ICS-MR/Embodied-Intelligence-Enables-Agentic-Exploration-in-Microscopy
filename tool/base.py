from typing import Dict, List, Callable, Optional
import inspect

# 预设say方法模板
SAY_METHOD_TEMPLATE = '''def say(message: str):
    """
    Outputs a log message with `[ACTION]`, `[INFO]`, or `[ERROR]` prefix. Ensures consistent logging format.
    """
    print(f'robot says: {message}')'''


def format_method_with_docstring(def_line: str, docstring: str) -> str:
    if not docstring.strip():
        return f"{def_line}\n    pass  # No docstring"
    lines = docstring.splitlines()
    if len(lines) == 1:
        doc_block = f'    """{lines[0]}"""'
    else:
        doc_block = '    """\n' + '\n'.join(f"    {line}" for line in lines) + '\n    """'
    return f"{def_line}\n{doc_block}"


# 标记函数为工具函数
def tool_func(func: Callable) -> Callable:
    """装饰器：仅标记函数为工具函数，不立即注册"""
    func._is_tool_func = True
    return func


class ToolMeta(type):
    """元类：在类创建时扫描所有被 @tool_func 标记的函数，注册到子类的独立注册表"""
    def __new__(cls, name, bases, attrs):
        # 创建类
        subclass = super().__new__(cls, name, bases, attrs)
        
        # 初始化子类的独立注册表
        subclass._tool_func_registry = {}
        
        # 扫描所有在类中定义的属性
        for attr_name, attr_value in attrs.items():
            if callable(attr_value) and getattr(attr_value, '_is_tool_func', False):
                # 是被 @tool_func 标记的函数
                try:
                    sig = inspect.signature(attr_value)
                    params = list(sig.parameters.values())
                    
                    # 移除 self（如果是实例方法）
                    if params and params[0].name == 'self' and params[0].kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ):
                        params = params[1:]
                    
                    filtered_sig = sig.replace(parameters=params)
                    signature_str = f"def {attr_name}{filtered_sig}:"
                except Exception as e:
                    signature_str = f"def {attr_name}() -> <signature_parse_failed>:"
                    print(f"[WARNING] 提取函数 {attr_name} 签名失败：{e}")
                
                docstring = (attr_value.__doc__ or "").strip()
                
                subclass._tool_func_registry[attr_name] = {
                    "signature": signature_str,
                    "docstring": docstring
                }
        
        return subclass


class BaseTool(metaclass=ToolMeta):
    """基类，提供工具函数查询接口"""
    
    @classmethod
    def get_public_methods(cls) -> List[str]:
        method_names = list(cls._tool_func_registry.keys())
        method_names.sort()
        return method_names

    @classmethod
    def get_tool_methods(cls) -> List[Dict[str, str]]:
        return [
            {
                "name": func_name,
                "signature": func_info["signature"],
                "docstring": func_info["docstring"]
            }
            for func_name, func_info in cls._tool_func_registry.items()
        ]

    @classmethod
    def get_formatted_tool_methods(cls, inject_say: bool = True) -> str:
        method_strs = []
        for tool_method in cls.get_tool_methods():
            method_str = format_method_with_docstring(
                tool_method["signature"],
                tool_method["docstring"]
            )
            method_strs.append(method_str)

        if inject_say and "def say(" not in "\n".join(method_strs):
            method_strs.append(SAY_METHOD_TEMPLATE)

        return "\n\n".join(method_strs)
    
