import os
import shutil
import ast
from typing import Set, Optional, Dict, Any, List, Tuple
from openai import OpenAI

# === 配置常量（可提取到独立配置文件）===
class Config:
    """项目配置类，集中管理所有配置项，减少硬编码"""
    # 提示词模板
    PROMPTS: Dict[str, str] = {
        "system_role": """
You are a Python programming assistant skilled in understanding code, function definitions, and generating relevant examples.
Output format:
# Example input
Task instruction based on the function
# Example output
Function usage example
""",
        "docstring": """You are a professional Python developer. Generate a concise and accurate Google-style docstring in English for the following function.
Requirements:
- Include Args and Returns if applicable
- Do NOT include function definition, triple quotes, code fences, or extra explanations
- Return ONLY the docstring content

Function signature:
{signature}

Generated docstring:
""",
        "usage": """You are a Python expert. Based on the following method definitions, generate a concise and practical usage example.

Requirements:
- Use English comments.
- Include necessary class instantiation and method calls.
- Cover the main functionalities.
- Output format must strictly follow:
# Example input
<A brief task description>
# Example output
<Directly runnable Python code without any markdown code fences (e.g., no ```python or ```)>.

Method definitions:
{methods}

Generate the usage example:
""",
        "summary": """You are a technical documentation expert. Summarize the core capabilities of the following Python class based on its public methods.
Requirements:
- Use English
- List by functional modules (start each with "- ")
- One sentence per point, concise and clear
- Do NOT repeat method names; abstract their purpose
- Ignore placeholder texts like '自动生成 docstring 失败'
- Do NOT include code, examples, or prefixes

Methods:
{methods}

Capabilities summary:
""",
        "task_example": """You are a task planning expert. Generate a usage example in the specified format based on the tool's capabilities.

Tool name: {class_name}
Capabilities:
{capabilities}

Requirements:
- Generate a realistic, specific user instruction (# Example input)
- Output must strictly follow:
<Task Ready>
{{"Status": "OK"}}
</Task Ready>
<Task steps>
[
    {{
        "subtask_index": 1,
        "module": "{class_name}",
        "command": "Clear, specific action in English, starting with a verb"
    }}
]
</Task steps>
- The 'command' must be based ONLY on the given capabilities
- Both input and command must be in English
- ONLY output # Example input and # Example output sections, no extra text

Generate example:
"""
    }

    # 默认路径配置
    DEFAULT_TOOL_FOLDER = "tool"
    DEFAULT_TASK_MANAGER_FILE = 'prompts/task_manager_full.py'
    SAY_METHOD_TEMPLATE = ""  # 从tool.base导入，此处先占位，初始化时加载


# === 工具扫描类：负责扫描、查询、导入继承BaseTool的类 ===
class ToolScanner:
    """工具扫描器，负责扫描工具目录、识别继承BaseTool的类、动态导入类"""

    def __init__(self, tool_folder: str = Config.DEFAULT_TOOL_FOLDER):
        self.tool_folder = tool_folder
        self.inherit_classes: List[Tuple[str, str]] = []  # （类名, 文件路径）

    def scan_base_tool_subclasses(self) -> List[Tuple[str, str]]:
        """扫描所有继承BaseTool的类，返回结果列表"""
        if not os.path.isdir(self.tool_folder):
            print(f"⚠️  工具目录 {self.tool_folder} 不存在，返回空列表")
            return []

        self.inherit_classes = []
        for root, _, files in os.walk(self.tool_folder):
            for file_name in files:
                if not file_name.endswith(".py") or file_name == "__init__.py":
                    continue

                file_path = os.path.join(root, file_name)
                try:
                    self._parse_file_for_base_tool(file_path)
                except Exception as e:
                    print(f"⚠️  解析文件 {file_path} 失败：{str(e)}")
                    continue

        return self.inherit_classes

    def _parse_file_for_base_tool(self, file_path: str) -> None:
        """解析单个Python文件，识别继承BaseTool的类"""
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        tree = ast.parse(file_content, filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "BaseTool":
                        self.inherit_classes.append((node.name, file_path))
                        print(f"✅ 找到继承BaseTool的类：{node.name}（位于{file_path}）")
                        break

    def dynamic_import_class(self, class_name: str, file_path: str) -> Optional[Any]:
        """动态导入指定类，返回类对象（失败返回None）"""
        try:
            # 构建模块名：tool.xxx
            module_name = f"{self.tool_folder}.{os.path.splitext(os.path.basename(file_path))[0]}"
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as e:
            print(f"⚠️  导入类 {class_name} 失败：{e}")
            return None


# === LLM代理类：负责LLM调用，封装OpenAI API ===
class LLMAgent:
    """LLM Agent compatible with OpenAI API (e.g., OpenAI, Qwen via DashScope)."""

    def __init__(
            self,
            api_key: str,
            base_url: str = "https://api.openai.com/v1",
            model: str = "gpt-3.5-turbo",
            temperature: float = 0.7,
            timeout: int = 30
    ):
        print(f"[INFO] Initializing LLM Agent with model: {model}")
        self.client = OpenAI(api_key=api_key, base_url=base_url.strip())
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.system_prompt = Config.PROMPTS["system_role"].strip()

    def generate(self, prompt: str) -> str:
        """调用LLM生成结果，返回格式化字符串"""
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
                "type": "text"
            },
            {
                "role": "user",
                "content": prompt.strip(),
                "type": "text"
            }
        ]

        try:
            print(f"[INFO] Calling LLM API with prompt (first 50 chars): {prompt[:50]}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                timeout=self.timeout,
            )
            result = response.choices[0].message.content.strip()
            print(f"[SUCCESS] LLM generation completed (result length: {len(result)} chars)")
            return result
        except Exception as e:
            error_msg = f"[ERROR] LLM API call failed: {str(e)}"
            print(error_msg)
            return f"❌ LLM Error: {e}"

    def safe_generate(self, prompt: str, fallback: str = "LLM generation failed.") -> str:
        """安全调用LLM，失败返回兜底值"""
        try:
            result = self.generate(prompt)
            if not result or "❌" in result or not result.strip():
                print(f"[WARNING] LLM returned empty/invalid result: {fallback[:30]}...")
                return fallback
            return result
        except Exception as e:
            print(f"[ERROR] Safe LLM call failed: {str(e)}, using fallback")
            return fallback


# === 工具文档生成类：负责生成提示词、使用示例、能力摘要 ===
class ToolDocGenerator:
    """工具文档生成器，负责生成工具提示词文件、使用示例、能力摘要"""

    def __init__(self, llm_agent: LLMAgent):
        self.llm_agent = llm_agent
        self.prompts = Config.PROMPTS

    def generate_class_usage_example(self, public_methods: str) -> str:
        """生成类使用示例"""
        print("[INFO] Generating usage example for the class methods")
        prompt = self.prompts["usage"].format(methods=public_methods)
        return self.llm_agent.safe_generate(prompt)

    def summarize_class_capabilities(self, public_methods: str) -> str:
        """生成类能力摘要"""
        print("[INFO] Generating capabilities summary for the class")
        prompt = self.prompts["summary"].format(methods=public_methods)
        raw_summary = self.llm_agent.safe_generate(prompt)

        # 清理摘要内容
        lines = [
            line for line in raw_summary.splitlines()
            if not line.strip().startswith(("Capabilities summary", "Summary", "：", ":", "```", "Note", "Requirement"))
        ]
        return "\n".join(lines).strip()

    def generate_task_example(self, class_name: str, capability_summary: str) -> str:
        """生成任务示例"""
        print(f"[INFO] Generating task example for class: {class_name}")
        prompt = self.prompts["task_example"].format(
            class_name=class_name,
            capabilities=capability_summary
        )
        return self.llm_agent.safe_generate(prompt)

    def generate_tool_prompt_file(self, target_class: str, public_methods: str) -> str:
        """生成工具提示词文件，返回文件绝对路径"""
        print(f"[INFO] Generating tool usage prompt for class: {target_class}")
        output_file = f"{target_class}_usage.py"
        usage_example = self.generate_class_usage_example(public_methods)

        # 构建提示词内容
        header = self._build_prompt_header(target_class)
        methods_section = "# API function\n" + public_methods + "\n\n"
        usage_section = "# Usage Example\n" + usage_example.strip()
        full_content = f"prompt_{target_class} = '''{header}{methods_section}{usage_section}'''"

        # 写入文件并返回路径
        abs_output_file = os.path.abspath(output_file)
        with open(abs_output_file, 'w', encoding='utf-8') as f:
            f.write(full_content)

        print(f"✅ Prompt file generated successfully: {abs_output_file}")
        return abs_output_file

    @staticmethod
    def _build_prompt_header(target_class: str) -> str:
        """构建提示词文件头部内容，抽离为独立方法"""
        return f'''import cv2 as cv
import numpy as np
# Prohibit importing other Python libraries.

# Role and Objectives
Role: A professional assistant specialized in generating Python code for controlling {target_class} systems based on user instructions.
Core Objective: Ensure the generated code is secure and complies with hardware constraints.

# Behavioral Constraints
- Language use: English
- Determine parameters based on Behavioral Constraints and Current environment
## Hardware security control
- All motion and imaging commands must include parameter verification.
## Context-Aware
- Fully utilize user-provided file information to avoid assuming non-existent files/parameters
## Decision-Making Mechanism
- No assumptions are allowed.
- Prioritize using the provided API functions to complete tasks. Carefully read the API function definitions before answering to ensure that the tasks can be completed and comply with the function definitions.
- Saving mechanism: Use the provided functions to read and save files.
- Execute each instruction sequentially and completely without skipping, merging, or reordering.
'''


# === 任务管理器更新类：负责备份、更新、还原任务管理器 ===
class TaskManagerUpdater:
    """任务管理器更新器，负责备份、更新、还原任务管理器文件"""

    def __init__(self, task_manager_file: str = Config.DEFAULT_TASK_MANAGER_FILE):
        self.task_manager_file = task_manager_file
        self.backup_file = f"{self.task_manager_file}.backup"
        self._ensure_dir_exists()

    def _ensure_dir_exists(self) -> None:
        """确保任务管理器文件所在目录存在"""
        dir_name = os.path.dirname(self.task_manager_file)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

    def backup_task_manager(self) -> None:
        """备份任务管理器原始文件"""
        if os.path.exists(self.task_manager_file):
            shutil.copy2(self.task_manager_file, self.backup_file)
            print(f"✅ Created task manager backup: {self.backup_file}")
        else:
            # 创建空备份文件
            with open(self.backup_file, 'w', encoding='utf-8') as f:
                f.write("# 任务管理器初始备份\n###Tool Description\n###Tool usage\n")
            print(f"✅ Created empty task manager backup: {self.backup_file}")

    def update_task_manager(self, class_name: str, capability_summary: str, task_example: str) -> None:
        """更新任务管理器文件，写入类的描述和示例"""
        try:
            print(f"[INFO] Updating task manager file: {self.task_manager_file}")
            self._create_default_task_manager_if_not_exists()

            # 读取原有内容并更新
            with open(self.task_manager_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            output_lines = self._insert_content_to_lines(lines, class_name, capability_summary, task_example)

            # 写入更新后的内容
            with open(self.task_manager_file, 'w', encoding='utf-8') as f:
                f.writelines(output_lines)

            print(f"✅ Task manager file updated successfully: {self.task_manager_file}")
        except Exception as e:
            print(f"⚠️  Failed to update {self.task_manager_file}: {e}")

    def _create_default_task_manager_if_not_exists(self) -> None:
        """若任务管理器文件不存在，创建默认模板"""
        if not os.path.exists(self.task_manager_file):
            with open(self.task_manager_file, 'w', encoding='utf-8') as f:
                f.write("# 任务管理器自动生成模板\n###Tool Description\n###Tool usage\n")
            print(f"✅ 自动创建任务管理器文件：{self.task_manager_file}")

    @staticmethod
    def _insert_content_to_lines(lines: List[str], class_name: str, summary: str, example: str) -> List[str]:
        """在指定标记后插入内容，返回更新后的行列表"""
        tool_desc_marker = '###Tool Description\n'
        tool_usage_marker = '###Tool usage\n'
        output_lines = []

        for line in lines:
            output_lines.append(line)
            if line.strip() == tool_desc_marker.strip():
                output_lines.append(f"{class_name}\n{summary}\n\n")
            if line.strip() == tool_usage_marker.strip():
                output_lines.append(f"{example}\n\n")

        return output_lines

    def restore_task_manager(self) -> None:
        """从备份文件还原任务管理器原始状态"""
        if os.path.exists(self.backup_file):
            shutil.copy2(self.backup_file, self.task_manager_file)
            print(f"\n✅ Restored task manager file to original state: {self.task_manager_file}")


# === 资源清理类：负责清理临时文件 ===
class ResourceCleaner:
    """资源清理器，负责清理生成的临时提示词文件和备份文件"""

    @staticmethod
    def clean_up(prompt_file_path: str, backup_file_path: str) -> None:
        """执行资源清理，删除临时提示词文件和备份文件"""
        print("\n" + "=" * 30 + " START CLEANUP " + "=" * 30)

        # 清理提示词文件
        ResourceCleaner._delete_file(prompt_file_path, "temporary prompt file")

        # 清理备份文件
        ResourceCleaner._delete_file(backup_file_path, "task manager backup file")

        print("=" * 31 + " CLEANUP DONE " + "=" * 31)

    @staticmethod
    def _delete_file(file_path: str, file_desc: str) -> None:
        """删除单个文件，包含容错处理"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ Deleted {file_desc}: {file_path}")
            else:
                print(f"ℹ️ {file_desc} not found: {file_path} (skip deletion)")
        except Exception as e:
            print(f"⚠️  Failed to delete {file_desc}: {e}")


# === 核心流水线类：整合所有模块，提供端到端处理流程 ===
class ToolProcessingPipeline:
    """工具处理流水线，整合所有子模块，提供完整的处理流程"""

    def __init__(self, openai_api_key: str, base_url: str, model_name: str):
        # 初始化配置
        self._init_llm_agent(openai_api_key, base_url, model_name)
        self._init_sub_modules()

        self.generated_prompt_files: List[str] = []  # ← 新增：记录所有生成的提示词文件
        # 加载外部常量
        self._load_say_method_template()

    def _init_llm_agent(self, api_key: str, base_url: str, model_name: str) -> None:
        """初始化LLM Agent"""
        self.llm_agent = LLMAgent(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0.2,
            timeout=15
        )

    def _init_sub_modules(self) -> None:
        """初始化所有子模块"""
        self.tool_scanner = ToolScanner()
        self.doc_generator = ToolDocGenerator(self.llm_agent)
        self.task_manager = TaskManagerUpdater()
        self.resource_cleaner = ResourceCleaner()

    def _load_say_method_template(self) -> None:
        """加载SAY_METHOD_TEMPLATE从tool.base"""
        try:
            from tool.base import SAY_METHOD_TEMPLATE
            Config.SAY_METHOD_TEMPLATE = SAY_METHOD_TEMPLATE
        except ImportError:
            print("⚠️  未能从tool.base导入SAY_METHOD_TEMPLATE，使用空模板")


    def run_pipeline(self) -> None:
        """运行完整的工具处理流水线"""
        try:
            # 1. 扫描所有继承BaseTool的类
            inherit_classes = self.tool_scanner.scan_base_tool_subclasses()
            if not inherit_classes:
                print("⚠️  未找到任何继承BaseTool的类，程序终止")
                return

            # 2. 备份任务管理器
            self.task_manager.backup_task_manager()

            # 3. 批量处理每个类
            self._process_all_classes(inherit_classes)


        except Exception as e:
            print(f"\n[FATAL ERROR] Program failed: {str(e)}")

    def _process_all_classes(self, inherit_classes: List[Tuple[str, str]]) -> None:
        """批量处理所有继承BaseTool的类"""
        for class_name, file_path in inherit_classes:
            print(f"\n" + "-" * 40 + f" 处理类：{class_name} " + "-" * 40)

            # 动态导入类
            target_class = self.tool_scanner.dynamic_import_class(class_name, file_path)
            if not target_class:
                continue

            # 获取公共方法
            public_methods = target_class.get_formatted_tool_methods(inject_say=True)

            # 生成提示词文件
            prompt_file_path = self.doc_generator.generate_tool_prompt_file(class_name, public_methods)

            self.generated_prompt_files.append(prompt_file_path)

            # 生成能力摘要和任务示例
            capability_summary = self.doc_generator.summarize_class_capabilities(public_methods)
            task_example = self.doc_generator.generate_task_example(class_name, capability_summary)

            # 更新任务管理器
            self.task_manager.update_task_manager(class_name, capability_summary, task_example)


    def release(self):
        for prompt_file in self.generated_prompt_files:
            self.resource_cleaner._delete_file(prompt_file, "generated prompt file")

        self.task_manager.restore_task_manager()

        self.resource_cleaner._delete_file(self.task_manager.backup_file, "backup file")
        # 5. 输出完成信息
        print("\n" + "=" * 50)
        print("[COMPLETE] All classes processed successfully!")
        print("=" * 50)


# === 程序入口 ===
if __name__ == "__main__":
    print("=" * 50)
    print("[START] Starting process (auto find BaseTool classes + OOP encapsulation)")
    print("=" * 50)

    try:
        # 加载API配置
        from config.agent_config import openai_api_key, base_url, model_name

        # 校验API密钥
        if not openai_api_key or openai_api_key == "your-openai-api-key":
            raise EnvironmentError("Please set a valid OPENAI_API_KEY in config/agent_config.py")

        # 初始化并运行流水线
        pipeline = ToolProcessingPipeline(openai_api_key, base_url, model_name)
        
        pipeline.run_pipeline()

        pipeline.release()

    except ImportError:
        print("[ERROR] Could not import config from config.agent_config - please check the file exists")
    except Exception as e:
        print(f"\n[FATAL ERROR] Program terminated: {str(e)}")
