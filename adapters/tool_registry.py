from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tool.ports import validate_port_implementation


@dataclass
class ToolBinding:
    name: str
    env: Any
    executor: Any
    role: str = ""
    public_callables: Dict[str, Any] = field(default_factory=dict)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    expose_public_callables: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._bindings: Dict[str, ToolBinding] = {}

    def register_tool(
        self,
        name: str,
        env: Any,
        executor: Any,
        *,
        role: str = "",
        validate_role: bool = False,
        expose_public_callables: bool = False,
    ) -> None:
        if name in self._bindings:
            raise ValueError(f"Platform '{name}' is already registered")

        if validate_role and role:
            validate_port_implementation(env, role)
        methods = env.get_public_methods() if hasattr(env, "get_public_methods") else []
        public_callables = {
            method_name: getattr(env, method_name)
            for method_name in methods
            if hasattr(env, method_name)
        }
        conflicts = self._find_public_callable_conflicts(name, public_callables)
        if expose_public_callables and conflicts:
            detail = ", ".join(
                f"'{method_name}' already provided by '{owner}'"
                for method_name, owner in conflicts.items()
            )
            raise ValueError(f"Platform '{name}' exposes duplicate public methods: {detail}")

        metadata = env.get_tool_descriptors() if hasattr(env, "get_tool_descriptors") else []
        self._bindings[name] = ToolBinding(
            name=name,
            env=env,
            executor=executor,
            role=role,
            public_callables=public_callables,
            metadata=metadata,
            expose_public_callables=expose_public_callables,
        )

    def register_platform(self, name: str, env: Any, executor: Any, *, port_kind: str) -> None:
        self.register_tool(name, env, executor, role=port_kind, validate_role=True, expose_public_callables=True)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "platform": binding.name,
                "role": binding.role,
                "methods": sorted(binding.public_callables.keys()),
                "metadata": binding.metadata,
            }
            for binding in self._bindings.values()
        ]

    def get_tool(self, name: str) -> Optional[ToolBinding]:
        return self._bindings.get(name)

    def get_executor(self, name: str) -> Any:
        binding = self.get_tool(name)
        return binding.executor if binding else None

    def all_public_callables(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for binding in self._bindings.values():
            if not binding.expose_public_callables:
                continue
            for method_name, method in binding.public_callables.items():
                owner = self._public_callable_owner(method_name, exclude_platform=binding.name)
                if owner is not None:
                    raise ValueError(
                        f"Duplicate public method '{method_name}' is registered by both "
                        f"'{owner}' and '{binding.name}'"
                    )
                result[method_name] = method
        return result

    def allowed_call_names(self) -> set[str]:
        return set(self.all_public_callables().keys())

    def _public_callable_owner(self, method_name: str, *, exclude_platform: str | None = None) -> Optional[str]:
        for binding_name, binding in self._bindings.items():
            if exclude_platform is not None and binding_name == exclude_platform:
                continue
            if not binding.expose_public_callables:
                continue
            if method_name in binding.public_callables:
                return binding_name
        return None

    def _find_public_callable_conflicts(
        self,
        platform_name: str,
        public_callables: Dict[str, Any],
    ) -> Dict[str, str]:
        conflicts: Dict[str, str] = {}
        for method_name in public_callables:
            owner = self._public_callable_owner(method_name, exclude_platform=platform_name)
            if owner is not None:
                conflicts[method_name] = owner
        return conflicts
