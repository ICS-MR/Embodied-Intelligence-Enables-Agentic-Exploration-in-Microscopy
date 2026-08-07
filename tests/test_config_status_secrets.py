import asyncio
from types import SimpleNamespace

from api.config_routes import get_config_status


class _FakeRuntimeManager:
    def __init__(self) -> None:
        self.system_status = SimpleNamespace(
            initialized=False,
            initializing=False,
            error="",
            message="ok",
            system_phase="ready_to_start",
            failure_step="",
        )

    def get_preview_status(self) -> dict[str, str]:
        return {"preview_phase": "idle"}

    def current_snapshot(self) -> dict[str, object]:
        return {"system": {}, "agent": {}, "startup": {}}

    def get_transmitted_light_runtime_info(self) -> dict[str, object]:
        return {}


def test_config_status_returns_loaded_secrets(monkeypatch) -> None:
    secret_snapshot = {
        "system": {"CONFIG_PATH": "demo.cfg"},
        "agent": {
            "microscope_mode": "demo",
            "image_analysis_mode": "mock",
            "segmentation_mode": "mock",
            "clarify_enabled": False,
            "checker_enabled": True,
            "openai_api_key": "sk-test-1234",
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-4.1",
            "vlm_api_key": "vlm-test-5678",
            "vlm_base_url": "https://api.openai.com/v1",
            "vlm_model_name": "gpt-4.1",
            "masked": {
                "openai_api_key": "sk-t****1234",
                "vlm_api_key": "vlm-****5678",
            },
        },
        "startup": {},
    }
    persisted_snapshot = {
        "system": {"CONFIG_PATH": "persisted.cfg"},
        "agent": secret_snapshot["agent"],
        "startup": {},
    }

    monkeypatch.setattr("api.config_routes.read_config_snapshot", lambda: secret_snapshot)
    monkeypatch.setattr(
        "api.config_routes.read_public_config_snapshot",
        lambda **kwargs: persisted_snapshot,
    )
    monkeypatch.setattr(
        "api.config_routes.check_snapshot_assets",
        lambda snapshot: SimpleNamespace(ready=True),
    )

    response = asyncio.run(get_config_status(runtime_manager=_FakeRuntimeManager()))

    assert response.agent.openai_api_key == "sk-test-1234"
    assert response.agent.vlm_api_key == "vlm-test-5678"
    assert response.agent.masked["openai_api_key"] == "sk-t****1234"

