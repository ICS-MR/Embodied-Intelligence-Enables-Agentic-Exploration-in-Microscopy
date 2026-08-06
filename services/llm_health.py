"""Lightweight model connectivity checks for the configuration UI."""

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from adapters.llm_clients import create_chat_completion


@dataclass
class LLMConnectionConfig:
    openai_api_key: str
    base_url: str
    model_name: str
    llm_seed: int | None = None


@dataclass
class VLMConnectionConfig:
    vlm_api_key: str
    vlm_base_url: str
    vlm_model_name: str
    llm_seed: int | None = None


def validate_llm_connection(config: LLMConnectionConfig) -> None:
    api_key = str(config.openai_api_key or "").strip()
    model_name = str(config.model_name or "").strip()
    base_url = str(config.base_url or "").strip()
    if not api_key:
        raise ValueError("LLM API key is empty.")
    if not model_name:
        raise ValueError("LLM model name is empty.")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    create_chat_completion(
        client,
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a connectivity test endpoint."},
            {"role": "user", "content": "Reply with OK."},
        ],
        temperature=0.0,
        seed=config.llm_seed,
        max_tokens=8,
        retries=1,
        timeout=15,
    )


def validate_vlm_connection(config: VLMConnectionConfig) -> None:
    api_key = str(config.vlm_api_key or "").strip()
    model_name = str(config.vlm_model_name or "").strip()
    base_url = str(config.vlm_base_url or "").strip()
    if not api_key:
        raise ValueError("VLM API key is empty.")
    if not model_name:
        raise ValueError("VLM model name is empty.")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    create_chat_completion(
        client,
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a vision connectivity test endpoint."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with OK if you can read this image request."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
                            )
                        },
                    },
                ],
            },
        ],
        temperature=0.0,
        seed=config.llm_seed,
        max_tokens=8,
        retries=1,
        timeout=20,
    )
