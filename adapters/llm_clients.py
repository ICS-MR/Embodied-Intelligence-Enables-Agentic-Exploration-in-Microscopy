import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from openai import APIConnectionError, OpenAI, RateLimitError


logger = logging.getLogger(__name__)
MAX_COMPLETION_TOKENS = 5120


@dataclass
class ClientBundle:
    llm_client: OpenAI
    vlm_client: OpenAI


def build_openai_clients(model_config: Any) -> ClientBundle:
    return ClientBundle(
        llm_client=OpenAI(api_key=model_config.openai_api_key, base_url=model_config.base_url),
        vlm_client=OpenAI(api_key=model_config.vlm_api_key, base_url=model_config.vlm_base_url),
    )


def create_chat_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    seed: Optional[int] = None,
    stop: Optional[list[str]] = None,
    stream: bool = False,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    retry_interval: float = 3.0,
    **extra_kwargs: Any,
) -> Any:
    normalized_max_tokens = _normalize_max_tokens(max_tokens)
    if not stream and "qwen" in model.lower():
        extra_body = dict(extra_kwargs.get("extra_body") or {})
        extra_body.setdefault("enable_thinking", False)
        extra_kwargs["extra_body"] = extra_body

    attempt = 0
    while True:
        attempt += 1
        try:
            logger.info(
                "Sending model request: model=%s stream=%s max_tokens=%s attempt=%s/%s",
                model,
                stream,
                normalized_max_tokens,
                attempt,
                retries,
            )
            request_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stop": stop,
                "stream": stream,
                "max_tokens": normalized_max_tokens,
                **extra_kwargs,
            }
            if seed is not None:
                request_kwargs["seed"] = seed
            return client.chat.completions.create(**request_kwargs)
        except (RateLimitError, APIConnectionError) as exc:
            if _is_invalid_parameter_error(exc):
                logger.error(
                    "Model request rejected due to invalid parameters: model=%s max_tokens=%s error=%s",
                    model,
                    normalized_max_tokens,
                    exc,
                )
                raise
            if attempt >= retries:
                raise
            logger.warning("Model request failed (%s/%s): %s", attempt, retries, exc)
            time.sleep(retry_interval)


def _normalize_max_tokens(max_tokens: Optional[int]) -> Optional[int]:
    if max_tokens is None:
        return None

    normalized = int(max_tokens)
    if normalized < 1:
        logger.warning("Invalid max_tokens=%s; using 1 instead.", max_tokens)
        return 1
    if normalized > MAX_COMPLETION_TOKENS:
        logger.warning(
            "max_tokens=%s exceeds provider limit; clamping to %s.",
            max_tokens,
            MAX_COMPLETION_TOKENS,
        )
        return MAX_COMPLETION_TOKENS
    return normalized


def _is_invalid_parameter_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "invalidparameter" in message
        or "invalid_parameter" in message
        or "invalid_request_error" in message
        or "max_tokens should be" in message
    )


def stream_chat_completion_text(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    stop: Optional[list[str]] = None,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    retry_interval: float = 3.0,
    **extra_kwargs: Any,
) -> Iterator[str]:
    stream = create_chat_completion(
        client,
        model=model,
        messages=messages,
        temperature=temperature,
        stop=stop,
        stream=True,
        max_tokens=max_tokens,
        retries=retries,
        retry_interval=retry_interval,
        **extra_kwargs,
    )
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        delta = getattr(chunk.choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            yield content
