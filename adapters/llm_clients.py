import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from openai import APIConnectionError, BadRequestError, OpenAI, OpenAIError, RateLimitError


logger = logging.getLogger(__name__)
MAX_COMPLETION_TOKENS = int(os.getenv("EIMS_MAX_COMPLETION_TOKENS", "5120"))
QWEN_MAX_COMPLETION_TOKENS = int(os.getenv("EIMS_QWEN_MAX_COMPLETION_TOKENS", "8192"))
# Planner-only output ceiling so verbose task plans are not truncated by the
# generic 5120-token default. Override with EIMS_PLANNER_MAX_COMPLETION_TOKENS.
PLANNER_MAX_COMPLETION_TOKENS = int(os.getenv("EIMS_PLANNER_MAX_COMPLETION_TOKENS", "12000"))
# Exponential-backoff knobs for transient LLM connection/rate-limit retries.
RETRY_MAX_INTERVAL_SECONDS = float(os.getenv("EIMS_RETRY_MAX_INTERVAL_SECONDS", "60.0"))
RETRY_JITTER_FACTOR = float(os.getenv("EIMS_RETRY_JITTER_FACTOR", "0.3"))
# Default output cap for calls that do not specify max_tokens. Without this, the
# request goes out unbounded (up to the model max, e.g. 65536), which can trip
# OpenRouter credit checks (402). Override with EIMS_DEFAULT_MAX_TOKENS.
DEFAULT_MAX_TOKENS = int(os.getenv("EIMS_DEFAULT_MAX_TOKENS", "8192"))


@dataclass
class ClientBundle:
    llm_client: OpenAI
    vlm_client: OpenAI


def build_openai_clients(model_config: Any) -> ClientBundle:
    return ClientBundle(
        llm_client=OpenAI(api_key=model_config.openai_api_key, base_url=model_config.base_url),
        vlm_client=OpenAI(api_key=model_config.vlm_api_key, base_url=model_config.vlm_base_url),
    )


# Server-side rejection markers that indicate the configured model does not
# accept image content (i.e. it is a text-only model, not a vision model).
_VLM_IMAGE_REJECTION_MARKERS = (
    "unexpected item type in content",
    "image content is not supported",
    "image is not supported",
    "does not support image",
    "not support image input",
    "unsupported image",
    "image_url is not supported",
)


def explain_vlm_image_rejection(model: str, error_text: str) -> str:
    """Return an actionable hint when a VLM request is rejected because the
    configured model does not accept image input.

    Returns an empty string when the error does not look like a modality
    rejection, so callers can fall back to surfacing the raw error.
    """
    lowered = (error_text or "").lower()
    if not any(marker in lowered for marker in _VLM_IMAGE_REJECTION_MARKERS):
        return ""
    return (
        f"The configured VLM model '{model}' rejected image input, which usually means "
        "it is a text-only model. Set vlm_model_name to a vision-capable model "
        "(e.g. qwen3-vl-plus, qwen-vl-max, qwen-vl-plus, gpt-4o)."
    )


def _exponential_retry_delay(base_interval: float, attempt: int) -> float:
    """Exponential backoff (with jitter) before the next LLM retry.

    delay = base * 2^(attempt-1), capped at RETRY_MAX_INTERVAL_SECONDS and
    scaled by a random factor around RETRY_JITTER_FACTOR (0 disables jitter).
    """
    delay = float(base_interval) * (2.0 ** max(int(attempt) - 1, 0))
    delay = min(delay, RETRY_MAX_INTERVAL_SECONDS)
    jitter = RETRY_JITTER_FACTOR
    if jitter > 0:
        delay = delay * random.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)
    return max(0.0, delay)


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
    provider_max_tokens: Optional[int] = None,
    retries: int = 3,
    retry_interval: float = 3.0,
    **extra_kwargs: Any,
) -> Any:
    normalized_max_tokens = _normalize_max_tokens(
        max_tokens,
        model=model,
        provider_limit=provider_max_tokens,
    )
    normalized_seed = _normalize_seed(seed)
    _apply_reasoning_controls(model, extra_kwargs)
    _apply_disable_thinking_controls(model, extra_kwargs)

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
                "stream": stream,
                "max_tokens": normalized_max_tokens,
                **extra_kwargs,
            }
            if stop:
                request_kwargs["stop"] = stop
            if normalized_seed is not None:
                request_kwargs["seed"] = normalized_seed
            response = client.chat.completions.create(**request_kwargs)
            if not stream and not _response_final_content(response):
                # Reasoning-capable models sometimes emit only a thinking trace and
                # leave message.content empty (no final answer). That is not a valid
                # response: retry instead of treating empty content as success.
                if attempt >= retries:
                    raise OpenAIError(
                        "Model returned empty content (no final answer) after %s attempts: %s. "
                        "The model likely emitted only reasoning/thinking without a final answer; "
                        "use a non-reasoning model, disable thinking, increase max_tokens, or "
                        "switch endpoint." % (attempt, model)
                    )
                had_thinking_controls = _has_disable_thinking_controls(extra_kwargs)
                _apply_disable_thinking_controls(model, extra_kwargs)
                if not had_thinking_controls and _has_disable_thinking_controls(extra_kwargs):
                    logger.warning(
                        "Model returned empty content (no final answer) attempt=%s/%s model=%s; "
                        "retrying with thinking disabled. message fields: %s",
                        attempt,
                        retries,
                        model,
                        _describe_message_fields(response),
                    )
                elif had_thinking_controls:
                    logger.warning(
                        "Model returned empty content (no final answer) attempt=%s/%s model=%s; "
                        "retrying (thinking already disabled). message fields: %s",
                        attempt,
                        retries,
                        model,
                        _describe_message_fields(response),
                    )
                else:
                    logger.warning(
                        "Model returned empty content (no final answer) attempt=%s/%s model=%s; "
                        "retrying with unchanged parameters (no disable-thinking option for this model). "
                        "message fields: %s",
                        attempt,
                        retries,
                        model,
                        _describe_message_fields(response),
                    )
                time.sleep(_exponential_retry_delay(retry_interval, attempt))
                continue
            return response
        except BadRequestError as exc:
            if _is_invalid_reasoning_control_error(exc):
                logger.warning(
                    "Model request rejected reasoning/thinking controls; retrying without them: model=%s error=%s",
                    model,
                    exc,
                )
                _remove_reasoning_controls(extra_kwargs)
                _remove_disable_thinking_controls(extra_kwargs)
                continue
            raise
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
            time.sleep(_exponential_retry_delay(retry_interval, attempt))


def _has_disable_thinking_controls(extra_kwargs: dict[str, Any]) -> bool:
    extra_body = dict(extra_kwargs.get("extra_body") or {})
    return ("enable_thinking" in extra_body) or ("thinking" in extra_body)


def _apply_disable_thinking_controls(model: str, extra_kwargs: dict[str, Any]) -> None:
    """Ask reasoning-capable models to emit a direct final answer (message.content).

    Reasoning traces (message.reasoning_content / thinking) are not a usable
    answer; if the model leaves content empty the request is retried in
    create_chat_completion. These provider-specific parameters disable thinking
    where supported.
    """
    model_lower = str(model or "").lower()
    if "qwen" in model_lower:
        extra_body = dict(extra_kwargs.get("extra_body") or {})
        extra_body.setdefault("enable_thinking", False)
        extra_kwargs["extra_body"] = extra_body
    elif "kimi" in model_lower:
        extra_body = dict(extra_kwargs.get("extra_body") or {})
        extra_body.setdefault("thinking", {"type": "disabled"})
        extra_kwargs["extra_body"] = extra_body


def _remove_disable_thinking_controls(extra_kwargs: dict[str, Any]) -> None:
    extra_body = dict(extra_kwargs.get("extra_body") or {})
    extra_body.pop("enable_thinking", None)
    extra_body.pop("thinking", None)
    if extra_body:
        extra_kwargs["extra_body"] = extra_body
    else:
        extra_kwargs.pop("extra_body", None)


def _describe_message_fields(response: Any) -> str:
    """Diagnostic: summarize the fields of the response message (for empty-content debugging)."""
    try:
        message = response.choices[0].message
    except Exception:
        return "<unreadable response>"
    parts = []
    for field in ("content", "reasoning_content", "reasoning", "thinking"):
        value = getattr(message, field, None)
        if isinstance(value, str):
            parts.append(f"{field}={len(value)}ch")
        elif value is not None:
            parts.append(f"{field}={type(value).__name__}")
    return "{ " + ", ".join(parts) + " }"


def _response_final_content(response: Any) -> str:
    """Extract the final answer text from a non-streaming chat completion.

    Only message.content counts as the final answer; reasoning traces are not
    returned even if present.
    """
    try:
        message = response.choices[0].message
    except Exception:
        return ""
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _apply_reasoning_controls(model: str, extra_kwargs: dict[str, Any]) -> None:
    if "gpt-5" not in model.lower():
        return
    reasoning_effort = os.getenv("EIMS_REASONING_EFFORT", "").strip()
    verbosity = os.getenv("EIMS_VERBOSITY", "").strip()
    if reasoning_effort:
        extra_kwargs.setdefault("reasoning_effort", reasoning_effort)
    if verbosity:
        extra_kwargs.setdefault("verbosity", verbosity)


def _has_reasoning_controls(extra_kwargs: dict[str, Any]) -> bool:
    return "reasoning_effort" in extra_kwargs or "verbosity" in extra_kwargs


def _remove_reasoning_controls(extra_kwargs: dict[str, Any]) -> None:
    extra_kwargs.pop("reasoning_effort", None)
    extra_kwargs.pop("verbosity", None)


def _is_invalid_reasoning_control_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "reasoning_effort" in message
        or "verbosity" in message
        or "enable_thinking" in message
        or "thinking" in message
        or "unsupported parameter" in message
        or "unknown parameter" in message
        or "unrecognized request argument" in message
    )


def _normalize_max_tokens(
    max_tokens: Optional[int],
    *,
    model: str,
    provider_limit: Optional[int] = None,
) -> Optional[int]:
    if max_tokens is None:
        normalized = DEFAULT_MAX_TOKENS
    else:
        normalized = int(max_tokens)
    if normalized < 1:
        logger.warning("Invalid max_tokens=%s; using 1 instead.", max_tokens)
        return 1
    provider_limit = MAX_COMPLETION_TOKENS if provider_limit is None else int(provider_limit)
    if "qwen" in model.lower():
        provider_limit = min(provider_limit, QWEN_MAX_COMPLETION_TOKENS)
    if normalized > provider_limit:
        logger.warning(
            "max_tokens=%s exceeds provider limit; clamping to %s.",
            max_tokens,
            provider_limit,
        )
        return provider_limit
    return normalized


def _normalize_seed(seed: Any) -> Optional[int]:
    if seed is None:
        return None
    if isinstance(seed, str):
        seed = seed.strip()
        if not seed:
            return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        logger.warning("Invalid seed=%r; omitting seed from model request.", seed)
        return None


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
