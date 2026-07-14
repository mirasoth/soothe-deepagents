"""Provider-agnostic utility for robust LLM call retries."""
# ruff: noqa: D102,PLR2004,TC003

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        out.append(current)
        current = current.__cause__ or current.__context__
    return out


def _is_rate_limit_error(exc: Exception) -> bool:
    for link in _iter_exception_chain(exc):
        if type(link).__name__ == "RateLimitError":
            return True
        response = getattr(link, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
        text = str(link).lower()
        if "429" in text or "rate limit" in text or "throttl" in text:
            return True
    return False


def _retry_after_seconds(exc: Exception) -> float | None:
    for link in _iter_exception_chain(exc):
        response = getattr(link, "response", None)
        if response is None:
            continue
        headers = getattr(response, "headers", None)
        if headers is None:
            continue
        retry_after = headers.get("retry-after")
        if retry_after is None:
            continue
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            continue
    return None


@dataclass(frozen=True, slots=True)
class LLMCallPolicyConfig:
    """Retry/timeout settings for :func:`run_llm_call_with_policy`."""

    timeout_seconds: float = 60.0
    retry_on_timeout: bool = True
    max_timeout_retries: int = 2
    timeout_retry_multiplier: float = 1.5
    retry_on_rate_limit: bool = True
    max_rate_limit_retries: int = 3
    rate_limit_backoff_base: float = 2.0
    rate_limit_backoff_max: float = 60.0
    respect_retry_after_header: bool = True
    rate_limit_retry_timeout_seconds: float | None = None

    def timeout_for_attempt(self, *, timeout_attempt: int, rate_limit_attempt: int) -> float:
        if rate_limit_attempt > 0 and self.rate_limit_retry_timeout_seconds is not None:
            return self.rate_limit_retry_timeout_seconds
        escalated = self.timeout_seconds * (self.timeout_retry_multiplier**timeout_attempt)
        return max(0.1, escalated)


def _backoff_for_rate_limit(config: LLMCallPolicyConfig, attempt: int, exc: Exception) -> float:
    if config.respect_retry_after_header:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, config.rate_limit_backoff_max)
    return min(config.rate_limit_backoff_base * (2**attempt), config.rate_limit_backoff_max)


async def run_llm_call_with_policy(
    call: Callable[[], Awaitable[T]],
    *,
    config: LLMCallPolicyConfig,
) -> T:
    """Run ``call`` with timeout/retry policy.

    ``call`` must be an async callable with no positional arguments.
    """
    timeout_attempt = 0
    rate_limit_attempt = 0
    max_timeout_attempts = config.max_timeout_retries + 1 if config.retry_on_timeout else 1
    max_rate_limit_attempts = config.max_rate_limit_retries + 1 if config.retry_on_rate_limit else 1

    while True:
        timeout = config.timeout_for_attempt(
            timeout_attempt=timeout_attempt,
            rate_limit_attempt=rate_limit_attempt,
        )
        try:
            return await asyncio.wait_for(call(), timeout=timeout)
        except TimeoutError:
            timeout_attempt += 1
            if timeout_attempt >= max_timeout_attempts:
                raise
            await asyncio.sleep(float(timeout_attempt))
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise
            rate_limit_attempt += 1
            if rate_limit_attempt >= max_rate_limit_attempts:
                raise
            await asyncio.sleep(_backoff_for_rate_limit(config, rate_limit_attempt - 1, exc))
