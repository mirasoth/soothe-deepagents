"""Common LLM rate-limit middleware and shared retry runner.

Provider-agnostic middleware that throttles model calls at the API-call layer
(not at thread/task orchestration level) and applies timeout / 429 retry policy.

Concurrency primitives are safe across multiple asyncio event loops in one
process: per-budget ``asyncio.Semaphore`` instances are keyed by running loop,
and process-wide slots use a threading-coordinated cross-loop semaphore.
"""
# ruff: noqa: ANN401,C901,D102,D103,D105,D107,EM101,FBT001,FBT002,FBT003,PLR0912,PLR0915,PLR2004,SIM105,SIM117,TC002,TC003,TRY003,TRY300

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Self, TypeVar

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from soothe_deepagents.middleware.llm_call_policy import LLMCallPolicyConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _is_api_rate_limit_error(exc: Exception) -> bool:
    for link in _iter_exception_chain(exc):
        if type(link).__name__ == "RateLimitError":
            return True
        response = getattr(link, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
        text = str(link).lower()
        if "429" in text or "rate limit" in text or "throttling" in text:
            return True
    return False


def _is_concurrency_throttle_error(exc: Exception) -> bool:
    """True when a 429 is about in-flight concurrency, not (only) RPM."""
    for link in _iter_exception_chain(exc):
        text = str(link).lower()
        if "concurrency" in text or "throttling" in text:
            return True
        response = getattr(link, "response", None)
        if response is None:
            continue
        try:
            body = response.json()
            error_obj = body.get("error", {}) if isinstance(body, dict) else {}
            message = str(error_obj.get("message", "") or body).lower()
            code = str(error_obj.get("code", "")).lower()
            if "concurrency" in message or "throttling" in message or code == "throttling":
                return True
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return False


def _is_transient_connection_error(exc: Exception) -> bool:
    exc_type_name = type(exc).__name__
    transient_types = {
        "ConnectionError",
        "ConnectError",
        "NetworkError",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "RemoteProtocolError",
        "LocalProtocolError",
        "StreamError",
    }
    if exc_type_name in transient_types:
        return True
    module_name = str(type(exc).__module__)
    if "httpx" in module_name and exc_type_name in {
        "ConnectError",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "StreamConsumed",
        "RemoteProtocolError",
    }:
        return True
    if "aiohttp" in module_name and exc_type_name in {
        "ClientConnectionError",
        "ClientConnectorError",
        "ClientOSError",
        "ClientPayloadError",
        "ClientResponseError",
        "ServerTimeoutError",
        "ClientTimeout",
    }:
        return True
    text = str(exc).lower()
    keywords = (
        "connection error",
        "connection refused",
        "connection reset",
        "connection closed",
        "network unreachable",
        "network error",
        "timeout",
        "timed out",
        "socket error",
        "ssl error",
        "tls error",
        "certificate error",
        "eof occurred in violation of protocol",
        "protocol error",
        "stream error",
        "temporary failure",
    )
    return any(k in text for k in keywords)


def _extract_retry_after_seconds(exc: Exception) -> float | None:
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


def _extract_rate_limit_info(exc: Exception) -> dict[str, Any]:
    result: dict[str, Any] = {
        "retry_after_seconds": None,
        "rpm_limit_hint": None,
        "provider_name": None,
    }
    response = None
    for link in _iter_exception_chain(exc):
        response = getattr(link, "response", None)
        if response is not None:
            break
    if response is None:
        return result

    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            result["retry_after_seconds"] = float(retry_after)
        except (TypeError, ValueError):
            pass

    try:
        body = response.json()
        error_obj = body.get("error", {})
        if result["retry_after_seconds"] is None:
            for key in ("retry_after", "wait_seconds"):
                if key in error_obj:
                    try:
                        result["retry_after_seconds"] = float(error_obj[key])
                        break
                    except (TypeError, ValueError):
                        pass
        rate_limit_obj = error_obj.get("rate_limit", {})
        if isinstance(rate_limit_obj, dict) and "limit" in rate_limit_obj:
            try:
                result["rpm_limit_hint"] = int(rate_limit_obj["limit"])
            except (TypeError, ValueError):
                pass
        message = str(error_obj.get("message", "") or body).lower()
        if "dashscope" in message or "qwen" in message:
            result["provider_name"] = "dashscope"
        elif "zhipu" in message or "glm" in message:
            result["provider_name"] = "zhipu"
        elif "kimi" in message or "moonshot" in message:
            result["provider_name"] = "kimi"
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return result


def estimate_model_request_prompt_chars(request: ModelRequest[Any]) -> int:
    total = 0
    system_prompt = getattr(request, "system_prompt", None)
    if isinstance(system_prompt, str):
        total += len(system_prompt)
    elif system_prompt:
        total += len(str(system_prompt))
    for msg in getattr(request, "messages", []):
        total += len(str(getattr(msg, "content", "") or ""))
    return total


class EnhancedTimeoutError(TimeoutError):
    """Timeout error carrying retry metadata."""

    def __init__(self, timeout_seconds: int, retries: int, prompt_chars: int, thread_id: str) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.prompt_chars = prompt_chars
        self.thread_id = thread_id
        parts = [f"LLM call timed out after {retries} retries", f"({timeout_seconds}s final timeout)"]
        if prompt_chars > 50_000:
            parts.append(f"- large prompt ({prompt_chars:,} chars)")
        super().__init__(" ".join(parts))


def _safe_set_result(fut: asyncio.Future[None]) -> None:
    if not fut.done():
        fut.set_result(None)


class CrossLoopSemaphore:
    """Process-wide semaphore usable from any asyncio event loop.

    ``asyncio.Semaphore`` binds to one event loop; this type coordinates slots
    with a ``threading.Lock`` and wakes waiters via ``call_soon_threadsafe``.

    ``value == 0`` means unlimited (acquire/release are no-ops).
    """

    def __init__(self, value: int) -> None:
        if value < 0:
            raise ValueError("value must be >= 0 (0 = unlimited)")
        self._max = value
        self._value = value
        self._lock = threading.Lock()
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = []

    @property
    def max_value(self) -> int:
        return self._max

    @property
    def unlimited(self) -> bool:
        return self._max == 0

    @property
    def available(self) -> int:
        with self._lock:
            # -1 signals unlimited capacity for stats consumers.
            return -1 if self._max == 0 else self._value

    def set_max(self, new_max: int) -> None:
        """Adjust capacity; wake waiters when capacity increases or becomes unlimited."""
        new_max = max(0, int(new_max))
        with self._lock:
            if new_max == 0:
                self._max = 0
                self._value = 0
                self._wake_all_waiters_unlocked()
                return
            if self._max == 0:
                # Was unlimited; install a finite pool.
                self._max = new_max
                self._value = new_max
                return
            delta = new_max - self._max
            self._max = new_max
            if delta > 0:
                self._value += delta
                self._wake_waiters_unlocked()
            elif delta < 0:
                self._value = max(0, self._value + delta)

    def _wake_all_waiters_unlocked(self) -> None:
        while self._waiters:
            loop, fut = self._waiters.pop(0)
            if fut.done():
                continue
            loop.call_soon_threadsafe(_safe_set_result, fut)

    def _wake_waiters_unlocked(self) -> None:
        while self._waiters and self._value > 0:
            loop, fut = self._waiters.pop(0)
            if fut.done():
                continue
            self._value -= 1
            loop.call_soon_threadsafe(_safe_set_result, fut)

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] | None = None
        with self._lock:
            if self._max == 0:
                return
            if self._value > 0:
                self._value -= 1
                return
            fut = loop.create_future()
            self._waiters.append((loop, fut))
        if fut is None:
            msg = "cross-loop semaphore waiter missing"
            raise RuntimeError(msg)
        try:
            await fut
        except asyncio.CancelledError:
            with self._lock:
                self._waiters = [(waiter_loop, waiter_fut) for waiter_loop, waiter_fut in self._waiters if waiter_fut is not fut]
                if fut.done() and not fut.cancelled():
                    # Slot was granted then cancelled before use — release it.
                    self._release_unlocked()
            raise

    def release(self) -> None:
        with self._lock:
            if self._max == 0:
                return
            self._release_unlocked()

    def _release_unlocked(self) -> None:
        while self._waiters:
            loop, fut = self._waiters.pop(0)
            if fut.done():
                continue
            loop.call_soon_threadsafe(_safe_set_result, fut)
            return
        if self._value < self._max:
            self._value += 1

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        self.release()


@dataclass
class ThreadBudget:
    """Per-budget-key RPM tracker and per-event-loop concurrency slots.

    ``semaphore_max`` caps concurrent in-flight LLM calls for one budget key
    on a single asyncio event loop. Semaphores are not shared across loops.
    """

    rpm_limit: int
    semaphore_max: int
    request_times: list[float] = field(default_factory=list)
    _semaphores: dict[int, asyncio.Semaphore] = field(default_factory=dict, init=False, repr=False)
    _sem_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def semaphore_for_running_loop(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        key = id(loop)
        with self._sem_lock:
            sem = self._semaphores.get(key)
            if sem is None:
                sem = asyncio.Semaphore(self.semaphore_max)
                self._semaphores[key] = sem
            return sem

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        async with self.semaphore_for_running_loop():
            yield

    async def wait_for_rpm_slot(self) -> None:
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60.0]
        if len(self.request_times) >= self.rpm_limit:
            oldest = self.request_times[0]
            wait_seconds = oldest + 60.0 - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = time.time()
                self.request_times = [t for t in self.request_times if now - t < 60.0]

    def record_request(self) -> float:
        now = time.time()
        self.request_times.append(now)
        return now

    def get_stats(self) -> dict[str, Any]:
        now = time.time()
        recent = [t for t in self.request_times if now - t < 60.0]
        with self._sem_lock:
            available = sum(sem._value for sem in self._semaphores.values()) if self._semaphores else self.semaphore_max
            loop_count = len(self._semaphores)
        return {
            "rpm_limit": self.rpm_limit,
            "requests_in_last_minute": len(recent),
            "semaphore_available": available,
            "event_loop_semaphores": loop_count,
        }


def resolve_llm_budget_key(thread_id: str | None) -> str:
    return thread_id or "default"


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    return getattr(config, key, default)


def effective_llm_call_timeout(config: Any, *, timeout_attempts: int, rate_limit_attempts: int) -> int:
    policy = LLMCallPolicyConfig(
        timeout_seconds=float(_cfg_get(config, "call_timeout_seconds", 600)),
        retry_on_timeout=bool(_cfg_get(config, "retry_on_timeout", True)),
        max_timeout_retries=int(_cfg_get(config, "max_timeout_retries", 2)),
        timeout_retry_multiplier=float(_cfg_get(config, "timeout_retry_multiplier", 1.2)),
        retry_on_rate_limit=bool(_cfg_get(config, "retry_on_rate_limit", True)),
        max_rate_limit_retries=int(_cfg_get(config, "max_rate_limit_retries", 3)),
        rate_limit_backoff_base=float(_cfg_get(config, "rate_limit_backoff_base", 2.0)),
        rate_limit_backoff_max=float(_cfg_get(config, "rate_limit_backoff_max", 60.0)),
        respect_retry_after_header=bool(_cfg_get(config, "respect_retry_after_header", True)),
        rate_limit_retry_timeout_seconds=float(_cfg_get(config, "rate_limit_retry_timeout_seconds", 120)),
    )
    timeout = int(policy.timeout_for_attempt(timeout_attempt=timeout_attempts, rate_limit_attempt=rate_limit_attempts))
    cap = int(_cfg_get(config, "call_timeout_max_seconds", timeout))
    return min(timeout, cap)


def calc_rate_limit_backoff(
    attempt: int,
    exc: Exception | None,
    *,
    base: float,
    backoff_max: float,
    respect_retry_after: bool,
) -> float:
    if respect_retry_after and exc is not None:
        retry_after = _extract_retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, backoff_max)
    return min(base * (2**attempt), backoff_max)


def _emit_llm_retry_event(*, attempt: int, max_attempts: int, error_type: str, thread_id: str | None) -> None:
    # Upstream no-op hook. Consumers can wrap/override.
    _ = (attempt, max_attempts, error_type, thread_id)


class LLMRateLimitRegistry:
    """Process-wide registry for per-budget-key RPM budgets and global LLM slots."""

    _shared: LLMRateLimitRegistry | None = None

    def __init__(self) -> None:
        self._rpm_limit_global = 60
        self._concurrent_limit_per_thread = 8
        # 0 = no process-wide concurrent cap (per-budget limits still apply).
        self._global_concurrent_limit = 0
        self._thread_budgets: dict[str, ThreadBudget] = {}
        self._budget_lock = threading.Lock()
        self._global_semaphore = CrossLoopSemaphore(self._global_concurrent_limit)

    @classmethod
    def shared(cls) -> LLMRateLimitRegistry:
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._shared = None

    @property
    def thread_budgets(self) -> dict[str, ThreadBudget]:
        return self._thread_budgets

    @property
    def rpm_limit_global(self) -> int:
        return self._rpm_limit_global

    @property
    def global_concurrent_limit(self) -> int:
        return self._global_concurrent_limit

    @property
    def global_semaphore(self) -> CrossLoopSemaphore:
        return self._global_semaphore

    def update_limits(
        self,
        *,
        requests_per_minute: int,
        concurrent_limit_per_thread: int,
        global_concurrent_limit: int | None = None,
    ) -> None:
        self._rpm_limit_global = requests_per_minute
        self._concurrent_limit_per_thread = concurrent_limit_per_thread
        if global_concurrent_limit is not None:
            self._set_global_concurrent_limit(global_concurrent_limit)

    def _set_global_concurrent_limit(self, new_limit: int) -> None:
        # 0 = unlimited; positive values are hard caps (max 500).
        new_limit = max(0, min(int(new_limit), 500))
        self._global_concurrent_limit = new_limit
        self._global_semaphore.set_max(new_limit)

    def adjust_global_concurrent_limit(self, new_limit: int, reason: str) -> None:
        """Shrink/grow a finite global cap. Never introduces unlimited (0)."""
        _ = reason
        new_limit = max(1, min(int(new_limit), 500))
        if new_limit == self._global_concurrent_limit:
            return
        logger.info(
            "Adjusting global LLM concurrent limit %d -> %d (%s)",
            self._global_concurrent_limit,
            new_limit,
            reason,
        )
        self._set_global_concurrent_limit(new_limit)

    def get_budget(self, thread_id: str) -> ThreadBudget:
        with self._budget_lock:
            if thread_id not in self._thread_budgets:
                active_threads = len(self._thread_budgets)
                thread_rpm = max(self._rpm_limit_global // (active_threads + 1), 10)
                self._thread_budgets[thread_id] = ThreadBudget(
                    rpm_limit=thread_rpm,
                    semaphore_max=self._concurrent_limit_per_thread,
                )
            return self._thread_budgets[thread_id]

    def _redistribute_budgets_unlocked(self) -> None:
        active_threads = len(self._thread_budgets)
        if active_threads <= 0:
            return
        thread_rpm = max(self._rpm_limit_global // active_threads, 10)
        for budget in self._thread_budgets.values():
            budget.rpm_limit = thread_rpm

    def cleanup_thread_budget(self, thread_id: str) -> None:
        with self._budget_lock:
            if thread_id not in self._thread_budgets:
                return
            del self._thread_budgets[thread_id]
            self._redistribute_budgets_unlocked()

    def adjust_rpm_limit(self, new_limit: int, reason: str) -> None:
        _ = reason
        new_limit = max(5, min(new_limit, 10_000))
        with self._budget_lock:
            if new_limit == self._rpm_limit_global:
                return
            self._rpm_limit_global = new_limit
            self._redistribute_budgets_unlocked()


async def run_llm_call_with_policy(
    call: Callable[[], Awaitable[T]],
    *,
    config: Any,
    budget_key: str,
    thread_id: str | None = None,
    prompt_chars: int = 0,
    log_prefix: str = "LLM",
    log: logging.Logger | None = None,
) -> T:
    call_log = log or logger
    registry = LLMRateLimitRegistry.shared()
    budget = registry.get_budget(budget_key)
    telemetry_id = thread_id or budget_key

    timeout_attempts = 0
    rate_limit_attempts = 0
    max_timeout_attempts = int(_cfg_get(config, "max_timeout_retries", 2)) + 1 if bool(_cfg_get(config, "retry_on_timeout", True)) else 1
    max_rate_limit_attempts = int(_cfg_get(config, "max_rate_limit_retries", 3)) + 1 if bool(_cfg_get(config, "retry_on_rate_limit", True)) else 1

    while True:
        eff_timeout = effective_llm_call_timeout(
            config,
            timeout_attempts=timeout_attempts,
            rate_limit_attempts=rate_limit_attempts,
        )
        retry_sleep: float | None = None
        retry_error_type: str | None = None
        retry_attempt = 0
        retry_max = 0

        async with registry.global_semaphore:
            async with budget.acquire():
                await budget.wait_for_rpm_slot()
                try:
                    result = await asyncio.wait_for(call(), timeout=eff_timeout)
                    budget.record_request()
                    return result
                except TimeoutError:
                    timeout_attempts += 1
                    if timeout_attempts >= 2:
                        current_rpm = registry.rpm_limit_global
                        reduced_rpm = max(int(current_rpm * 0.8), 10)
                        registry.adjust_rpm_limit(reduced_rpm, reason="consecutive timeout backpressure")
                    if timeout_attempts < max_timeout_attempts:
                        retry_sleep = 1.0 * timeout_attempts
                        retry_error_type = "timeout"
                        retry_attempt = timeout_attempts
                        retry_max = max_timeout_attempts
                    else:
                        raise EnhancedTimeoutError(
                            timeout_seconds=eff_timeout,
                            retries=max_timeout_attempts - 1,
                            prompt_chars=prompt_chars,
                            thread_id=telemetry_id or budget_key,
                        ) from None
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if _is_api_rate_limit_error(exc):
                        info = _extract_rate_limit_info(exc)
                        if info["rpm_limit_hint"] is not None:
                            registry.adjust_rpm_limit(
                                info["rpm_limit_hint"],
                                reason=f"429 from {info['provider_name'] or 'provider'}",
                            )
                        if _is_concurrency_throttle_error(exc) and registry.global_concurrent_limit > 0:
                            # Only shrink when a finite global cap is configured.
                            # Unlimited (0) stays unlimited; retries/backoff handle 429s.
                            reduced = max(1, registry.global_concurrent_limit - 1)
                            registry.adjust_global_concurrent_limit(
                                reduced,
                                reason=f"429 concurrency from {info['provider_name'] or 'provider'}",
                            )
                        rate_limit_attempts += 1
                        if rate_limit_attempts < max_rate_limit_attempts:
                            retry_sleep = calc_rate_limit_backoff(
                                rate_limit_attempts - 1,
                                exc,
                                base=float(_cfg_get(config, "rate_limit_backoff_base", 2.0)),
                                backoff_max=float(_cfg_get(config, "rate_limit_backoff_max", 60.0)),
                                respect_retry_after=bool(_cfg_get(config, "respect_retry_after_header", True)),
                            )
                            retry_error_type = "rate_limit"
                            retry_attempt = rate_limit_attempts
                            retry_max = max_rate_limit_attempts
                        else:
                            raise
                    elif _is_transient_connection_error(exc):
                        connection_attempts = 0
                        while connection_attempts < 3:
                            connection_attempts += 1
                            await asyncio.sleep(2.0 * connection_attempts)
                            try:
                                result = await asyncio.wait_for(call(), timeout=eff_timeout)
                                budget.record_request()
                                return result
                            except Exception as retry_exc:
                                if not _is_transient_connection_error(retry_exc):
                                    raise
                                exc = retry_exc
                        raise
                    else:
                        raise

        if retry_sleep is not None and retry_error_type is not None:
            call_log.warning(
                "%s %s (attempt %d/%d) - retrying in %.1fs (thread_id=%s)",
                log_prefix,
                "timeout" if retry_error_type == "timeout" else "429",
                retry_attempt,
                retry_max,
                retry_sleep,
                telemetry_id,
            )
            _emit_llm_retry_event(
                attempt=retry_attempt,
                max_attempts=retry_max,
                error_type=retry_error_type,
                thread_id=telemetry_id,
            )
            await asyncio.sleep(retry_sleep)


class LLMRateLimitMiddleware(AgentMiddleware):
    """Rate limiting middleware for model calls."""

    name = "LLMRateLimitMiddleware"

    def __init__(
        self,
        requests_per_minute: int = 120,
        max_concurrent_requests_per_thread: int = 10,
        max_concurrent_requests_global: int = 0,
        call_timeout_seconds: int = 600,
        call_timeout_max_seconds: int = 900,
        retry_on_timeout: bool = True,
        max_timeout_retries: int = 10,
        timeout_retry_multiplier: float = 1.2,
        retry_on_rate_limit: bool = True,
        max_rate_limit_retries: int = 10,
        rate_limit_backoff_base: float = 2.0,
        rate_limit_backoff_max: float = 60.0,
        respect_retry_after_header: bool = True,
        rate_limit_retry_timeout_seconds: int = 120,
    ) -> None:
        super().__init__()
        self._concurrent_limit_per_thread = max_concurrent_requests_per_thread
        self._policy_config = type(
            "_PolicyConfig",
            (),
            {
                "call_timeout_seconds": call_timeout_seconds,
                "call_timeout_max_seconds": max(call_timeout_max_seconds, call_timeout_seconds),
                "retry_on_timeout": retry_on_timeout,
                "max_timeout_retries": max_timeout_retries,
                "timeout_retry_multiplier": timeout_retry_multiplier,
                "retry_on_rate_limit": retry_on_rate_limit,
                "max_rate_limit_retries": max_rate_limit_retries,
                "rate_limit_backoff_base": rate_limit_backoff_base,
                "rate_limit_backoff_max": rate_limit_backoff_max,
                "respect_retry_after_header": respect_retry_after_header,
                "rate_limit_retry_timeout_seconds": rate_limit_retry_timeout_seconds,
            },
        )()
        LLMRateLimitRegistry.shared().update_limits(
            requests_per_minute=requests_per_minute,
            concurrent_limit_per_thread=max_concurrent_requests_per_thread,
            global_concurrent_limit=max_concurrent_requests_global,
        )

    @property
    def _thread_budgets(self) -> dict[str, ThreadBudget]:
        return LLMRateLimitRegistry.shared().thread_budgets

    @property
    def _rpm_limit_global(self) -> int:
        return LLMRateLimitRegistry.shared().rpm_limit_global

    @_rpm_limit_global.setter
    def _rpm_limit_global(self, value: int) -> None:
        LLMRateLimitRegistry.shared()._rpm_limit_global = value

    @staticmethod
    def _thread_id_from_request(request: ModelRequest[Any]) -> str:
        runtime = getattr(request, "runtime", None)
        config = getattr(runtime, "config", None) if runtime is not None else None
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
            if isinstance(configurable, dict):
                thread_id = configurable.get("thread_id")
                if isinstance(thread_id, str) and thread_id:
                    return thread_id
        return "default"

    async def _get_thread_budget(self, thread_id: str) -> ThreadBudget:
        # Async for caller compat; registry lookup is synchronous.
        return LLMRateLimitRegistry.shared().get_budget(thread_id)

    def cleanup_thread_budget(self, thread_id: str) -> None:
        LLMRateLimitRegistry.shared().cleanup_thread_budget(thread_id)

    def adjust_rpm_limit(self, new_limit: int, reason: str) -> None:
        LLMRateLimitRegistry.shared().adjust_rpm_limit(new_limit, reason)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        budget_key = self._thread_id_from_request(request)

        async def _invoke() -> ModelResponse[Any]:
            return await handler(request)

        return await run_llm_call_with_policy(
            _invoke,
            config=self._policy_config,
            budget_key=budget_key,
            thread_id=budget_key,
            prompt_chars=estimate_model_request_prompt_chars(request),
            log_prefix="LLM call",
            log=logger,
        )

    def get_stats(self) -> dict[str, Any]:
        registry = LLMRateLimitRegistry.shared()
        return {
            "mode": "budget_key_and_global",
            "global_rpm_limit": registry.rpm_limit_global,
            "per_thread_concurrent_limit": self._concurrent_limit_per_thread,
            "global_concurrent_limit": registry.global_concurrent_limit,
            "global_slots_available": registry.global_semaphore.available,
            "active_threads": len(registry.thread_budgets),
            "thread_budgets": {k: v.get_stats() for k, v in registry.thread_budgets.items()},
        }
