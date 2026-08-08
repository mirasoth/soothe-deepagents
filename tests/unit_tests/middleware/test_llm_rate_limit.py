"""Regression tests for multi-event-loop rate limiting and global LLM slots."""

from __future__ import annotations

import asyncio
import threading

import pytest

from soothe_deepagents.middleware.llm_rate_limit import (
    CrossLoopSemaphore,
    LLMRateLimitMiddleware,
    LLMRateLimitRegistry,
    run_llm_call_with_policy,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    LLMRateLimitRegistry.reset_for_tests()
    yield
    LLMRateLimitRegistry.reset_for_tests()


class _Policy:
    call_timeout_seconds = 5
    call_timeout_max_seconds = 5
    retry_on_timeout = False
    max_timeout_retries = 0
    timeout_retry_multiplier = 1.0
    retry_on_rate_limit = False
    max_rate_limit_retries = 0
    rate_limit_backoff_base = 1.0
    rate_limit_backoff_max = 1.0
    respect_retry_after_header = False
    rate_limit_retry_timeout_seconds = 1


async def _echo(value: str = "ok") -> str:
    await asyncio.sleep(0.01)
    return value


def test_cross_loop_semaphore_caps_across_event_loops() -> None:
    sem = CrossLoopSemaphore(2)
    started = threading.Barrier(3)
    in_flight = 0
    peak = 0
    lock = threading.Lock()
    errors: list[BaseException] = []

    def worker() -> None:
        nonlocal in_flight, peak

        async def _run() -> None:
            nonlocal in_flight, peak
            started.wait()
            async with sem:
                with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                await asyncio.sleep(0.05)
                with lock:
                    in_flight -= 1

        try:
            asyncio.run(_run())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    assert errors == []
    assert peak <= 2


def test_multi_event_loop_budget_semaphores_do_not_raise() -> None:
    """Shared registry + distinct asyncio loops must not bind-loop RuntimeError."""
    LLMRateLimitRegistry.shared().update_limits(
        requests_per_minute=10_000,
        concurrent_limit_per_thread=4,
        global_concurrent_limit=8,
    )
    errors: list[BaseException] = []
    results: list[str] = []
    barrier = threading.Barrier(4)

    def worker(budget_key: str) -> None:
        async def _run() -> None:
            barrier.wait()
            out = await run_llm_call_with_policy(
                _echo,
                config=_Policy(),
                budget_key=budget_key,
                thread_id=budget_key,
            )
            results.append(out)

        try:
            asyncio.run(_run())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"budget-{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert errors == []
    assert results == ["ok"] * 4
    assert "bound to a different event loop" not in " ".join(str(e) for e in errors)


def test_global_concurrent_limit_zero_is_unlimited() -> None:
    """global_concurrent_limit=0 must not cap across budget keys."""
    LLMRateLimitRegistry.shared().update_limits(
        requests_per_minute=10_000,
        concurrent_limit_per_thread=8,
        global_concurrent_limit=0,
    )
    in_flight = 0
    peak = 0
    lock = threading.Lock()
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    async def _slow() -> str:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        with lock:
            in_flight -= 1
        return "ok"

    def worker(i: int) -> None:
        async def _run() -> None:
            barrier.wait()
            await run_llm_call_with_policy(
                _slow,
                config=_Policy(),
                budget_key=f"k-{i}",
                thread_id=f"k-{i}",
            )

        try:
            asyncio.run(_run())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert errors == []
    assert peak == 4
    assert LLMRateLimitRegistry.shared().global_concurrent_limit == 0
    assert LLMRateLimitRegistry.shared().global_semaphore.available == -1


def test_concurrency_429_does_not_shrink_unlimited_global() -> None:
    registry = LLMRateLimitRegistry.shared()
    registry.update_limits(
        requests_per_minute=10_000,
        concurrent_limit_per_thread=8,
        global_concurrent_limit=0,
    )

    class _Concurrency429Error(Exception):
        pass

    calls = {"n": 0}

    async def _fail_once() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            msg = "Error code: 429 - concurrency allocated quota exceeded. throttling"
            raise _Concurrency429Error(msg)
        return "ok"

    policy = _Policy()
    policy.retry_on_rate_limit = True  # type: ignore[attr-defined]
    policy.max_rate_limit_retries = 2  # type: ignore[attr-defined]
    policy.rate_limit_backoff_base = 0.01  # type: ignore[attr-defined]
    policy.rate_limit_backoff_max = 0.01  # type: ignore[attr-defined]

    async def _run() -> str:
        return await run_llm_call_with_policy(
            _fail_once,
            config=policy,
            budget_key="b1",
        )

    assert asyncio.run(_run()) == "ok"
    assert registry.global_concurrent_limit == 0


def test_global_concurrent_limit_across_budget_keys() -> None:
    LLMRateLimitRegistry.shared().update_limits(
        requests_per_minute=10_000,
        concurrent_limit_per_thread=8,
        global_concurrent_limit=2,
    )
    in_flight = 0
    peak = 0
    lock = threading.Lock()
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    async def _slow() -> str:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        with lock:
            in_flight -= 1
        return "ok"

    def worker(i: int) -> None:
        async def _run() -> None:
            barrier.wait()
            await run_llm_call_with_policy(
                _slow,
                config=_Policy(),
                budget_key=f"k-{i}",
                thread_id=f"k-{i}",
            )

        try:
            asyncio.run(_run())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert errors == []
    assert peak <= 2


def test_middleware_wires_global_limit() -> None:
    mw = LLMRateLimitMiddleware(
        requests_per_minute=100,
        max_concurrent_requests_per_thread=3,
        max_concurrent_requests_global=5,
    )
    stats = mw.get_stats()
    assert stats["global_concurrent_limit"] == 5
    assert stats["per_thread_concurrent_limit"] == 3
    assert LLMRateLimitRegistry.shared().global_concurrent_limit == 5


def test_concurrency_429_shrinks_global_limit() -> None:
    registry = LLMRateLimitRegistry.shared()
    registry.update_limits(
        requests_per_minute=10_000,
        concurrent_limit_per_thread=8,
        global_concurrent_limit=4,
    )

    class _Concurrency429Error(Exception):
        pass

    calls = {"n": 0}

    async def _fail_once() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            msg = "Error code: 429 - concurrency allocated quota exceeded. throttling"
            raise _Concurrency429Error(msg)
        return "ok"

    policy = _Policy()
    policy.retry_on_rate_limit = True  # type: ignore[attr-defined]
    policy.max_rate_limit_retries = 2  # type: ignore[attr-defined]
    policy.rate_limit_backoff_base = 0.01  # type: ignore[attr-defined]
    policy.rate_limit_backoff_max = 0.01  # type: ignore[attr-defined]

    async def _run() -> str:
        return await run_llm_call_with_policy(
            _fail_once,
            config=policy,
            budget_key="b1",
        )

    assert asyncio.run(_run()) == "ok"
    assert registry.global_concurrent_limit == 3


def test_same_budget_key_on_two_loops_uses_distinct_semaphores() -> None:
    registry = LLMRateLimitRegistry.shared()
    registry.update_limits(
        requests_per_minute=10_000,
        concurrent_limit_per_thread=1,
        global_concurrent_limit=8,
    )
    seen: dict[str, int] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        async def _run() -> None:
            barrier.wait()
            budget = registry.get_budget("shared-key")
            sem = budget.semaphore_for_running_loop()
            with lock:
                seen[name] = id(sem)
            async with budget.acquire():
                await asyncio.sleep(0.01)

        asyncio.run(_run())

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert seen["a"] != seen["b"]


def test_cleanup_thread_budget_is_sync_safe() -> None:
    registry = LLMRateLimitRegistry.shared()
    registry.get_budget("t1")
    registry.get_budget("t2")
    assert set(registry.thread_budgets) == {"t1", "t2"}
    registry.cleanup_thread_budget("t1")
    assert set(registry.thread_budgets) == {"t2"}
    # Must not require a running event loop
    registry.adjust_rpm_limit(50, reason="test")
    assert registry.rpm_limit_global == 50
