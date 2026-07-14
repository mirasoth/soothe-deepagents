"""Focused reliability tests that complement migrated regression coverage.

Most middleware behavior is covered in
``test_migrated_middleware_regressions.py``. Keep this file for non-overlapping
checks only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from soothe_deepagents.middleware.llm_call_policy import LLMCallPolicyConfig, run_llm_call_with_policy
from soothe_deepagents.middleware.reliability import NetworkToolErrorsMiddleware


@pytest.mark.asyncio
async def test_network_tool_errors_middleware_returns_tool_message_for_connection_refused() -> None:
    middleware = NetworkToolErrorsMiddleware()
    request = SimpleNamespace(tool_call={"name": "fetch_url", "id": "call-1"})

    async def handler(_request: object) -> ToolMessage:
        error_message = "Connect call failed ('127.0.0.1', 8080)"
        raise ConnectionRefusedError(error_message)

    result = await middleware.awrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "Connection refused" in str(result.content)


@pytest.mark.asyncio
async def test_llm_call_policy_retries_rate_limit_then_succeeds() -> None:
    class RateLimitError(Exception):
        pass

    attempts = {"count": 0}

    async def call() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            error_message = "rate limit"
            raise RateLimitError(error_message)
        return "ok"

    result = await run_llm_call_with_policy(
        call,
        config=LLMCallPolicyConfig(
            timeout_seconds=0.5,
            retry_on_rate_limit=True,
            max_rate_limit_retries=2,
            rate_limit_backoff_base=0.001,
            rate_limit_backoff_max=0.01,
        ),
    )
    assert result == "ok"
    assert attempts["count"] == 2
