"""Unit tests for reliability middleware primitives."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

from soothe_deepagents.middleware.llm_call_policy import LLMCallPolicyConfig, run_llm_call_with_policy
from soothe_deepagents.middleware.reliability import (
    InvalidToolHintsMiddleware,
    NetworkToolErrorsMiddleware,
    ToolOutputCapMiddleware,
)
from soothe_deepagents.middleware.tool_timeout import ToolTimeoutMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


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
async def test_invalid_tool_hints_middleware_appends_actionable_hint() -> None:
    middleware = InvalidToolHintsMiddleware()
    request = SimpleNamespace(tool_call={"name": "read_command", "args": {"command": "ls"}, "id": "call-1"})

    async def handler(_request: object) -> ToolMessage:
        return ToolMessage(content="read_command is not a valid tool", name="read_command", tool_call_id="call-1")

    result = await middleware.awrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    assert "Hint:" in str(result.content)
    assert "run_command" in str(result.content)


@pytest.mark.asyncio
async def test_tool_output_cap_middleware_caps_tool_result_and_model_request_messages() -> None:
    middleware = ToolOutputCapMiddleware(
        default_max_chars=20,
        code_exec_max_chars=12,
    )

    tool_request = SimpleNamespace(tool_call={"name": "run_command", "id": "call-1"})

    async def tool_handler(_request: object) -> ToolMessage:
        return ToolMessage(
            content="abcdefghijklmnopqrstuvwxyz",
            name="run_command",
            tool_call_id="call-1",
        )

    tool_result = await middleware.awrap_tool_call(tool_request, tool_handler)
    assert isinstance(tool_result, ToolMessage)
    assert len(str(tool_result.content)) <= 12

    captured: list[ModelRequest] = []

    async def model_handler(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    request = ModelRequest(
        model=model,
        messages=[
            ToolMessage(content="x" * 200, name="read_file", tool_call_id="call-2"),
            AIMessage(content="next"),
        ],
    )
    await middleware.awrap_model_call(request, model_handler)
    assert captured
    capped_tool_msg = captured[0].messages[0]
    assert isinstance(capped_tool_msg, ToolMessage)
    assert len(str(capped_tool_msg.content)) <= 20


@pytest.mark.asyncio
async def test_tool_timeout_middleware_returns_error_on_timeout() -> None:
    middleware = ToolTimeoutMiddleware(default_timeout_seconds=0.01)
    request = SimpleNamespace(tool_call={"name": "slow_tool", "id": "call-1", "args": {}})

    async def handler(_request: object) -> ToolMessage:
        await asyncio.sleep(0.1)
        return ToolMessage(content="done", name="slow_tool", tool_call_id="call-1")

    result = await middleware.awrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "timed out" in str(result.content)


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
