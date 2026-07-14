"""Regression tests migrated from Soothe middleware suite."""

from __future__ import annotations

import asyncio
import ssl
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from soothe_deepagents.middleware.reliability import (
    InvalidToolHintsMiddleware,
    NetworkToolErrorsMiddleware,
    ToolOutputCapMiddleware,
)
from soothe_deepagents.middleware.tool_timeout import ToolTimeoutMiddleware


@pytest.mark.asyncio
async def test_network_tool_errors_middleware_maps_ssl_error_to_tool_message() -> None:
    middleware = NetworkToolErrorsMiddleware()
    request = SimpleNamespace(
        tool_call={
            "id": "call-1",
            "name": "requests_get",
            "args": {"url": "https://mcap.dev"},
        }
    )

    async def failing_handler(_req: object) -> ToolMessage:
        raise ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED]")

    result = await middleware.awrap_tool_call(request, failing_handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "certificate" in str(result.content).lower()


@pytest.mark.asyncio
async def test_network_tool_errors_middleware_reraises_unrelated_errors() -> None:
    middleware = NetworkToolErrorsMiddleware()
    request = SimpleNamespace(tool_call={"id": "call-2", "name": "grep", "args": {}})

    async def failing_handler(_req: object) -> ToolMessage:
        msg = "not a network error"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="not a network error"):
        await middleware.awrap_tool_call(request, failing_handler)


@pytest.mark.asyncio
async def test_invalid_tool_hints_middleware_appends_hint_for_hallucinated_tool() -> None:
    middleware = InvalidToolHintsMiddleware()
    request = SimpleNamespace(tool_call={"name": "read_command", "args": {"command": "ls"}, "id": "c1"})

    async def handler(_req: object) -> ToolMessage:
        return ToolMessage(
            content="Error: read_command is not a valid tool, try one of [run_command].",
            tool_call_id="c1",
            name="read_command",
            status="error",
        )

    result = await middleware.awrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    assert "Hint:" in str(result.content)
    assert "run_command" in str(result.content)


@pytest.mark.asyncio
async def test_invalid_tool_hints_middleware_enhances_command_wrapped_result() -> None:
    middleware = InvalidToolHintsMiddleware()
    request = SimpleNamespace(tool_call={"name": "read_command", "args": {"command": "ls"}, "id": "c3"})

    async def handler(_req: object) -> Command:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Error: read_command is not a valid tool, try one of [run_command].",
                        tool_call_id="c3",
                        name="read_command",
                        status="error",
                    )
                ]
            }
        )

    result = await middleware.awrap_tool_call(request, handler)
    assert isinstance(result, Command)
    update = result.update
    assert isinstance(update, dict)
    message = update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Hint:" in str(message.content)
    assert "run_command" in str(message.content)


@pytest.mark.asyncio
async def test_tool_output_cap_middleware_truncates_tool_and_command_messages() -> None:
    middleware = ToolOutputCapMiddleware(default_max_chars=20, code_exec_max_chars=12)
    request = SimpleNamespace(tool_call={"name": "run_command", "id": "tc1"})

    async def tool_handler(_req: object) -> ToolMessage:
        return ToolMessage(content="x" * 200, tool_call_id="tc1", name="run_command")

    tool_result = await middleware.awrap_tool_call(request, tool_handler)
    assert isinstance(tool_result, ToolMessage)
    assert len(str(tool_result.content)) <= 12
    assert len(str(tool_result.content)) <= 12

    async def command_handler(_req: object) -> Command:
        return Command(
            update={
                "messages": [
                    ToolMessage(content="y" * 200, tool_call_id="tc2", name="read_file"),
                ]
            }
        )

    command_result = await middleware.awrap_tool_call(
        SimpleNamespace(tool_call={"name": "read_file", "id": "tc2"}),
        command_handler,
    )
    assert isinstance(command_result, Command)
    update = command_result.update
    assert isinstance(update, dict)
    tool_message = update["messages"][0]
    assert isinstance(tool_message, ToolMessage)
    assert len(str(tool_message.content)) <= 20


@pytest.mark.asyncio
async def test_tool_timeout_middleware_timeout_and_counters() -> None:
    middleware = ToolTimeoutMiddleware(default_timeout_seconds=0.05)
    request = SimpleNamespace(tool_call={"name": "slow_tool", "id": "call-1", "args": {}})

    async def slow_handler(_req: object) -> ToolMessage:
        await asyncio.sleep(0.2)
        return ToolMessage(content="ok", tool_call_id="call-1", name="slow_tool")

    result = await middleware.awrap_tool_call(request, slow_handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "timed out" in str(result.content)
    assert middleware.get_timeout_stats()["timeout_count"] == 1


@pytest.mark.asyncio
async def test_tool_timeout_middleware_honors_timeout_arg_for_run_command() -> None:
    middleware = ToolTimeoutMiddleware(
        default_timeout_seconds=5.0,
        honor_timeout_arg_for=frozenset({"run_command"}),
        max_timeout_seconds=0.1,
    )
    request = SimpleNamespace(
        tool_call={
            "name": "run_command",
            "id": "call-2",
            "args": {"command": "sleep 9", "timeout": 3600},
        }
    )

    async def slow_handler(_req: object) -> ToolMessage:
        await asyncio.sleep(0.2)
        return ToolMessage(content="ok", tool_call_id="call-2", name="run_command")

    result = await middleware.awrap_tool_call(request, slow_handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "run_command" in str(result.content)


@pytest.mark.asyncio
async def test_tool_timeout_middleware_batched_requests_bypass_timeout() -> None:
    middleware = ToolTimeoutMiddleware(default_timeout_seconds=0.01)
    request = SimpleNamespace(
        tool_call={"name": "slow_tool", "id": "call-3", "args": {}},
        metadata={"_batched": True},
    )

    async def slow_handler(_req: object) -> ToolMessage:
        await asyncio.sleep(0.05)
        return ToolMessage(content="batched_ok", tool_call_id="call-3", name="slow_tool")

    result = await middleware.awrap_tool_call(request, slow_handler)
    assert isinstance(result, ToolMessage)
    assert result.content == "batched_ok"
    assert middleware.get_timeout_stats()["timeout_count"] == 0
