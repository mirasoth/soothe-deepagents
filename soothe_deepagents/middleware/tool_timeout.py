"""Generic tool timeout middleware for deep agents."""
# ruff: noqa: TC003

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest
    from langgraph.types import Command


def _parse_positive_timeout(value: object) -> float | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if not isinstance(value, int | float | str):
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


class ToolTimeoutMiddleware(AgentMiddleware):
    """Wrap tool calls with configurable timeout safeguards.

    The middleware is intentionally runtime-agnostic and returns a ToolMessage on
    timeout so the model can recover instead of failing the whole step.
    """

    name = "ToolTimeoutMiddleware"

    def __init__(
        self,
        *,
        default_timeout_seconds: float,
        per_tool_timeout_seconds: Mapping[str, float] | None = None,
        skip_tools: frozenset[str] | None = None,
        honor_timeout_arg_for: frozenset[str] | None = None,
        max_timeout_seconds: float | None = None,
    ) -> None:
        """Initialize timeout behavior for tool calls.

        Args:
            default_timeout_seconds: Fallback timeout for tools without overrides.
            per_tool_timeout_seconds: Optional per-tool timeout mapping.
            skip_tools: Tool names excluded from timeout wrapping.
            honor_timeout_arg_for: Tool names allowed to honor input `timeout`.
            max_timeout_seconds: Optional global clamp for resolved timeout values.
        """
        super().__init__()
        self._default_timeout_seconds = float(default_timeout_seconds)
        self._per_tool_timeout_seconds = dict(per_tool_timeout_seconds or {})
        self._skip_tools = skip_tools or frozenset()
        self._honor_timeout_arg_for = honor_timeout_arg_for or frozenset()
        self._max_timeout_seconds = float(max_timeout_seconds) if max_timeout_seconds else None
        self._timeout_count = 0

    def _resolve_timeout(self, tool_name: str, tool_args: object) -> float:
        if tool_name in self._honor_timeout_arg_for and isinstance(tool_args, dict):
            requested = _parse_positive_timeout(tool_args.get("timeout"))
            if requested is not None:
                if self._max_timeout_seconds is not None:
                    return min(requested, self._max_timeout_seconds)
                return requested

        timeout = self._per_tool_timeout_seconds.get(tool_name, self._default_timeout_seconds)
        if self._max_timeout_seconds is not None:
            return min(timeout, self._max_timeout_seconds)
        return timeout

    @staticmethod
    def _timeout_message(tool_name: str, timeout_seconds: float) -> str:
        return (
            f"Error: Tool '{tool_name}' timed out after {timeout_seconds:.1f}s. "
            "Try narrowing scope, increasing timeout for bounded work, or using a background flow."
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[object]],
    ) -> ToolMessage | Command[object]:
        """Run synchronous tool calls with timeout protection."""
        metadata = getattr(request, "metadata", None) or {}
        if isinstance(metadata, dict) and metadata.get("_batched"):
            return handler(request)

        tool_call = request.tool_call or {}
        tool_name = str(tool_call.get("name", ""))
        if tool_name in self._skip_tools:
            return handler(request)

        timeout = self._resolve_timeout(tool_name, tool_call.get("args", {}))
        tool_call_id = tool_call.get("id")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(handler, request)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                self._timeout_count += 1
                return ToolMessage(
                    content=self._timeout_message(tool_name, timeout),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[object]]],
    ) -> ToolMessage | Command[object]:
        """Run asynchronous tool calls with timeout protection."""
        metadata = getattr(request, "metadata", None) or {}
        if isinstance(metadata, dict) and metadata.get("_batched"):
            return await handler(request)

        tool_call = request.tool_call or {}
        tool_name = str(tool_call.get("name", ""))
        if tool_name in self._skip_tools:
            return await handler(request)

        timeout = self._resolve_timeout(tool_name, tool_call.get("args", {}))
        tool_call_id = tool_call.get("id")

        try:
            async with asyncio.timeout(timeout):
                return await handler(request)
        except TimeoutError:
            self._timeout_count += 1
            return ToolMessage(
                content=self._timeout_message(tool_name, timeout),
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
            )

    def get_timeout_stats(self) -> dict[str, int]:
        """Return timeout counters for observability and tests."""
        return {"timeout_count": self._timeout_count}
