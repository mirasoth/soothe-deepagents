"""Reusable reliability middleware primitives for deep agents."""
# ruff: noqa: ANN401,D102,D103,D107,PLR0911,PLW2901,SIM108,TC002,TC003

from __future__ import annotations

import errno
import re
import ssl
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

_INVALID_TOOL_ERROR_MARKER = "is not a valid tool"
_TRUNCATION_SUFFIX = "\n...[truncated for model context]"
_DEFAULT_CODE_EXEC_TOOLS = frozenset({"execute", "run_command", "run_python"})

_SHELL_TOOL_ALIASES = frozenset(
    {
        "read_command",
        "execute_command",
        "shell",
        "bash",
        "cmd",
        "exec",
        "run_shell",
        "terminal",
    }
)
_READ_FILE_ALIASES = frozenset({"cat", "view_file", "view", "show_file"})
_GREP_ALIASES = frozenset({"search", "search_text", "find_text"})
_GLOB_ALIASES = frozenset({"find", "search_files", "find_files"})
_OPERAND_IN_NAME_RE = re.compile(
    r"^(?P<base>ls|read_file|glob|grep|run_command)\s+(?P<operand>.+)$",
    re.IGNORECASE,
)


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        out.append(current)
        current = current.__cause__ or current.__context__
    return out


def is_recoverable_tool_network_error(exc: BaseException) -> bool:
    """Return True when a tool network failure should be surfaced as ToolMessage."""
    for e in _iter_exception_chain(exc):
        if isinstance(e, ConnectionRefusedError):
            return True
        if isinstance(e, OSError) and getattr(e, "errno", None) == errno.ECONNREFUSED:
            return True
        if isinstance(e, ssl.SSLCertVerificationError):
            return True
        typ = type(e).__name__
        if typ in {"ClientConnectorCertificateError", "CertificateError"}:
            return True
        msg = str(e).lower()
        if "certificate verify failed" in msg or "certification_verify_failed" in msg:
            return True
    return False


def format_tool_network_error(exc: BaseException) -> str:
    """Format recoverable tool network errors with short, actionable guidance."""
    combined = " ".join(str(e) for e in _iter_exception_chain(exc))
    if "certificate verify failed" in combined.lower():
        return "TLS certificate verification failed for the requested URL. Try another source or adjust TLS verification settings if appropriate."
    connect_call = re.search(r"Connect call failed\s*\(\s*'([^']+)'\s*,\s*(\d+)", combined)
    if connect_call:
        host, port = connect_call.group(1), connect_call.group(2)
        return f"Connection refused to {host}:{port} - nothing is listening there. Start the service or correct the host/port."
    return str(exc)


def is_invalid_tool_error(content: str) -> bool:
    return _INVALID_TOOL_ERROR_MARKER in content


def _extract_tool_message_content(result: ToolMessage | Command[Any]) -> str | None:
    if isinstance(result, ToolMessage):
        return str(result.content or "")
    if isinstance(result, Command):
        update = result.update
        if not isinstance(update, dict):
            return None
        messages = update.get("messages")
        if not isinstance(messages, list):
            return None
        for msg in messages:
            if isinstance(msg, ToolMessage):
                return str(msg.content or "")
    return None


def _append_hint_to_tool_result(
    result: ToolMessage | Command[Any],
    *,
    hint: str,
) -> ToolMessage | Command[Any]:
    if isinstance(result, ToolMessage):
        content = str(result.content or "")
        if hint in content:
            return result
        return result.model_copy(update={"content": f"{content.rstrip()}\n\n{hint}"})
    if isinstance(result, Command):
        update = result.update
        if not isinstance(update, dict):
            return result
        messages = update.get("messages")
        if not isinstance(messages, list):
            return result
        changed = False
        patched: list[Any] = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = str(msg.content or "")
                if hint not in content:
                    msg = msg.model_copy(update={"content": f"{content.rstrip()}\n\n{hint}"})
                    changed = True
            patched.append(msg)
        if changed:
            return Command(update={**update, "messages": patched})
    return result


def _coerce_args(args: Any) -> dict[str, Any]:
    return args if isinstance(args, dict) else {}


def _sanitize_hallucinated_tool_name(tool_name: str) -> tuple[str, dict[str, Any]]:
    raw = (tool_name or "").strip()
    if not raw:
        return raw, {}
    match = _OPERAND_IN_NAME_RE.match(raw)
    if not match:
        return raw, {}
    base = match.group("base").lower()
    operand = match.group("operand").strip().strip("'\"")
    if base == "ls":
        return "ls", {"path": operand}
    if base == "read_file":
        return "read_file", {"file_path": operand}
    if base == "glob":
        return "glob", {"pattern": operand}
    if base == "grep":
        return "grep", {"pattern": operand}
    if base == "run_command":
        return "run_command", {"command": operand}
    return base, {}


def suggest_invalid_tool_hint(tool_name: str, args: Any) -> str | None:
    normalized_name, embedded_args = _sanitize_hallucinated_tool_name(tool_name)
    merged_args = {**embedded_args, **_coerce_args(args)}
    lower = normalized_name.lower()

    if lower in _SHELL_TOOL_ALIASES:
        if merged_args.get("command") is not None:
            return "Hint: use run_command with the same command argument for shell output."
        return "Hint: use run_command for shell commands, or grep to search file contents."
    if lower in _READ_FILE_ALIASES:
        return "Hint: use read_file with file_path."
    if lower in _GREP_ALIASES:
        return "Hint: use grep with pattern (and optional path/glob)."
    if lower in _GLOB_ALIASES:
        return "Hint: use glob with a path pattern to find files."
    if embedded_args and lower == "run_command":
        return "Hint: use run_command with command in args, not in the tool name."
    if embedded_args and lower == "read_file":
        return "Hint: use read_file with file_path in args, not in the tool name."
    return None


def _truncate_content(content: Any, max_chars: int) -> Any:
    if isinstance(content, str):
        text = content
    else:
        text = str(content)
    if len(text) <= max_chars:
        return content
    if max_chars <= len(_TRUNCATION_SUFFIX):
        return text[:max_chars]
    return text[: max_chars - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


def _truncate_tool_message(msg: ToolMessage, max_chars: int) -> ToolMessage:
    capped = _truncate_content(msg.content, max_chars)
    if capped is msg.content:
        return msg
    return msg.model_copy(update={"content": capped})


class NetworkToolErrorsMiddleware(AgentMiddleware):
    """Surface recoverable network failures as ToolMessage instead of raising."""

    name = "NetworkToolErrorsMiddleware"

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        try:
            return await handler(request)
        except Exception as exc:
            if not is_recoverable_tool_network_error(exc):
                raise
            tool_call = request.tool_call or {}
            tool_name = str(tool_call.get("name", "tool"))
            return ToolMessage(
                content=f"Error: {format_tool_network_error(exc)}",
                tool_call_id=tool_call.get("id"),
                name=tool_name,
                status="error",
            )


class InvalidToolHintsMiddleware(AgentMiddleware):
    """Append actionable hints when the model invokes an invalid tool name."""

    name = "InvalidToolHintsMiddleware"

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call or {}
        tool_name = str(tool_call.get("name", ""))
        args = tool_call.get("args", {})
        result = await handler(request)
        content = _extract_tool_message_content(result)
        if not content or not is_invalid_tool_error(content):
            return result
        hint = suggest_invalid_tool_hint(tool_name, args)
        if not hint:
            return result
        return _append_hint_to_tool_result(result, hint=hint)


class ToolOutputCapMiddleware(AgentMiddleware):
    """Truncate large tool outputs before they enter model context."""

    name = "ToolOutputCapMiddleware"

    def __init__(
        self,
        *,
        default_max_chars: int,
        code_exec_max_chars: int | None = None,
        code_exec_tools: frozenset[str] | None = None,
    ) -> None:
        super().__init__()
        self._default_max_chars = default_max_chars
        self._code_exec_max_chars = code_exec_max_chars
        self._code_exec_tools = code_exec_tools or _DEFAULT_CODE_EXEC_TOOLS

    def _cap_for_tool(self, tool_name: str) -> int:
        if self._code_exec_max_chars is not None and tool_name in self._code_exec_tools:
            return self._code_exec_max_chars
        return self._default_max_chars

    def _cap_messages(self, messages: list[Any]) -> list[Any]:
        out: list[Any] = []
        changed = False
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = str(getattr(msg, "name", None) or "")
                capped = _truncate_tool_message(msg, self._cap_for_tool(tool_name))
                if capped is not msg:
                    changed = True
                out.append(capped)
            else:
                out.append(msg)
        return out if changed else messages

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Cap tool-call outputs before they are written back to state."""
        result = await handler(request)
        tool_call = request.tool_call or {}
        tool_name = str(tool_call.get("name", ""))
        max_chars = self._cap_for_tool(tool_name)

        if isinstance(result, ToolMessage):
            return _truncate_tool_message(result, max_chars)

        if isinstance(result, Command):
            update = result.update
            if isinstance(update, dict):
                messages = update.get("messages")
                if isinstance(messages, list):
                    patched: list[Any] = []
                    for msg in messages:
                        if isinstance(msg, ToolMessage):
                            msg_tool_name = str(getattr(msg, "name", None) or tool_name)
                            patched.append(_truncate_tool_message(msg, self._cap_for_tool(msg_tool_name)))
                        else:
                            patched.append(msg)
                    return Command(update={**update, "messages": patched})
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Cap historical tool messages before sending a model request."""
        messages = list(getattr(request, "messages", None) or [])
        capped = self._cap_messages(messages)
        if capped is not messages:
            request = request.override(messages=capped)
        return await handler(request)
