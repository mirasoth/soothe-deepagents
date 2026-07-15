"""Middleware to patch dangling tool calls in the messages history."""

from typing import Annotated, Any, NotRequired, cast

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


class PatchToolCallsState(AgentState):
    """Private rolling cursor and answered IDs for incremental patch scans."""

    _patch_scan_cursor: NotRequired[Annotated[int, PrivateStateAttr]]
    _patch_answered_tool_ids: NotRequired[Annotated[list[str], PrivateStateAttr]]


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls in the messages history."""

    state_schema = PatchToolCallsState

    def before_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002, C901, PLR0912
        """Before the agent runs, handle dangling tool calls from any AIMessage."""
        state_typed = cast("PatchToolCallsState", state)
        messages = state_typed["messages"]
        if not messages:
            return None

        total_messages = len(messages)
        raw_cursor = state_typed.get("_patch_scan_cursor", 0)
        cursor = raw_cursor if isinstance(raw_cursor, int) else 0
        if cursor < 0 or cursor > total_messages:
            cursor = 0
        raw_answered_ids = state_typed.get("_patch_answered_tool_ids", [])
        answered_ids = {tool_id for tool_id in raw_answered_ids if isinstance(tool_id, str)} if isinstance(raw_answered_ids, list) else set[str]()

        # Include the previous message in the scan window so edge transitions
        # across turns are still validated incrementally.
        scan_start = max(cursor - 1, 0)
        scanned_messages = messages[scan_start:]
        if not any(isinstance(msg, AIMessage) for msg in scanned_messages):
            return None
        for msg in scanned_messages:
            if msg.type == "tool" and isinstance(msg, ToolMessage) and msg.tool_call_id is not None:
                answered_ids.add(msg.tool_call_id)

        if not any(
            tool_call["id"] is not None and tool_call["id"] not in answered_ids
            for msg in scanned_messages
            if isinstance(msg, AIMessage)
            for tool_call in (*msg.tool_calls, *msg.invalid_tool_calls)
        ):
            update: dict[str, Any] = {}
            if cursor != total_messages:
                update["_patch_scan_cursor"] = total_messages
            sorted_answered_ids = sorted(answered_ids)
            if raw_answered_ids != sorted_answered_ids:
                update["_patch_answered_tool_ids"] = sorted_answered_ids
            return update or None

        patched_scanned_messages: list[AnyMessage] = []
        for msg in scanned_messages:
            patched_scanned_messages.append(msg)
            if not isinstance(msg, AIMessage):
                continue
            for tool_call in (*msg.tool_calls, *msg.invalid_tool_calls):
                tool_call_id = tool_call["id"]
                if tool_call_id is None or tool_call_id in answered_ids:
                    continue
                name = tool_call["name"] or "unknown"
                if tool_call.get("type") == "invalid_tool_call":
                    content = f"Tool call {name} with id {tool_call_id} could not be executed - arguments were malformed or truncated."
                else:
                    content = f"Tool call {name} with id {tool_call_id} was cancelled - another message came in before it could be completed."
                answered_ids.add(tool_call_id)
                patched_scanned_messages.append(ToolMessage(content=content, name=name, tool_call_id=tool_call_id))

        patched_messages = [*messages[:scan_start], *patched_scanned_messages]

        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *patched_messages],
            "_patch_scan_cursor": len(patched_messages),
            "_patch_answered_tool_ids": sorted(answered_ids),
        }
