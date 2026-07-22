"""FilesystemMiddleware binds runtime workspace without host config objects."""

from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage

from soothe_deepagents.middleware.filesystem import FilesystemMiddleware


class _BindableBackend:
    """Minimal backend exposing bind_workspace for duck-typing."""

    def __init__(self) -> None:
        self.bound: list[str] = []

    def bind_workspace(self, workspace: str) -> None:
        self.bound.append(str(workspace))


def test_filesystem_middleware_binds_workspace_from_runtime_config() -> None:
    backend = _BindableBackend()
    middleware = FilesystemMiddleware(backend=backend)  # type: ignore[arg-type]
    runtime = ToolRuntime(
        state={"messages": [AIMessage(content="hi")]},
        context={},
        config={"configurable": {"workspace": "/project-root"}},
        stream_writer=lambda _chunk: None,
        tools=[],
        tool_call_id="call_1",
        store=None,
    )

    resolved = middleware._get_backend(runtime)

    assert resolved is backend
    assert backend.bound == ["/project-root"]


def test_filesystem_middleware_binds_workspace_from_state_when_config_missing() -> None:
    backend = _BindableBackend()
    middleware = FilesystemMiddleware(backend=backend)  # type: ignore[arg-type]
    runtime = ToolRuntime(
        state={"workspace": "/from-state", "messages": []},
        context={},
        config={"configurable": {}},
        stream_writer=lambda _chunk: None,
        tools=[],
        tool_call_id="call_2",
        store=None,
    )

    middleware._get_backend(runtime)

    assert backend.bound == ["/from-state"]


def test_filesystem_middleware_skips_bind_without_workspace() -> None:
    backend = _BindableBackend()
    middleware = FilesystemMiddleware(backend=backend)  # type: ignore[arg-type]
    runtime = ToolRuntime(
        state={"messages": []},
        context={},
        config={"configurable": {}},
        stream_writer=lambda _chunk: None,
        tools=[],
        tool_call_id="call_3",
        store=None,
    )

    middleware._get_backend(runtime)

    assert backend.bound == []


def test_filesystem_middleware_skips_bind_when_backend_has_no_binder() -> None:
    class _PlainBackend:
        pass

    backend = _PlainBackend()
    middleware = FilesystemMiddleware(backend=backend)  # type: ignore[arg-type]
    runtime = ToolRuntime(
        state={},
        context={},
        config={"configurable": {"workspace": "/project"}},
        stream_writer=lambda _chunk: None,
        tools=[],
        tool_call_id="call_4",
        store=None,
    )

    # Must not raise when backend lacks bind_workspace.
    assert middleware._get_backend(runtime) is backend
