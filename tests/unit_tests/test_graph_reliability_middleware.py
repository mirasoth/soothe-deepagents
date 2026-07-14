"""Graph wiring tests for reliability middleware options."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from soothe_deepagents.graph import create_deep_agent
from soothe_deepagents.middleware.reliability import (
    InvalidToolHintsMiddleware,
    NetworkToolErrorsMiddleware,
    ToolOutputCapMiddleware,
)
from soothe_deepagents.middleware.tool_timeout import ToolTimeoutMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def _middleware_names(middleware: list[object]) -> list[str]:
    return [type(m).__name__ for m in middleware]


class _StubSummarizationMW(AgentMiddleware):
    pass


def test_create_deep_agent_does_not_add_reliability_middleware_by_default() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    with (
        patch("soothe_deepagents.graph.create_agent") as mock_create,
        patch(
            "soothe_deepagents.graph.create_summarization_middleware",
            return_value=_StubSummarizationMW(),
        ),
    ):
        mock_create.return_value.with_config.return_value = MagicMock()
        create_deep_agent(model=model)
        middleware = mock_create.call_args.kwargs["middleware"]
        names = _middleware_names(middleware)
        assert "ToolTimeoutMiddleware" not in names
        assert "NetworkToolErrorsMiddleware" not in names
        assert "InvalidToolHintsMiddleware" not in names
        assert "ToolOutputCapMiddleware" not in names


def test_create_deep_agent_adds_opt_in_reliability_middleware_in_order() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    with (
        patch("soothe_deepagents.graph.create_agent") as mock_create,
        patch(
            "soothe_deepagents.graph.create_summarization_middleware",
            return_value=_StubSummarizationMW(),
        ),
    ):
        mock_create.return_value.with_config.return_value = MagicMock()
        create_deep_agent(
            model=model,
            reliability_middleware={
                "tool_timeout_seconds": 1.0,
                "network_tool_errors": True,
                "invalid_tool_hints": True,
                "tool_output_cap_chars": 1200,
                "tool_output_cap_code_exec_chars": 600,
                "tool_timeout_honor_timeout_arg_for": ["execute"],
            },
        )
        middleware = mock_create.call_args.kwargs["middleware"]
        timeout_idx = next(i for i, m in enumerate(middleware) if isinstance(m, ToolTimeoutMiddleware))
        network_idx = next(i for i, m in enumerate(middleware) if isinstance(m, NetworkToolErrorsMiddleware))
        invalid_idx = next(i for i, m in enumerate(middleware) if isinstance(m, InvalidToolHintsMiddleware))
        output_cap_idx = next(i for i, m in enumerate(middleware) if isinstance(m, ToolOutputCapMiddleware))
        assert timeout_idx < network_idx < invalid_idx < output_cap_idx
