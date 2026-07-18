from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from soothe_deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.utils import (
    TOY_BASKETBALL_RESEARCH,
    ResearchMiddleware,
    ResearchMiddlewareWithTools,
    WeatherToolMiddleware,
    assert_all_deepagent_qualities,
    get_soccer_scores,
    get_weather,
    sample_tool,
)


def _parent_model_with_task_call(subagent_type: str, call_id: str = "call_task") -> GenericFakeChatModel:
    """Build a fake parent model that emits one `task` tool call then a final text response."""
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "Delegate to subagent",
                                "subagent_type": subagent_type,
                            },
                            "id": call_id,
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Done."),
            ]
        )
    )


def _parent_model_with_parallel_task_calls(call_specs: list[tuple[str, str]]) -> GenericFakeChatModel:
    """Build a fake parent model that emits multiple parallel `task` tool calls then a final response."""
    tool_calls = [
        {
            "name": "task",
            "args": {"description": "Delegate to subagent", "subagent_type": stype},
            "id": cid,
            "type": "tool_call",
        }
        for stype, cid in call_specs
    ]
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=tool_calls),
                AIMessage(content="Done."),
            ]
        )
    )


def _subagent_model(final_text: str) -> GenericFakeChatModel:
    """Build a fake subagent model that emits a single final text response."""
    return GenericFakeChatModel(messages=iter([AIMessage(content=final_text)]))


class TestDeepAgents:
    def test_deep_agent_with_subagents(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [get_weather],
                "model": _subagent_model("The weather in Tokyo is sunny."),
            }
        ]
        agent = create_deep_agent(
            model=_parent_model_with_task_call("weather_agent"),
            tools=[sample_tool],
            subagents=subagents,
        )
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "weather_agent" for tool_call in tool_calls)

    def test_deep_agent_with_subagents_gen_purpose(self):
        subagents = [
            {
                "name": "general-purpose",
                "description": "Use this agent for general purpose tasks",
                "system_prompt": "You are a general purpose agent.",
                "tools": [sample_tool],
                "model": _subagent_model("Sample tool result retrieved."),
            }
        ]
        agent = create_deep_agent(
            model=_parent_model_with_task_call("general-purpose"),
            tools=[sample_tool],
            subagents=subagents,
        )
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [HumanMessage(content="Use the general purpose subagent to call the sample tool")]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "general-purpose" for tool_call in tool_calls)

    def test_deep_agent_with_subagents_with_middleware(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [],
                "model": _subagent_model("The weather in Tokyo is sunny."),
                "middleware": [WeatherToolMiddleware()],
            }
        ]
        agent = create_deep_agent(
            model=_parent_model_with_task_call("weather_agent"),
            tools=[sample_tool],
            subagents=subagents,
        )
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "weather_agent" for tool_call in tool_calls)

    def test_deep_agent_with_custom_subagents(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [get_weather],
                "model": _subagent_model("The weather in Tokyo is sunny."),
            },
            {
                "name": "soccer_agent",
                "description": "Use this agent to get the latest soccer scores",
                "runnable": create_agent(
                    model=_subagent_model("Manchester City won 3-1."),
                    tools=[get_soccer_scores],
                    system_prompt="You are a soccer agent.",
                ),
            },
        ]
        agent = create_deep_agent(
            model=_parent_model_with_parallel_task_calls([("weather_agent", "call_weather"), ("soccer_agent", "call_soccer")]),
            tools=[sample_tool],
            subagents=subagents,
        )
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [HumanMessage(content="Look up the weather in Tokyo, and the latest scores for Manchester City!")]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "weather_agent" for tool_call in tool_calls)
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "soccer_agent" for tool_call in tool_calls)

    @pytest.mark.xfail(strict=False, reason="Subagent middleware double-writes extended state keys in same step")
    def test_deep_agent_with_extended_state_and_subagents(self):
        subagents = [
            {
                "name": "basketball_info_agent",
                "description": "Use this agent to get surface level info on any basketball topic",
                "system_prompt": "You are a basketball info agent.",
                "middleware": [ResearchMiddlewareWithTools()],
            }
        ]
        agent = create_deep_agent(
            model=_parent_model_with_task_call("basketball_info_agent"),
            tools=[sample_tool],
            subagents=subagents,
            middleware=[ResearchMiddleware()],
        )
        assert_all_deepagent_qualities(agent)
        assert "research" in agent.stream_channels
        result = agent.invoke({"messages": [HumanMessage(content="Get surface level info on lebron james")]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "basketball_info_agent" for tool_call in tool_calls)
        assert TOY_BASKETBALL_RESEARCH in result["research"]

    def test_deep_agent_with_subagents_no_tools(self):
        subagents = [
            {
                "name": "basketball_info_agent",
                "description": "Use this agent to get surface level info on any basketball topic",
                "system_prompt": "You are a basketball info agent.",
                "model": _subagent_model("LeBron James is a basketball player."),
            }
        ]
        agent = create_deep_agent(
            model=_parent_model_with_task_call("basketball_info_agent"),
            tools=[sample_tool],
            subagents=subagents,
        )
        assert_all_deepagent_qualities(agent)
        result = agent.invoke(
            {"messages": [HumanMessage(content="Use the basketball info subagent to call the sample tool")]},
        )
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "basketball_info_agent" for tool_call in tool_calls)

    def test_response_format_tool_strategy(self):
        class StructuredOutput(BaseModel):
            pokemon: list[str]

        agent = create_deep_agent(
            model=GenericFakeChatModel(
                messages=iter(
                    [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "StructuredOutput",
                                    "args": {"pokemon": ["Bulbasaur", "Charmander", "Squirtle"]},
                                    "id": "call_struct",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                    ]
                )
            ),
            response_format=ToolStrategy(schema=StructuredOutput),
        )
        response = agent.invoke({"messages": [{"role": "user", "content": "Who are all of the Kanto starters?"}]})
        structured_output = response["structured_response"]
        assert len(structured_output.pokemon) == 3
