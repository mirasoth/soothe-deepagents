"""Deep Agents package."""

from soothe_deepagents._version import __version__
from soothe_deepagents.graph import (
    DeepAgentState,
    SystemPromptConfig,
    create_deep_agent,
)
from soothe_deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware
from soothe_deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission, FsToolName
from soothe_deepagents.middleware.memory import MemoryMiddleware
from soothe_deepagents.middleware.rubric import RubricMiddleware
from soothe_deepagents.middleware.subagents import (
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from soothe_deepagents.profiles.harness.harness_profiles import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    HarnessProfileConfig,
    register_harness_profile,
)
from soothe_deepagents.profiles.provider.provider_profiles import (
    ProviderProfile,
    register_provider_profile,
)

__all__ = [
    "AsyncSubAgent",
    "AsyncSubAgentMiddleware",
    "CompiledSubAgent",
    "DeepAgentState",
    "FilesystemMiddleware",
    "FilesystemPermission",
    "FsToolName",
    "GeneralPurposeSubagentProfile",
    "HarnessProfile",
    "HarnessProfileConfig",
    "MemoryMiddleware",
    "ProviderProfile",
    "RubricMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "SystemPromptConfig",
    "__version__",
    "create_deep_agent",
    "register_harness_profile",
    "register_provider_profile",
]
