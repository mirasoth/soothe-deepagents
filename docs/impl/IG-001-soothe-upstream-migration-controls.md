# IG-001: Upstream migration controls for Soothe integration

## Context

Soothe currently carries runtime monkeypatches under:

- `soothe/foundation/coreagent/coding/patches/execute_filter.py`
- `soothe/foundation/coreagent/coding/patches/task_tool.py`
- `soothe/foundation/coreagent/coding/patches/summarization.py`

Those patches target `soothe_deepagents` internals and are brittle across upstream changes.

## Goal

Provide first-class upstream controls in `soothe-deepagents` so Soothe can migrate off monkeypatches and rely on supported APIs.

## Scope

1. Add `create_deep_agent(...)` controls to shape the default stack without patching internals.
2. Add `SubAgentMiddleware`/task-tool controls for parent-owned state merge and workspace propagation.
3. Add tests for all new controls.

Out of scope:

- Removing Soothe-side patches in this repository.
- Introducing behavior changes that break default SDK behavior.

## Design

### 1) create_deep_agent controls

Add optional parameters:

- `enable_general_purpose_subagent: bool | None = None`
  - `None`: keep current behavior (profile-driven).
  - `True`/`False`: explicit per-call override.
- `filesystem_tools: Sequence[FsToolName] | Literal["all"] | None = None`
  - Forwarded to all default `FilesystemMiddleware(...)` instances (main + generated subagents).
  - Enables callers to exclude `execute` without monkeypatching middleware `__init__`.

### 2) task tool state controls

Add optional parameters:

- `parent_owned_state_keys: frozenset[str] | None = None` on `SubAgentMiddleware`.
- Thread this through to `_build_task_tool(...)`.

Behavior:

- Subagent result merge excludes:
  - existing excluded keys
  - private state keys
  - parent-owned state keys
- Prevents invalid concurrent writes for parent-managed channels (e.g. `workspace` LastValue channel).

### 3) workspace propagation for subagents

In task invocation state prep:

- If `runtime.config.configurable.workspace` exists and subagent state does not already contain `workspace`, inject it into subagent state.

This keeps workspace availability consistent for subagents even when workspace is carried in config instead of graph state.

## Validation plan

Add/extend unit tests to cover:

1. `create_deep_agent(..., filesystem_tools=...)` removes `execute` from visible/default filesystem tool sets.
2. `enable_general_purpose_subagent=False` suppresses auto-added default general-purpose subagent.
3. `SubAgentMiddleware(..., parent_owned_state_keys={"workspace"})` drops `workspace` from subagent result state merge.
4. Task tool injects `workspace` from `runtime.config.configurable.workspace` when absent from runtime state.

Run required repository checks:

```bash
python -m ruff format .
python -m ruff check .
pytest tests/unit_tests
```

## Rollout

Backward-compatible release:

- All new parameters are optional and default to current behavior.
- No existing caller changes are required.
