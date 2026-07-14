# Upstream Reuse Matrix

This document records the Soothe middleware split decisions used to upstream
framework-grade behavior into `soothe-deepagents`.

## Upstreamed Here

- `NetworkToolErrorsMiddleware`
  - Generic exception-to-ToolMessage conversion for recoverable network failures.
- `InvalidToolHintsMiddleware`
  - Generic hints for invalid tool-name errors returned by the runtime.
- `ToolOutputCapMiddleware`
  - Generic truncation guard for tool outputs before model context.
- `ToolTimeoutMiddleware`
  - Generic timeout wrapper around tool calls.
- `run_llm_call_with_policy` + `LLMCallPolicyConfig`
  - Provider-agnostic retry/timeout utility for future middleware composition.

## Kept In Soothe

- Identity, policy, workspace context, per-turn model override, and role routing
  middleware remain in Soothe due to daemon/runtime coupling.
- Soothe prompt orchestration and progressive skill/MCP activation remain in
  Soothe due to state-channel and event-system coupling.
- Edit coalescing remains Soothe-specific until deepagents adopts a matching
  batched edit contract.

## Integration Policy

- New reliability middleware in `soothe-deepagents` is opt-in via
  `create_deep_agent(..., reliability_middleware=...)`.
- Default behavior is unchanged when options are omitted.
