# soothe-deepagents Development Rules

## MUST: Verification Before Finishing Implementation

Before marking any coding task as complete, you MUST run all of the following:

1. Format checks/fixes
2. Lint checks
3. Unit tests

Minimum required commands:

```bash
python -m ruff format .
python -m ruff check .
pytest tests/unit_tests
```

## Completion Gate

Do not consider implementation finished until:

- formatting passes
- lint passes
- relevant unit tests pass (and at least `tests/unit_tests` succeeds for CI baseline)

If any command fails, continue fixing code and rerun verification until green.
