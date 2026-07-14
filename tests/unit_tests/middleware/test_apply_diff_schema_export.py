"""Regression tests for ApplyDiffSchema public export surface."""

from pydantic import BaseModel

from soothe_deepagents.backends.state import StateBackend
from soothe_deepagents.middleware import ApplyDiffSchema as PackageApplyDiffSchema
from soothe_deepagents.middleware.filesystem import ApplyDiffSchema, FilesystemMiddleware


def test_apply_diff_schema_is_public_basemodel() -> None:
    assert PackageApplyDiffSchema is ApplyDiffSchema
    assert issubclass(ApplyDiffSchema, BaseModel)
    assert {"file_path", "diff"} <= set(ApplyDiffSchema.model_fields)
    for field_name, field_info in ApplyDiffSchema.model_fields.items():
        assert field_info.description, f"{field_name} missing description"


def test_filesystem_middleware_exposes_apply_diff_tool() -> None:
    middleware = FilesystemMiddleware(backend=StateBackend())
    assert any(tool.name == "apply_diff" for tool in middleware.tools)
    apply_diff = next(tool for tool in middleware.tools if tool.name == "apply_diff")
    assert apply_diff.args_schema is ApplyDiffSchema
