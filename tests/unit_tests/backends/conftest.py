"""Shared fixtures for backend unit tests."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.constants import CONF
from langgraph.runtime import CONFIG_KEY_RUNTIME, Runtime


@contextmanager
def store_backend_runtime_context():
    """Provide a LangGraph runtime so StoreBackend namespace factories resolve."""
    runtime = Runtime(context=None)
    token = var_child_runnable_config.set({CONF: {CONFIG_KEY_RUNTIME: runtime}})
    try:
        yield runtime
    finally:
        var_child_runnable_config.reset(token)


@pytest.fixture(autouse=True)
def _langgraph_runtime_for_store_backend(request: pytest.FixtureRequest):
    """Auto-enable runtime context for StoreBackend tests outside graph execution."""
    module_name = request.module.__name__
    store_backend_modules = (
        "test_store_backend",
        "test_store_backend_async",
        "test_file_format",
        "test_backwards_compat",
    )
    if not any(module_name.endswith(suffix) for suffix in store_backend_modules):
        yield
        return
    with store_backend_runtime_context():
        yield
