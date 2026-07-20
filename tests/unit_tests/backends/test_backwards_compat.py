"""Backwards compatibility tests for file format after the API cut.

V1 and v2 writes both persist plain `str` content. Legacy `list[str]` items
seeded in the store still raise `TypeError` when materialized.
"""

import pytest
from langgraph.store.memory import InMemoryStore

from soothe_deepagents.backends.store import StoreBackend
from soothe_deepagents.backends.utils import _to_legacy_file_data, create_file_data


class TestV1WriteStorageShape:
    """V1 writes store plain str payloads like v2."""

    def test_v1_write_stores_str_content(self) -> None:
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format="v1")

        result = be.write("/project/main.py", "import os\nprint('hello')\n")
        assert result.error is None

        item = mem_store.get(("filesystem",), "/project/main.py")
        assert isinstance(item.value["content"], str)
        assert item.value["content"] == "import os\nprint('hello')\n"
        assert "encoding" not in item.value


class TestLegacyListContentRaisesTypeError:
    """Legacy list[str] store items are rejected when read back."""

    def _seed_legacy_item(self, mem_store: InMemoryStore, path: str, content: str) -> None:
        mem_store.put(
            ("filesystem",),
            path,
            _to_legacy_file_data(create_file_data(content)),
        )

    def test_read_raises_type_error(self) -> None:
        mem_store = InMemoryStore()
        self._seed_legacy_item(mem_store, "/old/file.txt", "hello\nworld")
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format="v2")

        with pytest.raises(TypeError, match="list\\[str\\]"):
            be.read("/old/file.txt")

    def test_edit_raises_type_error(self) -> None:
        mem_store = InMemoryStore()
        self._seed_legacy_item(mem_store, "/old/code.py", "foo\nbar")
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format="v2")

        with pytest.raises(TypeError, match="list\\[str\\]"):
            be.edit("/old/code.py", "bar", "qux")

    def test_grep_raises_type_error_when_legacy_items_present(self) -> None:
        mem_store = InMemoryStore()
        self._seed_legacy_item(mem_store, "/src/legacy.py", "def legacy():\n    pass")
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format="v2")
        be.write("/src/modern.py", "def modern():\n    pass")

        with pytest.raises(TypeError, match="list\\[str\\]"):
            be.grep("def", path="/")

    def test_download_raises_type_error(self) -> None:
        mem_store = InMemoryStore()
        self._seed_legacy_item(mem_store, "/data.csv", "alpha\nbeta")
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format="v2")

        with pytest.raises(TypeError, match="list\\[str\\]"):
            be.download_files(["/data.csv"])


class TestV1OperationsUsePlainStr:
    """V1 backend read/edit round-trips use plain str content."""

    def test_v1_write_then_read(self) -> None:
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format="v1")
        be.write("/app.py", "hello\nworld")

        read_result = be.read("/app.py")
        assert read_result.error is None
        assert read_result.file_data is not None
        assert read_result.file_data["content"] == "hello\nworld"

    def test_v1_write_then_edit(self) -> None:
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format="v1")
        be.write("/app.py", "hello\nworld")

        edit_result = be.edit("/app.py", "world", "there")
        assert edit_result.error is None

        read_result = be.read("/app.py")
        assert read_result.error is None
        assert read_result.file_data is not None
        assert read_result.file_data["content"] == "hello\nthere"


class TestV2OperationsAlongsideLegacyStoreData:
    """V2 operations remain available even when legacy items exist in the store."""

    def test_write_new_v2_file_alongside_legacy_checkpoint(self) -> None:
        legacy_data = _to_legacy_file_data(create_file_data("old content"))
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/old/file.txt", legacy_data)
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format="v2")

        result = be.write("/new/file.txt", "new content")
        assert result.error is None

        item = mem_store.get(("filesystem",), "/new/file.txt")
        assert isinstance(item.value["content"], str)
        assert item.value["encoding"] == "utf-8"

        read_result = be.read("/new/file.txt")
        assert read_result.error is None
        assert read_result.file_data is not None
        assert read_result.file_data["content"] == "new content"
