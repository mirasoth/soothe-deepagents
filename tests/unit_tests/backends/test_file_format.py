"""Tests for current file data storage format and helpers."""

import base64

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from soothe_deepagents.backends.state import StateBackend
from soothe_deepagents.backends.store import StoreBackend
from soothe_deepagents.backends.utils import (
    _to_legacy_file_data,
    compile_grep_include_glob,
    create_file_data,
    file_data_to_string,
    grep_matches_from_files,
)
from soothe_deepagents.middleware.filesystem import FilesystemState

NS = lambda _rt: ("filesystem",)  # noqa: E731


def test_text_round_trip() -> None:
    fd = create_file_data("hello\nworld")
    assert isinstance(fd["content"], str)
    assert fd["content"] == "hello\nworld"
    assert fd["encoding"] == "utf-8"
    assert file_data_to_string(fd) == "hello\nworld"


def test_legacy_list_content_is_readable() -> None:
    fd = {"content": ["hello", "world", ""]}

    assert file_data_to_string(fd) == "hello\nworld\n"  # type: ignore[arg-type]


def test_empty_legacy_list_content_is_readable() -> None:
    fd = {"content": []}

    assert file_data_to_string(fd) == ""  # type: ignore[arg-type]


def test_legacy_list_content_rejects_non_string_items() -> None:
    fd = {"content": ["hello", 1]}

    with pytest.raises(TypeError, match="got list"):
        file_data_to_string(fd)  # type: ignore[arg-type]


def test_binary_round_trip() -> None:
    original = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    b64_str = base64.standard_b64encode(original).decode("ascii")
    fd = create_file_data(b64_str, encoding="base64")
    assert fd["content"] == b64_str
    assert fd["encoding"] == "base64"
    assert base64.standard_b64decode(fd["content"]) == original


@pytest.mark.parametrize("backend_type", [StateBackend, StoreBackend])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("replacement", ["ordinary replacement text", "café\n日本語", "aGVsbG8=", ""])
async def test_text_write_replaces_binary_encoding(backend_type: type[StateBackend] | type[StoreBackend], mode: str, replacement: str) -> None:
    backend = StoreBackend(store=InMemoryStore(), namespace=NS) if backend_type is StoreBackend else StateBackend()
    path = "/notes.txt"
    binary = b"\xff\xfe\xfd"

    async def exercise(_state: FilesystemState) -> dict:
        uploaded = await backend.aupload_files([(path, binary)]) if mode == "async" else backend.upload_files([(path, binary)])
        assert uploaded[0].error is None
        original = await backend.aread(path) if mode == "async" else backend.read(path)
        assert original.file_data is not None
        assert original.file_data["encoding"] == "base64"
        downloaded = await backend.adownload_files([path]) if mode == "async" else backend.download_files([path])
        assert downloaded[0].content == binary

        written = await backend.awrite(path, replacement) if mode == "async" else backend.write(path, replacement)
        assert written.error is None
        result = await backend.aread(path) if mode == "async" else backend.read(path)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == replacement
        assert result.file_data["encoding"] == "utf-8"
        assert result.file_data["created_at"] == original.file_data["created_at"]
        downloaded = await backend.adownload_files([path]) if mode == "async" else backend.download_files([path])
        assert downloaded[0].error is None
        assert downloaded[0].content == replacement.encode("utf-8")
        return {}

    builder = StateGraph(FilesystemState)
    builder.add_node("exercise", exercise)
    builder.add_edge(START, "exercise")
    builder.add_edge("exercise", END)
    await builder.compile().ainvoke({"messages": []})


def test_store_upload_binary() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    responses = be.upload_files([("/images/test.png", png_bytes)])

    assert len(responses) == 1
    assert responses[0].error is None
    item = mem_store.get(("filesystem",), "/images/test.png")
    assert item is not None
    assert item.value["encoding"] == "base64"
    assert isinstance(item.value["content"], str)


def test_store_upload_download_binary_round_trip() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)

    original_bytes = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    be.upload_files([("/images/photo.png", original_bytes)])

    responses = be.download_files(["/images/photo.png"])
    assert len(responses) == 1
    assert responses[0].error is None
    assert responses[0].content == original_bytes


def test_store_upload_text() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)

    text_bytes = b"Hello, world!\nLine 2"
    responses = be.upload_files([("/docs/readme.txt", text_bytes)])

    assert len(responses) == 1
    assert responses[0].error is None
    item = mem_store.get(("filesystem",), "/docs/readme.txt")
    assert item is not None
    assert item.value["encoding"] == "utf-8"
    assert item.value["content"] == "Hello, world!\nLine 2"


def test_grep_new_format() -> None:
    fd = create_file_data("import os\nprint('hello')\nimport sys")
    files = {"/src/main.py": fd}
    result = grep_matches_from_files(files, "import", path="/")
    assert result.matches is not None
    assert len(result.matches) == 2
    assert result.matches[0]["line"] == 1
    assert result.matches[0]["text"] == "import os"
    assert result.matches[1]["line"] == 3
    assert result.matches[1]["text"] == "import sys"


def test_grep_glob_filters_by_filename() -> None:
    files = {
        "/src/main.py": create_file_data("import os"),
        "/src/notes.txt": create_file_data("import os"),
    }
    result = grep_matches_from_files(files, "import", path="/", glob="*.py")
    assert result.matches is not None
    assert len(result.matches) == 1
    assert result.matches[0]["path"] == "/src/main.py"


def test_grep_glob_brace_expansion() -> None:
    files = {
        "/a.py": create_file_data("hit"),
        "/b.md": create_file_data("hit"),
        "/c.txt": create_file_data("hit"),
    }
    result = grep_matches_from_files(files, "hit", path="/", glob="*.{py,md}")
    assert result.matches is not None
    paths = sorted(m["path"] for m in result.matches)
    assert paths == ["/a.py", "/b.md"]


def test_grep_glob_matches_nothing() -> None:
    files = {
        "/a.py": create_file_data("hit"),
        "/b.md": create_file_data("hit"),
    }
    result = grep_matches_from_files(files, "hit", path="/", glob="*.rs")
    assert result.matches == []


def test_grep_glob_repeated_pattern_stays_correct() -> None:
    first = {"/x.py": create_file_data("hit"), "/x.md": create_file_data("hit")}
    second = {"/y.py": create_file_data("hit"), "/y.txt": create_file_data("hit")}
    r1 = grep_matches_from_files(first, "hit", path="/", glob="*.py")
    r2 = grep_matches_from_files(second, "hit", path="/", glob="*.py")
    assert r1.matches is not None
    assert r2.matches is not None
    assert [m["path"] for m in r1.matches] == ["/x.py"]
    assert [m["path"] for m in r2.matches] == ["/y.py"]


def test_store_upload_utf8_content_stored_as_text() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)

    be.upload_files([("/docs/notes.txt", b"Hello, world!")])

    item = mem_store.get(("filesystem",), "/docs/notes.txt")
    assert item is not None
    assert item.value["encoding"] == "utf-8"
    assert item.value["content"] == "Hello, world!"


def test_store_upload_non_utf8_content_stored_as_base64() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)

    raw = b"\x89PNG\r\n\x1a\n" + b"\xff\xfe" + b"\x00" * 20
    be.upload_files([("/images/photo.png", raw)])

    item = mem_store.get(("filesystem",), "/images/photo.png")
    assert item is not None
    assert item.value["encoding"] == "base64"
    assert base64.standard_b64decode(item.value["content"]) == raw


def _ns(_rt: object) -> tuple[str, ...]:
    return ("filesystem",)


def test_legacy_list_content_is_joined():
    """Legacy `list[str]` content is joined with newlines instead of rejected."""
    legacy_fd = {
        "content": ["line1", "line2"],
        "encoding": "utf-8",
        "created_at": "2025-01-01T00:00:00+00:00",
        "modified_at": "2025-01-01T00:00:00+00:00",
    }
    assert file_data_to_string(legacy_fd) == "line1\nline2"


def test_non_string_content_still_raises_type_error():
    """Content that is neither `str` nor legacy `list[str]` stays an error."""
    bad_fd = {
        "content": 42,
        "encoding": "utf-8",
        "created_at": "2025-01-01T00:00:00+00:00",
        "modified_at": "2025-01-01T00:00:00+00:00",
    }
    with pytest.raises(TypeError):
        file_data_to_string(bad_fd)


def test_new_format_no_warning():
    fd = {"content": "hello", "encoding": "utf-8", "created_at": "2025-01-01T00:00:00+00:00", "modified_at": "2025-01-01T00:00:00+00:00"}
    assert file_data_to_string(fd) == "hello"


def test_store_legacy_list_content_read_is_joined():
    """Legacy `list[str]` store items are joined back into text."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)
    mem_store.put(
        ("filesystem",),
        "/legacy/file.txt",
        {
            "content": ["line1", "line2", "line3"],
            "encoding": "utf-8",
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-01T00:00:00+00:00",
        },
    )
    read_result = be.read("/legacy/file.txt")
    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "line1\nline2\nline3"

    responses = be.download_files(["/legacy/file.txt"])
    assert responses[0].error is None
    assert responses[0].content == b"line1\nline2\nline3"


def test_store_legacy_list_content_read_v1_is_joined():
    """Reading legacy list[str] content through StoreBackend (v1 format) joins the lines."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS, file_format="v1")
    mem_store.put(
        ("filesystem",),
        "/old/file.txt",
        {
            "content": ["alpha", "beta"],
            "encoding": "utf-8",
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-01T00:00:00+00:00",
        },
    )
    read_result = be.read("/old/file.txt")
    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "alpha\nbeta"

    responses = be.download_files(["/old/file.txt"])
    assert responses[0].error is None
    assert responses[0].content == b"alpha\nbeta"


def test_compile_glob_is_cached():
    """`compile_grep_include_glob` returns the identical matcher for a repeated pattern.

    This is the optimization the change exists to provide: the compiled matcher
    is reused across calls rather than recompiled per candidate file.
    """
    assert compile_grep_include_glob("*.py") is compile_grep_include_glob("*.py")
    assert compile_grep_include_glob("*.py") is not compile_grep_include_glob("*.md")


def test_grep_legacy_format_joins_lines():
    """Legacy `list[str]` content is joined before grep scans it."""
    legacy_fd = {
        "content": ["def foo():", "    return 42", "def bar():", "    return 0"],
        "encoding": "utf-8",
        "created_at": "2025-01-01T00:00:00+00:00",
        "modified_at": "2025-01-01T00:00:00+00:00",
    }
    files = {"/src/funcs.py": legacy_fd}
    result = grep_matches_from_files(files, "def", path="/")
    assert result.error is None
    assert [match["path"] for match in (result.matches or [])] == ["/src/funcs.py"] * 2


def test_store_write_v1_stores_plain_str():
    """StoreBackend with file_format='v1' stores content as plain str."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS, file_format="v1")
    be.write("/docs/readme.txt", "line1\nline2\nline3")

    item = mem_store.get(("filesystem",), "/docs/readme.txt")
    assert item is not None
    assert item.value["content"] == "line1\nline2\nline3"
    assert "encoding" not in item.value


def test_store_edit_v1_round_trip():
    """Editing v1 plain str content works normally."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS, file_format="v1")
    be.write("/docs/readme.txt", "hello\nworld")

    edit_result = be.edit("/docs/readme.txt", "world", "there")
    assert edit_result.error is None

    read_result = be.read("/docs/readme.txt")
    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "hello\nthere"


def test_store_write_v1_readable():
    """Files stored in v1 format can be read back as plain str."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS, file_format="v1")
    be.write("/file.txt", "aaa\nbbb")

    read_result = be.read("/file.txt")
    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "aaa\nbbb"


def test_store_write_v1():
    """StoreBackend with file_format='v1' stores plain str content."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS, file_format="v1")
    result = be.write("/docs/readme.txt", "alpha\nbeta")
    assert result.error is None
    item = mem_store.get(("filesystem",), "/docs/readme.txt")
    assert item.value["content"] == "alpha\nbeta"
    assert "encoding" not in item.value


def test_store_edit_as_list_v1_reads_legacy_content():
    """Editing seeded v1 list[str] content joins it before applying the edit."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS, file_format="v1")
    legacy = _to_legacy_file_data(create_file_data("hello\nworld"))
    mem_store.put(("filesystem",), "/file.txt", legacy)
    edit_result = be.edit("/file.txt", "world", "there")
    assert edit_result.error is None

    read_result = be.read("/file.txt")
    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "hello\nthere"
