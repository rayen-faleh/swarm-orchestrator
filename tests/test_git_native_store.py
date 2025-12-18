"""
Tests for git-native session metadata store.

Tests SessionStore CRUD operations, persistence, and concurrent access safety.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime

from swarm_orchestrator.backends.git_native_store import SessionStore, SessionRecord


class TestSessionRecord:
    """Tests for SessionRecord dataclass."""

    def test_session_record_creation(self):
        """SessionRecord can be created with all fields."""
        record = SessionRecord(
            name="test-session",
            status="running",
            branch="git-native/test-session",
            worktree_path="/path/to/worktree",
            created_at="2024-01-01T00:00:00Z",
            spec_content="Task content here",
            pid=12345,
        )
        assert record.name == "test-session"
        assert record.status == "running"
        assert record.branch == "git-native/test-session"
        assert record.worktree_path == "/path/to/worktree"
        assert record.spec_content == "Task content here"
        assert record.pid == 12345

    def test_session_record_pid_defaults_to_none(self):
        """SessionRecord pid defaults to None."""
        record = SessionRecord(
            name="test",
            status="running",
            branch="git-native/test",
        )
        assert record.pid is None

    def test_session_record_to_dict(self):
        """SessionRecord can be serialized to dict."""
        record = SessionRecord(
            name="test",
            status="spec",
            branch="git-native/test",
            pid=5678,
        )
        d = record.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "spec"
        assert d["branch"] == "git-native/test"
        assert d["worktree_path"] is None
        assert d["pid"] == 5678

    def test_session_record_from_dict(self):
        """SessionRecord can be deserialized from dict."""
        d = {
            "name": "test",
            "status": "running",
            "branch": "git-native/test",
            "worktree_path": "/path",
            "created_at": "2024-01-01T00:00:00Z",
            "spec_content": "content",
            "pid": 9999,
        }
        record = SessionRecord.from_dict(d)
        assert record.name == "test"
        assert record.status == "running"
        assert record.worktree_path == "/path"
        assert record.pid == 9999

    def test_session_record_from_dict_without_pid(self):
        """SessionRecord from_dict handles missing pid field."""
        d = {
            "name": "test",
            "status": "running",
            "branch": "git-native/test",
        }
        record = SessionRecord.from_dict(d)
        assert record.pid is None


class TestSessionStore:
    """Tests for SessionStore class."""

    @pytest.fixture
    def temp_store_path(self, tmp_path):
        """Create a temporary store file path."""
        return tmp_path / ".swarm" / "sessions.json"

    @pytest.fixture
    def store(self, temp_store_path):
        """Create a SessionStore with temporary path."""
        return SessionStore(store_path=temp_store_path)

    def test_save_creates_directory(self, store, temp_store_path):
        """save() creates parent directory if needed."""
        record = SessionRecord(
            name="test",
            status="spec",
            branch="git-native/test",
        )
        store.save(record)
        assert temp_store_path.parent.exists()
        assert temp_store_path.exists()

    def test_save_and_get(self, store):
        """save() persists record that can be retrieved with get()."""
        record = SessionRecord(
            name="my-session",
            status="running",
            branch="git-native/my-session",
            worktree_path="/path/to/worktree",
            created_at="2024-01-01T00:00:00Z",
            spec_content="Task spec here",
        )
        store.save(record)
        retrieved = store.get("my-session")

        assert retrieved is not None
        assert retrieved.name == "my-session"
        assert retrieved.status == "running"
        assert retrieved.worktree_path == "/path/to/worktree"
        assert retrieved.spec_content == "Task spec here"

    def test_get_nonexistent_returns_none(self, store):
        """get() returns None for nonexistent session."""
        result = store.get("does-not-exist")
        assert result is None

    def test_save_updates_existing(self, store):
        """save() updates existing record if name matches."""
        record = SessionRecord(name="test", status="spec", branch="git-native/test")
        store.save(record)

        updated = SessionRecord(
            name="test",
            status="running",
            branch="git-native/test",
            worktree_path="/new/path",
        )
        store.save(updated)

        retrieved = store.get("test")
        assert retrieved.status == "running"
        assert retrieved.worktree_path == "/new/path"

    def test_list_empty_store(self, store):
        """list() returns empty list for empty store."""
        result = store.list()
        assert result == []

    def test_list_returns_all_records(self, store):
        """list() returns all stored records."""
        store.save(SessionRecord(name="a", status="spec", branch="git-native/a"))
        store.save(SessionRecord(name="b", status="running", branch="git-native/b"))
        store.save(SessionRecord(name="c", status="reviewed", branch="git-native/c"))

        result = store.list()
        assert len(result) == 3
        names = {r.name for r in result}
        assert names == {"a", "b", "c"}

    def test_delete_removes_record(self, store):
        """delete() removes a record from the store."""
        store.save(SessionRecord(name="to-delete", status="spec", branch="git-native/to-delete"))
        assert store.get("to-delete") is not None

        store.delete("to-delete")
        assert store.get("to-delete") is None

    def test_delete_nonexistent_is_noop(self, store):
        """delete() does nothing for nonexistent session."""
        store.delete("does-not-exist")  # Should not raise

    def test_load_from_empty_file(self, temp_store_path):
        """load() handles missing file gracefully."""
        store = SessionStore(store_path=temp_store_path)
        # Should not raise, just return empty
        result = store.list()
        assert result == []

    def test_persistence_across_instances(self, temp_store_path):
        """Data persists across SessionStore instances."""
        store1 = SessionStore(store_path=temp_store_path)
        store1.save(SessionRecord(
            name="persistent",
            status="running",
            branch="git-native/persistent",
            spec_content="persisted content",
        ))

        # Create new instance with same path
        store2 = SessionStore(store_path=temp_store_path)
        retrieved = store2.get("persistent")

        assert retrieved is not None
        assert retrieved.name == "persistent"
        assert retrieved.spec_content == "persisted content"

    def test_concurrent_access_safety(self, temp_store_path):
        """Multiple stores can safely read/write concurrently."""
        store1 = SessionStore(store_path=temp_store_path)
        store2 = SessionStore(store_path=temp_store_path)

        store1.save(SessionRecord(name="a", status="spec", branch="git-native/a"))
        store2.save(SessionRecord(name="b", status="spec", branch="git-native/b"))

        # Both should see both records after reload
        result1 = store1.list()
        result2 = store2.list()

        assert len(result1) == 2
        assert len(result2) == 2

    def test_file_format_is_valid_json(self, store, temp_store_path):
        """Store file is valid JSON."""
        store.save(SessionRecord(name="test", status="spec", branch="git-native/test"))

        with open(temp_store_path) as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_handles_corrupted_file(self, temp_store_path):
        """Store handles corrupted JSON gracefully."""
        temp_store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_store_path.write_text("not valid json {{{")

        store = SessionStore(store_path=temp_store_path)
        # Should not raise, treat as empty
        result = store.list()
        assert result == []
