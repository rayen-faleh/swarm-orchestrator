"""
Tests for git-native worktree backend.

Tests GitNativeWorktreeBackend implementation using subprocess git commands.
"""

import subprocess
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from swarm_orchestrator.backends.git_native import GitNativeWorktreeBackend
from swarm_orchestrator.backends.git_native_store import SessionStore, SessionRecord
from swarm_orchestrator.backends.base import SessionInfo, DiffResult


class TestGitNativeWorktreeBackend:
    """Tests for GitNativeWorktreeBackend class."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary git repository for testing."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_path, check=True, capture_output=True)
        # Create initial commit
        (repo_path / "README.md").write_text("# Test")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)
        return repo_path

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary SessionStore."""
        return SessionStore(store_path=tmp_path / ".swarm" / "sessions.json")

    @pytest.fixture
    def backend(self, temp_repo, store):
        """Create a GitNativeWorktreeBackend instance."""
        return GitNativeWorktreeBackend(repo_path=temp_repo, store=store)

    def test_create_session_creates_worktree(self, backend, temp_repo):
        """create_session creates a git worktree and stores metadata."""
        info = backend.create_session("test-session", "Task content here")

        assert info.name == "test-session"
        assert info.status == "running"
        assert info.branch == "git-native/test-session"
        assert info.worktree_path is not None
        assert Path(info.worktree_path).exists()

        # Verify worktree was created
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=temp_repo,
            capture_output=True,
            text=True,
        )
        assert "git-native/test-session" in result.stdout

    def test_create_session_stores_metadata(self, backend, store):
        """create_session stores session metadata in SessionStore."""
        backend.create_session("meta-test", "Task content")

        record = store.get("meta-test")
        assert record is not None
        assert record.name == "meta-test"
        assert record.status == "running"
        assert record.branch == "git-native/meta-test"
        assert record.spec_content == "Task content"

    def test_delete_session_removes_worktree(self, backend, temp_repo):
        """delete_session removes the git worktree."""
        backend.create_session("to-delete", "content")
        backend.delete_session("to-delete")

        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=temp_repo,
            capture_output=True,
            text=True,
        )
        assert "to-delete" not in result.stdout

    def test_delete_session_removes_metadata(self, backend, store):
        """delete_session removes session metadata from store."""
        backend.create_session("to-delete-meta", "content")
        backend.delete_session("to-delete-meta")

        assert store.get("to-delete-meta") is None

    def test_delete_session_removes_branch(self, backend, temp_repo):
        """delete_session removes the session branch."""
        backend.create_session("branch-delete", "content")
        backend.delete_session("branch-delete")

        result = subprocess.run(
            ["git", "branch", "--list", "git-native/branch-delete"],
            cwd=temp_repo,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == ""

    def test_get_session_returns_info(self, backend):
        """get_session returns SessionInfo for existing session."""
        backend.create_session("get-test", "content")

        info = backend.get_session("get-test")
        assert info is not None
        assert info.name == "get-test"
        assert info.status == "running"
        assert info.branch == "git-native/get-test"

    def test_get_session_returns_none_for_nonexistent(self, backend):
        """get_session returns None for nonexistent session."""
        info = backend.get_session("does-not-exist")
        assert info is None

    def test_list_sessions_returns_all(self, backend):
        """list_sessions returns all sessions."""
        backend.create_session("list-a", "content a")
        backend.create_session("list-b", "content b")

        sessions = backend.list_sessions()
        names = {s.name for s in sessions}
        assert "list-a" in names
        assert "list-b" in names

    def test_list_sessions_filter_active(self, backend, store):
        """list_sessions with filter='active' returns running and reviewed, excludes specs."""
        backend.create_session("running-test", "content")
        # Manually add a reviewed session via store
        store.save(SessionRecord(
            name="reviewed-test",
            status="reviewed",
            branch="git-native/reviewed-test",
        ))
        # Manually add a spec session via store
        store.save(SessionRecord(
            name="spec-test",
            status="spec",
            branch="git-native/spec-test",
        ))

        sessions = backend.list_sessions(filter_type="active")
        names = {s.name for s in sessions}
        assert "running-test" in names
        assert "reviewed-test" in names
        assert "spec-test" not in names

    def test_get_diff_returns_changes(self, backend, temp_repo):
        """get_diff returns the diff against parent branch."""
        info = backend.create_session("diff-test", "content")

        # Make a change in the worktree
        worktree_path = Path(info.worktree_path)
        (worktree_path / "new_file.py").write_text("print('hello')\n")
        subprocess.run(["git", "add", "."], cwd=worktree_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add file"], cwd=worktree_path, check=True, capture_output=True)

        diff = backend.get_diff("diff-test")
        assert isinstance(diff, DiffResult)
        assert "new_file.py" in diff.files
        assert "print('hello')" in diff.content

    def test_get_diff_empty_for_no_changes(self, backend):
        """get_diff returns empty result when no changes."""
        backend.create_session("no-changes", "content")

        diff = backend.get_diff("no-changes")
        assert isinstance(diff, DiffResult)
        assert diff.files == []
        assert diff.content == ""

    def test_merge_session_squash_merges(self, backend, temp_repo):
        """merge_session squash-merges changes to parent branch."""
        info = backend.create_session("merge-test", "content")

        # Make changes in worktree
        worktree_path = Path(info.worktree_path)
        (worktree_path / "merged_file.py").write_text("def merged(): pass\n")
        subprocess.run(["git", "add", "."], cwd=worktree_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add merged file"], cwd=worktree_path, check=True, capture_output=True)

        backend.merge_session("merge-test", "Merge: add merged feature")

        # Check file exists in main repo
        result = subprocess.run(
            ["git", "show", "HEAD:merged_file.py"],
            cwd=temp_repo,
            capture_output=True,
            text=True,
        )
        assert "def merged(): pass" in result.stdout

    def test_merge_session_cleans_up(self, backend, temp_repo, store):
        """merge_session removes worktree and metadata after merge."""
        info = backend.create_session("cleanup-test", "content")
        worktree_path = info.worktree_path

        # Make a commit
        (Path(worktree_path) / "cleanup.py").write_text("# cleanup")
        subprocess.run(["git", "add", "."], cwd=worktree_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Cleanup commit"], cwd=worktree_path, check=True, capture_output=True)

        backend.merge_session("cleanup-test", "Merge cleanup")

        # Worktree should be removed
        assert not Path(worktree_path).exists()
        # Metadata should be removed
        assert store.get("cleanup-test") is None


class TestGitNativeWorktreeBackendEdgeCases:
    """Edge case tests for GitNativeWorktreeBackend."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary git repository."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_path, check=True, capture_output=True)
        (repo_path / "README.md").write_text("# Test")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)
        return repo_path

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary SessionStore."""
        return SessionStore(store_path=tmp_path / ".swarm" / "sessions.json")

    @pytest.fixture
    def backend(self, temp_repo, store):
        """Create a GitNativeWorktreeBackend instance."""
        return GitNativeWorktreeBackend(repo_path=temp_repo, store=store)

    def test_delete_nonexistent_session_is_noop(self, backend):
        """delete_session for nonexistent session does not raise."""
        backend.delete_session("nonexistent")  # Should not raise

    def test_delete_session_force_with_uncommitted(self, backend):
        """delete_session with force=True removes even with uncommitted changes."""
        info = backend.create_session("force-delete", "content")
        worktree_path = Path(info.worktree_path)

        # Create uncommitted changes
        (worktree_path / "uncommitted.py").write_text("# uncommitted")

        backend.delete_session("force-delete", force=True)
        assert not worktree_path.exists()

    def test_create_session_unique_worktree_paths(self, backend):
        """Each session gets a unique worktree path."""
        info1 = backend.create_session("unique-a", "content")
        info2 = backend.create_session("unique-b", "content")

        assert info1.worktree_path != info2.worktree_path

    def test_get_diff_with_multiple_commits(self, backend):
        """get_diff captures all commits on the branch."""
        info = backend.create_session("multi-commit", "content")
        worktree_path = Path(info.worktree_path)

        # First commit
        (worktree_path / "file1.py").write_text("# file1")
        subprocess.run(["git", "add", "."], cwd=worktree_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "First"], cwd=worktree_path, check=True, capture_output=True)

        # Second commit
        (worktree_path / "file2.py").write_text("# file2")
        subprocess.run(["git", "add", "."], cwd=worktree_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Second"], cwd=worktree_path, check=True, capture_output=True)

        diff = backend.get_diff("multi-commit")
        assert "file1.py" in diff.files
        assert "file2.py" in diff.files
