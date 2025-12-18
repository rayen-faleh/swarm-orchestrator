"""
Git-native session metadata store.

Provides JSON-based persistence for git worktree session metadata,
enabling git-native backend to track sessions without Schaltwerk.
"""

import json
import fcntl
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class SessionRecord:
    """Session metadata record for git-native worktrees."""

    name: str
    status: str
    branch: str
    worktree_path: str | None = None
    created_at: str | None = None
    spec_content: str | None = None
    pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            status=data["status"],
            branch=data["branch"],
            worktree_path=data.get("worktree_path"),
            created_at=data.get("created_at"),
            spec_content=data.get("spec_content"),
            pid=data.get("pid"),
        )


class SessionStore:
    """
    JSON-based session metadata store.

    Persists session records to .swarm/sessions.json with file locking
    for concurrent access safety.
    """

    def __init__(self, store_path: Path | str | None = None):
        """
        Initialize the session store.

        Args:
            store_path: Path to the store file. Defaults to .swarm/sessions.json
        """
        if store_path is None:
            store_path = Path(".swarm") / "sessions.json"
        self._path = Path(store_path)

    def _load(self) -> dict[str, SessionRecord]:
        """Load all records from disk."""
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return {
                r["name"]: SessionRecord.from_dict(r)
                for r in data.get("sessions", [])
            }
        except (json.JSONDecodeError, KeyError):
            return {}

    def _write(self, records: dict[str, SessionRecord]) -> None:
        """Write all records to disk with locking."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(
                    {"sessions": [r.to_dict() for r in records.values()]},
                    f,
                    indent=2,
                )
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def save(self, record: SessionRecord) -> None:
        """Save or update a session record."""
        records = self._load()
        records[record.name] = record
        self._write(records)

    def get(self, name: str) -> SessionRecord | None:
        """Get a session record by name."""
        return self._load().get(name)

    def list(self) -> list[SessionRecord]:
        """List all session records."""
        return list(self._load().values())

    def delete(self, name: str) -> None:
        """Delete a session record by name."""
        records = self._load()
        if name in records:
            del records[name]
            self._write(records)
