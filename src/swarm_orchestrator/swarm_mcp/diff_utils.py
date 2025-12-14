"""
Utilities for processing and condensing git diffs.

Provides functions to:
- Condense diffs (remove context lines, normalize whitespace)
- Extract statistics (files changed, lines added/deleted)
- Generate summaries for quick agent comparison
"""

import re
from dataclasses import dataclass, field


@dataclass
class DiffStats:
    """Statistics extracted from a git diff."""
    files_changed: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_deleted: int = 0

    @property
    def total_changes(self) -> int:
        return self.lines_added + self.lines_deleted

    def to_dict(self) -> dict:
        return {
            "files": self.files_changed,
            "files_count": len(self.files_changed),
            "added": self.lines_added,
            "deleted": self.lines_deleted,
            "total": self.total_changes,
        }


@dataclass
class ImplementationSummary:
    """Structured summary of an agent's implementation."""
    agent_id: str
    session_name: str
    stats: DiffStats
    condensed_diff: str  # -U0 -w equivalent
    full_diff: str  # Original full diff (kept for reference)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "session_name": self.session_name,
            "summary": {
                "files_changed": self.stats.files_changed,
                "stats": self.stats.to_dict(),
            },
            "condensed_diff": self.condensed_diff,
            "full_diff_size": len(self.full_diff),
        }

    def to_response_dict(self, include_full_diff: bool = False) -> dict:
        """Generate response dict for MCP tool response."""
        result = {
            "agent_id": self.agent_id,
            "session_name": self.session_name,
            "summary": {
                "files_changed": self.stats.files_changed,
                "stats": self.stats.to_dict(),
            },
            "condensed_diff": self.condensed_diff,
        }
        if include_full_diff:
            result["full_diff"] = self.full_diff
        return result


def extract_diff_stats(diff: str) -> DiffStats:
    """
    Extract statistics from a git diff.

    Args:
        diff: Raw git diff text

    Returns:
        DiffStats with files changed and line counts
    """
    stats = DiffStats()

    for line in diff.split('\n'):
        # Extract file names from diff headers
        if line.startswith('diff --git'):
            # Format: diff --git a/path/to/file b/path/to/file
            match = re.search(r'diff --git a/(.*) b/(.*)', line)
            if match:
                file_path = match.group(2)
                if file_path not in stats.files_changed:
                    stats.files_changed.append(file_path)

        # Count added lines (excluding diff metadata)
        elif line.startswith('+') and not line.startswith('+++'):
            stats.lines_added += 1

        # Count deleted lines (excluding diff metadata)
        elif line.startswith('-') and not line.startswith('---'):
            stats.lines_deleted += 1

    return stats


def condense_diff(diff: str, normalize_whitespace: bool = True) -> str:
    """
    Condense a git diff by removing context lines (like -U0).

    Keeps only:
    - File headers (diff --git, ---, +++)
    - Hunk headers (@@)
    - Added lines (+)
    - Deleted lines (-)

    Args:
        diff: Raw git diff text
        normalize_whitespace: If True, normalize whitespace in changed lines

    Returns:
        Condensed diff with only changed lines
    """
    condensed_lines = []

    for line in diff.split('\n'):
        # Always keep diff metadata
        if line.startswith('diff --git'):
            condensed_lines.append(line)
            continue

        # Keep file headers (but simplify)
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            condensed_lines.append(line)
            continue

        # Keep hunk headers (simplified - just marker)
        if line.startswith('@@'):
            # Extract just the line numbers for context
            match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if match:
                condensed_lines.append(f"@@ -{match.group(1)} +{match.group(2)} @@")
            else:
                condensed_lines.append("@@")
            continue

        # Keep added lines
        if line.startswith('+'):
            if normalize_whitespace:
                # Normalize internal whitespace but keep the + prefix
                content = line[1:]
                # Collapse multiple spaces/tabs to single space
                content = re.sub(r'[ \t]+', ' ', content)
                # Strip trailing whitespace
                content = content.rstrip()
                condensed_lines.append('+' + content)
            else:
                condensed_lines.append(line.rstrip())
            continue

        # Keep deleted lines
        if line.startswith('-'):
            if normalize_whitespace:
                content = line[1:]
                content = re.sub(r'[ \t]+', ' ', content)
                content = content.rstrip()
                condensed_lines.append('-' + content)
            else:
                condensed_lines.append(line.rstrip())
            continue

        # Skip context lines (lines starting with space) and other metadata
        # This is the key difference from full diff - we drop context

    return '\n'.join(condensed_lines)


def create_implementation_summary(
    agent_id: str,
    session_name: str,
    full_diff: str,
) -> ImplementationSummary:
    """
    Create a structured implementation summary from a raw diff.

    Args:
        agent_id: The agent's ID
        session_name: The Schaltwerk session name
        full_diff: The complete git diff

    Returns:
        ImplementationSummary with stats and condensed diff
    """
    stats = extract_diff_stats(full_diff)
    condensed = condense_diff(full_diff, normalize_whitespace=True)

    return ImplementationSummary(
        agent_id=agent_id,
        session_name=session_name,
        stats=stats,
        condensed_diff=condensed,
        full_diff=full_diff,
    )


def format_diff_for_comparison(implementations: list[ImplementationSummary]) -> str:
    """
    Format multiple implementations for easy agent comparison.

    Args:
        implementations: List of implementation summaries

    Returns:
        Formatted string with side-by-side comparison
    """
    lines = ["## Implementation Comparison\n"]
    lines.append("| Agent | Files | +Added | -Deleted | Total |")
    lines.append("|-------|-------|--------|----------|-------|")

    for impl in implementations:
        s = impl.stats
        lines.append(
            f"| {impl.agent_id} | {len(s.files_changed)} | "
            f"+{s.lines_added} | -{s.lines_deleted} | {s.total_changes} |"
        )

    lines.append("\n### Files Changed by Agent\n")
    for impl in implementations:
        lines.append(f"**{impl.agent_id}**: {', '.join(impl.stats.files_changed)}")

    return '\n'.join(lines)
