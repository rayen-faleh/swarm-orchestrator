"""
Voting and consensus logic for comparing agent outputs.
"""

import hashlib
import re
from dataclasses import dataclass, field


@dataclass
class VoteGroup:
    """A group of sessions with identical output."""

    diff_hash: str
    diff_content: str
    sessions: list[str] = field(default_factory=list)

    @property
    def vote_count(self) -> int:
        return len(self.sessions)


@dataclass
class VoteResult:
    """Result of voting on agent outputs."""

    groups: list[VoteGroup]
    winner: VoteGroup | None
    total_votes: int
    consensus_reached: bool
    confidence: float  # 0.0 to 1.0


def normalize_diff(diff: str) -> str:
    """
    Normalize a diff for comparison.

    - Remove diff metadata (timestamps, commit hashes)
    - Normalize hunk headers
    - Strip trailing whitespace
    - Remove empty lines at start/end
    """
    lines = []
    for line in diff.split("\n"):
        # Skip git diff metadata
        if line.startswith("diff --git"):
            continue
        if line.startswith("index "):
            continue
        if line.startswith("---") or line.startswith("+++"):
            # Keep file paths but normalize
            parts = line.split("\t")
            lines.append(parts[0])
            continue

        # Normalize hunk headers (remove line numbers, keep structure)
        if line.startswith("@@"):
            lines.append("@@")
            continue

        # Keep content lines, strip trailing whitespace
        lines.append(line.rstrip())

    # Remove empty lines at start/end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    return "\n".join(lines)


def hash_diff(diff: str) -> str:
    """Create a hash of a normalized diff for grouping."""
    normalized = normalize_diff(diff)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def group_by_output(sessions_with_diffs: dict[str, str]) -> list[VoteGroup]:
    """
    Group sessions by their diff output.

    Args:
        sessions_with_diffs: Dict mapping session name to diff content

    Returns:
        List of VoteGroups, sorted by vote count (descending)
    """
    groups: dict[str, VoteGroup] = {}

    for session, diff in sessions_with_diffs.items():
        diff_hash = hash_diff(diff)

        if diff_hash not in groups:
            groups[diff_hash] = VoteGroup(
                diff_hash=diff_hash,
                diff_content=diff,
                sessions=[],
            )

        groups[diff_hash].sessions.append(session)

    # Sort by vote count descending
    sorted_groups = sorted(groups.values(), key=lambda g: g.vote_count, reverse=True)
    return sorted_groups


def find_consensus(
    sessions_with_diffs: dict[str, str],
    min_votes: int | None = None,
) -> VoteResult:
    """
    Determine if there's consensus among agent outputs.

    Args:
        sessions_with_diffs: Dict mapping session name to diff content
        min_votes: Minimum votes needed for consensus (default: majority)

    Returns:
        VoteResult with consensus information
    """
    total_votes = len(sessions_with_diffs)

    if total_votes == 0:
        return VoteResult(
            groups=[],
            winner=None,
            total_votes=0,
            consensus_reached=False,
            confidence=0.0,
        )

    # Default to simple majority
    if min_votes is None:
        min_votes = (total_votes // 2) + 1

    groups = group_by_output(sessions_with_diffs)

    # Check if top group has enough votes
    winner = None
    consensus_reached = False

    if groups and groups[0].vote_count >= min_votes:
        winner = groups[0]
        consensus_reached = True

    # Calculate confidence (winning votes / total)
    confidence = groups[0].vote_count / total_votes if groups else 0.0

    return VoteResult(
        groups=groups,
        winner=winner,
        total_votes=total_votes,
        consensus_reached=consensus_reached,
        confidence=confidence,
    )


def format_vote_summary(result: VoteResult) -> str:
    """Format vote result as a human-readable summary."""
    lines = []
    lines.append(f"Total votes: {result.total_votes}")
    lines.append(f"Consensus: {'Yes' if result.consensus_reached else 'No'}")
    lines.append(f"Confidence: {result.confidence:.1%}")
    lines.append("")
    lines.append("Vote distribution:")

    for i, group in enumerate(result.groups):
        marker = "→" if group == result.winner else " "
        lines.append(
            f"  {marker} Group {i + 1}: {group.vote_count} votes "
            f"({group.vote_count / result.total_votes:.1%}) "
            f"[{', '.join(group.sessions)}]"
        )

    return "\n".join(lines)
