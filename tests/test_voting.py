"""
Tests for the voting module.

TDD approach: These tests define the expected behavior of the voting system.
"""

import pytest
from swarm_orchestrator.voting import (
    normalize_diff,
    hash_diff,
    group_by_output,
    find_consensus,
    format_vote_summary,
    VoteGroup,
    VoteResult,
)


class TestNormalizeDiff:
    """Tests for diff normalization."""

    def test_removes_git_metadata(self):
        """Should remove diff --git and index lines."""
        diff = """diff --git a/file.py b/file.py
index abc123..def456 100644
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
+new line
 existing"""
        normalized = normalize_diff(diff)
        assert "diff --git" not in normalized
        assert "index " not in normalized
        assert "+new line" in normalized

    def test_normalizes_hunk_headers(self):
        """Should normalize @@ hunk headers to just @@."""
        diff = "@@ -1,5 +1,6 @@\n+new"
        normalized = normalize_diff(diff)
        assert "@@" in normalized
        assert "-1,5" not in normalized

    def test_strips_trailing_whitespace(self):
        """Should strip trailing whitespace from lines."""
        diff = "+new line   \n-old line\t"
        normalized = normalize_diff(diff)
        assert normalized == "+new line\n-old line"

    def test_removes_empty_lines_at_edges(self):
        """Should remove empty lines at start and end."""
        diff = "\n\n+content\n\n"
        normalized = normalize_diff(diff)
        assert normalized == "+content"

    def test_preserves_content_lines(self):
        """Should preserve actual diff content."""
        diff = """+def hello():
+    return "world"
-def old():
-    pass"""
        normalized = normalize_diff(diff)
        assert '+def hello():' in normalized
        assert '-def old():' in normalized


class TestHashDiff:
    """Tests for diff hashing."""

    def test_identical_diffs_same_hash(self):
        """Identical diffs should produce the same hash."""
        diff1 = "+new line\n-old line"
        diff2 = "+new line\n-old line"
        assert hash_diff(diff1) == hash_diff(diff2)

    def test_different_diffs_different_hash(self):
        """Different diffs should produce different hashes."""
        diff1 = "+new line"
        diff2 = "+other line"
        assert hash_diff(diff1) != hash_diff(diff2)

    def test_whitespace_variations_same_hash(self):
        """Diffs differing only in trailing whitespace should match."""
        diff1 = "+new line"
        diff2 = "+new line   "
        assert hash_diff(diff1) == hash_diff(diff2)

    def test_hash_is_deterministic(self):
        """Same input should always produce same hash."""
        diff = "+content"
        hash1 = hash_diff(diff)
        hash2 = hash_diff(diff)
        assert hash1 == hash2


class TestGroupByOutput:
    """Tests for grouping sessions by output."""

    def test_groups_identical_outputs(self):
        """Sessions with identical output should be in same group."""
        sessions = {
            "agent-0": "+same content",
            "agent-1": "+same content",
            "agent-2": "+different",
        }
        groups = group_by_output(sessions)

        assert len(groups) == 2
        # First group should have 2 sessions (same content)
        assert groups[0].vote_count == 2
        assert set(groups[0].sessions) == {"agent-0", "agent-1"}

    def test_all_different_outputs(self):
        """Each unique output should be its own group."""
        sessions = {
            "agent-0": "+content A",
            "agent-1": "+content B",
            "agent-2": "+content C",
        }
        groups = group_by_output(sessions)

        assert len(groups) == 3
        assert all(g.vote_count == 1 for g in groups)

    def test_all_same_outputs(self):
        """All identical outputs should be one group."""
        sessions = {
            "agent-0": "+same",
            "agent-1": "+same",
            "agent-2": "+same",
        }
        groups = group_by_output(sessions)

        assert len(groups) == 1
        assert groups[0].vote_count == 3

    def test_sorted_by_vote_count(self):
        """Groups should be sorted by vote count descending."""
        sessions = {
            "agent-0": "+A",
            "agent-1": "+B",
            "agent-2": "+B",
            "agent-3": "+B",
            "agent-4": "+C",
        }
        groups = group_by_output(sessions)

        assert groups[0].vote_count == 3  # B
        assert groups[1].vote_count == 1  # A or C

    def test_empty_sessions(self):
        """Empty input should return empty groups."""
        groups = group_by_output({})
        assert groups == []


class TestFindConsensus:
    """Tests for consensus finding."""

    def test_unanimous_consensus(self):
        """All agents agree - highest confidence."""
        sessions = {
            "agent-0": "+same",
            "agent-1": "+same",
            "agent-2": "+same",
        }
        result = find_consensus(sessions)

        assert result.consensus_reached is True
        assert result.confidence == 1.0
        assert result.winner is not None
        assert result.winner.vote_count == 3

    def test_majority_consensus(self):
        """Majority agrees - consensus reached."""
        sessions = {
            "agent-0": "+winner",
            "agent-1": "+winner",
            "agent-2": "+loser",
        }
        result = find_consensus(sessions)

        assert result.consensus_reached is True
        assert result.confidence == pytest.approx(2 / 3)
        assert result.winner is not None
        assert "agent-0" in result.winner.sessions or "agent-1" in result.winner.sessions

    def test_no_consensus_all_different(self):
        """No majority - no consensus."""
        sessions = {
            "agent-0": "+A",
            "agent-1": "+B",
            "agent-2": "+C",
        }
        result = find_consensus(sessions)

        assert result.consensus_reached is False
        assert result.winner is None
        assert result.confidence == pytest.approx(1 / 3)

    def test_no_consensus_split_vote(self):
        """Even split - no consensus."""
        sessions = {
            "agent-0": "+A",
            "agent-1": "+A",
            "agent-2": "+B",
            "agent-3": "+B",
        }
        result = find_consensus(sessions)

        assert result.consensus_reached is False
        assert result.winner is None

    def test_custom_min_votes(self):
        """Custom minimum votes threshold."""
        sessions = {
            "agent-0": "+winner",
            "agent-1": "+winner",
            "agent-2": "+loser",
            "agent-3": "+other",
            "agent-4": "+another",
        }

        # With min_votes=2, should have consensus
        result = find_consensus(sessions, min_votes=2)
        assert result.consensus_reached is True

        # With min_votes=3, should not
        result = find_consensus(sessions, min_votes=3)
        assert result.consensus_reached is False

    def test_empty_sessions(self):
        """Empty input should return no consensus."""
        result = find_consensus({})

        assert result.consensus_reached is False
        assert result.winner is None
        assert result.total_votes == 0
        assert result.confidence == 0.0

    def test_single_session(self):
        """Single session should have consensus (1/1)."""
        sessions = {"agent-0": "+only one"}
        result = find_consensus(sessions)

        assert result.consensus_reached is True
        assert result.confidence == 1.0

    def test_five_agents_three_agree(self):
        """3/5 majority with 5 agents."""
        sessions = {
            "agent-0": "+winner",
            "agent-1": "+winner",
            "agent-2": "+winner",
            "agent-3": "+loser-A",
            "agent-4": "+loser-B",
        }
        result = find_consensus(sessions)

        assert result.consensus_reached is True
        assert result.confidence == 0.6
        assert result.winner.vote_count == 3


class TestFormatVoteSummary:
    """Tests for vote summary formatting."""

    def test_formats_consensus_result(self):
        """Should format a consensus result nicely."""
        result = VoteResult(
            groups=[
                VoteGroup(diff_hash="abc", diff_content="+win", sessions=["a", "b"]),
                VoteGroup(diff_hash="def", diff_content="+lose", sessions=["c"]),
            ],
            winner=VoteGroup(diff_hash="abc", diff_content="+win", sessions=["a", "b"]),
            total_votes=3,
            consensus_reached=True,
            confidence=2 / 3,
        )

        summary = format_vote_summary(result)

        assert "Total votes: 3" in summary
        assert "Consensus: Yes" in summary
        assert "66.7%" in summary

    def test_formats_no_consensus_result(self):
        """Should format a no-consensus result."""
        result = VoteResult(
            groups=[],
            winner=None,
            total_votes=0,
            consensus_reached=False,
            confidence=0.0,
        )

        summary = format_vote_summary(result)

        assert "Consensus: No" in summary
