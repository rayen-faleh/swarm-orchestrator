# Wave 1D: Voting Logic

## Overview
Wave 1D implements the core voting and consensus detection logic for the multi-agent system. This wave introduces the VoteCounter class, which analyzes agent outputs to determine if consensus has been reached and calculates confidence scores. The voting mechanism uses exact code matching after normalization to group identical solutions and identify majority agreement (3+ out of 5 agents).

This is a FOUNDATION wave that establishes the consensus algorithm used throughout the system.

## Dependencies
- **Requires**: Wave 1A (Project Bootstrap)
- **Parallel with**: Wave 1B (Agent Interface), Wave 1C (Provider Interface)
- **Enables**: Wave 2C (Orchestrator)

## User Stories

### US-4.1: Vote Counter

**As a** system orchestrator
**I want** to count votes from multiple agent results
**So that** I can determine if consensus has been reached and identify the winning solution

#### Description
Implement a VoteCounter class that processes a list of AgentResult objects and determines consensus through majority voting. The system groups results by exact code match (after normalization) and identifies when 3 or more agents (out of 5) produce identical solutions.

#### Acceptance Criteria
- [ ] VoteCounter class exists with count_votes method
- [ ] Method accepts List[AgentResult] and returns VoteResult
- [ ] Code normalization strips whitespace and normalizes newlines
- [ ] Results are grouped by normalized code (exact match)
- [ ] VoteResult contains groups dict mapping normalized code to list of AgentResults
- [ ] VoteResult contains winner (AgentResult or None)
- [ ] VoteResult contains confidence score (float 0.0-1.0)
- [ ] VoteResult contains is_consensus boolean flag
- [ ] Majority threshold is 3+ votes out of 5
- [ ] Confidence = (votes_for_winner / total_votes)
- [ ] Winner is None when no group has >= 3 votes
- [ ] All edge cases handled (empty list, single result, ties)

### Technical Implementation

#### Component Architecture

**voting.py** (~100 LOC)

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from .types import AgentResult


@dataclass
class VoteResult:
    """Result of vote counting process"""
    groups: Dict[str, List[AgentResult]]  # normalized_code -> list of results
    winner: Optional[AgentResult]  # First result from winning group, or None
    confidence: float  # votes_for_winner / total_votes (0.0-1.0)
    is_consensus: bool  # True if winner has >= 3 votes


class VoteCounter:
    """Counts votes from agent results to determine consensus"""

    MAJORITY_THRESHOLD = 3  # Minimum votes needed for consensus

    def normalize_code(self, code: str) -> str:
        """
        Normalize code string for comparison.

        Steps:
        1. Strip leading/trailing whitespace
        2. Normalize all newlines to \n
        3. Remove extra blank lines
        4. Strip whitespace from each line

        Args:
            code: Raw code string

        Returns:
            Normalized code string
        """
        # Strip outer whitespace
        normalized = code.strip()

        # Normalize line endings (CRLF -> LF)
        normalized = normalized.replace('\r\n', '\n')

        # Split into lines, strip each, remove empty lines
        lines = [line.strip() for line in normalized.split('\n')]
        lines = [line for line in lines if line]

        # Rejoin
        return '\n'.join(lines)

    def count_votes(self, results: List[AgentResult]) -> VoteResult:
        """
        Count votes from agent results and determine consensus.

        Algorithm:
        1. Normalize each code string
        2. Group results by normalized code (exact match)
        3. Count votes per group
        4. Find group with maximum votes
        5. Check if max group >= MAJORITY_THRESHOLD (3 votes)
        6. Calculate confidence score
        7. Return VoteResult

        Args:
            results: List of AgentResult objects from agents

        Returns:
            VoteResult with groups, winner, confidence, and consensus flag
        """
        # Edge case: empty results
        if not results:
            return VoteResult(
                groups={},
                winner=None,
                confidence=0.0,
                is_consensus=False
            )

        # Group by normalized code
        groups: Dict[str, List[AgentResult]] = {}
        for result in results:
            normalized = self.normalize_code(result.code)
            if normalized not in groups:
                groups[normalized] = []
            groups[normalized].append(result)

        # Find winning group (most votes)
        max_group_code = max(groups.keys(), key=lambda k: len(groups[k]))
        max_group = groups[max_group_code]
        max_votes = len(max_group)

        # Check consensus threshold
        is_consensus = max_votes >= self.MAJORITY_THRESHOLD
        winner = max_group[0] if is_consensus else None

        # Calculate confidence
        total_votes = len(results)
        confidence = max_votes / total_votes if total_votes > 0 else 0.0

        return VoteResult(
            groups=groups,
            winner=winner,
            confidence=confidence,
            is_consensus=is_consensus
        )
```

**Key Design Decisions:**
- **Exact matching**: After normalization, codes must match exactly (no fuzzy matching in v1)
- **Normalization strategy**: Removes formatting differences but preserves code structure
- **Majority rule**: 3/5 votes ensures true majority (>50%)
- **Confidence metric**: Simple ratio provides interpretable score
- **First-in-group winner**: When multiple identical results exist, use first one submitted

#### Testing Strategy

**test_voting.py** (~150 LOC)

```python
import pytest
from swarm_agents.voting import VoteCounter, VoteResult
from swarm_agents.types import AgentResult


class TestVoteCounter:
    """Test suite for VoteCounter class"""

    @pytest.fixture
    def counter(self):
        """VoteCounter instance"""
        return VoteCounter()

    @pytest.fixture
    def sample_code(self):
        """Sample code string"""
        return "def add(a, b):\n    return a + b"

    def test_unanimous_vote(self, counter, sample_code):
        """Test 5/5 agents agree - should have 100% confidence"""
        results = [
            AgentResult(code=sample_code, provider="openai", model="gpt-4", tokens=100)
            for _ in range(5)
        ]

        vote_result = counter.count_votes(results)

        assert vote_result.is_consensus is True
        assert vote_result.confidence == 1.0
        assert vote_result.winner is not None
        assert vote_result.winner.code == sample_code
        assert len(vote_result.groups) == 1
        assert len(vote_result.groups[counter.normalize_code(sample_code)]) == 5

    def test_majority_vote(self, counter):
        """Test 3/5 agree - should have 60% confidence"""
        winning_code = "def add(a, b):\n    return a + b"
        losing_code_1 = "def add(a, b):\n    return a+b"  # Different formatting
        losing_code_2 = "def add(x, y):\n    return x + y"  # Different vars

        results = [
            AgentResult(code=winning_code, provider="openai", model="gpt-4", tokens=100),
            AgentResult(code=winning_code, provider="anthropic", model="claude", tokens=90),
            AgentResult(code=winning_code, provider="google", model="gemini", tokens=95),
            AgentResult(code=losing_code_1, provider="openai", model="gpt-3.5", tokens=80),
            AgentResult(code=losing_code_2, provider="anthropic", model="claude", tokens=85),
        ]

        vote_result = counter.count_votes(results)

        assert vote_result.is_consensus is True
        assert vote_result.confidence == 0.6
        assert vote_result.winner is not None
        assert counter.normalize_code(vote_result.winner.code) == counter.normalize_code(winning_code)
        assert len(vote_result.groups) == 3  # Three unique solutions

    def test_no_consensus(self, counter):
        """Test all different results - no consensus"""
        results = [
            AgentResult(code=f"def solution_{i}():\n    return {i}", provider="openai", model="gpt-4", tokens=100)
            for i in range(5)
        ]

        vote_result = counter.count_votes(results)

        assert vote_result.is_consensus is False
        assert vote_result.winner is None
        assert len(vote_result.groups) == 5
        assert vote_result.confidence == 0.2  # 1/5 = 20% for each

    def test_whitespace_normalization(self, counter):
        """Test that whitespace differences are normalized"""
        code_variants = [
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n\n    return a + b",  # Extra blank line
            "  def add(a, b):\n    return a + b  ",  # Leading/trailing spaces
            "def add(a, b):\r\n    return a + b",  # Windows line endings
            "def add(a, b):\n    return a + b\n\n",  # Trailing newlines
        ]

        results = [
            AgentResult(code=code, provider="openai", model="gpt-4", tokens=100)
            for code in code_variants
        ]

        vote_result = counter.count_votes(results)

        assert vote_result.is_consensus is True
        assert vote_result.confidence == 1.0
        assert len(vote_result.groups) == 1  # All normalized to same code

    def test_vote_result_includes_all_groups(self, counter):
        """Test that VoteResult preserves all vote groups"""
        code_a = "solution_a"
        code_b = "solution_b"
        code_c = "solution_c"

        results = [
            AgentResult(code=code_a, provider="openai", model="gpt-4", tokens=100),
            AgentResult(code=code_a, provider="anthropic", model="claude", tokens=90),
            AgentResult(code=code_a, provider="google", model="gemini", tokens=95),
            AgentResult(code=code_b, provider="openai", model="gpt-3.5", tokens=80),
            AgentResult(code=code_c, provider="anthropic", model="claude", tokens=85),
        ]

        vote_result = counter.count_votes(results)

        assert len(vote_result.groups) == 3
        assert len(vote_result.groups[counter.normalize_code(code_a)]) == 3
        assert len(vote_result.groups[counter.normalize_code(code_b)]) == 1
        assert len(vote_result.groups[counter.normalize_code(code_c)]) == 1

    def test_tie_goes_to_first_group(self, counter):
        """Test that when groups tie, none wins (no consensus)"""
        code_a = "solution_a"
        code_b = "solution_b"

        results = [
            AgentResult(code=code_a, provider="openai", model="gpt-4", tokens=100),
            AgentResult(code=code_a, provider="anthropic", model="claude", tokens=90),
            AgentResult(code=code_b, provider="google", model="gemini", tokens=95),
            AgentResult(code=code_b, provider="openai", model="gpt-3.5", tokens=80),
        ]

        vote_result = counter.count_votes(results)

        assert vote_result.is_consensus is False  # 2 votes < 3 threshold
        assert vote_result.winner is None
        assert vote_result.confidence == 0.5  # 2/4 = 50%

    def test_empty_results(self, counter):
        """Test handling of empty results list"""
        vote_result = counter.count_votes([])

        assert vote_result.is_consensus is False
        assert vote_result.winner is None
        assert vote_result.confidence == 0.0
        assert len(vote_result.groups) == 0

    def test_single_result(self, counter):
        """Test handling of single result"""
        result = AgentResult(code="solution", provider="openai", model="gpt-4", tokens=100)
        vote_result = counter.count_votes([result])

        assert vote_result.is_consensus is False  # 1 < 3 threshold
        assert vote_result.winner is None
        assert vote_result.confidence == 1.0  # 1/1 = 100% but still no consensus
        assert len(vote_result.groups) == 1

    def test_normalize_code_preserves_structure(self, counter):
        """Test that normalization preserves code structure"""
        code = """
        def complex_function():
            if True:
                for i in range(10):
                    print(i)
            return None
        """

        normalized = counter.normalize_code(code)

        # Should preserve indentation relationships
        lines = normalized.split('\n')
        assert all(line.strip() for line in lines)  # No empty lines
        assert 'def complex_function():' in normalized
        assert 'if True:' in normalized
        assert 'for i in range(10):' in normalized
```

**Test Coverage Goals:**
- Line coverage: >95%
- Branch coverage: >90%
- All edge cases covered
- All normalization scenarios tested

#### File Structure
```
src/swarm_agents/
└── voting.py                 # VoteCounter and VoteResult classes (~100 LOC)

tests/
└── test_voting.py            # Comprehensive test suite (~150 LOC)
```

## Success Criteria
- [ ] All acceptance criteria met for US-4.1
- [ ] VoteCounter correctly identifies consensus (3+ votes)
- [ ] Code normalization handles whitespace, newlines, and formatting
- [ ] Confidence scores calculated accurately
- [ ] VoteResult contains all required fields
- [ ] All tests pass with >95% coverage
- [ ] No external dependencies beyond stdlib
- [ ] Type hints on all public methods
- [ ] Docstrings on all public classes and methods

## Integration Points
- **Wave 2C (Orchestrator)**: Will use VoteCounter to analyze agent results
- **Wave 1A (Types)**: Uses AgentResult type defined in types.py
- **Future waves**: Voting logic may be extended for recursive consensus

## Estimated Effort
**2-3 hours**

**Breakdown:**
- Implementation: 1 hour
  - VoteCounter class: 30 min
  - Normalization logic: 20 min
  - VoteResult dataclass: 10 min
- Testing: 1 hour
  - Unit tests: 40 min
  - Edge cases: 20 min
- Documentation: 30 min
- Code review & refinement: 30 min

## Risk Assessment
**Low Risk** - Self-contained module with clear requirements

**Potential Issues:**
- Normalization may need tuning based on real-world code variations
- Exact matching might be too strict for some use cases

**Mitigation:**
- Comprehensive test coverage for normalization edge cases
- Design allows for future enhancement to fuzzy matching if needed
- Keep normalization logic isolated for easy modification

## Future Enhancements (Post-MVP)
- Fuzzy code matching (AST-based comparison)
- Weighted voting based on agent performance history
- Configurable majority thresholds
- Similarity scores between non-matching groups
- Support for partial consensus (e.g., 2/5 with high similarity)
