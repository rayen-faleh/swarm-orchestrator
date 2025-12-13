# Wave 4A: CLI and Integration Testing

**Status:** Not Started
**Wave Type:** INTEGRATION (Final Wave)
**Requires:** Wave 3A (LangGraph Workflow)
**Enables:** None (Final Wave)

## Overview

This wave implements the command-line interface and comprehensive end-to-end integration testing for the multi-agent consensus system. It provides a user-friendly CLI using Click and validates the entire system with real Ollama integration tests.

## User Stories

### US-6.1: Main Entry Point

**Goal:** Create a professional CLI for the multi-agent consensus system

**Acceptance Criteria:**
- CLI built with Click library
- `--query` argument for task description (required)
- `--model` argument to override default model (default: qwen3-coder)
- `--agents` argument to set agent count (default: 5)
- Pretty-printed output showing:
  - Task decomposition (subtasks)
  - Consensus results per subtask
  - Final combined code
  - Confidence scores
- Exit code 0 on success, 1 on failure
- Clear error messages for all failure modes

**Implementation:**

```python
# src/swarm_agents/main.py

import click
import sys
from typing import Dict, Any
from swarm_agents.workflow import build_workflow
from swarm_agents.llm import OllamaClient


@click.command()
@click.option('--query', '-q', required=True, help='The coding task to execute')
@click.option('--model', '-m', default='qwen3-coder', help='Ollama model to use')
@click.option('--agents', '-a', default=5, type=int, help='Number of agents')
def cli(query: str, model: str, agents: int):
    """Multi-agent consensus code generator.

    This tool uses multiple AI agents to collaboratively solve coding tasks
    through consensus voting. The system decomposes complex tasks, generates
    multiple solutions, and selects the best code through democratic voting.

    Examples:
        swarm-agents --query "Create a function to calculate fibonacci"
        swarm-agents -q "Implement quicksort" -m deepseek-coder -a 7
    """
    click.echo(f"🤖 Starting multi-agent consensus system...")
    click.echo(f"   Model: {model}")
    click.echo(f"   Agents: {agents}")
    click.echo(f"   Query: {query}\n")

    try:
        # Validate inputs
        if agents < 2:
            raise click.BadParameter("Number of agents must be at least 2")
        if agents > 10:
            click.echo("⚠️  Warning: Using more than 10 agents may be slow", err=True)

        # Initialize client and workflow
        client = OllamaClient(model=model)
        graph = build_workflow(client, agent_count=agents)

        # Execute workflow
        click.echo("🔄 Executing workflow...\n")
        result = graph.invoke({"query": query})

        # Pretty print results
        print_results(result)

        # Success exit
        sys.exit(0)

    except KeyboardInterrupt:
        click.echo("\n⚠️  Interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


def print_results(result: Dict[str, Any]) -> None:
    """Pretty print the workflow results.

    Args:
        result: The workflow output dictionary containing:
            - subtasks: List of decomposed tasks
            - consensus_results: List of consensus outcomes per subtask
            - final_output: Combined final code
            - metadata: Additional execution information
    """
    # Header
    click.echo("=" * 70)
    click.echo("📋 TASK DECOMPOSITION")
    click.echo("=" * 70)

    subtasks = result.get("subtasks", [])
    if subtasks:
        for i, subtask in enumerate(subtasks, 1):
            description = subtask.description if hasattr(subtask, 'description') else str(subtask)
            click.echo(f"  {i}. {description}")
    else:
        click.echo("  [Single atomic task - no decomposition needed]")

    # Consensus results
    click.echo("\n" + "=" * 70)
    click.echo("🗳️  CONSENSUS RESULTS")
    click.echo("=" * 70)

    consensus_results = result.get("consensus_results", [])
    if consensus_results:
        for i, consensus in enumerate(consensus_results, 1):
            click.echo(f"\nSubtask {i}:")

            # Confidence score
            confidence = consensus.confidence if hasattr(consensus, 'confidence') else 0.0
            click.echo(f"  Confidence: {confidence:.1%}")

            # Vote distribution
            votes = consensus.vote_distribution if hasattr(consensus, 'vote_distribution') else {}
            if votes:
                click.echo(f"  Vote Distribution:")
                for option, count in sorted(votes.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / sum(votes.values())) * 100 if sum(votes.values()) > 0 else 0
                    click.echo(f"    • Option {option}: {count} votes ({percentage:.0f}%)")

            # Selected solution
            if hasattr(consensus, 'selected_solution'):
                click.echo(f"  Selected: Option {consensus.selected_solution}")
    else:
        click.echo("  [No consensus voting performed]")

    # Final output
    click.echo("\n" + "=" * 70)
    click.echo("✅ FINAL OUTPUT")
    click.echo("=" * 70)

    final_output = result.get("final_output", "")
    if final_output:
        click.echo(final_output)
    else:
        click.echo("  [No output generated]")

    # Metadata (if present)
    metadata = result.get("metadata", {})
    if metadata:
        click.echo("\n" + "=" * 70)
        click.echo("📊 METADATA")
        click.echo("=" * 70)
        for key, value in metadata.items():
            click.echo(f"  {key}: {value}")

    click.echo("\n" + "=" * 70)
    click.echo("✨ Execution Complete")
    click.echo("=" * 70)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
```

**Files:**
- `src/swarm_agents/main.py` (~150 LOC)

**Tests:**
- `tests/test_cli.py` - CLI argument parsing and output formatting

---

### US-7.1: End-to-End Integration Test

**Goal:** Validate the entire system with real Ollama integration tests

**Acceptance Criteria:**
- Integration tests with real Ollama (skippable if not available)
- Test atomic task (single function)
- Test decomposed task (multiple functions)
- Verify consensus is reached
- Verify output code is valid Python (ast.parse)
- All tests pass when Ollama is available
- Tests are skipped gracefully when Ollama is unavailable

**Implementation:**

```python
# tests/test_cli.py

import pytest
from click.testing import CliRunner
from swarm_agents.main import cli
from unittest.mock import Mock, patch


class TestCLI:
    """Test suite for CLI interface."""

    def test_cli_requires_query(self):
        """Test that --query argument is required."""
        runner = CliRunner()
        result = runner.invoke(cli, [])

        assert result.exit_code != 0
        assert "Missing option '--query'" in result.output or "Error" in result.output

    def test_cli_accepts_query(self):
        """Test that CLI accepts and processes query."""
        runner = CliRunner()

        # Mock the workflow
        mock_result = {
            "subtasks": [],
            "consensus_results": [],
            "final_output": "def hello(): return 'world'"
        }

        with patch('swarm_agents.main.build_workflow') as mock_build:
            mock_graph = Mock()
            mock_graph.invoke.return_value = mock_result
            mock_build.return_value = mock_graph

            result = runner.invoke(cli, ['--query', 'Create hello function'])

            assert result.exit_code == 0
            assert "Starting multi-agent consensus system" in result.output

    def test_cli_shows_confidence(self):
        """Test that CLI displays confidence scores."""
        runner = CliRunner()

        # Mock consensus result with confidence
        mock_consensus = Mock()
        mock_consensus.confidence = 0.85
        mock_consensus.vote_distribution = {"A": 4, "B": 1}
        mock_consensus.selected_solution = "A"

        mock_result = {
            "subtasks": [Mock(description="Task 1")],
            "consensus_results": [mock_consensus],
            "final_output": "def test(): pass"
        }

        with patch('swarm_agents.main.build_workflow') as mock_build:
            mock_graph = Mock()
            mock_graph.invoke.return_value = mock_result
            mock_build.return_value = mock_graph

            result = runner.invoke(cli, ['-q', 'Test task'])

            assert result.exit_code == 0
            assert "85.0%" in result.output or "Confidence" in result.output

    def test_cli_custom_model(self):
        """Test that CLI accepts custom model argument."""
        runner = CliRunner()

        mock_result = {
            "subtasks": [],
            "consensus_results": [],
            "final_output": "def test(): pass"
        }

        with patch('swarm_agents.main.build_workflow') as mock_build:
            with patch('swarm_agents.main.OllamaClient') as mock_client:
                mock_graph = Mock()
                mock_graph.invoke.return_value = mock_result
                mock_build.return_value = mock_graph

                result = runner.invoke(cli, [
                    '--query', 'Test',
                    '--model', 'deepseek-coder'
                ])

                # Verify custom model was passed to OllamaClient
                mock_client.assert_called_once_with(model='deepseek-coder')
                assert result.exit_code == 0

    def test_cli_custom_agents(self):
        """Test that CLI accepts custom agent count."""
        runner = CliRunner()

        mock_result = {
            "subtasks": [],
            "consensus_results": [],
            "final_output": "def test(): pass"
        }

        with patch('swarm_agents.main.build_workflow') as mock_build:
            mock_graph = Mock()
            mock_graph.invoke.return_value = mock_result
            mock_build.return_value = mock_graph

            result = runner.invoke(cli, [
                '--query', 'Test',
                '--agents', '7'
            ])

            # Verify custom agent count was passed to build_workflow
            mock_build.assert_called_once()
            call_args = mock_build.call_args
            assert call_args[1]['agent_count'] == 7
            assert result.exit_code == 0

    def test_cli_invalid_agent_count(self):
        """Test that CLI rejects invalid agent counts."""
        runner = CliRunner()

        # Test too few agents
        result = runner.invoke(cli, [
            '--query', 'Test',
            '--agents', '1'
        ])
        assert result.exit_code != 0
        assert "at least 2" in result.output.lower()

    def test_cli_handles_errors_gracefully(self):
        """Test that CLI handles errors with proper exit codes."""
        runner = CliRunner()

        with patch('swarm_agents.main.build_workflow') as mock_build:
            mock_build.side_effect = Exception("Connection error")

            result = runner.invoke(cli, ['--query', 'Test'])

            assert result.exit_code == 1
            assert "Error" in result.output


# tests/test_integration.py

import pytest
import ast
from swarm_agents.workflow import build_workflow
from swarm_agents.llm import OllamaClient


def ollama_available() -> bool:
    """Check if Ollama is available and running.

    Returns:
        bool: True if Ollama is available, False otherwise
    """
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


# Skip all integration tests if Ollama is not available
pytestmark = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not available - skipping integration tests"
)


@pytest.mark.integration
class TestEndToEndIntegration:
    """Integration tests with real Ollama."""

    @pytest.fixture
    def client(self):
        """Create OllamaClient for testing."""
        return OllamaClient(model="qwen3-coder")

    @pytest.fixture
    def workflow(self, client):
        """Build workflow for testing."""
        return build_workflow(client, agent_count=5)

    def test_end_to_end_atomic_task(self, workflow):
        """Test atomic task (single function) execution.

        An atomic task should:
        - Not be decomposed into subtasks
        - Execute directly through consensus
        - Produce valid Python code
        """
        query = "Create a function called 'factorial' that calculates the factorial of a number using recursion."

        result = workflow.invoke({"query": query})

        # Verify structure
        assert "final_output" in result
        assert isinstance(result["final_output"], str)
        assert len(result["final_output"]) > 0

        # Verify it's valid Python
        try:
            ast.parse(result["final_output"])
        except SyntaxError as e:
            pytest.fail(f"Generated code is not valid Python: {e}")

        # Verify the function is present
        assert "factorial" in result["final_output"]
        assert "def factorial" in result["final_output"]

    def test_end_to_end_complex_task(self, workflow):
        """Test decomposed task (multiple functions) execution.

        A complex task should:
        - Be decomposed into multiple subtasks
        - Execute consensus for each subtask
        - Combine results into coherent code
        - Produce valid Python code
        """
        query = """Create a simple calculator module with the following functions:
        1. add(a, b) - returns sum
        2. subtract(a, b) - returns difference
        3. multiply(a, b) - returns product
        4. divide(a, b) - returns quotient (handle division by zero)
        """

        result = workflow.invoke({"query": query})

        # Verify decomposition occurred
        assert "subtasks" in result
        subtasks = result.get("subtasks", [])
        assert len(subtasks) > 1, "Complex task should be decomposed"

        # Verify consensus results exist
        assert "consensus_results" in result
        consensus_results = result.get("consensus_results", [])
        assert len(consensus_results) > 0, "Should have consensus results"

        # Verify final output
        assert "final_output" in result
        final_code = result["final_output"]
        assert isinstance(final_code, str)
        assert len(final_code) > 0

        # Verify it's valid Python
        try:
            ast.parse(final_code)
        except SyntaxError as e:
            pytest.fail(f"Generated code is not valid Python: {e}")

        # Verify all functions are present
        assert "def add" in final_code
        assert "def subtract" in final_code
        assert "def multiply" in final_code
        assert "def divide" in final_code

    def test_output_is_valid_python(self, workflow):
        """Test that output is always valid Python syntax.

        This test ensures the consensus mechanism produces
        syntactically correct Python code.
        """
        queries = [
            "Create a function to check if a number is prime",
            "Implement bubble sort",
            "Create a class representing a bank account with deposit and withdraw methods"
        ]

        for query in queries:
            result = workflow.invoke({"query": query})
            code = result.get("final_output", "")

            # Must be non-empty
            assert len(code) > 0, f"Empty output for query: {query}"

            # Must be valid Python
            try:
                ast.parse(code)
            except SyntaxError as e:
                pytest.fail(f"Invalid Python for query '{query}': {e}\nCode:\n{code}")

    def test_consensus_is_reached(self, workflow):
        """Test that consensus is reached for all subtasks.

        Every subtask should have:
        - A confidence score
        - Vote distribution
        - A selected solution
        """
        query = "Create functions for binary search and linear search"

        result = workflow.invoke({"query": query})

        consensus_results = result.get("consensus_results", [])
        assert len(consensus_results) > 0, "Should have consensus results"

        for i, consensus in enumerate(consensus_results):
            # Check confidence exists and is valid
            assert hasattr(consensus, 'confidence'), f"Consensus {i} missing confidence"
            assert 0.0 <= consensus.confidence <= 1.0, f"Invalid confidence: {consensus.confidence}"

            # Check vote distribution exists
            assert hasattr(consensus, 'vote_distribution'), f"Consensus {i} missing votes"
            votes = consensus.vote_distribution
            assert isinstance(votes, dict), "Vote distribution should be a dict"
            assert len(votes) > 0, "Should have at least one vote"

            # Check selected solution exists
            assert hasattr(consensus, 'selected_solution'), f"Consensus {i} missing selection"
            assert consensus.selected_solution is not None

    def test_different_models(self):
        """Test that system works with different Ollama models.

        This test verifies model switching functionality.
        """
        models_to_test = ["qwen3-coder"]  # Add more models if available

        query = "Create a function to reverse a string"

        for model in models_to_test:
            try:
                client = OllamaClient(model=model)
                workflow = build_workflow(client, agent_count=3)
                result = workflow.invoke({"query": query})

                # Verify basic structure
                assert "final_output" in result
                code = result["final_output"]

                # Verify valid Python
                ast.parse(code)

            except Exception as e:
                # Skip if model not available
                if "not found" in str(e).lower():
                    pytest.skip(f"Model {model} not available")
                else:
                    raise

    def test_various_agent_counts(self, client):
        """Test that system works with different agent counts.

        Consensus should work with:
        - Minimum agents (2)
        - Small team (3-5)
        - Larger team (7-9)
        """
        query = "Create a function to check if a string is a palindrome"

        for agent_count in [2, 3, 5, 7]:
            workflow = build_workflow(client, agent_count=agent_count)
            result = workflow.invoke({"query": query})

            # Verify result structure
            assert "final_output" in result
            assert len(result["final_output"]) > 0

            # Verify consensus occurred
            consensus_results = result.get("consensus_results", [])
            if consensus_results:
                for consensus in consensus_results:
                    # Vote count should match agent count
                    total_votes = sum(consensus.vote_distribution.values())
                    assert total_votes == agent_count, \
                        f"Expected {agent_count} votes, got {total_votes}"


@pytest.mark.integration
def test_ollama_connection():
    """Test that we can connect to Ollama."""
    try:
        import ollama
        models = ollama.list()
        assert models is not None
        print(f"Available models: {[m['name'] for m in models.get('models', [])]}")
    except Exception as e:
        pytest.fail(f"Cannot connect to Ollama: {e}")
```

**Files:**
- `tests/test_cli.py` (~120 LOC)
- `tests/test_integration.py` (~250 LOC)

**Test Markers:**
- `@pytest.mark.integration` - Marks tests that require Ollama
- Use `pytest -m integration` to run only integration tests
- Use `pytest -m "not integration"` to skip integration tests

---

## Dependencies

### Production Dependencies
```toml
[tool.poetry.dependencies]
click = "^8.1.7"  # CLI framework
```

### Development Dependencies
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
```

### Package Entry Point
```toml
[tool.poetry.scripts]
swarm-agents = "swarm_agents.main:main"
```

---

## File Structure

```
swarm/
├── src/
│   └── swarm_agents/
│       ├── __init__.py
│       ├── main.py              # NEW: CLI entry point
│       ├── models.py            # From Wave 1
│       ├── llm.py               # From Wave 1
│       ├── prompts.py           # From Wave 2A
│       ├── decomposer.py        # From Wave 2A
│       ├── generator.py         # From Wave 2B
│       ├── consensus.py         # From Wave 2C
│       └── workflow.py          # From Wave 3A
├── tests/
│   ├── test_cli.py              # NEW: CLI tests
│   ├── test_integration.py      # NEW: E2E integration tests
│   ├── test_models.py
│   ├── test_llm.py
│   ├── test_decomposer.py
│   ├── test_generator.py
│   ├── test_consensus.py
│   └── test_workflow.py
├── docs/
│   └── swarm-agents/
│       ├── wave-1-core-types.md
│       ├── wave-2a-decomposer.md
│       ├── wave-2b-generator.md
│       ├── wave-2c-consensus.md
│       ├── wave-3a-workflow.md
│       └── wave-4a-cli-integration.md  # THIS FILE
├── pyproject.toml
└── README.md
```

---

## Testing Strategy

### Unit Tests (test_cli.py)
- Mock all external dependencies (workflow, LLM client)
- Test CLI argument parsing
- Test output formatting
- Test error handling
- Test exit codes
- Fast execution (~1 second total)

### Integration Tests (test_integration.py)
- Require real Ollama connection
- Test full workflow execution
- Test multiple scenarios (atomic, complex)
- Validate Python syntax of output
- Verify consensus mechanism
- Slower execution (~30-60 seconds total)

### Test Execution
```bash
# Run all tests
poetry run pytest

# Run only unit tests (fast)
poetry run pytest -m "not integration"

# Run only integration tests (slow, requires Ollama)
poetry run pytest -m integration

# Run with coverage
poetry run pytest --cov=swarm_agents --cov-report=html

# Run specific test file
poetry run pytest tests/test_cli.py -v
```

---

## Usage Examples

### Basic Usage
```bash
# Default settings (qwen3-coder, 5 agents)
swarm-agents --query "Create a function to calculate fibonacci numbers"

# Short form
swarm-agents -q "Implement binary search"
```

### Custom Model
```bash
swarm-agents --query "Create a sorting algorithm" --model deepseek-coder
swarm-agents -q "Create a sorting algorithm" -m deepseek-coder
```

### Custom Agent Count
```bash
swarm-agents --query "Create a web scraper" --agents 7
swarm-agents -q "Create a web scraper" -a 7
```

### Combined Options
```bash
swarm-agents \
  --query "Create a REST API client with error handling" \
  --model qwen3-coder \
  --agents 5
```

---

## Expected Output Format

```
🤖 Starting multi-agent consensus system...
   Model: qwen3-coder
   Agents: 5
   Query: Create a function to calculate fibonacci numbers

🔄 Executing workflow...

======================================================================
📋 TASK DECOMPOSITION
======================================================================
  [Single atomic task - no decomposition needed]

======================================================================
🗳️  CONSENSUS RESULTS
======================================================================

Subtask 1:
  Confidence: 87.5%
  Vote Distribution:
    • Option A: 4 votes (80%)
    • Option B: 1 votes (20%)
  Selected: Option A

======================================================================
✅ FINAL OUTPUT
======================================================================
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number using recursion.

    Args:
        n: The position in the Fibonacci sequence (0-indexed)

    Returns:
        The Fibonacci number at position n

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

======================================================================
✨ Execution Complete
======================================================================
```

---

## Error Handling

### Connection Errors
```
❌ Error: Cannot connect to Ollama. Is it running?
Exit code: 1
```

### Invalid Arguments
```
Error: Missing option '--query' / '-q'.
Exit code: 2 (Click default)
```

### Validation Errors
```
❌ Error: Number of agents must be at least 2
Exit code: 1
```

### Workflow Errors
```
❌ Error: Failed to reach consensus after maximum retries
Exit code: 1
```

---

## Success Criteria

- [ ] CLI accepts all required and optional arguments
- [ ] Pretty-printed output is clear and informative
- [ ] Exit codes are correct (0 for success, 1 for failure)
- [ ] All unit tests pass
- [ ] All integration tests pass (when Ollama available)
- [ ] Integration tests are skipped gracefully (when Ollama unavailable)
- [ ] Generated code is valid Python (verified with ast.parse)
- [ ] Consensus is reached for all subtasks
- [ ] System works with different models
- [ ] System works with different agent counts (2-10)
- [ ] Error messages are clear and actionable
- [ ] Package can be installed and run as `swarm-agents` command

---

## Definition of Done

- [ ] `src/swarm_agents/main.py` implemented
- [ ] `tests/test_cli.py` written and passing
- [ ] `tests/test_integration.py` written and passing
- [ ] Entry point configured in pyproject.toml
- [ ] CLI tested with real Ollama
- [ ] Documentation updated with usage examples
- [ ] All tests pass (`pytest`)
- [ ] Code coverage ≥ 85%
- [ ] Code reviewed and approved
- [ ] No type errors (`mypy src/`)
- [ ] No linting errors (`ruff check src/`)

---

## Notes

- Integration tests should be marked with `@pytest.mark.integration`
- Use `ollama_available()` helper to skip tests when Ollama is not available
- CLI should handle keyboard interrupts gracefully (Ctrl+C)
- Consider adding `--verbose` flag for debugging in future iterations
- Consider adding `--output` flag to save results to file in future iterations
- The CLI should work immediately after `poetry install`
- Use Click's built-in help system for documentation

---

## Next Steps

After Wave 4A completion, the multi-agent consensus system will be:
- ✅ Fully functional with CLI interface
- ✅ Thoroughly tested with integration tests
- ✅ Ready for deployment and usage
- ✅ Installable as a Python package

Future enhancements could include:
- Web UI for visualization
- Result caching
- Progress bars for long-running tasks
- Configuration file support
- Plugin system for custom agents
- Metrics and telemetry
- Distributed execution
