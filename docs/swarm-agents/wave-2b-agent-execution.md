# Wave 2B: Agent Execution

## Overview
**Wave Type:** ENHANCEMENT
**Wave Number:** 2B
**Name:** Agent Execution
**Status:** Not Started

## Dependencies
**Requires:**
- Wave 1B: Ollama Client (MUST be completed)
- Wave 1C: Data Models (MUST be completed)

**Can Run in Parallel With:**
- Wave 2A: Task Decomposition
- Wave 2C: Consensus Mechanism

## Objectives
Implement the core agent execution system that:
1. Creates individual agents with unique identities
2. Executes code generation tasks using LLM
3. Manages pools of multiple agents
4. Extracts and validates Python code from LLM responses
5. Returns structured results with metadata

## User Stories

### US-3.1: Single Agent Executor
**As a** system developer
**I want** individual agents that can execute code generation tasks
**So that** I can generate multiple solution candidates for each subtask

**Acceptance Criteria:**
- Agent class initialized with OllamaClient
- Each agent has unique agent_id (UUID)
- Agent has execute(subtask: Subtask) -> AgentResult method
- Uses optimized system prompt for Python code generation
- Extracts Python code from markdown code blocks (```python ... ```)
- Returns AgentResult with:
  - agent_id: UUID of the agent
  - subtask_id: UUID of the subtask
  - code: Extracted Python code string
  - reasoning: LLM's explanation/reasoning
  - timestamp: When the result was generated

**Implementation Notes:**
- Use regex or string parsing for code block extraction
- Handle cases where no code block is present
- Preserve code formatting and indentation
- Extract reasoning text outside code blocks

---

### US-3.2: Multi-Agent Pool
**As a** consensus system
**I want** to execute the same task with multiple agents in parallel
**So that** I can collect diverse solution candidates for consensus

**Acceptance Criteria:**
- AgentPool class with configurable agent_count (default: 5)
- execute_all(subtask: Subtask) -> List[AgentResult] method
- Sequential execution (one agent at a time for local model)
- All agents receive identical subtask input
- Results collected in order of execution
- Each result has different agent_id
- All results reference same subtask_id

**Implementation Notes:**
- Sequential execution prevents overwhelming local Ollama instance
- Consider future async execution for cloud models
- Maintain agent identity consistency across executions
- Pool size should be configurable for different consensus requirements

## System Prompt Template

```python
SYSTEM_PROMPT = """You are an expert Python programmer. Generate clean, working Python code for the given task.

Rules:
1. Output ONLY the code in a python code block
2. No explanations outside the code block
3. Include necessary imports
4. Use type hints
5. Keep functions focused and simple

Task: {subtask.description}
Context: {subtask.context}
"""
```

**Prompt Design Rationale:**
- Clear instruction to output code in markdown blocks
- Emphasizes code quality and Python best practices
- Minimal but sufficient context for code generation
- Type hints improve code clarity and IDE support
- Focused functions are easier to test and validate

## File Structure

```
src/swarm_agents/
└── agent.py                 # ~200 LOC
    ├── Agent class
    ├── AgentPool class
    └── Code extraction utilities

tests/
├── test_agent.py           # ~150 LOC
│   ├── Agent unit tests
│   └── Code extraction tests
└── test_agent_pool.py      # ~100 LOC
    └── AgentPool tests
```

## Key Classes

### Agent Class
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import uuid
import re

from swarm_agents.client import OllamaClient
from swarm_agents.models import Subtask, AgentResult


class Agent:
    """Individual agent that executes code generation tasks."""

    def __init__(self, client: OllamaClient):
        """
        Initialize agent with unique ID.

        Args:
            client: OllamaClient instance for LLM communication
        """
        self.agent_id = str(uuid.uuid4())
        self.client = client

    def execute(self, subtask: Subtask) -> AgentResult:
        """
        Execute a subtask and generate code solution.

        Args:
            subtask: Subtask to solve

        Returns:
            AgentResult with generated code and metadata

        Raises:
            ValueError: If no code block found in response
        """
        # Format system prompt with subtask details
        prompt = self._format_prompt(subtask)

        # Get LLM response
        response = self.client.generate(prompt)

        # Extract code from markdown code blocks
        code = self._extract_code(response)

        # Extract reasoning (text outside code blocks)
        reasoning = self._extract_reasoning(response)

        # Create result
        return AgentResult(
            agent_id=self.agent_id,
            subtask_id=subtask.subtask_id,
            code=code,
            reasoning=reasoning,
            timestamp=datetime.utcnow()
        )

    def _format_prompt(self, subtask: Subtask) -> str:
        """Format system prompt with subtask details."""
        return f"""You are an expert Python programmer. Generate clean, working Python code for the given task.

Rules:
1. Output ONLY the code in a python code block
2. No explanations outside the code block
3. Include necessary imports
4. Use type hints
5. Keep functions focused and simple

Task: {subtask.description}
Context: {subtask.context if subtask.context else 'None'}
"""

    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from markdown code blocks.

        Args:
            response: LLM response text

        Returns:
            Extracted Python code

        Raises:
            ValueError: If no python code block found
        """
        # Pattern to match ```python ... ``` blocks
        pattern = r'```python\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            # Try without language specifier
            pattern = r'```\s*(.*?)```'
            matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            raise ValueError("No code block found in LLM response")

        # Take first code block and strip whitespace
        return matches[0].strip()

    def _extract_reasoning(self, response: str) -> str:
        """
        Extract reasoning text (content outside code blocks).

        Args:
            response: LLM response text

        Returns:
            Reasoning text or empty string
        """
        # Remove all code blocks
        cleaned = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        return cleaned.strip()
```

### AgentPool Class
```python
class AgentPool:
    """Pool of agents for multi-agent consensus."""

    def __init__(self, client: OllamaClient, agent_count: int = 5):
        """
        Initialize pool with specified number of agents.

        Args:
            client: OllamaClient instance shared by all agents
            agent_count: Number of agents to create (default: 5)
        """
        self.client = client
        self.agent_count = agent_count
        self.agents = [Agent(client) for _ in range(agent_count)]

    def execute_all(self, subtask: Subtask) -> list[AgentResult]:
        """
        Execute subtask with all agents sequentially.

        Sequential execution prevents overwhelming local Ollama instance.
        All agents receive identical subtask input.

        Args:
            subtask: Subtask to solve

        Returns:
            List of AgentResult from all agents, in execution order
        """
        results = []

        for agent in self.agents:
            result = agent.execute(subtask)
            results.append(result)

        return results

    def get_agent_ids(self) -> list[str]:
        """Get list of all agent IDs in the pool."""
        return [agent.agent_id for agent in self.agents]
```

## Test Cases

### test_agent.py
```python
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from swarm_agents.agent import Agent
from swarm_agents.client import OllamaClient
from swarm_agents.models import Subtask, AgentResult


def test_agent_has_unique_id():
    """Each agent should have a unique UUID."""
    client = Mock(spec=OllamaClient)
    agent1 = Agent(client)
    agent2 = Agent(client)

    assert agent1.agent_id != agent2.agent_id
    assert len(agent1.agent_id) == 36  # UUID format


def test_agent_execute_returns_result():
    """Agent execute should return AgentResult."""
    client = Mock(spec=OllamaClient)
    client.generate.return_value = """Here's the solution:
```python
def add(a: int, b: int) -> int:
    return a + b
```
This function adds two numbers."""

    agent = Agent(client)
    subtask = Subtask(
        subtask_id="test-123",
        description="Create an add function",
        context="Simple addition"
    )

    result = agent.execute(subtask)

    assert isinstance(result, AgentResult)
    assert result.agent_id == agent.agent_id
    assert result.subtask_id == "test-123"
    assert "def add" in result.code
    assert isinstance(result.timestamp, datetime)


def test_agent_extracts_code_from_markdown():
    """Agent should extract code from markdown code blocks."""
    client = Mock(spec=OllamaClient)
    agent = Agent(client)

    response = """Here is the code:
```python
def hello():
    print("Hello")
```
That's it!"""

    code = agent._extract_code(response)
    assert code == 'def hello():\n    print("Hello")'
    assert "Here is the code" not in code


def test_agent_extracts_code_without_language():
    """Agent should handle code blocks without language specifier."""
    client = Mock(spec=OllamaClient)
    agent = Agent(client)

    response = """```
def test():
    pass
```"""

    code = agent._extract_code(response)
    assert "def test():" in code


def test_agent_raises_on_no_code_block():
    """Agent should raise ValueError if no code block found."""
    client = Mock(spec=OllamaClient)
    agent = Agent(client)

    response = "Just some text without code blocks"

    with pytest.raises(ValueError, match="No code block found"):
        agent._extract_code(response)


def test_agent_result_includes_timestamp():
    """AgentResult should include timestamp."""
    client = Mock(spec=OllamaClient)
    client.generate.return_value = "```python\npass\n```"

    agent = Agent(client)
    subtask = Subtask(
        subtask_id="test-456",
        description="Test task",
        context=None
    )

    before = datetime.utcnow()
    result = agent.execute(subtask)
    after = datetime.utcnow()

    assert before <= result.timestamp <= after


def test_agent_extracts_reasoning():
    """Agent should extract reasoning text outside code blocks."""
    client = Mock(spec=OllamaClient)
    agent = Agent(client)

    response = """This solution uses recursion.
```python
def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)
```
Time complexity is O(n)."""

    reasoning = agent._extract_reasoning(response)
    assert "recursion" in reasoning.lower()
    assert "def factorial" not in reasoning
    assert "complexity" in reasoning.lower()
```

### test_agent_pool.py
```python
import pytest
from unittest.mock import Mock, MagicMock

from swarm_agents.agent import AgentPool
from swarm_agents.client import OllamaClient
from swarm_agents.models import Subtask, AgentResult


def test_pool_creates_correct_number_of_agents():
    """AgentPool should create specified number of agents."""
    client = Mock(spec=OllamaClient)

    pool = AgentPool(client, agent_count=5)
    assert len(pool.agents) == 5

    pool2 = AgentPool(client, agent_count=10)
    assert len(pool2.agents) == 10


def test_pool_default_agent_count():
    """AgentPool should default to 5 agents."""
    client = Mock(spec=OllamaClient)
    pool = AgentPool(client)

    assert pool.agent_count == 5
    assert len(pool.agents) == 5


def test_pool_executes_all_agents():
    """AgentPool should execute task with all agents."""
    client = Mock(spec=OllamaClient)
    client.generate.return_value = "```python\npass\n```"

    pool = AgentPool(client, agent_count=3)
    subtask = Subtask(
        subtask_id="test-789",
        description="Test task",
        context=None
    )

    results = pool.execute_all(subtask)

    assert len(results) == 3
    assert all(isinstance(r, AgentResult) for r in results)


def test_pool_results_have_different_agent_ids():
    """Each result should have different agent_id."""
    client = Mock(spec=OllamaClient)
    client.generate.return_value = "```python\npass\n```"

    pool = AgentPool(client, agent_count=4)
    subtask = Subtask(
        subtask_id="test-101",
        description="Test task",
        context=None
    )

    results = pool.execute_all(subtask)
    agent_ids = [r.agent_id for r in results]

    assert len(agent_ids) == len(set(agent_ids))  # All unique


def test_pool_results_have_same_subtask_id():
    """All results should reference the same subtask_id."""
    client = Mock(spec=OllamaClient)
    client.generate.return_value = "```python\npass\n```"

    pool = AgentPool(client, agent_count=3)
    subtask = Subtask(
        subtask_id="test-202",
        description="Test task",
        context=None
    )

    results = pool.execute_all(subtask)
    subtask_ids = [r.subtask_id for r in results]

    assert all(sid == "test-202" for sid in subtask_ids)


def test_pool_executes_sequentially():
    """AgentPool should execute agents sequentially (not in parallel)."""
    client = Mock(spec=OllamaClient)
    call_order = []

    def track_call(*args, **kwargs):
        call_order.append(len(call_order))
        return "```python\npass\n```"

    client.generate.side_effect = track_call

    pool = AgentPool(client, agent_count=3)
    subtask = Subtask(
        subtask_id="test-303",
        description="Test task",
        context=None
    )

    pool.execute_all(subtask)

    # Calls should be sequential: [0, 1, 2]
    assert call_order == [0, 1, 2]


def test_pool_get_agent_ids():
    """AgentPool should return list of all agent IDs."""
    client = Mock(spec=OllamaClient)
    pool = AgentPool(client, agent_count=3)

    agent_ids = pool.get_agent_ids()

    assert len(agent_ids) == 3
    assert all(isinstance(aid, str) for aid in agent_ids)
    assert len(set(agent_ids)) == 3  # All unique
```

## Technical Implementation Details

### Code Extraction Strategy
1. **Primary pattern**: Match ````python ... ```blocks
2. **Fallback pattern**: Match ```` ... ``` blocks without language
3. **Error handling**: Raise ValueError if no blocks found
4. **Whitespace**: Strip leading/trailing whitespace from extracted code
5. **Multiple blocks**: Take first block only

### Sequential Execution Rationale
- Local Ollama instances have limited concurrent capacity
- Sequential execution ensures consistent performance
- Prevents resource exhaustion on development machines
- Future enhancement: Async execution for cloud deployments

### Agent Identity
- UUID4 ensures unique agent IDs
- Agent IDs persist across multiple executions
- Useful for tracking agent performance in consensus
- Enables future agent-specific fine-tuning

### Error Handling
- ValueError if no code block found in LLM response
- Client errors propagated to caller
- Invalid subtask data caught by Pydantic models

## Integration Points

### Dependencies (MUST exist before this wave)
- `swarm_agents.client.OllamaClient` from Wave 1B
- `swarm_agents.models.Subtask` from Wave 1C
- `swarm_agents.models.AgentResult` from Wave 1C

### Provided to Future Waves
- `swarm_agents.agent.Agent` class
- `swarm_agents.agent.AgentPool` class
- Code extraction utilities

### Used By
- Wave 2C: Consensus Mechanism (uses AgentPool)
- Wave 3: Full Pipeline (orchestrates agent execution)

## Success Criteria

### Functional Requirements
- [ ] Agent generates unique UUID on initialization
- [ ] Agent executes subtasks using OllamaClient
- [ ] Agent extracts code from markdown blocks
- [ ] Agent returns AgentResult with all required fields
- [ ] AgentPool creates configurable number of agents
- [ ] AgentPool executes all agents sequentially
- [ ] All results have unique agent_id
- [ ] All results reference same subtask_id

### Non-Functional Requirements
- [ ] Code extraction handles edge cases (no language, multiple blocks)
- [ ] Proper error handling for missing code blocks
- [ ] Type hints on all public methods
- [ ] 100% test coverage for Agent and AgentPool
- [ ] Tests use mocks (no actual LLM calls)
- [ ] Documentation strings on all public methods

### Quality Gates
- [ ] All tests pass
- [ ] No linting errors (ruff check)
- [ ] Type checking passes (mypy)
- [ ] Test coverage >= 90%

## Performance Considerations

### Sequential vs Parallel Execution
- **Current**: Sequential execution for local Ollama
- **Rationale**: Local models have limited concurrent capacity
- **Future**: Consider async execution for cloud models
- **Tradeoff**: Slower but more reliable for development

### Memory Usage
- Agent pool size directly impacts memory
- Each agent holds reference to shared client
- Default 5 agents is reasonable for most machines
- Consider lower count for resource-constrained environments

### LLM Response Time
- Each agent execution waits for LLM response
- Total time = agent_count * avg_response_time
- 5 agents × 2s avg = ~10s per subtask
- Acceptable for prototype, optimize in production

## Future Enhancements

### Phase 2 Improvements
- [ ] Async execution with configurable concurrency
- [ ] Agent-specific temperature/sampling parameters
- [ ] Retry logic for failed generations
- [ ] Streaming support for partial results
- [ ] Agent performance metrics and tracking

### Advanced Features
- [ ] Multi-model support (different LLMs per agent)
- [ ] Dynamic agent pool scaling
- [ ] Code validation before returning result
- [ ] Caching for identical subtasks
- [ ] Agent specialization (web, data, algorithms, etc.)

## Risk Assessment

### High Risk
- **LLM availability**: Ollama must be running and responsive
  - Mitigation: Clear error messages, retry logic in future
- **Code extraction failures**: LLM may not follow format
  - Mitigation: Fallback patterns, clear error messages

### Medium Risk
- **Performance**: Sequential execution may be slow
  - Mitigation: Acceptable for prototype, async in future
- **Resource usage**: Multiple agents consume memory
  - Mitigation: Configurable pool size, reasonable defaults

### Low Risk
- **UUID collisions**: Statistically negligible
- **Type errors**: Caught by Pydantic and mypy

## Testing Strategy

### Unit Tests
- Mock OllamaClient to avoid LLM dependencies
- Test code extraction with various markdown formats
- Verify agent identity and result metadata
- Test pool execution order and result collection

### Integration Tests (Future)
- Test with actual Ollama instance
- Validate end-to-end code generation
- Performance benchmarks with real LLM

### Edge Cases
- No code blocks in response
- Multiple code blocks (take first)
- Code blocks without language specifier
- Empty code blocks
- Malformed markdown

## Documentation Requirements

### Code Documentation
- [ ] Docstrings on all classes and public methods
- [ ] Type hints on all function signatures
- [ ] Inline comments for complex logic (regex, extraction)
- [ ] Module-level docstring explaining purpose

### User Documentation
- [ ] README section on agent execution
- [ ] Example usage of Agent and AgentPool
- [ ] Configuration guide (agent count, client setup)
- [ ] Troubleshooting common issues

## Acceptance Checklist

Before marking this wave as complete:
- [ ] All user stories implemented
- [ ] All test cases passing
- [ ] Code coverage >= 90%
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)
- [ ] Documentation complete
- [ ] Integration with Wave 1B and 1C verified
- [ ] Code review completed
- [ ] Performance acceptable (< 15s for 5 agents)

## Estimated Effort
- **Development**: 4-6 hours
- **Testing**: 2-3 hours
- **Documentation**: 1-2 hours
- **Total**: 7-11 hours

## Notes
- Keep prompt simple and focused on code generation
- Consider future expansion to support different task types
- Agent identity will be useful for performance analysis later
- Sequential execution is intentional for local development
