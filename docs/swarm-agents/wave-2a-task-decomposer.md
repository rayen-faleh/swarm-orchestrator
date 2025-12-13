# Wave 2A: Task Decomposer

## Overview

**Wave Type:** ENHANCEMENT
**Status:** Not Started
**Dependencies:** Wave 1B (Ollama Client), Wave 1C (Data Models)
**Parallel Execution:** Can run in parallel with Wave 2B, Wave 2C

This wave implements the Task Decomposer component, which analyzes coding tasks and breaks them down into atomic subtasks when necessary. The decomposer uses an LLM to determine if a task is already atomic (single function) or complex (requires multiple functions), and generates appropriate subtask breakdowns.

## User Story

**US-2.2: Decomposer Node**

As a swarm orchestrator, I need a task decomposer that can:
- Accept a coding query and determine if it's atomic or complex
- Break complex tasks into 2-5 clear, actionable subtasks
- Return atomic tasks unchanged (as single subtask)
- Handle malformed LLM responses gracefully with retry logic
- Provide structured output with subtask IDs, descriptions, context, and dependencies

## Architecture

### Component Structure

```
src/swarm_agents/
└── decomposer.py  (~150 LOC)
    ├── DecompositionError (Exception)
    └── Decomposer (Class)

tests/
└── test_decomposer.py (~200 LOC)
```

### Key Classes

```python
class DecompositionError(Exception):
    """Raised when decomposition fails after retries."""
    pass


class Decomposer:
    """Decomposes coding tasks into atomic subtasks using LLM."""

    def __init__(self, client: OllamaClient, max_retries: int = 3):
        """
        Initialize the decomposer.

        Args:
            client: OllamaClient instance for LLM communication
            max_retries: Maximum number of retries for malformed responses
        """
        self.client = client
        self.max_retries = max_retries

    def decompose(self, query: str) -> DecompositionResult:
        """
        Decompose a coding query into subtasks.

        Args:
            query: The coding task to decompose

        Returns:
            DecompositionResult containing subtasks

        Raises:
            DecompositionError: If decomposition fails after max retries
        """
        pass

    def _build_prompt(self, query: str) -> str:
        """Build the decomposition prompt with system instructions."""
        pass

    def _parse_response(self, response: str) -> DecompositionResult:
        """
        Parse JSON response from LLM.

        Args:
            response: Raw LLM response string

        Returns:
            DecompositionResult object

        Raises:
            ValueError: If response cannot be parsed
        """
        pass

    def _validate_subtasks(self, subtasks: list[Subtask]) -> None:
        """
        Validate that subtasks meet requirements.

        Args:
            subtasks: List of subtasks to validate

        Raises:
            ValueError: If validation fails
        """
        pass
```

## System Prompt

The decomposer uses the following system prompt for LLM communication:

```
You are a task decomposer for a coding agent swarm. Given a coding task, determine if it's atomic (single function/operation) or complex (needs multiple functions/operations).

Output JSON format:
{
  "is_atomic": true/false,
  "subtasks": [
    {
      "id": "1",
      "description": "Clear, actionable description of what to implement",
      "context": "Additional context or constraints for this subtask",
      "dependencies": []
    }
  ]
}

Rules:
1. For atomic tasks: Return single subtask with the original query as description
2. For complex tasks: Break into 2-5 independent subtasks
3. Each subtask should be implementable by a single agent
4. Subtask IDs must be unique strings
5. Dependencies array contains IDs of subtasks that must complete first
6. Descriptions must be clear, specific, and actionable
7. Include relevant context (libraries, patterns, constraints)

Example atomic task:
Query: "Write a function to calculate fibonacci numbers"
Response: {"is_atomic": true, "subtasks": [{"id": "1", "description": "Write a function to calculate fibonacci numbers", "context": "Use recursive or iterative approach", "dependencies": []}]}

Example complex task:
Query: "Build a REST API for user management"
Response: {
  "is_atomic": false,
  "subtasks": [
    {"id": "1", "description": "Create User model with validation", "context": "Include email, password hash, created_at fields", "dependencies": []},
    {"id": "2", "description": "Implement user registration endpoint", "context": "POST /users with email/password validation", "dependencies": ["1"]},
    {"id": "3", "description": "Implement user authentication endpoint", "context": "POST /auth/login with JWT token generation", "dependencies": ["1"]},
    {"id": "4", "description": "Implement get user profile endpoint", "context": "GET /users/:id with auth middleware", "dependencies": ["1", "3"]}
  ]
}
```

## Implementation Details

### Decomposition Logic

1. **Prompt Construction**
   - Combine system prompt with user query
   - Include formatting instructions
   - Request structured JSON output

2. **LLM Interaction**
   - Send prompt to OllamaClient
   - Set appropriate temperature (0.3 for consistency)
   - Use JSON mode if available

3. **Response Parsing**
   - Parse JSON response
   - Validate structure and required fields
   - Convert to DecompositionResult and Subtask objects
   - Handle missing or malformed fields

4. **Retry Logic**
   - Catch JSON parsing errors
   - Retry with clarified prompt (up to max_retries)
   - Include error feedback in retry prompt
   - Raise DecompositionError if all retries fail

5. **Validation**
   - Ensure 1-5 subtasks returned
   - Verify all subtask IDs are unique
   - Check all dependency IDs reference valid subtasks
   - Validate no circular dependencies
   - Ensure descriptions are non-empty

### Error Handling

```python
# Retry with feedback on parsing errors
for attempt in range(self.max_retries):
    try:
        response = self.client.generate(prompt)
        result = self._parse_response(response)
        self._validate_subtasks(result.subtasks)
        return result
    except (json.JSONDecodeError, ValueError) as e:
        if attempt == self.max_retries - 1:
            raise DecompositionError(f"Failed to decompose after {self.max_retries} attempts: {e}")
        # Add error feedback to prompt for retry
        prompt = self._build_retry_prompt(query, str(e))
```

## Test Cases

### test_decomposer.py

```python
import pytest
from swarm_agents.decomposer import Decomposer, DecompositionError
from swarm_agents.models import DecompositionResult, Subtask
from swarm_agents.ollama_client import OllamaClient


class TestDecomposer:
    """Test suite for Decomposer class."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create a mock OllamaClient."""
        return mocker.Mock(spec=OllamaClient)

    @pytest.fixture
    def decomposer(self, mock_client):
        """Create a Decomposer instance with mock client."""
        return Decomposer(client=mock_client)

    def test_atomic_task_not_decomposed(self, decomposer, mock_client):
        """Test that atomic tasks return single subtask."""
        query = "Write a function to calculate fibonacci numbers"
        mock_response = {
            "is_atomic": True,
            "subtasks": [
                {
                    "id": "1",
                    "description": query,
                    "context": "Use iterative approach",
                    "dependencies": []
                }
            ]
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        result = decomposer.decompose(query)

        assert result.is_atomic is True
        assert len(result.subtasks) == 1
        assert result.subtasks[0].description == query
        assert result.subtasks[0].dependencies == []

    def test_complex_task_decomposed(self, decomposer, mock_client):
        """Test that complex tasks are broken into multiple subtasks."""
        query = "Build a REST API for user management"
        mock_response = {
            "is_atomic": False,
            "subtasks": [
                {
                    "id": "1",
                    "description": "Create User model",
                    "context": "Include validation",
                    "dependencies": []
                },
                {
                    "id": "2",
                    "description": "Implement registration endpoint",
                    "context": "POST /users",
                    "dependencies": ["1"]
                },
                {
                    "id": "3",
                    "description": "Implement authentication endpoint",
                    "context": "POST /auth/login",
                    "dependencies": ["1"]
                }
            ]
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        result = decomposer.decompose(query)

        assert result.is_atomic is False
        assert len(result.subtasks) == 3
        assert result.subtasks[1].dependencies == ["1"]
        assert result.subtasks[2].dependencies == ["1"]

    def test_decomposer_handles_malformed_response(self, decomposer, mock_client):
        """Test retry logic for malformed JSON responses."""
        query = "Write a sorting function"

        # First attempt: malformed JSON
        # Second attempt: valid JSON
        mock_client.generate.side_effect = [
            "This is not JSON",
            json.dumps({
                "is_atomic": True,
                "subtasks": [
                    {
                        "id": "1",
                        "description": query,
                        "context": "",
                        "dependencies": []
                    }
                ]
            })
        ]

        result = decomposer.decompose(query)

        assert mock_client.generate.call_count == 2
        assert len(result.subtasks) == 1

    def test_subtasks_have_unique_ids(self, decomposer, mock_client):
        """Test validation fails for duplicate subtask IDs."""
        mock_response = {
            "is_atomic": False,
            "subtasks": [
                {"id": "1", "description": "Task 1", "context": "", "dependencies": []},
                {"id": "1", "description": "Task 2", "context": "", "dependencies": []}
            ]
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        with pytest.raises(DecompositionError):
            decomposer.decompose("Build something")

    def test_decomposer_retries_on_json_error(self, decomposer, mock_client):
        """Test that decomposer retries on JSON parsing errors."""
        mock_client.generate.side_effect = [
            "Invalid JSON 1",
            "Invalid JSON 2",
            "Invalid JSON 3"
        ]

        decomposer_with_retries = Decomposer(client=mock_client, max_retries=3)

        with pytest.raises(DecompositionError) as exc_info:
            decomposer_with_retries.decompose("Test query")

        assert "Failed to decompose after 3 attempts" in str(exc_info.value)
        assert mock_client.generate.call_count == 3

    def test_validates_dependency_references(self, decomposer, mock_client):
        """Test validation fails for invalid dependency references."""
        mock_response = {
            "is_atomic": False,
            "subtasks": [
                {"id": "1", "description": "Task 1", "context": "", "dependencies": []},
                {"id": "2", "description": "Task 2", "context": "", "dependencies": ["999"]}
            ]
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        with pytest.raises(DecompositionError):
            decomposer.decompose("Build something")

    def test_validates_subtask_count(self, decomposer, mock_client):
        """Test validation fails for too many subtasks."""
        mock_response = {
            "is_atomic": False,
            "subtasks": [
                {"id": str(i), "description": f"Task {i}", "context": "", "dependencies": []}
                for i in range(1, 7)  # 6 subtasks
            ]
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        with pytest.raises(DecompositionError):
            decomposer.decompose("Build something")

    def test_validates_no_circular_dependencies(self, decomposer, mock_client):
        """Test validation fails for circular dependencies."""
        mock_response = {
            "is_atomic": False,
            "subtasks": [
                {"id": "1", "description": "Task 1", "context": "", "dependencies": ["2"]},
                {"id": "2", "description": "Task 2", "context": "", "dependencies": ["1"]}
            ]
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        with pytest.raises(DecompositionError):
            decomposer.decompose("Build something")

    def test_empty_description_rejected(self, decomposer, mock_client):
        """Test validation fails for empty descriptions."""
        mock_response = {
            "is_atomic": True,
            "subtasks": [
                {"id": "1", "description": "", "context": "", "dependencies": []}
            ]
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        with pytest.raises(DecompositionError):
            decomposer.decompose("Build something")
```

## Acceptance Criteria

- [ ] Decomposer class accepts OllamaClient in constructor
- [ ] decompose() method returns DecompositionResult
- [ ] Atomic tasks are recognized and returned as single subtask
- [ ] Complex tasks are broken into 2-5 subtasks
- [ ] Each subtask has unique ID, description, context, and dependencies
- [ ] LLM receives properly formatted system prompt
- [ ] JSON parsing errors trigger retry logic (up to max_retries)
- [ ] Validation ensures unique IDs and valid dependency references
- [ ] Validation prevents circular dependencies
- [ ] Validation enforces 1-5 subtask limit
- [ ] DecompositionError raised after max retries exceeded
- [ ] All test cases pass with >90% coverage

## Integration Points

### Dependencies
- **Wave 1B (Ollama Client):** Uses OllamaClient.generate() for LLM communication
- **Wave 1C (Data Models):** Returns DecompositionResult and Subtask objects

### Downstream Impact
- **Wave 3A (Coordinator):** Will use Decomposer to break down initial queries
- **Wave 4A (Consensus Evaluator):** May use subtask structure for evaluation

## Performance Considerations

- **Retry Strategy:** Default max_retries=3 balances reliability vs. latency
- **LLM Temperature:** Use 0.3 for consistent, deterministic decomposition
- **Prompt Length:** System prompt ~500 tokens, leaves room for complex queries
- **Validation Overhead:** O(n²) for dependency validation, acceptable for n≤5

## Edge Cases

1. **Empty Query:** Raise ValueError immediately
2. **Very Long Query:** Truncate or summarize before sending to LLM
3. **Ambiguous Complexity:** LLM decides atomic vs. complex (no exact rules)
4. **Missing Context Field:** Default to empty string
5. **Non-Array Dependencies:** Convert to empty array
6. **Self-Dependency:** Validation error (ID in own dependencies)

## Documentation Requirements

- [ ] Docstrings for all public methods
- [ ] Type hints for all parameters and returns
- [ ] README section explaining decomposition strategy
- [ ] Example usage in documentation
- [ ] System prompt documented with examples

## Example Usage

```python
from swarm_agents.ollama_client import OllamaClient
from swarm_agents.decomposer import Decomposer

# Initialize client and decomposer
client = OllamaClient(base_url="http://localhost:11434")
decomposer = Decomposer(client=client)

# Decompose a simple task
result = decomposer.decompose("Write a function to reverse a string")
print(f"Atomic: {result.is_atomic}")
print(f"Subtasks: {len(result.subtasks)}")
# Output:
# Atomic: True
# Subtasks: 1

# Decompose a complex task
result = decomposer.decompose("Build a user authentication system with JWT")
print(f"Atomic: {result.is_atomic}")
print(f"Subtasks: {len(result.subtasks)}")
for subtask in result.subtasks:
    print(f"  {subtask.id}: {subtask.description}")
    if subtask.dependencies:
        print(f"    Depends on: {', '.join(subtask.dependencies)}")
# Output:
# Atomic: False
# Subtasks: 4
#   1: Create User model with password hashing
#   2: Implement user registration endpoint
#     Depends on: 1
#   3: Implement JWT token generation
#     Depends on: 1
#   4: Implement protected route middleware
#     Depends on: 3
```

## Success Metrics

- **Decomposition Accuracy:** >85% of complex tasks properly decomposed
- **Retry Rate:** <20% of requests require retries
- **Response Time:** <2s average for decomposition (including LLM call)
- **Test Coverage:** >90% code coverage
- **No Circular Dependencies:** 100% detection rate in validation

## Timeline Estimate

- **Implementation:** 4-6 hours
- **Testing:** 3-4 hours
- **Documentation:** 1-2 hours
- **Total:** 8-12 hours

## Notes

- This wave is foundational for the coordinator (Wave 3A)
- Consider adding caching for repeated queries in future enhancements
- May need to tune system prompt based on real-world LLM performance
- JSON mode support varies by model (fallback to regex parsing if needed)
