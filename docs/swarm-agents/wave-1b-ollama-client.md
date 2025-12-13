# Wave 1B: Ollama Client Wrapper

## Overview

Wave 1B implements a robust wrapper around the Ollama library to provide a standardized interface for LLM interactions across the swarm-agents system. This foundation component enables all agent-based functionality by abstracting away the complexity of LLM communication, error handling, and retry logic.

The OllamaClient class serves as the primary interface for generating responses from local LLM models, with intelligent error handling and configurable retry mechanisms to ensure reliability in production environments.

## Dependencies

- **Requires**: Wave 1A (Project Bootstrap)
- **Parallel with**: Wave 1C (Agent Configuration Schema), Wave 1D (Consensus Protocol Interface)
- **Enables**: Wave 2A (Agent Implementation), Wave 2B (Response Aggregation)

## User Stories

### US-1.2: Ollama Client Wrapper

**As a** developer building agent-based systems
**I want** a reliable wrapper around the Ollama LLM client
**So that** I can generate responses with built-in error handling and retry logic

#### Story Details

Create a production-ready OllamaClient class that:
- Wraps the ollama Python library with a clean, simple interface
- Provides configurable model selection (default: qwen3-coder)
- Implements exponential backoff retry logic for transient failures
- Handles connection errors gracefully with custom exceptions
- Supports timeout configuration to prevent hanging requests
- Logs all LLM interactions for debugging and monitoring

#### Acceptance Criteria

- [ ] OllamaClient class initializes with sensible defaults
- [ ] `generate(prompt: str) -> str` method returns LLM responses
- [ ] Custom `OllamaConnectionError` exception for connection failures
- [ ] Retry logic attempts up to 3 times (configurable) with exponential backoff
- [ ] Timeout configuration prevents indefinite waiting (default: 120s)
- [ ] All errors are logged with appropriate context
- [ ] Model can be configured at initialization
- [ ] Unit tests cover happy path and error scenarios
- [ ] Test coverage exceeds 90%

## Technical Implementation

### Component Architecture

#### **llm.py** (~80-100 LOC)

```python
"""
LLM client wrapper for Ollama integration.

This module provides a robust interface to the Ollama LLM service with
built-in error handling, retry logic, and timeout management.
"""

import time
import logging
from typing import Optional
import ollama


logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when connection to Ollama service fails."""
    pass


class OllamaClient:
    """
    Wrapper around the ollama library with retry logic and error handling.

    Attributes:
        model: Name of the LLM model to use (e.g., 'qwen3-coder')
        retries: Maximum number of retry attempts on failure
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        model: str = "qwen3-coder",
        retries: int = 3,
        timeout: int = 120
    ):
        """
        Initialize the Ollama client.

        Args:
            model: Model name to use for generation (default: qwen3-coder)
            retries: Number of retry attempts on failure (default: 3)
            timeout: Request timeout in seconds (default: 120)
        """
        self.model = model
        self.retries = retries
        self.timeout = timeout
        logger.info(
            f"Initialized OllamaClient with model={model}, "
            f"retries={retries}, timeout={timeout}"
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM with retry logic.

        Args:
            prompt: Input prompt for the LLM

        Returns:
            Generated text response from the model

        Raises:
            OllamaConnectionError: If connection fails after all retries
            ValueError: If prompt is empty or invalid
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        last_error = None

        for attempt in range(self.retries):
            try:
                logger.debug(
                    f"Attempt {attempt + 1}/{self.retries} to generate "
                    f"response for prompt (length={len(prompt)})"
                )

                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={'timeout': self.timeout}
                )

                result = response.get('response', '')
                logger.info(
                    f"Successfully generated response (length={len(result)})"
                )
                return result

            except (ConnectionError, TimeoutError, Exception) as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}"
                )

                if attempt < self.retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        # All retries exhausted
        error_msg = (
            f"Failed to connect to Ollama after {self.retries} attempts. "
            f"Last error: {last_error}"
        )
        logger.error(error_msg)
        raise OllamaConnectionError(error_msg) from last_error
```

**Key Design Decisions:**

1. **Exponential Backoff**: Retry delays increase exponentially (1s, 2s, 4s) to handle temporary service disruptions
2. **Comprehensive Logging**: All attempts, successes, and failures are logged for debugging
3. **Custom Exception**: `OllamaConnectionError` provides clear error signaling
4. **Input Validation**: Empty prompts are rejected early with `ValueError`
5. **Configurable Defaults**: Sensible defaults (qwen3-coder, 3 retries, 120s timeout) work for most cases

### Testing Strategy

#### **tests/test_ollama_client.py** (~150-200 LOC)

**Test Cases:**

1. **Initialization Tests**
   - `test_client_initialization_with_default_model`: Verify default model is qwen3-coder
   - `test_client_initialization_with_custom_model`: Verify custom model is set correctly
   - `test_client_initialization_with_custom_retries`: Verify retry count configuration
   - `test_client_initialization_with_custom_timeout`: Verify timeout configuration

2. **Happy Path Tests**
   - `test_generate_returns_string`: Verify successful generation returns string
   - `test_generate_with_simple_prompt`: Test basic prompt-response flow
   - `test_generate_with_complex_prompt`: Test multi-line, complex prompts

3. **Error Handling Tests**
   - `test_generate_raises_on_empty_prompt`: Verify ValueError for empty input
   - `test_generate_handles_connection_error`: Mock connection failure, verify exception
   - `test_generate_retries_on_failure`: Verify retry logic is invoked
   - `test_generate_succeeds_on_second_retry`: Verify recovery after transient failure
   - `test_generate_exhausts_retries`: Verify failure after all retries exhausted

4. **Integration Tests** (optional, requires Ollama running)
   - `test_integration_real_ollama`: End-to-end test with actual Ollama service

**Mock Strategy:**
- Use `unittest.mock.patch` to mock `ollama.generate`
- Mock responses should match actual Ollama response structure
- Mock connection errors with `ConnectionError` and `TimeoutError`

**Example Test Structure:**

```python
import pytest
from unittest.mock import patch, MagicMock
from swarm_agents.llm import OllamaClient, OllamaConnectionError


class TestOllamaClientInitialization:
    def test_client_initialization_with_default_model(self):
        client = OllamaClient()
        assert client.model == "qwen3-coder"
        assert client.retries == 3
        assert client.timeout == 120

    def test_client_initialization_with_custom_model(self):
        client = OllamaClient(model="llama2")
        assert client.model == "llama2"


class TestOllamaClientGenerate:
    @patch('ollama.generate')
    def test_generate_returns_string(self, mock_generate):
        mock_generate.return_value = {'response': 'Test response'}
        client = OllamaClient()
        result = client.generate("Test prompt")
        assert result == 'Test response'
        assert isinstance(result, str)

    def test_generate_raises_on_empty_prompt(self):
        client = OllamaClient()
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            client.generate("")

    @patch('ollama.generate')
    def test_generate_handles_connection_error(self, mock_generate):
        mock_generate.side_effect = ConnectionError("Service unavailable")
        client = OllamaClient(retries=2)

        with pytest.raises(OllamaConnectionError):
            client.generate("Test prompt")

        assert mock_generate.call_count == 2
```

### File Structure

```
src/swarm_agents/
├── __init__.py
└── llm.py                    # ~80-100 LOC

tests/
├── __init__.py
└── test_ollama_client.py     # ~150-200 LOC
```

### Dependencies

**Runtime:**
- `ollama` - Official Ollama Python client

**Development:**
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities

## Success Criteria

- [ ] OllamaClient class implemented in `src/swarm_agents/llm.py`
- [ ] All acceptance criteria met
- [ ] Unit tests pass with >90% coverage
- [ ] Code follows project style guidelines (PEP 8, type hints)
- [ ] Documentation strings complete for all public methods
- [ ] Integration test passes with local Ollama instance
- [ ] Error handling verified through fault injection tests
- [ ] Logging output is clear and actionable

## Validation Steps

1. **Unit Testing**
   ```bash
   pytest tests/test_ollama_client.py -v --cov=src/swarm_agents/llm
   ```

2. **Manual Integration Test**
   ```bash
   # Ensure Ollama is running locally
   ollama serve

   # Run integration test
   pytest tests/test_ollama_client.py::test_integration_real_ollama
   ```

3. **Error Scenario Testing**
   ```bash
   # Stop Ollama to test connection errors
   # Run tests to verify proper error handling
   pytest tests/test_ollama_client.py::TestOllamaClientGenerate::test_generate_handles_connection_error
   ```

## Estimated Effort

**2-3 hours**

- Implementation: 1-1.5 hours
- Unit testing: 0.5-1 hour
- Integration testing: 0.5 hour
- Documentation: 0.25 hour

## Notes

- This wave is marked as FOUNDATION because it provides critical infrastructure used by all subsequent agent implementations
- The retry logic is essential for production reliability given potential network instability
- Default model (qwen3-coder) is optimized for code generation tasks but can be swapped for other models
- Timeout of 120s is conservative; adjust based on model performance and hardware
- Consider adding response caching in future waves for repeated prompts

## Next Steps

After completing Wave 1B, proceed with:
- **Wave 1C**: Agent Configuration Schema (can run in parallel)
- **Wave 1D**: Consensus Protocol Interface (can run in parallel)
- **Wave 2A**: Agent Implementation (requires 1B, 1C, 1D)
