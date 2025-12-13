# Wave 1A: Project Bootstrap

## Overview
This foundational wave establishes the core Python project structure, dependency management, and development environment for the multi-agent consensus system. It sets up the essential tooling and project layout that all subsequent waves will build upon. This includes configuring the Python package with pyproject.toml, installing necessary dependencies (LangGraph, LangChain, Ollama), and establishing testing infrastructure with pytest.

## Dependencies
- **Requires**: None (foundation wave)
- **Parallel with**: None (must complete first)
- **Enables**: Wave 1B (Agent Interface), Wave 1C (Ollama Integration), Wave 1D (Graph State)

## User Stories

### US-1.1: Project Setup and Dependencies

**As a** developer
**I want** a properly configured Python project with all necessary dependencies
**So that** I can begin implementing the multi-agent consensus system with LangGraph and Ollama

### Acceptance Criteria
- [ ] Python 3.11+ project initialized with modern pyproject.toml configuration
- [ ] Project follows src-layout structure with swarm_agents package
- [ ] All core dependencies (LangGraph, LangChain, Ollama) are installable via uv
- [ ] Development dependencies (pytest, pytest-asyncio) are configured
- [ ] Tests directory structure is in place with conftest.py
- [ ] .gitignore properly configured for Python projects (including .venv for uv)
- [ ] Package is installable in editable mode: `uv pip install -e .`
- [ ] Import verification test passes for all core dependencies

### Technical Implementation

#### Files to Create

**1. pyproject.toml**
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "swarm-agents"
version = "0.1.0"
description = "Multi-agent consensus system using LangGraph and Ollama"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Swarm Team"}
]
keywords = ["langgraph", "ollama", "multi-agent", "consensus", "ai"]

dependencies = [
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-community>=0.3.0",
    "ollama>=0.4.0",
    "pydantic>=2.0.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "black>=24.0.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0",
]

[project.scripts]
swarm = "swarm_agents.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --strict-markers --cov=swarm_agents --cov-report=term-missing"

[tool.black]
line-length = 100
target-version = ["py311"]
include = '\.pyi?$'

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "W", "I", "N", "UP", "ANN", "B", "A", "C4", "DTZ", "PIE", "T20", "SIM"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**2. src/swarm_agents/__init__.py**
```python
"""
Swarm Agents: Multi-agent consensus system using LangGraph and Ollama.

This package provides a framework for creating collaborative agent systems
that can reach consensus through structured deliberation and voting.
"""

__version__ = "0.1.0"
__author__ = "Swarm Team"

__all__ = ["__version__", "__author__"]
```

**3. tests/__init__.py**
```python
"""Test suite for swarm_agents package."""
```

**4. tests/conftest.py**
```python
"""
Pytest configuration and shared fixtures for swarm_agents tests.

This module provides common test fixtures and configuration used across
the test suite, including mock Ollama clients and test state objects.
"""

import pytest
from typing import Generator


@pytest.fixture(scope="session")
def project_root():
    """Provide the project root directory path."""
    import pathlib
    return pathlib.Path(__file__).parent.parent


@pytest.fixture
def sample_prompt():
    """Provide a sample prompt for testing agent responses."""
    return "What is the capital of France?"


@pytest.fixture
def sample_agent_config():
    """Provide sample agent configuration for testing."""
    return {
        "name": "test_agent",
        "model": "llama3.2:1b",
        "temperature": 0.7,
        "system_prompt": "You are a helpful test agent."
    }


# Placeholder for future Ollama mock fixtures
# Will be implemented in Wave 1C
```

**5. tests/test_setup.py**
```python
"""
Test suite for project setup and dependency verification.

Verifies that all required dependencies are properly installed and
importable, ensuring the development environment is correctly configured.
"""

import sys
import pytest


def test_python_version():
    """Verify Python version is 3.11 or higher."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"


def test_langgraph_import():
    """Verify LangGraph can be imported."""
    try:
        import langgraph
        assert langgraph is not None
    except ImportError as e:
        pytest.fail(f"Failed to import langgraph: {e}")


def test_langchain_import():
    """Verify LangChain can be imported."""
    try:
        import langchain
        from langchain_community import llms
        assert langchain is not None
        assert llms is not None
    except ImportError as e:
        pytest.fail(f"Failed to import langchain: {e}")


def test_ollama_import():
    """Verify Ollama Python client can be imported."""
    try:
        import ollama
        assert ollama is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ollama: {e}")


def test_pydantic_import():
    """Verify Pydantic can be imported."""
    try:
        from pydantic import BaseModel
        assert BaseModel is not None
    except ImportError as e:
        pytest.fail(f"Failed to import pydantic: {e}")


def test_click_import():
    """Verify Click can be imported."""
    try:
        import click
        assert click is not None
    except ImportError as e:
        pytest.fail(f"Failed to import click: {e}")


def test_package_version():
    """Verify swarm_agents package version."""
    import swarm_agents
    assert hasattr(swarm_agents, "__version__")
    assert swarm_agents.__version__ == "0.1.0"


@pytest.mark.asyncio
async def test_pytest_asyncio_configured():
    """Verify pytest-asyncio is properly configured."""
    # If this test runs, asyncio mode is working
    import asyncio
    await asyncio.sleep(0.001)
    assert True
```

**6. .gitignore**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
env.bak/
venv.bak/
.venv/

# Testing
.pytest_cache/
.coverage
.coverage.*
htmlcov/
.tox/
.nox/
coverage.xml
*.cover
.hypothesis/

# Type Checking
.mypy_cache/
.dmypy.json
dmypy.json
.pytype/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Environment Variables
.env
.env.local
.env.*.local

# Documentation
docs/_build/
site/

# Logs
*.log
logs/

# Database
*.db
*.sqlite
*.sqlite3

# Temporary Files
*.tmp
*.bak
temp/
tmp/
```

**7. README.md**
```markdown
# Swarm Agents

Multi-agent consensus system using LangGraph and Ollama.

## Overview

Swarm Agents is a framework for building collaborative AI agent systems that reach consensus through structured deliberation and voting. Built on LangGraph for orchestration and Ollama for local LLM execution.

## Requirements

- Python 3.11 or higher
- Ollama installed and running locally

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd swarm
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

## Verification

Run the test suite to verify installation:
```bash
uv run pytest tests/test_setup.py -v
```

All tests should pass, confirming that dependencies are correctly installed.

## Project Structure

```
swarm/
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # This file
├── src/
│   └── swarm_agents/      # Main package
│       └── __init__.py
└── tests/                 # Test suite
    ├── __init__.py
    ├── conftest.py        # Shared test fixtures
    └── test_setup.py      # Setup verification tests
```

## Development

### Running Tests
```bash
uv run pytest              # Run all tests
uv run pytest -v           # Verbose output
uv run pytest --cov        # With coverage report
```

### Code Quality
```bash
uv run black src/ tests/         # Format code
uv run ruff check src/ tests/    # Lint code
uv run mypy src/                 # Type check
```

## Next Steps

After completing this bootstrap wave, the following components will be added:
- Agent interface definitions (Wave 1B)
- Ollama integration layer (Wave 1C)
- LangGraph state management (Wave 1D)

## License

MIT
```

#### Directory Structure
```
swarm/
├── .gitignore
├── README.md
├── pyproject.toml
├── src/
│   └── swarm_agents/
│       └── __init__.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_setup.py
```

#### Installation Steps
1. Create directory structure (if not exists)
2. Write all configuration and source files
3. Initialize git repository (if not initialized)
4. Create virtual environment: `uv venv`
5. Activate virtual environment: `source .venv/bin/activate`
6. Install package in editable mode: `uv pip install -e ".[dev]"`
7. Verify installation: `uv run pytest tests/test_setup.py -v`

#### Testing Strategy

**Unit Tests (tests/test_setup.py)**
- `test_python_version()`: Ensures Python 3.11+ is being used
- `test_langgraph_import()`: Verifies LangGraph installation
- `test_langchain_import()`: Verifies LangChain and langchain-community
- `test_ollama_import()`: Verifies Ollama Python client
- `test_pydantic_import()`: Verifies Pydantic for data validation
- `test_click_import()`: Verifies Click for CLI support
- `test_package_version()`: Confirms package metadata is correct
- `test_pytest_asyncio_configured()`: Validates async test support

**Success Metrics**
- All 8 setup tests pass with 100% success rate
- Package is pip-installable without errors
- All dependencies resolve without conflicts
- Coverage report shows test infrastructure is working

## Success Criteria

### Functional Success
- [ ] Project installs cleanly with `uv pip install -e ".[dev]"` without errors
- [ ] All 8 dependency verification tests pass
- [ ] Package can be imported: `import swarm_agents` works in Python REPL
- [ ] pytest discovers and runs tests successfully
- [ ] Async tests execute properly with pytest-asyncio

### Code Quality Success
- [ ] pyproject.toml follows modern Python packaging standards
- [ ] Project structure follows src-layout best practices
- [ ] .gitignore covers all common Python artifacts
- [ ] README provides clear setup and verification instructions
- [ ] All files have proper docstrings and type hints where applicable

### Integration Success
- [ ] Virtual environment creation with `uv venv` succeeds on macOS/Linux/Windows
- [ ] No dependency version conflicts during installation
- [ ] Package metadata (version, author) is accessible programmatically
- [ ] Test fixtures in conftest.py are discoverable by pytest

### Documentation Success
- [ ] README.md explains installation process clearly
- [ ] Code comments explain configuration choices
- [ ] File structure is intuitive for new developers
- [ ] Next steps are clearly documented

## Estimated Effort

**Total Time**: 1-2 hours

**Breakdown**:
- Project structure creation: 15 minutes
- pyproject.toml configuration: 20 minutes
- .gitignore and README: 15 minutes
- Test infrastructure setup: 20 minutes
- Verification and debugging: 20-30 minutes
- Documentation: 10 minutes

**Complexity**: Low - This is standard Python project scaffolding

**Risk Factors**:
- Minimal - Dependency version conflicts are the main risk
- Mitigation: Use version ranges in pyproject.toml for flexibility

## Notes

### Design Decisions
- **Src-layout**: Chosen over flat layout to prevent import issues and follow modern best practices
- **pyproject.toml**: Using modern Python packaging (PEP 517/518) instead of setup.py
- **Version pinning**: Using minimum versions with flexibility to avoid lock-in
- **Test structure**: Separate conftest.py for reusable fixtures across future test modules

### Future Enhancements
- Pre-commit hooks for code quality (black, ruff, mypy)
- GitHub Actions CI/CD pipeline
- Docker containerization for consistent environments
- Documentation site with MkDocs

### Dependencies Rationale
- **LangGraph**: Core framework for building stateful multi-agent workflows
- **LangChain**: Provides abstractions for LLM interactions and chains
- **langchain-community**: Community integrations including Ollama
- **Ollama**: Python client for local LLM execution
- **Pydantic**: Data validation and settings management with type safety
- **Click**: User-friendly CLI creation (future CLI commands)
- **pytest/pytest-asyncio**: Modern testing framework with async support
- **black/ruff/mypy**: Code quality tools for maintainability
