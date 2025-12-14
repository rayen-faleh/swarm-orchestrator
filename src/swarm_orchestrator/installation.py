"""
Detection of swarm installation context.

Provides utilities to detect:
- Whether the current directory is a Python project
- Whether swarm is installed locally in the project or globally (pipx/system)
"""

import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class InstallationType(Enum):
    """How swarm-orchestrator is installed."""
    LOCAL = "local"      # Installed in project's venv/editable install
    PIPX = "pipx"        # Installed via pipx
    SYSTEM = "system"    # Installed in system Python
    UNKNOWN = "unknown"  # Cannot determine


@dataclass
class PythonProject:
    """Information about a detected Python project."""
    root: Path
    has_pyproject_toml: bool = False
    has_setup_py: bool = False
    has_requirements_txt: bool = False
    has_setup_cfg: bool = False

    @property
    def is_valid(self) -> bool:
        """Whether this appears to be a valid Python project."""
        return any([
            self.has_pyproject_toml,
            self.has_setup_py,
            self.has_requirements_txt,
            self.has_setup_cfg,
        ])


@dataclass
class InstallationContext:
    """Complete information about swarm's installation context."""
    installation_type: InstallationType
    python_project: Optional[PythonProject]
    swarm_location: Optional[Path]  # Where swarm package is installed

    @property
    def is_local_install(self) -> bool:
        """Whether swarm is installed locally in a project."""
        return self.installation_type == InstallationType.LOCAL

    @property
    def is_global_install(self) -> bool:
        """Whether swarm is installed globally (pipx or system)."""
        return self.installation_type in (InstallationType.PIPX, InstallationType.SYSTEM)

    @property
    def in_python_project(self) -> bool:
        """Whether we're in a Python project directory."""
        return self.python_project is not None and self.python_project.is_valid


def detect_python_project(directory: Optional[Path] = None) -> Optional[PythonProject]:
    """
    Detect if a directory is a Python project.

    Looks for common Python project markers:
    - pyproject.toml
    - setup.py
    - setup.cfg
    - requirements.txt

    Args:
        directory: Directory to check (defaults to current working directory)

    Returns:
        PythonProject if markers found, None otherwise
    """
    if directory is None:
        directory = Path.cwd()

    directory = Path(directory).resolve()

    if not directory.is_dir():
        return None

    project = PythonProject(
        root=directory,
        has_pyproject_toml=(directory / "pyproject.toml").is_file(),
        has_setup_py=(directory / "setup.py").is_file(),
        has_requirements_txt=(directory / "requirements.txt").is_file(),
        has_setup_cfg=(directory / "setup.cfg").is_file(),
    )

    return project if project.is_valid else None


def _get_swarm_package_location() -> Optional[Path]:
    """Get the location where swarm-orchestrator is installed."""
    try:
        import swarm_orchestrator
        location = Path(swarm_orchestrator.__file__).parent
        return location.resolve()
    except ImportError:
        return None


def _is_in_pipx() -> bool:
    """Check if swarm is running from a pipx installation."""
    location = _get_swarm_package_location()
    if location is None:
        return False

    # pipx installs to ~/.local/pipx/venvs/<package>/...
    # or on some systems ~/.local/share/pipx/venvs/<package>/...
    location_str = str(location)
    return "pipx" in location_str and "venvs" in location_str


def _is_in_project_venv(project_root: Path) -> bool:
    """Check if swarm is installed in the project's virtual environment."""
    location = _get_swarm_package_location()
    if location is None:
        return False

    project_root = project_root.resolve()

    # Check if package location is under project root
    # This handles both .venv and venv directories, plus editable installs
    try:
        location.relative_to(project_root)
        return True
    except ValueError:
        pass

    # Also check if we're running from the same venv as the project
    # by checking if sys.prefix is under the project
    try:
        Path(sys.prefix).resolve().relative_to(project_root)
        return True
    except ValueError:
        pass

    return False


def _is_editable_install() -> bool:
    """Check if swarm is installed as an editable/development install."""
    location = _get_swarm_package_location()
    if location is None:
        return False

    # Editable installs typically have the source directly accessible
    # and are in a src/ directory structure
    return "src" in str(location) and "site-packages" not in str(location)


def detect_installation_type(project_root: Optional[Path] = None) -> InstallationType:
    """
    Detect how swarm-orchestrator is installed.

    Args:
        project_root: Root of the Python project to check against

    Returns:
        InstallationType indicating installation method
    """
    if _is_in_pipx():
        return InstallationType.PIPX

    if project_root is not None:
        if _is_in_project_venv(project_root):
            return InstallationType.LOCAL
        if _is_editable_install():
            return InstallationType.LOCAL

    # Check if we're in site-packages (system or user install)
    location = _get_swarm_package_location()
    if location is not None and "site-packages" in str(location):
        return InstallationType.SYSTEM

    return InstallationType.UNKNOWN


def detect_installation_context(directory: Optional[Path] = None) -> InstallationContext:
    """
    Detect complete installation context for swarm.

    This is the main entry point for installation detection. It determines:
    - Whether we're in a Python project
    - How swarm is installed (local, pipx, system)
    - Where swarm is installed

    Args:
        directory: Directory to check (defaults to current working directory)

    Returns:
        InstallationContext with all detected information
    """
    python_project = detect_python_project(directory)
    project_root = python_project.root if python_project else None

    return InstallationContext(
        installation_type=detect_installation_type(project_root),
        python_project=python_project,
        swarm_location=_get_swarm_package_location(),
    )
