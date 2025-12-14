"""
Tests for installation detection module.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from swarm_orchestrator.installation import (
    InstallationType,
    PythonProject,
    InstallationContext,
    detect_python_project,
    detect_installation_type,
    detect_installation_context,
    _is_in_pipx,
    _is_in_project_venv,
    _is_editable_install,
    _get_swarm_package_location,
)


class TestPythonProject:
    """Tests for PythonProject dataclass."""

    def test_is_valid_with_pyproject(self, tmp_path):
        """Project with pyproject.toml is valid."""
        project = PythonProject(
            root=tmp_path,
            has_pyproject_toml=True,
        )
        assert project.is_valid

    def test_is_valid_with_setup_py(self, tmp_path):
        """Project with setup.py is valid."""
        project = PythonProject(
            root=tmp_path,
            has_setup_py=True,
        )
        assert project.is_valid

    def test_is_valid_with_requirements(self, tmp_path):
        """Project with requirements.txt is valid."""
        project = PythonProject(
            root=tmp_path,
            has_requirements_txt=True,
        )
        assert project.is_valid

    def test_is_valid_with_setup_cfg(self, tmp_path):
        """Project with setup.cfg is valid."""
        project = PythonProject(
            root=tmp_path,
            has_setup_cfg=True,
        )
        assert project.is_valid

    def test_is_invalid_without_markers(self, tmp_path):
        """Project without any markers is invalid."""
        project = PythonProject(root=tmp_path)
        assert not project.is_valid


class TestInstallationContext:
    """Tests for InstallationContext dataclass."""

    def test_is_local_install(self, tmp_path):
        """LOCAL installation type is correctly identified."""
        ctx = InstallationContext(
            installation_type=InstallationType.LOCAL,
            python_project=PythonProject(root=tmp_path, has_pyproject_toml=True),
            swarm_location=tmp_path / ".venv" / "lib",
        )
        assert ctx.is_local_install
        assert not ctx.is_global_install

    def test_is_global_install_pipx(self, tmp_path):
        """PIPX installation is global."""
        ctx = InstallationContext(
            installation_type=InstallationType.PIPX,
            python_project=None,
            swarm_location=Path.home() / ".local" / "pipx",
        )
        assert ctx.is_global_install
        assert not ctx.is_local_install

    def test_is_global_install_system(self, tmp_path):
        """SYSTEM installation is global."""
        ctx = InstallationContext(
            installation_type=InstallationType.SYSTEM,
            python_project=None,
            swarm_location=Path("/usr/lib/python3/site-packages"),
        )
        assert ctx.is_global_install
        assert not ctx.is_local_install

    def test_in_python_project_true(self, tmp_path):
        """in_python_project is True when project exists and is valid."""
        ctx = InstallationContext(
            installation_type=InstallationType.LOCAL,
            python_project=PythonProject(root=tmp_path, has_pyproject_toml=True),
            swarm_location=None,
        )
        assert ctx.in_python_project

    def test_in_python_project_false_no_project(self):
        """in_python_project is False when no project."""
        ctx = InstallationContext(
            installation_type=InstallationType.SYSTEM,
            python_project=None,
            swarm_location=None,
        )
        assert not ctx.in_python_project

    def test_in_python_project_false_invalid(self, tmp_path):
        """in_python_project is False when project is invalid."""
        ctx = InstallationContext(
            installation_type=InstallationType.SYSTEM,
            python_project=PythonProject(root=tmp_path),  # No markers
            swarm_location=None,
        )
        assert not ctx.in_python_project


class TestDetectPythonProject:
    """Tests for detect_python_project function."""

    def test_detects_pyproject_toml(self, tmp_path):
        """Detects project with pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'")

        project = detect_python_project(tmp_path)

        assert project is not None
        assert project.has_pyproject_toml
        assert project.root == tmp_path.resolve()

    def test_detects_setup_py(self, tmp_path):
        """Detects project with setup.py."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")

        project = detect_python_project(tmp_path)

        assert project is not None
        assert project.has_setup_py

    def test_detects_requirements_txt(self, tmp_path):
        """Detects project with requirements.txt."""
        (tmp_path / "requirements.txt").write_text("requests>=2.0")

        project = detect_python_project(tmp_path)

        assert project is not None
        assert project.has_requirements_txt

    def test_detects_setup_cfg(self, tmp_path):
        """Detects project with setup.cfg."""
        (tmp_path / "setup.cfg").write_text("[metadata]\nname = test")

        project = detect_python_project(tmp_path)

        assert project is not None
        assert project.has_setup_cfg

    def test_detects_multiple_markers(self, tmp_path):
        """Detects project with multiple markers."""
        (tmp_path / "pyproject.toml").write_text("[project]")
        (tmp_path / "requirements.txt").write_text("")

        project = detect_python_project(tmp_path)

        assert project is not None
        assert project.has_pyproject_toml
        assert project.has_requirements_txt

    def test_returns_none_for_empty_directory(self, tmp_path):
        """Returns None for directory without project markers."""
        project = detect_python_project(tmp_path)
        assert project is None

    def test_returns_none_for_nonexistent_directory(self, tmp_path):
        """Returns None for non-existent directory."""
        project = detect_python_project(tmp_path / "nonexistent")
        assert project is None

    def test_uses_cwd_by_default(self, tmp_path, monkeypatch):
        """Uses current directory when no argument given."""
        (tmp_path / "pyproject.toml").write_text("[project]")
        monkeypatch.chdir(tmp_path)

        project = detect_python_project()

        assert project is not None
        assert project.has_pyproject_toml


class TestIsPipx:
    """Tests for _is_in_pipx helper."""

    def test_detects_pipx_location(self):
        """Detects pipx installation path."""
        with patch(
            "swarm_orchestrator.installation._get_swarm_package_location",
            return_value=Path.home() / ".local" / "pipx" / "venvs" / "swarm" / "lib",
        ):
            assert _is_in_pipx()

    def test_rejects_non_pipx_location(self):
        """Rejects non-pipx installation path."""
        with patch(
            "swarm_orchestrator.installation._get_swarm_package_location",
            return_value=Path("/usr/lib/python3/site-packages"),
        ):
            assert not _is_in_pipx()

    def test_handles_none_location(self):
        """Handles case where package location is None."""
        with patch(
            "swarm_orchestrator.installation._get_swarm_package_location",
            return_value=None,
        ):
            assert not _is_in_pipx()


class TestIsInProjectVenv:
    """Tests for _is_in_project_venv helper."""

    def test_detects_package_in_project(self, tmp_path):
        """Detects package installed in project venv."""
        venv_path = tmp_path / ".venv" / "lib" / "python3.11" / "site-packages" / "swarm"
        with patch(
            "swarm_orchestrator.installation._get_swarm_package_location",
            return_value=venv_path,
        ):
            assert _is_in_project_venv(tmp_path)

    def test_rejects_package_outside_project(self, tmp_path):
        """Rejects package installed outside project."""
        with patch(
            "swarm_orchestrator.installation._get_swarm_package_location",
            return_value=Path("/usr/lib/python3/site-packages/swarm"),
        ):
            with patch("sys.prefix", str(Path("/usr"))):
                assert not _is_in_project_venv(tmp_path)

    def test_handles_none_location(self, tmp_path):
        """Handles case where package location is None."""
        with patch(
            "swarm_orchestrator.installation._get_swarm_package_location",
            return_value=None,
        ):
            assert not _is_in_project_venv(tmp_path)


class TestIsEditableInstall:
    """Tests for _is_editable_install helper."""

    def test_detects_editable_install(self):
        """Detects editable/development install."""
        with patch(
            "swarm_orchestrator.installation._get_swarm_package_location",
            return_value=Path("/home/user/project/src/swarm_orchestrator"),
        ):
            assert _is_editable_install()

    def test_rejects_regular_install(self):
        """Rejects regular (non-editable) install."""
        with patch(
            "swarm_orchestrator.installation._get_swarm_package_location",
            return_value=Path("/usr/lib/python3/site-packages/swarm_orchestrator"),
        ):
            assert not _is_editable_install()

    def test_handles_none_location(self):
        """Handles case where package location is None."""
        with patch(
            "swarm_orchestrator.installation._get_swarm_package_location",
            return_value=None,
        ):
            assert not _is_editable_install()


class TestDetectInstallationType:
    """Tests for detect_installation_type function."""

    def test_detects_pipx(self, tmp_path):
        """Detects pipx installation."""
        with patch("swarm_orchestrator.installation._is_in_pipx", return_value=True):
            assert detect_installation_type(tmp_path) == InstallationType.PIPX

    def test_detects_local_in_venv(self, tmp_path):
        """Detects local installation in project venv."""
        with patch("swarm_orchestrator.installation._is_in_pipx", return_value=False):
            with patch("swarm_orchestrator.installation._is_in_project_venv", return_value=True):
                assert detect_installation_type(tmp_path) == InstallationType.LOCAL

    def test_detects_local_editable(self, tmp_path):
        """Detects local editable installation."""
        with patch("swarm_orchestrator.installation._is_in_pipx", return_value=False):
            with patch("swarm_orchestrator.installation._is_in_project_venv", return_value=False):
                with patch("swarm_orchestrator.installation._is_editable_install", return_value=True):
                    assert detect_installation_type(tmp_path) == InstallationType.LOCAL

    def test_detects_system(self, tmp_path):
        """Detects system installation."""
        with patch("swarm_orchestrator.installation._is_in_pipx", return_value=False):
            with patch("swarm_orchestrator.installation._is_in_project_venv", return_value=False):
                with patch("swarm_orchestrator.installation._is_editable_install", return_value=False):
                    with patch(
                        "swarm_orchestrator.installation._get_swarm_package_location",
                        return_value=Path("/usr/lib/python3/site-packages/swarm"),
                    ):
                        assert detect_installation_type(tmp_path) == InstallationType.SYSTEM

    def test_returns_unknown_when_undetermined(self, tmp_path):
        """Returns UNKNOWN when installation type cannot be determined."""
        with patch("swarm_orchestrator.installation._is_in_pipx", return_value=False):
            with patch("swarm_orchestrator.installation._is_in_project_venv", return_value=False):
                with patch("swarm_orchestrator.installation._is_editable_install", return_value=False):
                    with patch(
                        "swarm_orchestrator.installation._get_swarm_package_location",
                        return_value=Path("/some/random/path"),
                    ):
                        assert detect_installation_type(tmp_path) == InstallationType.UNKNOWN


class TestDetectInstallationContext:
    """Tests for detect_installation_context function."""

    def test_detects_local_in_python_project(self, tmp_path):
        """Detects local install in Python project."""
        (tmp_path / "pyproject.toml").write_text("[project]")

        with patch("swarm_orchestrator.installation._is_in_pipx", return_value=False):
            with patch("swarm_orchestrator.installation._is_in_project_venv", return_value=True):
                ctx = detect_installation_context(tmp_path)

        assert ctx.installation_type == InstallationType.LOCAL
        assert ctx.in_python_project
        assert ctx.is_local_install

    def test_detects_pipx_outside_project(self, tmp_path):
        """Detects pipx install outside Python project."""
        # Empty directory, no project markers

        with patch("swarm_orchestrator.installation._is_in_pipx", return_value=True):
            with patch(
                "swarm_orchestrator.installation._get_swarm_package_location",
                return_value=Path.home() / ".local" / "pipx" / "venvs" / "swarm",
            ):
                ctx = detect_installation_context(tmp_path)

        assert ctx.installation_type == InstallationType.PIPX
        assert not ctx.in_python_project
        assert ctx.is_global_install

    def test_uses_cwd_by_default(self, tmp_path, monkeypatch):
        """Uses current directory when no argument given."""
        (tmp_path / "pyproject.toml").write_text("[project]")
        monkeypatch.chdir(tmp_path)

        with patch("swarm_orchestrator.installation._is_in_pipx", return_value=False):
            with patch("swarm_orchestrator.installation._is_in_project_venv", return_value=True):
                ctx = detect_installation_context()

        assert ctx.in_python_project

    def test_includes_swarm_location(self, tmp_path):
        """Includes swarm package location in context."""
        location = Path("/some/path/to/swarm")

        with patch("swarm_orchestrator.installation._is_in_pipx", return_value=False):
            with patch("swarm_orchestrator.installation._is_in_project_venv", return_value=False):
                with patch("swarm_orchestrator.installation._is_editable_install", return_value=False):
                    with patch(
                        "swarm_orchestrator.installation._get_swarm_package_location",
                        return_value=location,
                    ):
                        ctx = detect_installation_context(tmp_path)

        assert ctx.swarm_location == location
