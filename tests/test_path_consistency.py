#!/usr/bin/env python3
"""
Test that all path configurations are consistent across the project.
This test ensures that Python, Makefile, and Dockerfile use the same paths.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from chipiron.utils.path_variables import (
    EXTERNAL_DATA_DIR,
    GUI_DIR,
    LICHESS_PGN_DIR,
    PROJECT_ROOT,
    STOCKFISH_DIR,
    SYZYGY_TABLES_DIR,
)


def load_env_file():
    """Load environment variables from .env file."""
    env_vars = {}
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key] = value
    return env_vars


class TestPathConsistency:
    """Test suite for path configuration consistency."""

    def test_env_file_exists(self):
        """Test that .env file exists."""
        env_file = project_root / ".env"
        assert env_file.exists(), "The .env configuration file should exist"

    def test_required_env_vars_present(self):
        """Test that all required environment variables are defined in .env."""
        env_vars = load_env_file()
        required_vars = [
            "EXTERNAL_DATA_DIR",
            "LICHESS_PGN_DIR",
            "SYZYGY_TABLES_DIR",
            "STOCKFISH_DIR",
            "GUI_DIR",
            "LICHESS_PGN_FILE",
            "STOCKFISH_BINARY_PATH",
            "STOCKFISH_VERSION",
            "SYZYGY_SOURCE",
            "STOCKFISH_URL",
            "DATA_SOURCE",
        ]

        for var in required_vars:
            assert var in env_vars, (
                f"Required environment variable {var} not found in .env"
            )
            assert env_vars[var].strip(), f"Environment variable {var} is empty in .env"

    def test_python_paths_match_env(self):
        """Test that Python path variables match .env definitions."""
        env_vars = load_env_file()

        # Test directory paths
        path_mappings = {
            "EXTERNAL_DATA_DIR": EXTERNAL_DATA_DIR,
            "LICHESS_PGN_DIR": LICHESS_PGN_DIR,
            "SYZYGY_TABLES_DIR": SYZYGY_TABLES_DIR,
            "STOCKFISH_DIR": STOCKFISH_DIR,
            "GUI_DIR": GUI_DIR,
        }

        for env_key, python_path in path_mappings.items():
            if env_key in env_vars:
                expected_path = PROJECT_ROOT / env_vars[env_key]
                assert python_path == expected_path, (
                    f"Python path {env_key} doesn't match .env: "
                    f"Python={python_path}, .env={expected_path}"
                )

    def test_project_root_is_correct(self):
        """Test that PROJECT_ROOT points to the actual project root."""
        # Project root should contain these key files
        key_files = ["pyproject.toml", "Makefile", ".env", "README.md"]
        for file_name in key_files:
            file_path = PROJECT_ROOT / file_name
            assert file_path.exists(), (
                f"Key file {file_name} not found at PROJECT_ROOT: {PROJECT_ROOT}"
            )

    def test_critical_directories_exist(self):
        """Test that critical directories exist."""
        critical_dirs = [PROJECT_ROOT, EXTERNAL_DATA_DIR]

        for dir_path in critical_dirs:
            assert dir_path.exists(), f"Critical directory does not exist: {dir_path}"
            assert dir_path.is_dir(), f"Path exists but is not a directory: {dir_path}"

    def test_makefile_uses_env_vars(self):
        """Test that Makefile includes .env file and uses environment variables."""
        makefile_path = PROJECT_ROOT / "Makefile"
        assert makefile_path.exists(), "Makefile should exist"

        with open(makefile_path, "r", encoding="utf-8") as f:
            makefile_content = f.read()

        # Check that Makefile includes .env
        assert "include .env" in makefile_content, "Makefile should include .env file"
        assert "export" in makefile_content, (
            "Makefile should export environment variables"
        )

        # Check that Makefile uses environment variables
        env_var_usage = [
            "$(EXTERNAL_DATA_DIR)",
            "$(SYZYGY_TABLES_DIR)",
            "$(STOCKFISH_DIR)",
            "${SYZYGY_SOURCE}",
        ]
        for var_usage in env_var_usage:
            assert var_usage in makefile_content, f"Makefile should use {var_usage}"

    def test_dockerfile_has_env_vars(self):
        """Test that Dockerfile defines environment variables matching .env."""
        dockerfile_path = PROJECT_ROOT / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile should exist"

        with open(dockerfile_path, "r", encoding="utf-8") as f:
            dockerfile_content = f.read()

        env_vars = load_env_file()

        # Check that Dockerfile has ENV statements for key variables
        key_env_vars = [
            "EXTERNAL_DATA_DIR",
            "SYZYGY_TABLES_DIR",
            "STOCKFISH_DIR",
            "GUI_DIR",
        ]
        for env_var in key_env_vars:
            assert f"ENV {env_var}=" in dockerfile_content, (
                f"Dockerfile should define ENV {env_var}"
            )

    def test_env_values_are_relative_paths(self):
        """Test that .env path values are relative (not absolute)."""
        env_vars = load_env_file()

        path_vars = [
            "EXTERNAL_DATA_DIR",
            "LICHESS_PGN_DIR",
            "SYZYGY_TABLES_DIR",
            "STOCKFISH_DIR",
            "GUI_DIR",
            "LICHESS_PGN_FILE",
            "STOCKFISH_BINARY_PATH",
        ]

        for var in path_vars:
            if var in env_vars:
                value = env_vars[var]
                assert not os.path.isabs(value), (
                    f"Environment variable {var} should be a relative path, got: {value}"
                )

    def test_urls_are_valid_format(self):
        """Test that URL environment variables have valid format."""
        env_vars = load_env_file()

        url_vars = ["SYZYGY_SOURCE", "STOCKFISH_URL", "DATA_SOURCE"]

        for var in url_vars:
            if var in env_vars:
                value = env_vars[var]
                assert value.startswith(("http://", "https://")), (
                    f"URL variable {var} should start with http:// or https://, got: {value}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
