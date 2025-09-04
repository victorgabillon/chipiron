#!/usr/bin/env python3
"""
Chipiron Integration Test Script

This script performs a complete end-to-end test of the Chipiron installation:
1. Creates a temporary conda environment
2. Clones the repository to a temporary directory
3. Runs the makefile installation process
4. Tests all major components (GUI, Stockfish, Syzygy tables)
5. Reports detailed results

Usage:
    python scripts/integration_test.py [--repo-url URL] [--keep-temp] [--verbose]
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from chipiron.utils.logger import chipiron_logger


class IntegrationTestResult:
    """Container for test results"""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.errors = []
        self.warnings = []

    def add_result(
        self,
        test_name: str,
        success: bool,
        message: str = "",
        details: dict[str, Any] | None = None,
    ):
        """Add a test result"""
        self.results[test_name] = {
            "success": success,
            "message": message,
            "details": details or {},
            "timestamp": time.time(),
        }
        if not success:
            self.errors.append("%s: %s" % (test_name, message))
        chipiron_logger.info("%s %s: %s", "✅" if success else "❌", test_name, message)

    def add_warning(self, test_name: str, message: str):
        """Add a warning"""
        self.warnings.append("%s: %s" % (test_name, message))
        chipiron_logger.warning("⚠️  %s: %s", test_name, message)

    def get_summary(self) -> dict[str, Any]:
        """Get test summary"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r["success"])
        duration = time.time() - self.start_time

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "duration_seconds": duration,
            "errors": self.errors,
            "warnings": self.warnings,
            "detailed_results": self.results,
        }


class ChipironIntegrationTester:
    """Main integration tester class"""

    def __init__(
        self,
        repo_url: str,
        keep_temp: bool = False,
        verbose: bool = False,
        skip_syzygy: bool = False,
        conda_env: str = None,
    ):
        self.repo_url = repo_url
        self.keep_temp = keep_temp
        self.verbose = verbose
        self.skip_syzygy = skip_syzygy
        self.use_existing_env = conda_env is not None
        self.result = IntegrationTestResult()
        self.temp_dir = None
        self.conda_env_name = conda_env  # Use provided env name if given
        self.project_dir = None
        self.python_version = None

    def get_python_version_from_pyproject(self) -> str:
        """Extract Python version requirement from pyproject.toml"""
        pyproject_path = Path(self.project_dir) / "pyproject.toml"

        if not pyproject_path.exists():
            chipiron_logger.warning(
                "pyproject.toml not found, defaulting to Python 3.12"
            )
            return "3.12"

        try:
            with open(pyproject_path, "r") as f:
                content = f.read()

            # Look for requires-python line
            match = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                python_req = match.group(1)
                # Extract version number from requirement like ">=3.12" or "~=3.12"
                version_match = re.search(r"(\d+\.\d+)", python_req)
                if version_match:
                    version = version_match.group(1)
                    chipiron_logger.info(
                        "Found Python version requirement: %s -> using %s",
                        python_req,
                        version,
                    )
                    return version

            chipiron_logger.warning(
                "Could not parse Python version from pyproject.toml, defaulting to 3.12"
            )
            return "3.12"

        except (OSError, IOError, re.error) as e:
            chipiron_logger.warning(
                "Error reading pyproject.toml: %s, defaulting to Python 3.12", e
            )
            return "3.12"

    def run_command(
        self,
        cmd: list[str],
        cwd: str = None,
        check: bool = True,
        capture_output: bool = True,
        timeout: int = 300,
        show_progress: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run a command with proper error handling and logging"""
        if self.verbose:
            chipiron_logger.info(
                "Running: %s in %s", " ".join(cmd), cwd or "current dir"
            )

        try:
            if show_progress and not capture_output:
                # For long-running commands, show live output
                chipiron_logger.info(
                    "📥 Starting download process (this may take several minutes)..."
                )
                process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                output_lines = []
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        output_lines.append(output.strip())
                        # Show progress indicators
                        if any(
                            keyword in output.lower()
                            for keyword in [
                                "downloading",
                                "download",
                                "fetching",
                                "progress",
                                "%",
                                "kb",
                                "mb",
                            ]
                        ):
                            chipiron_logger.info("📦 %s", output.strip())
                        elif self.verbose:
                            chipiron_logger.debug(output.strip())

                rc = process.poll()
                result = subprocess.CompletedProcess(
                    cmd, rc, "\n".join(output_lines), ""
                )
                if check and rc != 0:
                    raise subprocess.CalledProcessError(
                        rc, cmd, result.stdout, result.stderr
                    )
                return result
            else:
                # Regular command execution
                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    check=check,
                    capture_output=capture_output,
                    text=True,
                    timeout=timeout,
                )
                if self.verbose and result.stdout:
                    chipiron_logger.debug("STDOUT: %s", result.stdout)
                if result.stderr and self.verbose:
                    chipiron_logger.debug("STDERR: %s", result.stderr)
                return result
        except subprocess.TimeoutExpired:
            chipiron_logger.error(
                "Command timed out after %ds: %s", timeout, " ".join(cmd)
            )
            raise
        except subprocess.CalledProcessError as e:
            chipiron_logger.error("Command failed: %s", " ".join(cmd))
            chipiron_logger.error("Return code: %s", e.returncode)
            if e.stdout:
                chipiron_logger.error("STDOUT: %s", e.stdout)
            if e.stderr:
                chipiron_logger.error("STDERR: %s", e.stderr)
            raise

    def setup_temporary_environment(self):
        """Create temporary directory and setup for conda environment"""
        chipiron_logger.info("🚀 Setting up temporary environment...")

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="chipiron_test_")
        chipiron_logger.info("Created temporary directory: %s", self.temp_dir)

        # Generate unique conda environment name

        self.conda_env_name = f"chipiron_test_{uuid.uuid4().hex[:8]}"

        self.result.add_result(
            "temporary_setup",
            True,
            "Created temporary directory and environment name: %s"
            % self.conda_env_name,
        )

    def create_conda_environment(self):
        """Create conda environment with the correct Python version"""
        chipiron_logger.info("🐍 Creating conda environment...")

        # Determine Python version from pyproject.toml
        self.python_version = self.get_python_version_from_pyproject()

        try:
            # Create conda environment with the correct Python version
            self.run_command(
                [
                    "conda",
                    "create",
                    "-n",
                    self.conda_env_name,
                    f"python={self.python_version}",
                    "-y",
                ]
            )
            self.result.add_result(
                "conda_environment_creation",
                True,
                "Created conda environment: %s with Python %s"
                % (self.conda_env_name, self.python_version),
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            self.result.add_result(
                "conda_environment_creation",
                False,
                "Failed to create conda environment: %s" % str(e),
            )
            raise

    def clone_repository(self):
        """Clone the repository to temporary directory"""
        chipiron_logger.info("📦 Cloning repository...")

        self.project_dir = Path(self.temp_dir) / "chipiron"

        try:
            self.run_command(["git", "clone", self.repo_url, str(self.project_dir)])
            self.result.add_result(
                "repository_clone", True, "Cloned repository to %s" % self.project_dir
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            self.result.add_result(
                "repository_clone", False, "Failed to clone repository: %s" % str(e)
            )
            raise

    def run_makefile_installation(self):
        """Run the makefile installation process"""
        chipiron_logger.info("🔧 Running makefile installation...")
        chipiron_logger.info(
            "⏳ This includes downloading Stockfish, dependencies, and optionally Syzygy tables..."
        )
        if not self.skip_syzygy:
            chipiron_logger.info(
                "📥 Syzygy table downloads may take 10-20 minutes depending on your connection"
            )
        else:
            chipiron_logger.info("⚡ Skipping Syzygy tables for faster testing")

        # Activate conda environment and run make setup
        conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env_name}"

        try:
            if self.skip_syzygy:
                # Run installation without Syzygy tables but include Stockfish
                self.run_command(
                    [
                        "bash",
                        "-c",
                        f"{conda_activate} && cd {self.project_dir} && make init-no-syzygy",
                    ],
                    timeout=600,
                    capture_output=False,
                    show_progress=True,
                )  # 10 minutes timeout
            else:
                # Run full installation including Syzygy tables and Stockfish
                self.run_command(
                    [
                        "bash",
                        "-c",
                        f"{conda_activate} && cd {self.project_dir} && make init",
                    ],
                    timeout=1800,
                    capture_output=False,
                    show_progress=True,
                )  # 30 minutes timeout

            self.result.add_result(
                "makefile_setup",
                True,
                "Makefile setup completed successfully %s"
                % ("(without Syzygy tables)" if self.skip_syzygy else ""),
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            self.result.add_result(
                "makefile_setup", False, "Makefile setup failed: %s" % str(e)
            )
            raise

    def test_python_installation(self):
        """Test if chipiron can be imported"""
        chipiron_logger.info("🐍 Testing Python installation...")

        conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env_name}"

        try:
            _ = self.run_command(
                [
                    "bash",
                    "-c",
                    f"{conda_activate} && cd {self.project_dir} && python -c 'import chipiron; print(\"Chipiron imported successfully\")'",
                ]
            )

            self.result.add_result(
                "python_import", True, "Chipiron module imported successfully"
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            self.result.add_result(
                "python_import", False, "Failed to import chipiron: %s" % str(e)
            )

    def test_stockfish_installation(self):
        """Test if Stockfish is properly installed and accessible"""
        chipiron_logger.info("♟️  Testing Stockfish installation...")

        conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env_name}"

        try:
            # Create a temporary Python script for testing
            python_script = """
import sys, subprocess, os
try:
    from chipiron.utils.path_variables import STOCKFISH_BINARY_PATH
    print(f"Looking for Stockfish at: {STOCKFISH_BINARY_PATH}")
    if os.path.exists(STOCKFISH_BINARY_PATH):
        print("Stockfish binary found, testing execution...")
        result = subprocess.run([str(STOCKFISH_BINARY_PATH)], input="quit\\n", text=True, capture_output=True, timeout=5)
        print(f"Return code: {result.returncode}")
        if result.stdout: print(f"STDOUT: {result.stdout[:200]}")
        if result.stderr: print(f"STDERR: {result.stderr[:200]}")
        if result.returncode == 0:
            version = result.stderr.split("\\n")[0] if result.stderr else "Unknown"
            print(f"Stockfish version: {version}")
            print("Stockfish working!")
        else:
            print(f"Stockfish failed to run with return code {result.returncode}")
    else:
        print(f"Stockfish binary not found at: {STOCKFISH_BINARY_PATH}")
except Exception as e:
    print(f"Stockfish test error: {e}")
"""

            # Write to temporary file and execute
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(python_script)
                temp_script = f.name

            try:
                result = self.run_command(
                    [
                        "bash",
                        "-c",
                        f"{conda_activate} && cd {self.project_dir} && python {temp_script}",
                    ]
                )
            finally:
                # Clean up temp file
                if os.path.exists(temp_script):
                    os.unlink(temp_script)

            if "Stockfish working!" in result.stdout:
                version_line = [
                    line
                    for line in result.stdout.split("\n")
                    if "Stockfish version:" in line
                ]
                version = (
                    version_line[0].replace("Stockfish version:", "").strip()
                    if version_line
                    else "Unknown"
                )

                self.result.add_result(
                    "stockfish_installation",
                    True,
                    "Stockfish is working. Version: %s" % version,
                    {"version": version},
                )
            else:
                self.result.add_result(
                    "stockfish_installation",
                    False,
                    "Stockfish binary exists but failed to run properly",
                )

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            self.result.add_result(
                "stockfish_installation", False, "Stockfish test failed: %s" % str(e)
            )

    def _download_syzygy_tables(self):
        """Download Syzygy tables if they're missing"""
        chipiron_logger.info("📥 Downloading Syzygy tables for testing...")

        conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env_name}"

        try:
            # Run make target to download Syzygy tables - use the variable name from .env
            # Ensure clean environment to avoid SYZYGY_DESTINATION pollution
            self.run_command(
                [
                    "bash",
                    "-c",
                    f"unset SYZYGY_DESTINATION && {conda_activate} && cd {self.project_dir} && make syzygy-tables",
                ],
                timeout=1200,
                capture_output=False,
                show_progress=True,
            )  # 20 minutes timeout

            chipiron_logger.info("✅ Syzygy tables downloaded successfully")

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            chipiron_logger.warning("⚠️  Failed to download Syzygy tables: %s", str(e))
            # Don't fail the test completely, just note that tables aren't available

    def test_syzygy_tables(self):
        """Test if Syzygy tables are accessible"""
        chipiron_logger.info("📊 Testing Syzygy tables...")

        conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env_name}"

        # First check if Syzygy tables exist
        tables_exist = False
        try:
            # Quick check if tables directory exists and has files
            check_script = """
import os
from chipiron.utils.path_variables import SYZYGY_TABLES_DIR
if os.path.exists(SYZYGY_TABLES_DIR):
    files = [f for f in os.listdir(SYZYGY_TABLES_DIR) if f.endswith('.rtbw') or f.endswith('.rtbz')]
    print(f"SYZYGY_EXISTS:{len(files) > 0}")
else:
    print("SYZYGY_EXISTS:False")
"""

            # Write to temporary file and execute
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(check_script)
                temp_script = f.name

            try:
                result = self.run_command(
                    [
                        "bash",
                        "-c",
                        f"{conda_activate} && cd {self.project_dir} && python {temp_script}",
                    ]
                )
                tables_exist = "SYZYGY_EXISTS:True" in result.stdout
            finally:
                # Clean up temp file
                if os.path.exists(temp_script):
                    os.unlink(temp_script)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            chipiron_logger.warning(
                "⚠️  Could not check for existing Syzygy tables: %s", e
            )

        # Download tables if they don't exist AND we're not skipping Syzygy
        if not tables_exist and not self.skip_syzygy:
            chipiron_logger.info(
                "📥 Syzygy tables not found in fresh repository. Downloading them now..."
            )
            self._download_syzygy_tables()
        elif not tables_exist and self.skip_syzygy:
            chipiron_logger.info(
                "⚡ Syzygy tables not found, but skipping download due to --skip-syzygy flag"
            )

        try:
            # Create a temporary Python script for testing
            python_script = """
import os
try:
    # Debug current working directory and environment
    print(f'Current working directory: {os.getcwd()}')
    print(f'Python path: {os.environ.get("PYTHONPATH", "Not set")}')

    from chipiron.utils.path_variables import SYZYGY_TABLES_DIR, PROJECT_ROOT
    tables_path = SYZYGY_TABLES_DIR
    print(f'PROJECT_ROOT: {PROJECT_ROOT}')
    print(f'Expected Syzygy directory: {tables_path}')
    print(f'Absolute Syzygy path: {tables_path.absolute() if hasattr(tables_path, "absolute") else os.path.abspath(tables_path)}')

    if os.path.exists(tables_path):
        print(f'Directory exists: {tables_path}')
        all_files = os.listdir(tables_path)
        print(f'Total files found: {len(all_files)}')
        print(f'All files in directory: {all_files[:10]}...' if len(all_files) > 10 else f'All files in directory: {all_files}')

        syzygy_files = [f for f in all_files if f.endswith('.rtbw') or f.endswith('.rtbz')]
        duplicate_files = [f for f in all_files if f.endswith('.1')]
        print(f'Found {len(syzygy_files)} Syzygy table files')
        print(f'Found {len(duplicate_files)} duplicate .1 files')

        if syzygy_files:
            print('Syzygy tables available!')
            print(f'Sample files: {syzygy_files[:5]}')
        else:
            print('No Syzygy table files found')
            # Check for common alternatives
            dat_files = [f for f in all_files if f.endswith('.dat')]
            rtbw_files = [f for f in all_files if '.rtbw' in f]
            rtbz_files = [f for f in all_files if '.rtbz' in f]
            print(f'Files containing .rtbw: {len(rtbw_files)}')
            print(f'Files containing .rtbz: {len(rtbz_files)}')
            if dat_files:
                print(f'Found {len(dat_files)} .dat files instead')
    else:
        print(f'Syzygy tables directory not found: {tables_path}')
        # Check if parent directory exists and list its contents
        parent_dir = os.path.dirname(tables_path)
        if os.path.exists(parent_dir):
            print(f'Parent directory exists: {parent_dir}')
            contents = os.listdir(parent_dir)
            print(f'Parent directory contents: {contents}')
            # Check if syzygy-tables with different case exists
            for item in contents:
                if 'syzygy' in item.lower():
                    print(f'Found syzygy-related directory: {item}')
                    full_path = os.path.join(parent_dir, item)
                    if os.path.isdir(full_path):
                        subcontents = os.listdir(full_path)
                        print(f'Contents of {item}: {len(subcontents)} files')
                        syzygy_sub = [f for f in subcontents if f.endswith('.rtbw') or f.endswith('.rtbz')]
                        print(f'Syzygy files in {item}: {len(syzygy_sub)}')
        else:
            print(f'Parent directory also not found: {parent_dir}')

except Exception as e:
    import traceback
    print(f'Syzygy test error: {e}')
    print(f'Traceback: {traceback.format_exc()}')
"""

            # Write to temporary file and execute
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(python_script)
                temp_script = f.name

            try:
                result = self.run_command(
                    [
                        "bash",
                        "-c",
                        f"{conda_activate} && cd {self.project_dir} && python {temp_script}",
                    ]
                )
            finally:
                # Clean up temp file

                if os.path.exists(temp_script):
                    os.unlink(temp_script)

            if "Syzygy tables available!" in result.stdout:
                file_count_line = [
                    line
                    for line in result.stdout.split("\n")
                    if "Found" in line and "files" in line
                ]
                file_count = file_count_line[0].split()[1] if file_count_line else "0"

                self.result.add_result(
                    "syzygy_tables",
                    True,
                    "Syzygy tables are accessible with %s table files" % file_count,
                    {"file_count": file_count},
                )
            elif "Found" in result.stdout and "files" in result.stdout:
                # Even if we have some files but not the full set, consider it a partial success
                file_count_line = [
                    line
                    for line in result.stdout.split("\n")
                    if "Found" in line and "files" in line
                ]
                file_count = (
                    int(file_count_line[0].split()[1]) if file_count_line else 0
                )

                if file_count > 0:
                    self.result.add_warning(
                        "syzygy_tables",
                        "Partial Syzygy tables available (%s files) - some downloads may have failed"
                        % file_count,
                    )
                    self.result.add_result(
                        "syzygy_tables",
                        True,
                        "Partial Syzygy tables available with %s table files"
                        % file_count,
                        {"file_count": file_count, "partial": True},
                    )
                else:
                    # Extract debug output for analysis
                    debug_lines = result.stdout.split("\n")
                    debug_info = (
                        "\n".join(debug_lines[:20]) if debug_lines else "No output"
                    )
                    self.result.add_result(
                        "syzygy_tables",
                        False,
                        "Syzygy tables directory exists but no table files found. Debug: %s"
                        % debug_info,
                    )
            else:
                # Extract debug output for analysis
                debug_lines = result.stdout.split("\n")
                debug_info = "\n".join(debug_lines[:20]) if debug_lines else "No output"
                self.result.add_result(
                    "syzygy_tables",
                    False,
                    "Syzygy tables directory exists but no table files found. Debug: %s"
                    % debug_info,
                )

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            self.result.add_result(
                "syzygy_tables", False, "Syzygy tables test failed: %s" % str(e)
            )

    def test_gui_functionality(self):
        """Test GUI functionality (PNG display capability)"""
        chipiron_logger.info("🖼️  Testing GUI functionality...")

        conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env_name}"

        try:
            # Create a temporary Python script for testing
            python_script = """
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")  # Use non-interactive backend
    import numpy as np
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    fig.savefig("/tmp/test_plot.png")
    plt.close()
    print("GUI dependencies working!")
except Exception as e:
    print(f"GUI test failed: {e}")
"""

            # Write to temporary file and execute
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(python_script)
                temp_script = f.name

            try:
                result = self.run_command(
                    [
                        "bash",
                        "-c",
                        f"{conda_activate} && cd {self.project_dir} && python {temp_script}",
                    ]
                )
            finally:
                # Clean up temp file

                if os.path.exists(temp_script):
                    os.unlink(temp_script)

            if "GUI dependencies working!" in result.stdout:
                self.result.add_result(
                    "gui_functionality",
                    True,
                    "GUI dependencies are working and can generate PNG files",
                )
            else:
                self.result.add_result(
                    "gui_functionality", False, "GUI dependencies test failed"
                )

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            self.result.add_result(
                "gui_functionality", False, "GUI functionality test failed: %s" % str(e)
            )

    def test_basic_game_functionality(self):
        """Test basic game functionality"""
        chipiron_logger.info("🎮 Testing basic game functionality...")

        conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env_name}"

        try:
            # Create a temporary Python script for testing
            python_script = """
try:
    from chipiron.environments.chess_env.board import create_board_chi
    print("Attempting to create chess board...")
    board = create_board_chi()
    print(f"Initial board created: {len(str(board))} characters")
    print("Basic game functionality working!")
except Exception as e:
    import traceback
    print(f"Game test failed: {e}")
    print(f"Traceback: {traceback.format_exc()}")
"""

            # Write to temporary file and execute
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(python_script)
                temp_script = f.name

            try:
                result = self.run_command(
                    [
                        "bash",
                        "-c",
                        f"{conda_activate} && cd {self.project_dir} && python {temp_script}",
                    ]
                )
            finally:
                # Clean up temp file

                if os.path.exists(temp_script):
                    os.unlink(temp_script)

            if "Basic game functionality working!" in result.stdout:
                self.result.add_result(
                    "basic_game_functionality",
                    True,
                    "Basic chess game functionality is working",
                )
            else:
                self.result.add_result(
                    "basic_game_functionality",
                    False,
                    "Basic game functionality test failed",
                )

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            self.result.add_result(
                "basic_game_functionality",
                False,
                "Basic game functionality test failed: %s" % str(e),
            )

    def test_syzygy_dtz_functionality(self):
        """Test that SyzygyChiTable.dtz(board) works and returns an int for a known tablebase position."""
        chipiron_logger.info("🧪 Testing Syzygy DTZ functionality...")
        conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env_name}"
        try:
            # Create a temporary Python script for testing SyzygyChiTable.dtz
            python_script = """
import sys
from chipiron.players.boardevaluators.table_base.syzygy_python import SyzygyChiTable
from chipiron.environments.chess_env.board import create_board_chi
from chipiron.utils.path_variables import SYZYGY_TABLES_DIR

# KQ vs K position (should be in tablebase)
fen = "8/8/8/8/8/8/5K2/6Qk w - - 0 1"
board = create_board_chi(fen)
table = SyzygyChiTable(SYZYGY_TABLES_DIR)
try:
    dtz = table.dtz(board)
    print(f"DTZ value: {dtz}")
    print("Syzygy DTZ test passed!")
except Exception as e:
    print(f"Syzygy DTZ test failed: {e}")
    sys.exit(1)
"""
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(python_script)
                temp_script = f.name
            try:
                result = self.run_command(
                    [
                        "bash",
                        "-c",
                        f"{conda_activate} && cd {self.project_dir} && python {temp_script}",
                    ]
                )
            finally:
                if os.path.exists(temp_script):
                    os.remove(temp_script)
            if "Syzygy DTZ test passed!" in result.stdout:
                self.result.add_result(
                    "syzygy_dtz_functionality",
                    True,
                    "SyzygyChiTable.dtz(board) returned a valid DTZ value",
                )
            else:
                self.result.add_result(
                    "syzygy_dtz_functionality",
                    False,
                    "Syzygy DTZ test failed: %s" % result.stdout.strip(),
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            self.result.add_result(
                "syzygy_dtz_functionality", False, "Syzygy DTZ test failed: %s" % str(e)
            )

    def test_stockfish_player_functionality(self):
        """Test that StockfishPlayer can select a move for a simple position."""
        chipiron_logger.info("🧪 Testing StockfishPlayer functionality...")
        conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env_name}"
        try:
            python_script = """
import sys
from chipiron.players.move_selector.stockfish import StockfishPlayer
from chipiron.environments.chess_env.board import create_board_chi
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes

# Simple position: starting position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
board = create_board_chi(fen)
player = StockfishPlayer(type=MoveSelectorTypes.Stockfish)
try:
    move_rec = player.select_move(board, move_seed=42)
    print(f"Stockfish selected move: {move_rec.move}")
    print("StockfishPlayer test passed!")
except Exception as e:
    print(f"StockfishPlayer test failed: {e}")
    sys.exit(1)
"""
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(python_script)
                temp_script = f.name
            try:
                result = self.run_command(
                    [
                        "bash",
                        "-c",
                        f"{conda_activate} && cd {self.project_dir} && python {temp_script}",
                    ]
                )
            finally:
                if os.path.exists(temp_script):
                    os.remove(temp_script)
            if "StockfishPlayer test passed!" in result.stdout:
                self.result.add_result(
                    "stockfish_player_functionality",
                    True,
                    "StockfishPlayer.select_move(board, move_seed) returned a move",
                )
            else:
                self.result.add_result(
                    "stockfish_player_functionality",
                    False,
                    "StockfishPlayer test failed: %s" % result.stdout.strip(),
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            self.result.add_result(
                "stockfish_player_functionality",
                False,
                "StockfishPlayer test failed: %s" % str(e),
            )

    def cleanup(self):
        """Clean up temporary resources"""
        chipiron_logger.info("🧹 Cleaning up...")

        # Remove conda environment
        if self.conda_env_name:
            try:
                self.run_command(
                    ["conda", "env", "remove", "-n", self.conda_env_name, "-y"]
                )
                chipiron_logger.info(
                    "Removed conda environment: %s", self.conda_env_name
                )
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                OSError,
            ) as e:
                chipiron_logger.warning("Failed to remove conda environment: %s", e)

        # Remove temporary directory
        if self.temp_dir and not self.keep_temp:
            try:
                shutil.rmtree(self.temp_dir)
                chipiron_logger.info("Removed temporary directory: %s", self.temp_dir)
            except OSError as e:
                chipiron_logger.warning("Failed to remove temporary directory: %s", e)
        elif self.temp_dir and self.keep_temp:
            chipiron_logger.info("Keeping temporary directory: %s", self.temp_dir)

    def run_full_test_suite(self):
        """Run the complete integration test suite"""
        chipiron_logger.info("🔬 Starting Chipiron Integration Test Suite")
        try:
            # Setup phase
            self.setup_temporary_environment()
            self.clone_repository()
            self.create_conda_environment()
            # Installation phase
            self.run_makefile_installation()
            # Testing phase
            self.test_python_installation()
            self.test_stockfish_installation()
            if not self.skip_syzygy:
                self.test_syzygy_tables()
                self.test_syzygy_dtz_functionality()
            else:
                self.test_syzygy_tables()
            self.test_stockfish_player_functionality()
            self.test_gui_functionality()
            self.test_basic_game_functionality()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            chipiron_logger.error("Integration test suite failed: %s", e)
        finally:
            # Always clean up
            self.cleanup()
        # Generate final report
        return self.generate_report()

    def generate_report(self) -> dict[Any, Any]:
        """Generate final test report"""
        summary = self.result.get_summary()

        chipiron_logger.info("📋 INTEGRATION TEST REPORT")
        chipiron_logger.info("=" * 50)
        chipiron_logger.info("Total Tests: %s", summary["total_tests"])
        chipiron_logger.info("Successful: %s", summary["successful_tests"])
        chipiron_logger.info("Failed: %s", summary["failed_tests"])
        chipiron_logger.info("Success Rate: %.1f%%", summary["success_rate"] * 100)
        chipiron_logger.info("Duration: %.1f seconds", summary["duration_seconds"])

        if summary["errors"]:
            chipiron_logger.error("❌ ERRORS:")
            for error in summary["errors"]:
                chipiron_logger.error("  - %s", error)

        if summary["warnings"]:
            chipiron_logger.warning("⚠️  WARNINGS:")
            for warning in summary["warnings"]:
                chipiron_logger.warning("  - %s", warning)

        # Save detailed report to file
        report_file = "integration_test_report.json"
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2)
        chipiron_logger.info("📄 Detailed report saved to: %s", report_file)

        return summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Chipiron Integration Test Suite")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/victorgabillon/chipiron.git",
        help="Git repository URL to test",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after test completion",
    )
    parser.add_argument(
        "--skip-syzygy",
        action="store_true",
        help="Skip Syzygy table downloads for faster testing",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        chipiron_logger.setLevel(logging.DEBUG)

    # Check if conda is available
    try:
        subprocess.run(["conda", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        chipiron_logger.error("❌ Conda is not available. Please install conda first.")
        sys.exit(1)

    # Check if git is available
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        chipiron_logger.error("❌ Git is not available. Please install git first.")
        sys.exit(1)

    # Run the integration test
    tester = ChipironIntegrationTester(
        repo_url=args.repo_url,
        keep_temp=args.keep_temp,
        verbose=args.verbose,
        skip_syzygy=args.skip_syzygy,
    )

    summary = tester.run_full_test_suite()

    # Exit with appropriate code
    if summary["failed_tests"] == 0:
        chipiron_logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        chipiron_logger.error("💥 %s test(s) failed!", summary["failed_tests"])
        sys.exit(1)


if __name__ == "__main__":
    main()
