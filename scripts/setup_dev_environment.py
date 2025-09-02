#!/usr/bin/env python3
"""
Development Environment Setup Script for Plansheet Scanner

This script automates the setup of a modern Python development environment
with all necessary tools, pre-commit hooks, and configurations.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Optional


class DevEnvironmentSetup:
    """Manages the development environment setup process."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.venv_path = self.project_root / "venv"
        self.is_windows = platform.system() == "Windows"
        self.python_cmd = "python" if self.is_windows else "python3"
        self.pip_cmd = "pip" if self.is_windows else "pip3"
        
    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> bool:
        """Run a shell command and return success status."""
        try:
            print(f"Running: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {' '.join(command)}")
            print(f"Error: {e}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            return False
    
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        try:
            result = subprocess.run(
                [self.python_cmd, "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.strip()
            print(f"Found Python: {version}")
            
            # Extract version number
            version_parts = version.split()[1].split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            if major < 3 or (major == 3 and minor < 8):
                print("Error: Python 3.8 or higher is required")
                return False
            
            return True
        except Exception as e:
            print(f"Error checking Python version: {e}")
            return False
    
    def create_virtual_environment(self) -> bool:
        """Create a virtual environment if it doesn't exist."""
        if self.venv_path.exists():
            print(f"Virtual environment already exists at {self.venv_path}")
            return True
        
        print("Creating virtual environment...")
        return self.run_command([self.python_cmd, "-m", "venv", str(self.venv_path)])
    
    def activate_virtual_environment(self) -> bool:
        """Activate the virtual environment and update PATH."""
        if self.is_windows:
            activate_script = self.venv_path / "Scripts" / "activate.bat"
            if not activate_script.exists():
                print("Error: Virtual environment activation script not found")
                return False
        else:
            activate_script = self.venv_path / "bin" / "activate"
            if not activate_script.exists():
                print("Error: Virtual environment activation script not found")
                return False
        
        # Update PATH for current process
        if self.is_windows:
            scripts_path = self.venv_path / "Scripts"
        else:
            scripts_path = self.venv_path / "bin"
        
        os.environ["PATH"] = f"{scripts_path}{os.pathsep}{os.environ['PATH']}"
        
        # Update pip and setuptools
        print("Updating pip and setuptools...")
        self.run_command([self.pip_cmd, "install", "--upgrade", "pip", "setuptools", "wheel"])
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install project dependencies."""
        print("Installing core dependencies...")
        if not self.run_command([self.pip_cmd, "install", "-r", "requirements.txt"]):
            return False
        
        print("Installing development dependencies...")
        if not self.run_command([self.pip_cmd, "install", "-r", "requirements-dev.txt"]):
            return False
        
        return True
    
    def install_pre_commit_hooks(self) -> bool:
        """Install pre-commit hooks."""
        print("Installing pre-commit hooks...")
        return self.run_command([self.pip_cmd, "run", "pre-commit", "install"])
    
    def run_initial_checks(self) -> bool:
        """Run initial code quality checks."""
        print("Running initial code quality checks...")
        
        checks = [
            (["black", "--check", "src/", "tests/"], "Black formatting check"),
            (["isort", "--check-only", "src/", "tests/"], "Import sorting check"),
            (["flake8", "src/", "tests/"], "Linting check"),
        ]
        
        all_passed = True
        for command, description in checks:
            print(f"\nRunning {description}...")
            if not self.run_command(command):
                print(f"Warning: {description} failed")
                all_passed = False
        
        return all_passed
    
    def create_development_config(self) -> bool:
        """Create development configuration files."""
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Create .env.example
        env_example = self.project_root / ".env.example"
        if not env_example.exists():
            env_content = """# Development Environment Configuration
# Copy this file to .env and update values as needed

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/plansheet_scanner

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# ML Model Configuration
MODEL_CACHE_DIR=models/
TRAINING_DATA_DIR=training_data/

# External Services
OPENAI_API_KEY=your_openai_api_key_here
TESSERACT_PATH=/usr/bin/tesseract

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/plansheet_scanner.log

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=localhost,127.0.0.1

# Testing
TEST_DATABASE_URL=postgresql://user:password@localhost:5432/plansheet_scanner_test
COVERAGE_THRESHOLD=80
"""
            env_example.write_text(env_content)
            print("Created .env.example file")
        
        return True
    
    def setup_git_hooks(self) -> bool:
        """Set up additional Git hooks for development."""
        git_hooks_dir = self.project_root / ".git" / "hooks"
        
        # Create pre-push hook for running tests
        pre_push_hook = git_hooks_dir / "pre-push"
        if not pre_push_hook.exists():
            hook_content = """#!/bin/sh
# Pre-push hook to run tests before pushing

echo "Running tests before push..."
cd "$(dirname "$0")/../.."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Run tests
python -m pytest tests/ -v --tb=short

# Exit with test result
exit $?
"""
            pre_push_hook.write_text(hook_content)
            os.chmod(pre_push_hook, 0o755)
            print("Created pre-push Git hook")
        
        return True
    
    def print_setup_complete(self):
        """Print setup completion message with next steps."""
        print("\n" + "="*60)
        print("🎉 Development Environment Setup Complete!")
        print("="*60)
        
        print("\n📋 Next Steps:")
        print("1. Activate your virtual environment:")
        if self.is_windows:
            print(f"   {self.venv_path}\\Scripts\\activate")
        else:
            print(f"   source {self.venv_path}/bin/activate")
        
        print("\n2. Run tests to verify installation:")
        print("   python -m pytest tests/ -v")
        
        print("\n3. Format your code:")
        print("   black src/ tests/")
        print("   isort src/ tests/")
        
        print("\n4. Run pre-commit hooks:")
        print("   pre-commit run --all-files")
        
        print("\n5. Start developing!")
        print("   # Your code is ready for development")
        
        print("\n🔧 Available Commands:")
        print("   python -m pytest          # Run tests")
        print("   black src/ tests/         # Format code")
        print("   isort src/ tests/         # Sort imports")
        print("   flake8 src/ tests/        # Lint code")
        print("   mypy src/                 # Type check")
        print("   pre-commit run --all-files # Run all quality checks")
        
        print("\n📚 Documentation:")
        print("   - README.md: Project overview and setup")
        print("   - CONTRIBUTING.md: Development guidelines")
        print("   - pyproject.toml: Project configuration")
        
        print("\n🚀 Happy Coding!")
        print("="*60)
    
    def setup(self) -> bool:
        """Run the complete setup process."""
        print("🚀 Setting up Plansheet Scanner Development Environment")
        print("="*60)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Activating virtual environment", self.activate_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Installing pre-commit hooks", self.install_pre_commit_hooks),
            ("Creating development config", self.create_development_config),
            ("Setting up Git hooks", self.setup_git_hooks),
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"❌ {step_name} failed")
                return False
            print(f"✅ {step_name} completed")
        
        print("\n🔍 Running initial quality checks...")
        self.run_initial_checks()
        
        self.print_setup_complete()
        return True


def main():
    """Main entry point for the setup script."""
    setup = DevEnvironmentSetup()
    
    try:
        success = setup.setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
