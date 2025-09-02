#!/usr/bin/env python3
"""
Test Setup Script for Plansheet Scanner

This script verifies that the development environment is properly configured
and all necessary tools are available.
"""

import sys
import importlib
from pathlib import Path
from typing import List, Tuple


def test_python_version() -> Tuple[bool, str]:
    """Test Python version compatibility."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} ✗ (3.8+ required)"


def test_imports() -> List[Tuple[str, bool, str]]:
    """Test critical package imports."""
    packages = [
        # Core ML packages
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "scikit-learn"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        
        # Development tools
        ("pytest", "pytest"),
        ("black", "Black"),
        ("isort", "isort"),
        ("flake8", "flake8"),
        ("mypy", "mypy"),
        ("pre_commit", "pre-commit"),
        
        # Additional tools
        ("streamlit", "Streamlit"),
        ("fastapi", "FastAPI"),
        ("plotly", "Plotly"),
    ]
    
    results = []
    for package, name in packages:
        try:
            importlib.import_module(package)
            results.append((name, True, f"{name} ✓"))
        except ImportError:
            results.append((name, False, f"{name} ✗ (not installed)"))
    
    return results


def test_project_structure() -> List[Tuple[str, bool, str]]:
    """Test project structure and key files."""
    project_root = Path(__file__).parent
    
    required_files = [
        ("pyproject.toml", "Project configuration"),
        ("requirements.txt", "Core dependencies"),
        ("requirements-dev.txt", "Development dependencies"),
        (".pre-commit-config.yaml", "Pre-commit hooks"),
        ("README.md", "Project documentation"),
        ("CONTRIBUTING.md", "Contributing guidelines"),
        ("CHANGELOG.md", "Change log"),
        ("SETUP_GUIDE.md", "Setup guide"),
    ]
    
    required_dirs = [
        ("src/", "Source code"),
        ("tests/", "Test suite"),
        ("scripts/", "Utility scripts"),
        (".github/workflows/", "GitHub Actions"),
    ]
    
    results = []
    
    # Check files
    for file_path, description in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            results.append((description, True, f"{description} ✓"))
        else:
            results.append((description, False, f"{description} ✗ (missing)"))
    
    # Check directories
    for dir_path, description in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            results.append((description, True, f"{description} ✓"))
        else:
            results.append((description, False, f"{description} ✗ (missing)"))
    
    return results


def test_git_hooks() -> Tuple[bool, str]:
    """Test if Git hooks are properly configured."""
    project_root = Path(__file__).parent
    hooks_dir = project_root / ".git" / "hooks"
    
    if not hooks_dir.exists():
        return False, "Git repository not found ✗"
    
    # Check for pre-commit hook
    pre_commit_hook = hooks_dir / "pre-commit"
    if pre_commit_hook.exists():
        return True, "Git hooks configured ✓"
    else:
        return False, "Git hooks not configured ✗ (run: pre-commit install)"


def run_tests() -> None:
    """Run all setup tests and display results."""
    print("🚀 Plansheet Scanner Development Environment Test")
    print("=" * 60)
    
    # Test Python version
    print("\n📋 Python Version Check:")
    version_ok, version_msg = test_python_version()
    print(f"  {version_msg}")
    
    # Test package imports
    print("\n📦 Package Import Tests:")
    import_results = test_imports()
    for _, _, msg in import_results:
        print(f"  {msg}")
    
    # Test project structure
    print("\n🏗️  Project Structure Tests:")
    structure_results = test_project_structure()
    for _, _, msg in structure_results:
        print(f"  {msg}")
    
    # Test Git hooks
    print("\n🔧 Git Configuration Tests:")
    hooks_ok, hooks_msg = test_git_hooks()
    print(f"  {hooks_msg}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    all_tests = [version_ok] + [r[1] for r in import_results] + [r[1] for r in structure_results] + [hooks_ok]
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"  Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your development environment is ready.")
        print("\n📋 Next steps:")
        print("  1. Activate your virtual environment")
        print("  2. Run: python -m pytest tests/ -v")
        print("  3. Start developing!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
        print("\n🔧 To fix issues:")
        print("  1. Run: python scripts/setup_dev_environment.py")
        print("  2. Or follow the manual setup in SETUP_GUIDE.md")
    
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
