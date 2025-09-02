#!/usr/bin/env python3
"""
GitHub Upload Validation Script
Validates the workspace is ready for GitHub upload based on agent instructions.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return Path(file_path).exists()

def run_command(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, timeout=300)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def check_essential_files() -> Dict[str, bool]:
    """Check if all essential files exist."""
    print("🔍 Checking essential files...")
    
    essential_files = {
        "pyproject.toml": "Modern Python packaging",
        ".pre-commit-config.yaml": "Code quality hooks",
        "requirements.txt": "Dependencies",
        "README.md": "Project documentation",
        "LICENSE": "MIT License",
        ".gitignore": "Git ignore patterns",
        "CHANGELOG.md": "Version history",
        "CONTRIBUTING.md": "Contributing guidelines",
        ".github/workflows/ci.yml": "CI/CD pipeline",
        ".github/pull_request_template.md": "PR template",
        ".github/dependabot.yml": "Dependency updates"
    }
    
    results = {}
    for file_path, description in essential_files.items():
        exists = check_file_exists(file_path)
        results[file_path] = exists
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path} - {description}")
    
    return results

def check_python_packages() -> Dict[str, bool]:
    """Check if required Python packages are installed."""
    print("\n🐍 Checking Python packages...")
    
    required_packages = [
        "black", "isort", "flake8", "mypy", "pytest", "bandit", "safety"
    ]
    
    results = {}
    for package in required_packages:
        try:
            __import__(package)
            results[package] = True
            print(f"   ✅ {package}")
        except ImportError:
            results[package] = False
            print(f"   ❌ {package} - Not installed")
    
    return results

def run_code_quality_checks() -> Dict[str, bool]:
    """Run code quality checks."""
    print("\n🔧 Running code quality checks...")
    
    checks = {
        "black": ["black", "--check", "src/", "tests/"],
        "isort": ["isort", "--check-only", "src/", "tests/"],
        "flake8": ["flake8", "src/", "tests/"],
        "mypy": ["mypy", "src/"],
    }
    
    results = {}
    for check_name, cmd in checks.items():
        print(f"   Running {check_name}...")
        exit_code, stdout, stderr = run_command(cmd)
        success = exit_code == 0
        results[check_name] = success
        status = "✅" if success else "❌"
        print(f"   {status} {check_name}")
        
        if not success and stderr:
            print(f"      Error: {stderr[:200]}...")
    
    return results

def run_security_checks() -> Dict[str, bool]:
    """Run security checks."""
    print("\n🔒 Running security checks...")
    
    checks = {
        "bandit": ["bandit", "-r", "src/", "-f", "json"],
        "safety": ["safety", "check", "--json"]
    }
    
    results = {}
    for check_name, cmd in checks.items():
        print(f"   Running {check_name}...")
        exit_code, stdout, stderr = run_command(cmd)
        success = exit_code == 0
        results[check_name] = success
        status = "✅" if success else "❌"
        print(f"   {status} {check_name}")
        
        if stdout and check_name == "safety":
            try:
                safety_data = json.loads(stdout)
                if safety_data:
                    print(f"      ⚠️  Found {len(safety_data)} security issues")
                else:
                    print(f"      ✅ No security issues found")
            except json.JSONDecodeError:
                pass
    
    return results

def run_tests() -> Dict[str, Any]:
    """Run tests and get coverage."""
    print("\n🧪 Running tests...")
    
    # Run pytest with coverage
    cmd = ["pytest", "--cov=src", "--cov-report=term-missing", "--cov-report=json"]
    exit_code, stdout, stderr = run_command(cmd)
    
    results = {
        "tests_passed": exit_code == 0,
        "coverage_data": None
    }
    
    if exit_code == 0:
        print("   ✅ Tests passed")
        
        # Try to parse coverage data
        try:
            coverage_file = Path("htmlcov/coverage.json")
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    results["coverage_data"] = coverage_data
                    
                    # Extract total coverage
                    if "totals" in coverage_data:
                        total_coverage = coverage_data["totals"]["percent_covered"]
                        print(f"   📊 Coverage: {total_coverage:.1f}%")
                        results["total_coverage"] = total_coverage
        except Exception as e:
            print(f"   ⚠️  Could not parse coverage data: {e}")
    else:
        print("   ❌ Tests failed")
        if stderr:
            print(f"      Error: {stderr[:200]}...")
    
    return results

def check_git_status() -> Dict[str, Any]:
    """Check git repository status."""
    print("\n📝 Checking git status...")
    
    results = {}
    
    # Check if git is initialized
    exit_code, stdout, stderr = run_command(["git", "status"])
    if exit_code == 0:
        results["git_initialized"] = True
        print("   ✅ Git repository initialized")
        
        # Check for uncommitted changes
        if "nothing to commit" in stdout:
            results["clean_working_directory"] = True
            print("   ✅ Working directory is clean")
        else:
            results["clean_working_directory"] = False
            print("   ⚠️  Working directory has uncommitted changes")
            
        # Check for remote
        exit_code, stdout, stderr = run_command(["git", "remote", "-v"])
        if exit_code == 0 and stdout.strip():
            results["remote_configured"] = True
            print("   ✅ Remote repository configured")
        else:
            results["remote_configured"] = False
            print("   ⚠️  No remote repository configured")
    else:
        results["git_initialized"] = False
        print("   ❌ Git repository not initialized")
    
    return results

def generate_summary_report(
    essential_files: Dict[str, bool],
    packages: Dict[str, bool],
    quality_checks: Dict[str, bool],
    security_checks: Dict[str, bool],
    tests: Dict[str, Any],
    git_status: Dict[str, Any]
) -> None:
    """Generate a summary report."""
    print("\n" + "="*60)
    print("📊 GITHUB UPLOAD VALIDATION SUMMARY")
    print("="*60)
    
    # Essential files
    essential_files_passed = all(essential_files.values())
    print(f"\n📁 Essential Files: {'✅ READY' if essential_files_passed else '❌ ISSUES'}")
    for file_path, exists in essential_files.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path}")
    
    # Python packages
    packages_passed = all(packages.values())
    print(f"\n🐍 Python Packages: {'✅ READY' if packages_passed else '❌ ISSUES'}")
    for package, installed in packages.items():
        status = "✅" if installed else "❌"
        print(f"   {status} {package}")
    
    # Code quality
    quality_passed = all(quality_checks.values())
    print(f"\n🔧 Code Quality: {'✅ READY' if quality_passed else '❌ ISSUES'}")
    for check, passed in quality_checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    # Security
    security_passed = all(security_checks.values())
    print(f"\n🔒 Security: {'✅ READY' if security_passed else '❌ ISSUES'}")
    for check, passed in security_checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    # Tests
    tests_passed = tests.get("tests_passed", False)
    print(f"\n🧪 Tests: {'✅ READY' if tests_passed else '❌ ISSUES'}")
    if tests_passed:
        coverage = tests.get("total_coverage", 0)
        print(f"   📊 Coverage: {coverage:.1f}%")
    
    # Git status
    git_ready = git_status.get("git_initialized", False)
    print(f"\n📝 Git Status: {'✅ READY' if git_ready else '❌ ISSUES'}")
    if git_ready:
        clean = git_status.get("clean_working_directory", False)
        remote = git_status.get("remote_configured", False)
        print(f"   {'✅' if clean else '⚠️'} Working directory clean")
        print(f"   {'✅' if remote else '⚠️'} Remote configured")
    
    # Overall status
    all_checks = [
        essential_files_passed,
        packages_passed,
        quality_passed,
        security_passed,
        tests_passed,
        git_ready
    ]
    
    overall_ready = all(all_checks)
    
    print("\n" + "="*60)
    if overall_ready:
        print("🎉 READY FOR GITHUB UPLOAD!")
        print("✅ All checks passed. Your workspace is ready for GitHub upload.")
        print("\nNext steps:")
        print("1. Create GitHub repository")
        print("2. Add remote origin")
        print("3. Push to GitHub")
        print("4. Verify CI/CD pipelines")
    else:
        print("⚠️  ISSUES FOUND - Please fix before uploading")
        print("❌ Some checks failed. Please address the issues above.")
        print("\nRecommended fixes:")
        if not packages_passed:
            print("- Install missing Python packages: pip install -e '.[dev]'")
        if not quality_passed:
            print("- Fix code quality issues: black src/ tests/ && isort src/ tests/")
        if not tests_passed:
            print("- Fix failing tests")
        if not git_ready:
            print("- Initialize git repository and commit changes")
    
    print("="*60)

def main():
    """Main validation function."""
    print("🚀 GitHub Upload Validation")
    print("="*40)
    print("Validating workspace readiness for GitHub upload...")
    
    # Run all checks
    essential_files = check_essential_files()
    packages = check_python_packages()
    quality_checks = run_code_quality_checks()
    security_checks = run_security_checks()
    tests = run_tests()
    git_status = check_git_status()
    
    # Generate summary report
    generate_summary_report(
        essential_files, packages, quality_checks,
        security_checks, tests, git_status
    )

if __name__ == "__main__":
    main()
