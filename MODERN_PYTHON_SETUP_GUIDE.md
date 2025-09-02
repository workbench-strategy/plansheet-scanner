# Modern Python Environment & Organization Guide

## 🎯 Overview

This guide provides comprehensive agent prompts and best practices for setting up a modern, lean, and mean Python project environment. It covers everything from project structure to CI/CD pipelines, making your Python projects production-ready and maintainable.

## 🏗️ Project Structure

### Lean & Mean Directory Layout

```
project-name/
├── pyproject.toml          # Modern Python packaging (replaces setup.py)
├── .python-version         # Python version specification
├── .pre-commit-config.yaml # Code quality hooks
├── .github/
│   ├── workflows/          # CI/CD pipelines
│   ├── ISSUE_TEMPLATE/     # Issue templates
│   ├── pull_request_template.md
│   └── dependabot.yml      # Automated dependency updates
├── src/
│   └── package_name/       # Main package (src layout)
│       ├── __init__.py     # Package initialization
│       ├── core/           # Core functionality
│       ├── cli/            # Command-line interface
│       └── utils/          # Utilities
├── tests/                  # Test suite
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── .gitignore             # Comprehensive ignore patterns
├── README.md              # Project overview
└── CHANGELOG.md           # Version history
```

## 📦 Modern Python Packaging

### pyproject.toml Configuration

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "your-package-name"
version = "1.0.0"
description = "Your package description"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your-email@example.com"}
]
keywords = ["your", "keywords"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    # Core dependencies with version pinning
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.0.0,<3.0.0",
    "requests>=2.28.0,<3.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0,<8.0.0",
    "black>=23.7.0,<24.0.0",
    "isort>=5.12.0,<6.0.0",
    "flake8>=6.0.0,<7.0.0",
    "mypy>=1.5.0,<2.0.0",
    "pre-commit>=3.3.0,<4.0.0",
    "bandit>=1.7.0,<2.0.0",
    "safety>=2.3.0,<3.0.0",
]
test = [
    "pytest>=7.4.0,<8.0.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "pytest-benchmark>=4.0.0,<5.0.0",
]
docs = [
    "mkdocs>=1.5.0,<2.0.0",
    "mkdocs-material>=9.2.0,<10.0.0",
    "mkdocstrings>=0.22.0,<1.0.0",
]
performance = [
    "memory-profiler>=0.61.0,<1.0.0",
    "py-spy>=0.3.0,<1.0.0",
    "line-profiler>=4.0.0,<5.0.0",
]

[project.urls]
Homepage = "https://github.com/your-username/your-repo"
Documentation = "https://your-package.readthedocs.io"
Repository = "https://github.com/your-username/your-repo"
Issues = "https://github.com/your-username/your-repo/issues"

[project.scripts]
your-command = "src.package_name.cli.main:main"

# Tool configurations
[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
disallow_untyped_defs = true
strict_equality = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = [
    "-v",
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "performance: marks tests as performance tests",
]
```

## 🔧 Code Quality & Standards

### Pre-commit Configuration

```yaml
repos:
  # General hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: check-case-conflict
      - id: check-docstring-first
      - id: check-json
      - id: check-toml
      - id: debug-statements
      - id: requirements-txt-fixer

  # Code formatting
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  # Import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  # Linting
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  # Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]

  # Security scanning
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.0
    hooks:
      - id: bandit
        args: [-r, ., -f, json, -o, bandit-report.json]

  # Safety checks
  - repo: https://github.com/PyCQA/safety
    rev: 2.3.0
    hooks:
      - id: safety
        args: [--json, --output, safety-report.json]

  # Commit message formatting
  - repo: https://github.com/commitizen-tools/commitizen
    rev: v3.5.2
    hooks:
      - id: commitizen
        stages: [commit-msg]
```

## 🚀 GitHub Pro Workflow

### CI/CD Pipeline

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  release:
    types: [published]

env:
  PYTHON_VERSION: "3.11"

jobs:
  # Quality checks
  quality:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run pre-commit hooks
        run: pre-commit run --all-files

      - name: Run security scan
        run: |
          bandit -r src/ -f json -o bandit-report.json
          safety check --json --output safety-report.json

  # Testing matrix
  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('**/pyproject.toml') }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test,dev]"

      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml --cov-report=term-missing

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: false

  # Performance testing
  performance:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test,performance]"

      - name: Run performance tests
        run: |
          pytest -m performance --benchmark-only

  # Documentation build
  docs:
    name: Build Documentation
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[docs]"

      - name: Build documentation
        run: |
          mkdocs build

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site

  # Build and publish
  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [test, quality]
    if: github.event_name == 'release'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### Dependabot Configuration

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "America/New_York"
    open-pull-requests-limit: 10
    reviewers:
      - "your-github-username"
    assignees:
      - "your-github-username"
    commit-message:
      prefix: "pip"
      include: "scope"
    labels:
      - "dependencies"
      - "python"
    ignore:
      # Ignore major version updates for critical dependencies
      - dependency-name: "numpy"
        update-types: ["version-update:semver-major"]

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "America/New_York"
    open-pull-requests-limit: 5
    reviewers:
      - "your-github-username"
    assignees:
      - "your-github-username"
    commit-message:
      prefix: "ci"
      include: "scope"
    labels:
      - "dependencies"
      - "github-actions"

  # Security updates (high priority)
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "daily"
      time: "06:00"
      timezone: "America/New_York"
    open-pull-requests-limit: 5
    reviewers:
      - "your-github-username"
    assignees:
      - "your-github-username"
    commit-message:
      prefix: "security"
      include: "scope"
    labels:
      - "dependencies"
      - "security"
      - "high-priority"
```

## 📋 Issue and PR Templates

### Bug Report Template

```markdown
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ['your-github-username']

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment (please complete the following information):**
 - OS: [e.g. Windows 10, macOS 12.0, Ubuntu 20.04]
 - Python Version: [e.g. 3.8, 3.9, 3.10, 3.11]
 - Package Version: [e.g. 1.0.0]
 - Installation Method: [e.g. pip, conda, source]

**Error logs**
If applicable, paste the error logs here:

```

# Paste error logs here
```

**Checklist**
- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided all required information
- [ ] I have included error logs if applicable
- [ ] I have tested with the latest version
```

### Feature Request Template

```markdown
---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: ['enhancement', 'needs-triage']
assignees: ['your-github-username']

---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions.

**Use case**
Describe a specific use case where this feature would be valuable:

**Priority**
- [ ] Low - Nice to have
- [ ] Medium - Important for workflow
- [ ] High - Critical for core functionality
- [ ] Urgent - Blocking current work

**Implementation complexity**
- [ ] Simple - Minor changes
- [ ] Medium - Moderate effort
- [ ] Complex - Significant development time
- [ ] Very Complex - Major architectural changes

**Checklist**
- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a clear use case
- [ ] I have considered alternatives
- [ ] This feature aligns with project goals
```

### Pull Request Template

```markdown
## Description
Brief description of changes made in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)
- [ ] Test updates
- [ ] CI/CD improvements

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass (if applicable)
- [ ] Manual testing completed
- [ ] All pre-commit hooks pass

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules

## Related Issues
Closes #(issue number)
Related to #(issue number)

## Performance Impact
- [ ] No performance impact
- [ ] Performance improvement
- [ ] Performance regression (explain below)

## Breaking Changes
- [ ] No breaking changes
- [ ] Breaking changes (explain below)

## Additional Notes
Any additional information that reviewers should know.
```

## 🎯 Agent Prompts for Common Tasks

### Setting Up a New Project

**Prompt:**
```
Create a new modern Python project with the following requirements:

1. Use pyproject.toml instead of setup.py
2. Implement src layout for the package
3. Set up comprehensive testing with pytest
4. Configure pre-commit hooks for code quality
5. Set up GitHub Actions CI/CD pipeline
6. Add type hints throughout the codebase
7. Implement proper logging and error handling
8. Create comprehensive documentation structure
9. Set up performance monitoring and benchmarking
10. Configure security scanning and dependency management

Follow the modern Python best practices:
- Use dependency groups (main, dev, test, docs, performance)
- Implement proper version pinning
- Set up automated quality checks
- Configure comprehensive testing
- Add performance profiling tools
- Implement security scanning
- Set up automated documentation building
- Configure release automation

Project name: [PROJECT_NAME]
Description: [PROJECT_DESCRIPTION]
Python version: 3.8+
```

### Adding New Features

**Prompt:**
```
When implementing new features in this modern Python project:

1. Follow the established project structure and patterns
2. Add comprehensive type hints for all functions and classes
3. Write unit tests with pytest for all new functionality
4. Add integration tests for complex workflows
5. Update documentation with usage examples
6. Follow the established code formatting standards (Black, isort)
7. Add performance benchmarks for performance-critical code
8. Implement proper error handling and logging
9. Update the CLI interface if applicable
10. Add security considerations for user inputs

Requirements:
- Feature: [FEATURE_DESCRIPTION]
- Performance requirements: [PERFORMANCE_REQUIREMENTS]
- Security considerations: [SECURITY_CONSIDERATIONS]
- Testing requirements: [TESTING_REQUIREMENTS]
- Documentation needs: [DOCUMENTATION_NEEDS]

Follow the established patterns in the codebase and ensure all quality checks pass.
```

### Performance Optimization

**Prompt:**
```
Optimize the performance of this Python code following modern best practices:

1. Profile the code to identify bottlenecks
2. Use vectorized operations with NumPy/Pandas where possible
3. Implement caching for expensive computations
4. Use multiprocessing for CPU-intensive tasks
5. Optimize memory usage and implement garbage collection
6. Use appropriate data structures for the task
7. Consider GPU acceleration with PyTorch if applicable
8. Implement lazy loading for large datasets
9. Add performance monitoring and metrics
10. Create performance benchmarks for regression testing

Code to optimize:
[PASTE_CODE_HERE]

Performance requirements:
- Target execution time: [TARGET_TIME]
- Memory constraints: [MEMORY_CONSTRAINTS]
- CPU usage: [CPU_REQUIREMENTS]

Use the established performance monitoring tools in the project:
- pytest-benchmark for benchmarking
- memory-profiler for memory analysis
- py-spy for CPU profiling
- cProfile for detailed profiling
```

### Security Hardening

**Prompt:**
```
Implement security hardening for this Python application:

1. Validate and sanitize all user inputs
2. Use environment variables for sensitive configuration
3. Implement proper authentication and authorization
4. Add rate limiting for API endpoints
5. Use secure random number generation
6. Implement proper session management
7. Add input validation and sanitization
8. Use HTTPS for all external communications
9. Implement proper error handling without information disclosure
10. Add security logging and monitoring

Security requirements:
- Input validation: [VALIDATION_REQUIREMENTS]
- Authentication: [AUTH_REQUIREMENTS]
- Data protection: [PROTECTION_REQUIREMENTS]
- Compliance: [COMPLIANCE_REQUIREMENTS]

Use the established security tools in the project:
- bandit for security scanning
- safety for dependency vulnerability checking
- pre-commit hooks for security checks
- Automated security testing in CI/CD
```

## 🔧 Development Workflow

### Daily Development Commands

```bash
# Set up development environment
pip install -e ".[dev,test,docs,performance]"
pre-commit install

# Daily workflow
git checkout -b feature/your-feature
# Make changes
pre-commit run --all-files  # Run quality checks
pytest  # Run tests
pytest -m performance --benchmark-only  # Run performance tests
git add .
git commit -m "feat: add your feature"
git push origin feature/your-feature
# Create PR
```

### Code Review Checklist

- [ ] Type hints are present and correct
- [ ] All tests pass
- [ ] Code follows formatting standards (Black, isort)
- [ ] No linting errors (flake8)
- [ ] Security scan passes (bandit, safety)
- [ ] Documentation is updated
- [ ] Performance impact is assessed
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate
- [ ] No hardcoded secrets

### Release Process

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Create release branch
git checkout -b release/v1.0.0
# Update version
git commit -m "chore: bump version to 1.0.0"
git push origin release/v1.0.0
# Create PR to main
# After merge, create GitHub release
# CI/CD will automatically build and publish to PyPI
```

## 📊 Monitoring and Metrics

### Performance Monitoring

```python
import time
import logging
from functools import wraps
from typing import Any, Callable

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            logging.info(
                f"Function {func.__name__} executed in {execution_time:.4f}s, "
                f"memory: {memory_used / 1024 / 1024:.2f}MB"
            )
    
    return wrapper
```

### Logging Configuration

```python
import logging
import sys
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """Set up structured logging configuration."""
    
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(funcName)s:%(lineno)d - %(message)s"
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
        force=True
    )
```

## 🎯 Best Practices Summary

### Code Quality
- Use type hints throughout
- Follow PEP 8 with Black formatting
- Sort imports with isort
- Use comprehensive docstrings
- Implement proper error handling
- Add logging for debugging

### Testing
- Write unit tests for all functions
- Use pytest for testing framework
- Implement integration tests
- Add performance benchmarks
- Maintain high test coverage
- Use test markers for organization

### Security
- Validate all inputs
- Use environment variables for secrets
- Implement proper authentication
- Add security scanning
- Follow principle of least privilege
- Regular dependency updates

### Performance
- Profile code regularly
- Use vectorized operations
- Implement caching strategies
- Optimize memory usage
- Use appropriate data structures
- Monitor performance metrics

### Documentation
- Keep README updated
- Add API documentation
- Include usage examples
- Document configuration options
- Maintain changelog
- Add architecture documentation

### DevOps
- Use modern Python packaging
- Implement CI/CD pipelines
- Automate quality checks
- Set up monitoring
- Use dependency management
- Implement release automation

This guide provides a comprehensive foundation for modern Python development that is lean, mean, and production-ready. Follow these practices to create maintainable, scalable, and secure Python applications.
