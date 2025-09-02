# Contributing to Plansheet Scanner

Thank you for your interest in contributing to Plansheet Scanner! This document provides guidelines and standards for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)
- [Documentation](#documentation)
- [Support](#support)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of machine learning and computer vision concepts

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/plansheet-scanner-new.git
   cd plansheet-scanner-new
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/original-username/plansheet-scanner-new.git
   ```

## Development Setup

### Automated Setup

Run the development environment setup script:

```bash
python scripts/setup_dev_environment.py
```

### Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Verify installation:
   ```bash
   python -m pytest tests/ -v
   ```

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use isort with Black profile
- **Type hints**: Required for all function parameters and return values
- **Docstrings**: Google-style docstrings for all public functions and classes

### Code Formatting

All code is automatically formatted using:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run formatting before committing:
```bash
black src/ tests/
isort src/ tests/
```

### Import Organization

Follow this import order:
1. Standard library imports
2. Third-party imports
3. Local application imports

Example:
```python
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from plansheet_scanner.core.models import BaseModel
from plansheet_scanner.utils.helpers import load_data
```

### Type Hints

Use type hints for all functions:

```python
def process_image(
    image_path: Path,
    model_config: ModelConfig,
    output_dir: Optional[Path] = None
) -> ProcessingResult:
    """
    Process an image using the specified model configuration.
    
    Args:
        image_path: Path to the input image
        model_config: Configuration for the ML model
        output_dir: Optional output directory for results
        
    Returns:
        ProcessingResult containing the analysis results
        
    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If model_config is invalid
    """
    # Implementation here
```

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors with appropriate levels
- Handle edge cases gracefully

```python
try:
    result = model.predict(data)
except ModelNotTrainedError:
    logger.error("Model must be trained before prediction")
    raise
except ValueError as e:
    logger.error(f"Invalid input data: {e}")
    raise
```

## Testing Guidelines

### Test Structure

- Place tests in the `tests/` directory
- Mirror the source code structure
- Use descriptive test names
- Group related tests in classes

### Test Naming Convention

```python
def test_function_name_expected_behavior():
    """Test description."""
    # Test implementation

def test_function_name_error_condition():
    """Test error handling."""
    # Test implementation
```

### Test Categories

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.slow
def test_expensive_operation():
    """Test that takes a long time."""
    pass

@pytest.mark.integration
def test_database_integration():
    """Test database operations."""
    pass

@pytest.mark.unit
def test_unit_function():
    """Test individual function."""
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not slow"
pytest -m integration
pytest -m unit

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v
```

### Test Coverage

- Maintain at least 80% test coverage
- Focus on critical business logic
- Mock external dependencies
- Test both success and failure cases

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Run code quality checks:
   ```bash
   pre-commit run --all-files
   ```
3. Update documentation if needed
4. Add tests for new functionality

### Pull Request Template

Use this template for pull requests:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Code Review**: At least one maintainer must approve
3. **Testing**: Ensure all tests pass on multiple Python versions
4. **Documentation**: Update relevant documentation
5. **Merge**: Squash and merge when approved

## Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release tag
- [ ] Deploy to PyPI

### Creating a Release

```bash
# Update version
bump2version patch  # or minor/major

# Create and push tag
git push --tags

# Build and upload to PyPI
python -m build
twine upload dist/*
```

## Documentation

### Code Documentation

- Use Google-style docstrings
- Include examples in docstrings
- Document all public APIs
- Keep docstrings up to date

### Project Documentation

- Update README.md for new features
- Maintain API documentation
- Include setup and usage examples
- Document configuration options

### Documentation Standards

- Use clear, concise language
- Include code examples
- Provide troubleshooting guides
- Keep documentation current

## Support

### Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check the docs directory and README files

### Communication

- Be respectful and professional
- Provide context for issues
- Use clear, descriptive language
- Follow the project's communication guidelines

## Additional Resources

- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [mypy Type Checker](https://mypy.readthedocs.io/)

---

Thank you for contributing to Plansheet Scanner! Your contributions help make this project better for everyone.
