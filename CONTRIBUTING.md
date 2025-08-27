# Contributing to PlanSheet Scanner

Thank you for your interest in contributing to the PlanSheet Scanner project! This document provides guidelines and information for contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your changes
4. **Make your changes** following the guidelines below
5. **Test your changes** thoroughly
6. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda for package management

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/plansheet-scanner-new.git
cd plansheet-scanner-new

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Code Style Guidelines

### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and under 50 lines when possible
- Use type hints where appropriate

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(core): add new cable matching algorithm
fix(extractor): resolve PDF parsing edge case
docs(readme): update installation instructions
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run tests with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_cable_matcher.py
```

### Writing Tests

- Place test files in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies when appropriate
- Include integration tests for complex workflows

## Pull Request Process

1. **Update documentation** if your changes affect user-facing functionality
2. **Add tests** for new features or bug fixes
3. **Ensure all tests pass** before submitting
4. **Update the CHANGELOG.md** with a brief description of your changes
5. **Request review** from maintainers

### Pull Request Template

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Other (please describe)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Project Structure

```
plansheet-scanner-new/
├── src/                    # Core source code
│   ├── core/              # Core functionality
│   └── cli/               # Command-line interfaces
├── tests/                 # Test files
├── docs/                  # Documentation
├── templates/             # Template files
├── scripts/               # Utility scripts
└── requirements.txt       # Python dependencies
```

## Areas for Contribution

### High Priority
- Bug fixes and performance improvements
- Enhanced error handling and logging
- Additional unit and integration tests
- Documentation improvements

### Medium Priority
- New feature development
- UI/UX improvements
- Additional file format support
- Performance optimizations

### Low Priority
- Code refactoring
- Style improvements
- Additional examples and tutorials

## Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Documentation**: Check the `docs/` directory for detailed documentation

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to PlanSheet Scanner!
