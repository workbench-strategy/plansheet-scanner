# Setup Guide for Plansheet Scanner

This guide will help you set up a complete development environment for the Plansheet Scanner project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Development Tools](#development-tools)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Platform-Specific Notes](#platform-specific-notes)

## Prerequisites

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **Python**: 3.8 or higher (3.9+ recommended)
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space minimum
- **Git**: Latest stable version

### Python Installation

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Install with "Add Python to PATH" checked
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

#### macOS
```bash
# Using Homebrew (recommended)
brew install python

# Or download from python.org
# Verify installation
python3 --version
pip3 --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Verify installation
python3 --version
pip3 --version
```

### Git Installation

#### Windows
Download and install from [git-scm.com](https://git-scm.com/)

#### macOS
```bash
brew install git
```

#### Linux
```bash
sudo apt install git
```

## Quick Start

### Automated Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/plansheet-scanner-new.git
   cd plansheet-scanner-new
   ```

2. **Run the setup script**:
   ```bash
   python scripts/setup_dev_environment.py
   ```

3. **Activate the virtual environment**:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Verify installation**:
   ```bash
   python -m pytest tests/ -v
   ```

### Manual Setup

If you prefer manual setup or encounter issues with the automated script:

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Detailed Setup

### 1. Repository Setup

```bash
# Clone the repository
git clone https://github.com/your-username/plansheet-scanner-new.git
cd plansheet-scanner-new

# Add upstream remote (if contributing)
git remote add upstream https://github.com/original-username/plansheet-scanner-new.git

# Verify remotes
git remote -v
```

### 2. Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Verify activation
which python  # Should point to venv directory
pip list      # Should show minimal packages
```

### 3. Dependencies Installation

```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Verify installation
pip list
```

### 4. Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files (optional)
pre-commit run --all-files
```

### 5. Configuration Files

```bash
# Copy environment configuration
cp .env.example .env

# Edit configuration as needed
# Windows: notepad .env
# macOS/Linux: nano .env
```

## Development Tools

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Automated quality checks

### Testing Tools

- **pytest**: Test framework
- **pytest-cov**: Coverage reporting
- **pytest-benchmark**: Performance testing

### Development Utilities

- **ipython**: Enhanced Python shell
- **jupyter**: Interactive notebooks
- **debugpy**: Debugging support

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
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
```

### IDE Configuration

#### VS Code

1. Install Python extension
2. Select Python interpreter (venv)
3. Install recommended extensions:
   - Python
   - Python Docstring Generator
   - Python Test Explorer
   - GitLens

#### PyCharm

1. Open project
2. Configure Python interpreter (venv)
3. Enable type checking
4. Configure pytest

### Git Configuration

```bash
# Set up Git configuration
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Set up pre-commit hooks
pre-commit install

# Verify hooks are active
ls -la .git/hooks/
```

## Troubleshooting

### Common Issues

#### Virtual Environment Issues

**Problem**: Virtual environment not activating
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Verify activation
which python
pip list
```

**Problem**: Permission denied on activation
```bash
# Fix permissions (macOS/Linux)
chmod +x venv/bin/activate
```

#### Dependency Installation Issues

**Problem**: Build errors on Windows
```bash
# Install Visual C++ build tools
# Or use pre-compiled wheels
pip install --only-binary=all package_name
```

**Problem**: SSL certificate errors
```bash
# Update certificates
pip install --upgrade certifi
```

#### Pre-commit Issues

**Problem**: Pre-commit hooks failing
```bash
# Update pre-commit
pip install --upgrade pre-commit

# Reinstall hooks
pre-commit install

# Run manually to see errors
pre-commit run --all-files
```

### Performance Issues

**Problem**: Slow dependency installation
```bash
# Use faster package index
pip install -i https://pypi.org/simple/ package_name

# Or use conda for scientific packages
conda install numpy pandas scikit-learn
```

**Problem**: Memory issues during ML operations
```bash
# Reduce batch sizes in configuration
# Use smaller models for development
# Monitor memory usage with psutil
```

## Platform-Specific Notes

### Windows

- Use `python` instead of `python3`
- Use `venv\Scripts\activate` for virtual environment
- Install Visual C++ build tools for some packages
- Use Windows Subsystem for Linux (WSL) for better compatibility

### macOS

- Use `python3` and `pip3` commands
- Install Xcode command line tools: `xcode-select --install`
- Use Homebrew for system dependencies
- Consider using pyenv for Python version management

### Linux (Ubuntu/Debian)

- Install system dependencies: `sudo apt install build-essential python3-dev`
- Use `python3` and `pip3` commands
- Consider using pyenv for Python version management
- Install tesseract: `sudo apt install tesseract-ocr`

## Verification

### Test Installation

```bash
# Run test suite
python -m pytest tests/ -v

# Check code quality
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

### Check Dependencies

```bash
# List installed packages
pip list

# Check for outdated packages
pip list --outdated

# Verify specific packages
python -c "import torch; print(torch.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

## Next Steps

After successful setup:

1. **Read the documentation**:
   - [README.md](README.md)
   - [CONTRIBUTING.md](CONTRIBUTING.md)
   - [API Documentation](docs/)

2. **Explore the codebase**:
   - Start with `src/plansheet_scanner/` for core functionality
   - Check `tests/` for examples
   - Review `examples/` for usage patterns

3. **Run examples**:
   ```bash
   python examples/basic_usage.py
   ```

4. **Start developing**:
   - Create a feature branch
   - Make your changes
   - Run tests and quality checks
   - Submit a pull request

## Support

If you encounter issues:

1. **Check this guide** for common solutions
2. **Search existing issues** on GitHub
3. **Create a new issue** with detailed information
4. **Join discussions** in GitHub Discussions

---

**Happy Coding! 🚀**
