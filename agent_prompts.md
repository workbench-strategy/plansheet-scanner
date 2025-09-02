# Agent Prompts for Plansheet Scanner Workspace

This file contains specific agent prompts and templates for common development tasks in this workspace. Use these prompts with Cursor to get consistent, high-quality implementations.

## 🏗️ **MODERN PYTHON ENVIRONMENT & ORGANIZATION PROMPTS**

### Lean & Mean Project Structure Setup

**Prompt:**
```
Create a modern, lean Python project structure following these principles:
- Minimal dependencies with clear separation of concerns
- Modern Python packaging standards (pyproject.toml over setup.py)
- Comprehensive testing with pytest
- Type hints throughout
- Automated CI/CD with GitHub Actions
- Clear documentation with mkdocs
- Dependency management with poetry or pip-tools
- Pre-commit hooks for code quality

Required structure:
```
plansheet-scanner/
├── pyproject.toml          # Modern Python packaging
├── .python-version         # Python version specification
├── .pre-commit-config.yaml # Code quality hooks
├── .github/
│   └── workflows/          # CI/CD pipelines
├── src/
│   └── plansheet_scanner/  # Main package
├── tests/                  # Test suite
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── .gitignore             # Comprehensive ignore patterns
├── README.md              # Project overview
└── CHANGELOG.md           # Version history
```

### Modern Python Packaging Setup

**Prompt:**
```
Convert this project to use modern Python packaging standards:

1. Replace setup.py with pyproject.toml
2. Use build-system specification
3. Implement proper dependency groups (main, dev, test, docs)
4. Add type checking with mypy
5. Configure pytest with coverage
6. Set up pre-commit hooks for:
   - Code formatting (black)
   - Import sorting (isort)
   - Linting (flake8)
   - Type checking (mypy)
   - Security scanning (bandit)

Example pyproject.toml structure:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "plansheet-scanner"
version = "1.0.0"
description = "ML-powered plansheet scanner for engineering drawings"
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "torch>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.7.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
]
test = ["pytest>=7.4.0", "pytest-cov>=4.1.0"]
docs = ["mkdocs>=1.5.0", "mkdocs-material>=9.2.0"]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=src --cov-report=html --cov-report=term-missing"
```

### GitHub Pro Workflow Setup

**Prompt:**
```
Set up professional GitHub workflows for this project:

1. Create comprehensive CI/CD pipeline:
   - Automated testing on multiple Python versions
   - Code quality checks (linting, formatting, type checking)
   - Security scanning
   - Documentation building and deployment
   - Release automation

2. Implement branch protection rules:
   - Require PR reviews
   - Require status checks to pass
   - Require up-to-date branches
   - Enforce linear history

3. Set up issue templates and PR templates

4. Configure automated dependency updates with Dependabot

Example GitHub Actions workflow:
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run pre-commit hooks
      run: pre-commit run --all-files
    
    - name: Run tests
      run: pytest
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Dependency Management Optimization

**Prompt:**
```
Optimize dependency management for this ML project:

1. Analyze current requirements.txt and identify:
   - Unused dependencies
   - Version conflicts
   - Security vulnerabilities
   - Performance bottlenecks

2. Create dependency groups:
   - Core: Essential functionality
   - ML: Machine learning libraries
   - Dev: Development tools
   - Test: Testing frameworks
   - Docs: Documentation tools

3. Implement dependency pinning strategy:
   - Pin major versions for stability
   - Use dependency lock files
   - Regular security updates

4. Set up automated dependency management:
   - Dependabot configuration
   - Automated vulnerability scanning
   - Dependency update workflows

Example optimized requirements structure:
```toml
[project]
dependencies = [
    # Core dependencies
    "opencv-python>=4.8.0,<5.0.0",
    "Pillow>=10.0.0,<11.0.0",
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.0.0,<3.0.0",
    
    # ML dependencies
    "scikit-learn>=1.3.0,<2.0.0",
    "torch>=2.0.0,<3.0.0",
    "transformers>=4.30.0,<5.0.0",
    
    # Utilities
    "tqdm>=4.65.0,<5.0.0",
    "PyYAML>=6.0,<7.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0,<8.0.0",
    "black>=23.7.0,<24.0.0",
    "mypy>=1.5.0,<2.0.0",
    "pre-commit>=3.3.0,<4.0.0",
]
```

### Code Quality & Standards Enforcement

**Prompt:**
```
Implement comprehensive code quality standards:

1. Set up pre-commit hooks for:
   - Code formatting (black)
   - Import sorting (isort)
   - Linting (flake8)
   - Type checking (mypy)
   - Security scanning (bandit)
   - Commit message formatting

2. Configure tools for consistency:
   - Black configuration for consistent formatting
   - isort configuration for import organization
   - flake8 configuration for style enforcement
   - mypy configuration for type checking

3. Set up automated quality checks:
   - GitHub Actions for CI/CD
   - Code coverage reporting
   - Performance benchmarking
   - Security vulnerability scanning

Example pre-commit configuration:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Performance Optimization Setup

**Prompt:**
```
Set up performance monitoring and optimization tools:

1. Implement performance profiling:
   - cProfile integration
   - Memory profiling with memory_profiler
   - CPU profiling with py-spy
   - Benchmarking with pytest-benchmark

2. Set up monitoring:
   - Application metrics collection
   - Performance regression detection
   - Resource usage monitoring
   - Automated performance testing

3. Configure optimization tools:
   - Cython for critical paths
   - Numba for numerical computations
   - Multiprocessing for parallel processing
   - Caching strategies

Example performance setup:
```python
# performance_monitor.py
import cProfile
import pstats
import io
from functools import wraps
from memory_profiler import profile

def performance_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        with open(f'profile_{func.__name__}.txt', 'w') as f:
            f.write(s.getvalue())
        
        return result
    return wrapper

@profile
def memory_intensive_function():
    # Function implementation
    pass
```

### Documentation & Knowledge Management

**Prompt:**
```
Set up comprehensive documentation system:

1. Implement mkdocs with material theme:
   - API documentation
   - User guides
   - Developer guides
   - Architecture documentation
   - Performance benchmarks

2. Set up automated documentation:
   - Auto-generated API docs
   - Code examples
   - Interactive tutorials
   - Version-specific docs

3. Create knowledge management:
   - Decision records (ADRs)
   - Architecture decision log
   - Performance optimization notes
   - Troubleshooting guides

Example mkdocs configuration:
```yaml
site_name: PlanSheet Scanner
site_description: ML-powered plansheet scanner for engineering drawings
site_author: HNTB DIS SEA_DTS Python Working Group

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - search.highlight
    - search.share

plugins:
  - search
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: [src]
          options:
            show_source: true
            show_root_heading: true

nav:
  - Home: index.md
  - User Guide:
    - Getting Started: user-guide/getting-started.md
    - Installation: user-guide/installation.md
    - Configuration: user-guide/configuration.md
  - Developer Guide:
    - Architecture: dev-guide/architecture.md
    - Contributing: dev-guide/contributing.md
    - Testing: dev-guide/testing.md
  - API Reference:
    - Core: api/core.md
    - ML Models: api/ml-models.md
    - Utilities: api/utilities.md
```

## 🚀 **NEW AGENT INTEGRATION PROMPTS**

### Enhanced Agent System Overview

**System Context:**
```
The plansheet scanner now includes 10 specialized agents:
1. PerformanceReviewer (InterdisciplinaryReviewer) - Algorithmic complexity analysis
2. CodeHighlighter - Deprecated function detection and syntax highlighting
3. TrafficPlanReviewer - Batch processing with memory optimization
4. affine_transform - Robust error handling for geospatial transformations
5. legend_extractor - Auto-cropping with color clustering
6. overlay - KMZ generation with multiple DPI options
7. manual_georef - GeoJSON export functionality
8. line_matcher - OpenCV-based line detection and matching
9. cable_entity_pipeline - PDF entity extraction with ML
10. InterdisciplinaryReviewer - Multi-perspective code analysis

All agents follow consistent patterns: comprehensive error handling, CLI interfaces, 
unit tests, logging, and integration capabilities.
```

### Performance Review Integration

**Prompt:**
```
When implementing new features that may have performance implications, integrate with the PerformanceReviewer:

1. Import and use PerformanceReviewer from interdisciplinary_reviewer.py
2. Add performance analysis to your code review workflow
3. Include complexity detection for nested loops and inefficient operations
4. Provide optimization recommendations
5. Follow the patterns established in tests/test_performance_reviewer.py

Example integration:
```python
from src.core.interdisciplinary_reviewer import PerformanceReviewer

def analyze_performance(code: str) -> Dict[str, Any]:
    reviewer = PerformanceReviewer()
    return reviewer.analyze_complexity(code)
```
```

### Code Highlighting Integration

**Prompt:**
```
When working with code analysis or syntax highlighting, use the enhanced CodeHighlighter:

1. Import CodeHighlighter from src.core.code_companion
2. Use deprecated function detection for Python code
3. Include warning messages in highlighted output
4. Follow patterns in tests/test_code_highlighter.py
5. Add new deprecated patterns as needed

Example usage:
```python
from src.core.code_companion import CodeHighlighter

def highlight_code_with_warnings(code: str) -> HighlightedCode:
    highlighter = CodeHighlighter()
    return highlighter.highlight_code(code, "python")
```
```

### Batch Processing Integration

**Prompt:**
```
When implementing batch processing features, follow the TrafficPlanReviewer patterns:

1. Use memory optimization with gc.collect() and psutil monitoring
2. Implement comprehensive JSON output with metadata
3. Include progress tracking and error handling
4. Add CLI options for batch processing
5. Follow patterns in tests/test_batch_review.py

Key requirements:
- Memory-efficient processing for large datasets
- Detailed JSON output with summary statistics
- Graceful error handling and recovery
- Progress reporting and logging
```

### Geospatial Processing Integration

**Prompt:**
```
When working with geospatial data, use the enhanced geospatial agents:

1. Use manual_georef.py for GeoJSON export functionality
2. Use overlay.py for KMZ generation with validation
3. Use affine_transform with robust error handling
4. Include coordinate validation and CRS support
5. Follow patterns in tests/test_affine_transform.py

Example geospatial workflow:
```python
from src.core.manual_georef import export_control_points_to_geojson
from src.core.overlay import generate_kmz_overlay
from src.core.kmz_matcher import affine_transform

# Export control points
export_control_points_to_geojson(control_points, "output.geojson")

# Generate KMZ overlay
generate_kmz_overlay(pdf_path, pages, dpi, control_box, "output.kmz")

# Apply affine transformation with error handling
transform_matrix = affine_transform(control_points)
```
```

### Computer Vision Integration

**Prompt:**
```
When implementing computer vision features, use the enhanced vision agents:

1. Use legend_extractor.py for auto-cropping with color clustering
2. Use line_matcher.py for line detection and matching
3. Use cable_entity_pipeline.py for entity extraction
4. Include confidence scoring and validation
5. Follow OpenCV best practices

Example computer vision workflow:
```python
from src.core.legend_extractor import extract_symbols_from_legend
from src.core.line_matcher import LineMatcher
from src.core.cable_entity_pipeline import CableEntityPipeline

# Auto-crop legend symbols
symbols = extract_symbols_from_legend(pdf_path, page_num, use_auto_crop=True)

# Match lines between images
matcher = LineMatcher()
matches = matcher.detect_and_match(image1, image2)

# Extract cable entities
pipeline = CableEntityPipeline()
entities = pipeline.parse_pdf(pdf_path)
```
```

### Entity Extraction Integration

**Prompt:**
```
When implementing entity extraction from documents, use the CableEntityPipeline patterns:

1. Create dataclasses for entity representation
2. Implement multiple extraction methods (text, visual, ML)
3. Include confidence scoring and validation
4. Add comprehensive statistics and reporting
5. Follow patterns in tests/test_cable_entity_pipeline.py

Key requirements:
- Multiple extraction methods with fallback
- Confidence scoring for all extractions
- Comprehensive error handling
- JSON export with metadata
- CLI interface with filtering options
```

## 🔧 **DEVELOPMENT WORKFLOW PROMPTS**

### New Feature Development

**Prompt:**
```
When implementing new features in the plansheet scanner:

1. **Agent Integration**: Check if your feature should integrate with existing agents
2. **Error Handling**: Use robust error handling patterns from existing agents
3. **CLI Interface**: Add comprehensive CLI options following existing patterns
4. **Unit Tests**: Create tests following patterns in tests/ directory
5. **Logging**: Use structured logging with appropriate levels
6. **Documentation**: Update relevant documentation files

Required components:
- Main functionality in src/core/
- CLI interface with argparse
- Comprehensive unit tests
- Error handling and validation
- Logging and progress reporting
- JSON output for programmatic use
```

### Testing Integration

**Prompt:**
```
When creating tests for new features:

1. **Follow Existing Patterns**: Use patterns from tests/test_*.py files
2. **Mock Dependencies**: Mock external services and file operations
3. **Edge Cases**: Test error conditions and boundary cases
4. **Integration Tests**: Test agent interactions where applicable
5. **Performance Tests**: Include performance benchmarks for critical paths

Test structure:
```python
class TestNewFeature:
    def setup_method(self):
        """Set up test fixtures."""
        
    def teardown_method(self):
        """Clean up test fixtures."""
        
    def test_success_case(self):
        """Test successful execution."""
        
    def test_error_handling(self):
        """Test error conditions."""
        
    def test_integration_with_agents(self):
        """Test integration with existing agents."""
```
```

### CLI Development

**Prompt:**
```
When creating CLI interfaces:

1. **Use argparse**: Follow patterns from existing CLI modules
2. **Help Text**: Provide comprehensive help with examples
3. **Validation**: Validate all inputs with clear error messages
4. **Progress Reporting**: Show progress for long-running operations
5. **Output Options**: Support both human-readable and JSON output

CLI structure:
```python
def main():
    parser = argparse.ArgumentParser(
        description="Feature description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python module.py input --output result.json
  python module.py --batch --verbose
        """
    )
    
    parser.add_argument("input", help="Input description")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    try:
        # Implementation
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
```
```

### Error Handling Patterns

**Prompt:**
```
When implementing error handling:

1. **Specific Exceptions**: Use specific exception types (ValueError, FileNotFoundError, etc.)
2. **Clear Messages**: Provide descriptive error messages with context
3. **Graceful Degradation**: Handle errors gracefully when possible
4. **Logging**: Log errors with appropriate levels
5. **Recovery**: Provide recovery options when feasible

Error handling pattern:
```python
def process_data(input_path: str) -> Result:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        # Processing logic
        return result
    except SpecificError as e:
        logger.error(f"Processing failed: {e}")
        raise RuntimeError(f"Failed to process {input_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```
```

### Logging Integration

**Prompt:**
```
When implementing logging:

1. **Structured Logging**: Use Python's logging module with timestamps
2. **Appropriate Levels**: Use DEBUG, INFO, WARNING, ERROR appropriately
3. **Context**: Include relevant context in log messages
4. **Progress**: Log progress for long-running operations
5. **Configuration**: Allow log level configuration via CLI

Logging pattern:
```python
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_with_logging():
    logger.info("Starting processing")
    logger.debug("Processing details")
    logger.warning("Non-critical issue")
    logger.error("Critical error")
```
```

## 🧪 **TESTING AND VALIDATION PROMPTS**

### Agent Integration Testing

**Prompt:**
```
When testing agent integrations:

1. **Mock Dependencies**: Mock external services and file operations
2. **Test Interactions**: Test how agents work together
3. **Error Scenarios**: Test error propagation between agents
4. **Performance**: Test performance impact of agent interactions
5. **Edge Cases**: Test boundary conditions and unusual inputs

Integration test pattern:
```python
class TestAgentIntegration:
    def test_agent_workflow(self):
        """Test complete agent workflow."""
        with patch('external_service') as mock_service:
            # Test agent interactions
            result = workflow.run()
            assert result.success
            mock_service.assert_called()
```
```

### Performance Testing

**Prompt:**
```
When testing performance:

1. **Benchmarks**: Create performance benchmarks for critical paths
2. **Memory Usage**: Monitor memory usage for large datasets
3. **Scalability**: Test with different data sizes
4. **Optimization**: Identify bottlenecks and optimization opportunities
5. **Comparison**: Compare performance with existing implementations

Performance test pattern:
```python
def test_performance_benchmark():
    """Test performance with realistic data."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    result = process_large_dataset()
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    assert end_time - start_time < 30  # 30 second limit
    assert end_memory - start_memory < 100 * 1024 * 1024  # 100MB limit
```
```

## 📚 **DOCUMENTATION PROMPTS**

### Agent Documentation

**Prompt:**
```
When documenting new agents:

1. **Overview**: Provide clear overview of agent purpose and capabilities
2. **Usage Examples**: Include practical usage examples
3. **Integration**: Document how to integrate with other agents
4. **Configuration**: Document all configuration options
5. **Troubleshooting**: Include common issues and solutions

Documentation structure:
```markdown
# Agent Name

## Overview
Brief description of agent purpose and capabilities.

## Usage
```python
from src.core.agent_name import AgentClass

agent = AgentClass()
result = agent.process(input_data)
```

## Integration
How to integrate with other agents and systems.

## Configuration
Available configuration options and their effects.

## Troubleshooting
Common issues and their solutions.
```
```

### API Documentation

**Prompt:**
```
When documenting APIs:

1. **Type Hints**: Include comprehensive type hints
2. **Docstrings**: Write detailed docstrings with examples
3. **Parameters**: Document all parameters and return values
4. **Exceptions**: Document all possible exceptions
5. **Examples**: Include practical usage examples

API documentation pattern:
```python
def process_data(input_path: str, 
                output_path: Optional[str] = None,
                config: Dict[str, Any] = None) -> ProcessingResult:
    """
    Process data from input file and optionally save to output file.
    
    Args:
        input_path: Path to input file
        output_path: Optional path for output file
        config: Optional configuration dictionary
        
    Returns:
        ProcessingResult object with results and metadata
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input data is invalid
        RuntimeError: If processing fails
        
    Example:
        >>> result = process_data("input.txt", "output.json")
        >>> print(result.success)
        True
    """
```
```

## 🔄 **MAINTENANCE AND ENHANCEMENT PROMPTS**

### Code Refactoring

**Prompt:**
```
When refactoring existing code:

1. **Agent Integration**: Ensure refactored code integrates with existing agents
2. **Backward Compatibility**: Maintain backward compatibility where possible
3. **Testing**: Update tests to reflect refactored code
4. **Documentation**: Update documentation to reflect changes
5. **Performance**: Ensure refactoring doesn't degrade performance

Refactoring checklist:
- [ ] Code integrates with existing agents
- [ ] Backward compatibility maintained
- [ ] Tests updated and passing
- [ ] Documentation updated
- [ ] Performance validated
- [ ] Error handling improved
```

### Feature Enhancement

**Prompt:**
```
When enhancing existing features:

1. **Agent Compatibility**: Ensure enhancements work with existing agents
2. **Configuration**: Add configuration options for new features
3. **Testing**: Add tests for new functionality
4. **Documentation**: Update documentation for new features
5. **Performance**: Ensure enhancements don't degrade performance

Enhancement pattern:
```python
def enhanced_feature(input_data: InputData, 
                    new_option: bool = False,
                    config: Dict[str, Any] = None) -> EnhancedResult:
    """
    Enhanced version of existing feature with new capabilities.
    
    Args:
        input_data: Input data (existing parameter)
        new_option: New feature flag (new parameter)
        config: Configuration for new features (new parameter)
        
    Returns:
        Enhanced result with new capabilities
    """
    # Implement enhancement while maintaining existing functionality
```
```

## 🎯 **SPECIFIC AGENT USAGE PROMPTS**

### Using PerformanceReviewer

**Prompt:**
```
When analyzing code performance:

1. Import PerformanceReviewer from interdisciplinary_reviewer
2. Use analyze_complexity() for algorithmic analysis
3. Use detect_inefficient_operations() for performance issues
4. Use generate_optimization_recommendations() for suggestions
5. Integrate results into overall risk assessment

Example:
```python
from src.core.interdisciplinary_reviewer import PerformanceReviewer

def analyze_code_performance(code: str) -> Dict[str, Any]:
    reviewer = PerformanceReviewer()
    
    complexity = reviewer.analyze_complexity(code)
    inefficiencies = reviewer.detect_inefficient_operations(code)
    recommendations = reviewer.generate_optimization_recommendations(code)
    
    return {
        "complexity": complexity,
        "inefficiencies": inefficiencies,
        "recommendations": recommendations
    }
```
```

### Using CodeHighlighter

**Prompt:**
```
When highlighting code with warnings:

1. Import CodeHighlighter from code_companion
2. Use highlight_code() with language specification
3. Check for deprecated function warnings
4. Include warning messages in output
5. Use appropriate color coding

Example:
```python
from src.core.code_companion import CodeHighlighter

def highlight_with_warnings(code: str, language: str = "python") -> HighlightedCode:
    highlighter = CodeHighlighter()
    highlighted = highlighter.highlight_code(code, language)
    
    # Check for warnings
    if highlighted.warnings:
        print("⚠️ Warnings detected:")
        for warning in highlighted.warnings:
            print(f"  - {warning}")
    
    return highlighted
```
```

### Using Batch Processing

**Prompt:**
```
When implementing batch processing:

1. Follow TrafficPlanReviewer patterns for memory optimization
2. Use gc.collect() for garbage collection
3. Monitor memory with psutil
4. Generate comprehensive JSON output
5. Include progress reporting

Example:
```python
import gc
import psutil
import json

def batch_process_files(file_list: List[str]) -> Dict[str, Any]:
    results = []
    total_files = len(file_list)
    
    for i, file_path in enumerate(file_list):
        # Process file
        result = process_single_file(file_path)
        results.append(result)
        
        # Memory optimization
        if i % 10 == 0:
            gc.collect()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"Progress: {i+1}/{total_files}, Memory: {memory_usage:.1f}MB")
    
    return {
        "metadata": {
            "total_files": total_files,
            "processed_files": len(results),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024
        },
        "results": results
    }
```
```

## 🚨 **EMERGENCY AND TROUBLESHOOTING PROMPTS**

### Debugging Agent Issues

**Prompt:**
```
When debugging agent issues:

1. **Check Logs**: Review all log files for error messages
2. **Validate Inputs**: Ensure all inputs meet validation requirements
3. **Test Isolation**: Test each agent in isolation
4. **Check Dependencies**: Verify all dependencies are installed
5. **Review Configuration**: Check configuration parameters

Debugging checklist:
- [ ] Check logs for error messages
- [ ] Validate input data format
- [ ] Test agent in isolation
- [ ] Verify dependencies
- [ ] Check configuration
- [ ] Review error handling
- [ ] Test with minimal data
```

### Performance Optimization

**Prompt:**
```
When optimizing performance:

1. **Profile Code**: Use profiling tools to identify bottlenecks
2. **Memory Analysis**: Monitor memory usage and optimize
3. **Algorithm Review**: Review algorithms for efficiency
4. **Caching**: Implement caching where appropriate
5. **Parallelization**: Consider parallel processing for large datasets

Optimization pattern:
```python
import cProfile
import pstats
import io

def profile_function(func, *args, **kwargs):
    """Profile a function and return performance statistics."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    return result, s.getvalue()
```
```

---

## 📋 **QUICK REFERENCE CHECKLIST**

### New Feature Development
- [ ] Integrate with existing agents where appropriate
- [ ] Implement comprehensive error handling
- [ ] Add CLI interface with help text
- [ ] Create unit tests following existing patterns
- [ ] Add structured logging
- [ ] Include JSON output options
- [ ] Update documentation

### Agent Integration
- [ ] Import required agents from src.core
- [ ] Follow established patterns and interfaces
- [ ] Handle errors gracefully
- [ ] Include confidence scoring where applicable
- [ ] Add progress reporting
- [ ] Test integration thoroughly

### Testing
- [ ] Follow patterns in tests/ directory
- [ ] Mock external dependencies
- [ ] Test error conditions
- [ ] Include integration tests
- [ ] Add performance benchmarks
- [ ] Test edge cases

### Documentation
- [ ] Update relevant documentation files
- [ ] Include usage examples
- [ ] Document configuration options
- [ ] Add troubleshooting section
- [ ] Update API documentation
- [ ] Include integration examples
