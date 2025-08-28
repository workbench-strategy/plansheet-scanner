# Agent Prompts for Plansheet Scanner Workspace

This file contains specific agent prompts and templates for common development tasks in this workspace. Use these prompts with Cursor to get consistent, high-quality implementations.

## ML Model Development Prompts

### Feature Importance Analysis Implementation

**Prompt:**
```
Implement a function in adaptive_reviewer.py to calculate feature importance from the trained RandomForest model. 
Requirements:
- Input: model name (string), feature names (list)
- Output: JSON with feature names and importance scores sorted descending
- Add unit tests using pytest
- Ensure compatibility with existing PlanFeatureExtractor
- Handle edge cases: invalid model names, mismatched feature counts
- Include proper error handling and logging
- Follow existing code patterns and style
```

**Expected Implementation Pattern:**
```python
def calculate_feature_importance(self, model_name: str, feature_names: List[str]) -> Dict[str, float]:
    """
    Calculate feature importance from trained RandomForest model.
    
    Args:
        model_name: Name of the model to analyze (e.g., 'random_forest', 'gradient_boosting')
        feature_names: List of feature names corresponding to model features
        
    Returns:
        Dictionary with feature names and importance scores sorted in descending order
        
    Raises:
        ValueError: If model not found or not a tree-based model
        ValueError: If feature_names length doesn't match model features
        KeyError: If model_name not in self.models
    """
    # Implementation here
```

### Model Training and Evaluation

**Prompt:**
```
Add a comprehensive model evaluation function to adaptive_reviewer.py that:
- Calculates accuracy, precision, recall, F1-score
- Generates confusion matrix visualization
- Provides feature importance analysis
- Saves evaluation results to JSON
- Includes cross-validation scores
- Handles both classification and regression tasks
- Uses existing training data structure
```

### Data Preprocessing Pipeline

**Prompt:**
```
Implement a robust data preprocessing pipeline in PlanFeatureExtractor that:
- Handles missing values appropriately
- Normalizes numerical features
- Encodes categorical variables
- Detects and handles outliers
- Validates data quality
- Provides preprocessing statistics
- Saves preprocessing parameters for inference
- Follows scikit-learn pipeline patterns
```

## Testing Prompts

### Unit Test Generation

**Prompt:**
```
Create comprehensive unit tests for [FUNCTION_NAME] in [MODULE_NAME]:
- Test successful execution with valid inputs
- Test error handling with invalid inputs
- Test edge cases and boundary conditions
- Mock external dependencies appropriately
- Use pytest fixtures for common test data
- Include parameterized tests for multiple scenarios
- Test both positive and negative cases
- Ensure 90%+ code coverage
- Follow existing test patterns in tests/ directory
```

### Integration Test Setup

**Prompt:**
```
Set up integration tests for the ML pipeline that:
- Test end-to-end workflow from data loading to prediction
- Use realistic test data from test_data/ directory
- Validate model persistence and loading
- Test feedback collection and retraining
- Mock external services and APIs
- Include performance benchmarks
- Test error recovery scenarios
- Generate test reports with coverage metrics
```

## Code Quality and Refactoring

### Code Review and Optimization

**Prompt:**
```
Review and optimize [FILE_PATH] for:
- Performance bottlenecks and optimization opportunities
- Code duplication and refactoring needs
- Error handling improvements
- Type hint completeness
- Documentation quality
- Test coverage gaps
- Security vulnerabilities
- Follow PEP 8 standards
- Suggest specific improvements with code examples
```

### Refactoring Large Functions

**Prompt:**
```
Refactor the [FUNCTION_NAME] function in [FILE_PATH] to:
- Break down into smaller, focused functions
- Improve readability and maintainability
- Reduce cyclomatic complexity
- Extract common patterns into utility functions
- Add proper error handling
- Improve type hints and documentation
- Maintain backward compatibility
- Add unit tests for new functions
```

## Documentation and API Development

### API Documentation

**Prompt:**
```
Create comprehensive API documentation for [MODULE_NAME]:
- Document all public functions and classes
- Include usage examples with realistic data
- Explain parameter types and return values
- Document error conditions and exceptions
- Provide integration examples
- Include performance considerations
- Add troubleshooting section
- Follow Google docstring format
- Generate documentation that works with Sphinx
```

### README Updates

**Prompt:**
```
Update the README.md for [FEATURE_NAME] to include:
- Clear installation instructions
- Usage examples with code snippets
- Configuration options and environment variables
- Troubleshooting common issues
- Performance benchmarks
- Integration with existing systems
- API reference links
- Contributing guidelines
- Changelog for recent updates
```

## Performance and Monitoring

### Performance Optimization

**Prompt:**
```
Optimize the performance of [FUNCTION_NAME] in [FILE_PATH]:
- Profile the current implementation
- Identify bottlenecks and memory usage
- Implement vectorized operations where possible
- Add caching for expensive computations
- Optimize data structures and algorithms
- Reduce I/O operations
- Add performance monitoring and metrics
- Include benchmark comparisons
- Maintain code readability
```

### Logging and Monitoring

**Prompt:**
```
Implement comprehensive logging and monitoring for [MODULE_NAME]:
- Add structured logging with appropriate levels
- Include performance metrics and timing
- Log model training progress and metrics
- Monitor memory usage and resource consumption
- Add error tracking and alerting
- Include user activity and usage statistics
- Provide debugging information
- Configure log rotation and retention
- Follow existing logging patterns
```

## Security and Best Practices

### Security Audit

**Prompt:**
```
Perform a security audit of [MODULE_NAME]:
- Identify potential security vulnerabilities
- Review input validation and sanitization
- Check for hardcoded secrets or credentials
- Validate file handling and path operations
- Review authentication and authorization
- Check for SQL injection or code injection risks
- Validate data serialization and deserialization
- Review logging for sensitive information
- Suggest security improvements
```

### Input Validation

**Prompt:**
```
Implement comprehensive input validation for [FUNCTION_NAME]:
- Validate data types and formats
- Check for required fields and dependencies
- Sanitize user inputs
- Handle edge cases and malicious inputs
- Provide clear error messages
- Log validation failures
- Use appropriate validation libraries
- Follow existing validation patterns
- Add unit tests for validation logic
```

## Specific Implementation Examples

### RandomForest Feature Importance (Your Example)

**Complete Implementation Prompt:**
```
Implement a function in adaptive_reviewer.py to calculate feature importance from the trained RandomForest model. 
Requirements:
- Input: model name (string), feature names (list)
- Output: JSON with feature names and importance scores sorted descending
- Add unit tests using pytest
- Ensure compatibility with existing PlanFeatureExtractor
- Handle edge cases: invalid model names, mismatched feature counts
- Include proper error handling and logging
- Follow existing code patterns and style
- Add integration with existing model loading system
- Include validation for tree-based models only
- Provide both raw importance scores and normalized percentages
```

### XGBoost Integration (Your Example)

**Complete Implementation Prompt:**
```
Extend FoundationTrainer to support XGBoost as an additional model type.
Tasks:
- Add XGBoost to train_foundation_models()
- Update _save_foundation_model() to handle XGBoost
- Ensure cross-validation metrics are logged
- Maintain backward compatibility with RandomForest and GradientBoosting
- Add proper error handling for missing XGBoost dependency
- Include XGBoost-specific metadata and feature importance analysis
- Update model loading system to handle XGBoost models
- Add comprehensive unit tests for XGBoost functionality
```

### Performance Review Perspective (Your Example)

**Complete Implementation Prompt:**
```
Add a new perspective to InterdisciplinaryReviewer for 'Performance Review'.
Requirements:
- Detect nested loops and high time complexity patterns
- Assign severity levels based on O(n^2) or worse
- Include recommendations for optimization
- Integrate into overall risk score calculation
- Detect inefficient list operations (insert(0), pop(0), etc.)
- Identify memory-intensive operations
- Provide specific optimization recommendations with complexity improvements
- Include deep nesting detection and recursion analysis
- Add comprehensive unit tests for all performance analysis features
- Maintain consistency with existing reviewer patterns and interfaces
```

### Expected Test Structure:
```python
import pytest
from unittest.mock import Mock, patch
import numpy as np
from src.core.adaptive_reviewer import AdaptiveReviewer

class TestFeatureImportance:
    def test_calculate_feature_importance_success(self):
        """Test successful feature importance calculation."""
        # Test implementation
        
    def test_calculate_feature_importance_invalid_model(self):
        """Test error handling for invalid model name."""
        # Test implementation
        
    def test_calculate_feature_importance_mismatched_features(self):
        """Test error handling for mismatched feature names."""
        # Test implementation
        
    def test_calculate_feature_importance_non_tree_model(self):
        """Test error handling for non-tree-based models."""
        # Test implementation
```

## Usage Instructions

1. Copy the relevant prompt template
2. Replace placeholder text (e.g., [FUNCTION_NAME], [MODULE_NAME]) with actual values
3. Use with Cursor's AI features for consistent, high-quality implementations
4. Customize prompts based on specific project needs
5. Update prompts as the codebase evolves

## Best Practices for Using These Prompts

- Always specify the exact file path and function name
- Include specific requirements and constraints
- Mention existing patterns and compatibility needs
- Request comprehensive error handling
- Ask for unit tests and documentation
- Specify performance requirements when relevant
- Include integration requirements with existing systems
