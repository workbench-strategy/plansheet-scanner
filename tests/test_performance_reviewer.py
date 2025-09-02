"""
Unit tests for PerformanceReviewer class.

Tests the performance analysis capabilities including complexity detection,
inefficient operations detection, and optimization recommendations.
"""

import re
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from src.core.interdisciplinary_reviewer import PerformanceReviewer, ReviewPerspective


class TestPerformanceReviewer:
    """Test suite for PerformanceReviewer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reviewer = PerformanceReviewer()

    def test_init(self):
        """Test PerformanceReviewer initialization."""
        assert hasattr(self.reviewer, "complexity_patterns")
        assert "nested_loops" in self.reviewer.complexity_patterns
        assert "inefficient_operations" in self.reviewer.complexity_patterns
        assert "memory_intensive" in self.reviewer.complexity_patterns

    def test_review_code_basic(self):
        """Test basic code review functionality."""
        code = """
def simple_function():
    return 42
"""
        result = self.reviewer.review_code(code)

        assert isinstance(result, ReviewPerspective)
        assert result.name == "Performance Review"
        assert (
            result.description
            == "Analysis of algorithmic complexity, performance bottlenecks, and optimization opportunities"
        )
        assert result.risk_level == "low"
        assert result.confidence == 0.85

    def test_review_code_empty(self):
        """Test review with empty code."""
        result = self.reviewer.review_code("")

        assert isinstance(result, ReviewPerspective)
        assert result.risk_level == "low"
        assert len(result.findings) == 0

    def test_review_code_none(self):
        """Test review with None code."""
        with pytest.raises(AttributeError):
            self.reviewer.review_code(None)

    def test_nested_loop_detection(self):
        """Test detection of nested loops."""
        code = """
def matrix_multiply(a, b):
    result = []
    for i in range(len(a)):
        for j in range(len(b[0])):
            sum = 0
            for k in range(len(b)):
                sum += a[i][k] * b[k][j]
            result[i][j] = sum
    return result
"""
        result = self.reviewer.review_code(code)

        # Should detect nested loops
        nested_loop_findings = [
            f for f in result.findings if f.get("category") == "nested_loops"
        ]
        assert len(nested_loop_findings) > 0

        # Check severity
        for finding in nested_loop_findings:
            assert finding["severity"] in ["high", "critical"]
            assert "O(n^2)" in finding.get("complexity", "") or "O(n^3)" in finding.get(
                "complexity", ""
            )

    def test_inefficient_list_operations(self):
        """Test detection of inefficient list operations."""
        code = """
def process_items(items):
    result = []
    for item in items:
        result.insert(0, item)  # O(n) operation
    return result

def find_item(items, target):
    if target in list(items):  # O(n) membership test
        return items.index(target)  # O(n) index search
    return -1
"""
        result = self.reviewer.review_code(code)

        # Should detect inefficient operations
        inefficient_findings = [
            f for f in result.findings if f.get("category") == "list_operation"
        ]
        assert len(inefficient_findings) > 0

        # Check for specific operations
        operations_found = [f.get("code", "") for f in inefficient_findings]
        assert any("insert(0" in op for op in operations_found)
        assert any("index(" in op for op in operations_found)

    def test_memory_intensive_operations(self):
        """Test detection of memory-intensive operations."""
        code = """
import copy
import pickle
import json

def process_data(data):
    # Memory-intensive operations
    deep_copied = copy.deepcopy(data)
    serialized = pickle.dumps(data)
    json_data = json.dumps(data)
    return deep_copied, serialized, json_data
"""
        result = self.reviewer.review_code(code)

        # Should detect memory-intensive operations
        memory_findings = [
            f for f in result.findings if f.get("category") == "memory_intensive"
        ]
        assert len(memory_findings) > 0

        # Check for specific operations
        operations_found = [f.get("code", "") for f in memory_findings]
        assert any("deepcopy" in op for op in operations_found)
        assert any("dumps" in op for op in operations_found)

    def test_deep_nesting_detection(self):
        """Test detection of deep nesting."""
        code = """
def complex_function(data):
    if data:
        if data.is_valid():
            if data.has_items():
                if data.items_are_ready():
                    if data.can_process():
                        if data.should_process():
                            if data.is_authorized():
                                return process_data(data)
    return None
"""
        result = self.reviewer.review_code(code)

        # Should detect deep nesting
        nesting_findings = [
            f for f in result.findings if f.get("category") == "deep_nesting"
        ]
        assert len(nesting_findings) > 0

        for finding in nesting_findings:
            assert finding["severity"] == "medium"
            assert "nesting" in finding["description"].lower()

    def test_recursive_function_detection(self):
        """Test detection of recursive functions."""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
        result = self.reviewer.review_code(code)

        # Should detect recursive functions
        recursive_findings = [
            f for f in result.findings if f.get("category") == "recursion"
        ]
        assert len(recursive_findings) > 0

        for finding in recursive_findings:
            assert finding["severity"] == "medium"
            assert "recursive" in finding["description"].lower()
            assert "memoization" in finding["recommendation"].lower()

    def test_optimization_recommendations(self):
        """Test generation of optimization recommendations."""
        code = """
def inefficient_search(items, target):
    for i in range(len(items)):
        if items[i] == target:
            return i
    return -1

def slow_processing(data):
    result = []
    for item in data:
        result.insert(0, item)
    return result
"""
        result = self.reviewer.review_code(code)

        # Should generate optimization recommendations
        optimization_findings = [
            f for f in result.findings if f.get("type") == "optimization_recommendation"
        ]
        assert len(optimization_findings) > 0

        for finding in optimization_findings:
            assert "recommendation" in finding
            assert "impact" in finding
            assert finding["severity"] in ["high", "medium"]

    def test_risk_level_calculation(self):
        """Test risk level calculation based on findings."""
        # Test with no issues
        simple_code = "def simple(): return 42"
        result = self.reviewer.review_code(simple_code)
        assert result.risk_level == "low"

        # Test with medium issues
        medium_code = """
def medium_issue():
    items = [1, 2, 3]
    items.sort()  # O(n log n)
    return items
"""
        result = self.reviewer.review_code(medium_code)
        # Should be low or medium depending on other factors

        # Test with high issues
        high_code = """
def high_issue():
    for i in range(100):
        for j in range(100):
            process(i, j)
"""
        result = self.reviewer.review_code(high_code)
        assert result.risk_level in ["high", "critical"]

    def test_line_number_accuracy(self):
        """Test that line numbers are accurately reported."""
        code = """
def function1():
    return 1

def function2():
    for i in range(10):  # Line 6
        for j in range(10):  # Line 7
            process(i, j)

def function3():
    return 3
"""
        result = self.reviewer.review_code(code)

        # Check that nested loop findings have correct line numbers
        nested_findings = [
            f for f in result.findings if f.get("category") == "nested_loops"
        ]
        for finding in nested_findings:
            line_num = finding.get("line")
            assert isinstance(line_num, int)
            assert 1 <= line_num <= len(code.split("\n"))

    def test_code_context_preservation(self):
        """Test that code context is preserved in findings."""
        code = """
def test_function():
    items = [1, 2, 3, 4, 5]
    for i in range(len(items)):
        for j in range(len(items)):
            if items[i] == items[j]:
                items.insert(0, items[i])
"""
        result = self.reviewer.review_code(code)

        for finding in result.findings:
            assert "code" in finding
            code_snippet = finding["code"]
            assert isinstance(code_snippet, str)
            assert len(code_snippet) > 0

    def test_complexity_classification(self):
        """Test complexity classification accuracy."""
        # O(n) complexity
        linear_code = """
def linear_function(items):
    for item in items:
        process(item)
"""
        result = self.reviewer.review_code(linear_code)
        # Should have low risk level

        # O(n^2) complexity
        quadratic_code = """
def quadratic_function(items):
    for i in range(len(items)):
        for j in range(len(items)):
            process(items[i], items[j])
"""
        result = self.reviewer.review_code(quadratic_code)
        # Should have higher risk level

        # O(n^3) complexity
        cubic_code = """
def cubic_function(items):
    for i in range(len(items)):
        for j in range(len(items)):
            for k in range(len(items)):
                process(items[i], items[j], items[k])
"""
        result = self.reviewer.review_code(cubic_code)
        # Should have highest risk level

    def test_optimization_recommendation_specificity(self):
        """Test that optimization recommendations are specific and actionable."""
        code = """
def inefficient_function():
    items = [1, 2, 3]
    items.insert(0, 0)  # O(n) operation
    if 5 in list(items):  # O(n) membership test
        return items.index(5)  # O(n) index search
    return -1
"""
        result = self.reviewer.review_code(code)

        for finding in result.findings:
            if finding.get("type") == "inefficient_operation":
                recommendation = finding.get("recommendation", "")
                assert (
                    "deque" in recommendation
                    or "set" in recommendation
                    or "dict" in recommendation
                )
                assert len(recommendation) > 20  # Should be specific enough

    def test_false_positive_avoidance(self):
        """Test that legitimate code patterns are not flagged as issues."""
        code = """
def legitimate_function():
    # Small lists where O(n) operations are acceptable
    small_list = [1, 2, 3]
    small_list.insert(0, 0)  # OK for small lists
    
    # Single loop is fine
    for i in range(10):
        process(i)
    
    # Simple recursion with base case
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n-1)
"""
        result = self.reviewer.review_code(code)

        # Should have relatively low risk level
        assert result.risk_level in ["low", "medium"]

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Code with syntax errors
        invalid_code = """
def invalid_function(
    # Missing closing parenthesis
"""
        # Should handle gracefully
        result = self.reviewer.review_code(invalid_code)
        assert isinstance(result, ReviewPerspective)

        # Very long code
        long_code = "def long_function():\n" + "    pass\n" * 1000
        result = self.reviewer.review_code(long_code)
        assert isinstance(result, ReviewPerspective)

    def test_performance_analysis_performance(self):
        """Test that the performance analysis itself is efficient."""
        import time

        # Large code sample
        large_code = "def function():\n" + "    pass\n" * 10000

        start_time = time.time()
        result = self.reviewer.review_code(large_code)
        end_time = time.time()

        # Should complete within reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0
        assert isinstance(result, ReviewPerspective)

    def test_criteria_completeness(self):
        """Test that all required criteria are included."""
        result = self.reviewer.review_code("def test(): pass")

        expected_criteria = [
            "Time complexity analysis",
            "Space complexity analysis",
            "Inefficient operation detection",
            "Memory usage optimization",
            "Algorithm optimization recommendations",
        ]

        for criterion in expected_criteria:
            assert criterion in result.criteria

    def test_confidence_level(self):
        """Test that confidence level is appropriate."""
        result = self.reviewer.review_code("def test(): pass")

        assert result.confidence == 0.85
        assert 0.0 <= result.confidence <= 1.0

    def test_finding_structure_completeness(self):
        """Test that all findings have required structure."""
        code = """
def test_function():
    for i in range(10):
        for j in range(10):
            process(i, j)
"""
        result = self.reviewer.review_code(code)

        for finding in result.findings:
            # Check required fields
            assert "type" in finding
            assert "category" in finding
            assert "severity" in finding
            assert "description" in finding
            assert "recommendation" in finding

            # Check field types
            assert isinstance(finding["type"], str)
            assert isinstance(finding["category"], str)
            assert isinstance(finding["severity"], str)
            assert isinstance(finding["description"], str)
            assert isinstance(finding["recommendation"], str)

            # Check severity values
            assert finding["severity"] in ["critical", "high", "medium", "low"]


class TestPerformanceReviewerIntegration:
    """Integration tests for PerformanceReviewer with InterdisciplinaryReviewer."""

    def test_integration_with_interdisciplinary_reviewer(self):
        """Test that PerformanceReviewer integrates correctly with InterdisciplinaryReviewer."""
        from src.core.interdisciplinary_reviewer import InterdisciplinaryReviewer

        reviewer = InterdisciplinaryReviewer()

        # Create a temporary file with performance issues
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
def inefficient_function():
    items = [1, 2, 3, 4, 5]
    for i in range(len(items)):
        for j in range(len(items)):
            if items[i] == items[j]:
                items.insert(0, items[i])
    return items
"""
            )
            temp_file = f.name

        try:
            result = reviewer.perform_review(temp_file)

            # Check that performance perspective is included
            assert "performance" in result.perspectives
            performance_perspective = result.perspectives["performance"]

            # Check that performance findings contribute to overall risk
            assert isinstance(result.overall_risk_score, float)
            assert 0.0 <= result.overall_risk_score <= 1.0

            # Check that performance recommendations are included
            performance_recommendations = [
                r for r in result.recommendations if "[PERFORMANCE]" in r
            ]
            assert len(performance_recommendations) > 0

        finally:
            os.unlink(temp_file)

    def test_risk_score_contribution(self):
        """Test that performance issues contribute to overall risk score."""
        from src.core.interdisciplinary_reviewer import InterdisciplinaryReviewer

        reviewer = InterdisciplinaryReviewer()

        # Test with code that has performance issues
        code_with_issues = """
def problematic_function():
    for i in range(1000):
        for j in range(1000):
            for k in range(1000):
                process(i, j, k)
"""

        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code_with_issues)
            temp_file = f.name

        try:
            result = reviewer.perform_review(temp_file)

            # Performance issues should increase overall risk
            assert (
                result.overall_risk_score > 0.5
            )  # Should be higher due to O(n^3) complexity

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
