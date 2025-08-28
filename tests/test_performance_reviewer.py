"""
Unit tests for PerformanceReviewer functionality.
"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.interdisciplinary_reviewer import PerformanceReviewer


class TestPerformanceReviewer:
    """Test cases for PerformanceReviewer functionality."""
    
    @pytest.fixture
    def performance_reviewer(self):
        """Create PerformanceReviewer instance for testing."""
        return PerformanceReviewer()
    
    def test_nested_loops_detection(self, performance_reviewer):
        """Test detection of nested loops."""
        code_with_nested_loops = """
def process_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] *= 2
    return matrix
"""
        
        perspective = performance_reviewer.review_code(code_with_nested_loops)
        
        # Check that nested loops are detected
        nested_loop_findings = [f for f in perspective.findings if f['category'] == 'nested_loops']
        assert len(nested_loop_findings) > 0
        
        # Check severity and complexity
        finding = nested_loop_findings[0]
        assert finding['severity'] == 'high'
        assert finding['complexity'] == 'O(n^2) or worse'
        assert 'nested loop' in finding['description'].lower()
    
    def test_inefficient_list_operations(self, performance_reviewer):
        """Test detection of inefficient list operations."""
        code_with_inefficient_ops = """
def process_data(data_list):
    result = []
    for item in data_list:
        result.insert(0, item)  # Inefficient operation
    return result

def search_item(items, target):
    if target in list(items):  # Inefficient membership test
        return True
    return False
"""
        
        perspective = performance_reviewer.review_code(code_with_inefficient_ops)
        
        # Check for inefficient operations
        inefficient_findings = [f for f in perspective.findings if f['category'] == 'list_operation']
        assert len(inefficient_findings) >= 2  # Should find both insert(0) and in list()
        
        # Check specific operations
        insert_findings = [f for f in inefficient_findings if 'insert(0' in f['code']]
        assert len(insert_findings) > 0
        assert insert_findings[0]['severity'] == 'high'
        
        membership_findings = [f for f in inefficient_findings if 'in list(' in f['code']]
        assert len(membership_findings) > 0
    
    def test_quadratic_operations_detection(self, performance_reviewer):
        """Test detection of quadratic operations."""
        code_with_quadratic_ops = """
def create_matrix(n):
    matrix = []
    for i in range(n):
        matrix.append([])  # Potentially quadratic
    return matrix

def process_range(n):
    return list(range(n, n*2))  # Range to list conversion
"""
        
        perspective = performance_reviewer.review_code(code_with_quadratic_ops)
        
        # Check for quadratic operations
        quadratic_findings = [f for f in perspective.findings if f['category'] == 'quadratic_operation']
        assert len(quadratic_findings) > 0
        
        # Check complexity notation
        for finding in quadratic_findings:
            assert finding['complexity'] == 'O(n^2)'
            assert finding['severity'] == 'medium'
    
    def test_deep_nesting_detection(self, performance_reviewer):
        """Test detection of deeply nested structures."""
        code_with_deep_nesting = """
def complex_function(data):
    if data:
        for item in data:
            if item > 0:
                for subitem in item:
                    if subitem:
                        for subsubitem in subitem:
                            if subsubitem:
                                for final_item in subsubitem:
                                    print(final_item)
"""
        
        perspective = performance_reviewer.review_code(code_with_deep_nesting)
        
        # Check for deep nesting
        nesting_findings = [f for f in perspective.findings if f['category'] == 'deep_nesting']
        assert len(nesting_findings) > 0
        
        # Check that line numbers are captured
        for finding in nesting_findings:
            assert 'line' in finding
            assert finding['severity'] == 'medium'
    
    def test_memory_intensive_operations(self, performance_reviewer):
        """Test detection of memory-intensive operations."""
        code_with_memory_ops = """
import pickle
import json

def save_data(data):
    return pickle.dumps(data)  # Memory intensive

def serialize_data(data):
    return json.dumps(data)  # Memory intensive
"""
        
        perspective = performance_reviewer.review_code(code_with_memory_ops)
        
        # Check for memory issues
        memory_findings = [f for f in perspective.findings if f['category'] == 'memory_intensive']
        assert len(memory_findings) >= 2  # Should find both pickle.dumps and json.dumps
        
        # Check severity and recommendations
        for finding in memory_findings:
            assert finding['severity'] == 'medium'
            assert 'streaming' in finding['recommendation'].lower() or 'chunked' in finding['recommendation'].lower()
    
    def test_optimization_recommendations(self, performance_reviewer):
        """Test generation of optimization recommendations."""
        code_with_issues = """
def inefficient_function(data):
    result = []
    for item in data:
        result.insert(0, item)  # Inefficient
    for item in data:
        if item in list(data):  # Inefficient
            result.append(item)
    return result
"""
        
        perspective = performance_reviewer.review_code(code_with_issues)
        
        # Check for optimization recommendations
        optimization_findings = [f for f in perspective.findings if f['type'] == 'optimization_recommendation']
        assert len(optimization_findings) > 0
        
        # Check that recommendations include specific suggestions
        for finding in optimization_findings:
            assert 'deque' in finding['recommendation'] or 'set' in finding['recommendation']
            assert 'impact' in finding
    
    def test_risk_level_determination(self, performance_reviewer):
        """Test risk level determination based on findings."""
        # Code with high severity issues
        high_risk_code = """
def high_risk_function(data):
    for i in data:
        for j in data:
            for k in data:
                result.insert(0, i)  # Multiple high severity issues
"""
        
        perspective = performance_reviewer.review_code(high_risk_code)
        assert perspective.risk_level == 'high'
        
        # Code with medium severity issues
        medium_risk_code = """
def medium_risk_function(data):
    result = []
    for item in data:
        result.append([])  # Medium severity
    return result
"""
        
        perspective = performance_reviewer.review_code(medium_risk_code)
        assert perspective.risk_level == 'medium'
        
        # Code with no issues
        low_risk_code = """
def low_risk_function(data):
    result = []
    for item in data:
        result.append(item)
    return result
"""
        
        perspective = performance_reviewer.review_code(low_risk_code)
        assert perspective.risk_level == 'low'
    
    def test_specific_optimization_recommendations(self, performance_reviewer):
        """Test specific optimization recommendations for different operations."""
        # Test insert(0) recommendation
        recommendation = performance_reviewer._get_optimization_recommendation('result.insert(0, item)')
        assert 'deque' in recommendation
        assert 'O(1)' in recommendation
        
        # Test pop(0) recommendation
        recommendation = performance_reviewer._get_optimization_recommendation('result.pop(0)')
        assert 'deque' in recommendation
        assert 'O(1)' in recommendation
        
        # Test membership test recommendation
        recommendation = performance_reviewer._get_optimization_recommendation('item in list(data)')
        assert 'set' in recommendation
        assert 'O(1)' in recommendation
    
    def test_complexity_analysis_integration(self, performance_reviewer):
        """Test that complexity analysis integrates with overall review."""
        complex_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
"""
        
        perspective = performance_reviewer.review_code(complex_code)
        
        # Check that multiple types of findings are generated
        categories = set(finding['category'] for finding in perspective.findings)
        assert 'nested_loops' in categories
        
        # Check that recommendations are generated
        optimization_findings = [f for f in perspective.findings if f['type'] == 'optimization_recommendation']
        assert len(optimization_findings) > 0
        
        # Check perspective metadata
        assert perspective.name == "Performance Review"
        assert "complexity" in perspective.description.lower()
        assert len(perspective.criteria) >= 5
        assert perspective.confidence > 0.8
    
    def test_edge_cases(self, performance_reviewer):
        """Test edge cases and boundary conditions."""
        # Empty code
        perspective = performance_reviewer.review_code("")
        assert perspective.risk_level == 'low'
        assert len(perspective.findings) == 0
        
        # Code with only comments
        comment_code = """
# This is a comment
# Another comment
"""
        perspective = performance_reviewer.review_code(comment_code)
        assert perspective.risk_level == 'low'
        
        # Code with single line
        single_line_code = "result = [1, 2, 3]"
        perspective = performance_reviewer.review_code(single_line_code)
        assert perspective.risk_level == 'low'
    
    def test_recursion_detection(self, performance_reviewer):
        """Test detection of recursive patterns."""
        recursive_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # Recursive without memoization
"""
        
        perspective = performance_reviewer.review_code(recursive_code)
        
        # Check for recursion findings
        recursion_findings = [f for f in perspective.findings if f['category'] == 'recursion']
        assert len(recursion_findings) > 0
        
        # Check recommendation includes memoization
        finding = recursion_findings[0]
        assert 'memoization' in finding['recommendation'].lower()
        assert 'O(2^n)' in finding['complexity']


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
