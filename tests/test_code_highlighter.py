"""
Unit tests for CodeHighlighter class.

Tests the code highlighting capabilities including deprecated function detection,
syntax highlighting, and warning generation.
"""

import re
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from src.core.code_companion import CodeAnalyzer, CodeHighlighter, HighlightedCode


class TestCodeHighlighter:
    """Test suite for CodeHighlighter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = Mock(spec=CodeAnalyzer)
        self.analyzer.analyze_complexity.return_value = {
            "cyclomatic_complexity": 1,
            "nested_depth": 1,
            "lines_of_code": 10,
        }
        self.highlighter = CodeHighlighter(self.analyzer)

    def test_init(self):
        """Test CodeHighlighter initialization."""
        assert hasattr(self.highlighter, "highlight_patterns")
        assert "python" in self.highlighter.highlight_patterns
        assert hasattr(self.highlighter, "deprecated_patterns")
        assert "python" in self.highlighter.deprecated_patterns

    def test_highlight_code_basic(self):
        """Test basic code highlighting functionality."""
        code = """
def simple_function():
    return 42
"""
        result = self.highlighter.highlight_code(code)

        assert isinstance(result, HighlightedCode)
        assert result.code == code
        assert isinstance(result.highlights, list)
        assert isinstance(result.suggestions, list)
        assert isinstance(result.warnings, list)
        assert result.complexity_score == 1

    def test_syntax_highlights(self):
        """Test syntax-based highlighting."""
        code = """
def test_function():
    # This is a comment
    string_var = "hello world"
    number_var = 42
    return True
"""
        result = self.highlighter.highlight_code(code)

        # Check that various syntax elements are highlighted
        highlight_types = [h["type"] for h in result.highlights]
        assert "keywords" in highlight_types
        assert "comments" in highlight_types
        assert "strings" in highlight_types
        assert "numbers" in highlight_types
        assert "functions" in highlight_types

    def test_deprecated_asyncio_get_event_loop(self):
        """Test detection of deprecated asyncio.get_event_loop()."""
        code = """
import asyncio

async def main():
    loop = asyncio.get_event_loop()
    await asyncio.sleep(1)
"""
        result = self.highlighter.highlight_code(code)

        # Check for deprecated function highlight
        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) > 0

        # Check specific deprecated function
        asyncio_highlight = None
        for highlight in deprecated_highlights:
            if "asyncio.get_event_loop" in highlight["text"]:
                asyncio_highlight = highlight
                break

        assert asyncio_highlight is not None
        assert asyncio_highlight["color"] == "red"
        assert asyncio_highlight["severity"] == "high"
        assert "deprecated" in asyncio_highlight["message"].lower()
        assert "asyncio.get_running_loop" in asyncio_highlight["replacement"]

    def test_deprecated_collections_abc(self):
        """Test detection of deprecated collections imports."""
        code = """
from collections import MutableMapping, MutableSequence

class MyDict(MutableMapping):
    pass
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) > 0

        # Check for collections.abc warning
        collections_warnings = [w for w in result.warnings if "collections.abc" in w]
        assert len(collections_warnings) > 0

    def test_deprecated_urllib_functions(self):
        """Test detection of deprecated urllib functions."""
        code = """
import urllib

encoded = urllib.quote("hello world")
decoded = urllib.unquote(encoded)
params = urllib.urlencode({"key": "value"})
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert (
            len(deprecated_highlights) >= 3
        )  # Should detect quote, unquote, and urlencode

        # Check for urllib.parse warnings
        urllib_warnings = [w for w in result.warnings if "urllib.parse" in w]
        assert len(urllib_warnings) >= 3

    def test_deprecated_urllib2(self):
        """Test detection of deprecated urllib2 imports."""
        code = """
import urllib2
from urllib2 import urlopen

response = urllib2.urlopen("http://example.com")
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) > 0

        urllib2_warnings = [w for w in result.warnings if "urllib2" in w]
        assert len(urllib2_warnings) > 0

    def test_deprecated_string_constants(self):
        """Test detection of deprecated string constants."""
        code = """
import string

letters = string.letters
lowercase = string.lowercase
uppercase = string.uppercase
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) >= 3

        string_warnings = [w for w in result.warnings if "string.ascii" in w]
        assert len(string_warnings) >= 3

    def test_deprecated_imp_module(self):
        """Test detection of deprecated imp module."""
        code = """
import imp
from imp import load_source, load_module

module = imp.load_source('test', 'test.py')
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) > 0

        imp_warnings = [w for w in result.warnings if "importlib" in w]
        assert len(imp_warnings) > 0

    def test_deprecated_os_path_walk(self):
        """Test detection of deprecated os.path.walk."""
        code = """
import os

def callback(arg, dirname, fnames):
    pass

os.path.walk('/path', callback, None)
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) > 0

        walk_warnings = [w for w in result.warnings if "os.walk" in w]
        assert len(walk_warnings) > 0

    def test_deprecated_unittest_assertRaisesRegexp(self):
        """Test detection of deprecated unittest.assertRaisesRegexp."""
        code = """
import unittest

class TestCase(unittest.TestCase):
    def test_something(self):
        self.assertRaisesRegexp(ValueError, "invalid", int, "not_a_number")
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) > 0

        unittest_warnings = [w for w in result.warnings if "assertRaisesRegex" in w]
        assert len(unittest_warnings) > 0

    def test_multiple_deprecated_functions(self):
        """Test detection of multiple deprecated functions in same code."""
        code = """
import urllib
import string
import asyncio

def process_data():
    encoded = urllib.quote("data")
    letters = string.letters
    loop = asyncio.get_event_loop()
    return encoded, letters, loop
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) >= 3

        # Check that multiple deprecated function warnings are generated
        deprecated_warnings = [w for w in result.warnings if "deprecated" in w.lower()]
        assert len(deprecated_warnings) >= 3

    def test_no_deprecated_functions(self):
        """Test that code without deprecated functions has no deprecated highlights."""
        code = """
import urllib.parse
import string
import asyncio

async def process_data():
    encoded = urllib.parse.quote("data")
    letters = string.ascii_letters
    loop = asyncio.get_running_loop()
    return encoded, letters, loop
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) == 0

        deprecated_warnings = [w for w in result.warnings if "deprecated" in w.lower()]
        assert len(deprecated_warnings) == 0

    def test_highlight_color_mapping(self):
        """Test that deprecated functions get red color."""
        code = """
import asyncio

loop = asyncio.get_event_loop()
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) > 0

        for highlight in deprecated_highlights:
            assert highlight["color"] == "red"

    def test_deprecated_function_metadata(self):
        """Test that deprecated function highlights contain all required metadata."""
        code = """
import asyncio

loop = asyncio.get_event_loop()
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) > 0

        highlight = deprecated_highlights[0]
        required_fields = [
            "type",
            "start",
            "end",
            "text",
            "color",
            "message",
            "severity",
            "replacement",
            "deprecation_type",
        ]

        for field in required_fields:
            assert field in highlight
            assert highlight[field] is not None

    def test_severity_levels(self):
        """Test that different deprecated functions have appropriate severity levels."""
        code = """
import asyncio
import string
import urllib

# High severity
loop = asyncio.get_event_loop()
encoded = urllib.quote("data")

# Medium severity
letters = string.letters
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]

        severities = [h["severity"] for h in deprecated_highlights]
        assert "high" in severities
        assert "medium" in severities

    def test_replacement_suggestions(self):
        """Test that replacement suggestions are provided."""
        code = """
import asyncio
import urllib

loop = asyncio.get_event_loop()
encoded = urllib.quote("data")
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]

        for highlight in deprecated_highlights:
            assert "replacement" in highlight
            assert highlight["replacement"] is not None
            assert len(highlight["replacement"]) > 0

    def test_line_number_accuracy(self):
        """Test that deprecated function highlights have accurate positions."""
        code = """
import asyncio

def test_function():
    loop = asyncio.get_event_loop()  # Line 4
    return loop
"""
        result = self.highlighter.highlight_code(code)

        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) > 0

        highlight = deprecated_highlights[0]
        assert highlight["start"] < highlight["end"]
        assert highlight["start"] >= 0
        assert highlight["end"] <= len(code)

    def test_complexity_metrics_integration(self):
        """Test that complexity metrics are properly integrated."""
        # Mock analyzer to return specific metrics
        self.analyzer.analyze_complexity.return_value = {
            "cyclomatic_complexity": 15,
            "nested_depth": 6,
            "lines_of_code": 60,
        }

        code = """
def complex_function():
    if True:
        if False:
            for i in range(10):
                if i > 5:
                    return i
    return 0
"""
        result = self.highlighter.highlight_code(code)

        assert result.complexity_score == 15

        # Check that suggestions are generated based on complexity
        complexity_suggestions = [
            s for s in result.suggestions if "complex" in s.lower()
        ]
        assert len(complexity_suggestions) > 0

    def test_warning_generation(self):
        """Test that appropriate warnings are generated."""
        code = """
import asyncio
import urllib

# Multiple deprecated functions
loop = asyncio.get_event_loop()
encoded = urllib.quote("data")
decoded = urllib.unquote(encoded)
"""
        result = self.highlighter.highlight_code(code)

        # Should have warnings for deprecated functions
        deprecated_warnings = [w for w in result.warnings if "deprecated" in w.lower()]
        assert len(deprecated_warnings) >= 3

        # Should have multiple deprecated function warnings
        assert len(deprecated_warnings) >= 3

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty code
        result = self.highlighter.highlight_code("")
        assert isinstance(result, HighlightedCode)
        assert result.code == ""

        # Code with only comments
        comment_code = "# This is a comment\n# Another comment"
        result = self.highlighter.highlight_code(comment_code)
        assert isinstance(result, HighlightedCode)

        # Code with syntax errors (should still highlight what it can)
        invalid_code = "def invalid_function(\n    # Missing closing parenthesis"
        result = self.highlighter.highlight_code(invalid_code)
        assert isinstance(result, HighlightedCode)

    def test_language_specific_behavior(self):
        """Test that deprecated function detection only works for Python."""
        code = """
import asyncio

loop = asyncio.get_event_loop()
"""

        # Test with Python
        result_python = self.highlighter.highlight_code(code, "python")
        deprecated_python = [
            h for h in result_python.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_python) > 0

        # Test with non-Python language
        result_other = self.highlighter.highlight_code(code, "javascript")
        deprecated_other = [
            h for h in result_other.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_other) == 0

    def test_performance_with_large_code(self):
        """Test performance with larger code samples."""
        import time

        # Create large code sample with deprecated functions
        large_code = (
            """
import asyncio
import urllib
import string

def large_function():
    """
            + "\n".join(
                [f"    loop = asyncio.get_event_loop()  # Line {i}" for i in range(100)]
            )
            + """
    return True
"""
        )

        start_time = time.time()
        result = self.highlighter.highlight_code(large_code)
        end_time = time.time()

        # Should complete within reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        assert isinstance(result, HighlightedCode)

        # Should detect multiple deprecated function instances
        deprecated_highlights = [
            h for h in result.highlights if h["type"] == "deprecated_function"
        ]
        assert len(deprecated_highlights) >= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
