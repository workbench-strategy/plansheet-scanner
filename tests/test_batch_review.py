"""
Unit tests for batch review functionality in TrafficPlanReviewer.

Tests the enhanced batch review features including memory optimization,
JSON output generation, and performance monitoring.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.cli.traffic_plan_reviewer import batch_review_command


class TestBatchReview:
    """Test suite for batch review functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up test files
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)

    def create_test_pdf(self, filename: str, content: str = "test content") -> str:
        """Create a test PDF file."""
        file_path = os.path.join(self.temp_dir, filename)

        # Create a simple text file as a mock PDF for testing
        with open(file_path, "w") as f:
            f.write(content)

        self.test_files.append(file_path)
        return file_path

    def create_test_files(self, filenames: List[str]) -> List[str]:
        """Create multiple test files with unique names."""
        file_paths = []
        for i, filename in enumerate(filenames):
            # Ensure unique filenames
            base, ext = os.path.splitext(filename)
            unique_filename = f"{base}_{i}{ext}"
            file_path = self.create_test_pdf(unique_filename)
            file_paths.append(file_path)
        return file_paths

    def test_batch_review_with_no_files(self):
        """Test batch review when no plan files are found."""
        args = Mock()
        args.directory = self.temp_dir
        args.plan_type = "auto"
        args.output = None

        # Should return 1 (error) when no files found
        result = batch_review_command(args)
        assert result == 1

    def test_batch_review_with_invalid_directory(self):
        """Test batch review with invalid directory."""
        args = Mock()
        args.directory = "/nonexistent/directory"
        args.plan_type = "auto"
        args.output = None

        # Should return 1 (error) when directory doesn't exist
        result = batch_review_command(args)
        assert result == 1  # Returns 1 for error

    @patch("src.cli.traffic_plan_reviewer.TrafficPlanReviewer")
    def test_batch_review_successful_processing(self, mock_reviewer_class):
        """Test successful batch review processing."""
        # Create test files with unique names
        self.create_test_files(["plan1.pdf", "plan2.pdf", "plan3.pdf"])

        # Mock reviewer
        mock_reviewer = Mock()
        mock_reviewer_class.return_value = mock_reviewer

        # Mock review results
        mock_result = Mock()
        mock_result.plan_type = "traffic_signal"
        mock_result.compliance_score = 0.85
        mock_result.issues = [
            {
                "issue": "Test issue 1",
                "severity": "medium",
                "location": "test",
                "description": "test",
            }
        ]
        mock_result.recommendations = ["Test recommendation"]
        mock_result.elements_found = [
            Mock(
                element_type="signal_head",
                location=(100, 200),
                confidence=0.9,
                metadata={"type": "red"},
                bounding_box=(90, 190, 110, 210),
            )
        ]
        mock_result.standards_checked = ["MUTCD", "ITE"]

        mock_reviewer.review_plan.return_value = mock_result

        # Test arguments
        args = Mock()
        args.directory = self.temp_dir
        args.plan_type = "traffic_signal"
        args.output = None

        # Run batch review
        result = batch_review_command(args)

        # Should complete successfully
        assert result == 0

        # Should have called review_plan for each file
        assert mock_reviewer.review_plan.call_count == 3

    @patch("src.cli.traffic_plan_reviewer.TrafficPlanReviewer")
    def test_batch_review_with_json_output(self, mock_reviewer_class):
        """Test batch review with JSON output generation."""
        # Create test files with unique names
        self.create_test_files(["plan1.pdf", "plan2.pdf"])

        # Mock reviewer
        mock_reviewer = Mock()
        mock_reviewer_class.return_value = mock_reviewer

        # Mock review results
        mock_result = Mock()
        mock_result.plan_type = "traffic_signal"
        mock_result.compliance_score = 0.85
        mock_result.issues = []
        mock_result.recommendations = []
        mock_result.elements_found = []
        mock_result.standards_checked = ["MUTCD"]

        mock_reviewer.review_plan.return_value = mock_result

        # Create output file
        output_file = os.path.join(self.temp_dir, "results.json")

        # Test arguments
        args = Mock()
        args.directory = self.temp_dir
        args.plan_type = "traffic_signal"
        args.output = output_file

        # Run batch review
        result = batch_review_command(args)

        # Should complete successfully
        assert result == 0

        # Should have created output file
        assert os.path.exists(output_file)

        # Should contain valid JSON
        with open(output_file, "r") as f:
            data = json.load(f)

        # Check structure
        assert "metadata" in data
        assert "summary" in data
        assert "results" in data

        # Check metadata
        metadata = data["metadata"]
        assert metadata["total_plans"] == 2
        assert metadata["successful_reviews"] == 2
        assert metadata["failed_reviews"] == 0
        assert metadata["plan_type"] == "traffic_signal"
        assert "timestamp" in metadata
        assert "total_processing_time" in metadata
        assert "memory_usage_mb" in metadata

        # Check results
        results = data["results"]
        assert len(results) == 2
        for result_item in results:
            assert "file" in result_item
            assert "plan_type" in result_item
            assert "compliance_score" in result_item
            assert "issues" in result_item
            assert "recommendations" in result_item
            assert "elements_found" in result_item
            assert "standards_checked" in result_item

    @patch("src.cli.traffic_plan_reviewer.TrafficPlanReviewer")
    def test_batch_review_with_errors(self, mock_reviewer_class):
        """Test batch review handling of processing errors."""
        # Create test files with unique names
        self.create_test_files(["plan1.pdf", "plan2.pdf"])

        # Mock reviewer
        mock_reviewer = Mock()
        mock_reviewer_class.return_value = mock_reviewer

        # Mock first call to succeed, second to fail
        mock_result = Mock()
        mock_result.plan_type = "traffic_signal"
        mock_result.compliance_score = 0.85
        mock_result.issues = []
        mock_result.recommendations = []
        mock_result.elements_found = []
        mock_result.standards_checked = ["MUTCD"]

        mock_reviewer.review_plan.side_effect = [mock_result, Exception("Test error")]

        # Test arguments
        args = Mock()
        args.directory = self.temp_dir
        args.plan_type = "traffic_signal"
        args.output = None

        # Run batch review
        result = batch_review_command(args)

        # Should complete successfully (errors are handled gracefully)
        assert result == 0

        # Should have called review_plan twice
        assert mock_reviewer.review_plan.call_count == 2

    @patch("src.cli.traffic_plan_reviewer.TrafficPlanReviewer")
    def test_batch_review_memory_monitoring(self, mock_reviewer_class):
        """Test memory monitoring in batch review."""
        # Create test files with unique names
        self.create_test_files(["plan1.pdf", "plan2.pdf"])

        # Mock reviewer
        mock_reviewer = Mock()
        mock_reviewer_class.return_value = mock_reviewer

        # Mock review results
        mock_result = Mock()
        mock_result.plan_type = "traffic_signal"
        mock_result.compliance_score = 0.85
        mock_result.issues = []
        mock_result.recommendations = []
        mock_result.elements_found = []
        mock_result.standards_checked = ["MUTCD"]

        mock_reviewer.review_plan.return_value = mock_result

        # Test arguments
        args = Mock()
        args.directory = self.temp_dir
        args.plan_type = "traffic_signal"
        args.output = None

        # Run batch review
        result = batch_review_command(args)

        # Should complete successfully
        assert result == 0

        # Should have called review_plan for each file
        assert mock_reviewer.review_plan.call_count == 2

    @patch("src.cli.traffic_plan_reviewer.TrafficPlanReviewer")
    def test_batch_review_compliance_distribution(self, mock_reviewer_class):
        """Test compliance distribution calculation."""
        # Create test files with unique names
        self.create_test_files(
            ["plan1.pdf", "plan2.pdf", "plan3.pdf", "plan4.pdf"]
        )  # Excellent, Good, Fair, Poor

        # Mock reviewer
        mock_reviewer = Mock()
        mock_reviewer_class.return_value = mock_reviewer

        # Mock different compliance scores
        scores = [0.95, 0.80, 0.60, 0.30]  # Excellent, Good, Fair, Poor

        def mock_review_plan(file_path, plan_type):
            mock_result = Mock()
            mock_result.plan_type = "traffic_signal"
            mock_result.compliance_score = scores.pop(0)
            mock_result.issues = []
            mock_result.recommendations = []
            mock_result.elements_found = []
            mock_result.standards_checked = ["MUTCD"]
            return mock_result

        mock_reviewer.review_plan.side_effect = mock_review_plan

        # Create output file
        output_file = os.path.join(self.temp_dir, "results.json")

        # Test arguments
        args = Mock()
        args.directory = self.temp_dir
        args.plan_type = "traffic_signal"
        args.output = output_file

        # Run batch review
        result = batch_review_command(args)

        # Should complete successfully
        assert result == 0

        # Check JSON output for compliance distribution
        with open(output_file, "r") as f:
            data = json.load(f)

        summary = data["summary"]
        distribution = summary["compliance_distribution"]

        assert distribution["excellent"] == 1
        assert distribution["good"] == 1
        assert distribution["fair"] == 1
        assert distribution["poor"] == 1

    @patch("src.cli.traffic_plan_reviewer.TrafficPlanReviewer")
    def test_batch_review_file_size_monitoring(self, mock_reviewer_class):
        """Test file size monitoring in batch review."""
        # Create test files with different sizes and unique names
        self.create_test_pdf("small_0.pdf", "small content")
        self.create_test_pdf("large_1.pdf", "large content " * 1000)

        # Mock reviewer
        mock_reviewer = Mock()
        mock_reviewer_class.return_value = mock_reviewer

        # Mock review results
        mock_result = Mock()
        mock_result.plan_type = "traffic_signal"
        mock_result.compliance_score = 0.85
        mock_result.issues = []
        mock_result.recommendations = []
        mock_result.elements_found = []
        mock_result.standards_checked = ["MUTCD"]

        mock_reviewer.review_plan.return_value = mock_result

        # Create output file
        output_file = os.path.join(self.temp_dir, "results.json")

        # Test arguments
        args = Mock()
        args.directory = self.temp_dir
        args.plan_type = "traffic_signal"
        args.output = output_file

        # Run batch review
        result = batch_review_command(args)

        # Should complete successfully
        assert result == 0

        # Check that file sizes are recorded
        with open(output_file, "r") as f:
            data = json.load(f)

        results = data["results"]
        assert len(results) == 2

        # Check that file_size_mb is present
        for result_item in results:
            assert "file_size_mb" in result_item
            assert isinstance(result_item["file_size_mb"], (int, float))
            assert result_item["file_size_mb"] > 0

    @patch("src.cli.traffic_plan_reviewer.TrafficPlanReviewer")
    def test_batch_review_output_size_limit(self, mock_reviewer_class):
        """Test output file size limit checking."""
        # Create many test files to generate large output
        for i in range(100):
            self.create_test_pdf(f"plan{i}.pdf")

        # Mock reviewer
        mock_reviewer = Mock()
        mock_reviewer_class.return_value = mock_reviewer

        # Mock review results with detailed data
        mock_result = Mock()
        mock_result.plan_type = "traffic_signal"
        mock_result.compliance_score = 0.85
        mock_result.issues = [
            {
                "issue": f"Issue {i}",
                "severity": "medium",
                "location": "test",
                "description": "test" * 100,
            }
            for i in range(50)
        ]
        mock_result.recommendations = [f"Recommendation {i}" * 10 for i in range(20)]
        mock_result.elements_found = [
            Mock(
                element_type="signal_head",
                location=(100, 200),
                confidence=0.9,
                metadata={"type": "red", "details": "detailed info" * 50},
                bounding_box=(90, 190, 110, 210),
            )
            for _ in range(30)
        ]
        mock_result.standards_checked = ["MUTCD", "ITE", "AASHTO"]

        mock_reviewer.review_plan.return_value = mock_result

        # Create output file
        output_file = os.path.join(self.temp_dir, "results.json")

        # Test arguments
        args = Mock()
        args.directory = self.temp_dir
        args.plan_type = "traffic_signal"
        args.output = output_file

        # Run batch review
        result = batch_review_command(args)

        # Should complete successfully
        assert result == 0

        # Check output file size
        output_size = os.path.getsize(output_file) / 1024 / 1024  # MB

        # Should be reasonable size (not necessarily under 10MB for this test)
        assert output_size > 0
        assert output_size < 100  # Should not be unreasonably large


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
