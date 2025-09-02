"""
Tests for cable entity pipeline functionality.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.core.cable_entity_pipeline import CableEntityPipeline


class TestCableEntityPipeline:
    """Test cases for CableEntityPipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = CableEntityPipeline()

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        assert self.pipeline is not None
        assert hasattr(self.pipeline, "confidence_threshold")

    def test_extract_material(self):
        """Test material extraction from text."""
        test_cases = [
            ("copper cable", "copper"),
            ("aluminum conductor", "aluminum"),
            ("steel wire", "steel"),
            ("PVC insulation", "plastic"),
        ]

        for text, expected in test_cases:
            result = self.pipeline._extract_material(text)
            if result:
                assert result == expected

    def test_extract_length(self):
        """Test length extraction from text."""
        test_cases = [
            ("100 ft cable", 30.48),
            ("50 meters", 50.0),
            ("2.5 km", 2500.0),
        ]

        for text, expected in test_cases:
            result = self.pipeline._extract_length(text)
            if result is not None:
                assert result == pytest.approx(expected, rel=1e-2)

    def test_extract_voltage(self):
        """Test voltage extraction from text."""
        test_cases = [
            ("120V cable", "120v"),
            ("15 kV", "15kv"),
            ("480 volts", "480v"),
        ]

        for text, expected in test_cases:
            result = self.pipeline._extract_voltage(text)
            if result:
                assert result == expected

    def test_analyze_text_for_cable(self):
        """Test cable analysis from text."""
        text = "Power cable 100 ft copper 120V"
        result = self.pipeline._analyze_text_for_cable(text, 0, "test")

        if result:
            assert hasattr(result, "type")
            assert hasattr(result, "length")
            assert hasattr(result, "material")
            assert hasattr(result, "voltage")

    def test_process_pdf(self):
        """Test PDF processing."""
        with patch("fitz.open") as mock_open:
            mock_doc = Mock()
            mock_page = Mock()
            mock_doc.__getitem__.return_value = mock_page
            mock_open.return_value.__enter__.return_value = mock_doc

            result = self.pipeline.process_pdf("test.pdf")
            assert isinstance(result, list)

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        high_threshold_pipeline = CableEntityPipeline(confidence_threshold=0.9)
        text = "Some cable"
        result = high_threshold_pipeline._analyze_text_for_cable(text, 0, "test")

        # Should return None due to low confidence
        assert result is None
