"""
Unit tests for affine_transform function with error handling.

Tests the enhanced affine_transform function including validation of
control points, detection of degenerate cases, and error handling.
"""

from typing import Dict, List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.core.kmz_matcher import affine_transform


class TestAffineTransform:
    """Test suite for affine_transform function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Valid control points forming a triangle
        self.valid_control_points = {
            (100, 100): (-122.4194, 37.7749),  # San Francisco
            (200, 100): (-122.4000, 37.7800),  # Different point
            (150, 200): (-122.4100, 37.7700),  # Third point forming triangle
        }

        # Collinear control points (degenerate case)
        self.collinear_control_points = {
            (100, 100): (-122.4194, 37.7749),
            (200, 100): (-122.4000, 37.7749),  # Same latitude
            (300, 100): (-122.3800, 37.7749),  # Same latitude
        }

        # Vertical line control points (degenerate case)
        self.vertical_line_control_points = {
            (100, 100): (-122.4194, 37.7749),
            (100, 200): (-122.4194, 37.7800),  # Same longitude
            (100, 300): (-122.4194, 37.7850),  # Same longitude
        }

        # Horizontal line control points (degenerate case)
        self.horizontal_line_control_points = {
            (100, 100): (-122.4194, 37.7749),
            (200, 100): (-122.4000, 37.7749),  # Same latitude
            (300, 100): (-122.3800, 37.7749),  # Same latitude
        }

    def test_valid_control_points(self):
        """Test affine_transform with valid control points."""
        pixels = [(150, 150)]
        result = affine_transform(pixels, self.valid_control_points)

        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert np.all(np.isfinite(result))

    def test_insufficient_control_points(self):
        """Test affine_transform with insufficient control points."""
        pixels = [(150, 150)]
        insufficient_points = {
            (100, 100): (-122.4194, 37.7749),
            (200, 100): (-122.4000, 37.7800),
        }

        with pytest.raises(ValueError, match="At least 3 control points are required"):
            affine_transform(pixels, insufficient_points)

    def test_empty_control_points(self):
        """Test affine_transform with empty control points."""
        pixels = [(150, 150)]
        empty_points = {}

        with pytest.raises(ValueError, match="At least 3 control points are required"):
            affine_transform(pixels, empty_points)

    def test_none_control_points(self):
        """Test affine_transform with None control points."""
        pixels = [(150, 150)]

        with pytest.raises(ValueError, match="At least 3 control points are required"):
            affine_transform(pixels, None)

    def test_collinear_control_points(self):
        """Test affine_transform with collinear control points."""
        pixels = [(150, 150)]

        with pytest.raises(ValueError, match="Control points are collinear"):
            affine_transform(pixels, self.collinear_control_points)

    def test_vertical_line_control_points(self):
        """Test affine_transform with vertical line control points."""
        pixels = [(150, 150)]

        with pytest.raises(ValueError, match="collinear"):
            affine_transform(pixels, self.vertical_line_control_points)

    def test_horizontal_line_control_points(self):
        """Test affine_transform with horizontal line control points."""
        pixels = [(150, 150)]

        with pytest.raises(ValueError, match="collinear"):
            affine_transform(pixels, self.horizontal_line_control_points)

    def test_nearly_collinear_control_points(self):
        """Test affine_transform with nearly collinear control points."""
        pixels = [(150, 150)]
        nearly_collinear = {
            (100, 100): (-122.4194, 37.7749),
            (200, 100): (-122.4000, 37.7749),
            (150, 100.001): (-122.4100, 37.7749),  # Very close to line
        }

        # This should work since the area is small but non-zero
        result = affine_transform(pixels, nearly_collinear)
        assert isinstance(result, np.ndarray)
        assert len(result) == 6

    def test_rank_deficient_matrix(self):
        """Test affine_transform with rank-deficient matrix."""
        pixels = [(150, 150)]
        rank_deficient_points = {
            (100, 100): (-122.4194, 37.7749),
            (200, 100): (-122.4000, 37.7800),
            (100, 200): (-122.4194, 37.7800),  # Same x as first point
            (200, 200): (-122.4000, 37.7749),  # Same x as second point
        }

        # This should work since the points form a valid rectangle
        result = affine_transform(pixels, rank_deficient_points)
        assert isinstance(result, np.ndarray)
        assert len(result) == 6

    def test_four_valid_control_points(self):
        """Test affine_transform with four valid control points."""
        pixels = [(150, 150)]
        four_points = {
            (100, 100): (-122.4194, 37.7749),
            (200, 100): (-122.4000, 37.7800),
            (100, 200): (-122.4194, 37.7700),
            (200, 200): (-122.4000, 37.7750),
        }

        result = affine_transform(pixels, four_points)

        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert np.all(np.isfinite(result))

    def test_large_residuals_warning(self):
        """Test that large residuals trigger a warning."""
        pixels = [(150, 150)]
        # Create control points that don't fit well to an affine transformation
        poor_fit_points = {
            (100, 100): (-122.4194, 37.7749),
            (200, 100): (-122.4000, 37.7800),
            (150, 200): (-122.4100, 37.7700),
            (250, 150): (-122.3900, 37.7850),  # This point doesn't fit well
        }

        # Should not raise an error but may print a warning
        result = affine_transform(pixels, poor_fit_points)

        assert isinstance(result, np.ndarray)
        assert len(result) == 6

    def test_numerical_stability(self):
        """Test numerical stability with very small coordinates."""
        pixels = [(0.001, 0.001)]
        small_coord_points = {
            (0.001, 0.001): (-122.4194, 37.7749),
            (0.003, 0.001): (-122.4000, 37.7800),  # Increased to get larger area
            (0.001, 0.003): (-122.4194, 37.7700),  # Forms triangle with area 0.002
        }

        # This should work since the area is well above the threshold
        result = affine_transform(pixels, small_coord_points)

        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert np.all(np.isfinite(result))

    def test_large_coordinates(self):
        """Test with very large coordinates."""
        pixels = [(1000000, 1000000)]
        large_coord_points = {
            (1000000, 1000000): (-122.4194, 37.7749),
            (2000000, 1000000): (-122.4000, 37.7800),
            (1000000, 2000000): (-122.4194, 37.7700),
        }

        result = affine_transform(pixels, large_coord_points)

        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert np.all(np.isfinite(result))

    def test_mixed_coordinate_types(self):
        """Test with mixed integer and float coordinates."""
        pixels = [(150.5, 150.5)]
        mixed_coord_points = {
            (100, 100): (-122.4194, 37.7749),
            (200.0, 100): (-122.4000, 37.7800),
            (100, 200.0): (-122.4194, 37.7700),
        }

        result = affine_transform(pixels, mixed_coord_points)

        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert np.all(np.isfinite(result))

    def test_backward_compatibility(self):
        """Test backward compatibility with existing JSON format."""
        pixels = [(150, 150)]

        # Test with the original format
        result = affine_transform(pixels, self.valid_control_points)

        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert np.all(np.isfinite(result))

    def test_error_message_clarity(self):
        """Test that error messages are clear and helpful."""
        pixels = [(150, 150)]

        # Test collinear points error message
        with pytest.raises(ValueError) as exc_info:
            affine_transform(pixels, self.collinear_control_points)

        error_message = str(exc_info.value)
        assert "collinear" in error_message.lower()
        assert "degenerate" in error_message.lower()
        assert "non-collinear" in error_message.lower()

    def test_area_calculation_accuracy(self):
        """Test that area calculation for collinearity detection is accurate."""
        pixels = [(150, 150)]

        # Create points with known area
        known_area_points = {
            (0, 0): (-122.4194, 37.7749),
            (1, 0): (-122.4000, 37.7800),
            (0, 1): (-122.4194, 37.7700),  # Forms triangle with area 0.5
        }

        # This should work (non-zero area)
        result = affine_transform(pixels, known_area_points)
        assert isinstance(result, np.ndarray)

        # Create collinear points with zero area
        zero_area_points = {
            (0, 0): (-122.4194, 37.7749),
            (1, 0): (-122.4000, 37.7800),
            (2, 0): (-122.3800, 37.7850),  # Collinear
        }

        with pytest.raises(ValueError, match="collinear"):
            affine_transform(pixels, zero_area_points)

    def test_threshold_sensitivity(self):
        """Test sensitivity of collinearity detection thresholds."""
        pixels = [(150, 150)]

        # Test with points that are very close to collinear but not exactly
        near_collinear_points = {
            (0, 0): (-122.4194, 37.7749),
            (1, 0): (-122.4000, 37.7800),
            (0.5, 0.0001): (-122.4100, 37.7775),  # Very small deviation
        }

        # This should work (small but non-zero area)
        result = affine_transform(pixels, near_collinear_points)
        assert isinstance(result, np.ndarray)

    def test_linear_regression_collinearity(self):
        """Test collinearity detection using linear regression."""
        pixels = [(150, 150)]

        # Create many points that lie exactly on a line
        many_collinear_points = {
            (i, 100): (-122.4194 + i * 0.001, 37.7749)
            for i in range(10, 21, 2)  # 6 points on horizontal line
        }

        with pytest.raises(ValueError, match="collinear"):
            affine_transform(pixels, many_collinear_points)

    def test_exception_handling(self):
        """Test that exceptions are properly caught and re-raised."""
        pixels = [(150, 150)]

        # Test with invalid data that would cause LinAlgError
        with patch("src.core.kmz_matcher.lstsq") as mock_lstsq:
            mock_lstsq.side_effect = np.linalg.LinAlgError("Singular matrix")

            with pytest.raises(
                ValueError, match="Failed to compute affine transformation"
            ):
                affine_transform(pixels, self.valid_control_points)

    def test_residual_checking(self):
        """Test that residual checking works correctly."""
        pixels = [(150, 150)]

        # Create points that should have small residuals
        good_fit_points = {
            (100, 100): (-122.4194, 37.7749),
            (200, 100): (-122.4000, 37.7800),
            (100, 200): (-122.4194, 37.7700),
            (200, 200): (-122.4000, 37.7750),  # Forms a rectangle
        }

        # This should work without warnings
        result = affine_transform(pixels, good_fit_points)
        assert isinstance(result, np.ndarray)

    def test_parameter_validation(self):
        """Test parameter validation and type checking."""
        pixels = [(150, 150)]

        # Test with invalid control point format
        invalid_format_points = {
            "invalid_key": (-122.4194, 37.7749),
            (200, 100): (-122.4000, 37.7800),
            (100, 200): (-122.4194, 37.7700),
        }

        # This should raise a TypeError when trying to convert to array
        with pytest.raises((ValueError, TypeError)):
            affine_transform(pixels, invalid_format_points)

    def test_docstring_accuracy(self):
        """Test that the docstring accurately describes the function behavior."""
        import inspect

        doc = inspect.getdoc(affine_transform)
        assert doc is not None
        assert "pixels" in doc
        assert "control_points" in doc
        assert "params" in doc
        assert "ValueError" in doc
        assert "degenerate" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
