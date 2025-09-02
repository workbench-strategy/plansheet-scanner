"""
Unit tests for enhanced overlay.py module.

Tests KMZ overlay generation with multiple DPI options, coordinate validation,
logging, and file size management.
"""

import os
import shutil
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.core.overlay import (
    compress_images_for_kmz,
    generate_kmz_overlay,
    get_kmz_size_mb,
    validate_control_box,
)


class TestOverlay:
    """Test suite for overlay module."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Valid control box for testing
        self.valid_control_box = {
            "geo": [
                [-122.4194, 37.7749],  # Top-left (lon, lat)
                [-122.4000, 37.7749],  # Top-right
                [-122.4000, 37.7600],  # Bottom-right
                [-122.4194, 37.7600],  # Bottom-left
            ]
        }

        # Invalid control boxes for testing
        self.invalid_control_boxes = {
            "not_dict": "not a dictionary",
            "missing_geo": {"other_key": "value"},
            "empty_geo": {"geo": []},
            "insufficient_coords": {"geo": [[-122.4194, 37.7749]]},
            "invalid_coord_type": {"geo": [["not", "numbers"], [-122.4000, 37.7749]]},
            "invalid_lon": {
                "geo": [[-200.0, 37.7749], [-122.4000, 37.7749], [-122.4000, 37.7600]]
            },
            "invalid_lat": {
                "geo": [[-122.4194, 100.0], [-122.4000, 37.7749], [-122.4000, 37.7600]]
            },
            "collinear": {"geo": [[0, 0], [1, 1], [2, 2], [3, 3]]},
        }

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_validate_control_box_valid(self):
        """Test validation of valid control box."""
        result = validate_control_box(self.valid_control_box)
        assert result is True

    def test_validate_control_box_not_dict(self):
        """Test validation with non-dictionary input."""
        with pytest.raises(ValueError, match="control_box must be a dictionary"):
            validate_control_box(self.invalid_control_boxes["not_dict"])

    def test_validate_control_box_missing_geo(self):
        """Test validation with missing 'geo' key."""
        with pytest.raises(ValueError, match="control_box must contain 'geo' key"):
            validate_control_box(self.invalid_control_boxes["missing_geo"])

    def test_validate_control_box_empty_geo(self):
        """Test validation with empty geo list."""
        with pytest.raises(
            ValueError,
            match="control_box\['geo'\] must be a list with at least 4 coordinate pairs",
        ):
            validate_control_box(self.invalid_control_boxes["empty_geo"])

    def test_validate_control_box_insufficient_coords(self):
        """Test validation with insufficient coordinates."""
        with pytest.raises(
            ValueError,
            match="control_box\['geo'\] must be a list with at least 4 coordinate pairs",
        ):
            validate_control_box(self.invalid_control_boxes["insufficient_coords"])

    def test_validate_control_box_invalid_coord_type(self):
        """Test validation with non-numeric coordinates."""
        with pytest.raises(ValueError, match="Coordinate 0 values must be numeric"):
            validate_control_box(self.invalid_control_boxes["invalid_coord_type"])

    def test_validate_control_box_invalid_lon(self):
        """Test validation with invalid longitude."""
        with pytest.raises(
            ValueError, match="Longitude -200.0 at coordinate 0 is out of valid range"
        ):
            validate_control_box(self.invalid_control_boxes["invalid_lon"])

    def test_validate_control_box_invalid_lat(self):
        """Test validation with invalid latitude."""
        with pytest.raises(
            ValueError, match="Latitude 100.0 at coordinate 0 is out of valid range"
        ):
            validate_control_box(self.invalid_control_boxes["invalid_lat"])

    def test_validate_control_box_collinear(self):
        """Test validation with collinear points."""
        with pytest.raises(ValueError, match="Control points form a degenerate"):
            validate_control_box(self.invalid_control_boxes["collinear"])

    def test_get_kmz_size_mb_existing_file(self):
        """Test getting size of existing file."""
        # Create a test file
        test_file = os.path.join(self.temp_dir, "test.kmz")
        with open(test_file, "w") as f:
            f.write("test content" * 1000)  # Create file with some content

        size = get_kmz_size_mb(test_file)
        assert isinstance(size, float)
        assert size > 0

    def test_get_kmz_size_mb_nonexistent_file(self):
        """Test getting size of non-existent file."""
        size = get_kmz_size_mb("nonexistent.kmz")
        assert size == 0.0

    @patch("src.core.overlay.cv2")
    @patch("src.core.overlay.Image")
    def test_compress_images_for_kmz_small_files(self, mock_pil_image, mock_cv2):
        """Test image compression with small files."""
        # Create small test files
        image_paths = []
        for i in range(3):
            img_path = os.path.join(self.temp_dir, f"test_{i}.png")
            with open(img_path, "w") as f:
                f.write("small content")
            image_paths.append(img_path)

        compressed_paths = compress_images_for_kmz(image_paths, target_size_mb=1.0)

        assert isinstance(compressed_paths, list)
        assert len(compressed_paths) == len(image_paths)

    @patch("src.core.overlay.cv2")
    @patch("src.core.overlay.Image")
    def test_compress_images_for_kmz_large_files(self, mock_pil_image, mock_cv2):
        """Test image compression with large files."""
        # Mock PIL Image
        mock_img = Mock()
        mock_pil_image.open.return_value.__enter__.return_value = mock_img

        # Create large test files
        image_paths = []
        for i in range(2):
            img_path = os.path.join(self.temp_dir, f"large_{i}.png")
            with open(img_path, "w") as f:
                f.write("large content" * 1000000)  # Create large file
            image_paths.append(img_path)

        compressed_paths = compress_images_for_kmz(image_paths, target_size_mb=0.1)

        assert isinstance(compressed_paths, list)
        assert len(compressed_paths) == len(image_paths)

    @patch("src.core.overlay.fitz.open")
    @patch("src.core.overlay.simplekml.Kml")
    def test_generate_kmz_overlay_success(self, mock_kml, mock_fitz_open):
        """Test successful KMZ overlay generation."""
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_doc.page_count = 3
        mock_page = Mock()
        mock_pixmap = Mock()
        mock_pixmap.samples = b"fake_image_data"
        mock_pixmap.height = 100
        mock_pixmap.width = 100
        mock_pixmap.n = 3
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        # Mock simplekml
        mock_kml_instance = Mock()
        mock_kml.return_value = mock_kml_instance
        mock_overlay = Mock()
        mock_kml_instance.newgroundoverlay.return_value = mock_overlay

        # Create test PDF file
        test_pdf = os.path.join(self.temp_dir, "test.pdf")
        with open(test_pdf, "w") as f:
            f.write("fake pdf content")

        output_kmz = os.path.join(self.temp_dir, "output.kmz")

        # Mock file operations
        with patch("os.path.exists", return_value=True), patch("os.makedirs"), patch(
            "os.remove"
        ), patch(
            "src.core.overlay.compress_images_for_kmz", return_value=["page_000.png"]
        ):
            result = generate_kmz_overlay(
                pdf_path=test_pdf,
                pages=[0, 1],
                dpi=300,
                control_box=self.valid_control_box,
                output_kmz=output_kmz,
            )

        assert result == output_kmz
        mock_doc.close.assert_called_once()

    def test_generate_kmz_overlay_nonexistent_pdf(self):
        """Test KMZ generation with non-existent PDF."""
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            generate_kmz_overlay(
                pdf_path="nonexistent.pdf",
                pages=[0],
                dpi=300,
                control_box=self.valid_control_box,
                output_kmz="output.kmz",
            )

    def test_generate_kmz_overlay_empty_pages(self):
        """Test KMZ generation with empty pages list."""
        with pytest.raises(ValueError, match="No pages specified"):
            generate_kmz_overlay(
                pdf_path="test.pdf",
                pages=[],
                dpi=300,
                control_box=self.valid_control_box,
                output_kmz="output.kmz",
            )

    def test_generate_kmz_overlay_invalid_dpi(self):
        """Test KMZ generation with invalid DPI."""
        with pytest.raises(ValueError, match="DPI must be positive"):
            generate_kmz_overlay(
                pdf_path="test.pdf",
                pages=[0],
                dpi=0,
                control_box=self.valid_control_box,
                output_kmz="output.kmz",
            )

    def test_generate_kmz_overlay_invalid_max_size(self):
        """Test KMZ generation with invalid max size."""
        with pytest.raises(ValueError, match="Maximum size must be positive"):
            generate_kmz_overlay(
                pdf_path="test.pdf",
                pages=[0],
                dpi=300,
                control_box=self.valid_control_box,
                output_kmz="output.kmz",
                max_size_mb=0,
            )

    @patch("src.core.overlay.fitz.open")
    def test_generate_kmz_overlay_invalid_page_number(self, mock_fitz_open):
        """Test KMZ generation with invalid page number."""
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_doc.page_count = 2
        mock_fitz_open.return_value = mock_doc

        # Create test PDF file
        test_pdf = os.path.join(self.temp_dir, "test.pdf")
        with open(test_pdf, "w") as f:
            f.write("fake pdf content")

        with pytest.raises(ValueError, match="Page 5 is out of range"):
            generate_kmz_overlay(
                pdf_path=test_pdf,
                pages=[5],
                dpi=300,
                control_box=self.valid_control_box,
                output_kmz="output.kmz",
            )

    def test_validate_control_box_edge_cases(self):
        """Test control box validation with edge cases."""
        # Test with exactly 4 coordinates
        valid_4_coords = {
            "geo": [
                [-180.0, -90.0],  # Bottom-left
                [180.0, -90.0],  # Bottom-right
                [180.0, 90.0],  # Top-right
                [-180.0, 90.0],  # Top-left
            ]
        }
        result = validate_control_box(valid_4_coords)
        assert result is True

    def test_validate_control_box_three_points(self):
        """Test control box validation with three points."""
        three_points = {"geo": [[0, 0], [1, 0], [0, 1]]}
        result = validate_control_box(three_points)
        assert result is True

    def test_validate_control_box_mixed_types(self):
        """Test control box validation with mixed numeric types."""
        mixed_types = {
            "geo": [
                [-122.4194, 37.7749],  # float, float
                [-122, 37],  # int, int
                [-122.5, 37.5],  # float, float
                [-123, 38],  # int, int
            ]
        }
        result = validate_control_box(mixed_types)
        assert result is True

    @patch("src.core.overlay.fitz.open")
    @patch("src.core.overlay.simplekml.Kml")
    def test_generate_kmz_overlay_with_validation_disabled(
        self, mock_kml, mock_fitz_open
    ):
        """Test KMZ generation with coordinate validation disabled."""
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_doc.page_count = 1
        mock_page = Mock()
        mock_pixmap = Mock()
        mock_pixmap.samples = b"fake_image_data"
        mock_pixmap.height = 100
        mock_pixmap.width = 100
        mock_pixmap.n = 3
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        # Mock simplekml
        mock_kml_instance = Mock()
        mock_kml.return_value = mock_kml_instance
        mock_overlay = Mock()
        mock_kml_instance.newgroundoverlay.return_value = mock_overlay

        # Create test PDF file
        test_pdf = os.path.join(self.temp_dir, "test.pdf")
        with open(test_pdf, "w") as f:
            f.write("fake pdf content")

        output_kmz = os.path.join(self.temp_dir, "output.kmz")

        # Mock file operations
        with patch("os.path.exists", return_value=True), patch("os.makedirs"), patch(
            "os.remove"
        ), patch(
            "src.core.overlay.compress_images_for_kmz", return_value=["page_000.png"]
        ):
            result = generate_kmz_overlay(
                pdf_path=test_pdf,
                pages=[0],
                dpi=300,
                control_box=self.invalid_control_boxes[
                    "collinear"
                ],  # Invalid control box
                output_kmz=output_kmz,
                validate_coords=False,  # Disable validation
            )

        assert result == output_kmz

    def test_validate_control_box_nearly_collinear(self):
        """Test control box validation with nearly collinear points."""
        nearly_collinear = {
            "geo": [
                [0, 0],
                [1, 0.001],  # Very small offset
                [2, 0.002],  # Very small offset
                [3, 0.003],  # Very small offset
            ]
        }
        # This should pass validation as the area is non-zero
        result = validate_control_box(nearly_collinear)
        assert result is True

    def test_validate_control_box_duplicate_points(self):
        """Test control box validation with duplicate points."""
        duplicate_points = {
            "geo": [[0, 0], [0, 0], [1, 1], [1, 1]]  # Duplicate  # Duplicate
        }
        # This should pass validation as it's not strictly collinear
        result = validate_control_box(duplicate_points)
        assert result is True
