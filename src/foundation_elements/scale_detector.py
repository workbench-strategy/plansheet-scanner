"""
Scale Detector

Detects and analyzes drawing scales in engineering drawings.
This is essential for understanding dimensions and measurements.
"""

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ScaleDetection:
    """Result of scale detection."""

    detected: bool
    scale_factor: float  # Drawing units per real units
    scale_text: str  # Original scale text
    scale_type: str  # "graphic", "text", "ratio"
    confidence: float
    position: Tuple[int, int]  # (x, y) coordinates
    size: Tuple[int, int]  # (width, height)
    units: str  # "feet", "meters", "inches", etc.
    graphic_scale_length: Optional[float] = None  # Length of graphic scale bar


class ScaleDetector:
    """
    Advanced scale detection system.

    Features:
    - Text-based scale detection (OCR)
    - Graphic scale bar detection
    - Multiple unit system support
    - Scale factor calculation
    - Confidence scoring
    """

    def __init__(self):
        """Initialize the scale detector."""
        # Scale text patterns
        self.scale_patterns = {
            "imperial": [
                r'(\d+(?:\.\d+)?)\s*["\']?\s*=\s*(\d+(?:\.\d+)?)\s*(ft|feet|foot)',
                r"(\d+(?:\.\d+)?)\s*inch(?:es)?\s*=\s*(\d+(?:\.\d+)?)\s*(ft|feet|foot)",
                r'scale\s*[:\s]*(\d+(?:\.\d+)?)\s*["\']?\s*=\s*(\d+(?:\.\d+)?)\s*(ft|feet|foot)',
                r'(\d+(?:\.\d+)?)\s*["\']?\s*=\s*(\d+(?:\.\d+)?)\s*(yd|yard|yards)',
                r'(\d+(?:\.\d+)?)\s*["\']?\s*=\s*(\d+(?:\.\d+)?)\s*(mi|mile|miles)',
            ],
            "metric": [
                r"(\d+(?:\.\d+)?)\s*(mm|millimeter|millimeters)\s*=\s*(\d+(?:\.\d+)?)\s*(m|meter|meters)",
                r"(\d+(?:\.\d+)?)\s*(cm|centimeter|centimeters)\s*=\s*(\d+(?:\.\d+)?)\s*(m|meter|meters)",
                r"scale\s*[:\s]*(\d+(?:\.\d+)?)\s*(mm|cm|m)\s*=\s*(\d+(?:\.\d+)?)\s*(m|meter|meters)",
                r"(\d+(?:\.\d+)?)\s*(mm|cm|m)\s*=\s*(\d+(?:\.\d+)?)\s*(km|kilometer|kilometers)",
            ],
            "ratio": [
                r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)",
                r"scale\s*[:\s]*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)",
            ],
        }

        # Common scale values
        self.common_scales = {
            "architectural": [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32, 64],
            "engineering": [
                1 / 10,
                1 / 20,
                1 / 30,
                1 / 40,
                1 / 50,
                1 / 60,
                1 / 80,
                1 / 100,
                1 / 200,
            ],
            "metric": [1 / 50, 1 / 100, 1 / 200, 1 / 500, 1 / 1000, 1 / 2000, 1 / 5000],
        }

        # Detection parameters
        self.min_confidence = 0.5
        self.min_scale_bar_length = 50  # pixels
        self.max_scale_bar_length = 500  # pixels

        logger.info("ScaleDetector initialized")

    def detect_scale(self, image: np.ndarray) -> ScaleDetection:
        """
        Detect scale in the given image.

        Args:
            image: Input image as numpy array (grayscale or BGR)

        Returns:
            ScaleDetection object with detection results
        """
        if image is None:
            return ScaleDetection(
                detected=False,
                scale_factor=1.0,
                scale_text="",
                scale_type="unknown",
                confidence=0.0,
                position=(0, 0),
                size=(0, 0),
                units="unknown",
            )

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Text-based scale detection
        text_result = self._detect_scale_by_text(gray)

        # 2. Graphic scale bar detection
        graphic_result = self._detect_graphic_scale(gray)

        # 3. Pattern-based detection
        pattern_result = self._detect_scale_patterns(gray)

        # Combine results
        best_result = self._combine_scale_results(
            text_result, graphic_result, pattern_result
        )

        return best_result

    def _detect_scale_by_text(self, gray_image: np.ndarray) -> Optional[ScaleDetection]:
        """Detect scale by finding scale text using OCR."""
        # This would integrate with OCR (pytesseract)
        # For now, return None - would need OCR implementation
        return None

    def _detect_graphic_scale(self, gray_image: np.ndarray) -> Optional[ScaleDetection]:
        """Detect graphic scale bars."""
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150)

        # Find lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=self.min_scale_bar_length,
            maxLineGap=10,
        )

        if lines is None:
            return None

        best_result = None
        best_confidence = 0.0

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line properties
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Check if line length is in reasonable range for scale bar
            if self.min_scale_bar_length <= length <= self.max_scale_bar_length:
                # Look for tick marks or divisions
                confidence = self._analyze_scale_bar_confidence(
                    gray_image, x1, y1, x2, y2
                )

                if confidence > best_confidence and confidence > self.min_confidence:
                    best_confidence = confidence
                    best_result = ScaleDetection(
                        detected=True,
                        scale_factor=1.0,  # Would need calibration
                        scale_text="Graphic scale bar",
                        scale_type="graphic",
                        confidence=confidence,
                        position=(min(x1, x2), min(y1, y2)),
                        size=(abs(x2 - x1), abs(y2 - y1)),
                        units="unknown",
                        graphic_scale_length=length,
                    )

        return best_result

    def _detect_scale_patterns(
        self, gray_image: np.ndarray
    ) -> Optional[ScaleDetection]:
        """Detect scale patterns in the image."""
        # This would look for common scale patterns
        # For now, return None - would need pattern recognition
        return None

    def _analyze_scale_bar_confidence(
        self, gray_image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> float:
        """Analyze confidence that a line represents a scale bar."""
        # Extract region around the line
        margin = 20
        x_min = max(0, min(x1, x2) - margin)
        x_max = min(gray_image.shape[1], max(x1, x2) + margin)
        y_min = max(0, min(y1, y2) - margin)
        y_max = min(gray_image.shape[0], max(y1, y2) + margin)

        roi = gray_image[y_min:y_max, x_min:x_max]

        # Look for tick marks or divisions
        # Apply edge detection to find perpendicular lines
        edges = cv2.Canny(roi, 30, 100)

        # Count edge pixels (tick marks)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Higher confidence for more structured patterns
        confidence = min(1.0, edge_density * 10)

        return confidence

    def _combine_scale_results(
        self,
        text_result: Optional[ScaleDetection],
        graphic_result: Optional[ScaleDetection],
        pattern_result: Optional[ScaleDetection],
    ) -> ScaleDetection:
        """Combine multiple scale detection results."""
        results = [
            r for r in [text_result, graphic_result, pattern_result] if r is not None
        ]

        if not results:
            return ScaleDetection(
                detected=False,
                scale_factor=1.0,
                scale_text="",
                scale_type="unknown",
                confidence=0.0,
                position=(0, 0),
                size=(0, 0),
                units="unknown",
            )

        # Return the result with highest confidence
        best_result = max(results, key=lambda r: r.confidence)
        return best_result

    def parse_scale_text(self, scale_text: str) -> Optional[ScaleDetection]:
        """
        Parse scale text to extract scale information.

        Args:
            scale_text: Text containing scale information

        Returns:
            ScaleDetection object if parsing successful
        """
        scale_text = scale_text.strip()

        # Try different patterns
        for unit_system, patterns in self.scale_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, scale_text, re.IGNORECASE)
                if match:
                    try:
                        if unit_system == "ratio":
                            drawing_units = float(match.group(1))
                            real_units = float(match.group(2))
                            scale_factor = drawing_units / real_units
                            units = "ratio"
                        else:
                            drawing_units = float(match.group(1))
                            real_units = float(match.group(2))
                            units = (
                                match.group(3) if len(match.groups()) > 2 else "unknown"
                            )
                            scale_factor = drawing_units / real_units

                        return ScaleDetection(
                            detected=True,
                            scale_factor=scale_factor,
                            scale_text=scale_text,
                            scale_type="text",
                            confidence=0.9,
                            position=(0, 0),
                            size=(0, 0),
                            units=units,
                        )
                    except (ValueError, ZeroDivisionError):
                        continue

        return None

    def validate_scale_factor(self, scale_factor: float) -> bool:
        """
        Validate if a scale factor is reasonable.

        Args:
            scale_factor: Scale factor to validate

        Returns:
            True if scale factor is reasonable
        """
        # Check against common scales
        for scale_list in self.common_scales.values():
            for common_scale in scale_list:
                # Allow 10% tolerance
                if abs(scale_factor - common_scale) / common_scale < 0.1:
                    return True

        # Check if scale factor is in reasonable range
        if 0.001 <= scale_factor <= 1000:
            return True

        return False

    def detect_from_pdf(self, pdf_path: str, page_number: int = 0) -> ScaleDetection:
        """
        Detect scale from a PDF page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-based)

        Returns:
            ScaleDetection object
        """
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            if page_number >= doc.page_count:
                logger.error(f"Page number {page_number} out of range")
                return ScaleDetection(
                    detected=False,
                    scale_factor=1.0,
                    scale_text="",
                    scale_type="unknown",
                    confidence=0.0,
                    position=(0, 0),
                    size=(0, 0),
                    units="unknown",
                )

            # Get page
            page = doc.load_page(page_number)

            # Convert to image
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )  # 2x scale for better detection
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))

            # Convert to numpy array
            image_array = np.array(pil_image)

            # Detect scale
            result = self.detect_scale(image_array)

            doc.close()
            return result

        except Exception as e:
            logger.error(f"Error detecting scale from PDF: {e}")
            return ScaleDetection(
                detected=False,
                scale_factor=1.0,
                scale_text="",
                scale_type="unknown",
                confidence=0.0,
                position=(0, 0),
                size=(0, 0),
                units="unknown",
            )

    def calculate_real_distance(
        self, pixel_distance: float, scale_detection: ScaleDetection
    ) -> float:
        """
        Calculate real-world distance from pixel distance.

        Args:
            pixel_distance: Distance in pixels
            scale_detection: Scale detection result

        Returns:
            Real-world distance in scale units
        """
        if not scale_detection.detected:
            return pixel_distance  # Return pixel distance if no scale detected

        # Convert pixel distance to real distance
        real_distance = pixel_distance / scale_detection.scale_factor

        return real_distance


def main():
    """Test the scale detector."""
    import argparse

    parser = argparse.ArgumentParser(description="Scale Detector")
    parser.add_argument("input", help="Input image or PDF file")
    parser.add_argument("--page", type=int, default=0, help="PDF page number (0-based)")
    parser.add_argument("--test-text", help="Test scale text parsing")

    args = parser.parse_args()

    # Initialize detector
    detector = ScaleDetector()

    # Test text parsing if provided
    if args.test_text:
        result = detector.parse_scale_text(args.test_text)
        print(f"Scale Text Parsing Results:")
        print(f"  Text: {args.test_text}")
        if result.detected:
            print(f"  Detected: {result.detected}")
            print(f"  Scale Factor: {result.scale_factor}")
            print(f"  Units: {result.units}")
            print(f"  Type: {result.scale_type}")
        else:
            print(f"  No scale detected")
        return

    # Detect scale
    if args.input.lower().endswith(".pdf"):
        result = detector.detect_from_pdf(args.input, args.page)
    else:
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        result = detector.detect_scale(image)

    # Print results
    print(f"Scale Detection Results:")
    print(f"  Detected: {result.detected}")
    if result.detected:
        print(f"  Scale Factor: {result.scale_factor}")
        print(f"  Scale Text: {result.scale_text}")
        print(f"  Type: {result.scale_type}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Position: {result.position}")
        print(f"  Size: {result.size}")
        print(f"  Units: {result.units}")
        if result.graphic_scale_length:
            print(f"  Graphic Scale Length: {result.graphic_scale_length:.1f} pixels")


if __name__ == "__main__":
    main()
