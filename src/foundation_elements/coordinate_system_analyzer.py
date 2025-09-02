"""
Coordinate System Analyzer

Analyzes coordinate systems, grid lines, and spatial references in engineering drawings.
This is essential for understanding spatial relationships and measurements.
"""

import io
import json
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
class GridLine:
    """Represents a grid line."""

    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    orientation: str  # "horizontal", "vertical"
    grid_number: Optional[str] = None
    confidence: float = 1.0


@dataclass
class CoordinateSystem:
    """Represents a coordinate system."""

    system_type: str  # "grid", "state_plane", "local", "geographic"
    origin: Tuple[float, float]
    orientation: float  # degrees
    units: str  # "feet", "meters", "inches"
    grid_spacing: Optional[float] = None
    grid_lines: List[GridLine] = None
    confidence: float = 1.0


@dataclass
class CoordinateAnalysis:
    """Result of coordinate system analysis."""

    detected: bool
    coordinate_system: Optional[CoordinateSystem]
    grid_lines: List[GridLine]
    total_grid_lines: int
    confidence: float
    spatial_references: List[str]


class CoordinateSystemAnalyzer:
    """
    Advanced coordinate system analysis system.

    Features:
    - Grid line detection
    - Coordinate system identification
    - Spatial reference extraction
    - Orientation analysis
    - Unit system detection
    """

    def __init__(self):
        """Initialize the coordinate system analyzer."""
        # Grid line detection parameters
        self.min_line_length = 100  # pixels
        self.max_line_gap = 10  # pixels
        self.angle_tolerance = 5.0  # degrees

        # Coordinate system patterns
        self.coordinate_patterns = {
            "state_plane": [
                r"\bSTATE PLANE\b",
                r"\bSPCS\b",
                r"\bNAD83\b",
                r"\bNAD27\b",
            ],
            "local": [r"\bLOCAL\b", r"\bSITE\b", r"\bPROJECT\b", r"\bSTATION\b"],
            "geographic": [r"\bLAT\b", r"\bLON\b", r"\bLATITUDE\b", r"\bLONGITUDE\b"],
        }

        logger.info("CoordinateSystemAnalyzer initialized")

    def analyze_coordinate_system(self, image: np.ndarray) -> CoordinateAnalysis:
        """
        Analyze coordinate system in the given image.

        Args:
            image: Input image as numpy array (grayscale or BGR)

        Returns:
            CoordinateAnalysis object with analysis results
        """
        if image is None:
            return CoordinateAnalysis(
                detected=False,
                coordinate_system=None,
                grid_lines=[],
                total_grid_lines=0,
                confidence=0.0,
                spatial_references=[],
            )

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Detect grid lines
        grid_lines = self._detect_grid_lines(gray)

        # 2. Analyze coordinate system
        coordinate_system = self._analyze_coordinate_system(gray, grid_lines)

        # 3. Extract spatial references
        spatial_references = self._extract_spatial_references(gray)

        # 4. Calculate overall confidence
        total_grid_lines = len(grid_lines)
        confidence = self._calculate_analysis_confidence(grid_lines, coordinate_system)

        return CoordinateAnalysis(
            detected=total_grid_lines > 0 or coordinate_system is not None,
            coordinate_system=coordinate_system,
            grid_lines=grid_lines,
            total_grid_lines=total_grid_lines,
            confidence=confidence,
            spatial_references=spatial_references,
        )

    def _detect_grid_lines(self, gray_image: np.ndarray) -> List[GridLine]:
        """Detect grid lines in the image."""
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150)

        # Find lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return []

        grid_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line properties
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Normalize angle
            angle = angle % 180
            if angle < 0:
                angle += 180

            # Determine orientation
            if (
                abs(angle) < self.angle_tolerance
                or abs(angle - 90) < self.angle_tolerance
            ):
                orientation = (
                    "horizontal" if abs(angle) < self.angle_tolerance else "vertical"
                )

                # Calculate confidence based on line properties
                confidence = self._calculate_grid_line_confidence(
                    gray_image, x1, y1, x2, y2
                )

                if confidence > 0.5:  # Minimum confidence threshold
                    grid_line = GridLine(
                        start_point=(float(x1), float(y1)),
                        end_point=(float(x2), float(y2)),
                        orientation=orientation,
                        confidence=confidence,
                    )
                    grid_lines.append(grid_line)

        return grid_lines

    def _calculate_grid_line_confidence(
        self, gray_image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> float:
        """Calculate confidence that a line is a grid line."""
        # Extract region around the line
        margin = 20
        x_min = max(0, min(x1, x2) - margin)
        x_max = min(gray_image.shape[1], max(x1, x2) + margin)
        y_min = max(0, min(y1, y2) - margin)
        y_max = min(gray_image.shape[0], max(y1, y2) + margin)

        roi = gray_image[y_min:y_max, x_min:x_max]

        # Analyze region properties
        # 1. Edge density (grid lines should have consistent edges)
        edges = cv2.Canny(roi, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # 2. Line straightness (grid lines should be straight)
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        straightness = min(1.0, line_length / 200)  # Longer lines get higher score

        # 3. Consistency (grid lines should be consistent in appearance)
        variance = np.var(roi)
        consistency = 1.0 - min(1.0, variance / 1000)

        # Combine factors
        confidence = edge_density * 0.4 + straightness * 0.3 + consistency * 0.3

        return confidence

    def _analyze_coordinate_system(
        self, gray_image: np.ndarray, grid_lines: List[GridLine]
    ) -> Optional[CoordinateSystem]:
        """Analyze the coordinate system based on grid lines and text."""
        if not grid_lines:
            return None

        # Analyze grid line patterns
        horizontal_lines = [
            line for line in grid_lines if line.orientation == "horizontal"
        ]
        vertical_lines = [line for line in grid_lines if line.orientation == "vertical"]

        # Calculate grid spacing
        grid_spacing = self._calculate_grid_spacing(horizontal_lines, vertical_lines)

        # Determine coordinate system type
        system_type = self._determine_coordinate_system_type(gray_image)

        # Estimate origin (top-left corner)
        origin = (0.0, 0.0)

        # Determine units (would need text analysis)
        units = "feet"  # Default assumption

        # Calculate orientation
        orientation = self._calculate_system_orientation(
            horizontal_lines, vertical_lines
        )

        return CoordinateSystem(
            system_type=system_type,
            origin=origin,
            orientation=orientation,
            units=units,
            grid_spacing=grid_spacing,
            grid_lines=grid_lines,
            confidence=0.8,
        )

    def _calculate_grid_spacing(
        self, horizontal_lines: List[GridLine], vertical_lines: List[GridLine]
    ) -> Optional[float]:
        """Calculate grid spacing from line positions."""
        if not horizontal_lines and not vertical_lines:
            return None

        # Calculate spacing from horizontal lines
        if len(horizontal_lines) > 1:
            y_positions = [line.start_point[1] for line in horizontal_lines]
            y_positions.sort()
            horizontal_spacing = np.mean(np.diff(y_positions))
        else:
            horizontal_spacing = None

        # Calculate spacing from vertical lines
        if len(vertical_lines) > 1:
            x_positions = [line.start_point[0] for line in vertical_lines]
            x_positions.sort()
            vertical_spacing = np.mean(np.diff(x_positions))
        else:
            vertical_spacing = None

        # Return average spacing
        if horizontal_spacing and vertical_spacing:
            return (horizontal_spacing + vertical_spacing) / 2
        elif horizontal_spacing:
            return horizontal_spacing
        elif vertical_spacing:
            return vertical_spacing
        else:
            return None

    def _determine_coordinate_system_type(self, gray_image: np.ndarray) -> str:
        """Determine the type of coordinate system."""
        # This would integrate with OCR to find coordinate system references
        # For now, return default type
        return "grid"

    def _calculate_system_orientation(
        self, horizontal_lines: List[GridLine], vertical_lines: List[GridLine]
    ) -> float:
        """Calculate the orientation of the coordinate system."""
        # For now, assume standard orientation (0 degrees)
        # This would be enhanced with more sophisticated analysis
        return 0.0

    def _extract_spatial_references(self, gray_image: np.ndarray) -> List[str]:
        """Extract spatial references from the image."""
        # This would integrate with OCR to find coordinate references
        # For now, return empty list
        return []

    def _calculate_analysis_confidence(
        self, grid_lines: List[GridLine], coordinate_system: Optional[CoordinateSystem]
    ) -> float:
        """Calculate overall confidence of the analysis."""
        if not grid_lines:
            return 0.0

        # Base confidence on number and quality of grid lines
        line_confidence = np.mean([line.confidence for line in grid_lines])
        line_count_factor = min(1.0, len(grid_lines) / 10)

        # System confidence
        system_confidence = coordinate_system.confidence if coordinate_system else 0.0

        # Combine factors
        confidence = (
            line_confidence * 0.6 + line_count_factor * 0.2 + system_confidence * 0.2
        )

        return confidence

    def detect_from_pdf(
        self, pdf_path: str, page_number: int = 0
    ) -> CoordinateAnalysis:
        """
        Analyze coordinate system from a PDF page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-based)

        Returns:
            CoordinateAnalysis object
        """
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            if page_number >= doc.page_count:
                logger.error(f"Page number {page_number} out of range")
                return CoordinateAnalysis(
                    detected=False,
                    coordinate_system=None,
                    grid_lines=[],
                    total_grid_lines=0,
                    confidence=0.0,
                    spatial_references=[],
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

            # Analyze coordinate system
            result = self.analyze_coordinate_system(image_array)

            doc.close()
            return result

        except Exception as e:
            logger.error(f"Error analyzing coordinate system from PDF: {e}")
            return CoordinateAnalysis(
                detected=False,
                coordinate_system=None,
                grid_lines=[],
                total_grid_lines=0,
                confidence=0.0,
                spatial_references=[],
            )

    def save_coordinate_analysis(
        self, coordinate_analysis: CoordinateAnalysis, output_path: str
    ) -> bool:
        """
        Save coordinate analysis results to JSON file.

        Args:
            coordinate_analysis: Coordinate analysis results
            output_path: Path to save JSON file

        Returns:
            True if saved successfully
        """
        try:
            # Convert to dictionary
            data = {
                "detected": coordinate_analysis.detected,
                "total_grid_lines": coordinate_analysis.total_grid_lines,
                "confidence": coordinate_analysis.confidence,
                "spatial_references": coordinate_analysis.spatial_references,
                "grid_lines": [
                    {
                        "start_point": line.start_point,
                        "end_point": line.end_point,
                        "orientation": line.orientation,
                        "grid_number": line.grid_number,
                        "confidence": line.confidence,
                    }
                    for line in coordinate_analysis.grid_lines
                ],
            }

            # Add coordinate system if available
            if coordinate_analysis.coordinate_system:
                data["coordinate_system"] = {
                    "system_type": coordinate_analysis.coordinate_system.system_type,
                    "origin": coordinate_analysis.coordinate_system.origin,
                    "orientation": coordinate_analysis.coordinate_system.orientation,
                    "units": coordinate_analysis.coordinate_system.units,
                    "grid_spacing": coordinate_analysis.coordinate_system.grid_spacing,
                    "confidence": coordinate_analysis.coordinate_system.confidence,
                }

            # Save to JSON
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Coordinate analysis saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving coordinate analysis: {e}")
            return False


def main():
    """Test the coordinate system analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description="Coordinate System Analyzer")
    parser.add_argument("input", help="Input image or PDF file")
    parser.add_argument("--page", type=int, default=0, help="PDF page number (0-based)")
    parser.add_argument("--output", help="Output JSON file for analysis results")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = CoordinateSystemAnalyzer()

    # Analyze coordinate system
    if args.input.lower().endswith(".pdf"):
        result = analyzer.detect_from_pdf(args.input, args.page)
    else:
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        result = analyzer.analyze_coordinate_system(image)

    # Print results
    print(f"Coordinate System Analysis Results:")
    print(f"  Detected: {result.detected}")
    print(f"  Total Grid Lines: {result.total_grid_lines}")
    print(f"  Confidence: {result.confidence:.3f}")

    if result.coordinate_system:
        print(f"  Coordinate System:")
        print(f"    Type: {result.coordinate_system.system_type}")
        print(f"    Units: {result.coordinate_system.units}")
        print(f"    Orientation: {result.coordinate_system.orientation:.1f}°")
        if result.coordinate_system.grid_spacing:
            print(
                f"    Grid Spacing: {result.coordinate_system.grid_spacing:.1f} pixels"
            )

    if result.grid_lines:
        print(f"  Grid Lines:")
        horizontal_count = len(
            [line for line in result.grid_lines if line.orientation == "horizontal"]
        )
        vertical_count = len(
            [line for line in result.grid_lines if line.orientation == "vertical"]
        )
        print(f"    Horizontal: {horizontal_count}")
        print(f"    Vertical: {vertical_count}")

    # Save results if output specified
    if args.output:
        analyzer.save_coordinate_analysis(result, args.output)


if __name__ == "__main__":
    main()
