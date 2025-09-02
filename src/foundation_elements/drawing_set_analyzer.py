"""
Drawing Set Analyzer

Analyzes drawing sets, sheet relationships, and match lines between drawings.
This integrates with the existing line matcher for comprehensive analysis.
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

# Import the existing line matcher
from src.core.line_matcher import LineMatcher, LineSegment, MatchedPair

logger = logging.getLogger(__name__)


@dataclass
class DrawingSheet:
    """Represents a drawing sheet."""

    sheet_id: str
    sheet_number: str
    sheet_title: str
    discipline: str
    file_path: str
    page_number: int
    image: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class MatchLine:
    """Represents a match line between drawings."""

    line_id: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    orientation: str  # "horizontal", "vertical"
    sheet1_id: str
    sheet2_id: str
    confidence: float
    matched_segments: List[MatchedPair] = None


@dataclass
class DrawingSetAnalysis:
    """Result of drawing set analysis."""

    total_sheets: int
    sheets: List[DrawingSheet]
    match_lines: List[MatchLine]
    sheet_relationships: Dict[str, List[str]]  # sheet_id -> connected_sheet_ids
    confidence: float
    analysis_complete: bool


class DrawingSetAnalyzer:
    """
    Advanced drawing set analysis system.

    Features:
    - Drawing sheet analysis
    - Match line detection using existing line matcher
    - Sheet relationship mapping
    - Cross-sheet element matching
    - Drawing set validation
    """

    def __init__(self):
        """Initialize the drawing set analyzer."""
        # Initialize the line matcher
        self.line_matcher = LineMatcher(
            min_line_length=50.0,
            max_angle_diff=10.0,
            distance_threshold=30.0,
            confidence_threshold=0.7,
        )

        # Analysis parameters
        self.min_match_line_length = 100  # pixels
        self.match_line_confidence_threshold = 0.6

        logger.info("DrawingSetAnalyzer initialized with LineMatcher integration")

    def analyze_drawing_set(self, drawing_files: List[str]) -> DrawingSetAnalysis:
        """
        Analyze a complete drawing set.

        Args:
            drawing_files: List of paths to drawing files (PDFs)

        Returns:
            DrawingSetAnalysis object with analysis results
        """
        if not drawing_files:
            return DrawingSetAnalysis(
                total_sheets=0,
                sheets=[],
                match_lines=[],
                sheet_relationships={},
                confidence=0.0,
                analysis_complete=False,
            )

        # 1. Load and analyze individual sheets
        sheets = self._load_drawing_sheets(drawing_files)

        # 2. Detect match lines between sheets
        match_lines = self._detect_match_lines(sheets)

        # 3. Build sheet relationships
        sheet_relationships = self._build_sheet_relationships(sheets, match_lines)

        # 4. Calculate overall confidence
        confidence = self._calculate_analysis_confidence(sheets, match_lines)

        return DrawingSetAnalysis(
            total_sheets=len(sheets),
            sheets=sheets,
            match_lines=match_lines,
            sheet_relationships=sheet_relationships,
            confidence=confidence,
            analysis_complete=True,
        )

    def _load_drawing_sheets(self, drawing_files: List[str]) -> List[DrawingSheet]:
        """Load and analyze individual drawing sheets."""
        sheets = []

        for file_path in drawing_files:
            try:
                # Extract sheet information from filename
                sheet_info = self._extract_sheet_info(file_path)

                # Load first page as image
                image = self._load_sheet_image(file_path, 0)

                sheet = DrawingSheet(
                    sheet_id=sheet_info["sheet_id"],
                    sheet_number=sheet_info["sheet_number"],
                    sheet_title=sheet_info["sheet_title"],
                    discipline=sheet_info["discipline"],
                    file_path=file_path,
                    page_number=0,
                    image=image,
                )
                sheets.append(sheet)

            except Exception as e:
                logger.error(f"Error loading sheet {file_path}: {e}")
                continue

        return sheets

    def _extract_sheet_info(self, file_path: str) -> Dict[str, str]:
        """Extract sheet information from filename."""
        filename = Path(file_path).stem

        # Default values
        sheet_info = {
            "sheet_id": filename,
            "sheet_number": "1",
            "sheet_title": "Unknown",
            "discipline": "unknown",
        }

        # Try to extract information from filename patterns
        # Example: "C-101_Traffic_Plan" -> sheet_number="C-101", discipline="traffic"

        # Look for sheet number patterns (C-101, S-201, etc.)
        sheet_number_match = re.search(r"([A-Z]-\d+)", filename, re.IGNORECASE)
        if sheet_number_match:
            sheet_info["sheet_number"] = sheet_number_match.group(1)

        # Look for discipline keywords
        discipline_keywords = {
            "traffic": ["traffic", "signal", "sign"],
            "electrical": ["electrical", "power", "lighting"],
            "structural": ["structural", "steel", "concrete"],
            "drainage": ["drainage", "storm", "sewer"],
            "civil": ["civil", "grading", "paving"],
        }

        filename_lower = filename.lower()
        for discipline, keywords in discipline_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                sheet_info["discipline"] = discipline
                break

        return sheet_info

    def _load_sheet_image(
        self, file_path: str, page_number: int = 0
    ) -> Optional[np.ndarray]:
        """Load sheet image from PDF."""
        try:
            # Open PDF
            doc = fitz.open(file_path)
            if page_number >= doc.page_count:
                logger.error(f"Page number {page_number} out of range for {file_path}")
                return None

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

            doc.close()
            return image_array

        except Exception as e:
            logger.error(f"Error loading image from {file_path}: {e}")
            return None

    def _detect_match_lines(self, sheets: List[DrawingSheet]) -> List[MatchLine]:
        """Detect match lines between drawing sheets."""
        match_lines = []

        # Compare each pair of sheets
        for i, sheet1 in enumerate(sheets):
            for j, sheet2 in enumerate(sheets[i + 1 :], i + 1):
                if sheet1.image is None or sheet2.image is None:
                    continue

                # Use the line matcher to find matching lines
                sheet_lines = self._detect_match_lines_between_sheets(sheet1, sheet2)

                for line in sheet_lines:
                    match_line = MatchLine(
                        line_id=f"match_{sheet1.sheet_id}_{sheet2.sheet_id}_{len(match_lines)}",
                        start_point=line.start_point,
                        end_point=line.end_point,
                        orientation=line.orientation,
                        sheet1_id=sheet1.sheet_id,
                        sheet2_id=sheet2.sheet_id,
                        confidence=line.confidence,
                    )
                    match_lines.append(match_line)

        return match_lines

    def _detect_match_lines_between_sheets(
        self, sheet1: DrawingSheet, sheet2: DrawingSheet
    ) -> List[MatchLine]:
        """Detect match lines between two specific sheets."""
        if sheet1.image is None or sheet2.image is None:
            return []

        # Convert images to grayscale
        gray1 = (
            cv2.cvtColor(sheet1.image, cv2.COLOR_BGR2GRAY)
            if len(sheet1.image.shape) == 3
            else sheet1.image
        )
        gray2 = (
            cv2.cvtColor(sheet2.image, cv2.COLOR_BGR2GRAY)
            if len(sheet2.image.shape) == 3
            else sheet2.image
        )

        # Use the line matcher to detect and match lines
        lines1 = self.line_matcher.detect_lines(gray1, method="combined")
        lines2 = self.line_matcher.detect_lines(gray2, method="combined")

        # Match lines between sheets
        matches = self.line_matcher.match_lines(lines1, lines2)

        # Convert matches to match lines
        match_lines = []
        for match in matches:
            if match.confidence >= self.match_line_confidence_threshold:
                # Determine orientation
                line1 = match.line1
                angle = line1.angle
                orientation = (
                    "horizontal"
                    if abs(angle) < 10 or abs(angle - 180) < 10
                    else "vertical"
                )

                # Check if line is long enough to be a match line
                length = line1.length
                if length >= self.min_match_line_length:
                    match_line = MatchLine(
                        line_id=f"match_{sheet1.sheet_id}_{sheet2.sheet_id}_{len(match_lines)}",
                        start_point=line1.start_point,
                        end_point=line1.end_point,
                        orientation=orientation,
                        sheet1_id=sheet1.sheet_id,
                        sheet2_id=sheet2.sheet_id,
                        confidence=match.confidence,
                        matched_segments=[match],
                    )
                    match_lines.append(match_line)

        return match_lines

    def _build_sheet_relationships(
        self, sheets: List[DrawingSheet], match_lines: List[MatchLine]
    ) -> Dict[str, List[str]]:
        """Build relationships between sheets based on match lines."""
        relationships = {sheet.sheet_id: [] for sheet in sheets}

        for match_line in match_lines:
            # Add bidirectional relationships
            if match_line.sheet2_id not in relationships[match_line.sheet1_id]:
                relationships[match_line.sheet1_id].append(match_line.sheet2_id)

            if match_line.sheet1_id not in relationships[match_line.sheet2_id]:
                relationships[match_line.sheet2_id].append(match_line.sheet1_id)

        return relationships

    def _calculate_analysis_confidence(
        self, sheets: List[DrawingSheet], match_lines: List[MatchLine]
    ) -> float:
        """Calculate overall confidence of the analysis."""
        if not sheets:
            return 0.0

        # Base confidence on number of sheets and match lines
        sheet_factor = min(1.0, len(sheets) / 10)  # More sheets = higher confidence

        # Match line confidence
        if match_lines:
            match_confidence = np.mean([line.confidence for line in match_lines])
            match_factor = min(
                1.0, len(match_lines) / 20
            )  # More match lines = higher confidence
        else:
            match_confidence = 0.0
            match_factor = 0.0

        # Sheet quality factor
        valid_sheets = sum(1 for sheet in sheets if sheet.image is not None)
        quality_factor = valid_sheets / len(sheets) if sheets else 0.0

        # Combine factors
        confidence = (
            sheet_factor * 0.3
            + match_confidence * 0.4
            + match_factor * 0.2
            + quality_factor * 0.1
        )

        return confidence

    def validate_drawing_set(self, analysis: DrawingSetAnalysis) -> Dict[str, Any]:
        """Validate the drawing set for completeness and consistency."""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": [],
        }

        # Check for isolated sheets (no match lines)
        isolated_sheets = []
        for sheet_id, connections in analysis.sheet_relationships.items():
            if not connections:
                isolated_sheets.append(sheet_id)

        if isolated_sheets:
            validation_results["warnings"].append(
                f"Isolated sheets found: {isolated_sheets}"
            )

        # Check for missing disciplines
        disciplines = set(sheet.discipline for sheet in analysis.sheets)
        expected_disciplines = {
            "traffic",
            "electrical",
            "structural",
            "drainage",
            "civil",
        }
        missing_disciplines = expected_disciplines - disciplines

        if missing_disciplines:
            validation_results["warnings"].append(
                f"Missing disciplines: {missing_disciplines}"
            )

        # Check match line consistency
        if analysis.match_lines:
            # Check for sheets with too many match lines (potential errors)
            sheet_match_counts = {}
            for match_line in analysis.match_lines:
                sheet_match_counts[match_line.sheet1_id] = (
                    sheet_match_counts.get(match_line.sheet1_id, 0) + 1
                )
                sheet_match_counts[match_line.sheet2_id] = (
                    sheet_match_counts.get(match_line.sheet2_id, 0) + 1
                )

            for sheet_id, count in sheet_match_counts.items():
                if count > 10:  # Arbitrary threshold
                    validation_results["warnings"].append(
                        f"Sheet {sheet_id} has many match lines ({count})"
                    )

        # Overall validation
        if validation_results["warnings"]:
            validation_results["is_valid"] = False

        return validation_results

    def save_drawing_set_analysis(
        self, analysis: DrawingSetAnalysis, output_path: str
    ) -> bool:
        """
        Save drawing set analysis results to JSON file.

        Args:
            analysis: Drawing set analysis results
            output_path: Path to save JSON file

        Returns:
            True if saved successfully
        """
        try:
            # Convert to dictionary
            data = {
                "total_sheets": analysis.total_sheets,
                "confidence": analysis.confidence,
                "analysis_complete": analysis.analysis_complete,
                "sheets": [
                    {
                        "sheet_id": sheet.sheet_id,
                        "sheet_number": sheet.sheet_number,
                        "sheet_title": sheet.sheet_title,
                        "discipline": sheet.discipline,
                        "file_path": sheet.file_path,
                        "page_number": sheet.page_number,
                        "confidence": sheet.confidence,
                    }
                    for sheet in analysis.sheets
                ],
                "match_lines": [
                    {
                        "line_id": line.line_id,
                        "start_point": line.start_point,
                        "end_point": line.end_point,
                        "orientation": line.orientation,
                        "sheet1_id": line.sheet1_id,
                        "sheet2_id": line.sheet2_id,
                        "confidence": line.confidence,
                    }
                    for line in analysis.match_lines
                ],
                "sheet_relationships": analysis.sheet_relationships,
            }

            # Save to JSON
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Drawing set analysis saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving drawing set analysis: {e}")
            return False


def main():
    """Test the drawing set analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description="Drawing Set Analyzer")
    parser.add_argument("drawing_files", nargs="+", help="Drawing files to analyze")
    parser.add_argument("--output", help="Output JSON file for analysis results")
    parser.add_argument("--validate", action="store_true", help="Validate drawing set")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = DrawingSetAnalyzer()

    # Analyze drawing set
    result = analyzer.analyze_drawing_set(args.drawing_files)

    # Print results
    print(f"Drawing Set Analysis Results:")
    print(f"  Total Sheets: {result.total_sheets}")
    print(f"  Match Lines: {len(result.match_lines)}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Analysis Complete: {result.analysis_complete}")

    if result.sheets:
        print(f"  Sheets:")
        for sheet in result.sheets:
            print(f"    {sheet.sheet_number}: {sheet.sheet_title} ({sheet.discipline})")

    if result.match_lines:
        print(f"  Match Lines:")
        for line in result.match_lines[:5]:  # Show first 5
            print(
                f"    {line.sheet1_id} <-> {line.sheet2_id} ({line.orientation}, conf: {line.confidence:.3f})"
            )

    # Validate if requested
    if args.validate:
        validation = analyzer.validate_drawing_set(result)
        print(f"  Validation:")
        print(f"    Valid: {validation['is_valid']}")
        if validation["warnings"]:
            print(f"    Warnings: {validation['warnings']}")
        if validation["issues"]:
            print(f"    Issues: {validation['issues']}")

    # Save results if output specified
    if args.output:
        analyzer.save_drawing_set_analysis(result, args.output)


if __name__ == "__main__":
    main()
