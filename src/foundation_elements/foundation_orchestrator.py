"""
Foundation Elements Orchestrator

Coordinates all foundation elements to provide comprehensive engineering drawing analysis.
This is the main entry point for foundation-level analysis.
"""

import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from .coordinate_system_analyzer import CoordinateAnalysis, CoordinateSystemAnalyzer
from .drawing_set_analyzer import DrawingSetAnalysis, DrawingSetAnalyzer
from .legend_extractor import LegendDetection, LegendExtractor

# Import all foundation elements
from .north_arrow_detector import NorthArrowDetection, NorthArrowDetector
from .notes_extractor import NotesDetection, NotesExtractor
from .scale_detector import ScaleDetection, ScaleDetector

logger = logging.getLogger(__name__)


@dataclass
class FoundationAnalysis:
    """Comprehensive foundation analysis results."""

    # Basic drawing information
    drawing_path: str
    page_number: int
    analysis_timestamp: str

    # Foundation element results
    north_arrow: NorthArrowDetection
    scale: ScaleDetection
    legend: LegendDetection
    notes: NotesDetection
    coordinate_system: CoordinateAnalysis

    # Overall analysis metrics
    overall_confidence: float
    foundation_score: float  # 0-100 score of foundation completeness
    missing_elements: List[str]
    recommendations: List[str]


class FoundationOrchestrator:
    """
    Main orchestrator for foundation elements analysis.

    This class coordinates all foundation elements to provide a comprehensive
    analysis of engineering drawings at the foundation level.
    """

    def __init__(self, template_dir: str = "templates"):
        """
        Initialize the foundation orchestrator.

        Args:
            template_dir: Directory containing templates for various detectors
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)

        # Initialize all foundation elements
        self.north_arrow_detector = NorthArrowDetector(
            str(self.template_dir / "north_arrows")
        )
        self.scale_detector = ScaleDetector()
        self.legend_extractor = LegendExtractor(str(self.template_dir / "legends"))
        self.notes_extractor = NotesExtractor()
        self.coordinate_analyzer = CoordinateSystemAnalyzer()
        self.drawing_set_analyzer = DrawingSetAnalyzer()

        logger.info("FoundationOrchestrator initialized with all foundation elements")

    def analyze_drawing(
        self, drawing_path: str, page_number: int = 0
    ) -> FoundationAnalysis:
        """
        Perform comprehensive foundation analysis on a single drawing.

        Args:
            drawing_path: Path to drawing file (PDF or image)
            page_number: Page number (0-based, for PDFs)

        Returns:
            FoundationAnalysis object with comprehensive results
        """
        logger.info(
            f"Starting foundation analysis of {drawing_path} (page {page_number})"
        )

        # Load drawing image
        image = self._load_drawing_image(drawing_path, page_number)
        if image is None:
            raise ValueError(f"Could not load image from {drawing_path}")

        # Perform all foundation element analyses
        north_arrow = self.north_arrow_detector.detect_north_arrow(image)
        scale = self.scale_detector.detect_scale(image)
        legend = self.legend_extractor.detect_legend(image)
        notes = self.notes_extractor.detect_notes(image)
        coordinate_system = self.coordinate_analyzer.analyze_coordinate_system(image)

        # Calculate overall metrics
        overall_confidence = self._calculate_overall_confidence(
            north_arrow, scale, legend, notes, coordinate_system
        )

        foundation_score = self._calculate_foundation_score(
            north_arrow, scale, legend, notes, coordinate_system
        )

        missing_elements = self._identify_missing_elements(
            north_arrow, scale, legend, notes, coordinate_system
        )

        recommendations = self._generate_recommendations(
            north_arrow, scale, legend, notes, coordinate_system, missing_elements
        )

        return FoundationAnalysis(
            drawing_path=drawing_path,
            page_number=page_number,
            analysis_timestamp=datetime.now().isoformat(),
            north_arrow=north_arrow,
            scale=scale,
            legend=legend,
            notes=notes,
            coordinate_system=coordinate_system,
            overall_confidence=overall_confidence,
            foundation_score=foundation_score,
            missing_elements=missing_elements,
            recommendations=recommendations,
        )

    def analyze_drawing_set(self, drawing_files: List[str]) -> Dict[str, Any]:
        """
        Analyze a complete drawing set.

        Args:
            drawing_files: List of paths to drawing files

        Returns:
            Dictionary with drawing set analysis results
        """
        logger.info(f"Starting drawing set analysis of {len(drawing_files)} files")

        # Analyze individual drawings
        individual_analyses = []
        for file_path in drawing_files:
            try:
                analysis = self.analyze_drawing(file_path, 0)
                individual_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                continue

        # Perform drawing set analysis
        drawing_set_analysis = self.drawing_set_analyzer.analyze_drawing_set(
            drawing_files
        )

        # Compile results
        results = {
            "drawing_set_analysis": drawing_set_analysis,
            "individual_analyses": individual_analyses,
            "summary": self._generate_drawing_set_summary(
                individual_analyses, drawing_set_analysis
            ),
        }

        return results

    def _load_drawing_image(
        self, drawing_path: str, page_number: int = 0
    ) -> Optional[np.ndarray]:
        """Load drawing image from file."""
        try:
            if drawing_path.lower().endswith(".pdf"):
                # Load from PDF
                doc = fitz.open(drawing_path)
                if page_number >= doc.page_count:
                    logger.error(f"Page number {page_number} out of range")
                    return None

                page = doc.load_page(page_number)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                image_array = np.array(pil_image)

                doc.close()
                return image_array
            else:
                # Load image file
                image = cv2.imread(drawing_path)
                if image is None:
                    logger.error(f"Could not load image: {drawing_path}")
                    return None
                return image

        except Exception as e:
            logger.error(f"Error loading image from {drawing_path}: {e}")
            return None

    def _calculate_overall_confidence(
        self,
        north_arrow: NorthArrowDetection,
        scale: ScaleDetection,
        legend: LegendDetection,
        notes: NotesDetection,
        coordinate_system: CoordinateAnalysis,
    ) -> float:
        """Calculate overall confidence of foundation analysis."""
        confidences = []

        # Weight different elements by importance
        if north_arrow.detected:
            confidences.append(north_arrow.confidence * 0.15)  # 15% weight

        if scale.detected:
            confidences.append(scale.confidence * 0.20)  # 20% weight

        if legend.detected:
            confidences.append(legend.confidence * 0.25)  # 25% weight

        if notes.detected:
            confidences.append(notes.confidence * 0.20)  # 20% weight

        if coordinate_system.detected:
            confidences.append(coordinate_system.confidence * 0.20)  # 20% weight

        # Return average confidence if any elements detected, otherwise 0
        return np.mean(confidences) if confidences else 0.0

    def _calculate_foundation_score(
        self,
        north_arrow: NorthArrowDetection,
        scale: ScaleDetection,
        legend: LegendDetection,
        notes: NotesDetection,
        coordinate_system: CoordinateAnalysis,
    ) -> float:
        """Calculate foundation completeness score (0-100)."""
        score = 0.0

        # North arrow (15 points)
        if north_arrow.detected:
            score += 15.0 * north_arrow.confidence

        # Scale (20 points)
        if scale.detected:
            score += 20.0 * scale.confidence

        # Legend (25 points)
        if legend.detected:
            score += 25.0 * legend.confidence

        # Notes (20 points)
        if notes.detected:
            score += 20.0 * notes.confidence

        # Coordinate system (20 points)
        if coordinate_system.detected:
            score += 20.0 * coordinate_system.confidence

        return min(100.0, score)

    def _identify_missing_elements(
        self,
        north_arrow: NorthArrowDetection,
        scale: ScaleDetection,
        legend: LegendDetection,
        notes: NotesDetection,
        coordinate_system: CoordinateAnalysis,
    ) -> List[str]:
        """Identify missing foundation elements."""
        missing = []

        if not north_arrow.detected:
            missing.append("north_arrow")

        if not scale.detected:
            missing.append("scale")

        if not legend.detected:
            missing.append("legend")

        if not notes.detected:
            missing.append("notes")

        if not coordinate_system.detected:
            missing.append("coordinate_system")

        return missing

    def _generate_recommendations(
        self,
        north_arrow: NorthArrowDetection,
        scale: ScaleDetection,
        legend: LegendDetection,
        notes: NotesDetection,
        coordinate_system: CoordinateAnalysis,
        missing_elements: List[str],
    ) -> List[str]:
        """Generate recommendations for improving foundation analysis."""
        recommendations = []

        # Recommendations based on missing elements
        if "north_arrow" in missing_elements:
            recommendations.append(
                "Add north arrow template to improve orientation detection"
            )

        if "scale" in missing_elements:
            recommendations.append(
                "Include scale bar or scale text for accurate measurements"
            )

        if "legend" in missing_elements:
            recommendations.append("Add legend section to improve symbol recognition")

        if "notes" in missing_elements:
            recommendations.append("Include notes section for specification extraction")

        if "coordinate_system" in missing_elements:
            recommendations.append(
                "Add grid lines or coordinate references for spatial analysis"
            )

        # Quality recommendations
        if north_arrow.detected and north_arrow.confidence < 0.7:
            recommendations.append(
                "Improve north arrow quality for better orientation accuracy"
            )

        if scale.detected and scale.confidence < 0.7:
            recommendations.append(
                "Enhance scale clarity for better measurement accuracy"
            )

        if legend.detected and legend.confidence < 0.7:
            recommendations.append(
                "Improve legend clarity for better symbol recognition"
            )

        return recommendations

    def _generate_drawing_set_summary(
        self,
        individual_analyses: List[FoundationAnalysis],
        drawing_set_analysis: DrawingSetAnalysis,
    ) -> Dict[str, Any]:
        """Generate summary of drawing set analysis."""
        if not individual_analyses:
            return {}

        # Calculate aggregate statistics
        foundation_scores = [
            analysis.foundation_score for analysis in individual_analyses
        ]
        confidences = [analysis.overall_confidence for analysis in individual_analyses]

        # Count missing elements
        missing_element_counts = {}
        for analysis in individual_analyses:
            for element in analysis.missing_elements:
                missing_element_counts[element] = (
                    missing_element_counts.get(element, 0) + 1
                )

        return {
            "total_drawings": len(individual_analyses),
            "average_foundation_score": np.mean(foundation_scores),
            "average_confidence": np.mean(confidences),
            "missing_elements_summary": missing_element_counts,
            "drawing_set_confidence": drawing_set_analysis.confidence,
            "match_lines_found": len(drawing_set_analysis.match_lines),
        }

    def save_analysis(self, analysis: FoundationAnalysis, output_path: str) -> bool:
        """
        Save foundation analysis results to JSON file.

        Args:
            analysis: Foundation analysis results
            output_path: Path to save JSON file

        Returns:
            True if saved successfully
        """
        try:
            # Convert to dictionary
            data = {
                "drawing_path": analysis.drawing_path,
                "page_number": analysis.page_number,
                "analysis_timestamp": analysis.analysis_timestamp,
                "overall_confidence": analysis.overall_confidence,
                "foundation_score": analysis.foundation_score,
                "missing_elements": analysis.missing_elements,
                "recommendations": analysis.recommendations,
                "north_arrow": {
                    "detected": analysis.north_arrow.detected,
                    "confidence": analysis.north_arrow.confidence,
                    "angle": analysis.north_arrow.angle,
                    "arrow_type": analysis.north_arrow.arrow_type,
                },
                "scale": {
                    "detected": analysis.scale.detected,
                    "confidence": analysis.scale.confidence,
                    "scale_factor": analysis.scale.scale_factor,
                    "units": analysis.scale.units,
                },
                "legend": {
                    "detected": analysis.legend.detected,
                    "confidence": analysis.legend.confidence,
                    "total_symbols": analysis.legend.total_symbols,
                    "discipline": analysis.legend.discipline,
                },
                "notes": {
                    "detected": analysis.notes.detected,
                    "confidence": analysis.notes.confidence,
                    "total_notes": analysis.notes.total_notes,
                    "text_density": analysis.notes.text_density,
                },
                "coordinate_system": {
                    "detected": analysis.coordinate_system.detected,
                    "confidence": analysis.coordinate_system.confidence,
                    "total_grid_lines": analysis.coordinate_system.total_grid_lines,
                },
            }

            # Save to JSON
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Foundation analysis saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving foundation analysis: {e}")
            return False


def main():
    """Test the foundation orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Foundation Elements Orchestrator")
    parser.add_argument("input", help="Input drawing file or directory")
    parser.add_argument("--output", help="Output JSON file for analysis results")
    parser.add_argument("--page", type=int, default=0, help="PDF page number (0-based)")
    parser.add_argument(
        "--template-dir", default="templates", help="Template directory"
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = FoundationOrchestrator(args.template_dir)

    # Check if input is file or directory
    input_path = Path(args.input)

    if input_path.is_file():
        # Analyze single drawing
        analysis = orchestrator.analyze_drawing(str(input_path), args.page)

        # Print results
        print(f"Foundation Analysis Results:")
        print(f"  Drawing: {analysis.drawing_path}")
        print(f"  Foundation Score: {analysis.foundation_score:.1f}/100")
        print(f"  Overall Confidence: {analysis.overall_confidence:.3f}")
        print(f"  Missing Elements: {analysis.missing_elements}")

        if analysis.recommendations:
            print(f"  Recommendations:")
            for rec in analysis.recommendations:
                print(f"    - {rec}")

        # Save results if output specified
        if args.output:
            orchestrator.save_analysis(analysis, args.output)

    elif input_path.is_dir():
        # Analyze drawing set
        drawing_files = (
            list(input_path.glob("*.pdf"))
            + list(input_path.glob("*.jpg"))
            + list(input_path.glob("*.png"))
        )
        drawing_files = [str(f) for f in drawing_files]

        if not drawing_files:
            print(f"No drawing files found in {input_path}")
            return

        results = orchestrator.analyze_drawing_set(drawing_files)

        # Print results
        print(f"Drawing Set Analysis Results:")
        print(f"  Total Drawings: {results['summary']['total_drawings']}")
        print(
            f"  Average Foundation Score: {results['summary']['average_foundation_score']:.1f}/100"
        )
        print(f"  Average Confidence: {results['summary']['average_confidence']:.3f}")
        print(f"  Match Lines Found: {results['summary']['match_lines_found']}")

        # Save results if output specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")

    else:
        print(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
