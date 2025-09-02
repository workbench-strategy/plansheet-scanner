"""
Enhanced Legend Extractor

Extracts and analyzes legends from engineering drawings.
This is critical for understanding symbols and their meanings.
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
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass
class LegendSymbol:
    """Represents a symbol in the legend."""

    symbol_id: str
    symbol_name: str
    symbol_type: str  # traffic, electrical, structural, etc.
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    description: str
    file_path: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class LegendDetection:
    """Result of legend detection."""

    detected: bool
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    symbols: List[LegendSymbol]
    legend_type: str  # "standard", "abbreviated", "detailed"
    discipline: str  # "traffic", "electrical", "structural", etc.
    total_symbols: int
    legend_text: Optional[str] = None


class LegendExtractor:
    """
    Advanced legend extraction system.

    Features:
    - Automatic legend region detection
    - Symbol extraction and classification
    - Multi-discipline legend support
    - OCR for text extraction
    - Template matching for common symbols
    """

    def __init__(self, template_dir: str = "templates/legends"):
        """
        Initialize the legend extractor.

        Args:
            template_dir: Directory containing legend templates
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)

        # Load templates
        self.templates = self._load_templates()

        # Legend detection parameters
        self.min_legend_size = (100, 50)  # minimum width, height
        self.max_legend_size = (800, 600)  # maximum width, height
        self.min_confidence = 0.6

        # Symbol patterns by discipline
        self.symbol_patterns = {
            "traffic": {
                "signal_symbols": ["TS", "TRAFFIC SIGNAL", "SIGNAL", "LIGHT"],
                "sign_symbols": ["STOP", "YIELD", "SPEED", "WARNING"],
                "detector_symbols": ["LOOP", "DETECTOR", "SENSOR", "CAMERA"],
            },
            "electrical": {
                "equipment_symbols": ["PANEL", "BOX", "JUNCTION", "TRANSFORMER"],
                "conduit_symbols": ["CONDUIT", "EMT", "PVC", "RMC"],
                "lighting_symbols": ["LIGHT", "LAMP", "FIXTURE", "POLE"],
            },
            "structural": {
                "beam_symbols": ["BEAM", "GIRDER", "COLUMN", "BRACE"],
                "foundation_symbols": ["FOOTING", "PILE", "CAISSON", "SLAB"],
                "connection_symbols": ["BOLT", "WELD", "PLATE", "ANGLE"],
            },
            "drainage": {
                "pipe_symbols": ["PIPE", "CULVERT", "MANHOLE", "CATCH BASIN"],
                "flow_symbols": ["FLOW", "DIRECTION", "GRADE", "SLOPE"],
            },
        }

        logger.info(f"LegendExtractor initialized with {len(self.templates)} templates")

    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Load legend templates from directory."""
        templates = {}

        if not self.template_dir.exists():
            logger.warning(f"Template directory not found: {self.template_dir}")
            return templates

        # Load existing templates
        for template_file in self.template_dir.glob("*.png"):
            try:
                template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    template_name = template_file.stem
                    templates[template_name] = template
                    logger.info(f"Loaded template: {template_name} ({template.shape})")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")

        return templates

    def detect_legend(self, image: np.ndarray) -> LegendDetection:
        """
        Detect legend region in the given image.

        Args:
            image: Input image as numpy array (grayscale or BGR)

        Returns:
            LegendDetection object with detection results
        """
        if image is None:
            return LegendDetection(
                detected=False,
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                symbols=[],
                legend_type="unknown",
                discipline="unknown",
                total_symbols=0,
            )

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Template-based legend detection
        template_result = self._detect_legend_by_template(gray)

        # 2. Pattern-based legend detection
        pattern_result = self._detect_legend_by_pattern(gray)

        # 3. Text-based legend detection
        text_result = self._detect_legend_by_text(gray)

        # Combine results
        best_result = self._combine_legend_results(
            template_result, pattern_result, text_result
        )

        # Extract symbols if legend detected
        if best_result.detected:
            best_result.symbols = self._extract_symbols_from_legend(
                gray, best_result.bbox
            )
            best_result.total_symbols = len(best_result.symbols)
            best_result.discipline = self._classify_legend_discipline(
                best_result.symbols
            )
            best_result.legend_type = self._classify_legend_type(best_result.symbols)

        return best_result

    def _detect_legend_by_template(
        self, gray_image: np.ndarray
    ) -> Optional[LegendDetection]:
        """Detect legend using template matching."""
        best_result = None
        best_confidence = 0.0

        for template_name, template in self.templates.items():
            # Template matching
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_confidence and max_val > self.min_confidence:
                best_confidence = max_val
                h, w = template.shape
                bbox = (max_loc[0], max_loc[1], w, h)

                best_result = LegendDetection(
                    detected=True,
                    bbox=bbox,
                    confidence=max_val,
                    symbols=[],
                    legend_type="template_matched",
                    discipline="unknown",
                    total_symbols=0,
                )

        return best_result

    def _detect_legend_by_pattern(
        self, gray_image: np.ndarray
    ) -> Optional[LegendDetection]:
        """Detect legend by analyzing patterns and layout."""
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_result = None
        best_confidence = 0.0

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Check size constraints
            if (
                w < self.min_legend_size[0]
                or h < self.min_legend_size[1]
                or w > self.max_legend_size[0]
                or h > self.max_legend_size[1]
            ):
                continue

            # Calculate confidence based on shape and content
            confidence = self._calculate_legend_confidence(gray_image, x, y, w, h)

            if confidence > best_confidence and confidence > self.min_confidence:
                best_confidence = confidence
                best_result = LegendDetection(
                    detected=True,
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    symbols=[],
                    legend_type="pattern_detected",
                    discipline="unknown",
                    total_symbols=0,
                )

        return best_result

    def _detect_legend_by_text(
        self, gray_image: np.ndarray
    ) -> Optional[LegendDetection]:
        """Detect legend by finding legend-related text."""
        # This would integrate with OCR to find "LEGEND", "SYMBOLS", etc.
        # For now, return None - would need OCR implementation
        return None

    def _calculate_legend_confidence(
        self, gray_image: np.ndarray, x: int, y: int, w: int, h: int
    ) -> float:
        """Calculate confidence that a region contains a legend."""
        # Extract region
        roi = gray_image[y : y + h, x : x + w]

        # Analyze region properties
        # 1. Edge density (legends have many edges from symbols)
        edges = cv2.Canny(roi, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # 2. Contour density (legends have many small contours)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_density = len(contours) / (w * h / 1000)  # contours per 1000 pixels

        # 3. Aspect ratio (legends are usually rectangular)
        aspect_ratio = w / h if h > 0 else 0
        aspect_confidence = 1.0 - abs(aspect_ratio - 2.0) / 2.0  # Prefer 2:1 ratio

        # Combine factors
        confidence = (
            edge_density * 0.4
            + min(contour_density / 10, 1.0) * 0.4
            + max(0, aspect_confidence) * 0.2
        )

        return confidence

    def _extract_symbols_from_legend(
        self, gray_image: np.ndarray, legend_bbox: Tuple[int, int, int, int]
    ) -> List[LegendSymbol]:
        """Extract individual symbols from the legend region."""
        x, y, w, h = legend_bbox
        legend_roi = gray_image[y : y + h, x : x + w]

        symbols = []

        # Apply edge detection
        edges = cv2.Canny(legend_roi, 30, 100)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for i, contour in enumerate(contours):
            # Get bounding rectangle
            cx, cy, cw, ch = cv2.boundingRect(contour)

            # Filter by size
            if cw < 10 or ch < 10 or cw > w // 2 or ch > h // 2:
                continue

            # Calculate confidence
            area = cv2.contourArea(contour)
            confidence = min(0.9, area / 1000)  # Normalize by area

            if confidence > 0.3:  # Minimum confidence threshold
                symbol = LegendSymbol(
                    symbol_id=f"symbol_{i}",
                    symbol_name=f"Symbol_{i}",
                    symbol_type="unknown",
                    bbox=(cx, cy, cw, ch),
                    confidence=confidence,
                    description=f"Extracted symbol {i}",
                )
                symbols.append(symbol)

        return symbols

    def _classify_legend_discipline(self, symbols: List[LegendSymbol]) -> str:
        """Classify the discipline of the legend based on symbols."""
        discipline_scores = {
            "traffic": 0,
            "electrical": 0,
            "structural": 0,
            "drainage": 0,
        }

        for symbol in symbols:
            symbol_text = symbol.symbol_name.upper()

            for discipline, patterns in self.symbol_patterns.items():
                for category, keywords in patterns.items():
                    for keyword in keywords:
                        if keyword in symbol_text:
                            discipline_scores[discipline] += symbol.confidence

        # Return discipline with highest score
        if discipline_scores:
            return max(discipline_scores, key=discipline_scores.get)
        else:
            return "unknown"

    def _classify_legend_type(self, symbols: List[LegendSymbol]) -> str:
        """Classify the type of legend based on content."""
        if len(symbols) > 20:
            return "detailed"
        elif len(symbols) > 10:
            return "standard"
        else:
            return "abbreviated"

    def _combine_legend_results(
        self,
        template_result: Optional[LegendDetection],
        pattern_result: Optional[LegendDetection],
        text_result: Optional[LegendDetection],
    ) -> LegendDetection:
        """Combine multiple legend detection results."""
        results = [
            r for r in [template_result, pattern_result, text_result] if r is not None
        ]

        if not results:
            return LegendDetection(
                detected=False,
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                symbols=[],
                legend_type="unknown",
                discipline="unknown",
                total_symbols=0,
            )

        # Return the result with highest confidence
        best_result = max(results, key=lambda r: r.confidence)
        return best_result

    def extract_symbols_from_legend(
        self, pdf_path: str, page_number: int, output_dir: str = "extracted_symbols"
    ) -> List[str]:
        """
        Extract symbols from legend using manual selection (legacy method).

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-based)
            output_dir: Directory to save extracted symbols

        Returns:
            List of file paths to saved symbol images
        """
        # This is the original manual selection method
        # Import the original function
        from src.core.legend_extractor import (
            extract_symbols_from_legend as original_extract,
        )

        return original_extract(pdf_path, page_number, output_dir)

    def detect_from_pdf(self, pdf_path: str, page_number: int = 0) -> LegendDetection:
        """
        Detect legend from a PDF page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-based)

        Returns:
            LegendDetection object
        """
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            if page_number >= doc.page_count:
                logger.error(f"Page number {page_number} out of range")
                return LegendDetection(
                    detected=False,
                    bbox=(0, 0, 0, 0),
                    confidence=0.0,
                    symbols=[],
                    legend_type="unknown",
                    discipline="unknown",
                    total_symbols=0,
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

            # Detect legend
            result = self.detect_legend(image_array)

            doc.close()
            return result

        except Exception as e:
            logger.error(f"Error detecting legend from PDF: {e}")
            return LegendDetection(
                detected=False,
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                symbols=[],
                legend_type="unknown",
                discipline="unknown",
                total_symbols=0,
            )

    def save_legend_analysis(
        self, legend_detection: LegendDetection, output_path: str
    ) -> bool:
        """
        Save legend analysis results to JSON file.

        Args:
            legend_detection: Legend detection results
            output_path: Path to save JSON file

        Returns:
            True if saved successfully
        """
        try:
            # Convert to dictionary
            data = {
                "detected": legend_detection.detected,
                "bbox": legend_detection.bbox,
                "confidence": legend_detection.confidence,
                "legend_type": legend_detection.legend_type,
                "discipline": legend_detection.discipline,
                "total_symbols": legend_detection.total_symbols,
                "symbols": [
                    {
                        "symbol_id": symbol.symbol_id,
                        "symbol_name": symbol.symbol_name,
                        "symbol_type": symbol.symbol_type,
                        "bbox": symbol.bbox,
                        "confidence": symbol.confidence,
                        "description": symbol.description,
                    }
                    for symbol in legend_detection.symbols
                ],
            }

            # Save to JSON
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Legend analysis saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving legend analysis: {e}")
            return False


def main():
    """Test the legend extractor."""
    import argparse

    parser = argparse.ArgumentParser(description="Legend Extractor")
    parser.add_argument("input", help="Input image or PDF file")
    parser.add_argument("--page", type=int, default=0, help="PDF page number (0-based)")
    parser.add_argument("--output", help="Output JSON file for analysis results")
    parser.add_argument(
        "--template-dir", default="templates/legends", help="Template directory"
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = LegendExtractor(args.template_dir)

    # Detect legend
    if args.input.lower().endswith(".pdf"):
        result = extractor.detect_from_pdf(args.input, args.page)
    else:
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        result = extractor.detect_legend(image)

    # Print results
    print(f"Legend Detection Results:")
    print(f"  Detected: {result.detected}")
    if result.detected:
        print(f"  BBox: {result.bbox}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Type: {result.legend_type}")
        print(f"  Discipline: {result.discipline}")
        print(f"  Total Symbols: {result.total_symbols}")

        # Save results if output specified
        if args.output:
            extractor.save_legend_analysis(result, args.output)


if __name__ == "__main__":
    main()
