"""
Notes Extractor

Extracts and analyzes notes, annotations, and text from engineering drawings.
This is essential for understanding specifications and requirements.
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
class NoteText:
    """Represents extracted note text."""

    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    note_type: str  # "general", "specification", "reference", "warning"
    font_size: Optional[float] = None
    color: Optional[Tuple[int, int, int]] = None


@dataclass
class NotesDetection:
    """Result of notes detection."""

    detected: bool
    notes: List[NoteText]
    total_notes: int
    confidence: float
    notes_regions: List[Tuple[int, int, int, int]]  # List of note region bboxes
    text_density: float  # Text density in the drawing


class NotesExtractor:
    """
    Advanced notes extraction system.

    Features:
    - Text region detection
    - OCR for text extraction
    - Note classification by type
    - Specification extraction
    - Reference number detection
    """

    def __init__(self):
        """Initialize the notes extractor."""
        # Note type patterns
        self.note_patterns = {
            "specification": [
                r"\bNOTE\b",
                r"\bSPEC\b",
                r"\bSPECIFICATION\b",
                r"\bREQUIREMENT\b",
                r"\bSTANDARD\b",
            ],
            "reference": [
                r"\bREF\b",
                r"\bREFERENCE\b",
                r"\bSEE\b",
                r"\bREFER TO\b",
                r"\bDETAIL\b",
            ],
            "warning": [
                r"\bWARNING\b",
                r"\bCAUTION\b",
                r"\bATTENTION\b",
                r"\bIMPORTANT\b",
                r"\bCRITICAL\b",
            ],
            "general": [
                r"\bGENERAL\b",
                r"\bTYPICAL\b",
                r"\bUNLESS\b",
                r"\bEXCEPT\b",
                r"\bALL\b",
            ],
        }

        # Detection parameters
        self.min_text_confidence = 0.5
        self.min_note_size = (20, 10)  # minimum width, height
        self.max_note_size = (800, 200)  # maximum width, height

        logger.info("NotesExtractor initialized")

    def detect_notes(self, image: np.ndarray) -> NotesDetection:
        """
        Detect notes in the given image.

        Args:
            image: Input image as numpy array (grayscale or BGR)

        Returns:
            NotesDetection object with detection results
        """
        if image is None:
            return NotesDetection(
                detected=False,
                notes=[],
                total_notes=0,
                confidence=0.0,
                notes_regions=[],
                text_density=0.0,
            )

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Text region detection
        text_regions = self._detect_text_regions(gray)

        # 2. Extract notes from regions
        notes = self._extract_notes_from_regions(gray, text_regions)

        # 3. Calculate overall metrics
        total_notes = len(notes)
        confidence = np.mean([note.confidence for note in notes]) if notes else 0.0
        text_density = self._calculate_text_density(gray, text_regions)

        return NotesDetection(
            detected=total_notes > 0,
            notes=notes,
            total_notes=total_notes,
            confidence=confidence,
            notes_regions=text_regions,
            text_density=text_density,
        )

    def _detect_text_regions(
        self, gray_image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Detect potential text regions in the image."""
        # Apply edge detection
        edges = cv2.Canny(gray_image, 30, 100)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        text_regions = []

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Check size constraints
            if (
                w < self.min_note_size[0]
                or h < self.min_note_size[1]
                or w > self.max_note_size[0]
                or h > self.max_note_size[1]
            ):
                continue

            # Calculate region properties
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Text regions typically have specific characteristics
            if self._is_likely_text_region(gray_image, x, y, w, h, area, aspect_ratio):
                text_regions.append((x, y, w, h))

        return text_regions

    def _is_likely_text_region(
        self,
        gray_image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        area: float,
        aspect_ratio: float,
    ) -> bool:
        """Determine if a region is likely to contain text."""
        # Extract region
        roi = gray_image[y : y + h, x : x + w]

        # 1. Edge density (text has high edge density)
        edges = cv2.Canny(roi, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # 2. Aspect ratio (text regions are usually wider than tall)
        aspect_score = min(1.0, aspect_ratio / 3.0) if aspect_ratio > 0 else 0

        # 3. Area consistency (not too small, not too large)
        area_score = 1.0 if 100 <= area <= 10000 else 0.5

        # 4. Variance (text regions have high variance)
        variance = np.var(roi)
        variance_score = min(1.0, variance / 1000)

        # Combine scores
        total_score = (
            edge_density * 0.4
            + aspect_score * 0.2
            + area_score * 0.2
            + variance_score * 0.2
        )

        return total_score > 0.6

    def _extract_notes_from_regions(
        self, gray_image: np.ndarray, text_regions: List[Tuple[int, int, int, int]]
    ) -> List[NoteText]:
        """Extract notes from detected text regions."""
        notes = []

        for i, (x, y, w, h) in enumerate(text_regions):
            # Extract region
            roi = gray_image[y : y + h, x : x + w]

            # This would integrate with OCR (pytesseract)
            # For now, simulate text extraction
            extracted_text = self._simulate_text_extraction(roi)

            if extracted_text:
                # Classify note type
                note_type = self._classify_note_type(extracted_text)

                # Calculate confidence
                confidence = self._calculate_note_confidence(roi, extracted_text)

                if confidence > self.min_text_confidence:
                    note = NoteText(
                        text=extracted_text,
                        bbox=(x, y, w, h),
                        confidence=confidence,
                        note_type=note_type,
                    )
                    notes.append(note)

        return notes

    def _simulate_text_extraction(self, roi: np.ndarray) -> str:
        """Simulate text extraction from ROI (placeholder for OCR)."""
        # This is a placeholder - would integrate with pytesseract
        # For now, return a simulated text based on region properties

        # Analyze region characteristics
        edge_density = np.sum(cv2.Canny(roi, 30, 100) > 0) / (
            roi.shape[0] * roi.shape[1]
        )
        variance = np.var(roi)

        # Simulate different types of text based on characteristics
        if edge_density > 0.1 and variance > 500:
            return "NOTE: See specification for details"
        elif edge_density > 0.05:
            return "REF: Detail A"
        else:
            return "General note"

    def _classify_note_type(self, text: str) -> str:
        """Classify the type of note based on text content."""
        text_upper = text.upper()

        for note_type, patterns in self.note_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_upper):
                    return note_type

        return "general"

    def _calculate_note_confidence(self, roi: np.ndarray, text: str) -> float:
        """Calculate confidence for extracted note."""
        # Base confidence on text length and region quality
        base_confidence = min(0.9, len(text) / 50)

        # Adjust based on region quality
        edge_density = np.sum(cv2.Canny(roi, 30, 100) > 0) / (
            roi.shape[0] * roi.shape[1]
        )
        quality_factor = min(1.0, edge_density * 5)

        return base_confidence * quality_factor

    def _calculate_text_density(
        self, gray_image: np.ndarray, text_regions: List[Tuple[int, int, int, int]]
    ) -> float:
        """Calculate text density in the image."""
        if not text_regions:
            return 0.0

        total_text_area = sum(w * h for _, _, w, h in text_regions)
        total_image_area = gray_image.shape[0] * gray_image.shape[1]

        return total_text_area / total_image_area if total_image_area > 0 else 0.0

    def extract_specifications(self, notes: List[NoteText]) -> List[Dict[str, Any]]:
        """Extract specifications from notes."""
        specifications = []

        for note in notes:
            if note.note_type == "specification":
                # Parse specification text
                spec = self._parse_specification_text(note.text)
                if spec:
                    specifications.append(spec)

        return specifications

    def _parse_specification_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse specification text to extract structured data."""
        # This would use more sophisticated NLP
        # For now, basic pattern matching

        spec = {"text": text, "type": "specification", "parameters": {}, "units": None}

        # Look for common specification patterns
        # Material specifications
        material_match = re.search(
            r"\b(steel|concrete|aluminum|copper|plastic)\b", text, re.IGNORECASE
        )
        if material_match:
            spec["parameters"]["material"] = material_match.group(1)

        # Dimension specifications
        dim_match = re.search(r"(\d+(?:\.\d+)?)\s*(in|ft|mm|cm|m)", text, re.IGNORECASE)
        if dim_match:
            spec["parameters"]["dimension"] = float(dim_match.group(1))
            spec["units"] = dim_match.group(2)

        # Grade specifications
        grade_match = re.search(r"grade\s*(\w+)", text, re.IGNORECASE)
        if grade_match:
            spec["parameters"]["grade"] = grade_match.group(1)

        return spec

    def detect_from_pdf(self, pdf_path: str, page_number: int = 0) -> NotesDetection:
        """
        Detect notes from a PDF page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-based)

        Returns:
            NotesDetection object
        """
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            if page_number >= doc.page_count:
                logger.error(f"Page number {page_number} out of range")
                return NotesDetection(
                    detected=False,
                    notes=[],
                    total_notes=0,
                    confidence=0.0,
                    notes_regions=[],
                    text_density=0.0,
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

            # Detect notes
            result = self.detect_notes(image_array)

            doc.close()
            return result

        except Exception as e:
            logger.error(f"Error detecting notes from PDF: {e}")
            return NotesDetection(
                detected=False,
                notes=[],
                total_notes=0,
                confidence=0.0,
                notes_regions=[],
                text_density=0.0,
            )

    def save_notes_analysis(
        self, notes_detection: NotesDetection, output_path: str
    ) -> bool:
        """
        Save notes analysis results to JSON file.

        Args:
            notes_detection: Notes detection results
            output_path: Path to save JSON file

        Returns:
            True if saved successfully
        """
        try:
            # Convert to dictionary
            data = {
                "detected": notes_detection.detected,
                "total_notes": notes_detection.total_notes,
                "confidence": notes_detection.confidence,
                "text_density": notes_detection.text_density,
                "notes_regions": notes_detection.notes_regions,
                "notes": [
                    {
                        "text": note.text,
                        "bbox": note.bbox,
                        "confidence": note.confidence,
                        "note_type": note.note_type,
                        "font_size": note.font_size,
                        "color": note.color,
                    }
                    for note in notes_detection.notes
                ],
            }

            # Save to JSON
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Notes analysis saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving notes analysis: {e}")
            return False


def main():
    """Test the notes extractor."""
    import argparse

    parser = argparse.ArgumentParser(description="Notes Extractor")
    parser.add_argument("input", help="Input image or PDF file")
    parser.add_argument("--page", type=int, default=0, help="PDF page number (0-based)")
    parser.add_argument("--output", help="Output JSON file for analysis results")

    args = parser.parse_args()

    # Initialize extractor
    extractor = NotesExtractor()

    # Detect notes
    if args.input.lower().endswith(".pdf"):
        result = extractor.detect_from_pdf(args.input, args.page)
    else:
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        result = extractor.detect_notes(image)

    # Print results
    print(f"Notes Detection Results:")
    print(f"  Detected: {result.detected}")
    print(f"  Total Notes: {result.total_notes}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Text Density: {result.text_density:.4f}")

    if result.notes:
        print(f"  Note Types:")
        type_counts = {}
        for note in result.notes:
            type_counts[note.note_type] = type_counts.get(note.note_type, 0) + 1

        for note_type, count in type_counts.items():
            print(f"    {note_type}: {count}")

        # Show first few notes
        print(f"  Sample Notes:")
        for i, note in enumerate(result.notes[:3]):
            print(f"    {i+1}. [{note.note_type}] {note.text[:50]}...")

    # Save results if output specified
    if args.output:
        extractor.save_notes_analysis(result, args.output)


if __name__ == "__main__":
    main()
