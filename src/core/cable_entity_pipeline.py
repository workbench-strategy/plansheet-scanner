import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CableEntity:
    """Represents a cable entity extracted from plan sheets."""

    id: str
    type: str
    length: Optional[float]
    material: Optional[str]
    voltage: Optional[str]
    diameter: Optional[float]
    location: Optional[Tuple[float, float]]
    confidence: float
    source_page: int
    extraction_method: str
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class CableEntityPipeline:
    """
    Pipeline for extracting cable entities from plan sheets.
    Supports multiple extraction methods including text analysis and visual detection.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the cable entity pipeline.

        Args:
            confidence_threshold: Minimum confidence for valid entities
        """
        self.confidence_threshold = confidence_threshold

        # Cable type patterns
        self.cable_patterns = {
            "power": r"\b(power|electrical|hv|mv|lv|voltage)\b",
            "fiber": r"\b(fiber|optical|fiberoptic|fiber-optic)\b",
            "copper": r"\b(copper|cat|ethernet|twisted|pair)\b",
            "coaxial": r"\b(coaxial|coax|rg|tv)\b",
            "control": r"\b(control|signal|instrumentation|telemetry)\b",
        }

        # Material patterns
        self.material_patterns = {
            "aluminum": r"\b(aluminum|al|alu)\b",
            "copper": r"\b(copper|cu)\b",
            "steel": r"\b(steel|st)\b",
            "plastic": r"\b(plastic|pvc|pe|hdpe)\b",
            "rubber": r"\b(rubber|neoprene|epdm)\b",
        }

        # Length patterns
        self.length_patterns = [
            r"(\d+(?:\.\d+)?)\s*(ft|feet|m|meter|km|kilometer)",
            r"length[:\s]*(\d+(?:\.\d+)?)",
            r"l[:\s]*(\d+(?:\.\d+)?)",
        ]

        # Voltage patterns
        self.voltage_patterns = [
            r"(\d+(?:\.\d+)?)\s*(v|volts|kv|kilovolt)",
            r"voltage[:\s]*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*v\b",
        ]

        logger.info("CableEntityPipeline initialized")

    def parse_pdf(
        self, pdf_path: str, pages: Optional[List[int]] = None
    ) -> List[CableEntity]:
        """
        Parse PDF and extract cable entities from specified pages.

        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to process (None for all pages)

        Returns:
            List of extracted CableEntity objects
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path}")

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count

        if pages is None:
            pages = list(range(total_pages))

        all_entities = []

        for page_num in pages:
            if page_num < 0 or page_num >= total_pages:
                logger.warning(f"Page {page_num} is out of range, skipping")
                continue

            logger.info(f"Processing page {page_num + 1}/{total_pages}")

            try:
                page_entities = self._extract_from_page(doc, page_num)
                all_entities.extend(page_entities)

                logger.info(
                    f"Extracted {len(page_entities)} entities from page {page_num}"
                )

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue

        doc.close()

        # Filter by confidence threshold
        filtered_entities = [
            entity
            for entity in all_entities
            if entity.confidence >= self.confidence_threshold
        ]

        logger.info(f"Total entities extracted: {len(filtered_entities)}")
        return filtered_entities

    def _extract_from_page(
        self, doc: fitz.Document, page_num: int
    ) -> List[CableEntity]:
        """
        Extract cable entities from a single page.

        Args:
            doc: PyMuPDF document
            page_num: Page number to process

        Returns:
            List of CableEntity objects
        """
        page = doc.load_page(page_num)
        entities = []

        # Method 1: Text extraction and analysis
        text_entities = self._extract_from_text(page, page_num)
        entities.extend(text_entities)

        # Method 2: Visual detection (if text extraction is insufficient)
        if len(text_entities) < 3:  # If few text entities found, try visual detection
            visual_entities = self._extract_from_visual(page, page_num)
            entities.extend(visual_entities)

        return entities

    def _extract_from_text(self, page: fitz.Page, page_num: int) -> List[CableEntity]:
        """
        Extract cable entities from page text.

        Args:
            page: PyMuPDF page object
            page_num: Page number

        Returns:
            List of CableEntity objects
        """
        entities = []

        # Extract text blocks
        text_blocks = page.get_text("dict")["blocks"]

        for block in text_blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    # Analyze text for cable information
                    entity = self._analyze_text_for_cable(
                        text, page_num, "text_analysis"
                    )
                    if entity:
                        entities.append(entity)

        return entities

    def _extract_from_visual(self, page: fitz.Page, page_num: int) -> List[CableEntity]:
        """
        Extract cable entities using visual detection.

        Args:
            page: PyMuPDF page object
            page_num: Page number

        Returns:
            List of CableEntity objects
        """
        entities = []

        # Convert page to image
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Detect line-like structures (potential cables)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Line detection using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10
        )

        if lines is not None:
            for i, line in enumerate(lines[:10]):  # Limit to first 10 lines
                x1, y1, x2, y2 = line[0]

                # Create entity for detected line
                entity = CableEntity(
                    id=f"cable_visual_{page_num}_{i}",
                    type="unknown",
                    length=self._calculate_line_length(x1, y1, x2, y2),
                    material=None,
                    voltage=None,
                    diameter=None,
                    location=((x1 + x2) / 2, (y1 + y2) / 2),
                    confidence=0.6,  # Lower confidence for visual detection
                    source_page=page_num,
                    extraction_method="visual_detection",
                    raw_text=f"Line detected at ({x1},{y1}) to ({x2},{y2})",
                )
                entities.append(entity)

        return entities

    def _analyze_text_for_cable(
        self, text: str, page_num: int, method: str
    ) -> Optional[CableEntity]:
        """
        Analyze text for cable-related information.

        Args:
            text: Text to analyze
            page_num: Page number
            method: Extraction method used

        Returns:
            CableEntity if cable information found, None otherwise
        """
        text_lower = text.lower()

        # Check if text contains cable-related keywords
        cable_keywords = [
            "cable",
            "wire",
            "conductor",
            "line",
            "fiber",
            "optical",
            "power",
        ]
        if not any(keyword in text_lower for keyword in cable_keywords):
            return None

        # Extract cable type
        cable_type = self._extract_cable_type(text_lower)

        # Extract material
        material = self._extract_material(text_lower)

        # Extract length
        length = self._extract_length(text_lower)

        # Extract voltage
        voltage = self._extract_voltage(text_lower)

        # Calculate confidence based on extracted information
        confidence = self._calculate_confidence(
            text_lower, cable_type, material, length, voltage
        )

        if confidence < self.confidence_threshold:
            return None

        # Generate unique ID
        entity_id = f"cable_{page_num}_{hash(text) % 10000}"

        return CableEntity(
            id=entity_id,
            type=cable_type or "unknown",
            length=length,
            material=material,
            voltage=voltage,
            diameter=None,  # Could be extracted with additional patterns
            location=None,  # Could be extracted from text position
            confidence=confidence,
            source_page=page_num,
            extraction_method=method,
            raw_text=text,
        )

    def _extract_cable_type(self, text: str) -> Optional[str]:
        """Extract cable type from text."""
        for cable_type, pattern in self.cable_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return cable_type
        return None

    def _extract_material(self, text: str) -> Optional[str]:
        """Extract material from text."""
        for material, pattern in self.material_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return material
        return None

    def _extract_length(self, text: str) -> Optional[float]:
        """Extract length from text."""
        for pattern in self.length_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower() if len(match.groups()) > 1 else None

                # Convert to meters
                if unit in ["ft", "feet"]:
                    value *= 0.3048
                elif unit in ["km", "kilometer"]:
                    value *= 1000

                return value
        return None

    def _extract_voltage(self, text: str) -> Optional[str]:
        """Extract voltage from text."""
        for pattern in self.voltage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                unit = match.group(2).lower() if len(match.groups()) > 1 else "v"
                return f"{value}{unit}"
        return None

    def _calculate_line_length(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate length of a line segment in pixels."""
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _calculate_confidence(
        self,
        text: str,
        cable_type: Optional[str],
        material: Optional[str],
        length: Optional[float],
        voltage: Optional[str],
    ) -> float:
        """
        Calculate confidence score for extracted entity.

        Args:
            text: Original text
            cable_type: Extracted cable type
            material: Extracted material
            length: Extracted length
            voltage: Extracted voltage

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.0

        # Base confidence for cable-related text
        if any(keyword in text for keyword in ["cable", "wire", "conductor"]):
            confidence += 0.3

        # Additional confidence for extracted attributes
        if cable_type:
            confidence += 0.2
        if material:
            confidence += 0.15
        if length:
            confidence += 0.15
        if voltage:
            confidence += 0.2

        # Bonus for multiple attributes
        attribute_count = sum(
            1 for attr in [cable_type, material, length, voltage] if attr is not None
        )
        if attribute_count >= 3:
            confidence += 0.1

        return min(confidence, 1.0)

    def save_entities(self, entities: List[CableEntity], output_path: str) -> str:
        """
        Save extracted entities to JSON file.

        Args:
            entities: List of CableEntity objects
            output_path: Output file path

        Returns:
            Path to saved file
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        # Convert entities to dictionaries
        entities_data = {
            "metadata": {
                "extraction_date": datetime.now().isoformat(),
                "total_entities": len(entities),
                "confidence_threshold": self.confidence_threshold,
                "source": "cable_entity_pipeline.py",
            },
            "entities": [entity.to_dict() for entity in entities],
        }

        # Save to JSON file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(entities_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(entities)} entities to: {output_path}")
            return output_path

        except Exception as e:
            raise IOError(f"Failed to save entities: {e}")

    def get_statistics(self, entities: List[CableEntity]) -> Dict[str, Any]:
        """
        Generate statistics for extracted entities.

        Args:
            entities: List of CableEntity objects

        Returns:
            Dictionary with statistics
        """
        if not entities:
            return {"total": 0}

        stats = {
            "total": len(entities),
            "by_type": {},
            "by_material": {},
            "by_extraction_method": {},
            "confidence_distribution": {
                "high": len([e for e in entities if e.confidence >= 0.8]),
                "medium": len([e for e in entities if 0.6 <= e.confidence < 0.8]),
                "low": len([e for e in entities if e.confidence < 0.6]),
            },
            "pages_processed": len(set(e.source_page for e in entities)),
        }

        # Count by type
        for entity in entities:
            entity_type = entity.type or "unknown"
            stats["by_type"][entity_type] = stats["by_type"].get(entity_type, 0) + 1

        # Count by material
        for entity in entities:
            material = entity.material or "unknown"
            stats["by_material"][material] = stats["by_material"].get(material, 0) + 1

        # Count by extraction method
        for entity in entities:
            method = entity.extraction_method
            stats["by_extraction_method"][method] = (
                stats["by_extraction_method"].get(method, 0) + 1
            )

        return stats


def main():
    """Main CLI function for cable entity extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract cable entities from plan sheets using text and visual analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cable_entity_pipeline.py input.pdf --output entities.json
  python cable_entity_pipeline.py input.pdf --pages 0 1 2 --confidence 0.7
  python cable_entity_pipeline.py input.pdf --all-pages --output detailed_entities.json
        """,
    )

    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output",
        "-o",
        default="cable_entities.json",
        help="Output JSON file path (default: cable_entities.json)",
    )
    parser.add_argument(
        "--pages", type=int, nargs="+", help="Specific pages to process (0-based)"
    )
    parser.add_argument("--all-pages", action="store_true", help="Process all pages")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--statistics", action="store_true", help="Print extraction statistics"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not os.path.exists(args.pdf_path):
        print(f"❌ Error: PDF file not found: {args.pdf_path}")
        return 1

    if args.confidence < 0 or args.confidence > 1:
        print("❌ Error: Confidence must be between 0 and 1")
        return 1

    try:
        # Create pipeline
        pipeline = CableEntityPipeline(confidence_threshold=args.confidence)

        # Determine pages to process
        pages = args.pages
        if args.all_pages:
            pages = None  # Process all pages

        # Extract entities
        print(f"🔍 Extracting cable entities from: {args.pdf_path}")
        entities = pipeline.parse_pdf(args.pdf_path, pages)

        if not entities:
            print("⚠️ No cable entities found")
            return 0

        # Save entities
        output_path = pipeline.save_entities(entities, args.output)
        print(f"✅ Saved {len(entities)} entities to: {output_path}")

        # Print statistics if requested
        if args.statistics:
            stats = pipeline.get_statistics(entities)
            print("\n📊 Extraction Statistics:")
            print(f"  Total entities: {stats['total']}")
            print(f"  Pages processed: {stats['pages_processed']}")
            print(f"  High confidence: {stats['confidence_distribution']['high']}")
            print(f"  Medium confidence: {stats['confidence_distribution']['medium']}")
            print(f"  Low confidence: {stats['confidence_distribution']['low']}")

            if stats["by_type"]:
                print("\n  By type:")
                for entity_type, count in stats["by_type"].items():
                    print(f"    {entity_type}: {count}")

            if stats["by_extraction_method"]:
                print("\n  By extraction method:")
                for method, count in stats["by_extraction_method"].items():
                    print(f"    {method}: {count}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
