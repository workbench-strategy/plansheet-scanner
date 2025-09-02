"""
North Arrow Detector

Detects and analyzes north arrows in engineering drawings to understand orientation.
This is a critical foundation element for proper spatial understanding.
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
class NorthArrowDetection:
    """Result of north arrow detection."""

    detected: bool
    angle: float  # Rotation angle in degrees
    confidence: float
    position: Tuple[int, int]  # (x, y) coordinates
    size: Tuple[int, int]  # (width, height)
    arrow_type: str  # "standard", "magnetic", "grid", "true"
    text_associated: Optional[str] = None
    template_matched: Optional[str] = None


class NorthArrowDetector:
    """
    Advanced north arrow detection system.

    Features:
    - Template matching with rotation
    - OCR for "NORTH" text detection
    - Multiple arrow type recognition
    - Confidence scoring
    - Position and orientation analysis
    """

    def __init__(self, template_dir: str = "templates/north_arrows"):
        """
        Initialize the north arrow detector.

        Args:
            template_dir: Directory containing north arrow templates
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)

        # Load templates
        self.templates = self._load_templates()

        # OCR setup for text detection
        self.north_text_patterns = [
            r"\bNORTH\b",
            r"\bN\b",
            r"\bTRUE NORTH\b",
            r"\bMAGNETIC NORTH\b",
            r"\bGRID NORTH\b",
        ]

        # Detection parameters
        self.min_confidence = 0.6
        self.rotation_step = 5.0  # degrees
        self.max_rotation = 360.0

        logger.info(
            f"NorthArrowDetector initialized with {len(self.templates)} templates"
        )

    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Load north arrow templates from directory."""
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

    def detect_north_arrow(self, image: np.ndarray) -> NorthArrowDetection:
        """
        Detect north arrow in the given image.

        Args:
            image: Input image as numpy array (grayscale or BGR)

        Returns:
            NorthArrowDetection object with detection results
        """
        if image is None:
            return NorthArrowDetection(
                detected=False,
                angle=0.0,
                confidence=0.0,
                position=(0, 0),
                size=(0, 0),
                arrow_type="unknown",
            )

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Template matching
        template_result = self._detect_by_template_matching(gray)

        # 2. Text-based detection
        text_result = self._detect_by_text(gray)

        # 3. Shape-based detection
        shape_result = self._detect_by_shape(gray)

        # Combine results
        best_result = self._combine_detection_results(
            template_result, text_result, shape_result
        )

        return best_result

    def _detect_by_template_matching(
        self, gray_image: np.ndarray
    ) -> Optional[NorthArrowDetection]:
        """Detect north arrow using template matching with rotation."""
        best_result = None
        best_confidence = 0.0

        for template_name, template in self.templates.items():
            # Try different rotations
            for angle in np.arange(0, self.max_rotation, self.rotation_step):
                # Rotate template
                rotated_template = self._rotate_image(template, angle)

                # Template matching
                result = cv2.matchTemplate(
                    gray_image, rotated_template, cv2.TM_CCOEFF_NORMED
                )
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > best_confidence and max_val > self.min_confidence:
                    best_confidence = max_val
                    best_result = NorthArrowDetection(
                        detected=True,
                        angle=angle,
                        confidence=max_val,
                        position=max_loc,
                        size=rotated_template.shape[::-1],  # (width, height)
                        arrow_type=self._classify_arrow_type(template_name),
                        template_matched=template_name,
                    )

        return best_result

    def _detect_by_text(self, gray_image: np.ndarray) -> Optional[NorthArrowDetection]:
        """Detect north arrow by finding 'NORTH' text."""
        # This would integrate with OCR (pytesseract)
        # For now, return None - would need OCR implementation
        return None

    def _detect_by_shape(self, gray_image: np.ndarray) -> Optional[NorthArrowDetection]:
        """Detect north arrow by analyzing shapes and contours."""
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_result = None
        best_confidence = 0.0

        for contour in contours:
            # Analyze contour properties
            area = cv2.contourArea(contour)
            if area < 100 or area > 10000:  # Filter by size
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0

            # Check if it looks like an arrow (pointed shape)
            confidence = self._calculate_arrow_confidence(contour, aspect_ratio)

            if confidence > best_confidence and confidence > self.min_confidence:
                best_confidence = confidence
                best_result = NorthArrowDetection(
                    detected=True,
                    angle=self._estimate_arrow_angle(contour),
                    confidence=confidence,
                    position=(x, y),
                    size=(w, h),
                    arrow_type="shape_detected",
                )

        return best_result

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

        return rotated

    def _calculate_arrow_confidence(
        self, contour: np.ndarray, aspect_ratio: float
    ) -> float:
        """Calculate confidence that a contour represents an arrow."""
        # Analyze contour shape
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        # Circularity (arrows are not circular)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

        # Aspect ratio confidence (arrows are usually elongated)
        aspect_confidence = min(1.0, aspect_ratio / 2.0) if aspect_ratio > 0 else 0

        # Combine factors
        confidence = (1.0 - circularity) * 0.6 + aspect_confidence * 0.4

        return confidence

    def _estimate_arrow_angle(self, contour: np.ndarray) -> float:
        """Estimate the angle of an arrow contour."""
        # Fit ellipse to contour
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]  # Rotation angle
            return angle
        else:
            # Fallback to bounding rectangle
            rect = cv2.minAreaRect(contour)
            return rect[2]

    def _classify_arrow_type(self, template_name: str) -> str:
        """Classify the type of north arrow based on template name."""
        template_name_lower = template_name.lower()

        if "magnetic" in template_name_lower:
            return "magnetic"
        elif "grid" in template_name_lower:
            return "grid"
        elif "true" in template_name_lower:
            return "true"
        else:
            return "standard"

    def _combine_detection_results(
        self,
        template_result: Optional[NorthArrowDetection],
        text_result: Optional[NorthArrowDetection],
        shape_result: Optional[NorthArrowDetection],
    ) -> NorthArrowDetection:
        """Combine multiple detection results to get the best one."""
        results = [
            r for r in [template_result, text_result, shape_result] if r is not None
        ]

        if not results:
            return NorthArrowDetection(
                detected=False,
                angle=0.0,
                confidence=0.0,
                position=(0, 0),
                size=(0, 0),
                arrow_type="unknown",
            )

        # Return the result with highest confidence
        best_result = max(results, key=lambda r: r.confidence)
        return best_result

    def detect_from_pdf(
        self, pdf_path: str, page_number: int = 0
    ) -> NorthArrowDetection:
        """
        Detect north arrow from a PDF page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-based)

        Returns:
            NorthArrowDetection object
        """
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            if page_number >= doc.page_count:
                logger.error(f"Page number {page_number} out of range")
                return NorthArrowDetection(
                    detected=False,
                    angle=0.0,
                    confidence=0.0,
                    position=(0, 0),
                    size=(0, 0),
                    arrow_type="unknown",
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

            # Detect north arrow
            result = self.detect_north_arrow(image_array)

            doc.close()
            return result

        except Exception as e:
            logger.error(f"Error detecting north arrow from PDF: {e}")
            return NorthArrowDetection(
                detected=False,
                angle=0.0,
                confidence=0.0,
                position=(0, 0),
                size=(0, 0),
                arrow_type="unknown",
            )

    def add_template(self, template_path: str, template_name: str) -> bool:
        """
        Add a new north arrow template.

        Args:
            template_path: Path to template image
            template_name: Name for the template

        Returns:
            True if template was added successfully
        """
        try:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                logger.error(f"Could not load template: {template_path}")
                return False

            # Save template
            template_file = self.template_dir / f"{template_name}.png"
            cv2.imwrite(str(template_file), template)

            # Add to loaded templates
            self.templates[template_name] = template

            logger.info(f"Added template: {template_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return False


def main():
    """Test the north arrow detector."""
    import argparse

    parser = argparse.ArgumentParser(description="North Arrow Detector")
    parser.add_argument("input", help="Input image or PDF file")
    parser.add_argument("--page", type=int, default=0, help="PDF page number (0-based)")
    parser.add_argument(
        "--template-dir", default="templates/north_arrows", help="Template directory"
    )

    args = parser.parse_args()

    # Initialize detector
    detector = NorthArrowDetector(args.template_dir)

    # Detect north arrow
    if args.input.lower().endswith(".pdf"):
        result = detector.detect_from_pdf(args.input, args.page)
    else:
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        result = detector.detect_north_arrow(image)

    # Print results
    print(f"North Arrow Detection Results:")
    print(f"  Detected: {result.detected}")
    if result.detected:
        print(f"  Angle: {result.angle:.1f}°")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Position: {result.position}")
        print(f"  Size: {result.size}")
        print(f"  Type: {result.arrow_type}")
        if result.template_matched:
            print(f"  Template: {result.template_matched}")


if __name__ == "__main__":
    main()
