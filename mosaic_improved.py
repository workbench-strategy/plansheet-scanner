#!/usr/bin/env python3
"""
North Arrow Detection Test - Focus on Template Rotation Scanning
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from dataclasses import dataclass
import cv2
import io
import math

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """Set up detailed logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('north_arrow_test.log'),
            logging.StreamHandler()
        ]
    )

@dataclass
class NorthDetectionResult:
    """Result of north arrow detection."""
    detected: bool
    angle: float  # Rotation angle needed to align north to top
    confidence: float
    template_angle: float  # Angle of template when best match found
    position: Tuple[int, int]  # Position of detected north arrow

class NorthArrowDetector:
    """Advanced north arrow detector using template rotation scanning."""
    
    def __init__(self, template_path: str = "templates/YetAnotherNorth/newnorth.png"):
        self.template_path = template_path
        self.logger = logging.getLogger(__name__)
        
        # Load template
        if not os.path.exists(template_path):
            self.logger.error(f"Template not found: {template_path}")
            self.template = None
        else:
            self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if self.template is None:
                self.logger.error(f"Could not load template: {template_path}")
            else:
                self.logger.info(f"Loaded template: {self.template.shape}")
    
    def detect_north_arrow(self, image: np.ndarray, rotation_step: float = 5.0) -> NorthDetectionResult:
        """
        Detect north arrow by rotating template and finding best match.
        
        Args:
            image: Grayscale image as numpy array
            rotation_step: Degrees to rotate template between attempts
            
        Returns:
            NorthDetectionResult with detection info
        """
        if self.template is None:
            return NorthDetectionResult(False, 0.0, 0.0, 0.0, (0, 0))
        
        self.logger.info(f"Scanning for north arrow with {rotation_step}° steps")
        
        best_confidence = 0.0
        best_angle = 0.0
        best_position = (0, 0)
        
        # Try different template rotations
        for angle in np.arange(0, 360, rotation_step):
            # Rotate template
            rotated_template = self._rotate_template(self.template, angle)
            
            # Template matching
            result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                best_angle = angle
                best_position = max_loc
                
                self.logger.debug(f"New best match at {angle}°: confidence {max_val:.3f}")
        
        # Determine if detection was successful
        threshold = 0.3
        detected = best_confidence >= threshold
        
        if detected:
            # Calculate rotation needed to align north to top
            # If template was rotated 45° to find match, we need to rotate image -45° to align north
            rotation_needed = -best_angle
            
            self.logger.info(f"North arrow detected!")
            self.logger.info(f"  Template rotation: {best_angle:.1f}°")
            self.logger.info(f"  Image rotation needed: {rotation_needed:.1f}°")
            self.logger.info(f"  Confidence: {best_confidence:.3f}")
            self.logger.info(f"  Position: {best_position}")
            
            return NorthDetectionResult(
                detected=True,
                angle=rotation_needed,
                confidence=best_confidence,
                template_angle=best_angle,
                position=best_position
            )
        else:
            self.logger.warning(f"No north arrow detected (best confidence: {best_confidence:.3f})")
            return NorthDetectionResult(False, 0.0, best_confidence, 0.0, (0, 0))
    
    def _rotate_template(self, template: np.ndarray, angle: float) -> np.ndarray:
        """Rotate template by given angle."""
        height, width = template.shape
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image size
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for new size
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(template, rotation_matrix, (new_width, new_height))
        
        return rotated
    
    def visualize_detection(self, image: np.ndarray, result: NorthDetectionResult, output_path: str):
        """Visualize the north arrow detection result."""
        if not result.detected:
            self.logger.warning("No detection to visualize")
            return
        
        # Convert to RGB for visualization
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image.copy()
        
        # Draw detection box
        template_h, template_w = self.template.shape
        x, y = result.position
        
        # Draw rectangle around detected north arrow
        cv2.rectangle(vis_image, (x, y), (x + template_w, y + template_h), (0, 255, 0), 2)
        
        # Draw text with detection info
        text = f"North: {result.angle:.1f}° (conf: {result.confidence:.3f})"
        cv2.putText(vis_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save visualization
        cv2.imwrite(output_path, vis_image)
        self.logger.info(f"Detection visualization saved to: {output_path}")

class NorthArrowTest:
    """Test class for north arrow detection."""
    
    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.detector = NorthArrowDetector()
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def test_sheet(self, page_num: int) -> NorthDetectionResult:
        """Test north arrow detection on a single sheet."""
        self.logger.info(f"\n=== Testing Sheet {page_num} ===")
        
        # Load page
        doc = fitz.open(self.pdf_path)
        page = doc.load_page(page_num)
        
        # Convert to grayscale numpy array
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Convert to grayscale
        gray_image = pil_image.convert('L')
        image_array = np.array(gray_image)
        
        self.logger.info(f"Image size: {image_array.shape}")
        
        # Detect north arrow
        result = self.detector.detect_north_arrow(image_array)
        
        # Visualize detection
        vis_path = os.path.join(self.output_dir, f"sheet_{page_num}_north_detection.png")
        self.detector.visualize_detection(image_array, result, vis_path)
        
        # If north arrow detected, rotate image and save
        if result.detected and abs(result.angle) > 1.0:  # Only rotate if significant
            self.logger.info(f"Rotating image by {result.angle:.1f}°")
            
            # Rotate PIL image
            rotated_pil = pil_image.rotate(-result.angle, expand=True)
            
            # Save rotated image
            rotated_path = os.path.join(self.output_dir, f"sheet_{page_num}_rotated.png")
            rotated_pil.save(rotated_path)
            self.logger.info(f"Rotated image saved to: {rotated_path}")
        
        doc.close()
        return result
    
    def test_multiple_sheets(self, max_sheets: int = 7):
        """Test north arrow detection on multiple sheets."""
        self.logger.info("Starting North Arrow Detection Test")
        
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        sheets_to_test = min(max_sheets, total_pages)
        
        self.logger.info(f"Testing {sheets_to_test} sheets out of {total_pages} total pages")
        
        results = []
        for page_num in range(sheets_to_test):
            result = self.test_sheet(page_num)
            results.append((page_num, result))
        
        # Summary
        self.logger.info("\n=== DETECTION SUMMARY ===")
        detected_count = sum(1 for _, result in results if result.detected)
        self.logger.info(f"Detected north arrows: {detected_count}/{len(results)}")
        
        for page_num, result in results:
            status = "✓" if result.detected else "✗"
            angle_info = f"({result.angle:.1f}°)" if result.detected else ""
            self.logger.info(f"Sheet {page_num}: {status} {angle_info} (conf: {result.confidence:.3f})")
        
        doc.close()
        return results

def main():
    parser = argparse.ArgumentParser(description='Test north arrow detection with template rotation')
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--sheets', type=int, default=7, help='Number of sheets to test')
    
    args = parser.parse_args()
    
    setup_logging()
    
    tester = NorthArrowTest(args.pdf, args.output)
    results = tester.test_multiple_sheets(args.sheets)
    
    print(f"\nTest complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()
