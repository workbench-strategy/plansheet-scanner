#!/usr/bin/env python3
"""
Efficient North Arrow Detector
Two-stage approach: coarse detection then fine-tuning
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import cv2
import io
import matplotlib.pyplot as plt

def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def rotate_template(template: np.ndarray, angle: float) -> np.ndarray:
    """Rotate template by given angle with precise rotation."""
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

def efficient_north_arrow_detection(image: np.ndarray, template: np.ndarray, output_dir: str, sheet_num: int):
    """
    Efficient north arrow detection with two-stage approach.
    """
    print(f"\n=== Sheet {sheet_num} Efficient North Arrow Detection ===")
    
    # Save original image
    cv2.imwrite(os.path.join(output_dir, f"sheet_{sheet_num}_original.png"), image)
    print(f"Saved original image: sheet_{sheet_num}_original.png")
    
    best_confidence = 0.0
    best_angle = 0.0
    best_position = (0, 0)
    
    # Stage 1: Coarse detection (15° steps)
    print("Stage 1: Coarse detection (15° steps)...")
    coarse_angles = np.arange(0, 360, 15)
    
    for angle in coarse_angles:
        # Rotate template
        rotated_template = rotate_template(template, angle)
        
        # Template matching
        result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_confidence:
            best_confidence = max_val
            best_angle = angle
            best_position = max_loc
            
            print(f"  Coarse best: {angle}° rotation, confidence: {max_val:.3f}")
    
    # Stage 2: Fine-tuning around best coarse angle (1° steps)
    print("Stage 2: Fine-tuning around best angle...")
    fine_start = max(0, best_angle - 15)
    fine_end = min(360, best_angle + 15)
    fine_angles = np.arange(fine_start, fine_end, 1)
    
    for angle in fine_angles:
        # Rotate template
        rotated_template = rotate_template(template, angle)
        
        # Template matching
        result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_confidence:
            best_confidence = max_val
            best_angle = angle
            best_position = max_loc
            
            print(f"  Fine best: {angle:.1f}° rotation, confidence: {max_val:.3f}")
    
    # Determine if detection was successful
    threshold = 0.35  # Balanced threshold
    detected = best_confidence >= threshold
    
    if detected:
        # Calculate rotation needed to align north to top
        rotation_needed = -best_angle
        
        print(f"✅ North arrow detected!")
        print(f"   Template rotation: {best_angle:.1f}°")
        print(f"   Image rotation needed: {rotation_needed:.1f}°")
        print(f"   Confidence: {best_confidence:.3f}")
        print(f"   Position: {best_position}")
        
        # Create visualization
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        x, y = best_position
        template_h, template_w = template.shape
        
        # Draw rectangle around detected north arrow
        cv2.rectangle(vis_image, (x, y), (x + template_w, y + template_h), (0, 255, 0), 3)
        
        # Draw text with detection info
        text = f"BEST: {best_angle:.1f}° rotation (conf: {best_confidence:.3f})"
        cv2.putText(vis_image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        text2 = f"Rotation needed: {rotation_needed:.1f}°"
        cv2.putText(vis_image, text2, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(output_dir, f"sheet_{sheet_num}_efficient_detection.png"), vis_image)
        print(f"Saved efficient detection: sheet_{sheet_num}_efficient_detection.png")
        
        # Save rotated image for verification
        if abs(rotation_needed) > 1.0:  # Only rotate if significant
            # Rotate the original PIL image
            pil_image = Image.fromarray(image)
            rotated_pil = pil_image.rotate(rotation_needed, expand=True)
            rotated_array = np.array(rotated_pil)
            
            # Save rotated image
            cv2.imwrite(os.path.join(output_dir, f"sheet_{sheet_num}_rotated.png"), rotated_array)
            print(f"Saved rotated image: sheet_{sheet_num}_rotated.png")
        
    else:
        print(f"❌ No north arrow detected (best confidence: {best_confidence:.3f})")
    
    return detected, best_angle, best_confidence, best_position

def test_sheet(pdf_path: str, page_num: int, template_path: str, output_dir: str):
    """Test efficient north arrow detection on a single sheet."""
    print(f"\n=== Testing Sheet {page_num} ===")
    
    # Load template
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        return None
    
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"❌ Could not load template: {template_path}")
        return None
    
    print(f"✅ Loaded template: {template.shape}")
    
    # Load page
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    
    # Convert to grayscale numpy array
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
    img_data = pix.tobytes("png")
    pil_image = Image.open(io.BytesIO(img_data))
    
    # Convert to grayscale
    gray_image = pil_image.convert('L')
    image_array = np.array(gray_image)
    
    print(f"✅ Image size: {image_array.shape}")
    
    # Detect and visualize north arrow
    result = efficient_north_arrow_detection(image_array, template, output_dir, page_num)
    
    doc.close()
    return result

def main():
    parser = argparse.ArgumentParser(description='Efficient north arrow detection with two-stage approach')
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--template', default='templates/north_improved/north_arrow_1.png', 
                       help='Path to north arrow template')
    parser.add_argument('--sheets', type=int, default=2, help='Number of sheets to test')
    parser.add_argument('--output', default='efficient_north_detection', help='Output directory')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("Efficient North Arrow Detection")
    print("=" * 50)
    print(f"PDF: {args.pdf}")
    print(f"Template: {args.template}")
    print(f"Output directory: {args.output}")
    print(f"Testing {args.sheets} sheets")
    print("=" * 50)
    
    results = []
    for page_num in range(args.sheets):
        result = test_sheet(args.pdf, page_num, args.template, args.output)
        if result is not None:
            results.append((page_num, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("EFFICIENT DETECTION SUMMARY")
    print("=" * 50)
    detected_count = sum(1 for _, (detected, _, _, _) in results if detected)
    print(f"Detected north arrows: {detected_count}/{len(results)}")
    
    for page_num, (detected, angle, confidence, position) in results:
        status = "✅" if detected else "❌"
        angle_info = f"({angle:.1f}° rotation)" if detected else ""
        print(f"Sheet {page_num}: {status} {angle_info} (conf: {confidence:.3f})")
    
    print(f"\n📁 Check the '{args.output}' directory for visualizations:")
    print(f"   - sheet_X_original.png: Original sheet image")
    print(f"   - sheet_X_efficient_detection.png: Efficient detection result")
    print(f"   - sheet_X_rotated.png: Rotated image (if significant rotation)")
    
    print(f"\n✅ Efficient detection complete! Features:")
    print(f"   - Two-stage approach: coarse (15°) then fine (1°)")
    print(f"   - Precise angle calculation")
    print(f"   - Fast and reliable")
    print(f"   - Automatic image rotation")

if __name__ == "__main__":
    main()


