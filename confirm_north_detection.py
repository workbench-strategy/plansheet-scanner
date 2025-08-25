#!/usr/bin/env python3
"""
North Arrow Detection Confirmation Script
Shows exactly what's happening with north arrow detection
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

def detect_and_visualize_north_arrow(image: np.ndarray, template: np.ndarray, output_dir: str, sheet_num: int):
    """
    Detect north arrow and create visualizations for confirmation.
    """
    print(f"\n=== Sheet {sheet_num} North Arrow Detection ===")
    
    # Save original image
    cv2.imwrite(os.path.join(output_dir, f"sheet_{sheet_num}_original.png"), image)
    print(f"Saved original image: sheet_{sheet_num}_original.png")
    
    # Save template for reference
    cv2.imwrite(os.path.join(output_dir, f"sheet_{sheet_num}_template.png"), template)
    print(f"Saved template: sheet_{sheet_num}_template.png")
    
    best_confidence = 0.0
    best_angle = 0.0
    best_position = (0, 0)
    all_results = []
    
    # Try different template rotations
    print("Scanning template rotations...")
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:  # Test 8 key angles
        # Rotate template
        rotated_template = rotate_template(template, angle)
        
        # Template matching
        result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        all_results.append((angle, max_val, max_loc))
        
        if max_val > best_confidence:
            best_confidence = max_val
            best_angle = angle
            best_position = max_loc
            
            print(f"  New best: {angle}° rotation, confidence: {max_val:.3f}")
    
    # Create visualization of all results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (angle, confidence, position) in enumerate(all_results):
        # Create detection visualization
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Draw detection box
        template_h, template_w = template.shape
        x, y = position
        
        # Draw rectangle around detected north arrow
        cv2.rectangle(vis_image, (x, y), (x + template_w, y + template_h), (0, 255, 0), 3)
        
        # Draw text with detection info
        text = f"{angle}°: {confidence:.3f}"
        cv2.putText(vis_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Show in subplot
        axes[i].imshow(vis_image)
        axes[i].set_title(f"{angle}° rotation\nConfidence: {confidence:.3f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sheet_{sheet_num}_all_rotations.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved all rotation results: sheet_{sheet_num}_all_rotations.png")
    
    # Determine if detection was successful
    threshold = 0.3
    detected = best_confidence >= threshold
    
    if detected:
        # Calculate rotation needed to align north to top
        rotation_needed = -best_angle
        
        print(f"✅ North arrow detected!")
        print(f"   Template rotation: {best_angle}°")
        print(f"   Image rotation needed: {rotation_needed}°")
        print(f"   Confidence: {best_confidence:.3f}")
        print(f"   Position: {best_position}")
        
        # Create best match visualization
        best_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        template_h, template_w = template.shape
        x, y = best_position
        
        # Draw rectangle around detected north arrow
        cv2.rectangle(best_vis, (x, y), (x + template_w, y + template_h), (0, 255, 0), 5)
        
        # Draw text with detection info
        text = f"BEST MATCH: {best_angle}° rotation"
        cv2.putText(best_vis, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        text2 = f"Confidence: {best_confidence:.3f}"
        cv2.putText(best_vis, text2, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(output_dir, f"sheet_{sheet_num}_best_match.png"), best_vis)
        print(f"Saved best match: sheet_{sheet_num}_best_match.png")
        
    else:
        print(f"❌ No north arrow detected (best confidence: {best_confidence:.3f})")
    
    return detected, best_angle, best_confidence, best_position

def test_sheet(pdf_path: str, page_num: int, template_path: str, output_dir: str):
    """Test north arrow detection on a single sheet."""
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
    result = detect_and_visualize_north_arrow(image_array, template, output_dir, page_num)
    
    doc.close()
    return result

def main():
    parser = argparse.ArgumentParser(description='Confirm north arrow detection with visualizations')
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--template', default='templates/north_improved/north_arrow_1.png', 
                       help='Path to north arrow template')
    parser.add_argument('--sheets', type=int, default=3, help='Number of sheets to test')
    parser.add_argument('--output', default='north_detection_confirmation', help='Output directory')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("North Arrow Detection Confirmation")
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
    print("DETECTION SUMMARY")
    print("=" * 50)
    detected_count = sum(1 for _, (detected, _, _, _) in results if detected)
    print(f"Detected north arrows: {detected_count}/{len(results)}")
    
    for page_num, (detected, angle, confidence, position) in results:
        status = "✅" if detected else "❌"
        angle_info = f"({angle}° rotation)" if detected else ""
        print(f"Sheet {page_num}: {status} {angle_info} (conf: {confidence:.3f})")
    
    print(f"\n📁 Check the '{args.output}' directory for visualizations:")
    print(f"   - sheet_X_original.png: Original sheet image")
    print(f"   - sheet_X_template.png: North arrow template used")
    print(f"   - sheet_X_all_rotations.png: All rotation attempts")
    print(f"   - sheet_X_best_match.png: Best detection result")
    
    print(f"\n✅ Confirmation complete! Review the images to verify detection accuracy.")

if __name__ == "__main__":
    main()


