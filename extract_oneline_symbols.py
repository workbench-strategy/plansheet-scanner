#!/usr/bin/env python
"""
ITS One-Line Symbol Extractor

This script automatically extracts ITS symbols from one-line diagrams 
using contour detection and saves them as template images.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import re
import fitz  # PyMuPDF

def extract_items_from_oneline(image_path, output_dir="templates/oneline_items", min_area=200, max_area=5000):
    """
    Automatically extracts potential ITS elements from a one-line diagram
    using contour detection.
    
    Args:
        image_path: Path to the one-line diagram image
        output_dir: Directory to save extracted items
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
    
    Returns:
        List of saved template file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return []
        
    # Create a copy for output
    output_image = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} potential ITS elements")
    
    saved_files = []
    
    # Process each contour
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter out very small or large contours
        if min_area <= area <= max_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + padding * 2)
            h = min(image.shape[0] - y, h + padding * 2)
            
            # Extract the region
            roi = image[y:y+h, x:x+w]
            
            if roi.size > 0:
                # Save the extracted item
                filename = os.path.join(output_dir, f"its_element_{i}.png")
                cv2.imwrite(filename, roi)
                saved_files.append(filename)
                
                # Draw rectangle on output image for visualization
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                print(f"Saved: {filename} (Area: {area})")
    
    # Save the visualization image
    vis_file = os.path.join(output_dir, "extraction_visualization.jpg")
    cv2.imwrite(vis_file, output_image)
    print(f"Saved visualization to {vis_file}")
    
    return saved_files

def extract_symbols_with_text_recognition(image_path, output_dir="templates/oneline_items"):
    """
    Extract ITS symbols from a one-line diagram using more advanced
    image processing and text recognition.
    
    This approach uses contour detection combined with Canny edge detection
    to better isolate ITS elements in the diagram.
    
    Args:
        image_path: Path to the one-line diagram image
        output_dir: Directory to save extracted items
    
    Returns:
        List of saved template file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return []
        
    # Create a copy for output
    output_image = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} potential regions")
    
    saved_files = []
    
    # Process each contour
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter out very small or large contours
        if 200 <= area <= 10000:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + padding * 2)
            h = min(image.shape[0] - y, h + padding * 2)
            
            # Check aspect ratio to filter out long thin rectangles (likely lines)
            aspect_ratio = float(w) / h
            if 0.2 <= aspect_ratio <= 5.0:  # Filter extreme aspect ratios
                # Extract the region
                roi = image[y:y+h, x:x+w]
                
                if roi.size > 0:
                    # Save the extracted item
                    filename = os.path.join(output_dir, f"its_element_{i}.png")
                    cv2.imwrite(filename, roi)
                    saved_files.append(filename)
                    
                    # Draw rectangle on output image for visualization
                    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    print(f"Saved: {filename} (Area: {area})")
    
    # Save the visualization image
    vis_file = os.path.join(output_dir, "extraction_visualization.jpg")
    cv2.imwrite(vis_file, output_image)
    print(f"Saved visualization to {vis_file}")
    
    return saved_files

def extract_interactive(image_path, output_dir="templates/oneline_items", max_display_size=1600):
    """
    Interactive extraction of ITS elements from a one-line diagram.
    Allows the user to select regions manually with zooming capabilities.
    
    Controls:
    - Click and drag to select a region
    - +/= to zoom in
    - - to zoom out
    - Arrow keys to pan when zoomed in
    - 'r' to reset view
    
    Args:
        image_path: Path to the one-line diagram image
        output_dir: Directory to save extracted items
        max_display_size: Maximum dimension for display
        
    Returns:
        List of saved template file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not read image {image_path}")
        return []
    
    # Store original dimensions
    original_height, original_width = original_img.shape[:2]
    print(f"Original image dimensions: {original_width}x{original_height} pixels")
    
    # Determine if we need to scale for initial display
    scale_factor = 1.0
    if original_height > max_display_size or original_width > max_display_size:
        if original_height > original_width:
            scale_factor = max_display_size / original_height
        else:
            scale_factor = max_display_size / original_width
        
        display_width = int(original_width * scale_factor)
        display_height = int(original_height * scale_factor)
        
        # Resize for display
        initial_display_img = cv2.resize(original_img, (display_width, display_height))
        print(f"Scaled for display: {display_width}x{display_height} pixels (scale factor: {scale_factor:.3f})")
    else:
        initial_display_img = original_img.copy()
    
    # Variables for zooming and panning
    zoom_level = 1.0
    offset_x, offset_y = 0, 0
    current_img = initial_display_img.copy()
    base_scale_factor = scale_factor
    
    # Keep original for extraction
    clone = original_img.copy()
    ref_pt = []
    cropping = False
    roi_count = 0
    saved_files = []
    element_name = ""
    
    def update_view():
        """Update the displayed image based on zoom level and offset"""
        nonlocal current_img, zoom_level, offset_x, offset_y, initial_display_img
        
        # Calculate the actual zoom and scale from original
        actual_scale = base_scale_factor / zoom_level
        
        # Determine the region of the original image to display
        h, w = original_img.shape[:2]
        view_h, view_w = int(h * actual_scale), int(w * actual_scale)
        
        # Calculate the center point adjusted by offset
        center_x = w // 2 - int(offset_x / actual_scale)
        center_y = h // 2 - int(offset_y / actual_scale)
        
        # Calculate the top-left corner of the view
        x1 = max(0, center_x - view_w // 2)
        y1 = max(0, center_y - view_h // 2)
        
        # Adjust if near the right/bottom edge
        x1 = min(x1, w - view_w)
        y1 = min(y1, h - view_h)
        
        # Ensure we don't go below 0
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        # Extract the region and resize to fit the display window
        region = original_img[y1:y1+view_h, x1:x1+view_w]
        if region.size > 0:  # Check if region is valid
            current_img = cv2.resize(region, (initial_display_img.shape[1], initial_display_img.shape[0]))
        else:
            # If the region is invalid, reset view
            zoom_level = 1.0
            offset_x, offset_y = 0, 0
            current_img = initial_display_img.copy()
    
    def click_and_crop(event, x, y, flags, param):
        nonlocal cropping, roi_count, element_name, ref_pt, zoom_level, base_scale_factor
        
        # Calculate the actual scale factor considering zoom
        actual_scale = base_scale_factor / zoom_level
        
        # Adjust coordinates for zoom and pan
        orig_x = int((x + offset_x) / actual_scale)
        orig_y = int((y + offset_y) / actual_scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_pt.clear()
            ref_pt.append((orig_x, orig_y))  # Store original image coordinates
            cropping = True
            
        elif event == cv2.EVENT_LBUTTONUP:
            if cropping:
                ref_pt.append((orig_x, orig_y))  # Store original image coordinates
                cropping = False
                
                # Create a copy of the current display image
                display_img = current_img.copy()
                
                # Convert original coordinates to display coordinates
                display_x1 = int((ref_pt[0][0] * actual_scale) - offset_x)
                display_y1 = int((ref_pt[0][1] * actual_scale) - offset_y)
                display_x2 = int((ref_pt[1][0] * actual_scale) - offset_x)
                display_y2 = int((ref_pt[1][1] * actual_scale) - offset_y)
                
                # Draw rectangle on display image
                cv2.rectangle(display_img, (display_x1, display_y1), (display_x2, display_y2), (0, 255, 0), 2)
                cv2.imshow("One-Line Diagram", display_img)
                
                # Extract ROI from original image
                x_min = min(ref_pt[0][0], ref_pt[1][0])
                y_min = min(ref_pt[0][1], ref_pt[1][1])
                x_max = max(ref_pt[0][0], ref_pt[1][0])
                y_max = max(ref_pt[0][1], ref_pt[1][1])
                
                # Ensure coordinates are within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(original_img.shape[1], x_max)
                y_max = min(original_img.shape[0], y_max)
                
                roi = clone[y_min:y_max, x_min:x_max]
                
                if roi.size > 0:
                    # Prompt for element name
                    print("Enter name for this ITS element (or press Enter for auto-naming):")
                    element_name = input().strip()
                    
                    # Generate filename
                    if element_name:
                        # Clean up filename
                        element_name = re.sub(r'[^\w\s-]', '', element_name)
                        element_name = re.sub(r'[-\s]+', '-', element_name)
                        filename = os.path.join(output_dir, f"{element_name}.png")
                    else:
                        filename = os.path.join(output_dir, f"its_element_{roi_count}.png")
                    
                    # Save the ROI
                    cv2.imwrite(filename, roi)
                    saved_files.append(filename)
                    roi_count += 1
                    print(f"Saved: {filename}")
    
    # Create window and set it to be resizable
    cv2.namedWindow("One-Line Diagram", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("One-Line Diagram", initial_display_img.shape[1], initial_display_img.shape[0])
    cv2.setMouseCallback("One-Line Diagram", click_and_crop)
    
    print("🖱️ Select each ITS element from the one-line diagram:")
    print("   1. Click and drag to create a rectangle around the element")
    print("   2. Release the mouse button to save")
    print("   3. Enter a name for the element when prompted")
    print("   4. Press ESC when done")
    print("\nZoom and Navigation Controls:")
    print("   - Press '+' or '=' to zoom in")
    print("   - Press '-' to zoom out")
    print("   - Use arrow keys to pan when zoomed in")
    print("   - Press 'r' to reset view")
    
    while True:
        cv2.imshow("One-Line Diagram", current_img)
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key to exit
        if key == 27:  
            break
            
        # Zoom in with + or =
        elif key == ord('+') or key == ord('='):
            zoom_level = min(10.0, zoom_level * 1.2)  # Limit max zoom
            update_view()
            print(f"Zoom level: {zoom_level:.2f}x")
            
        # Zoom out with -
        elif key == ord('-'):
            zoom_level = max(1.0, zoom_level / 1.2)  # Limit min zoom
            update_view()
            print(f"Zoom level: {zoom_level:.2f}x")
            
        # Pan with arrow keys
        elif key == 82 or key == ord('w'):  # Up arrow or 'w'
            offset_y -= 50 * zoom_level
            update_view()
        elif key == 84 or key == ord('s'):  # Down arrow or 's'
            offset_y += 50 * zoom_level
            update_view()
        elif key == 81 or key == ord('a'):  # Left arrow or 'a'
            offset_x -= 50 * zoom_level
            update_view()
        elif key == 83 or key == ord('d'):  # Right arrow or 'd'
            offset_x += 50 * zoom_level
            update_view()
            
        # Reset view with 'r'
        elif key == ord('r'):
            zoom_level = 1.0
            offset_x, offset_y = 0, 0
            current_img = initial_display_img.copy()
            print("View reset")
    
    cv2.destroyAllWindows()
    return saved_files

def extract_from_pdf(pdf_path, page_number, output_dir, mode="interactive", min_area=200, max_area=5000, max_display_size=1600):
    """
    Extract ITS elements from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to extract from (0-based)
        output_dir: Directory to save extracted items
        mode: Extraction mode ('auto', 'advanced', or 'interactive')
        min_area: Minimum contour area for auto mode
        max_area: Maximum contour area for auto mode
        max_display_size: Maximum display dimension for interactive mode
        
    Returns:
        List of saved file paths
    """
    # Create a temporary directory for the image if it doesn't exist
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Convert PDF page to image
    doc = fitz.open(pdf_path)
    if page_number < 0 or page_number >= doc.page_count:
        print(f"Error: Page number {page_number} is out of range for PDF with {doc.page_count} pages.")
        return []
    
    # Get the page and render it at high resolution (300 DPI)
    page = doc[page_number]
    pix = page.get_pixmap(dpi=300)
    temp_image_path = os.path.join(temp_dir, f"temp_page_{page_number}.png")
    pix.save(temp_image_path)
    
    print(f"Converted PDF page {page_number+1} to image: {temp_image_path}")
    
    # Extract from the image
    if mode == "auto":
        saved_files = extract_items_from_oneline(temp_image_path, output_dir, min_area, max_area)
    elif mode == "advanced":
        saved_files = extract_symbols_with_text_recognition(temp_image_path, output_dir)
    else:  # interactive
        saved_files = extract_interactive(temp_image_path, output_dir, max_display_size)
    
    # Optionally clean up the temporary file
    # os.remove(temp_image_path)
    
    return saved_files

def main():
    parser = argparse.ArgumentParser(description="Extract ITS elements from one-line diagrams")
    
    # Input source group
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to the one-line diagram image")
    input_group.add_argument("--pdf", help="Path to the PDF file")
    
    # PDF specific options
    parser.add_argument("--page", type=int, default=0, help="Page number for PDF (0-based index)")
    
    # Common options
    parser.add_argument("--output_dir", default="templates/oneline_items", 
                        help="Directory to save extracted items")
    parser.add_argument("--mode", choices=["auto", "advanced", "interactive"], default="interactive",
                        help="Extraction mode: auto (basic contours), advanced (edge detection), or interactive")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum contour area (auto mode)")
    parser.add_argument("--max-area", type=int, default=5000, help="Maximum contour area (auto mode)")
    parser.add_argument("--max-display-size", type=int, default=1600, 
                        help="Maximum dimension for display (interactive mode)")
    
    args = parser.parse_args()
    
    # Determine the input source
    input_path = args.image if args.image else args.pdf
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    try:
        # Extract ITS elements based on input type and selected mode
        if args.pdf:
            saved_files = extract_from_pdf(
                args.pdf,
                args.page,
                args.output_dir,
                args.mode,
                args.min_area,
                args.max_area,
                args.max_display_size
            )
        else:  # args.image is set
            if args.mode == "auto":
                saved_files = extract_items_from_oneline(
                    args.image,
                    args.output_dir,
                    args.min_area,
                    args.max_area
                )
            elif args.mode == "advanced":
                saved_files = extract_symbols_with_text_recognition(
                    args.image,
                    args.output_dir
                )
            else:  # interactive
                saved_files = extract_interactive(
                    args.image,
                    args.output_dir,
                    args.max_display_size
                )
        
        # Check if any files were saved
        if saved_files:
            print(f"\nExtracted {len(saved_files)} ITS elements. Saved to: {args.output_dir}")
        else:
            print("\nNo elements were extracted. Try adjusting parameters or using a different mode.")
            
        return 0
        
    except KeyboardInterrupt:
        print("\nExtraction canceled by user.")
        return 130
    except Exception as e:
        print(f"\nError during extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
