#!/usr/bin/env python3
"""
Debug script to check highlight coordinate accuracy.
This script will show exactly what coordinates are being found and where highlights are being placed.
"""

import fitz  # PyMuPDF
import sys
import os
from pathlib import Path

def debug_highlight_coordinates(pdf_path, search_text="CCTV"):
    """Debug the coordinate system for highlighting."""
    print(f"Debugging highlight coordinates for: {pdf_path}")
    print(f"Searching for text: '{search_text}'")
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        print(f"\nPDF has {len(doc)} pages")
        
        for page_num, page in enumerate(doc):
            print(f"\n=== Page {page_num + 1} ===")
            
            # Get page dimensions
            page_rect = page.rect
            print(f"Page dimensions: {page_rect}")
            print(f"Page width: {page_rect.width:.2f} points, height: {page_rect.height:.2f} points")
            
            # Search for the text
            text_instances = page.search_for(search_text)
            print(f"Found {len(text_instances)} instances of '{search_text}'")
            
            if text_instances:
                for i, inst in enumerate(text_instances):
                    print(f"\n  Instance {i+1}:")
                    print(f"    Raw coordinates: {inst}")
                    print(f"    x0: {inst[0]:.2f}, y0: {inst[1]:.2f}")
                    print(f"    x1: {inst[2]:.2f}, y1: {inst[3]:.2f}")
                    print(f"    Width: {inst[2] - inst[0]:.2f}, Height: {inst[3] - inst[1]:.2f}")
                    
                    # Check if coordinates are within page bounds
                    if (inst[0] >= 0 and inst[1] >= 0 and 
                        inst[2] <= page_rect.width and inst[3] <= page_rect.height):
                        print(f"    ✅ Coordinates are within page bounds")
                    else:
                        print(f"    ❌ Coordinates are OUTSIDE page bounds!")
                    
                    # Get the actual text at these coordinates
                    try:
                        text_at_coords = page.get_text("text", clip=inst)
                        print(f"    Text at coordinates: '{text_at_coords.strip()}'")
                    except Exception as e:
                        print(f"    Error getting text: {e}")
                    
                    # Try to add a highlight and see what happens
                    try:
                        # Create a highlight annotation
                        highlight = page.add_highlight_annot(inst)
                        print(f"    ✅ Highlight annotation created successfully")
                        
                        # Get the highlight's rectangle
                        if hasattr(highlight, 'rect'):
                            print(f"    Highlight rect: {highlight.rect}")
                        
                        # Remove the highlight for now (we're just testing)
                        page.delete_annot(highlight)
                        
                    except Exception as e:
                        print(f"    ❌ Error creating highlight: {e}")
                    
                    # Also try drawing a rectangle
                    try:
                        page.draw_rect(inst, color=(1, 0, 0), width=2)  # Red rectangle
                        print(f"    ✅ Rectangle drawn successfully")
                    except Exception as e:
                        print(f"    ❌ Error drawing rectangle: {e}")
            
            # Also check if there are any existing annotations
            annotations = list(page.annots())
            if annotations:
                print(f"\n  Existing annotations on page: {len(annotations)}")
                for j, annot in enumerate(annotations[:3]):  # Show first 3
                    print(f"    Annotation {j+1}: Type={annot.type}, Rect={annot.rect}")
        
        # Save a test version with highlights
        test_output = pdf_path.replace('.pdf', '_debug_highlights.pdf')
        doc.save(test_output)
        print(f"\nTest PDF with highlights saved to: {test_output}")
        
        doc.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        search_text = sys.argv[2] if len(sys.argv) > 2 else "CCTV"
    else:
        # Try to find a PDF in the current directory
        pdf_files = list(Path(".").glob("*.pdf"))
        if not pdf_files:
            print("Error: No PDF files found in current directory")
            print("Usage: python debug_highlight_coordinates.py <pdf_file> [search_text]")
            return
        
        pdf_path = str(pdf_files[0])
        search_text = "CCTV"
        print(f"Using PDF: {pdf_path}")
        print(f"Searching for: {search_text}")
    
    debug_highlight_coordinates(pdf_path, search_text)

if __name__ == "__main__":
    main()

