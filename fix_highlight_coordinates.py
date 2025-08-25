#!/usr/bin/env python3
"""
Fix script for highlight coordinate issues.
This script provides corrected coordinate handling for PDF highlighting.
"""

import fitz  # PyMuPDF
import sys
import os
from pathlib import Path

def fix_highlight_coordinates(pdf_path, search_text="CCTV"):
    """Fix highlight coordinate issues by ensuring proper coordinate system."""
    print(f"Fixing highlight coordinates for: {pdf_path}")
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
            
            # Get page dimensions and transformation matrix
            page_rect = page.rect
            print(f"Page dimensions: {page_rect}")
            
            # Check if page is rotated
            rotation = page.rotation
            print(f"Page rotation: {rotation} degrees")
            
            # Search for the text
            text_instances = page.search_for(search_text)
            print(f"Found {len(text_instances)} instances of '{search_text}'")
            
            if text_instances:
                for i, inst in enumerate(text_instances):
                    print(f"\n  Instance {i+1}:")
                    print(f"    Original coordinates: {inst}")
                    
                    # Convert to fitz.Rect for better handling
                    rect = fitz.Rect(inst)
                    print(f"    As fitz.Rect: {rect}")
                    
                    # Handle page rotation if needed
                    if rotation != 0:
                        print(f"    Page is rotated {rotation} degrees - coordinates may need adjustment")
                        
                        # For rotated pages, we might need to transform coordinates
                        # This is a simplified approach - you might need more sophisticated handling
                        if rotation == 90:
                            # 90 degree rotation: (x, y) -> (y, width-x)
                            adjusted_rect = fitz.Rect(
                                rect.y0, 
                                page_rect.width - rect.x1,
                                rect.y1, 
                                page_rect.width - rect.x0
                            )
                        elif rotation == 180:
                            # 180 degree rotation: (x, y) -> (width-x, height-y)
                            adjusted_rect = fitz.Rect(
                                page_rect.width - rect.x1,
                                page_rect.height - rect.y1,
                                page_rect.width - rect.x0,
                                page_rect.height - rect.y0
                            )
                        elif rotation == 270:
                            # 270 degree rotation: (x, y) -> (height-y, x)
                            adjusted_rect = fitz.Rect(
                                page_rect.height - rect.y1,
                                rect.x0,
                                page_rect.height - rect.y0,
                                rect.x1
                            )
                        else:
                            adjusted_rect = rect
                        
                        print(f"    Adjusted for rotation: {adjusted_rect}")
                        rect = adjusted_rect
                    
                    # Ensure coordinates are within page bounds
                    if rect.x0 < 0:
                        rect.x0 = 0
                    if rect.y0 < 0:
                        rect.y0 = 0
                    if rect.x1 > page_rect.width:
                        rect.x1 = page_rect.width
                    if rect.y1 > page_rect.height:
                        rect.y1 = page_rect.height
                    
                    print(f"    Final coordinates: {rect}")
                    
                    # Try to add highlight with corrected coordinates
                    try:
                        # Method 1: Use add_highlight_annot with rect
                        highlight = page.add_highlight_annot(rect)
                        print(f"    ✅ Highlight annotation created successfully")
                        
                        # Method 2: Also draw a rectangle for visual confirmation
                        page.draw_rect(rect, color=(1, 0, 0), width=2)  # Red rectangle
                        print(f"    ✅ Rectangle drawn for visual confirmation")
                        
                    except Exception as e:
                        print(f"    ❌ Error creating highlight: {e}")
                        
                        # Try alternative method using quads
                        try:
                            # Convert rect to quad format
                            quad = fitz.Quad(rect)
                            highlight = page.add_highlight_annot(quads=[quad])
                            print(f"    ✅ Highlight created using quad format")
                        except Exception as e2:
                            print(f"    ❌ Quad method also failed: {e2}")
            
            # Also check for any text extraction issues
            try:
                page_text = page.get_text()
                if search_text in page_text:
                    print(f"    ✅ Text '{search_text}' found in page text")
                else:
                    print(f"    ⚠️  Text '{search_text}' NOT found in page text (might be in image)")
            except Exception as e:
                print(f"    ❌ Error extracting page text: {e}")
        
        # Save the corrected version
        output_path = pdf_path.replace('.pdf', '_fixed_highlights.pdf')
        doc.save(output_path)
        print(f"\nFixed PDF saved to: {output_path}")
        
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
            print("Usage: python fix_highlight_coordinates.py <pdf_file> [search_text]")
            return
        
        pdf_path = str(pdf_files[0])
        search_text = "CCTV"
        print(f"Using PDF: {pdf_path}")
        print(f"Searching for: {search_text}")
    
    fix_highlight_coordinates(pdf_path, search_text)

if __name__ == "__main__":
    main()

