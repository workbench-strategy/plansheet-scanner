#!/usr/bin/env python
"""
Verify highlighting in PDF

This script checks if a PDF has highlights by counting the annotations.
"""

import fitz  # PyMuPDF
import sys
import os
from pathlib import Path

def verify_highlights(pdf_path):
    """Count annotations and drawn highlights in a PDF file"""
    print(f"Verifying highlights in: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return 1
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Count annotations and drawn objects
        total_annotations = 0
        page_with_content = 0
        
        # Dictionary to track colors
        colors = {}
        
        # Check for legend page
        has_legend = False
        
        print(f"\nPDF has {len(doc)} pages")
        
        # Check each page
        for page_num, page in enumerate(doc):
            # Look for annotations first
            annotations = page.annots()
            annotations = list(annotations) if annotations else []
            annot_count = len(annotations)
            
            # Look for direct drawing (rectangles, highlights)
            page_dict = page.get_textpage().extractDICT()
            
            # Check for "LEGEND" text which indicates a legend page
            page_text = page.get_text()
            if "LEGEND" in page_text or "Cable References" in page_text:
                has_legend = True
                print(f"  Page {page_num + 1}: Found legend page")
            
            # Count drawn objects by analyzing the page
            drawn_count = 0
            if hasattr(page, 'get_drawings'):
                drawings = page.get_drawings()
                drawn_count = len(drawings)
                
                # Count colors of drawn objects
                for drawing in drawings:
                    if 'rect' in drawing:
                        if drawing.get('color'):
                            color_key = str(drawing['color'])
                            if color_key in colors:
                                colors[color_key] += 1
                            else:
                                colors[color_key] = 1
                            drawn_count += 1
            
            # Add standard annotations
            for annot in annotations:
                stroke_color = annot.colors.get("stroke")
                if stroke_color:
                    color_key = str(stroke_color)
                    if color_key in colors:
                        colors[color_key] += 1
                    else:
                        colors[color_key] = 1
            
            count = annot_count + drawn_count
            if count > 0:
                page_with_content += 1
            
            total_annotations += count
            
            # Print details for first few pages with content
            if count > 0 and page_with_content <= 3:  # Only show first 3 pages with content
                print(f"  Page {page_num + 1}: {annot_count} annotations, {drawn_count} drawings")
                
                # Print examples of annotations on this page (max 2)
                for i, annot in enumerate(annotations[:2]):
                    rect = annot.rect
                    color = annot.colors.get("stroke", "unknown")
                    print(f"    - Annotation {i+1}: Type={annot.type}, Rect={rect}, Color={color}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Total content items: {total_annotations}")
        print(f"- Pages with content: {page_with_content}/{len(doc)}")
        print(f"- Legend page found: {has_legend}")
        print(f"- Unique colors used: {len(colors)}")
        
        # Print top colors
        print("\nTop highlight colors:")
        sorted_colors = sorted([(k, v) for k, v in colors.items()], key=lambda x: x[1], reverse=True)
        for color, count in sorted_colors[:5]:  # Show top 5 colors
            print(f"- Color {color}: {count} annotations")
        
        doc.close()
        return total_annotations
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 0

def main():
    """Main function"""
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Try to find the most recent highlighted PDF
        output_dir = Path("./output")
        if not output_dir.exists():
            print("Error: output directory not found")
            return 1
            
        highlighted_pdfs = list(output_dir.glob("*_highlighted.pdf"))
        if not highlighted_pdfs:
            print("Error: No highlighted PDFs found in output directory")
            return 1
            
        # Sort by modification time (newest first)
        highlighted_pdfs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        pdf_path = highlighted_pdfs[0]
        print(f"Using most recent highlighted PDF: {pdf_path}")
    
    annotation_count = verify_highlights(pdf_path)
    
    if annotation_count > 0:
        print(f"\n✅ PDF contains {annotation_count} highlights/annotations")
        return 0
    else:
        print(f"\n❌ PDF does not appear to contain any highlights")
        return 1

if __name__ == "__main__":
    sys.exit(main())
