#!/usr/bin/env python
"""
Quick Test for Cable Entity Pipeline

Tests the pipeline with a smaller portion of the PDF to run faster.
"""

import sys
import os
from pathlib import Path
import time
import fitz  # PyMuPDF

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.cable_entity_pipeline import run_pipeline

def create_smaller_pdf(input_pdf, output_pdf, max_pages=2):
    """Create a smaller version of the PDF with fewer pages for faster testing"""
    doc = fitz.open(input_pdf)
    
    # Limit pages
    page_count = min(len(doc), max_pages)
    
    # Create new document with just those pages
    new_doc = fitz.open()
    for i in range(page_count):
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
    
    # Save the smaller PDF
    new_doc.save(output_pdf)
    new_doc.close()
    doc.close()
    
    print(f"Created smaller PDF with {page_count} pages at {output_pdf}")
    return output_pdf

def main():
    """Run a quick test of the cable entity matching pipeline"""
    print("Running Quick Cable Entity Pipeline Test")
    start_time = time.time()
    
    # Create test directory
    test_dir = Path("./test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Define test files
    test_csv = Path("./Tables/WIM_Equipment.csv")
    original_pdf = Path("./Tables/M01-17-ITS-Tolling-2b.pdf")
    smaller_pdf = test_dir / "test_smaller.pdf"
    
    # Create a smaller version of the PDF with fewer pages
    create_smaller_pdf(original_pdf, smaller_pdf, max_pages=2)
    
    # Check if test files exist
    if not test_csv.exists():
        print(f"\n❌ CSV file not found: {test_csv}")
        return 1
        
    print("\nTest Files:")
    print(f"- CSV: {test_csv}")
    print(f"- PDF: {smaller_pdf} (smaller test version)")
    print(f"- Output Directory: {test_dir}")
    
    try:
        # Run the pipeline with test parameters and show more logging
        results = run_pipeline(
            csv_path=str(test_csv),
            pdf_path=str(smaller_pdf),
            output_dir=str(test_dir),
            border_width=0.8,
            fill_opacity=0.2
        )
        
        # Print results
        elapsed_time = time.time() - start_time
        print(f"\n✅ Test completed successfully in {elapsed_time:.2f} seconds!")
        print(f"- Output PDF: {results.get('output_pdf')}")
        print(f"- Report: {results.get('output_report')}")
        
        # Print summary statistics if available
        summary = results.get('summary', {})
        if summary:
            print(f"\nSummary:")
            print(f"- Entities found: {summary.get('entities_found', 0)}/{summary.get('total_entities', 0)}")
            print(f"- Total matches: {summary.get('total_matches', 0)}")
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
