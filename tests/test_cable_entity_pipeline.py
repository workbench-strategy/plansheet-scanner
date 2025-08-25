#!/usr/bin/env python
"""
Test script for Cable Entity Matching Pipeline

This script runs a simple test of the cable entity matching pipeline using sample data.
It demonstrates how to use the pipeline and verifies that it works correctly.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.cable_entity_pipeline import run_pipeline

def main():
    """Run a test of the cable entity matching pipeline"""
    print("Running Cable Entity Pipeline Test")
    
    # Create test directory
    test_dir = Path("./test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Define test files
    test_csv = Path("./Tables/WIM_Equipment.csv")
    test_pdf = Path("./Tables/M01-17-ITS-Tolling-2b.pdf")
    
    # Check if test files exist
    if not test_csv.exists() or not test_pdf.exists():
        print(f"\n❌ Test files not found. Please make sure these files exist:")
        print(f"  - CSV: {test_csv}")
        print(f"  - PDF: {test_pdf}")
        print("\nYou can copy your own files to these paths or modify this script.")
        return 1
        
    print("\nTest Files:")
    print(f"- CSV: {test_csv}")
    print(f"- PDF: {test_pdf}")
    print(f"- Output Directory: {test_dir}")
    
    try:
        # Run the pipeline with test parameters
        results = run_pipeline(
            csv_path=str(test_csv),
            pdf_path=str(test_pdf),
            output_dir=str(test_dir),
            border_width=0.8,  # Slightly thinner borders
            fill_opacity=0.2   # Slightly more opaque fills
        )
        
        # Print results
        print("\n✅ Test completed successfully!")
        print(f"- Output PDF: {results.get('output_pdf')}")
        print(f"- Report: {results.get('output_report')}")
        
        # Print summary statistics if available
        summary = results.get('summary', {})
        if summary:
            print(f"\nSummary:")
            print(f"- Entities found: {summary.get('entities_found', 0)}/{summary.get('total_entities', 0)}")
            print(f"- Total matches: {summary.get('total_matches', 0)}")
            
            metrics = summary.get('metrics', {})
            if metrics:
                print(f"\nMatch Statistics:")
                print(f"- Exact matches: {metrics.get('exact_matches', 0)}")
                print(f"- Variation matches: {metrics.get('variation_matches', 0)}")
                print(f"- New cable references: {metrics.get('new_cable_matches', 0)}")
                print(f"- Station callouts: {metrics.get('station_callouts', 0)}")
                
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
