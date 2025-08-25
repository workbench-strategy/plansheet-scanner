#!/usr/bin/env python
"""
Cable Entity Matcher - Main Runner

This script provides a convenient entry point to run the cable entity matching pipeline.
It searches for cable names in PDF documents and highlights them with unique colors.

Features:
- Automatic detection of "new" cable references
- Station callout association in the legend
- Reduced border width for cleaner highlighting
- Semi-transparent fill for improved readability
- Comprehensive report generation

Usage:
    python run_cable_matcher.py --csv <csv_file> --pdf <pdf_file> [options]
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import the pipeline module
from src.core.cable_entity_pipeline import run_pipeline

def main():
    """Main entry point for the cable entity matcher"""
    parser = argparse.ArgumentParser(
        description="Cable Entity Matcher - Find and highlight cable references in engineering documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--csv", 
        required=True,
        help="Path to CSV file with cable information"
    )
    
    parser.add_argument(
        "--pdf", 
        required=True,
        help="Path to PDF document to analyze"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="./output",
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--border-width", 
        type=float, 
        default=1.0,
        help="Width of highlight borders"
    )
    
    parser.add_argument(
        "--fill-opacity", 
        type=float, 
        default=0.15,
        help="Opacity of highlight fills (0-1)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable detailed logging"
    )
    
    parser.add_argument(
        "--one-line-mode",
        action="store_true",
        help="Enable one-line diagram mode (optimized for one-line diagrams without north arrow)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("cable_matcher.log"),
            logging.StreamHandler()
        ]
    )
    
    # Print banner
    print("=" * 80)
    print("CABLE ENTITY MATCHER".center(80))
    print("=" * 80)
    print(f"CSV: {args.csv}")
    print(f"PDF: {args.pdf}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Highlight Style: Border width={args.border_width}, Fill opacity={args.fill_opacity}")
    print("-" * 80)
    
    try:
        # Ensure CSV and PDF files exist
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"CSV file not found: {args.csv}")
        
        if not os.path.exists(args.pdf):
            raise FileNotFoundError(f"PDF file not found: {args.pdf}")
            
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        print("Processing... (this may take a few minutes for large documents)")
        
        # Run the pipeline
        results = run_pipeline(
            csv_path=args.csv,
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            border_width=args.border_width,
            fill_opacity=args.fill_opacity,
            one_line_mode=args.one_line_mode
        )
        
        # Print success message
        print("\n✅ Cable entity matching completed successfully!")
        print("-" * 80)
        print(f"Highlighted PDF: {results.get('output_pdf')}")
        print(f"Report: {results.get('output_report')}")
        
        # Print summary statistics
        summary = results.get('summary', {})
        if summary:
            print(f"\nSummary:")
            print(f"- Cable types found: {summary.get('entities_found', 0)}/{summary.get('total_entities', 0)}")
            print(f"- Total matches: {summary.get('total_matches', 0)}")
            
            metrics = summary.get("metrics", {})
            if metrics:
                print(f"\nMatch Types:")
                print(f"- Exact matches: {metrics.get('exact_matches', 0)}")
                print(f"- Variation matches: {metrics.get('variation_matches', 0)}")
                print(f"- New cable references: {metrics.get('new_cable_matches', 0)}")
                print(f"- Station callouts identified: {metrics.get('station_callouts', 0)}")
                print(f"- Duplicate matches prevented: {metrics.get('duplicate_avoidance', 0)}")
        
        print("-" * 80)
        print("Success! Output files include timestamps to prevent overwriting previous results.")
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
