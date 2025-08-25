#!/usr/bin/env python
"""
Cable Entity Matching CLI Interface

This script provides a command-line interface for the cable entity matching pipeline.
It allows users to analyze engineering documents for cable references.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to the path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

from core.cable_entity_pipeline import run_pipeline

def main():
    """Main entry point for the CLI tool"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cable Entity Matching Pipeline")
    
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
        help="Directory to save outputs (default: ./output)"
    )
    
    parser.add_argument(
        "--border-width", 
        type=float, 
        default=1.0,
        help="Width of highlight borders (default: 1.0)"
    )
    
    parser.add_argument(
        "--fill-opacity", 
        type=float, 
        default=0.15,
        help="Opacity of highlight fills, 0-1 (default: 0.15)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"Starting Cable Entity Matching Pipeline")
    print(f"CSV: {args.csv}")
    print(f"PDF: {args.pdf}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Highlight Style: Border width={args.border_width}, Fill opacity={args.fill_opacity}")
    print("\nProcessing...")
    
    try:
        # Run the pipeline
        results = run_pipeline(
            csv_path=args.csv,
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            border_width=args.border_width,
            fill_opacity=args.fill_opacity
        )
        
        # Print summary
        print("\n✅ Entity matching completed successfully:")
        print(f"- Highlighted PDF: {results.get('output_pdf')}")
        print(f"- Report: {results.get('output_report')}")
        
        summary = results.get("summary", {})
        if summary:
            print(f"- Entities found: {summary.get('entities_found', 0)}/{summary.get('total_entities', 0)}")
            print(f"- Total matches: {summary.get('total_matches', 0)}")
            
        metrics = summary.get("metrics", {})
        if metrics:
            print("\nMatch Statistics:")
            print(f"- Exact matches: {metrics.get('exact_matches', 0)}")
            print(f"- Variation matches: {metrics.get('variation_matches', 0)}")
            print(f"- New cable references: {metrics.get('new_cable_matches', 0)}")
            print(f"- Station callouts: {metrics.get('station_callouts', 0)}")
            print(f"- Duplicate matches prevented: {metrics.get('duplicate_avoidance', 0)}")
        
        print("\nOutput files include timestamps to prevent overwriting previous results.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
