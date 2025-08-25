#!/usr/bin/env python3
"""
Smart Plan Seamer CLI - Main Entry Point
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.smart_plan_seamer import SmartPlanSeamer

def setup_logging(verbose: bool = False, quiet: bool = False):
    """Set up logging with appropriate level."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_inputs(pdf_path: str, north_template: Optional[str] = None) -> bool:
    """Validate input files exist and are readable."""
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found: {pdf_path}", file=sys.stderr)
        return False
    
    if not os.access(pdf_path, os.R_OK):
        print(f"ERROR: Cannot read PDF file: {pdf_path}", file=sys.stderr)
        return False
    
    if north_template and not os.path.exists(north_template):
        print(f"ERROR: North template not found: {north_template}", file=sys.stderr)
        return False
    
    return True

def scan_command(args):
    """Execute the scan command (default behavior)."""
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not validate_inputs(args.pdf, args.north_template):
        sys.exit(1)
    
    try:
        # Initialize seamer
        seamer = SmartPlanSeamer(
            north_template_path=args.north_template,
            assume_north_up=args.assume_north_up,
            dpi_correlation=args.dpi,
            crop_content=not args.no_crop,
            max_canvas_inches=(args.max_width, args.max_height),
            correlation_threshold=args.threshold,
            stripe_width_inches=args.stripe_width,
            text_hints=not args.no_text_hints
        )
        
        # Process sheets
        result = seamer.process_sheets(
            pdf_path=args.pdf,
            output_dir=args.output,
            dry_run=False
        )
        
        # Print summary
        print("\n" + "="*60)
        print("SMART PLAN SEAMER - SCAN COMPLETE")
        print("="*60)
        print(f"Input PDF: {args.pdf}")
        print(f"Pages processed: {len(result.sheets)}")
        print(f"Matches found: {len(result.matches)}")
        print(f"Canvas size: {result.canvas_limits.scaled_width_inches:.1f}\" x {result.canvas_limits.scaled_height_inches:.1f}\"")
        print(f"Plotted scale: 1\" = {result.canvas_limits.plotted_scale_ft_per_inch:.1f}'")
        print(f"Output PDF: {result.output_pdf_path}")
        print(f"QA Report: {result.qa_report_path}")
        
        # Check for unplaced sheets
        unplaced = [s for s in result.sheets if s.notes == "unplaced_fallback"]
        if unplaced:
            print(f"\nWARNING: {len(unplaced)} sheets could not be placed:")
            for sheet in unplaced:
                print(f"  - {sheet.sheet_id} (page {sheet.page_num})")
        
        print("\nScan completed successfully!")
        
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def dry_run_command(args):
    """Execute the dry-run command (no PDF output)."""
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not validate_inputs(args.pdf, args.north_template):
        sys.exit(1)
    
    try:
        # Initialize seamer
        seamer = SmartPlanSeamer(
            north_template_path=args.north_template,
            assume_north_up=args.assume_north_up,
            dpi_correlation=args.dpi,
            crop_content=not args.no_crop,
            max_canvas_inches=(args.max_width, args.max_height),
            correlation_threshold=args.threshold,
            stripe_width_inches=args.stripe_width,
            text_hints=not args.no_text_hints
        )
        
        # Process sheets (dry run)
        result = seamer.process_sheets(
            pdf_path=args.pdf,
            output_dir=args.output,
            dry_run=True
        )
        
        # Print summary
        print("\n" + "="*60)
        print("SMART PLAN SEAMER - DRY RUN COMPLETE")
        print("="*60)
        print(f"Input PDF: {args.pdf}")
        print(f"Pages processed: {len(result.sheets)}")
        print(f"Matches found: {len(result.matches)}")
        print(f"Canvas size needed: {result.canvas_limits.original_width_inches:.1f}\" x {result.canvas_limits.original_height_inches:.1f}\"")
        print(f"Scale factor: {result.canvas_limits.scale_factor:.3f}")
        print(f"Plotted scale: 1\" = {result.canvas_limits.plotted_scale_ft_per_inch:.1f}'")
        print(f"QA Report: {result.qa_report_path}")
        
        # Check for unplaced sheets
        unplaced = [s for s in result.sheets if s.notes == "unplaced_fallback"]
        if unplaced:
            print(f"\nWARNING: {len(unplaced)} sheets would be unplaced:")
            for sheet in unplaced:
                print(f"  - {sheet.sheet_id} (page {sheet.page_num})")
        
        print("\nDry run completed successfully!")
        print("No output PDF was created.")
        
    except Exception as e:
        logger.error(f"Dry run failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Smart Plan Seamer - Intelligent PDF plan sheet mosaicking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic scan with default settings
  python -m src.cli.smart_seamer scan plans.pdf

  # Dry run to test without creating PDF
  python -m src.cli.smart_seamer dry-run plans.pdf

  # Custom north template and settings
  python -m src.cli.smart_seamer scan plans.pdf --north-template north_arrow.png --dpi 160

  # Assume north is up, no content cropping
  python -m src.cli.smart_seamer scan plans.pdf --assume-north-up --no-crop

  # Custom canvas size and correlation threshold
  python -m src.cli.smart_seamer scan plans.pdf --max-width 80 --max-height 60 --threshold 0.4

  # Verbose output for debugging
  python -m src.cli.smart_seamer scan plans.pdf --verbose

  # Quiet mode (errors only)
  python -m src.cli.smart_seamer scan plans.pdf --quiet
        """
    )
    
    # Global flags
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", 
                       help="Suppress info messages (errors only)")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Scan command
    scan_parser = subparsers.add_parser("scan", 
                                       help="Process plan sheets and create mosaic PDF")
    scan_parser.add_argument("pdf", help="Input PDF file")
    scan_parser.add_argument("--output", default="./output", 
                            help="Output directory (default: ./output)")
    scan_parser.add_argument("--north-template", 
                            help="Path to north arrow template image")
    scan_parser.add_argument("--assume-north-up", action="store_true",
                            help="Skip north detection, assume north is up")
    scan_parser.add_argument("--dpi", type=int, default=120,
                            help="DPI for correlation operations (default: 120)")
    scan_parser.add_argument("--no-crop", action="store_true",
                            help="Disable content cropping")
    scan_parser.add_argument("--max-width", type=float, default=100.0,
                            help="Maximum canvas width in inches (default: 100.0)")
    scan_parser.add_argument("--max-height", type=float, default=100.0,
                            help="Maximum canvas height in inches (default: 100.0)")
    scan_parser.add_argument("--threshold", type=float, default=0.35,
                            help="Minimum correlation score for matches (default: 0.35)")
    scan_parser.add_argument("--stripe-width", type=float, default=0.8,
                            help="Edge stripe width in inches (default: 0.8)")
    scan_parser.add_argument("--no-text-hints", action="store_true",
                            help="Disable text hints for matchline detection")
    scan_parser.set_defaults(func=scan_command)
    
    # Dry-run command
    dry_run_parser = subparsers.add_parser("dry-run",
                                          help="Test processing without creating PDF")
    dry_run_parser.add_argument("pdf", help="Input PDF file")
    dry_run_parser.add_argument("--output", default="./output",
                               help="Output directory for QA report (default: ./output)")
    dry_run_parser.add_argument("--north-template",
                               help="Path to north arrow template image")
    dry_run_parser.add_argument("--assume-north-up", action="store_true",
                               help="Skip north detection, assume north is up")
    dry_run_parser.add_argument("--dpi", type=int, default=120,
                               help="DPI for correlation operations (default: 120)")
    dry_run_parser.add_argument("--no-crop", action="store_true",
                               help="Disable content cropping")
    dry_run_parser.add_argument("--max-width", type=float, default=100.0,
                               help="Maximum canvas width in inches (default: 100.0)")
    dry_run_parser.add_argument("--max-height", type=float, default=100.0,
                               help="Maximum canvas height in inches (default: 100.0)")
    dry_run_parser.add_argument("--threshold", type=float, default=0.35,
                               help="Minimum correlation score for matches (default: 0.35)")
    dry_run_parser.add_argument("--stripe-width", type=float, default=0.8,
                               help="Edge stripe width in inches (default: 0.8)")
    dry_run_parser.add_argument("--no-text-hints", action="store_true",
                               help="Disable text hints for matchline detection")
    dry_run_parser.set_defaults(func=dry_run_command)
    
    args = parser.parse_args()
    
    # Handle no command specified (default to scan)
    if not args.command:
        if len(sys.argv) == 2 and not sys.argv[1].startswith('-'):
            # Single argument provided, treat as PDF for scan command
            args.command = "scan"
            args.pdf = sys.argv[1]
            args.output = "./output"
            args.north_template = None
            args.assume_north_up = False
            args.dpi = 120
            args.no_crop = False
            args.max_width = 100.0
            args.max_height = 100.0
            args.threshold = 0.35
            args.stripe_width = 0.8
            args.no_text_hints = False
            args.func = scan_command
        else:
            parser.print_help()
            sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose, args.quiet)
    
    # Execute command
    args.func(args)

if __name__ == "__main__":
    main()
