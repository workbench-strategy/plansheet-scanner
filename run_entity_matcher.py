#!/usr/bin/env python
"""
Test the Engineering Document Entity Matcher with the sample data.
This is a convenience script to run the entity matcher with the test data.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Main function to run the test."""
    script_dir = Path(__file__).parent
    cli_script = script_dir / "src" / "cli" / "entity_matcher_cli.py"
    
    if not cli_script.exists():
        print(f"Error: CLI script not found at {cli_script}")
        return 1
    
    # Define paths for sample data
    csv_path = script_dir / "Tables" / "WIM_Equipment.csv"
    pdf_path = script_dir / "Tables" / "M01-17-ITS-Tolling-2b.pdf"
    output_dir = script_dir / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if files exist
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return 1
        
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return 1
    
    # Build the command to run
    cmd = [
        sys.executable,
        str(cli_script),
        "--csv", str(csv_path),
        "--pdf", str(pdf_path),
        "--output-dir", str(output_dir),
        "--verbose"
    ]
    
    # Add a note about timestamps
    print("\nNOTE: Output files will include timestamps in their filenames")
    print("This ensures each run produces unique files without overwriting previous results")
    
    # Print the command
    print("Running command:")
    print(" ".join(cmd))
    print("\n" + "="*50 + "\n")
    
    # Execute the command
    os.execv(sys.executable, cmd)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
