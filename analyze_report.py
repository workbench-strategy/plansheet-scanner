#!/usr/bin/env python
"""
Report Analyzer for Cable Entity Pipeline

This script analyzes the JSON report produced by the cable entity pipeline
and displays a summary of the results.
"""

import json
import os
import sys
from pathlib import Path

def analyze_report(report_path):
    """Analyze and display results from a cable matching report"""
    try:
        # Load the report
        with open(report_path, 'r') as f:
            data = json.load(f)
            
        # Extract entity details
        entities = []
        for name, details in data['entity_details'].items():
            count = details['count']
            entities.append((name, count))
            
        # Sort by match count (descending)
        entities.sort(key=lambda x: x[1], reverse=True)
        
        # Display summary
        print("\nCABLE ENTITY MATCHING RESULTS")
        print("=" * 60)
        print(f"Total matches: {data['summary']['total_matches']}")
        print(f"Cables found: {data['summary']['entities_found']}/{data['summary']['total_entities']}")
        print(f"Exact matches: {data['summary']['metrics']['exact_matches']}")
        print(f"Variation matches: {data['summary']['metrics']['variation_matches']}")
        print(f"New cable references: {data['summary']['metrics']['new_cable_matches']}")
        print(f"Duplicate avoidance: {data['summary']['metrics']['duplicate_avoidance']}")
        
        # Display station range information if available
        if 'station_ranges' in data:
            print("\nSTATION RANGES")
            print("-" * 60)
            print(f"{'CABLE TYPE':<25} {'STATION RANGE':<25} {'PAGES':<20}")
            print("-" * 60)
            
            for entity, ranges in data['station_ranges'].items():
                for range_info in ranges:
                    start = range_info['start_station']
                    end = range_info['end_station']
                    pages = range_info['pages']
                    
                    # Format the page list (show up to 5)
                    if len(pages) <= 5:
                        page_str = ", ".join(str(p) for p in pages)
                    else:
                        page_str = ", ".join(str(p) for p in pages[:5])
                        page_str += f", ... ({len(pages) - 5} more)"
                    
                    print(f"{entity:<25} {start} to {end:<15} {page_str:<20}")
                
        print("=" * 60)
        
        # Display cables by match count
        print("\nCABLES FOUND (ordered by match count):")
        print("-" * 60)
        print(f"{'CABLE TYPE':<30} {'COUNT':<10} {'PAGES':<20}")
        print("-" * 60)
        
        for name, count in entities:
            if count > 0:
                details = data['entity_details'][name]
                page_list = details['pages']
                
                # Format the page list (show up to 5)
                if len(page_list) <= 5:
                    pages = ", ".join(str(p) for p in page_list)
                else:
                    pages = ", ".join(str(p) for p in page_list[:5])
                    pages += f", ... ({len(page_list) - 5} more)"
                    
                print(f"{name:<30} {count:<10} {pages:<20}")
        
        # Show cables not found
        not_found = [name for name, count in entities if count == 0]
        if not_found:
            print("\nCABLES NOT FOUND:")
            for name in not_found:
                print(f"- {name}")
        
        # Show report path
        print("\nThe complete report can be found at:")
        print(f"- {os.path.abspath(report_path)}")
        
    except Exception as e:
        print(f"Error analyzing report: {str(e)}")
        return 1
        
    return 0

def main():
    """Main function to process the report"""
    # Use the most recent report if no argument provided
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    else:
        # Find the most recent report in the output directory
        output_dir = Path("./output")
        if not output_dir.exists():
            print(f"Error: Output directory {output_dir} not found")
            return 1
            
        reports = list(output_dir.glob("*_report.json"))
        if not reports:
            print(f"Error: No reports found in {output_dir}")
            return 1
            
        # Sort by modification time (newest first)
        reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        report_path = reports[0]
        print(f"Using most recent report: {report_path}")
    
    return analyze_report(report_path)

if __name__ == "__main__":
    sys.exit(main())
