#!/usr/bin/env python3
"""
Run Real Plan Highlighter
Processes actual PDF files from the as_built_drawings directory.
"""

import os
import sys
from pathlib import Path
from enhanced_plan_highlighter import EnhancedPlanHighlighter

def run_on_real_plans():
    """Run the plan highlighter on real PDF files."""
    print("🎨 Running Real Plan Highlighter")
    print("=" * 50)
    
    # Initialize the highlighter
    highlighter = EnhancedPlanHighlighter()
    
    # Find real PDF files
    as_built_dir = Path("as_built_drawings")
    pdf_files = list(as_built_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in as_built_drawings directory")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files")
    
    # Select a few files to process (to avoid overwhelming output)
    sample_files = pdf_files[:3]  # Process first 3 files
    
    print(f"\n🔍 Processing {len(sample_files)} sample files:")
    for i, pdf_file in enumerate(sample_files, 1):
        print(f"   {i}. {pdf_file.name}")
    
    # Create plan data for each file
    plan_data_list = []
    
    for pdf_file in sample_files:
        # Extract basic info from filename
        filename = pdf_file.stem
        discipline = "unknown"
        
        # Try to determine discipline from filename
        if "ITS" in filename.upper():
            discipline = "its"
        elif "ELEC" in filename.upper() or "ELECTRICAL" in filename.upper():
            discipline = "electrical"
        elif "SIGNAL" in filename.upper():
            discipline = "traffic"
        elif "STRUCTURAL" in filename.upper() or "BRIDGE" in filename.upper():
            discipline = "structural"
        
        plan_data = {
            "drawing_id": filename,
            "sheet_title": filename,
            "discipline": discipline,
            "project_name": "Real As-Built Project",
            "construction_notes": f"Real as-built drawing: {filename}",
            "as_built_changes": [
                {"description": "As-built drawing from construction", "severity": "minor"}
            ],
            "file_path": str(pdf_file)
        }
        
        plan_data_list.append(plan_data)
    
    print(f"\n🚀 Processing real plans...")
    
    # Process the plans
    results = highlighter.highlight_multiple_plans(plan_data_list, "real_highlighted_plans")
    
    print(f"\n✅ Real plan processing complete!")
    print(f"   📁 Output directory: real_highlighted_plans")
    print(f"   📊 Plans processed: {results['total_plans']}")
    print(f"   🔍 Elements highlighted: {results['summary']['total_elements']}")
    print(f"   ⚠️  Violations found: {results['summary']['total_violations']}")
    print(f"   🔄 Changes detected: {results['summary']['total_changes']}")
    
    return results

def run_conduit_detector_on_real_plans():
    """Run the conduit detector on real PDF files."""
    print("\n🔍 Running Conduit Detector on Real Plans")
    print("=" * 50)
    
    from enhanced_conduit_detector import EnhancedConduitDetector
    
    # Initialize the detector
    detector = EnhancedConduitDetector()
    
    # Find real PDF files
    as_built_dir = Path("as_built_drawings")
    pdf_files = list(as_built_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in as_built_drawings directory")
        return
    
    # Select a sample file
    sample_file = pdf_files[0]  # Use first file
    
    print(f"🔍 Processing: {sample_file.name}")
    
    # Create plan data
    plan_data = {
        "drawing_id": sample_file.stem,
        "project_name": "Real As-Built Project",
        "sheet_number": "001",
        "sheet_title": sample_file.stem,
        "discipline": "electrical",
        "construction_notes": f"Real as-built drawing: {sample_file.name}",
        "as_built_changes": [
            {"description": "As-built drawing from construction", "severity": "minor"}
        ],
        "file_path": str(sample_file)
    }
    
    # Detect conduit in the real plan
    detection_results = detector.detect_conduit_in_plan(str(sample_file), plan_data)
    
    # Generate report
    report_path = detector.generate_conduit_report(detection_results, "real_conduit_detection_report.json")
    
    print(f"\n✅ Real conduit detection complete!")
    print(f"   📋 Report saved: {report_path}")
    print(f"   🔍 Elements detected: {len(detection_results['conduit_elements'])}")
    print(f"   🎯 Confidence score: {detection_results['confidence_score']:.1%}")
    
    return detection_results

if __name__ == "__main__":
    print("🎨 REAL PLAN HIGHLIGHTER DEMO")
    print("=" * 60)
    
    # Run plan highlighter on real files
    plan_results = run_on_real_plans()
    
    # Run conduit detector on real files
    conduit_results = run_conduit_detector_on_real_plans()
    
    print(f"\n🎉 REAL PLAN PROCESSING COMPLETE!")
    print("=" * 40)
    print("✅ Now processing actual PDF files from as_built_drawings")
    print("✅ Generating real highlighting reports")
    print("✅ Creating actual conduit detection analysis")
    print("✅ Check the 'real_highlighted_plans' directory for results")

