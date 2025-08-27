#!/usr/bin/env python3
"""
Clarify Processing Status
Analyze what was actually processed and scanner efficiency.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import time
from improved_ai_trainer import ImprovedAIEngineerTrainer

def analyze_what_was_processed():
    """Analyze what was actually processed vs what's available."""
    print("🔍 CLARIFYING PROCESSING STATUS")
    print("=" * 60)
    
    # Check available PDF files
    as_built_dir = Path("as_built_drawings")
    pdf_files = list(as_built_dir.rglob("*.pdf"))
    
    print(f"📄 AVAILABLE PDF FILES: {len(pdf_files)}")
    print("-" * 40)
    
    # Group by directory
    by_directory = {}
    for pdf_path in pdf_files:
        parent_dir = pdf_path.parent.name
        if parent_dir not in by_directory:
            by_directory[parent_dir] = []
        by_directory[parent_dir].append(pdf_path.name)
    
    for directory, files in by_directory.items():
        print(f"📁 {directory}: {len(files)} files")
        for filename in files[:3]:  # Show first 3 files
            print(f"   📄 {filename}")
        if len(files) > 3:
            print(f"   ... and {len(files) - 3} more")
        print()
    
    # Check what was actually processed
    print(f"🤖 PROCESSED DATA ANALYSIS")
    print("-" * 40)
    
    trainer = ImprovedAIEngineerTrainer()
    stats = trainer.get_training_statistics()
    
    print(f"📊 Total training examples: {stats['total_drawings']}")
    print(f"🎯 Models trained: {stats['models_trained']}")
    
    # Count real vs synthetic data
    real_data_count = 0
    synthetic_data_count = 0
    
    for drawing in trainer.as_built_drawings:
        if drawing.drawing_id.startswith("real_"):
            real_data_count += 1
        else:
            synthetic_data_count += 1
    
    print(f"📄 Real PDFs processed: {real_data_count}")
    print(f"🤖 Synthetic examples: {synthetic_data_count}")
    
    # Show real processed files
    if real_data_count > 0:
        print(f"\n🔍 REAL PDFS PROCESSED:")
        print("-" * 40)
        for drawing in trainer.as_built_drawings:
            if drawing.drawing_id.startswith("real_"):
                print(f"   📄 {drawing.drawing_id.replace('real_', '')}")
                print(f"      🏗️  Discipline: {drawing.discipline}")
                print(f"      📁 File: {drawing.file_path}")
                print()

def analyze_processing_efficiency():
    """Analyze how efficient the PDF processing is."""
    print(f"⚡ PROCESSING EFFICIENCY ANALYSIS")
    print("=" * 60)
    
    # Test processing time for a few files
    as_built_dir = Path("as_built_drawings")
    pdf_files = list(as_built_dir.rglob("*.pdf"))
    
    if len(pdf_files) == 0:
        print("❌ No PDF files found to test!")
        return
    
    print(f"🧪 Testing processing speed on sample files...")
    print()
    
    # Test first 3 files
    test_files = pdf_files[:3]
    total_time = 0
    
    for i, pdf_path in enumerate(test_files, 1):
        print(f"🔄 Test {i}: {pdf_path.name}")
        
        start_time = time.time()
        
        try:
            # Simulate the processing steps
            import fitz
            
            # Step 1: Open PDF
            doc = fitz.open(pdf_path)
            open_time = time.time() - start_time
            
            # Step 2: Extract text from first page
            page = doc.load_page(0)
            text = page.get_text()
            extract_time = time.time() - start_time - open_time
            
            # Step 3: Close document
            doc.close()
            total_file_time = time.time() - start_time
            
            total_time += total_file_time
            
            print(f"   ⏱️  Open time: {open_time:.3f}s")
            print(f"   ⏱️  Text extraction: {extract_time:.3f}s")
            print(f"   ⏱️  Total time: {total_file_time:.3f}s")
            print(f"   📊 Text length: {len(text)} chars")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
    
    avg_time = total_time / len(test_files)
    print(f"📈 EFFICIENCY SUMMARY:")
    print(f"   ⏱️  Average processing time: {avg_time:.3f}s per file")
    print(f"   📄 Estimated time for all {len(pdf_files)} files: {avg_time * len(pdf_files):.1f}s")
    print(f"   📄 Estimated time for all {len(pdf_files)} files: {avg_time * len(pdf_files) / 60:.1f} minutes")
    
    # Check file sizes
    total_size = sum(pdf_path.stat().st_size for pdf_path in pdf_files)
    total_size_mb = total_size / (1024 * 1024)
    print(f"   📊 Total PDF size: {total_size_mb:.1f} MB")
    print(f"   📊 Average file size: {total_size_mb / len(pdf_files):.1f} MB")

def show_processing_recommendations():
    """Show recommendations for efficient processing."""
    print(f"\n💡 PROCESSING RECOMMENDATIONS")
    print("=" * 60)
    
    as_built_dir = Path("as_built_drawings")
    pdf_files = list(as_built_dir.rglob("*.pdf"))
    
    print(f"📄 Current situation:")
    print(f"   - {len(pdf_files)} PDF files available")
    print(f"   - Only 6 files processed so far")
    print(f"   - Processing is relatively fast (~0.1s per file)")
    print()
    
    print(f"🚀 Recommended approach:")
    print(f"   1. Process all files in batches of 50-100")
    print(f"   2. Use parallel processing for speed")
    print(f"   3. Focus on high-priority projects first")
    print(f"   4. Set up background processing")
    print()
    
    print(f"⚡ Efficiency improvements:")
    print(f"   - Process only first 3 pages of each PDF")
    print(f"   - Skip files larger than 100MB")
    print(f"   - Use caching for repeated processing")
    print(f"   - Process during off-hours")

def main():
    """Main analysis function."""
    print("🔍 AS-BUILT PROCESSING STATUS CLARIFICATION")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    analyze_what_was_processed()
    analyze_processing_efficiency()
    show_processing_recommendations()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
