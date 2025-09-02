#!/usr/bin/env python3
"""
Discipline Classification System Demo

This script demonstrates how the discipline classification system works
using foundation elements and index symbol recognition.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the discipline classifier
from src.core.discipline_classifier import DisciplineClassifier, DisciplineClassification

def demo_discipline_classification():
    """Demonstrate the discipline classification system."""
    print("🚀 Discipline Classification System Demo")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize the classifier
    print("🔧 Initializing Discipline Classifier...")
    classifier = DisciplineClassifier()
    print("✅ Discipline Classifier initialized successfully")
    
    # Show system capabilities
    print("\n📊 System Capabilities:")
    stats = classifier.get_classification_statistics()
    print(f"  • Disciplines Supported: {len(stats['disciplines_supported'])}")
    print(f"  • Total Index Symbols: {stats['total_index_symbols']}")
    print(f"  • Symbol Recognizer Loaded: {stats['symbol_recognizer_loaded']}")
    print(f"  • Foundation Elements Available: {stats['foundation_elements_available']}")
    
    # Show discipline definitions
    print("\n🏗️ Supported Disciplines:")
    for discipline, definition in classifier.discipline_definitions.items():
        print(f"  • {discipline.title()}: {len(definition['index_symbols'])} index symbols")
        print(f"    - Sub-disciplines: {', '.join(definition['sub_disciplines'])}")
        print(f"    - Drawing types: {', '.join(definition['drawing_types'])}")
    
    # Demonstrate classification workflow
    print("\n🔄 Classification Workflow:")
    print("  1. Foundation Elements Analysis")
    print("     • North Arrow Detection")
    print("     • Scale Detection")
    print("     • Legend Extraction")
    print("     • Notes Extraction")
    print("     • Coordinate System Analysis")
    
    print("  2. Index Symbol Recognition")
    print("     • Symbol Detection using ML")
    print("     • Text-based Symbol Extraction")
    print("     • Symbol Variation Matching")
    
    print("  3. Multi-Stage Classification")
    print("     • Primary Discipline Classification")
    print("     • Sub-Discipline Classification")
    print("     • Drawing Type Classification")
    
    print("  4. Confidence and Evidence")
    print("     • Confidence Calculation")
    print("     • Supporting Evidence Compilation")
    print("     • Foundation Score Integration")
    
    # Show example index symbols
    print("\n🔍 Example Index Symbols by Discipline:")
    for discipline, definition in classifier.discipline_definitions.items():
        print(f"\n  📋 {discipline.upper()}:")
        for symbol_type, variations in definition['index_symbols'].items():
            print(f"    • {symbol_type}: {', '.join(variations[:3])}")
    
    # Demonstrate classification logic
    print("\n🧠 Classification Logic:")
    print("  • Index symbols are the primary driver for discipline classification")
    print("  • Foundation elements provide context and validation")
    print("  • Multi-stage classification ensures granular understanding")
    print("  • Confidence scoring validates classification quality")
    
    # Show output format
    print("\n📄 Output Format:")
    example_output = {
        "drawing_path": "example_electrical_plan.pdf",
        "classification": {
            "primary_discipline": "electrical",
            "sub_discipline": "power",
            "drawing_type": "power_plan",
            "confidence": 0.85,
            "supporting_evidence": [
                "Index symbol 'conduit' (0.92)",
                "Index symbol 'junction_box' (0.88)",
                "Legend evidence for electrical: 0.75",
                "Notes evidence for electrical: 0.65"
            ],
            "index_symbols": ["conduit", "junction_box", "transformer"],
            "foundation_score": 78.5,
            "classification_method": "foundation_elements_with_index_symbols",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    print(json.dumps(example_output, indent=2))
    
    # Show integration benefits
    print("\n🔗 Integration Benefits:")
    print("  • Uses your existing symbol recognition system")
    print("  • Integrates with all foundation elements")
    print("  • Leverages your existing line matcher")
    print("  • Maintains compatibility with existing workflows")
    print("  • Provides structured JSON output for integration")
    
    # Show next steps
    print("\n🚀 Next Steps:")
    print("  1. Test with real engineering drawings")
    print("  2. Train symbol recognition models for improved accuracy")
    print("  3. Validate classification accuracy on your data")
    print("  4. Integrate with existing workflows")
    print("  5. Begin Phase 2.2: Existing vs. Proposed Detection")
    
    print("\n🎉 Demo Complete!")
    print("The discipline classification system is ready for use with your engineering drawings.")


def demo_batch_classification():
    """Demonstrate batch classification capabilities."""
    print("\n🔄 Batch Classification Demo")
    print("=" * 40)
    
    # Look for sample data
    sample_dirs = ["yolo_processed_data_local", "real_world_data", "training_data"]
    
    for sample_dir in sample_dirs:
        if Path(sample_dir).exists():
            print(f"📁 Found sample data directory: {sample_dir}")
            
            # Look for PDF files
            pdf_files = list(Path(sample_dir).glob("*.pdf"))
            if pdf_files:
                print(f"  • Found {len(pdf_files)} PDF files")
                print(f"  • Ready for batch classification")
                print(f"  • Command: python src/core/discipline_classifier.py {sample_dir}/ --output batch_results.json")
                break
    
    print("\n📋 Batch Classification Features:")
    print("  • Process multiple drawings simultaneously")
    print("  • Generate discipline distribution statistics")
    print("  • Export results to JSON format")
    print("  • Error handling for failed classifications")
    print("  • Progress tracking and logging")


if __name__ == "__main__":
    demo_discipline_classification()
    demo_batch_classification()
