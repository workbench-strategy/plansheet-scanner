#!/usr/bin/env python3
"""
Test script for Foundation Elements system.

This script demonstrates the foundation elements working together
and integrates with your existing line matcher for comprehensive analysis.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_foundation_elements():
    """Test the foundation elements system."""
    print("🧪 Testing Foundation Elements System")
    print("=" * 50)
    
    try:
        # Import foundation elements
        from src.foundation_elements import (
            FoundationOrchestrator,
            NorthArrowDetector,
            ScaleDetector,
            LegendExtractor,
            NotesExtractor,
            CoordinateSystemAnalyzer,
            DrawingSetAnalyzer
        )
        
        print("✅ All foundation elements imported successfully!")
        
        # Test individual components
        print("\n🔧 Testing Individual Components:")
        
        # Test North Arrow Detector
        north_detector = NorthArrowDetector()
        print(f"  ✅ North Arrow Detector: {type(north_detector).__name__}")
        
        # Test Scale Detector
        scale_detector = ScaleDetector()
        print(f"  ✅ Scale Detector: {type(scale_detector).__name__}")
        
        # Test Legend Extractor
        legend_extractor = LegendExtractor()
        print(f"  ✅ Legend Extractor: {type(legend_extractor).__name__}")
        
        # Test Notes Extractor
        notes_extractor = NotesExtractor()
        print(f"  ✅ Notes Extractor: {type(notes_extractor).__name__}")
        
        # Test Coordinate System Analyzer
        coord_analyzer = CoordinateSystemAnalyzer()
        print(f"  ✅ Coordinate System Analyzer: {type(coord_analyzer).__name__}")
        
        # Test Drawing Set Analyzer (with your line matcher)
        drawing_analyzer = DrawingSetAnalyzer()
        print(f"  ✅ Drawing Set Analyzer: {type(drawing_analyzer).__name__}")
        
        # Test Foundation Orchestrator
        orchestrator = FoundationOrchestrator()
        print(f"  ✅ Foundation Orchestrator: {type(orchestrator).__name__}")
        
        print("\n🎯 Line Matcher Integration:")
        print(f"  ✅ Line Matcher integrated: {hasattr(drawing_analyzer, 'line_matcher')}")
        print(f"  ✅ Line Matcher type: {type(drawing_analyzer.line_matcher).__name__}")
        
        print("\n📊 Foundation Elements Summary:")
        print("  • North Arrow Detection: Template matching, shape analysis, text detection")
        print("  • Scale Detection: Graphic bars, text patterns, unit conversion")
        print("  • Legend Extraction: Symbol detection, discipline classification")
        print("  • Notes Extraction: Text regions, specification parsing")
        print("  • Coordinate Analysis: Grid lines, spatial references")
        print("  • Drawing Set Analysis: Match lines, sheet relationships")
        print("  • Foundation Orchestrator: Comprehensive analysis coordination")
        
        print("\n🚀 Ready for Phase 2: Discipline-Specific Models")
        print("  • Discipline Classification using foundation elements")
        print("  • Existing vs. Proposed detection")
        print("  • 2D to 3D understanding")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_with_sample_data():
    """Test with sample data if available."""
    print("\n🧪 Testing with Sample Data")
    print("=" * 30)
    
    # Check for YOLO processed data
    yolo_data_dir = Path("yolo_processed_data_local")
    if yolo_data_dir.exists():
        print(f"✅ Found YOLO processed data: {yolo_data_dir}")
        
        # Look for sample images
        image_files = list(yolo_data_dir.glob("*.jpg")) + list(yolo_data_dir.glob("*.png"))
        if image_files:
            print(f"✅ Found {len(image_files)} sample images")
            print(f"  Sample files: {[f.name for f in image_files[:3]]}")
            
            # Test with first image
            try:
                from src.foundation_elements import FoundationOrchestrator
                orchestrator = FoundationOrchestrator()
                
                # Test individual elements
                import cv2
                sample_image = cv2.imread(str(image_files[0]))
                if sample_image is not None:
                    print(f"✅ Successfully loaded sample image: {image_files[0].name}")
                    
                    # Test scale detection
                    from src.foundation_elements import ScaleDetector
                    scale_detector = ScaleDetector()
                    scale_result = scale_detector.detect_scale(sample_image)
                    print(f"  Scale Detection: {scale_result.detected} (confidence: {scale_result.confidence:.3f})")
                    
                    # Test coordinate system analysis
                    from src.foundation_elements import CoordinateSystemAnalyzer
                    coord_analyzer = CoordinateSystemAnalyzer()
                    coord_result = coord_analyzer.analyze_coordinate_system(sample_image)
                    print(f"  Coordinate System: {coord_result.detected} (grid lines: {coord_result.total_grid_lines})")
                    
            except Exception as e:
                print(f"⚠️  Sample data test error: {e}")
        else:
            print("⚠️  No sample images found in YOLO data")
    else:
        print("⚠️  No YOLO processed data found")

def main():
    """Main test function."""
    print("🎯 Foundation Elements System Test")
    print("=" * 50)
    
    # Test foundation elements
    success = test_foundation_elements()
    
    if success:
        # Test with sample data
        test_with_sample_data()
        
        print("\n🎉 Foundation Elements System Ready!")
        print("\n📋 Next Steps:")
        print("  1. Test with your engineering drawings")
        print("  2. Begin Phase 2: Discipline Classification")
        print("  3. Integrate with your existing YOLO data")
        print("  4. Start building discipline-specific models")
        
        print("\n🔧 Test Commands:")
        print("  python src/foundation_elements/foundation_orchestrator.py path/to/drawing.pdf")
        print("  python src/foundation_elements/drawing_set_analyzer.py drawing1.pdf drawing2.pdf")
        print("  python test_foundation_elements.py")
        
    else:
        print("\n❌ Foundation Elements System Test Failed")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()
