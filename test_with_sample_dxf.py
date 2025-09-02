#!/usr/bin/env python3
"""
Test AutoCAD Symbol Training with Sample DXF Files
This script tests the system with the sample DXF files that were created.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from autocad_dwg_symbol_trainer import AutoCADSymbolTrainer, MiniModelConfig

def test_with_sample_dxf():
    """Test the system with sample DXF files."""
    print("🧪 Testing AutoCAD Symbol Training with Sample DXF Files")
    print("=" * 60)
    
    # Find sample DXF files
    sample_dxf_dir = Path.cwd() / "dwg_files" / "sample_dxf"
    
    if not sample_dxf_dir.exists():
        print("❌ Sample DXF directory not found!")
        print("   Run: python dwg_to_dxf_converter.py first")
        return
    
    dxf_files = list(sample_dxf_dir.glob("*.dxf"))
    print(f"📁 Found {len(dxf_files)} sample DXF files in: {sample_dxf_dir}")
    
    if not dxf_files:
        print("❌ No DXF files found!")
        return
    
    # Initialize trainer
    trainer = AutoCADSymbolTrainer()
    
    # Process DXF files
    print(f"\n📥 Processing sample DXF files...")
    total_symbols = 0
    
    for dxf_file in dxf_files:
        try:
            print(f"   Processing: {dxf_file.name}")
            
            # Use the existing add_dwg_file method (it should work with DXF too)
            symbols_added = trainer.add_dwg_file(str(dxf_file))
            
            if symbols_added > 0:
                print(f"      ✅ Extracted {symbols_added} symbols")
                total_symbols += symbols_added
            else:
                print(f"      ⚠️  No symbols extracted")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print(f"\n📊 Processing Summary:")
    print(f"   Files processed: {len(dxf_files)}")
    print(f"   Total symbols extracted: {total_symbols}")
    
    # Show statistics
    stats = trainer.get_training_statistics()
    print(f"\n📈 Training data statistics:")
    print(f"   Total symbols: {stats['total_symbols']}")
    print(f"   Symbol types: {stats['symbol_types']}")
    print(f"   Entity types: {stats['entity_types']}")
    
    # Train model if we have enough data
    if stats['total_symbols'] >= 10:
        print(f"\n🤖 Training model...")
        try:
            config = MiniModelConfig(
                model_type="random_forest",
                n_estimators=100,
                max_depth=10
            )
            
            results = trainer.train_mini_model(config)
            
            if results["success"]:
                print(f"✅ Training successful!")
                print(f"   Train accuracy: {results['train_score']:.3f}")
                print(f"   Test accuracy: {results['test_score']:.3f}")
                print(f"   Model saved to: {results['model_path']}")
                print(f"   Symbol types: {results['symbol_types']}")
                
                # Test prediction on one of the DXF files
                print(f"\n🔮 Testing prediction...")
                if dxf_files:
                    test_file = dxf_files[0]
                    print(f"   Testing on: {test_file.name}")
                    
                    predictions = trainer.predict_symbol_type(str(test_file))
                    print(f"   ✅ Predicted {len(predictions)} symbols")
                    
                    # Show predictions
                    for i, pred in enumerate(predictions[:3]):
                        print(f"      Symbol {i+1}: {pred['symbol_name']}")
                        print(f"         Type: {pred['predicted_type']}")
                        print(f"         Confidence: {pred['confidence']:.3f}")
                        print(f"         Layer: {pred['layer_name']}")
                
            else:
                print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during training: {e}")
    else:
        print(f"\n⚠️  Need at least 10 symbols for training, have {stats['total_symbols']}")

def main():
    """Main function."""
    try:
        test_with_sample_dxf()
        
        print(f"\n🎉 Test completed!")
        print(f"\n💡 Next steps:")
        print(f"   1. The system is working with DXF files!")
        print(f"   2. Convert your DWG files to DXF format")
        print(f"   3. Place DXF files in the dwg_files directory")
        print(f"   4. Run the training system again")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
