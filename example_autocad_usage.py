#!/usr/bin/env python3
"""
Example: How to Use Your .dwg Files with AutoCAD Symbol Training
This script shows you exactly how to train models on your AutoCAD drawings.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from autocad_dwg_symbol_trainer import AutoCADSymbolTrainer, MiniModelConfig

def example_with_dwg_files():
    """Example of how to use .dwg files for training."""
    print("🎯 Example: Training on Your .dwg Files")
    print("=" * 50)
    
    # Initialize the trainer
    trainer = AutoCADSymbolTrainer()
    
    # Get the dwg_files directory
    dwg_files_dir = Path.cwd() / "dwg_files"
    
    print(f"\n📁 Looking for .dwg files in: {dwg_files_dir}")
    
    # Check if dwg_files directory exists and has .dwg files
    if dwg_files_dir.exists():
        dwg_files = list(dwg_files_dir.glob("*.dwg"))
        print(f"   Found {len(dwg_files)} .dwg files")
        
        if dwg_files:
            print(f"   Files found:")
            for dwg_file in dwg_files:
                print(f"      📄 {dwg_file.name}")
            
            # Example 1: Add individual files
            print(f"\n🔧 Example 1: Adding individual .dwg files")
            for dwg_file in dwg_files[:2]:  # Process first 2 files as example
                print(f"   Processing: {dwg_file.name}")
                try:
                    symbols_added = trainer.add_dwg_file(str(dwg_file))
                    print(f"      ✅ Extracted {symbols_added} symbols")
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            
            # Example 2: Add entire directory
            print(f"\n🔧 Example 2: Adding entire directory")
            try:
                total_symbols = trainer.add_dwg_directory(str(dwg_files_dir))
                print(f"   ✅ Total symbols from directory: {total_symbols}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Show current statistics
            stats = trainer.get_training_statistics()
            print(f"\n📊 Current training data:")
            print(f"   Total symbols: {stats['total_symbols']}")
            print(f"   Symbol types: {stats['symbol_types']}")
            print(f"   Entity types: {stats['entity_types']}")
            
            # Example 3: Train a model
            if stats['total_symbols'] >= 10:
                print(f"\n🤖 Example 3: Training a model")
                try:
                    config = MiniModelConfig(
                        model_type="random_forest",
                        n_estimators=50,
                        max_depth=8
                    )
                    
                    results = trainer.train_mini_model(config)
                    
                    if results["success"]:
                        print(f"   ✅ Training successful!")
                        print(f"   Train accuracy: {results['train_score']:.3f}")
                        print(f"   Test accuracy: {results['test_score']:.3f}")
                        print(f"   Model saved to: {results['model_path']}")
                        
                        # Example 4: Predict on a new file
                        print(f"\n🔮 Example 4: Predicting on a new .dwg file")
                        if dwg_files:
                            test_file = dwg_files[0]  # Use first file as test
                            print(f"   Testing on: {test_file.name}")
                            
                            predictions = trainer.predict_symbol_type(str(test_file))
                            print(f"   ✅ Predicted {len(predictions)} symbols")
                            
                            # Show first few predictions
                            for i, pred in enumerate(predictions[:3]):
                                print(f"      Symbol {i+1}: {pred['symbol_name']}")
                                print(f"         Type: {pred['predicted_type']}")
                                print(f"         Confidence: {pred['confidence']:.3f}")
                                print(f"         Layer: {pred['layer_name']}")
                    else:
                        print(f"   ❌ Training failed: {results.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"   ❌ Error during training: {e}")
            else:
                print(f"\n⚠️  Need at least 10 symbols for training, have {stats['total_symbols']}")
                print(f"   Add more .dwg files or use synthetic data for testing")
        
        else:
            print(f"   ⚠️  No .dwg files found in {dwg_files_dir}")
            print(f"   Please copy your .dwg files to this directory")
    
    else:
        print(f"   ⚠️  Directory {dwg_files_dir} does not exist")
        print(f"   Please run setup_autocad_training.py first")

def example_with_synthetic_data():
    """Example using synthetic data for testing."""
    print(f"\n🧪 Example: Testing with Synthetic Data")
    print("=" * 40)
    
    # Initialize trainer
    trainer = AutoCADSymbolTrainer()
    
    # Create synthetic symbols
    from autocad_dwg_symbol_trainer import AutoCADSymbol
    import numpy as np
    
    print(f"   Creating synthetic traffic symbols...")
    for i in range(10):
        symbol = AutoCADSymbol(
            symbol_id=f"synth_signal_{i}",
            symbol_name="traffic_signal",
            symbol_type="traffic",
            entity_type="AcDbCircle",
            geometry={
                "center": [100 + i * 15, 200 + i * 8],
                "radius": 1.5 + (i % 3) * 0.5,
                "area": np.pi * (1.5 + (i % 3) * 0.5) ** 2
            },
            layer_name="TRAFFIC_SIGNALS",
            color="RED",
            linetype="CONTINUOUS",
            lineweight=0.5,
            file_path="synthetic_traffic.dwg",
            bounding_box=(100 + i * 15 - 2, 200 + i * 8 - 2, 
                         100 + i * 15 + 2, 200 + i * 8 + 2)
        )
        trainer.autocad_symbols.append(symbol)
    
    print(f"   Creating synthetic electrical symbols...")
    for i in range(8):
        symbol = AutoCADSymbol(
            symbol_id=f"synth_conduit_{i}",
            symbol_name="electrical_conduit",
            symbol_type="electrical",
            entity_type="AcDbLine",
            geometry={
                "start_point": [300 + i * 20, 100 + i * 10],
                "end_point": [350 + i * 20, 100 + i * 10],
                "length": 50.0
            },
            layer_name="ELECTRICAL_CONDUIT",
            color="BLUE",
            linetype="CONTINUOUS",
            lineweight=0.4,
            file_path="synthetic_electrical.dwg"
        )
        trainer.autocad_symbols.append(symbol)
    
    # Show statistics
    stats = trainer.get_training_statistics()
    print(f"   📊 Synthetic data created:")
    print(f"      Total symbols: {stats['total_symbols']}")
    print(f"      Symbol types: {stats['symbol_types']}")
    
    # Train model
    print(f"   🤖 Training model on synthetic data...")
    config = MiniModelConfig(model_type="random_forest", n_estimators=30)
    results = trainer.train_mini_model(config)
    
    if results["success"]:
        print(f"   ✅ Training successful!")
        print(f"   Train accuracy: {results['train_score']:.3f}")
        print(f"   Test accuracy: {results['test_score']:.3f}")
        print(f"   Model saved to: {results['model_path']}")
    else:
        print(f"   ❌ Training failed: {results.get('error', 'Unknown error')}")

def main():
    """Main example function."""
    print("🚀 AutoCAD .dwg Symbol Training - Usage Examples")
    print("=" * 60)
    
    try:
        # Example with real .dwg files
        example_with_dwg_files()
        
        # Example with synthetic data
        example_with_synthetic_data()
        
        print(f"\n🎉 Examples completed!")
        print(f"\n💡 Key points:")
        print(f"   1. Put your .dwg files in the 'dwg_files' directory")
        print(f"   2. Use trainer.add_dwg_file() for individual files")
        print(f"   3. Use trainer.add_dwg_directory() for entire folders")
        print(f"   4. Train with trainer.train_mini_model()")
        print(f"   5. Predict with trainer.predict_symbol_type()")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
