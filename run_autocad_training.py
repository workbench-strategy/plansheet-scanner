#!/usr/bin/env python3
"""
Run AutoCAD Training with Your .dwg Files
This script helps you train models on your AutoCAD symbol files and troubleshoot issues.
"""

import os
import sys
from pathlib import Path
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from autocad_dwg_symbol_trainer import AutoCADSymbolTrainer, MiniModelConfig

def find_dwg_files():
    """Find all .dwg files in the dwg_files directory."""
    print("🔍 Finding .dwg files...")
    
    dwg_files_dir = Path.cwd() / "dwg_files"
    all_dwg_files = []
    
    if dwg_files_dir.exists():
        # Search recursively for .dwg files
        for dwg_file in dwg_files_dir.rglob("*.dwg"):
            if dwg_file.stat().st_size > 0:  # Skip empty files
                all_dwg_files.append(dwg_file)
    
    print(f"   Found {len(all_dwg_files)} .dwg files")
    
    # Group by directory
    files_by_dir = {}
    for dwg_file in all_dwg_files:
        parent_dir = dwg_file.parent.name
        if parent_dir not in files_by_dir:
            files_by_dir[parent_dir] = []
        files_by_dir[parent_dir].append(dwg_file)
    
    for dir_name, files in files_by_dir.items():
        print(f"   📁 {dir_name}: {len(files)} files")
    
    return all_dwg_files, files_by_dir

def test_single_file(trainer, dwg_file):
    """Test processing a single .dwg file."""
    print(f"   Testing: {dwg_file.name}")
    try:
        symbols_added = trainer.add_dwg_file(str(dwg_file))
        print(f"      ✅ Extracted {symbols_added} symbols")
        return True
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False

def run_training_with_files():
    """Run the training with the found .dwg files."""
    print("🚀 Running AutoCAD Training with Your .dwg Files")
    print("=" * 60)
    
    # Find .dwg files
    all_dwg_files, files_by_dir = find_dwg_files()
    
    if not all_dwg_files:
        print("❌ No .dwg files found!")
        print("   Make sure your .dwg files are in the dwg_files directory")
        return
    
    # Initialize trainer
    print(f"\n🤖 Initializing AutoCAD Symbol Trainer...")
    trainer = AutoCADSymbolTrainer()
    
    # Test a few files first
    print(f"\n🧪 Testing individual files...")
    test_files = all_dwg_files[:3]  # Test first 3 files
    successful_tests = 0
    
    for dwg_file in test_files:
        if test_single_file(trainer, dwg_file):
            successful_tests += 1
    
    if successful_tests == 0:
        print(f"\n❌ All test files failed!")
        print(f"   This might indicate a problem with:")
        print(f"   1. AutoCAD installation (for win32com method)")
        print(f"   2. File format or corruption")
        print(f"   3. Missing dependencies")
        return
    
    print(f"\n✅ {successful_tests}/{len(test_files)} test files processed successfully")
    
    # Process all files
    print(f"\n📥 Processing all .dwg files...")
    total_symbols = 0
    
    for dwg_file in all_dwg_files:
        try:
            symbols_added = trainer.add_dwg_file(str(dwg_file))
            total_symbols += symbols_added
            if symbols_added > 0:
                print(f"   ✅ {dwg_file.name}: {symbols_added} symbols")
        except Exception as e:
            print(f"   ❌ {dwg_file.name}: {e}")
    
    print(f"\n📊 Total symbols extracted: {total_symbols}")
    
    # Show statistics
    stats = trainer.get_training_statistics()
    print(f"\n📈 Training data statistics:")
    print(f"   Total symbols: {stats['total_symbols']}")
    print(f"   Symbol types: {stats['symbol_types']}")
    print(f"   Entity types: {stats['entity_types']}")
    print(f"   Files processed: {stats['files_processed']}")
    
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
                
                # Show feature importance
                if results.get("feature_importance"):
                    print(f"\n🎯 Top 5 Feature Importance:")
                    importance = results["feature_importance"]
                    top_features = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)[:5]
                    for idx, imp in top_features:
                        print(f"   Feature {idx}: {imp:.3f}")
                
                # Test prediction
                print(f"\n🔮 Testing prediction...")
                if all_dwg_files:
                    test_file = all_dwg_files[0]
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
                print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during training: {e}")
            traceback.print_exc()
    else:
        print(f"\n⚠️  Need at least 10 symbols for training, have {stats['total_symbols']}")
        print(f"   Try processing more .dwg files or check file formats")

def troubleshoot_issues():
    """Help troubleshoot common issues."""
    print(f"\n🔧 Troubleshooting Guide")
    print("=" * 40)
    
    # Check dependencies
    print(f"\n📋 Checking dependencies...")
    
    try:
        import win32com.client
        print(f"   ✅ win32com (AutoCAD COM) available")
    except ImportError:
        print(f"   ❌ win32com not available")
        print(f"      Install with: pip install pywin32")
        print(f"      Make sure AutoCAD is installed on Windows")
    
    try:
        import ezdxf
        print(f"   ✅ ezdxf available")
    except ImportError:
        print(f"   ❌ ezdxf not available")
        print(f"      Install with: pip install ezdxf")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        print(f"   ✅ scikit-learn available")
    except ImportError:
        print(f"   ❌ scikit-learn not available")
        print(f"      Install with: pip install scikit-learn")
    
    # Check file structure
    print(f"\n📁 Checking file structure...")
    dwg_files_dir = Path.cwd() / "dwg_files"
    if dwg_files_dir.exists():
        print(f"   ✅ dwg_files directory exists")
        dwg_count = len(list(dwg_files_dir.rglob("*.dwg")))
        print(f"   📄 Found {dwg_count} .dwg files")
    else:
        print(f"   ❌ dwg_files directory not found")
    
    # Check training data directory
    training_data_dir = Path.cwd() / "autocad_training_data"
    if training_data_dir.exists():
        print(f"   ✅ autocad_training_data directory exists")
    else:
        print(f"   ❌ autocad_training_data directory not found")
    
    # Check models directory
    models_dir = Path.cwd() / "autocad_models"
    if models_dir.exists():
        print(f"   ✅ autocad_models directory exists")
    else:
        print(f"   ❌ autocad_models directory not found")

def main():
    """Main function."""
    try:
        print("🚀 AutoCAD .dwg Symbol Training - Full Process")
        print("=" * 60)
        
        # Run troubleshooting first
        troubleshoot_issues()
        
        # Run training
        run_training_with_files()
        
        print(f"\n🎉 Process completed!")
        print(f"\n💡 Next steps:")
        print(f"   1. Check the autocad_models/ directory for trained models")
        print(f"   2. Use the trained model for predictions on new drawings")
        print(f"   3. Check README_AutoCAD_Symbol_Training.md for more details")
        
    except Exception as e:
        print(f"❌ Process failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
