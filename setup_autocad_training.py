#!/usr/bin/env python3
"""
AutoCAD Training Setup Script
Shows you exactly where to put your .dwg files and how to use them.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from autocad_dwg_symbol_trainer import AutoCADSymbolTrainer, MiniModelConfig

def setup_autocad_training():
    """Setup guide for AutoCAD .dwg training."""
    print("🚀 AutoCAD .dwg Symbol Training Setup")
    print("=" * 50)
    
    # Get current directory
    current_dir = Path.cwd()
    
    print(f"\n📁 Current working directory: {current_dir}")
    
    # Show where to put .dwg files
    print(f"\n📂 Where to put your .dwg files:")
    print(f"   Option 1: Create a 'dwg_files' folder in the current directory")
    print(f"   Option 2: Put them anywhere and specify the full path")
    print(f"   Option 3: Use existing folders like 'as_built_drawings'")
    
    # Create dwg_files directory if it doesn't exist
    dwg_files_dir = current_dir / "dwg_files"
    if not dwg_files_dir.exists():
        dwg_files_dir.mkdir(exist_ok=True)
        print(f"\n✅ Created directory: {dwg_files_dir}")
        print(f"   📁 Put your .dwg files in: {dwg_files_dir}")
    else:
        print(f"\n📁 Existing directory: {dwg_files_dir}")
        print(f"   📁 Your .dwg files can go in: {dwg_files_dir}")
    
    # Show example usage
    print(f"\n💡 Example usage:")
    print(f"   # Initialize trainer")
    print(f"   trainer = AutoCADSymbolTrainer()")
    print(f"   ")
    print(f"   # Add individual .dwg files")
    print(f"   trainer.add_dwg_file('{dwg_files_dir}/traffic_plan.dwg')")
    print(f"   trainer.add_dwg_file('{dwg_files_dir}/electrical_plan.dwg')")
    print(f"   ")
    print(f"   # Add entire directory")
    print(f"   trainer.add_dwg_directory('{dwg_files_dir}')")
    print(f"   ")
    print(f"   # Train model")
    print(f"   config = MiniModelConfig(model_type='random_forest')")
    print(f"   results = trainer.train_mini_model(config)")
    
    # Show existing directories that might contain .dwg files
    print(f"\n🔍 Existing directories that might contain .dwg files:")
    existing_dirs = []
    for item in current_dir.iterdir():
        if item.is_dir() and any(keyword in item.name.lower() for keyword in ['drawing', 'dwg', 'autocad', 'plan', 'as_built']):
            existing_dirs.append(item)
    
    if existing_dirs:
        for dir_path in existing_dirs:
            print(f"   📁 {dir_path.name}/")
            # Check if it contains .dwg files
            dwg_count = len(list(dir_path.glob("*.dwg")))
            if dwg_count > 0:
                print(f"      ✅ Contains {dwg_count} .dwg files")
            else:
                print(f"      ⚠️  No .dwg files found")
    else:
        print(f"   No obvious drawing directories found")
    
    # Show training data directory
    training_data_dir = current_dir / "autocad_training_data"
    print(f"\n📊 Training data will be saved to:")
    print(f"   📁 {training_data_dir}")
    
    # Show models directory
    models_dir = current_dir / "autocad_models"
    print(f"\n🤖 Trained models will be saved to:")
    print(f"   📁 {models_dir}")
    
    return dwg_files_dir

def demonstrate_usage():
    """Demonstrate how to use the system with .dwg files."""
    print(f"\n🎯 How to use your .dwg files:")
    print(f"=" * 40)
    
    # Initialize trainer
    trainer = AutoCADSymbolTrainer()
    
    # Show available extraction methods
    print(f"\n📋 Available extraction methods:")
    print(f"   {trainer.symbol_extractor.extraction_methods}")
    
    # Example with synthetic data (since we don't have real .dwg files)
    print(f"\n🧪 Testing with synthetic data first:")
    
    # Create some synthetic symbols for demonstration
    from autocad_dwg_symbol_trainer import AutoCADSymbol
    import numpy as np
    
    # Add a few synthetic symbols
    synthetic_symbols = []
    for i in range(5):
        symbol = AutoCADSymbol(
            symbol_id=f"demo_signal_{i}",
            symbol_name="traffic_signal",
            symbol_type="traffic",
            entity_type="AcDbCircle",
            geometry={
                "center": [100 + i * 20, 200 + i * 10],
                "radius": 1.5,
                "area": np.pi * 1.5 ** 2
            },
            layer_name="TRAFFIC_SIGNALS",
            color="RED",
            linetype="CONTINUOUS",
            lineweight=0.5,
            file_path="demo_traffic.dwg",
            bounding_box=(100 + i * 20 - 2, 200 + i * 10 - 2, 
                         100 + i * 20 + 2, 200 + i * 10 + 2)
        )
        synthetic_symbols.append(symbol)
    
    # Add to trainer
    for symbol in synthetic_symbols:
        trainer.autocad_symbols.append(symbol)
    
    print(f"   ✅ Added {len(synthetic_symbols)} demo symbols")
    
    # Show statistics
    stats = trainer.get_training_statistics()
    print(f"   📊 Current training data:")
    print(f"      Total symbols: {stats['total_symbols']}")
    print(f"      Symbol types: {stats['symbol_types']}")
    
    # Test feature extraction
    if trainer.autocad_symbols:
        sample_symbol = trainer.autocad_symbols[0]
        features = trainer.extract_features(sample_symbol)
        print(f"   🔍 Feature extraction test:")
        print(f"      Sample symbol: {sample_symbol.symbol_name}")
        print(f"      Feature vector length: {len(features)}")
    
    print(f"\n✅ System is ready to use with your .dwg files!")

def main():
    """Main setup function."""
    try:
        # Setup directories
        dwg_files_dir = setup_autocad_training()
        
        # Demonstrate usage
        demonstrate_usage()
        
        print(f"\n🎉 Setup complete!")
        print(f"\n📋 Next steps:")
        print(f"   1. Copy your .dwg files to: {dwg_files_dir}")
        print(f"   2. Run: python autocad_dwg_symbol_trainer.py")
        print(f"   3. Or use the training system in your own scripts")
        print(f"   4. Check the README_AutoCAD_Symbol_Training.md for detailed instructions")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
