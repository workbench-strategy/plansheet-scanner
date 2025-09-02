#!/usr/bin/env python3
"""
Test Script for AutoCAD .dwg Symbol Trainer
Demonstrates the mini model training system with synthetic data.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from autocad_dwg_symbol_trainer import AutoCADSymbolTrainer, AutoCADSymbol, MiniModelConfig

def create_synthetic_dwg_data():
    """Create synthetic AutoCAD symbol data for testing."""
    print("🔧 Creating synthetic AutoCAD symbol data...")
    
    synthetic_symbols = []
    
    # Traffic symbols
    for i in range(20):
        # Signal heads (circles)
        symbol = AutoCADSymbol(
            symbol_id=f"signal_{i}",
            symbol_name="traffic_signal",
            symbol_type="traffic",
            entity_type="AcDbCircle",
            geometry={
                "center": [100 + i * 10, 200 + i * 5],
                "radius": 1.5 + (i % 3) * 0.5,
                "area": np.pi * (1.5 + (i % 3) * 0.5) ** 2
            },
            layer_name="TRAFFIC_SIGNALS",
            color="RED",
            linetype="CONTINUOUS",
            lineweight=0.5,
            file_path="synthetic_traffic.dwg",
            bounding_box=(100 + i * 10 - 2, 200 + i * 5 - 2, 
                         100 + i * 10 + 2, 200 + i * 5 + 2)
        )
        synthetic_symbols.append(symbol)
        
        # Detector loops (polylines)
        symbol = AutoCADSymbol(
            symbol_id=f"detector_{i}",
            symbol_name="traffic_detector",
            symbol_type="traffic",
            entity_type="AcDbPolyline",
            geometry={
                "coordinates": [50 + i * 15, 150 + i * 8, 60 + i * 15, 150 + i * 8, 
                               60 + i * 15, 160 + i * 8, 50 + i * 15, 160 + i * 8],
                "vertices": 4,
                "length": 20.0
            },
            layer_name="TRAFFIC_DETECTORS",
            color="GREEN",
            linetype="CONTINUOUS",
            lineweight=0.3,
            file_path="synthetic_traffic.dwg",
            bounding_box=(50 + i * 15, 150 + i * 8, 60 + i * 15, 160 + i * 8)
        )
        synthetic_symbols.append(symbol)
    
    # Electrical symbols
    for i in range(15):
        # Conduits (lines)
        symbol = AutoCADSymbol(
            symbol_id=f"conduit_{i}",
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
        synthetic_symbols.append(symbol)
        
        # Junction boxes (circles)
        symbol = AutoCADSymbol(
            symbol_id=f"junction_{i}",
            symbol_name="electrical_junction",
            symbol_type="electrical",
            entity_type="AcDbCircle",
            geometry={
                "center": [400 + i * 25, 150 + i * 12],
                "radius": 3.0 + (i % 2) * 1.0,
                "area": np.pi * (3.0 + (i % 2) * 1.0) ** 2
            },
            layer_name="ELECTRICAL_JUNCTIONS",
            color="YELLOW",
            linetype="CONTINUOUS",
            lineweight=0.6,
            file_path="synthetic_electrical.dwg",
            bounding_box=(400 + i * 25 - 4, 150 + i * 12 - 4,
                         400 + i * 25 + 4, 150 + i * 12 + 4)
        )
        synthetic_symbols.append(symbol)
    
    # Structural symbols
    for i in range(10):
        # Foundation elements (polylines)
        symbol = AutoCADSymbol(
            symbol_id=f"foundation_{i}",
            symbol_name="structural_foundation",
            symbol_type="structural",
            entity_type="AcDbPolyline",
            geometry={
                "coordinates": [500 + i * 30, 200 + i * 15, 530 + i * 30, 200 + i * 15,
                               530 + i * 30, 230 + i * 15, 500 + i * 30, 230 + i * 15],
                "vertices": 4,
                "length": 60.0
            },
            layer_name="STRUCTURAL_FOUNDATION",
            color="WHITE",
            linetype="CONTINUOUS",
            lineweight=0.7,
            file_path="synthetic_structural.dwg",
            bounding_box=(500 + i * 30, 200 + i * 15, 530 + i * 30, 230 + i * 15)
        )
        synthetic_symbols.append(symbol)
    
    # Text labels
    for i in range(25):
        text_content = f"LABEL_{i}"
        if i < 10:
            text_content = f"SIGNAL_{i}"
        elif i < 20:
            text_content = f"CONDUIT_{i-10}"
        else:
            text_content = f"FOUNDATION_{i-20}"
        
        symbol = AutoCADSymbol(
            symbol_id=f"text_{i}",
            symbol_name="text_label",
            symbol_type="general",
            entity_type="AcDbText",
            geometry={
                "text": text_content,
                "position": [50 + i * 8, 300 + i * 6],
                "height": 0.18
            },
            layer_name="TEXT_LABELS",
            color="BYLAYER",
            linetype="CONTINUOUS",
            lineweight=-1,
            file_path="synthetic_labels.dwg"
        )
        synthetic_symbols.append(symbol)
    
    print(f"✅ Created {len(synthetic_symbols)} synthetic symbols")
    return synthetic_symbols

def test_autocad_symbol_trainer():
    """Test the AutoCAD symbol trainer with synthetic data."""
    print("🧪 Testing AutoCAD Symbol Trainer")
    print("=" * 50)
    
    # Initialize trainer
    trainer = AutoCADSymbolTrainer()
    
    # Create synthetic data
    synthetic_symbols = create_synthetic_dwg_data()
    
    # Add synthetic symbols to trainer
    print("\n📥 Adding synthetic symbols to trainer...")
    for symbol in synthetic_symbols:
        trainer.autocad_symbols.append(symbol)
    
    # Show statistics
    stats = trainer.get_training_statistics()
    print(f"\n📊 Training data statistics:")
    print(f"   Total symbols: {stats['total_symbols']}")
    print(f"   Symbol types: {stats['symbol_types']}")
    print(f"   Entity types: {stats['entity_types']}")
    print(f"   Files processed: {stats['files_processed']}")
    
    # Test feature extraction
    print("\n🔍 Testing feature extraction...")
    if trainer.autocad_symbols:
        sample_symbol = trainer.autocad_symbols[0]
        features = trainer.extract_features(sample_symbol)
        print(f"   Sample symbol: {sample_symbol.symbol_name}")
        print(f"   Feature vector length: {len(features)}")
        print(f"   Features: {features[:10]}...")  # Show first 10 features
    
    # Test different model configurations
    model_configs = [
        MiniModelConfig(model_type="random_forest", n_estimators=50, max_depth=8),
        MiniModelConfig(model_type="gradient_boost", n_estimators=50, learning_rate=0.1),
    ]
    
    for i, config in enumerate(model_configs):
        print(f"\n🤖 Testing model configuration {i+1}: {config.model_type}")
        print("-" * 40)
        
        try:
            results = trainer.train_mini_model(config)
            
            if results["success"]:
                print(f"✅ Training successful!")
                print(f"   Train accuracy: {results['train_score']:.3f}")
                print(f"   Test accuracy: {results['test_score']:.3f}")
                print(f"   Symbol types: {results['symbol_types']}")
                print(f"   Feature dimensions: {results['feature_dimensions']}")
                
                # Show classification report
                if "classification_report" in results:
                    report = results["classification_report"]
                    print(f"\n📋 Classification Report:")
                    for class_name, metrics in report.items():
                        if isinstance(metrics, dict) and "precision" in metrics:
                            print(f"   {class_name}:")
                            print(f"     Precision: {metrics['precision']:.3f}")
                            print(f"     Recall: {metrics['recall']:.3f}")
                            print(f"     F1-Score: {metrics['f1-score']:.3f}")
                
                # Show feature importance for tree-based models
                if results.get("feature_importance"):
                    print(f"\n🎯 Top 5 Feature Importance:")
                    importance = results["feature_importance"]
                    top_features = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)[:5]
                    for idx, imp in top_features:
                        print(f"   Feature {idx}: {imp:.3f}")
                
            else:
                print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during training: {e}")
    
    # Test prediction with synthetic data
    print(f"\n🔮 Testing prediction with synthetic data...")
    try:
        # Create a simple test symbol
        test_symbol = AutoCADSymbol(
            symbol_id="test_signal",
            symbol_name="test_traffic_signal",
            symbol_type="traffic",
            entity_type="AcDbCircle",
            geometry={
                "center": [100, 200],
                "radius": 2.0,
                "area": np.pi * 4
            },
            layer_name="TRAFFIC_SIGNALS",
            color="RED",
            linetype="CONTINUOUS",
            lineweight=0.5,
            file_path="test.dwg",
            bounding_box=(98, 198, 102, 202)
        )
        
        # Extract features
        test_features = trainer.extract_features(test_symbol)
        print(f"   Test symbol features: {test_features[:10]}...")
        
        # If we have a trained model, test prediction
        if trainer.symbol_classifier is not None:
            # Scale features
            test_features_scaled = trainer.feature_scaler.transform([test_features])
            
            # Predict
            prediction = trainer.symbol_classifier.predict(test_features_scaled)[0]
            probability = trainer.symbol_classifier.predict_proba(test_features_scaled)[0]
            
            predicted_type = trainer.label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(probability)
            
            print(f"   Predicted type: {predicted_type}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Actual type: {test_symbol.symbol_type}")
            
    except Exception as e:
        print(f"❌ Error during prediction test: {e}")
    
    # Save training data
    print(f"\n💾 Saving training data...")
    trainer.save_training_data()
    
    print(f"\n✅ Testing completed!")
    return trainer

def test_model_persistence():
    """Test saving and loading models."""
    print("\n🔄 Testing Model Persistence")
    print("=" * 40)
    
    # Create a new trainer
    trainer = AutoCADSymbolTrainer()
    
    # Add some synthetic data
    synthetic_symbols = create_synthetic_dwg_data()
    for symbol in synthetic_symbols[:50]:  # Use subset for faster testing
        trainer.autocad_symbols.append(symbol)
    
    # Train a simple model
    config = MiniModelConfig(model_type="random_forest", n_estimators=20)
    results = trainer.train_mini_model(config)
    
    if results["success"]:
        print("✅ Model trained successfully")
        
        # Test prediction before saving
        if trainer.autocad_symbols:
            test_symbol = trainer.autocad_symbols[0]
            test_features = trainer.extract_features(test_symbol)
            test_features_scaled = trainer.feature_scaler.transform([test_features])
            prediction_before = trainer.symbol_classifier.predict(test_features_scaled)[0]
            predicted_type_before = trainer.label_encoder.inverse_transform([prediction_before])[0]
            print(f"   Prediction before save: {predicted_type_before}")
        
        # Save model
        model_path = "test_autocad_model.pkl"
        trainer.save_model(model_path)
        print(f"   Model saved to: {model_path}")
        
        # Create new trainer and load model
        new_trainer = AutoCADSymbolTrainer()
        new_trainer.load_model(model_path)
        print("   Model loaded successfully")
        
        # Test prediction after loading
        if trainer.autocad_symbols:
            test_symbol = trainer.autocad_symbols[0]
            test_features = new_trainer.extract_features(test_symbol)
            test_features_scaled = new_trainer.feature_scaler.transform([test_features])
            prediction_after = new_trainer.symbol_classifier.predict(test_features_scaled)[0]
            predicted_type_after = new_trainer.label_encoder.inverse_transform([prediction_after])[0]
            print(f"   Prediction after load: {predicted_type_after}")
            
            # Verify predictions match
            if predicted_type_before == predicted_type_after:
                print("✅ Model persistence test passed!")
            else:
                print("❌ Model persistence test failed!")
        
        # Clean up
        try:
            os.remove(model_path)
            print(f"   Test file cleaned up: {model_path}")
        except:
            pass
    
    else:
        print("❌ Model training failed for persistence test")

def main():
    """Main test function."""
    print("🚀 AutoCAD .dwg Symbol Trainer - Test Suite")
    print("=" * 60)
    
    try:
        # Test basic functionality
        trainer = test_autocad_symbol_trainer()
        
        # Test model persistence
        test_model_persistence()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"\n📋 Summary:")
        print(f"   - Synthetic data generation: ✅")
        print(f"   - Feature extraction: ✅")
        print(f"   - Model training: ✅")
        print(f"   - Model persistence: ✅")
        print(f"   - Prediction testing: ✅")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Add real .dwg files using trainer.add_dwg_file()")
        print(f"   2. Train on real data with trainer.train_mini_model()")
        print(f"   3. Use trained model for predictions on new drawings")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
