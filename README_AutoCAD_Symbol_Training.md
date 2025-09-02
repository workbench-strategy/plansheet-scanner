# AutoCAD .dwg Symbol Training System

A specialized machine learning system for training mini models to recognize engineering symbols directly from AutoCAD .dwg files. This system can extract geometric entities, analyze their properties, and train lightweight models for symbol classification.

## 🚀 Features

### **Core Capabilities**
- **Direct .dwg File Processing** - Extract symbols from AutoCAD drawings using multiple methods
- **Multi-Method Extraction** - Support for both win32com (AutoCAD COM) and ezdxf libraries
- **Geometric Feature Analysis** - Extract meaningful features from AutoCAD entities
- **Mini Model Training** - Train lightweight models (Random Forest, Gradient Boosting, Neural Networks)
- **Symbol Classification** - Classify symbols by discipline (traffic, electrical, structural, etc.)
- **Model Persistence** - Save and load trained models for reuse

### **Supported AutoCAD Entities**
- **Circles** (AcDbCircle) - Signal heads, manholes, junction boxes
- **Polylines** (AcDbPolyline) - Detector loops, conduits, foundations
- **Lines** (AcDbLine) - Electrical wires, conduits, structural elements
- **Text** (AcDbText) - Labels, notes, annotations
- **Arcs** (AcDbArc) - Curved elements
- **Blocks** (AcDbBlockReference) - Complex symbols and assemblies

### **Symbol Types Recognized**
- **Traffic Symbols** - Signal heads, detector loops, signs
- **Electrical Symbols** - Conduits, junction boxes, cables
- **Structural Symbols** - Foundations, beams, columns
- **General Symbols** - Text labels, geometric shapes

## 📋 Requirements

### **Required Dependencies**
```bash
pip install scikit-learn joblib numpy pandas
```

### **Optional Dependencies**
```bash
# For AutoCAD COM integration (Windows only)
pip install pywin32

# For ezdxf library (cross-platform)
pip install ezdxf

# For neural network models
pip install torch torchvision
```

## 🛠️ Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd plansheet-scanner-new
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install AutoCAD integration** (choose one or both):
```bash
# For Windows with AutoCAD installed
pip install pywin32

# For cross-platform DWG reading
pip install ezdxf
```

## 🚀 Quick Start

### **1. Basic Usage**

```python
from autocad_dwg_symbol_trainer import AutoCADSymbolTrainer, MiniModelConfig

# Initialize trainer
trainer = AutoCADSymbolTrainer()

# Add .dwg files
trainer.add_dwg_file("path/to/drawing.dwg")
trainer.add_dwg_directory("path/to/dwg/folder")

# Train a mini model
config = MiniModelConfig(model_type="random_forest", n_estimators=100)
results = trainer.train_mini_model(config)

# Predict symbols in new drawings
predictions = trainer.predict_symbol_type("new_drawing.dwg")
```

### **2. Test the System**

Run the test script to verify everything works:

```bash
python test_autocad_symbol_trainer.py
```

This will:
- Create synthetic AutoCAD symbol data
- Test feature extraction
- Train multiple model types
- Test prediction capabilities
- Verify model persistence

## 📊 System Architecture

### **Core Components**

1. **AutoCADSymbolExtractor**
   - Extracts symbols from .dwg files
   - Supports multiple extraction methods
   - Processes different entity types

2. **AutoCADSymbolTrainer**
   - Manages training data
   - Extracts features from symbols
   - Trains and evaluates models

3. **MiniModelConfig**
   - Configures model parameters
   - Supports different model types
   - Controls training behavior

### **Data Flow**

```
.dwg Files → Symbol Extraction → Feature Engineering → Model Training → Prediction
```

## 🔧 Configuration

### **Model Configuration Options**

```python
config = MiniModelConfig(
    model_type="random_forest",      # "random_forest", "gradient_boost", "neural_network"
    feature_extraction="geometric",  # "geometric", "visual", "hybrid"
    max_depth=10,                   # Tree depth for tree-based models
    n_estimators=100,               # Number of trees/estimators
    learning_rate=0.1,              # Learning rate for gradient boosting
    batch_size=32,                  # Batch size for neural networks
    epochs=50,                      # Training epochs for neural networks
    validation_split=0.2            # Validation data split
)
```

### **Feature Extraction**

The system extracts comprehensive features from AutoCAD symbols:

- **Geometric Features**: Radius, area, length, vertices, coordinates
- **Layer Features**: Layer name analysis, keyword detection
- **Color Features**: Color classification and analysis
- **Lineweight Features**: Line thickness information
- **Bounding Box Features**: Width, height, area calculations

## 📈 Training Process

### **1. Data Collection**
```python
# Add individual files
trainer.add_dwg_file("traffic_plan.dwg")
trainer.add_dwg_file("electrical_plan.dwg")

# Add entire directories
trainer.add_dwg_directory("project_drawings/")
```

### **2. Feature Extraction**
```python
# Extract features from symbols
for symbol in trainer.autocad_symbols:
    features = trainer.extract_features(symbol)
    # Features include geometric, layer, color, and spatial information
```

### **3. Model Training**
```python
# Train with different configurations
configs = [
    MiniModelConfig(model_type="random_forest", n_estimators=100),
    MiniModelConfig(model_type="gradient_boost", n_estimators=100),
    MiniModelConfig(model_type="neural_network", epochs=50)
]

for config in configs:
    results = trainer.train_mini_model(config)
    print(f"Model: {config.model_type}")
    print(f"Accuracy: {results['test_score']:.3f}")
```

### **4. Model Evaluation**
```python
# Get training statistics
stats = trainer.get_training_statistics()
print(f"Total symbols: {stats['total_symbols']}")
print(f"Symbol types: {stats['symbol_types']}")
print(f"Entity types: {stats['entity_types']}")
```

## 🔮 Prediction and Inference

### **Predict Symbols in New Drawings**

```python
# Load a trained model
trainer.load_model("autocad_symbol_model_20241201_143022.pkl")

# Predict symbols in a new drawing
predictions = trainer.predict_symbol_type("new_project.dwg")

for pred in predictions:
    print(f"Symbol: {pred['symbol_name']}")
    print(f"Type: {pred['predicted_type']}")
    print(f"Confidence: {pred['confidence']:.3f}")
    print(f"Layer: {pred['layer_name']}")
```

### **Batch Processing**

```python
import os
from pathlib import Path

# Process multiple drawings
dwg_directory = Path("new_project_drawings")
for dwg_file in dwg_directory.glob("*.dwg"):
    predictions = trainer.predict_symbol_type(str(dwg_file))
    print(f"Processed {dwg_file.name}: {len(predictions)} symbols")
```

## 💾 Model Management

### **Save and Load Models**

```python
# Save trained model
model_path = "my_autocad_model.pkl"
trainer.save_model(model_path)

# Load model in new session
new_trainer = AutoCADSymbolTrainer()
new_trainer.load_model(model_path)

# Use loaded model for predictions
predictions = new_trainer.predict_symbol_type("test.dwg")
```

### **Model Versioning**

```python
# Save with timestamp
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f"autocad_model_v1_{timestamp}.pkl"
trainer.save_model(model_path)
```

## 📊 Performance and Optimization

### **Model Performance**

The system supports multiple model types with different characteristics:

- **Random Forest**: Fast training, good interpretability, feature importance
- **Gradient Boosting**: High accuracy, slower training, good for complex patterns
- **Neural Network**: Best for large datasets, requires more data

### **Feature Importance Analysis**

```python
# Get feature importance for tree-based models
if results.get("feature_importance"):
    importance = results["feature_importance"]
    top_features = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)[:10]
    
    for idx, imp in top_features:
        print(f"Feature {idx}: {imp:.3f}")
```

### **Optimization Tips**

1. **Data Quality**: Ensure .dwg files have proper layer organization
2. **Feature Engineering**: Add domain-specific features for better performance
3. **Model Selection**: Use Random Forest for quick prototyping, Gradient Boosting for production
4. **Hyperparameter Tuning**: Adjust model parameters based on your data characteristics

## 🔍 Troubleshooting

### **Common Issues**

1. **No AutoCAD COM Available**
   ```
   Warning: win32com not available. Install with: pip install pywin32
   ```
   - Install pywin32: `pip install pywin32`
   - Ensure AutoCAD is installed on Windows

2. **ezdxf Not Available**
   ```
   Warning: ezdxf not available. Install with: pip install ezdxf
   ```
   - Install ezdxf: `pip install ezdxf`
   - Works on all platforms

3. **Insufficient Training Data**
   ```
   Need at least 10 symbols for training, have X
   ```
   - Add more .dwg files with symbols
   - Use synthetic data for testing

4. **Model Training Fails**
   - Check data diversity (need multiple symbol types)
   - Verify feature extraction is working
   - Ensure proper data preprocessing

### **Debug Mode**

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
trainer = AutoCADSymbolTrainer()
symbols = trainer.symbol_extractor.extract_symbols_from_dwg("test.dwg")
print(f"Extracted {len(symbols)} symbols")
```

## 📚 Advanced Usage

### **Custom Feature Extraction**

```python
class CustomAutoCADSymbolTrainer(AutoCADSymbolTrainer):
    def extract_features(self, symbol):
        # Call parent method
        features = super().extract_features(symbol)
        
        # Add custom features
        if symbol.entity_type == "AcDbCircle":
            # Add custom circle features
            features.append(symbol.geometry.get("radius", 0) * 2)
        
        return features
```

### **Integration with Existing Systems**

```python
# Integrate with existing symbol recognition
from symbol_recognition_trainer import SymbolRecognitionTrainer

# Convert AutoCAD symbols to engineering symbols
autocad_trainer = AutoCADSymbolTrainer()
symbol_trainer = SymbolRecognitionTrainer()

# Extract symbols from .dwg
autocad_trainer.add_dwg_file("drawing.dwg")

# Convert and add to symbol trainer
for autocad_symbol in autocad_trainer.autocad_symbols:
    # Convert format and add to symbol trainer
    pass
```

## 🤝 Contributing

### **Adding New Entity Types**

1. Add processing method in `AutoCADSymbolExtractor`
2. Update feature extraction in `AutoCADSymbolTrainer`
3. Add tests for new functionality

### **Improving Symbol Classification**

1. Enhance layer name analysis
2. Add more geometric features
3. Implement context-aware classification

## 📄 License

This project is part of the Plansheet Scanner system. See the main LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test script to verify setup
3. Review the example code in this README
4. Check the main project documentation

---

**Note**: This system is designed to work with engineering drawings and AutoCAD .dwg files. For best results, ensure your drawings have proper layer organization and consistent symbol usage.
