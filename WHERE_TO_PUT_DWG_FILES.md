# Where to Put Your .dwg Files

## 📁 Quick Answer

**Put your .dwg files in this directory:**
```
C:\Users\jnanderson\OneDrive - HNTB\DIS SEA_DTS Python Working Group - GitHub-repos\plansheet-scanner-new\dwg_files\
```

## 🚀 How to Use Your .dwg Files

### Step 1: Copy Your Files
Copy your AutoCAD .dwg files to the `dwg_files` folder:
```
📁 dwg_files/
   📄 traffic_plan.dwg
   📄 electrical_plan.dwg
   📄 structural_plan.dwg
   📄 your_other_drawings.dwg
```

### Step 2: Run the Training
```python
from autocad_dwg_symbol_trainer import AutoCADSymbolTrainer, MiniModelConfig

# Initialize trainer
trainer = AutoCADSymbolTrainer()

# Add all .dwg files from the directory
trainer.add_dwg_directory("dwg_files")

# Train a model
config = MiniModelConfig(model_type="random_forest", n_estimators=100)
results = trainer.train_mini_model(config)

# Use the trained model
predictions = trainer.predict_symbol_type("new_drawing.dwg")
```

### Step 3: Or Use Individual Files
```python
# Add specific files
trainer.add_dwg_file("dwg_files/traffic_plan.dwg")
trainer.add_dwg_file("dwg_files/electrical_plan.dwg")
```

## 📂 Alternative Locations

You can also put .dwg files in these existing directories:
- `as_built_drawings/` - For as-built drawings
- `real_as_built_data/` - For real project data
- Any folder you specify with the full path

## 🧪 Test First

Run this to test the system:
```bash
python setup_autocad_training.py
python example_autocad_usage.py
```

## 📊 What Happens

1. **Symbol Extraction**: The system extracts symbols from your .dwg files
2. **Feature Engineering**: Converts symbols to 23-dimensional feature vectors
3. **Model Training**: Trains a mini model (Random Forest, Gradient Boosting, or Neural Network)
4. **Prediction**: Uses the trained model to predict symbols in new drawings

## 🎯 Supported Symbol Types

- **Traffic Symbols**: Signal heads, detector loops, signs
- **Electrical Symbols**: Conduits, junction boxes, cables
- **Structural Symbols**: Foundations, beams, columns
- **General Symbols**: Text labels, geometric shapes

## 🔧 Requirements

- **Windows**: AutoCAD installed (for win32com method)
- **Cross-platform**: Install ezdxf: `pip install ezdxf`
- **Python packages**: scikit-learn, numpy, pandas

## 💡 Tips

1. **Layer Organization**: Well-organized layers help with symbol classification
2. **File Size**: Start with smaller files for testing
3. **Symbol Variety**: Include different types of symbols for better training
4. **Consistent Naming**: Use consistent layer names across drawings

## 🆘 Need Help?

1. Run `python setup_autocad_training.py` to check your setup
2. Check `README_AutoCAD_Symbol_Training.md` for detailed documentation
3. Look at `example_autocad_usage.py` for working examples

---

**That's it! Just copy your .dwg files to the `dwg_files` folder and you're ready to train!** 🎉
