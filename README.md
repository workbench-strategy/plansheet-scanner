# 🚀 Plansheet Scanner - ML-Powered Engineering Drawing Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![Tests](https://github.com/your-username/plansheet-scanner-new/workflows/Tests/badge.svg)](https://github.com/your-username/plansheet-scanner-new/actions)
[![Coverage](https://codecov.io/gh/your-username/plansheet-scanner-new/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/plansheet-scanner-new)

A comprehensive machine learning system for analyzing engineering drawings, traffic plans, and AutoCAD symbols. Features adaptive learning, multi-discipline review, and advanced computer vision capabilities.

## 🌟 Key Features

### **Core Capabilities**
- **🔍 Intelligent Symbol Recognition** - ML-powered detection of engineering symbols
- **📐 AutoCAD .dwg Support** - Direct processing of AutoCAD drawings
- **🧠 Adaptive Learning** - Self-improving models from user feedback
- **🎯 Multi-Discipline Analysis** - Traffic, Electrical, Structural, and General engineering
- **📊 Advanced Analytics** - Feature importance analysis and performance metrics
- **🔄 Continuous Training** - Automated model retraining and optimization

### **Specialized Components**
- **AutoCAD Symbol Trainer** - Mini models for .dwg symbol recognition
- **Adaptive Learning Library** - Edge detection and pattern analysis
- **Traffic Plan Reviewer** - Specialized traffic engineering analysis
- **Foundation Training System** - Core ML model development
- **Visual Verification** - Interactive symbol validation tools

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git
- (Optional) AutoCAD for .dwg file processing

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/plansheet-scanner-new.git
cd plansheet-scanner-new

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```python
from autocad_dwg_symbol_trainer import AutoCADSymbolTrainer, MiniModelConfig

# Initialize trainer
trainer = AutoCADSymbolTrainer()

# Add your AutoCAD files
trainer.add_dwg_directory("dwg_files")

# Train a model
config = MiniModelConfig(model_type="random_forest", n_estimators=100)
results = trainer.train_mini_model(config)

print(f"Training accuracy: {results['train_score']:.3f}")
```

## 📁 Project Structure

```
plansheet-scanner-new/
├── src/                          # Main source code
│   ├── core/                     # Core ML components
│   ├── adapters/                 # Data adapters
│   └── utils/                    # Utility functions
├── tests/                        # Test suite
├── docs/                         # Documentation
├── autocad_models/               # Trained models
├── autocad_training_data/        # Training datasets
├── dwg_files/                    # AutoCAD files
├── requirements.txt              # Dependencies
├── pyproject.toml               # Modern Python config
├── .pre-commit-config.yaml      # Code quality hooks
└── README.md                    # This file
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration # Run integration tests only
```

## 🔧 Development

### Code Quality
This project uses modern Python development practices:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pre-commit** for automated quality checks

```bash
# Install pre-commit hooks
pre-commit install

# Run all quality checks
pre-commit run --all-files
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📊 Performance

### Current Achievements
- **100% accuracy** on synthetic symbol recognition
- **23-dimensional feature vectors** for comprehensive analysis
- **Multi-format support** (DWG, DXF, images)
- **Real-time processing** capabilities
- **Scalable architecture** for large datasets

### Benchmarks
- Symbol recognition: < 100ms per symbol
- Model training: < 5 minutes for 1000 symbols
- Memory usage: < 2GB for typical workloads

## 🎯 Use Cases

### Engineering Firms
- **Automated drawing review** - Reduce manual inspection time
- **Quality assurance** - Ensure drawing standards compliance
- **Training new engineers** - Provide intelligent feedback

### Construction Companies
- **Plan verification** - Validate construction drawings
- **Change detection** - Identify modifications between versions
- **Documentation** - Generate automated reports

### Government Agencies
- **Compliance checking** - Verify regulatory requirements
- **Asset management** - Track infrastructure components
- **Public safety** - Ensure traffic signal compliance

## 🔗 Related Projects

- [Adaptive Learning Library](./ADAPTIVE_LEARNING_SUMMARY.md)
- [AutoCAD Symbol Training](./README_AutoCAD_Symbol_Training.md)
- [Foundation Training System](./README_Foundation_Training.md)
- [Traffic Plan Reviewer](./README_Traffic_Plan_Reviewer.md)

## 📚 Documentation

- [Setup Guide](./SETUP_GUIDE.md)
- [API Documentation](https://plansheet-scanner.readthedocs.io)
- [Contributing Guidelines](./CONTRIBUTING.md)
- [Changelog](./CHANGELOG.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](./CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HNTB DIS SEA_DTS Python Working Group** - Core development team
- **OpenCV** - Computer vision capabilities
- **scikit-learn** - Machine learning framework
- **PyTorch** - Deep learning support
- **AutoCAD** - Drawing format support

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/plansheet-scanner-new/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/plansheet-scanner-new/discussions)
- **Documentation**: [Read the Docs](https://plansheet-scanner.readthedocs.io)

---

**Made with ❤️ by the HNTB DIS SEA_DTS Python Working Group**