# Symbol Recognition Training System

This system focuses on the foundational step of training an AI to recognize and understand engineering symbols from plans, legends, and notes. Once this foundation is established, you can move on to training on standards and requirements.

## Overview

The symbol recognition training system learns to:
- Identify engineering symbols from visual descriptions
- Parse legend references and understand symbol definitions
- Extract meaning from notes and specifications
- Classify symbols by discipline (traffic, electrical, structural, drainage)
- Build confidence in symbol identification

## Quick Start

### 1. Start the Background Training Service

```bash
# Start the symbol training service in the background
python symbol_training_control.py start

# Or start in foreground for debugging
python symbol_training_control.py start --foreground
```

### 2. Check Training Status

```bash
# Check if the service is running and view statistics
python symbol_training_control.py status
```

### 3. Monitor Training Progress

```bash
# View recent training logs
python symbol_training_control.py logs

# View more log lines
python symbol_training_control.py logs --lines 50
```

### 4. Stop the Service

```bash
# Stop the background training service
python symbol_training_control.py stop
```

## System Components

### Core Files

- **`symbol_recognition_trainer.py`** - Main training engine for symbol recognition
- **`symbol_background_trainer.py`** - Background service that runs training automatically
- **`symbol_training_control.py`** - Control interface for managing the training service
- **`config/symbol_training_config.json`** - Configuration settings

### Data Directories

- **`symbol_training_data/`** - Stores training data and symbol examples
- **`symbol_models/`** - Stores trained models
- **`logs/`** - Training logs and status files

## Configuration

The system is configured through `config/symbol_training_config.json`:

### Training Schedule
- **Interval**: How often to run training (default: 4 hours)
- **Max Time**: Maximum training session duration (default: 60 minutes)
- **Idle Threshold**: CPU usage threshold for starting training (default: 25%)

### Model Configuration
- **Min Examples**: Minimum examples needed to start training (default: 30)
- **Max Examples**: Maximum examples to generate (default: 500)
- **Validation Split**: Percentage for validation (default: 20%)

### Symbol-Specific Settings
- **Disciplines**: Engineering disciplines to train on
- **Focus Areas**: What aspects to focus on (visual, legend, notes)
- **Confidence Threshold**: Minimum confidence for predictions

## How It Works

### 1. Symbol Knowledge Base
The system includes a comprehensive knowledge base of engineering symbols organized by discipline:

- **Traffic**: Traffic signals, signs, markings
- **Electrical**: Junction boxes, conduits, equipment
- **Structural**: Beams, columns, connections
- **Drainage**: Catch basins, manholes, pipes

### 2. Training Data Generation
The system automatically generates diverse training examples with:
- Visual descriptions of symbols
- Legend references and variations
- Notes and specifications
- Context clues for identification

### 3. Model Training
Three main models are trained:
- **Symbol Classifier**: Identifies specific symbols
- **Discipline Classifier**: Categorizes by engineering discipline
- **Confidence Predictor**: Estimates identification confidence

### 4. Background Operation
The service runs automatically when:
- System is idle (low CPU usage)
- No intensive applications are running
- Sufficient memory is available

## Monitoring and Control

### Status Information
```bash
python symbol_training_control.py status
```

Shows:
- Service running status
- Last training time
- Training statistics (symbols, patterns, confidence)
- Performance metrics (CPU, memory, training time)
- Discipline and symbol distributions

### Log Monitoring
```bash
python symbol_training_control.py logs
```

Provides detailed training logs including:
- Training session start/stop times
- Data generation progress
- Model training results
- Performance metrics
- Error messages

### Configuration Review
```bash
python symbol_training_control.py config
```

Displays current configuration settings for review and verification.

## Training Process

### Phase 1: Data Generation
1. System generates diverse symbol examples
2. Creates variations in descriptions and references
3. Builds context clues for each discipline
4. Saves training data to disk

### Phase 2: Model Training
1. Extracts features from symbol data
2. Trains symbol classification model
3. Trains discipline classification model
4. Trains confidence prediction model
5. Evaluates model performance

### Phase 3: Continuous Learning
1. Monitors system resources
2. Runs training when system is idle
3. Generates additional data as needed
4. Updates models with new examples
5. Maintains performance metrics

## Next Steps

Once the symbol recognition foundation is established:

1. **Test Symbol Identification**: Verify the system can correctly identify symbols from plans
2. **Add Real-World Data**: Incorporate actual engineering plans and symbols
3. **Expand Knowledge Base**: Add more symbols and variations
4. **Move to Standards Training**: Begin training on code standards and requirements
5. **Integration**: Connect with plan review and analysis systems

## Troubleshooting

### Service Won't Start
- Check if required dependencies are installed
- Verify configuration file exists and is valid
- Check for existing processes and stop them first

### No Training Progress
- Check system resources (CPU, memory)
- Review idle threshold settings
- Check logs for error messages
- Verify data directories exist and are writable

### Poor Symbol Recognition
- Increase training data quantity
- Adjust confidence thresholds
- Review symbol knowledge base
- Check feature extraction logic

## Dependencies

Required Python packages:
- scikit-learn (for ML models)
- numpy (for numerical operations)
- pandas (for data handling)
- psutil (for system monitoring)
- schedule (for task scheduling)

Install with:
```bash
pip install scikit-learn numpy pandas psutil schedule
```

## File Structure

```
├── symbol_recognition_trainer.py      # Main training engine
├── symbol_background_trainer.py       # Background service
├── symbol_training_control.py         # Control interface
├── config/
│   └── symbol_training_config.json    # Configuration
├── symbol_training_data/              # Training data
├── symbol_models/                     # Trained models
└── logs/                              # Logs and status
    ├── symbol_training.log
    ├── symbol_training_status.json
    └── symbol_performance.log
```

## Support

For issues or questions:
1. Check the logs for error messages
2. Review configuration settings
3. Verify system requirements
4. Test with foreground mode for debugging

The symbol recognition system is designed to run quietly in the background, learning from your engineering plans without interrupting your work. Once it has a solid foundation in symbol recognition, you'll be ready to move on to the more complex task of training on standards and requirements.
