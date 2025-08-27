#!/usr/bin/env python3
"""
Feed the ML System - Demonstration Script
Shows how to feed the machine learning system with training data.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_image(width=1000, height=1000):
    """Create a sample image for training."""
    # Create a simple image with some patterns
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some structured patterns to simulate a traffic plan
    # Signal heads (red circles)
    cv2.circle(image, (100, 200), 20, (0, 0, 255), -1)  # Red signal
    cv2.circle(image, (100, 220), 20, (0, 255, 0), -1)  # Green signal
    
    # Detector loops (rectangles)
    cv2.rectangle(image, (80, 180), (120, 190), (255, 255, 0), 2)  # Yellow loop
    
    # Pedestrian button (blue circle)
    cv2.circle(image, (120, 240), 15, (255, 0, 0), -1)  # Blue button
    
    return image

def feed_foundation_trainer():
    """Demonstrate feeding the foundation trainer."""
    print("🤖 Feeding Foundation Trainer...")
    
    try:
        from src.core.foundation_trainer import FoundationTrainer
        
        # Initialize trainer
        trainer = FoundationTrainer("training_data", "models")
        
        # Create sample image
        sample_image = create_sample_image()
        
        # Load training data
        with open("sample_training_data.json", "r") as f:
            training_data = json.load(f)
        
        with open("sample_project_info.json", "r") as f:
            project_info = json.load(f)
        
        # Add as-built data
        trainer.add_as_built_data(
            plan_id=training_data["plan_id"],
            plan_type=training_data["plan_type"],
            as_built_image=sample_image,
            final_elements=training_data["final_elements"],
            construction_notes=training_data["construction_notes"],
            approval_date=datetime.now(),
            project_info=project_info
        )
        
        # Add review milestone
        trainer.add_review_milestone(
            plan_id=training_data["plan_id"],
            milestone="final",
            reviewer_comments=[
                "Signal head placement meets ITE standards",
                "Detector coverage adequate for traffic flow",
                "Pedestrian features properly implemented",
                "All elements installed per approved plans"
            ],
            approved_elements=training_data["final_elements"],
            rejected_elements=[],
            compliance_score=0.95,
            review_date=datetime.now()
        )
        
        print("✅ Successfully fed foundation trainer with sample data")
        
        # Extract features and train models
        print("🔍 Extracting training features...")
        trainer.extract_training_features()
        
        print("🤖 Training foundation models...")
        trainer.train_foundation_models(min_examples=1)
        
        # Show statistics
        stats = trainer.get_training_statistics()
        print(f"\n📊 Training Statistics:")
        print(f"   Total as-built records: {stats['total_as_built']}")
        print(f"   Total milestone records: {stats['total_milestones']}")
        print(f"   Total training examples: {stats['total_training_examples']}")
        print(f"   Models trained: {stats['models_trained']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch scikit-learn joblib opencv-python")
        return False
    except Exception as e:
        print(f"❌ Error feeding foundation trainer: {e}")
        return False

def feed_adaptive_reviewer():
    """Demonstrate feeding the adaptive reviewer."""
    print("\n🧠 Feeding Adaptive Reviewer...")
    
    try:
        from src.core.adaptive_reviewer import AdaptiveReviewer
        
        # Initialize adaptive reviewer
        reviewer = AdaptiveReviewer("models")
        
        # Create sample feedback
        original_prediction = {
            "elements": [
                {"type": "signal_head_red", "location": [100, 200], "confidence": 0.85},
                {"type": "signal_head_green", "location": [100, 220], "confidence": 0.82},
                {"type": "detector_loop", "location": [80, 180], "confidence": 0.78}
            ],
            "overall_confidence": 0.82
        }
        
        human_corrections = {
            "approved_elements": [
                {"type": "signal_head_red", "location": [100, 200]},
                {"type": "signal_head_green", "location": [100, 220]},
                {"type": "detector_loop", "location": [80, 180]},
                {"type": "pedestrian_button", "location": [120, 240]}
            ],
            "rejected_elements": [],
            "notes": "Good detection but missed pedestrian features"
        }
        
        # Record feedback
        reviewer.record_feedback(
            plan_id="sample_plan_001",
            plan_type="traffic_signal",
            reviewer_id="reviewer_001",
            original_prediction=original_prediction,
            human_corrections=human_corrections,
            confidence_score=0.82,
            review_time=180.0,
            notes="Good detection but missed pedestrian features"
        )
        
        print("✅ Successfully fed adaptive reviewer with sample feedback")
        
        # Train models
        print("🤖 Training models on feedback...")
        reviewer.train_models(min_examples=1)
        
        # Show statistics
        stats = reviewer.get_learning_statistics()
        print(f"\n📊 Learning Statistics:")
        print(f"   Total feedback: {stats['total_feedback']}")
        print(f"   Models trained: {stats['models_trained']}")
        print(f"   Average confidence: {stats['average_confidence']:.2f}")
        print(f"   Average review time: {stats['average_review_time']:.1f} seconds")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch scikit-learn joblib")
        return False
    except Exception as e:
        print(f"❌ Error feeding adaptive reviewer: {e}")
        return False

def main():
    """Main function to demonstrate feeding the ML system."""
    print("🚀 Feeding the ML System - Demonstration")
    print("=" * 50)
    
    # Create necessary directories
    Path("training_data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Feed foundation trainer
    foundation_success = feed_foundation_trainer()
    
    # Feed adaptive reviewer
    adaptive_success = feed_adaptive_reviewer()
    
    print("\n" + "=" * 50)
    if foundation_success and adaptive_success:
        print("🎉 Successfully fed both ML systems!")
        print("\nNext steps:")
        print("1. Add more real training data using the CLI tools:")
        print("   python src/cli/foundation_trainer.py add-as-built ...")
        print("   python src/cli/learning_manager.py collect-feedback ...")
        print("2. Train models with more data:")
        print("   python src/cli/foundation_trainer.py train-models --min-examples 20")
        print("   python src/cli/learning_manager.py train-models --min-examples 10")
        print("3. Monitor training progress:")
        print("   python src/cli/foundation_trainer.py show-statistics")
        print("   python src/cli/learning_manager.py show-statistics")
    else:
        print("⚠️  Some components failed. Check the error messages above.")
        print("Make sure all dependencies are installed:")
        print("pip install torch scikit-learn joblib opencv-python")

if __name__ == "__main__":
    # Import cv2 here to avoid import issues
    try:
        import cv2
        main()
    except ImportError:
        print("❌ OpenCV not installed. Install with: pip install opencv-python")
        print("Creating sample image without OpenCV...")
        # Create a simple numpy array instead
        def create_sample_image(width=1000, height=1000):
            return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        main()
