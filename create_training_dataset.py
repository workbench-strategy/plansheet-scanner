#!/usr/bin/env python3
"""
Create Training Dataset for ML System
Generates a comprehensive dataset to feed the machine learning system.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_image(plan_type="traffic_signal", width=1000, height=1000):
    """Create a sample image for training."""
    # Create a simple image with some patterns
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    if plan_type == "traffic_signal":
        # Signal heads (red circles)
        cv2.circle(image, (100, 200), 20, (0, 0, 255), -1)  # Red signal
        cv2.circle(image, (100, 220), 20, (0, 255, 0), -1)  # Green signal
        cv2.circle(image, (100, 240), 20, (255, 255, 0), -1)  # Yellow signal
        
        # Detector loops (rectangles)
        cv2.rectangle(image, (80, 180), (120, 190), (255, 255, 0), 2)  # Yellow loop
        cv2.rectangle(image, (80, 300), (120, 310), (255, 255, 0), 2)  # Yellow loop
        
        # Pedestrian button (blue circle)
        cv2.circle(image, (120, 240), 15, (255, 0, 0), -1)  # Blue button
        
    elif plan_type == "its":
        # ITS equipment (rectangles)
        cv2.rectangle(image, (100, 200), (150, 250), (0, 255, 255), -1)  # Cabinet
        cv2.rectangle(image, (200, 200), (220, 220), (255, 0, 255), -1)  # Camera
        cv2.rectangle(image, (300, 200), (320, 220), (128, 128, 128), -1)  # Detector
        
    elif plan_type == "mutcd_signing":
        # Signs (triangles and rectangles)
        cv2.rectangle(image, (100, 200), (200, 250), (255, 255, 0), -1)  # Warning sign
        cv2.rectangle(image, (250, 200), (350, 250), (255, 0, 0), -1)  # Stop sign
        cv2.rectangle(image, (400, 200), (500, 250), (0, 255, 0), -1)  # Guide sign
    
    return image

def generate_training_data():
    """Generate comprehensive training data."""
    print("📊 Generating Training Dataset...")
    
    # Create directories
    Path("training_data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Plan types and their characteristics
    plan_types = {
        "traffic_signal": {
            "element_types": ["signal_head_red", "signal_head_green", "signal_head_yellow", 
                            "detector_loop", "pedestrian_button", "pedestrian_signal"],
            "complexity_range": (0.3, 0.9),
            "budget_range": (200000, 800000)
        },
        "its": {
            "element_types": ["cctv_camera", "detector", "cabinet", "fiber_cable", "conduit"],
            "complexity_range": (0.4, 0.8),
            "budget_range": (100000, 500000)
        },
        "mutcd_signing": {
            "element_types": ["warning_sign", "regulatory_sign", "guide_sign", "post", "foundation"],
            "complexity_range": (0.2, 0.7),
            "budget_range": (50000, 300000)
        }
    }
    
    # Generate multiple training examples
    training_examples = []
    
    for i in range(25):  # Generate 25 training examples
        plan_type = random.choice(list(plan_types.keys()))
        plan_config = plan_types[plan_type]
        
        # Generate random elements
        num_elements = random.randint(3, 8)
        elements = []
        
        for j in range(num_elements):
            element_type = random.choice(plan_config["element_types"])
            location = [random.randint(50, 950), random.randint(50, 950)]
            confidence = random.uniform(0.7, 0.98)
            
            elements.append({
                "type": element_type,
                "location": location,
                "approved": random.choice([True, True, True, False]),  # 75% approved
                "confidence": confidence
            })
        
        # Generate project info
        complexity = random.uniform(*plan_config["complexity_range"])
        budget = random.randint(*plan_config["budget_range"])
        
        project_info = {
            "budget": budget,
            "duration_months": random.randint(3, 12),
            "complexity_score": complexity,
            "project_type": plan_type,
            "agency": random.choice(["WSDOT", "King County", "Seattle DOT", "Bellevue DOT"]),
            "region": random.choice(["Seattle", "Tacoma", "Bellevue", "Everett", "Spokane"]),
            "contractor": f"Contractor_{random.randint(1, 10)}",
            "engineer": f"Engineer_{random.randint(1, 5)}",
            "weather_conditions": random.choice(["dry", "wet", "snow"]),
            "traffic_volume": random.choice(["low", "medium", "high"]),
            "lanes": random.randint(2, 8)
        }
        
        # Generate construction notes
        construction_notes = f"Project completed successfully. {num_elements} elements installed. "
        if complexity > 0.7:
            construction_notes += "Complex project with multiple coordination requirements. "
        if budget > 500000:
            construction_notes += "High-budget project with extensive testing. "
        construction_notes += "All elements functioning per specifications."
        
        # Create training example
        example = {
            "plan_id": f"training_plan_{i:03d}",
            "plan_type": plan_type,
            "final_elements": elements,
            "construction_notes": construction_notes,
            "project_info": project_info,
            "approval_date": datetime.now() - timedelta(days=random.randint(1, 365)),
            "review_date": datetime.now() - timedelta(days=random.randint(1, 30))
        }
        
        training_examples.append(example)
    
    return training_examples

def feed_ml_system_with_dataset():
    """Feed the ML system with the generated dataset."""
    print("🤖 Feeding ML System with Generated Dataset...")
    
    try:
        from src.core.foundation_trainer import FoundationTrainer
        from src.core.adaptive_reviewer import AdaptiveReviewer
        
        # Generate training data
        training_examples = generate_training_data()
        
        # Initialize trainers
        foundation_trainer = FoundationTrainer("training_data", "models")
        adaptive_reviewer = AdaptiveReviewer("models")
        
        print(f"📊 Generated {len(training_examples)} training examples")
        
        # Feed foundation trainer
        print("🏗️ Feeding Foundation Trainer...")
        for i, example in enumerate(training_examples):
            # Create sample image
            sample_image = create_sample_image(example["plan_type"])
            
            # Add as-built data
            foundation_trainer.add_as_built_data(
                plan_id=example["plan_id"],
                plan_type=example["plan_type"],
                as_built_image=sample_image,
                final_elements=example["final_elements"],
                construction_notes=example["construction_notes"],
                approval_date=example["approval_date"],
                project_info=example["project_info"]
            )
            
            # Add review milestone
            approved_elements = [e for e in example["final_elements"] if e["approved"]]
            rejected_elements = [e for e in example["final_elements"] if not e["approved"]]
            
            foundation_trainer.add_review_milestone(
                plan_id=example["plan_id"],
                milestone="final",
                reviewer_comments=[
                    f"Plan type: {example['plan_type']}",
                    f"Complexity: {example['project_info']['complexity_score']:.2f}",
                    f"Budget: ${example['project_info']['budget']:,}",
                    "Review completed successfully"
                ],
                approved_elements=approved_elements,
                rejected_elements=rejected_elements,
                compliance_score=random.uniform(0.8, 0.98),
                review_date=example["review_date"]
            )
            
            if (i + 1) % 5 == 0:
                print(f"   Processed {i + 1}/{len(training_examples)} examples")
        
        # Feed adaptive reviewer with feedback
        print("🧠 Feeding Adaptive Reviewer...")
        for i, example in enumerate(training_examples):
            # Create original prediction (simulate system prediction)
            original_prediction = {
                "elements": [
                    {
                        "type": element["type"],
                        "location": element["location"],
                        "confidence": element["confidence"] * random.uniform(0.8, 1.2)
                    }
                    for element in example["final_elements"]
                ],
                "overall_confidence": random.uniform(0.7, 0.9)
            }
            
            # Create human corrections
            human_corrections = {
                "approved_elements": [e for e in example["final_elements"] if e["approved"]],
                "rejected_elements": [e for e in example["final_elements"] if not e["approved"]],
                "notes": f"Review of {example['plan_type']} plan. {len(example['final_elements'])} elements evaluated."
            }
            
            # Record feedback
            adaptive_reviewer.record_feedback(
                plan_id=example["plan_id"],
                plan_type=example["plan_type"],
                reviewer_id=f"reviewer_{random.randint(1, 5)}",
                original_prediction=original_prediction,
                human_corrections=human_corrections,
                confidence_score=original_prediction["overall_confidence"],
                review_time=random.uniform(60, 300),  # 1-5 minutes
                notes=human_corrections["notes"]
            )
            
            if (i + 1) % 5 == 0:
                print(f"   Processed {i + 1}/{len(training_examples)} feedback examples")
        
        # Train models
        print("🤖 Training Foundation Models...")
        foundation_trainer.extract_training_features()
        foundation_trainer.train_foundation_models(min_examples=20)
        
        print("🤖 Training Adaptive Models...")
        adaptive_reviewer.train_models(min_examples=20)
        
        # Show statistics
        print("\n📊 Training Results:")
        
        foundation_stats = foundation_trainer.get_training_statistics()
        print(f"🏗️ Foundation Trainer:")
        print(f"   Total as-built records: {foundation_stats['total_as_built']}")
        print(f"   Total milestone records: {foundation_stats['total_milestones']}")
        print(f"   Total training examples: {foundation_stats['total_training_examples']}")
        print(f"   Models trained: {foundation_stats['models_trained']}")
        
        adaptive_stats = adaptive_reviewer.get_learning_statistics()
        print(f"🧠 Adaptive Reviewer:")
        print(f"   Total feedback: {adaptive_stats['total_feedback']}")
        print(f"   Models trained: {adaptive_stats['models_trained']}")
        print(f"   Average confidence: {adaptive_stats['average_confidence']:.2f}")
        print(f"   Average review time: {adaptive_stats['average_review_time']:.1f} seconds")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch scikit-learn joblib opencv-python")
        return False
    except Exception as e:
        print(f"❌ Error feeding ML system: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Creating Training Dataset for ML System")
    print("=" * 60)
    
    success = feed_ml_system_with_dataset()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Successfully created and fed training dataset!")
        print("\nNext steps:")
        print("1. Monitor training progress:")
        print("   python src/cli/foundation_trainer.py show-statistics")
        print("   python src/cli/learning_manager.py show-statistics")
        print("2. Add real training data:")
        print("   python src/cli/foundation_trainer.py add-as-built ...")
        print("   python src/cli/learning_manager.py collect-feedback ...")
        print("3. Make predictions:")
        print("   python src/cli/foundation_trainer.py predict ...")
        print("4. Export training data:")
        print("   python src/cli/foundation_trainer.py export-data --format json --output training_data")
    else:
        print("⚠️  Failed to create training dataset. Check error messages above.")

if __name__ == "__main__":
    # Import cv2 here to avoid import issues
    try:
        import cv2
        main()
    except ImportError:
        print("❌ OpenCV not installed. Install with: pip install opencv-python")
        print("Creating sample images without OpenCV...")
        # Create a simple numpy array instead
        def create_sample_image(plan_type="traffic_signal", width=1000, height=1000):
            return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        main()
