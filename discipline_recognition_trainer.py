#!/usr/bin/env python3
"""
Discipline Recognition Trainer for Highway Project Drawings
Trains ML models to automatically categorize drawings by discipline based on:
- Drawing indexes and sheet numbers
- Notes and annotations
- Drawing content and symbols
- Project context
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
except ImportError:
    print("Warning: ML dependencies not installed. Install with: pip install scikit-learn joblib")

@dataclass
class DrawingInfo:
    """Information about a drawing for discipline recognition."""
    sheet_number: str
    sheet_title: str
    discipline_code: str
    notes: str
    symbols: List[str]
    project_type: str
    agency: str
    drawing_type: str  # plan, profile, detail, etc.
    file_path: str
    confidence: float = 1.0

@dataclass
class DisciplineTrainingExample:
    """Training example for discipline recognition."""
    features: Dict[str, Any]
    discipline: str
    confidence: float
    metadata: Dict[str, Any]

class DisciplineRecognitionTrainer:
    """Trains models to recognize disciplines in highway project drawings."""
    
    def __init__(self, data_dir: str = "discipline_training_data", model_dir: str = "discipline_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.training_examples = []
        self.discipline_mappings = {}
        
        # Models
        self.discipline_classifier = None
        self.text_vectorizer = None
        self.feature_scaler = None
        self.label_encoder = None
        
        # Discipline definitions for highway projects
        self.discipline_definitions = {
            "traffic": {
                "keywords": ["traffic", "signal", "sign", "marking", "detector", "loop", "pedestrian", "crosswalk"],
                "sheet_patterns": ["T-", "TS-", "TM-", "TP-"],
                "symbols": ["signal_head", "detector_loop", "sign_post", "crosswalk", "stop_bar"]
            },
            "electrical": {
                "keywords": ["electrical", "power", "conduit", "cable", "junction_box", "transformer", "voltage"],
                "sheet_patterns": ["E-", "EL-", "EP-"],
                "symbols": ["conduit", "junction_box", "transformer", "power_pole", "electrical_cable"]
            },
            "civil": {
                "keywords": ["drainage", "storm", "culvert", "inlet", "manhole", "pipe", "grading", "earthwork"],
                "sheet_patterns": ["C-", "CV-", "D-", "G-"],
                "symbols": ["catch_basin", "manhole", "culvert", "drainage_pipe", "grade_line"]
            },
            "structural": {
                "keywords": ["bridge", "beam", "column", "foundation", "reinforcement", "concrete", "steel"],
                "sheet_patterns": ["S-", "ST-", "B-"],
                "symbols": ["bridge_girder", "column", "foundation", "reinforcement", "expansion_joint"]
            },
            "mechanical": {
                "keywords": ["mechanical", "ventilation", "heating", "cooling", "duct", "equipment"],
                "sheet_patterns": ["M-", "HVAC-"],
                "symbols": ["duct", "air_handler", "ventilation_fan", "mechanical_equipment"]
            },
            "landscape": {
                "keywords": ["landscape", "irrigation", "planting", "tree", "shrub", "grass", "mulch"],
                "sheet_patterns": ["L-", "LS-"],
                "symbols": ["tree", "shrub", "irrigation_head", "landscape_area"]
            },
            "utilities": {
                "keywords": ["water", "sewer", "gas", "telecommunications", "fiber", "utility", "underground"],
                "sheet_patterns": ["U-", "UT-", "W-", "S-"],
                "symbols": ["water_line", "sewer_line", "gas_line", "fiber_cable", "utility_mark"]
            },
            "survey": {
                "keywords": ["survey", "control", "benchmark", "monument", "coordinate", "elevation"],
                "sheet_patterns": ["SV-", "SUR-"],
                "symbols": ["benchmark", "control_point", "monument", "survey_point"]
            }
        }
        
        # Load existing data
        self._load_training_data()
    
    def add_drawing_example(self, drawing_info: DrawingInfo):
        """Add a drawing example for training."""
        # Extract features
        features = self._extract_features(drawing_info)
        
        # Create training example
        example = DisciplineTrainingExample(
            features=features,
            discipline=drawing_info.discipline_code,
            confidence=drawing_info.confidence,
            metadata={
                "sheet_number": drawing_info.sheet_number,
                "sheet_title": drawing_info.sheet_title,
                "project_type": drawing_info.project_type,
                "agency": drawing_info.agency,
                "drawing_type": drawing_info.drawing_type
            }
        )
        
        self.training_examples.append(example)
        self._save_training_example(example)
        
        print(f"✅ Added drawing example: {drawing_info.sheet_number} - {drawing_info.discipline_code}")
    
    def _extract_features(self, drawing_info: DrawingInfo) -> Dict[str, Any]:
        """Extract features from drawing information."""
        features = {}
        
        # Text features
        all_text = f"{drawing_info.sheet_title} {drawing_info.notes}".lower()
        
        # Keyword matching for each discipline
        for discipline, definition in self.discipline_definitions.items():
            keyword_count = 0
            for keyword in definition["keywords"]:
                keyword_count += len(re.findall(rf'\b{keyword}\b', all_text))
            features[f"{discipline}_keyword_count"] = keyword_count
        
        # Sheet number pattern matching
        sheet_num = drawing_info.sheet_number.upper()
        for discipline, definition in self.discipline_definitions.items():
            pattern_match = 0
            for pattern in definition["sheet_patterns"]:
                if pattern in sheet_num:
                    pattern_match = 1
                    break
            features[f"{discipline}_pattern_match"] = pattern_match
        
        # Symbol matching
        symbols_text = " ".join(drawing_info.symbols).lower()
        for discipline, definition in self.discipline_definitions.items():
            symbol_count = 0
            for symbol in definition["symbols"]:
                symbol_count += len(re.findall(rf'\b{symbol}\b', symbols_text))
            features[f"{discipline}_symbol_count"] = symbol_count
        
        # Text-based features
        features["text_length"] = len(all_text)
        features["word_count"] = len(all_text.split())
        features["unique_words"] = len(set(all_text.split()))
        
        # Drawing type features
        drawing_type = drawing_info.drawing_type.lower()
        features["is_plan"] = 1 if "plan" in drawing_type else 0
        features["is_profile"] = 1 if "profile" in drawing_type else 0
        features["is_detail"] = 1 if "detail" in drawing_type else 0
        features["is_section"] = 1 if "section" in drawing_type else 0
        
        # Project context features
        project_type = drawing_info.project_type.lower()
        features["is_intersection"] = 1 if "intersection" in project_type else 0
        features["is_bridge"] = 1 if "bridge" in project_type else 0
        features["is_highway"] = 1 if "highway" in project_type else 0
        features["is_roadway"] = 1 if "roadway" in project_type else 0
        
        return features
    
    def train_discipline_model(self, min_examples: int = 10):
        """Train the discipline recognition model."""
        if len(self.training_examples) < min_examples:
            print(f"Need at least {min_examples} examples, have {len(self.training_examples)}")
            return
        
        print(f"🤖 Training discipline recognition model on {len(self.training_examples)} examples...")
        
        # Prepare data
        X = []
        y = []
        
        for example in self.training_examples:
            X.append(list(example.features.values()))
            y.append(example.discipline)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train_encoded)
        test_score = model.score(X_test_scaled, y_test_encoded)
        
        print(f"✅ Model trained successfully!")
        print(f"   Train accuracy: {train_score:.3f}")
        print(f"   Test accuracy: {test_score:.3f}")
        
        # Save model
        self.discipline_classifier = model
        self.feature_scaler = scaler
        self.label_encoder = label_encoder
        
        self._save_model(model, scaler, label_encoder)
        
        # Generate classification report
        y_pred = model.predict(X_test_scaled)
        print("\n📊 Classification Report:")
        print(classification_report(y_test, label_encoder.inverse_transform(y_pred)))
    
    def predict_discipline(self, drawing_info: DrawingInfo) -> Dict[str, Any]:
        """Predict discipline for a drawing."""
        if self.discipline_classifier is None:
            print("❌ Model not trained yet. Train the model first.")
            return {"discipline": "unknown", "confidence": 0.0}
        
        # Extract features
        features = self._extract_features(drawing_info)
        X = np.array([list(features.values())])
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Predict
        prediction_encoded = self.discipline_classifier.predict(X_scaled)[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence
        confidence = self.discipline_classifier.predict_proba(X_scaled)[0].max()
        
        return {
            "discipline": prediction,
            "confidence": confidence,
            "features": features
        }
    
    def generate_sample_data(self, num_examples: int = 50):
        """Generate sample training data for testing."""
        print(f"📊 Generating {num_examples} sample training examples...")
        
        # Sample data for different disciplines
        sample_data = [
            # Traffic examples
            DrawingInfo("T-001", "Traffic Signal Plan", "traffic", "Signal heads and detector loops shown", 
                       ["signal_head", "detector_loop"], "intersection", "WSDOT", "plan"),
            DrawingInfo("TS-002", "Traffic Signing Plan", "traffic", "Regulatory and warning signs", 
                       ["sign_post", "warning_sign"], "highway", "WSDOT", "plan"),
            
            # Electrical examples
            DrawingInfo("E-001", "Electrical Plan", "electrical", "Power distribution and conduit routing", 
                       ["conduit", "junction_box"], "intersection", "WSDOT", "plan"),
            DrawingInfo("EL-002", "Electrical Details", "electrical", "Electrical equipment and connections", 
                       ["transformer", "power_pole"], "highway", "WSDOT", "detail"),
            
            # Civil examples
            DrawingInfo("C-001", "Drainage Plan", "civil", "Storm drainage system and inlets", 
                       ["catch_basin", "drainage_pipe"], "highway", "WSDOT", "plan"),
            DrawingInfo("G-002", "Grading Plan", "civil", "Earthwork and grading details", 
                       ["grade_line", "cut_fill"], "intersection", "WSDOT", "plan"),
            
            # Structural examples
            DrawingInfo("S-001", "Bridge Plan", "structural", "Bridge structure and reinforcement", 
                       ["bridge_girder", "reinforcement"], "bridge", "WSDOT", "plan"),
            DrawingInfo("ST-002", "Structural Details", "structural", "Concrete and steel details", 
                       ["column", "foundation"], "bridge", "WSDOT", "detail"),
            
            # Utilities examples
            DrawingInfo("U-001", "Utility Plan", "utilities", "Underground utilities and conflicts", 
                       ["water_line", "sewer_line"], "highway", "WSDOT", "plan"),
            DrawingInfo("UT-002", "Telecommunications", "utilities", "Fiber optic and communication lines", 
                       ["fiber_cable", "utility_mark"], "intersection", "WSDOT", "plan"),
        ]
        
        # Generate variations
        for i in range(num_examples):
            base_example = sample_data[i % len(sample_data)]
            
            # Create variation
            variation = DrawingInfo(
                sheet_number=f"{base_example.sheet_number}-{i+1:03d}",
                sheet_title=f"{base_example.sheet_title} - Variation {i+1}",
                discipline_code=base_example.discipline_code,
                notes=f"{base_example.notes} (variation {i+1})",
                symbols=base_example.symbols,
                project_type=base_example.project_type,
                agency=base_example.agency,
                drawing_type=base_example.drawing_type,
                file_path=f"sample_drawing_{i+1:03d}.pdf"
            )
            
            self.add_drawing_example(variation)
        
        print(f"✅ Generated {num_examples} sample training examples")
    
    def _save_training_example(self, example: DisciplineTrainingExample):
        """Save training example to disk."""
        example_file = self.data_dir / f"example_{len(self.training_examples):04d}.json"
        
        with open(example_file, 'w') as f:
            json.dump({
                "features": example.features,
                "discipline": example.discipline,
                "confidence": example.confidence,
                "metadata": example.metadata,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    def _save_model(self, model, scaler, label_encoder):
        """Save trained model to disk."""
        model_file = self.model_dir / "discipline_classifier.joblib"
        scaler_file = self.model_dir / "feature_scaler.joblib"
        encoder_file = self.model_dir / "label_encoder.joblib"
        
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)
        joblib.dump(label_encoder, encoder_file)
        
        print(f"✅ Model saved to {self.model_dir}")
    
    def _load_training_data(self):
        """Load existing training data."""
        example_files = list(self.data_dir.glob("example_*.json"))
        
        for example_file in example_files:
            try:
                with open(example_file, 'r') as f:
                    data = json.load(f)
                
                example = DisciplineTrainingExample(
                    features=data["features"],
                    discipline=data["discipline"],
                    confidence=data["confidence"],
                    metadata=data["metadata"]
                )
                
                self.training_examples.append(example)
            except Exception as e:
                print(f"Warning: Could not load {example_file}: {e}")
        
        print(f"✅ Loaded {len(self.training_examples)} training examples")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about the training data."""
        if not self.training_examples:
            return {"total_examples": 0}
        
        disciplines = [ex.discipline for ex in self.training_examples]
        discipline_counts = {}
        for discipline in disciplines:
            discipline_counts[discipline] = discipline_counts.get(discipline, 0) + 1
        
        return {
            "total_examples": len(self.training_examples),
            "discipline_distribution": discipline_counts,
            "unique_disciplines": len(set(disciplines)),
            "model_trained": self.discipline_classifier is not None
        }

def main():
    """Main function to demonstrate discipline recognition training."""
    print("🏗️ Discipline Recognition Trainer for Highway Projects")
    print("=" * 60)
    
    # Initialize trainer
    trainer = DisciplineRecognitionTrainer()
    
    # Generate sample data
    trainer.generate_sample_data(num_examples=50)
    
    # Train model
    trainer.train_discipline_model(min_examples=20)
    
    # Test prediction
    test_drawing = DrawingInfo(
        "T-003", "Traffic Signal Timing", "traffic",
        "Signal timing and coordination plan with detector loops",
        ["signal_head", "detector_loop", "timing_plan"],
        "intersection", "WSDOT", "plan"
    )
    
    prediction = trainer.predict_discipline(test_drawing)
    print(f"\n🧪 Test Prediction:")
    print(f"   Drawing: {test_drawing.sheet_number} - {test_drawing.sheet_title}")
    print(f"   Predicted Discipline: {prediction['discipline']}")
    print(f"   Confidence: {prediction['confidence']:.3f}")
    
    # Show statistics
    stats = trainer.get_training_statistics()
    print(f"\n📊 Training Statistics:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Unique disciplines: {stats['unique_disciplines']}")
    print(f"   Model trained: {stats['model_trained']}")
    
    if stats['discipline_distribution']:
        print(f"   Discipline distribution:")
        for discipline, count in stats['discipline_distribution'].items():
            print(f"     {discipline}: {count}")

if __name__ == "__main__":
    main()
