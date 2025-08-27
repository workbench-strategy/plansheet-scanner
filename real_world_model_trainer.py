#!/usr/bin/env python3
"""
Real-World Model Trainer for As-Built Drawings
Handles thousands of real as-built drawings for practical AI engineer training.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
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
class AsBuiltData:
    """Simplified as-built data structure for real-world training."""
    drawing_id: str
    sheet_number: str
    sheet_title: str
    discipline: str
    notes: str
    code_references: List[str]
    review_comments: List[str]
    approval_status: str
    file_path: str

class RealWorldModelTrainer:
    """Trainer for real-world as-built data."""
    
    def __init__(self, data_dir: str = "real_world_data", model_dir: str = "trained_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.as_built_data = []
        
        # Models
        self.discipline_classifier = None
        self.violation_detector = None
        self.approval_predictor = None
        self.feature_scaler = None
        self.label_encoder = None
        self.text_vectorizer = None
        
        # Load existing data
        self._load_data()
    
    def add_as_built_drawing(self, drawing_data: AsBuiltData):
        """Add real as-built drawing data."""
        self.as_built_data.append(drawing_data)
        self._save_drawing_data(drawing_data)
        print(f"✅ Added: {drawing_data.sheet_number} - {drawing_data.discipline}")
    
    def train_all_models(self, min_examples: int = 100):
        """Train all models with real data."""
        if len(self.as_built_data) < min_examples:
            print(f"Need at least {min_examples} examples, have {len(self.as_built_data)}")
            return
        
        print(f"🤖 Training models on {len(self.as_built_data)} real examples...")
        
        # Prepare features
        X = self._extract_features()
        
        # Train discipline classifier
        self._train_discipline_model(X)
        
        # Train violation detector
        self._train_violation_model(X)
        
        # Train approval predictor
        self._train_approval_model(X)
        
        print("✅ All models trained and saved!")
    
    def _extract_features(self) -> np.ndarray:
        """Extract features from all drawings."""
        features = []
        
        for drawing in self.as_built_data:
            # Text features
            all_text = f"{drawing.sheet_title} {drawing.notes} {' '.join(drawing.review_comments)}"
            all_text = all_text.lower()
            
            # Basic features
            feature_vector = [
                len(all_text),  # Text length
                len(all_text.split()),  # Word count
                len(set(all_text.split())),  # Unique words
                len(drawing.code_references),  # Code references
                len(drawing.review_comments),  # Review comments
            ]
            
            # Discipline indicators
            for discipline in ["traffic", "electrical", "structural", "drainage", "mechanical"]:
                feature_vector.append(1 if discipline in drawing.discipline.lower() else 0)
            
            # Approval status
            for status in ["approved", "rejected", "conditional"]:
                feature_vector.append(1 if status in drawing.approval_status.lower() else 0)
            
            # Issue indicators
            feature_vector.extend([
                1 if "error" in all_text else 0,
                1 if "violation" in all_text else 0,
                1 if "problem" in all_text else 0,
                1 if "issue" in all_text else 0,
                1 if "correct" in all_text else 0,
                1 if "fix" in all_text else 0
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _train_discipline_model(self, X: np.ndarray):
        """Train discipline classification model."""
        print("   Training discipline classifier...")
        
        # Prepare labels
        disciplines = [d.discipline for d in self.as_built_data]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(disciplines)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"     Discipline classification - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Save model
        self.discipline_classifier = model
        self.label_encoder = label_encoder
        self.feature_scaler = scaler
        
        joblib.dump(model, self.model_dir / "discipline_classifier.joblib")
        joblib.dump(label_encoder, self.model_dir / "discipline_encoder.joblib")
        joblib.dump(scaler, self.model_dir / "feature_scaler.joblib")
    
    def _train_violation_model(self, X: np.ndarray):
        """Train violation detection model."""
        print("   Training violation detector...")
        
        # Create violation labels based on review comments
        y = np.array([1 if any("violation" in comment.lower() or "error" in comment.lower() 
                              for comment in d.review_comments) else 0 
                     for d in self.as_built_data])
        
        if sum(y) > 0 and sum(y) < len(y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            print(f"     Violation detection - Train: {train_score:.3f}, Test: {test_score:.3f}")
            
            self.violation_detector = model
            joblib.dump(model, self.model_dir / "violation_detector.joblib")
    
    def _train_approval_model(self, X: np.ndarray):
        """Train approval prediction model."""
        print("   Training approval predictor...")
        
        # Create approval labels
        approval_statuses = [d.approval_status for d in self.as_built_data]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(approval_statuses)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"     Approval prediction - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        self.approval_predictor = model
        joblib.dump(model, self.model_dir / "approval_predictor.joblib")
        joblib.dump(label_encoder, self.model_dir / "approval_encoder.joblib")
    
    def predict_drawing(self, drawing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Predict properties of a new drawing."""
        if self.discipline_classifier is None:
            return {"status": "models_not_trained"}
        
        # Extract features
        features = self._extract_single_features(drawing_info)
        X = np.array([features])
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Make predictions
        discipline_pred = self.discipline_classifier.predict(X_scaled)[0]
        predicted_discipline = self.label_encoder.inverse_transform([discipline_pred])[0]
        discipline_confidence = self.discipline_classifier.predict_proba(X_scaled)[0].max()
        
        violation_pred = False
        violation_confidence = 0.0
        if self.violation_detector:
            violation_pred = self.violation_detector.predict(X_scaled)[0]
            violation_confidence = self.violation_detector.predict_proba(X_scaled)[0].max()
        
        approval_pred = "unknown"
        approval_confidence = 0.0
        if self.approval_predictor:
            approval_pred_encoded = self.approval_predictor.predict(X_scaled)[0]
            approval_pred = self.approval_predictor.label_encoder.inverse_transform([approval_pred_encoded])[0]
            approval_confidence = self.approval_predictor.predict_proba(X_scaled)[0].max()
        
        return {
            "predicted_discipline": predicted_discipline,
            "discipline_confidence": discipline_confidence,
            "has_violations": bool(violation_pred),
            "violation_confidence": violation_confidence,
            "predicted_approval": approval_pred,
            "approval_confidence": approval_confidence,
            "recommendations": self._generate_recommendations(predicted_discipline, violation_pred)
        }
    
    def _extract_single_features(self, drawing_info: Dict[str, Any]) -> List[float]:
        """Extract features from single drawing info."""
        text = f"{drawing_info.get('sheet_title', '')} {drawing_info.get('notes', '')}".lower()
        
        features = [
            len(text),
            len(text.split()),
            len(set(text.split())),
            len(drawing_info.get('code_references', [])),
            len(drawing_info.get('review_comments', []))
        ]
        
        # Discipline indicators
        for discipline in ["traffic", "electrical", "structural", "drainage", "mechanical"]:
            features.append(1 if discipline in drawing_info.get('discipline', '').lower() else 0)
        
        # Approval status
        for status in ["approved", "rejected", "conditional"]:
            features.append(1 if status in drawing_info.get('approval_status', '').lower() else 0)
        
        # Issue indicators
        features.extend([
            1 if "error" in text else 0,
            1 if "violation" in text else 0,
            1 if "problem" in text else 0,
            1 if "issue" in text else 0,
            1 if "correct" in text else 0,
            1 if "fix" in text else 0
        ])
        
        return features
    
    def _generate_recommendations(self, discipline: str, has_violations: bool) -> List[str]:
        """Generate review recommendations."""
        recommendations = []
        
        if has_violations:
            recommendations.append("Review for code compliance violations")
        
        if discipline == "traffic":
            recommendations.append("Verify MUTCD compliance")
            recommendations.append("Check signal timing and coordination")
        elif discipline == "electrical":
            recommendations.append("Verify NEC compliance")
            recommendations.append("Check conduit sizing and grounding")
        elif discipline == "structural":
            recommendations.append("Verify AASHTO compliance")
            recommendations.append("Check load ratings and reinforcement")
        elif discipline == "drainage":
            recommendations.append("Verify drainage capacity")
            recommendations.append("Check pipe slopes and erosion control")
        
        return recommendations
    
    def _save_drawing_data(self, drawing_data: AsBuiltData):
        """Save drawing data to disk."""
        data_file = self.data_dir / f"drawing_{drawing_data.drawing_id}.json"
        
        with open(data_file, 'w') as f:
            json.dump({
                "drawing_id": drawing_data.drawing_id,
                "sheet_number": drawing_data.sheet_number,
                "sheet_title": drawing_data.sheet_title,
                "discipline": drawing_data.discipline,
                "notes": drawing_data.notes,
                "code_references": drawing_data.code_references,
                "review_comments": drawing_data.review_comments,
                "approval_status": drawing_data.approval_status,
                "file_path": drawing_data.file_path,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    def _load_data(self):
        """Load existing drawing data."""
        data_files = list(self.data_dir.glob("drawing_*.json"))
        
        for data_file in data_files:
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                drawing_data = AsBuiltData(
                    drawing_id=data["drawing_id"],
                    sheet_number=data["sheet_number"],
                    sheet_title=data["sheet_title"],
                    discipline=data["discipline"],
                    notes=data["notes"],
                    code_references=data["code_references"],
                    review_comments=data["review_comments"],
                    approval_status=data["approval_status"],
                    file_path=data["file_path"]
                )
                
                self.as_built_data.append(drawing_data)
            except Exception as e:
                print(f"Warning: Could not load {data_file}: {e}")
        
        print(f"✅ Loaded {len(self.as_built_data)} drawing examples")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.as_built_data:
            return {"total_drawings": 0}
        
        disciplines = [d.discipline for d in self.as_built_data]
        discipline_counts = {}
        for discipline in disciplines:
            discipline_counts[discipline] = discipline_counts.get(discipline, 0) + 1
        
        approval_statuses = [d.approval_status for d in self.as_built_data]
        status_counts = {}
        for status in approval_statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        violation_count = sum(1 for d in self.as_built_data 
                            if any("violation" in comment.lower() for comment in d.review_comments))
        
        return {
            "total_drawings": len(self.as_built_data),
            "discipline_distribution": discipline_counts,
            "approval_status_distribution": status_counts,
            "total_violations": violation_count,
            "models_trained": self.discipline_classifier is not None
        }

def main():
    """Main function to demonstrate real-world model training."""
    print("🏗️ Real-World Model Trainer for As-Built Drawings")
    print("=" * 60)
    
    # Initialize trainer
    trainer = RealWorldModelTrainer()
    
    # Add some sample real-world data
    sample_data = [
        AsBuiltData("001", "T-001", "Traffic Signal Plan", "traffic", 
                   "Signal heads and detector loops shown", ["MUTCD", "WSDOT Standards"],
                   ["Design meets requirements", "No violations found"], "approved", "traffic_001.pdf"),
        AsBuiltData("002", "E-001", "Electrical Plan", "electrical",
                   "Power distribution and conduit routing", ["NEC", "WSDOT Electrical"],
                   ["Conduit sizing adequate", "Grounding system proper"], "approved", "electrical_001.pdf"),
        AsBuiltData("003", "S-001", "Structural Plan", "structural",
                   "Bridge structure and reinforcement", ["AASHTO", "WSDOT Structural"],
                   ["Load ratings meet requirements", "Reinforcement adequate"], "approved", "structural_001.pdf"),
        AsBuiltData("004", "T-002", "Traffic Signal Plan", "traffic",
                   "Signal timing and coordination", ["MUTCD", "WSDOT Standards"],
                   ["Timing coordination issue found", "Violation: inadequate detector coverage"], "conditional", "traffic_002.pdf"),
        AsBuiltData("005", "E-002", "Electrical Plan", "electrical",
                   "Electrical equipment installation", ["NEC", "WSDOT Electrical"],
                   ["Undersized conduit detected", "Violation: conduit too small"], "rejected", "electrical_002.pdf")
    ]
    
    for data in sample_data:
        trainer.add_as_built_drawing(data)
    
    # Train models
    trainer.train_all_models(min_examples=5)
    
    # Test predictions
    test_drawings = [
        {
            "sheet_number": "T-003",
            "sheet_title": "Traffic Signal Plan",
            "discipline": "traffic",
            "notes": "Signal heads and pedestrian signals shown",
            "code_references": ["MUTCD"],
            "review_comments": ["Design review completed"],
            "approval_status": "pending"
        },
        {
            "sheet_number": "E-003",
            "sheet_title": "Electrical Plan",
            "discipline": "electrical",
            "notes": "Power distribution system",
            "code_references": ["NEC"],
            "review_comments": ["Electrical review in progress"],
            "approval_status": "pending"
        }
    ]
    
    print(f"\n🧪 Test Predictions:")
    for i, test_drawing in enumerate(test_drawings, 1):
        prediction = trainer.predict_drawing(test_drawing)
        print(f"\n   Test {i}: {test_drawing['sheet_number']} - {test_drawing['sheet_title']}")
        print(f"     Predicted Discipline: {prediction['predicted_discipline']} (confidence: {prediction['discipline_confidence']:.3f})")
        print(f"     Has Violations: {prediction['has_violations']} (confidence: {prediction['violation_confidence']:.3f})")
        print(f"     Predicted Approval: {prediction['predicted_approval']} (confidence: {prediction['approval_confidence']:.3f})")
        print(f"     Recommendations:")
        for rec in prediction['recommendations']:
            print(f"       - {rec}")
    
    # Show statistics
    stats = trainer.get_statistics()
    print(f"\n📊 Training Statistics:")
    print(f"   Total drawings: {stats['total_drawings']}")
    print(f"   Models trained: {stats['models_trained']}")
    
    if stats['discipline_distribution']:
        print(f"   Discipline distribution:")
        for discipline, count in stats['discipline_distribution'].items():
            print(f"     {discipline}: {count}")

if __name__ == "__main__":
    main()
