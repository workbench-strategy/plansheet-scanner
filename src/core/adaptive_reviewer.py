"""
Adaptive Traffic Plan Reviewer
Learns from feedback and improves over time using machine learning.
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# ML imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import joblib
except ImportError:
    print("Warning: ML dependencies not installed. Install with: pip install torch scikit-learn joblib")

@dataclass
class ReviewFeedback:
    """Stores feedback from human reviewers."""
    plan_id: str
    plan_type: str
    reviewer_id: str
    timestamp: datetime
    original_prediction: Dict[str, Any]
    human_corrections: Dict[str, Any]
    confidence_score: float
    review_time: float  # seconds
    notes: str = ""

@dataclass
class LearningExample:
    """Training example for the learning system."""
    plan_features: np.ndarray
    human_labels: np.ndarray
    predicted_labels: np.ndarray
    confidence_scores: np.ndarray
    plan_metadata: Dict[str, Any]

class PlanFeatureExtractor:
    """Extracts features from plan images for machine learning."""
    
    def __init__(self):
        self.feature_dim = 512  # Adjust based on your needs
        
    def extract_features(self, plan_image: np.ndarray, detected_elements: List[Any]) -> np.ndarray:
        """Extract numerical features from plan image and detected elements."""
        features = []
        
        # Basic image features
        features.extend(self._extract_image_features(plan_image))
        
        # Element-based features
        features.extend(self._extract_element_features(detected_elements))
        
        # Spatial relationship features
        features.extend(self._extract_spatial_features(detected_elements))
        
        # Compliance pattern features
        features.extend(self._extract_compliance_features(detected_elements))
        
        # Pad or truncate to fixed length
        features = self._normalize_features(features)
        
        return np.array(features)
    
    def _extract_image_features(self, image: np.ndarray) -> List[float]:
        """Extract basic image features."""
        features = []
        
        # Image statistics
        features.append(image.shape[0])  # height
        features.append(image.shape[1])  # width
        features.append(image.shape[2] if len(image.shape) > 2 else 1)  # channels
        
        # Color statistics
        if len(image.shape) > 2:
            for channel in range(min(3, image.shape[2])):
                features.append(np.mean(image[:, :, channel]))
                features.append(np.std(image[:, :, channel]))
        else:
            features.append(np.mean(image))
            features.append(np.std(image))
        
        # Edge density
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
        except:
            features.append(0.0)
        
        return features
    
    def _extract_element_features(self, elements: List[Any]) -> List[float]:
        """Extract features from detected elements."""
        features = []
        
        # Element counts by type
        element_types = {}
        for element in elements:
            element_type = element.element_type.split('_')[0]  # Get base type
            element_types[element_type] = element_types.get(element_type, 0) + 1
        
        # Count features for common element types
        for element_type in ['signal', 'sign', 'detector', 'marking', 'camera', 'sensor']:
            features.append(element_types.get(element_type, 0))
        
        # Confidence statistics
        if elements:
            confidences = [e.confidence for e in elements]
            features.append(np.mean(confidences))
            features.append(np.std(confidences))
            features.append(np.min(confidences))
            features.append(np.max(confidences))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Element size statistics
        if elements:
            areas = []
            for element in elements:
                if hasattr(element, 'metadata') and 'size' in element.metadata:
                    size = element.metadata['size']
                    if isinstance(size, (list, tuple)) and len(size) >= 2:
                        areas.append(size[0] * size[1])
            
            if areas:
                features.append(np.mean(areas))
                features.append(np.std(areas))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _extract_spatial_features(self, elements: List[Any]) -> List[float]:
        """Extract spatial relationship features."""
        features = []
        
        if len(elements) < 2:
            features.extend([0.0] * 10)  # Pad with zeros
            return features
        
        # Extract locations
        locations = [(e.location[0], e.location[1]) for e in elements]
        
        # Distance statistics
        distances = []
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                dist = np.sqrt((locations[i][0] - locations[j][0])**2 + 
                             (locations[i][1] - locations[j][1])**2)
                distances.append(dist)
        
        if distances:
            features.append(np.mean(distances))
            features.append(np.std(distances))
            features.append(np.min(distances))
            features.append(np.max(distances))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Spatial distribution
        x_coords = [loc[0] for loc in locations]
        y_coords = [loc[1] for loc in locations]
        
        features.append(np.std(x_coords))  # Horizontal spread
        features.append(np.std(y_coords))  # Vertical spread
        
        # Clustering measure
        if len(distances) > 0:
            features.append(np.percentile(distances, 25))  # Q1
            features.append(np.percentile(distances, 75))  # Q3
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _extract_compliance_features(self, elements: List[Any]) -> List[float]:
        """Extract compliance-related features."""
        features = []
        
        # Element type diversity
        element_types = [e.element_type.split('_')[0] for e in elements]
        unique_types = len(set(element_types))
        features.append(unique_types)
        
        # Compliance indicators (simplified)
        compliance_indicators = 0
        for element in elements:
            # Check for common compliance patterns
            if 'signal' in element.element_type:
                compliance_indicators += 1
            if 'detector' in element.element_type:
                compliance_indicators += 1
            if 'pedestrian' in element.element_type:
                compliance_indicators += 1
        
        features.append(compliance_indicators)
        
        # Pad to ensure consistent feature vector length
        while len(features) < 50:  # Adjust based on your needs
            features.append(0.0)
        
        return features
    
    def _normalize_features(self, features: List[float]) -> List[float]:
        """Normalize features to fixed length."""
        # Pad or truncate to self.feature_dim
        if len(features) < self.feature_dim:
            features.extend([0.0] * (self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]
        
        return features

class AdaptiveReviewer:
    """Adaptive reviewer that learns from feedback."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.feature_extractor = PlanFeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Feedback storage
        self.feedback_file = self.model_dir / "review_feedback.jsonl"
        self.training_data_file = self.model_dir / "training_data.pkl"
        
        # Load existing models if available
        self._load_models()
    
    def review_plan(self, plan_image: np.ndarray, detected_elements: List[Any], 
                   plan_type: str) -> Dict[str, Any]:
        """Review plan with adaptive learning capabilities."""
        
        # Extract features
        features = self.feature_extractor.extract_features(plan_image, detected_elements)
        
        # Get predictions from all models
        predictions = {}
        confidence_scores = {}
        
        for model_name, model in self.models.items():
            if model_name in self.scalers:
                features_scaled = self.scalers[model_name].transform([features])
                prediction = model.predict(features_scaled)[0]
                confidence = model.predict_proba(features_scaled)[0].max()
                
                predictions[model_name] = prediction
                confidence_scores[model_name] = confidence
        
        # Combine predictions (ensemble approach)
        final_prediction = self._ensemble_predictions(predictions, confidence_scores)
        
        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'final_prediction': final_prediction,
            'features': features.tolist(),
            'plan_type': plan_type,
            'timestamp': datetime.now().isoformat()
        }
    
    def record_feedback(self, plan_id: str, plan_type: str, reviewer_id: str,
                       original_prediction: Dict[str, Any], human_corrections: Dict[str, Any],
                       confidence_score: float, review_time: float, notes: str = ""):
        """Record human feedback for learning."""
        
        feedback = ReviewFeedback(
            plan_id=plan_id,
            plan_type=plan_type,
            reviewer_id=reviewer_id,
            timestamp=datetime.now(),
            original_prediction=original_prediction,
            human_corrections=human_corrections,
            confidence_score=confidence_score,
            review_time=review_time,
            notes=notes
        )
        
        # Save feedback
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(asdict(feedback), default=str) + '\n')
        
        # Create training example
        self._create_training_example(feedback)
    
    def train_models(self, min_examples: int = 10):
        """Train models on accumulated feedback."""
        
        if not self.training_data_file.exists():
            print("No training data available yet.")
            return
        
        # Load training data
        with open(self.training_data_file, 'rb') as f:
            training_data = pickle.load(f)
        
        if len(training_data) < min_examples:
            print(f"Need at least {min_examples} examples, have {len(training_data)}")
            return
        
        # Prepare data
        X = np.array([ex.plan_features for ex in training_data])
        y = np.array([ex.human_labels for ex in training_data])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train different model types
        model_types = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        for model_name, model in model_types.items():
            print(f"Training {model_name}...")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            print(f"{model_name} - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
            
            # Save model
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # Save to disk
            self._save_model(model_name, model, scaler)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning system."""
        stats = {
            'total_feedback': 0,
            'models_trained': len(self.models),
            'feedback_by_plan_type': {},
            'average_confidence': 0.0,
            'average_review_time': 0.0
        }
        
        if not self.feedback_file.exists():
            return stats
        
        # Read feedback file
        feedback_data = []
        with open(self.feedback_file, 'r') as f:
            for line in f:
                if line.strip():
                    feedback_data.append(json.loads(line))
        
        if not feedback_data:
            return stats
        
        stats['total_feedback'] = len(feedback_data)
        
        # Calculate statistics
        plan_types = [f['plan_type'] for f in feedback_data]
        confidences = [f['confidence_score'] for f in feedback_data]
        review_times = [f['review_time'] for f in feedback_data]
        
        stats['feedback_by_plan_type'] = pd.Series(plan_types).value_counts().to_dict()
        stats['average_confidence'] = np.mean(confidences)
        stats['average_review_time'] = np.mean(review_times)
        
        return stats
    
    def _ensemble_predictions(self, predictions: Dict[str, Any], 
                            confidence_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions from multiple models."""
        
        if not predictions:
            return {}
        
        # Simple ensemble: weighted average based on confidence
        ensemble_prediction = {}
        
        for model_name, prediction in predictions.items():
            confidence = confidence_scores.get(model_name, 0.5)
            
            # Weight the prediction by confidence
            for key, value in prediction.items():
                if key not in ensemble_prediction:
                    ensemble_prediction[key] = []
                ensemble_prediction[key].append((value, confidence))
        
        # Calculate weighted average
        final_prediction = {}
        for key, weighted_values in ensemble_prediction.items():
            total_weight = sum(weight for _, weight in weighted_values)
            if total_weight > 0:
                weighted_sum = sum(value * weight for value, weight in weighted_values)
                final_prediction[key] = weighted_sum / total_weight
            else:
                final_prediction[key] = 0.0
        
        return final_prediction
    
    def _create_training_example(self, feedback: ReviewFeedback):
        """Create training example from feedback."""
        
        # Extract features from original prediction
        features = np.array(feedback.original_prediction.get('features', []))
        
        # Create labels from human corrections
        human_labels = self._extract_labels_from_corrections(feedback.human_corrections)
        
        # Create predicted labels from original prediction
        predicted_labels = self._extract_labels_from_prediction(feedback.original_prediction)
        
        # Confidence scores
        confidence_scores = np.array([feedback.confidence_score])
        
        # Create training example
        example = LearningExample(
            plan_features=features,
            human_labels=human_labels,
            predicted_labels=predicted_labels,
            confidence_scores=confidence_scores,
            plan_metadata={
                'plan_id': feedback.plan_id,
                'plan_type': feedback.plan_type,
                'reviewer_id': feedback.reviewer_id,
                'timestamp': feedback.timestamp.isoformat()
            }
        )
        
        # Save training example
        self._save_training_example(example)
    
    def _extract_labels_from_corrections(self, corrections: Dict[str, Any]) -> np.ndarray:
        """Extract numerical labels from human corrections."""
        # This is a simplified version - you'd need to adapt based on your specific needs
        labels = []
        
        # Extract compliance scores
        if 'compliance_score' in corrections:
            labels.append(corrections['compliance_score'])
        else:
            labels.append(0.5)  # Default
        
        # Extract issue counts
        if 'issues' in corrections:
            labels.append(len(corrections['issues']))
        else:
            labels.append(0)
        
        # Extract element counts
        if 'elements_found' in corrections:
            labels.append(len(corrections['elements_found']))
        else:
            labels.append(0)
        
        # Pad to consistent length
        while len(labels) < 10:  # Adjust based on your needs
            labels.append(0.0)
        
        return np.array(labels)
    
    def _extract_labels_from_prediction(self, prediction: Dict[str, Any]) -> np.ndarray:
        """Extract numerical labels from model prediction."""
        # Similar to human corrections extraction
        labels = []
        
        if 'final_prediction' in prediction:
            final_pred = prediction['final_prediction']
            labels.append(final_pred.get('compliance_score', 0.5))
            labels.append(final_pred.get('issue_count', 0))
            labels.append(final_pred.get('element_count', 0))
        else:
            labels.extend([0.5, 0, 0])
        
        # Pad to consistent length
        while len(labels) < 10:
            labels.append(0.0)
        
        return np.array(labels)
    
    def _save_training_example(self, example: LearningExample):
        """Save training example to file."""
        
        # Load existing training data
        training_data = []
        if self.training_data_file.exists():
            with open(self.training_data_file, 'rb') as f:
                training_data = pickle.load(f)
        
        # Add new example
        training_data.append(example)
        
        # Save updated training data
        with open(self.training_data_file, 'wb') as f:
            pickle.dump(training_data, f)
    
    def _load_models(self):
        """Load trained models from disk."""
        
        for model_name in ['random_forest', 'gradient_boosting']:
            model_path = self.model_dir / f"{model_name}.joblib"
            scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
            
            if model_path.exists() and scaler_path.exists():
                try:
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                    print(f"Loaded {model_name} model")
                except Exception as e:
                    print(f"Error loading {model_name} model: {e}")
    
    def _save_model(self, model_name: str, model: Any, scaler: Any):
        """Save model and scaler to disk."""
        
        model_path = self.model_dir / f"{model_name}.joblib"
        scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
        
        try:
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print(f"Saved {model_name} model")
        except Exception as e:
            print(f"Error saving {model_name} model: {e}")

class FeedbackCollector:
    """Collects and manages feedback for the adaptive reviewer."""
    
    def __init__(self, adaptive_reviewer: AdaptiveReviewer):
        self.adaptive_reviewer = adaptive_reviewer
        self.feedback_queue = []
    
    def collect_feedback(self, plan_id: str, plan_type: str, reviewer_id: str,
                        original_prediction: Dict[str, Any], human_corrections: Dict[str, Any],
                        confidence_score: float, review_time: float, notes: str = ""):
        """Collect feedback from human reviewers."""
        
        # Record feedback
        self.adaptive_reviewer.record_feedback(
            plan_id=plan_id,
            plan_type=plan_type,
            reviewer_id=reviewer_id,
            original_prediction=original_prediction,
            human_corrections=human_corrections,
            confidence_score=confidence_score,
            review_time=review_time,
            notes=notes
        )
        
        # Add to queue for batch processing
        self.feedback_queue.append({
            'plan_id': plan_id,
            'timestamp': datetime.now(),
            'corrections': human_corrections
        })
        
        # Retrain models if enough new feedback
        if len(self.feedback_queue) >= 5:  # Adjust threshold as needed
            self._retrain_models()
    
    def _retrain_models(self):
        """Retrain models with new feedback."""
        print("Retraining models with new feedback...")
        self.adaptive_reviewer.train_models()
        self.feedback_queue.clear()  # Clear queue after training
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of collected feedback."""
        return self.adaptive_reviewer.get_learning_statistics()

def main():
    """Example usage of the adaptive reviewer."""
    
    # Initialize adaptive reviewer
    adaptive_reviewer = AdaptiveReviewer()
    
    # Example: Record feedback
    feedback_collector = FeedbackCollector(adaptive_reviewer)
    
    # Simulate feedback collection
    feedback_collector.collect_feedback(
        plan_id="plan_001",
        plan_type="traffic_signal",
        reviewer_id="reviewer_001",
        original_prediction={
            'compliance_score': 0.8,
            'issues': ['Signal height issue'],
            'elements_found': ['signal_head_red', 'signal_head_green']
        },
        human_corrections={
            'compliance_score': 0.7,
            'issues': ['Signal height issue', 'Missing pedestrian button'],
            'elements_found': ['signal_head_red', 'signal_head_green', 'pedestrian_button']
        },
        confidence_score=0.75,
        review_time=120.5,
        notes="Good detection but missed pedestrian features"
    )
    
    # Get learning statistics
    stats = adaptive_reviewer.get_learning_statistics()
    print("Learning Statistics:", stats)

if __name__ == "__main__":
    main()