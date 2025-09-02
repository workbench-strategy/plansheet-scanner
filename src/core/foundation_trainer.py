"""
Foundation Trainer for Traffic Plan Review
Trains models using as-builts and past reviewed plans without human feedback.
"""

import json
import logging
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

# ML imports
try:
    import joblib
    from sklearn.cluster import KMeans
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Add XGBoost import
    try:
        import xgboost as xgb

        XGBOOST_AVAILABLE = True
    except ImportError:
        print("Warning: XGBoost not installed. Install with: pip install xgboost")
        XGBOOST_AVAILABLE = False
except ImportError:
    print(
        "Warning: ML dependencies not installed. Install with: pip install scikit-learn joblib"
    )
    XGBOOST_AVAILABLE = False


@dataclass
class AsBuiltData:
    """Represents as-built data for training."""

    plan_id: str
    plan_type: str
    as_built_image: np.ndarray
    final_elements: List[Dict[str, Any]]  # Final approved elements
    construction_notes: str
    approval_date: datetime
    project_info: Dict[str, Any]


@dataclass
class ReviewMilestone:
    """Represents a review milestone with feedback."""

    plan_id: str
    milestone: str  # 'preliminary', 'final', 'construction', 'as_built'
    reviewer_comments: List[str]
    approved_elements: List[Dict[str, Any]]
    rejected_elements: List[Dict[str, Any]]
    compliance_score: float
    review_date: datetime


class FoundationTrainer:
    """Trains foundation models using as-builts and past reviewed plans."""

    def __init__(self, data_dir: str = "training_data", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

        # Data storage
        self.as_built_data = []
        self.review_milestones = []
        self.training_features = []
        self.training_labels = []

        # Models
        self.foundation_models = {}
        self.scalers = {}
        self.feature_extractors = {}

        # Load existing data
        self._load_training_data()

    def add_as_built_data(
        self,
        plan_id: str,
        plan_type: str,
        as_built_image: np.ndarray,
        final_elements: List[Dict[str, Any]],
        construction_notes: str,
        approval_date: datetime,
        project_info: Dict[str, Any],
    ):
        """Add as-built data for training."""

        as_built = AsBuiltData(
            plan_id=plan_id,
            plan_type=plan_type,
            as_built_image=as_built_image,
            final_elements=final_elements,
            construction_notes=construction_notes,
            approval_date=approval_date,
            project_info=project_info,
        )

        self.as_built_data.append(as_built)
        self._save_as_built_data(as_built)

        print(f"✅ Added as-built data for plan {plan_id}")

    def add_review_milestone(
        self,
        plan_id: str,
        milestone: str,
        reviewer_comments: List[str],
        approved_elements: List[Dict[str, Any]],
        rejected_elements: List[Dict[str, Any]],
        compliance_score: float,
        review_date: datetime,
    ):
        """Add review milestone data for training."""

        milestone_data = ReviewMilestone(
            plan_id=plan_id,
            milestone=milestone,
            reviewer_comments=reviewer_comments,
            approved_elements=approved_elements,
            rejected_elements=rejected_elements,
            compliance_score=compliance_score,
            review_date=review_date,
        )

        self.review_milestones.append(milestone_data)
        self._save_review_milestone(milestone_data)

        print(f"✅ Added review milestone for plan {plan_id} ({milestone})")

    def extract_training_features(self):
        """Extract features from as-built and review data."""

        print("🔍 Extracting training features from as-built and review data...")

        # Extract features from as-built data
        for as_built in self.as_built_data:
            features = self._extract_as_built_features(as_built)
            labels = self._extract_as_built_labels(as_built)

            self.training_features.append(features)
            self.training_labels.append(labels)

        # Extract features from review milestones
        for milestone in self.review_milestones:
            features = self._extract_milestone_features(milestone)
            labels = self._extract_milestone_labels(milestone)

            self.training_features.append(features)
            self.training_labels.append(labels)

        print(
            f"✅ Extracted features from {len(self.training_features)} training examples"
        )

    def train_foundation_models(self, min_examples: int = 20):
        """Train foundation models on as-built and review data."""

        if len(self.training_features) < min_examples:
            print(
                f"❌ Need at least {min_examples} examples, have {len(self.training_features)}"
            )
            return

        print(
            f"🤖 Training foundation models on {len(self.training_features)} examples..."
        )

        # Ensure all features have consistent dimensions
        max_features = max(len(features) for features in self.training_features)
        print(f"📊 Max feature dimension: {max_features}")

        # Pad all features to consistent length
        padded_features = []
        for features in self.training_features:
            padded = list(features)  # Convert to list if it's not already
            while len(padded) < max_features:
                padded.append(0.0)
            padded_features.append(padded)

        # Convert to numpy arrays
        X = np.array(padded_features)
        y = np.array(self.training_labels)

        print(f"📊 Feature array shape: {X.shape}")
        print(f"📊 Label array shape: {y.shape}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train different model types - using regression models for continuous labels
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor

        model_types = {
            "foundation_random_forest": RandomForestRegressor(
                n_estimators=200, random_state=42
            ),
            "foundation_gradient_boosting": MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=200, random_state=42)
            ),
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            model_types["foundation_xgboost"] = MultiOutputRegressor(
                xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    eval_metric="rmse",
                )
            )

        for model_name, model in model_types.items():
            print(f"Training {model_name}...")

            # Train model
            model.fit(X_train_scaled, y_train)

            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

            # Log detailed cross-validation metrics
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            print(f"  Train R² score: {train_score:.3f}")
            print(f"  Test R² score: {test_score:.3f}")
            print(f"  CV R² score: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")
            print(f"  CV scores: {cv_scores.tolist()}")

            # Log additional metrics for XGBoost
            if model_name == "foundation_xgboost" and hasattr(model, "estimators_"):
                # For MultiOutputRegressor, we need to access the underlying model
                base_model = model.estimators_[0]
                if hasattr(base_model, "feature_importances_"):
                    top_features = np.argsort(base_model.feature_importances_)[
                        -5:
                    ]  # Top 5 features
                    print(
                        f"  Top 5 feature importances: {base_model.feature_importances_[top_features]}"
                    )

            # Save model
            self.foundation_models[model_name] = model
            self.scalers[model_name] = scaler

            # Save to disk
            self._save_foundation_model(model_name, model, scaler)

        # Save training statistics
        self._save_training_statistics(X_train, X_test, y_train, y_test)

        # Log model availability summary
        available_models = list(self.foundation_models.keys())
        print(f"✅ Foundation models trained and saved: {available_models}")

        if XGBOOST_AVAILABLE and "foundation_xgboost" in available_models:
            print("✅ XGBoost model successfully trained and integrated")
        elif not XGBOOST_AVAILABLE:
            print(
                "⚠️  XGBoost not available - only RandomForest and GradientBoosting models trained"
            )

    def predict_with_foundation_models(
        self, plan_image: np.ndarray, detected_elements: List[Any]
    ) -> Dict[str, Any]:
        """Make predictions using foundation models."""

        if not self.foundation_models:
            print("❌ No foundation models trained yet")
            return {}

        # Extract features
        features = self._extract_plan_features(plan_image, detected_elements)

        predictions = {}
        confidence_scores = {}

        for model_name, model in self.foundation_models.items():
            if model_name in self.scalers:
                features_scaled = self.scalers[model_name].transform([features])
                prediction = model.predict(features_scaled)[0]
                confidence = model.predict_proba(features_scaled)[0].max()

                predictions[model_name] = prediction
                confidence_scores[model_name] = confidence

        # Ensemble prediction
        final_prediction = self._ensemble_predictions(predictions, confidence_scores)

        return {
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "final_prediction": final_prediction,
            "features": features.tolist(),
            "model_type": "foundation",
        }

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about training data."""

        stats = {
            "total_as_built": len(self.as_built_data),
            "total_milestones": len(self.review_milestones),
            "total_training_examples": len(self.training_features),
            "models_trained": len(self.foundation_models),
            "as_built_by_type": defaultdict(int),
            "milestones_by_type": defaultdict(int),
            "data_timeline": [],
        }

        # As-built statistics
        for as_built in self.as_built_data:
            stats["as_built_by_type"][as_built.plan_type] += 1
            stats["data_timeline"].append(
                {
                    "date": as_built.approval_date.isoformat(),
                    "type": "as_built",
                    "plan_type": as_built.plan_type,
                }
            )

        # Milestone statistics
        for milestone in self.review_milestones:
            stats["milestones_by_type"][milestone.milestone] += 1
            stats["data_timeline"].append(
                {
                    "date": milestone.review_date.isoformat(),
                    "type": "milestone",
                    "milestone": milestone.milestone,
                }
            )

        return stats

    def _extract_as_built_features(self, as_built: AsBuiltData) -> np.ndarray:
        """Extract features from as-built data."""
        features = []

        # Image features
        features.extend(self._extract_image_features(as_built.as_built_image))

        # Element features
        features.extend(self._extract_element_features(as_built.final_elements))

        # Project features
        features.extend(self._extract_project_features(as_built.project_info))

        # Construction notes features
        features.extend(self._extract_text_features(as_built.construction_notes))

        return np.array(features)

    def _extract_as_built_labels(self, as_built: AsBuiltData) -> np.ndarray:
        """Extract labels from as-built data."""
        labels = []

        # Compliance score (as-built is typically high compliance)
        labels.append(0.95)  # High compliance for as-built

        # Element count
        labels.append(len(as_built.final_elements))

        # Element diversity
        element_types = set()
        for element in as_built.final_elements:
            element_types.add(element.get("type", "unknown"))
        labels.append(len(element_types))

        # Pad to consistent length
        while len(labels) < 10:
            labels.append(0.0)

        return np.array(labels)

    def _extract_milestone_features(self, milestone: ReviewMilestone) -> np.ndarray:
        """Extract features from review milestone data."""
        features = []

        # Milestone type encoding
        milestone_encoding = {
            "preliminary": [1, 0, 0, 0],
            "final": [0, 1, 0, 0],
            "construction": [0, 0, 1, 0],
            "as_built": [0, 0, 0, 1],
        }
        features.extend(milestone_encoding.get(milestone.milestone, [0, 0, 0, 0]))

        # Approved vs rejected elements
        features.append(len(milestone.approved_elements))
        features.append(len(milestone.rejected_elements))
        features.append(milestone.compliance_score)

        # Comment features
        features.extend(
            self._extract_text_features(" ".join(milestone.reviewer_comments))
        )

        # Element features
        all_elements = milestone.approved_elements + milestone.rejected_elements
        features.extend(self._extract_element_features(all_elements))

        return np.array(features)

    def _extract_milestone_labels(self, milestone: ReviewMilestone) -> np.ndarray:
        """Extract labels from review milestone data."""
        labels = []

        # Compliance score
        labels.append(milestone.compliance_score)

        # Element counts
        labels.append(len(milestone.approved_elements))
        labels.append(len(milestone.rejected_elements))

        # Approval ratio
        total_elements = len(milestone.approved_elements) + len(
            milestone.rejected_elements
        )
        approval_ratio = (
            len(milestone.approved_elements) / total_elements
            if total_elements > 0
            else 1.0
        )
        labels.append(approval_ratio)

        # Pad to consistent length
        while len(labels) < 10:
            labels.append(0.0)

        return np.array(labels)

    def _extract_plan_features(
        self, plan_image: np.ndarray, detected_elements: List[Any]
    ) -> np.ndarray:
        """Extract features from a new plan for prediction."""
        features = []

        # Image features
        features.extend(self._extract_image_features(plan_image))

        # Element features
        element_dicts = []
        for element in detected_elements:
            element_dicts.append(
                {
                    "type": element.element_type,
                    "confidence": element.confidence,
                    "location": element.location,
                    "metadata": element.metadata,
                }
            )
        features.extend(self._extract_element_features(element_dicts))

        # Pad to consistent length
        while len(features) < 512:  # Match training feature dimension
            features.append(0.0)

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
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) > 2
                else image
            )
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
        except:
            features.append(0.0)

        return features

    def _extract_element_features(self, elements: List[Dict[str, Any]]) -> List[float]:
        """Extract features from element data."""
        features = []

        # Element counts by type
        element_types = defaultdict(int)
        for element in elements:
            element_type = element.get("type", "unknown").split("_")[0]
            element_types[element_type] += 1

        # Count features for common element types
        for element_type in [
            "signal",
            "sign",
            "detector",
            "marking",
            "camera",
            "sensor",
        ]:
            features.append(element_types[element_type])

        # Confidence statistics
        if elements:
            confidences = [e.get("confidence", 0.5) for e in elements]
            features.append(np.mean(confidences))
            features.append(np.std(confidences))
            features.append(np.min(confidences))
            features.append(np.max(confidences))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        return features

    def _extract_project_features(self, project_info: Dict[str, Any]) -> List[float]:
        """Extract features from project information."""
        features = []

        # Project size indicators
        features.append(project_info.get("budget", 0) / 1000000)  # Budget in millions
        features.append(project_info.get("duration_months", 0))
        features.append(project_info.get("complexity_score", 0))

        # Project type encoding
        project_types = ["intersection", "corridor", "freeway", "bridge", "tunnel"]
        for ptype in project_types:
            features.append(1.0 if project_info.get("project_type") == ptype else 0.0)

        return features

    def _extract_text_features(self, text: str) -> List[float]:
        """Extract features from text data."""
        features = []

        # Basic text features
        features.append(len(text))
        features.append(len(text.split()))
        features.append(len(text.split(".")))

        # Keyword features
        keywords = ["compliance", "approved", "rejected", "modification", "standard"]
        for keyword in keywords:
            features.append(text.lower().count(keyword))

        # Pad to consistent length
        while len(features) < 20:
            features.append(0.0)

        return features

    def _ensemble_predictions(
        self, predictions: Dict[str, Any], confidence_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine predictions from multiple foundation models."""

        if not predictions:
            return {}

        # Weighted average based on confidence
        ensemble_prediction = {}

        for model_name, prediction in predictions.items():
            confidence = confidence_scores.get(model_name, 0.5)

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

    def _save_as_built_data(self, as_built: AsBuiltData):
        """Save as-built data to disk."""
        data_file = self.data_dir / f"as_built_{as_built.plan_id}.pkl"

        # Convert image to bytes for storage
        as_built_dict = {
            "plan_id": as_built.plan_id,
            "plan_type": as_built.plan_type,
            "as_built_image": cv2.imencode(".png", as_built.as_built_image)[
                1
            ].tobytes(),
            "final_elements": as_built.final_elements,
            "construction_notes": as_built.construction_notes,
            "approval_date": as_built.approval_date,
            "project_info": as_built.project_info,
        }

        with open(data_file, "wb") as f:
            pickle.dump(as_built_dict, f)

    def _save_review_milestone(self, milestone: ReviewMilestone):
        """Save review milestone data to disk."""
        data_file = (
            self.data_dir / f"milestone_{milestone.plan_id}_{milestone.milestone}.pkl"
        )

        milestone_dict = {
            "plan_id": milestone.plan_id,
            "milestone": milestone.milestone,
            "reviewer_comments": milestone.reviewer_comments,
            "approved_elements": milestone.approved_elements,
            "rejected_elements": milestone.rejected_elements,
            "compliance_score": milestone.compliance_score,
            "review_date": milestone.review_date,
        }

        with open(data_file, "wb") as f:
            pickle.dump(milestone_dict, f)

    def _save_foundation_model(self, model_name: str, model: Any, scaler: Any):
        """Save foundation model to disk."""
        model_path = self.model_dir / f"{model_name}.joblib"
        scaler_path = self.model_dir / f"{model_name}_scaler.joblib"

        try:
            # Handle XGBoost models specially if needed
            if model_name == "foundation_xgboost" and XGBOOST_AVAILABLE:
                # XGBoost models can be saved with joblib, but we can add special handling here
                # For example, save additional metadata
                model_metadata = {
                    "model_type": "xgboost",
                    "n_estimators": model.n_estimators,
                    "learning_rate": model.learning_rate,
                    "max_depth": model.max_depth,
                    "feature_importances": model.feature_importances_.tolist()
                    if hasattr(model, "feature_importances_")
                    else None,
                }
                metadata_path = self.model_dir / f"{model_name}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(model_metadata, f, indent=2)
                print(f"  Saved {model_name} metadata")

            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print(f"  Saved {model_name}")
        except Exception as e:
            print(f"  Error saving {model_name}: {e}")

    def _save_training_statistics(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        """Save training statistics."""
        stats = {
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_dimension": X_train.shape[1],
            "label_dimension": y_train.shape[1],
            "training_date": datetime.now().isoformat(),
        }

        stats_file = self.model_dir / "foundation_training_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    def _load_training_data(self):
        """Load existing training data from disk."""

        # Load as-built data
        for data_file in self.data_dir.glob("as_built_*.pkl"):
            try:
                with open(data_file, "rb") as f:
                    data = pickle.load(f)

                # Convert image back from bytes
                image_bytes = data["as_built_image"]
                image_array = cv2.imdecode(
                    np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
                )

                as_built = AsBuiltData(
                    plan_id=data["plan_id"],
                    plan_type=data["plan_type"],
                    as_built_image=image_array,
                    final_elements=data["final_elements"],
                    construction_notes=data["construction_notes"],
                    approval_date=data["approval_date"],
                    project_info=data["project_info"],
                )

                self.as_built_data.append(as_built)
            except Exception as e:
                print(f"Error loading as-built data {data_file}: {e}")

        # Load review milestones
        for data_file in self.data_dir.glob("milestone_*.pkl"):
            try:
                with open(data_file, "rb") as f:
                    data = pickle.load(f)

                milestone = ReviewMilestone(
                    plan_id=data["plan_id"],
                    milestone=data["milestone"],
                    reviewer_comments=data["reviewer_comments"],
                    approved_elements=data["approved_elements"],
                    rejected_elements=data["rejected_elements"],
                    compliance_score=data["compliance_score"],
                    review_date=data["review_date"],
                )

                self.review_milestones.append(milestone)
            except Exception as e:
                print(f"Error loading milestone data {data_file}: {e}")

        print(
            f"✅ Loaded {len(self.as_built_data)} as-built records and {len(self.review_milestones)} milestone records"
        )

        # Load existing foundation models
        self._load_foundation_models()

    def _load_foundation_models(self):
        """Load existing foundation models from disk."""
        model_files = list(self.model_dir.glob("foundation_*.joblib"))

        for model_file in model_files:
            model_name = model_file.stem
            scaler_file = self.model_dir / f"{model_name}_scaler.joblib"

            if scaler_file.exists():
                try:
                    model = joblib.load(model_file)
                    scaler = joblib.load(scaler_file)

                    self.foundation_models[model_name] = model
                    self.scalers[model_name] = scaler

                    print(f"✅ Loaded {model_name}")

                    # Load XGBoost metadata if available
                    if model_name == "foundation_xgboost":
                        metadata_file = self.model_dir / f"{model_name}_metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                            print(
                                f"  XGBoost metadata: {metadata.get('n_estimators')} estimators, "
                                f"learning_rate={metadata.get('learning_rate')}, "
                                f"max_depth={metadata.get('max_depth')}"
                            )

                except Exception as e:
                    print(f"❌ Error loading {model_name}: {e}")

        if self.foundation_models:
            print(f"✅ Loaded {len(self.foundation_models)} foundation models")
        else:
            print("ℹ️  No existing foundation models found")


def main():
    """Example usage of the foundation trainer."""

    # Initialize foundation trainer
    trainer = FoundationTrainer()

    # Example: Add as-built data
    # (In practice, you'd load real as-built images and data)
    dummy_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

    trainer.add_as_built_data(
        plan_id="as_built_001",
        plan_type="traffic_signal",
        as_built_image=dummy_image,
        final_elements=[
            {"type": "signal_head_red", "location": (100, 200), "approved": True},
            {"type": "signal_head_green", "location": (100, 220), "approved": True},
            {"type": "detector_loop", "location": (80, 180), "approved": True},
        ],
        construction_notes="All elements installed per approved plans. No modifications required.",
        approval_date=datetime.now(),
        project_info={
            "budget": 500000,
            "duration_months": 6,
            "complexity_score": 0.7,
            "project_type": "intersection",
        },
    )

    # Example: Add review milestone
    trainer.add_review_milestone(
        plan_id="plan_001",
        milestone="final",
        reviewer_comments=[
            "Signal head placement meets ITE standards",
            "Detector coverage adequate for traffic flow",
            "Pedestrian features properly implemented",
        ],
        approved_elements=[
            {"type": "signal_head_red", "location": (100, 200)},
            {"type": "signal_head_green", "location": (100, 220)},
            {"type": "detector_loop", "location": (80, 180)},
        ],
        rejected_elements=[],
        compliance_score=0.95,
        review_date=datetime.now(),
    )

    # Extract features and train models
    trainer.extract_training_features()
    trainer.train_foundation_models(min_examples=1)

    # Get training statistics
    stats = trainer.get_training_statistics()
    print("Training Statistics:", stats)


if __name__ == "__main__":
    main()
