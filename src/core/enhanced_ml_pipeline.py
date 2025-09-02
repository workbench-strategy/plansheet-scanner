"""
Enhanced ML Pipeline for Plansheet Scanner

This module integrates YOLO processed data with existing foundation models
to create a comprehensive training and inference system.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMLPipeline:
    """
    Enhanced ML pipeline that integrates YOLO processed data with foundation models.
    """

    def __init__(
        self,
        data_dir: str = "yolo_processed_data_local",
        model_dir: str = "models",
        output_dir: str = "enhanced_models",
    ):
        """
        Initialize the enhanced ML pipeline.

        Args:
            data_dir: Directory containing YOLO processed data
            model_dir: Directory containing existing foundation models
            output_dir: Directory to save enhanced models
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.feature_data = None
        self.enhanced_features = None
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}

        logger.info(f"Enhanced ML Pipeline initialized with data_dir: {data_dir}")

    def analyze_yolo_data(self) -> Dict[str, Any]:
        """
        Analyze the YOLO processed data to understand patterns and characteristics.

        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing YOLO processed data...")

        features_dir = self.data_dir / "features"
        if not features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {features_dir}")

        # Load all feature files
        feature_files = list(features_dir.glob("*_features.json"))
        logger.info(f"Found {len(feature_files)} feature files")

        # Load and analyze features
        data = []
        for file in feature_files:
            try:
                with open(file, "r") as f:
                    features = json.load(f)
                data.append(features)
            except Exception as e:
                logger.warning(f"Error loading {file}: {e}")

        if not data:
            raise ValueError("No valid feature data found")

        # Create DataFrame for analysis
        df = pd.DataFrame(data)

        # Basic statistics
        analysis = {
            "total_pages": len(df),
            "unique_pdfs": df["source_pdf"].nunique()
            if "source_pdf" in df.columns
            else 0,
            "image_dimensions": {
                "width_mean": df["width"].mean(),
                "width_std": df["width"].std(),
                "height_mean": df["height"].mean(),
                "height_std": df["height"].std(),
                "aspect_ratio_mean": df["aspect_ratio"].mean(),
                "aspect_ratio_std": df["aspect_ratio"].std(),
            },
            "feature_statistics": {
                "line_count_mean": df["line_count"].mean(),
                "line_count_std": df["line_count"].std(),
                "contour_count_mean": df["contour_count"].mean(),
                "contour_count_std": df["contour_count"].std(),
                "edge_density_mean": df["edge_density"].mean(),
                "edge_density_std": df["edge_density"].std(),
                "brightness_mean": df["brightness_mean"].mean(),
                "brightness_std": df["brightness_std"].mean(),
            },
            "quality_metrics": {
                "processing_quality_distribution": df["processing_quality"]
                .value_counts()
                .to_dict()
                if "processing_quality" in df.columns
                else {}
            },
        }

        # Save analysis results
        analysis_file = self.output_dir / "yolo_data_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info(f"YOLO data analysis completed. Results saved to {analysis_file}")
        return analysis

    def create_enhanced_features(self) -> np.ndarray:
        """
        Create enhanced features by combining YOLO data with engineering-specific features.

        Returns:
            Enhanced feature matrix
        """
        logger.info("Creating enhanced features...")

        features_dir = self.data_dir / "features"
        feature_files = list(features_dir.glob("*_features.json"))

        enhanced_features = []

        for file in feature_files:
            try:
                with open(file, "r") as f:
                    features = json.load(f)

                # Extract basic features
                basic_features = [
                    features.get("width", 0),
                    features.get("height", 0),
                    features.get("aspect_ratio", 0),
                    features.get("color_variance", 0),
                    features.get("total_pixels", 0),
                    features.get("edge_density", 0),
                    features.get("line_count", 0),
                    features.get("contour_count", 0),
                    features.get("texture_variance", 0),
                    features.get("brightness_mean", 0),
                    features.get("brightness_std", 0),
                ]

                # Create engineering-specific features
                engineering_features = self._extract_engineering_features(features)

                # Combine features
                combined_features = basic_features + engineering_features
                enhanced_features.append(combined_features)

            except Exception as e:
                logger.warning(f"Error processing {file}: {e}")

        self.enhanced_features = np.array(enhanced_features)
        logger.info(
            f"Created enhanced features with shape: {self.enhanced_features.shape}"
        )

        return self.enhanced_features

    def _extract_engineering_features(self, features: Dict[str, Any]) -> List[float]:
        """
        Extract engineering-specific features from image data.

        Args:
            features: Dictionary containing basic image features

        Returns:
            List of engineering-specific feature values
        """
        # Calculate derived engineering features
        width = features.get("width", 1)
        height = features.get("height", 1)
        line_count = features.get("line_count", 0)
        contour_count = features.get("contour_count", 0)
        edge_density = features.get("edge_density", 0)

        engineering_features = [
            # Layout complexity features
            line_count / (width * height) * 1000000,  # Line density
            contour_count / (width * height) * 1000000,  # Contour density
            edge_density * 100,  # Edge density percentage
            # Drawing scale features
            width / height,  # Aspect ratio
            np.sqrt(width * height),  # Geometric mean of dimensions
            # Complexity indicators
            line_count / max(contour_count, 1),  # Lines per contour
            edge_density * line_count,  # Combined complexity
            # Quality indicators
            features.get("brightness_std", 0)
            / max(features.get("brightness_mean", 1), 1),  # Contrast ratio
            features.get("color_variance", 0) / 1000,  # Normalized color variance
            # Engineering drawing patterns
            min(line_count, 10000) / 1000,  # Normalized line count
            min(contour_count, 1000) / 100,  # Normalized contour count
        ]

        return engineering_features

    def create_synthetic_labels(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Create synthetic labels based on image characteristics for training.

        Args:
            feature_matrix: Matrix of enhanced features

        Returns:
            Array of synthetic labels
        """
        logger.info("Creating synthetic labels based on image characteristics...")

        # Create labels based on feature patterns
        labels = []

        for features in feature_matrix:
            # Extract key features for labeling
            line_density = features[12]  # Line density feature
            contour_density = features[13]  # Contour density feature
            edge_density = features[14]  # Edge density feature
            aspect_ratio = features[15]  # Aspect ratio
            complexity = features[18]  # Combined complexity

            # Create synthetic label based on characteristics
            if line_density > 50 and contour_density > 5:
                label = 0  # Complex engineering drawing
            elif edge_density > 0.05 and complexity > 100:
                label = 1  # Detailed technical drawing
            elif aspect_ratio > 1.5:
                label = 2  # Landscape layout
            elif line_density < 10 and contour_density < 2:
                label = 3  # Simple diagram
            else:
                label = 4  # Standard drawing

            labels.append(label)

        labels = np.array(labels)
        logger.info(
            f"Created {len(labels)} synthetic labels with distribution: {np.bincount(labels)}"
        )

        return labels

    def train_enhanced_models(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train enhanced models using the expanded dataset.

        Args:
            features: Feature matrix
            labels: Label array
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing training results and performance metrics
        """
        logger.info("Training enhanced models...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define models
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=random_state
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=random_state
            ),
        }

        results = {}

        for name, model in models.items():
            logger.info(f"Training {name}...")

            # Train model
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)

            # Performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

            # Feature importance
            if hasattr(model, "feature_importances_"):
                feature_importance = model.feature_importances_
            else:
                feature_importance = None

            results[name] = {
                "model": model,
                "scaler": scaler,
                "accuracy": accuracy,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "classification_report": classification_rep,
                "confusion_matrix": conf_matrix.tolist(),
                "feature_importance": feature_importance.tolist()
                if feature_importance is not None
                else None,
            }

            logger.info(
                f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
            )

        self.models = results
        self.performance_metrics = results

        # Save models and results
        self._save_enhanced_models()

        return results

    def _save_enhanced_models(self):
        """Save enhanced models and performance metrics."""
        logger.info("Saving enhanced models...")

        # Save models
        for name, result in self.models.items():
            model_file = self.output_dir / f"enhanced_{name}.joblib"
            scaler_file = self.output_dir / f"enhanced_{name}_scaler.joblib"

            joblib.dump(result["model"], model_file)
            joblib.dump(result["scaler"], scaler_file)

            logger.info(f"Saved {name} model to {model_file}")

        # Save performance metrics
        metrics_file = self.output_dir / "enhanced_model_performance.json"
        metrics_to_save = {}

        for name, result in self.models.items():
            metrics_to_save[name] = {
                "accuracy": result["accuracy"],
                "cv_mean": result["cv_mean"],
                "cv_std": result["cv_std"],
                "classification_report": result["classification_report"],
                "confusion_matrix": result["confusion_matrix"],
            }

        with open(metrics_file, "w") as f:
            json.dump(metrics_to_save, f, indent=2, default=str)

        logger.info(f"Saved performance metrics to {metrics_file}")

    def predict_with_enhanced_models(
        self, features: np.ndarray, model_name: str = "random_forest"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using enhanced models.

        Args:
            features: Feature matrix for prediction
            model_name: Name of the model to use

        Returns:
            Tuple of (predictions, prediction_probabilities)
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model {model_name} not found. Available models: {list(self.models.keys())}"
            )

        model_result = self.models[model_name]
        model = model_result["model"]
        scaler = model_result["scaler"]

        # Scale features
        features_scaled = scaler.transform(features)

        # Make predictions
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)

        return predictions, probabilities

    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report.

        Returns:
            Formatted performance report string
        """
        if not self.performance_metrics:
            return "No performance metrics available. Train models first."

        report = []
        report.append("=" * 60)
        report.append("ENHANCED ML PIPELINE PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        for name, metrics in self.performance_metrics.items():
            report.append(f"Model: {name.upper()}")
            report.append("-" * 40)
            report.append(f"Accuracy: {metrics['accuracy']:.4f}")
            report.append(
                f"Cross-Validation: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})"
            )
            report.append("")

            # Classification report
            report.append("Classification Report:")
            for label, scores in metrics["classification_report"].items():
                if isinstance(scores, dict):
                    report.append(f"  {label}:")
                    report.append(f"    Precision: {scores.get('precision', 0):.4f}")
                    report.append(f"    Recall: {scores.get('recall', 0):.4f}")
                    report.append(f"    F1-Score: {scores.get('f1-score', 0):.4f}")
            report.append("")

        return "\n".join(report)


def main():
    """Main function to demonstrate enhanced ML pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced ML Pipeline for Plansheet Scanner"
    )
    parser.add_argument(
        "--mode",
        choices=["analyze", "train", "predict"],
        default="train",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--data_dir",
        default="yolo_processed_data_local",
        help="Directory containing YOLO processed data",
    )
    parser.add_argument(
        "--model_dir", default="models", help="Directory containing existing models"
    )
    parser.add_argument(
        "--output_dir",
        default="enhanced_models",
        help="Directory to save enhanced models",
    )
    parser.add_argument(
        "--augment", action="store_true", help="Enable data augmentation"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Enable cross-validation"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = EnhancedMLPipeline(args.data_dir, args.model_dir, args.output_dir)

    if args.mode == "analyze":
        # Analyze YOLO data
        analysis = pipeline.analyze_yolo_data()
        print("YOLO Data Analysis Results:")
        print(json.dumps(analysis, indent=2))

    elif args.mode == "train":
        # Create enhanced features
        features = pipeline.create_enhanced_features()

        # Create synthetic labels
        labels = pipeline.create_synthetic_labels(features)

        # Train enhanced models
        results = pipeline.train_enhanced_models(features, labels)

        # Generate performance report
        report = pipeline.generate_performance_report()
        print(report)

        # Save report
        report_file = pipeline.output_dir / "performance_report.txt"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"\nPerformance report saved to: {report_file}")

    elif args.mode == "predict":
        # Load features and make predictions
        features = pipeline.create_enhanced_features()
        predictions, probabilities = pipeline.predict_with_enhanced_models(features)

        print(f"Made predictions for {len(predictions)} samples")
        print(f"Prediction distribution: {np.bincount(predictions)}")


if __name__ == "__main__":
    main()
