#!/usr/bin/env python3
"""
Fixed Enhanced Model Retraining Pipeline

Ensures multiple classes for proper model training.
"""

import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
from collections import Counter

class FixedEnhancedRetrainer:
    """
    Fixed retraining system that ensures multiple classes.
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.training_data = []
        self.feature_names = []
        
        print("🔄 Fixed Enhanced Model Retrainer Initialized")
        print("✅ Ensures multiple classes for training")
        print("✅ Improved synthetic labeling")
        print("✅ Robust feature extraction")
    
    def _setup_logging(self):
        """Setup logging for the retraining process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fixed_enhanced_retraining.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def extract_roadway_features(self, image_path):
        """
        Extract comprehensive roadway features from an image.
        """
        self.logger.info(f"Extracting roadway features from: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return None
        
        features = {}
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Basic image properties
            features.update(self._extract_basic_properties(image, gray))
            
            # 2. Line detection and analysis
            features.update(self._extract_line_features(gray))
            
            # 3. Edge detection features
            features.update(self._extract_edge_features(gray))
            
            # 4. Pattern analysis
            features.update(self._extract_pattern_features(gray))
            
            # 5. Text region analysis
            features.update(self._extract_text_features(gray))
            
            # 6. Roadway-specific features
            features.update(self._extract_roadway_specific_features(gray))
            
            self.logger.info(f"Successfully extracted {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def _extract_basic_properties(self, image, gray):
        """Extract basic image properties."""
        height, width = gray.shape
        aspect_ratio = width / height if height > 0 else 0
        
        # Calculate basic statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        min_intensity = np.min(gray)
        max_intensity = np.max(gray)
        
        # Calculate image complexity
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'laplacian_variance': laplacian_var,
            'image_area': width * height
        }
    
    def _extract_line_features(self, gray):
        """Extract line detection features."""
        # Multi-scale edge detection
        edges_canny = cv2.Canny(gray, 20, 60)
        edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges_sobel = np.uint8(np.absolute(edges_sobel))
        
        # Find lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges_canny,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=30,
            maxLineGap=10
        )
        
        total_lines = len(lines) if lines is not None else 0
        
        # Analyze line patterns
        horizontal_lines = 0
        vertical_lines = 0
        diagonal_lines = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Normalize angle
                if angle < 0:
                    angle += 180
                
                # Classify lines
                if abs(angle) <= 15 or abs(angle - 180) <= 15:
                    horizontal_lines += 1
                elif abs(angle - 90) <= 15:
                    vertical_lines += 1
                else:
                    diagonal_lines += 1
        
        # Calculate line densities
        line_density = total_lines / (gray.shape[0] * gray.shape[1]) * 1000000
        horizontal_density = horizontal_lines / (gray.shape[0] * gray.shape[1]) * 1000000
        vertical_density = vertical_lines / (gray.shape[0] * gray.shape[1]) * 1000000
        
        return {
            'total_lines': total_lines,
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'diagonal_lines': diagonal_lines,
            'line_density': line_density,
            'horizontal_density': horizontal_density,
            'vertical_density': vertical_density,
            'line_ratio_horizontal': horizontal_lines / max(total_lines, 1),
            'line_ratio_vertical': vertical_lines / max(total_lines, 1)
        }
    
    def _extract_edge_features(self, gray):
        """Extract edge detection features."""
        # Canny edges
        edges_canny = cv2.Canny(gray, 20, 60)
        edge_pixels_canny = np.sum(edges_canny > 0)
        edge_density_canny = edge_pixels_canny / (gray.shape[0] * gray.shape[1])
        
        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_magnitude = np.uint8(sobel_magnitude * 255 / sobel_magnitude.max())
        
        edge_pixels_sobel = np.sum(sobel_magnitude > 50)
        edge_density_sobel = edge_pixels_sobel / (gray.shape[0] * gray.shape[1])
        
        # Laplacian edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        edge_pixels_laplacian = np.sum(laplacian > 30)
        edge_density_laplacian = edge_pixels_laplacian / (gray.shape[0] * gray.shape[1])
        
        return {
            'edge_density_canny': edge_density_canny,
            'edge_density_sobel': edge_density_sobel,
            'edge_density_laplacian': edge_density_laplacian,
            'edge_pixels_canny': edge_pixels_canny,
            'edge_pixels_sobel': edge_pixels_sobel,
            'edge_pixels_laplacian': edge_pixels_laplacian,
            'total_edge_density': (edge_density_canny + edge_density_sobel + edge_density_laplacian) / 3
        }
    
    def _extract_pattern_features(self, gray):
        """Extract pattern analysis features."""
        # Contour analysis
        edges = cv2.Canny(gray, 20, 60)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_count = len(contours)
        total_contour_area = sum(cv2.contourArea(c) for c in contours)
        avg_contour_area = total_contour_area / max(contour_count, 1)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        morph_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        morph_open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        close_pixels = np.sum(morph_close > 0)
        open_pixels = np.sum(morph_open > 0)
        
        return {
            'contour_count': contour_count,
            'total_contour_area': total_contour_area,
            'avg_contour_area': avg_contour_area,
            'contour_density': contour_count / (gray.shape[0] * gray.shape[1]) * 10000,
            'morph_close_pixels': close_pixels,
            'morph_open_pixels': open_pixels,
            'morph_ratio': close_pixels / max(open_pixels, 1)
        }
    
    def _extract_text_features(self, gray):
        """Extract text region features."""
        # MSER for text region detection
        mser = cv2.MSER_create(min_area=100, max_area=5000)
        regions, _ = mser.detectRegions(gray)
        
        text_region_count = len(regions) if regions is not None else 0
        
        # Analyze text regions
        total_text_area = 0
        avg_text_size = 0
        
        if regions is not None:
            for region in regions:
                if len(region) >= 10:
                    hull = cv2.convexHull(region.reshape(-1, 1, 2))
                    area = cv2.contourArea(hull)
                    total_text_area += area
            
            avg_text_size = total_text_area / max(text_region_count, 1)
        
        text_density = text_region_count / (gray.shape[0] * gray.shape[1]) * 10000
        text_area_ratio = total_text_area / (gray.shape[0] * gray.shape[1])
        
        return {
            'text_region_count': text_region_count,
            'total_text_area': total_text_area,
            'avg_text_size': avg_text_size,
            'text_density': text_density,
            'text_area_ratio': text_area_ratio
        }
    
    def _extract_roadway_specific_features(self, gray):
        """Extract roadway-specific features."""
        # Estimate roadway elements based on line patterns
        edges = cv2.Canny(gray, 20, 60)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, 
                               minLineLength=30, maxLineGap=10)
        
        horizontal_lines = 0
        vertical_lines = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                if angle < 0:
                    angle += 180
                
                if abs(angle) <= 15 or abs(angle - 180) <= 15:
                    horizontal_lines += 1
                elif abs(angle - 90) <= 15:
                    vertical_lines += 1
        
        # Estimate roadway elements
        striping_lines = horizontal_lines // 10
        conduit_bundles = horizontal_lines // 20
        barriers_guardrails = horizontal_lines // 30
        illumination_signals = vertical_lines // 50
        index_patterns = vertical_lines // 100
        
        # Calculate complexity scores
        roadway_complexity = min((striping_lines + conduit_bundles + barriers_guardrails + 
                                illumination_signals + index_patterns) / 100.0, 1.0)
        
        infrastructure_score = min((conduit_bundles + barriers_guardrails + 
                                  illumination_signals) / 50.0, 1.0)
        
        return {
            'striping_lines': striping_lines,
            'conduit_bundles': conduit_bundles,
            'barriers_guardrails': barriers_guardrails,
            'illumination_signals': illumination_signals,
            'index_patterns': index_patterns,
            'roadway_complexity': roadway_complexity,
            'infrastructure_score': infrastructure_score,
            'total_roadway_elements': (striping_lines + conduit_bundles + barriers_guardrails + 
                                     illumination_signals + index_patterns)
        }
    
    def create_training_dataset(self, image_dir="yolo_processed_data_local/images", max_samples=500):
        """
        Create comprehensive training dataset from images.
        """
        self.logger.info("Creating enhanced training dataset")
        
        image_path = Path(image_dir)
        if not image_path.exists():
            self.logger.error(f"Image directory not found: {image_dir}")
            return
        
        # Find all images
        image_files = list(image_path.glob("*.png"))
        self.logger.info(f"Found {len(image_files)} images for training")
        
        # Limit samples for faster processing
        if len(image_files) > max_samples:
            image_files = image_files[:max_samples]
            self.logger.info(f"Limited to {max_samples} samples for faster processing")
        
        training_data = []
        
        for i, image_file in enumerate(image_files):
            if i % 50 == 0:
                self.logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
            
            # Extract features
            features = self.extract_roadway_features(image_file)
            if features:
                # Create synthetic label based on filename and features
                label = self._create_synthetic_label(image_file.name, features)
                
                training_data.append({
                    'filename': image_file.name,
                    'features': features,
                    'label': label
                })
        
        self.training_data = training_data
        self.logger.info(f"Created training dataset with {len(training_data)} samples")
        
        # Check class distribution
        labels = [sample['label'] for sample in training_data]
        label_counts = Counter(labels)
        self.logger.info(f"Label distribution: {dict(label_counts)}")
        
        # Ensure we have multiple classes
        if len(label_counts) < 2:
            self.logger.warning("Only one class detected! Forcing multiple classes...")
            self._force_multiple_classes()
        
        return training_data
    
    def _force_multiple_classes(self):
        """Force multiple classes by adjusting labels based on features."""
        for i, sample in enumerate(self.training_data):
            features = sample['features']
            
            # Create more diverse labels based on feature values
            if features.get('line_density', 0) > 50:
                sample['label'] = 'high_line_density'
            elif features.get('text_density', 0) > 30:
                sample['label'] = 'high_text_density'
            elif features.get('edge_density_canny', 0) > 0.05:
                sample['label'] = 'high_edge_density'
            elif features.get('roadway_complexity', 0) > 0.3:
                sample['label'] = 'roadway_infrastructure'
            elif features.get('conduit_bundles', 0) > 5:
                sample['label'] = 'electrical_conduits'
            elif features.get('striping_lines', 0) > 8:
                sample['label'] = 'roadway_striping'
            elif features.get('aspect_ratio', 0) > 1.5:
                sample['label'] = 'wide_format'
            elif features.get('aspect_ratio', 0) < 0.7:
                sample['label'] = 'tall_format'
            else:
                sample['label'] = 'general_drawing'
        
        # Check final distribution
        labels = [sample['label'] for sample in self.training_data]
        label_counts = Counter(labels)
        self.logger.info(f"Final label distribution: {dict(label_counts)}")
    
    def _create_synthetic_label(self, filename, features):
        """
        Create synthetic labels based on filename and extracted features.
        """
        filename_lower = filename.lower()
        
        # Discipline-based labeling
        if 'electrical' in filename_lower or features.get('conduit_bundles', 0) > 5:
            return 'electrical'
        elif 'structural' in filename_lower:
            return 'structural'
        elif 'civil' in filename_lower or features.get('roadway_complexity', 0) > 0.3:
            return 'civil'
        elif 'mechanical' in filename_lower:
            return 'mechanical'
        
        # Roadway-based labeling
        elif features.get('roadway_complexity', 0) > 0.5:
            return 'roadway_infrastructure'
        elif features.get('striping_lines', 0) > 10:
            return 'roadway_striping'
        elif features.get('conduit_bundles', 0) > 8:
            return 'electrical_conduits'
        elif features.get('infrastructure_score', 0) > 0.4:
            return 'infrastructure'
        
        # Pattern-based labeling
        elif features.get('line_density', 0) > 100:
            return 'high_detail'
        elif features.get('text_density', 0) > 50:
            return 'text_heavy'
        elif features.get('edge_density_canny', 0) > 0.1:
            return 'edge_rich'
        
        # Default
        return 'general_drawing'
    
    def train_enhanced_models(self):
        """
        Train enhanced models with comprehensive features.
        """
        self.logger.info("Training enhanced models")
        
        if not self.training_data:
            self.logger.error("No training data available")
            return
        
        # Prepare features and labels
        X = []
        y = []
        
        for sample in self.training_data:
            features = sample['features']
            X.append(list(features.values()))
            y.append(sample['label'])
        
        # Get feature names
        self.feature_names = list(self.training_data[0]['features'].keys())
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Check class distribution before training
        unique_classes = np.unique(y)
        self.logger.info(f"Classes for training: {unique_classes}")
        self.logger.info(f"Class counts: {dict(Counter(y))}")
        
        if len(unique_classes) < 2:
            self.logger.error("Still only one class! Cannot train models.")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        self.logger.info("Training Random Forest model")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting
        self.logger.info("Training Gradient Boosting model")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_score = rf_model.score(X_test_scaled, y_test)
        gb_score = gb_model.score(X_test_scaled, y_test)
        
        self.logger.info(f"Random Forest Accuracy: {rf_score:.4f}")
        self.logger.info(f"Gradient Boosting Accuracy: {gb_score:.4f}")
        
        # Cross-validation
        rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
        gb_cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5)
        
        self.logger.info(f"Random Forest CV Score: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")
        self.logger.info(f"Gradient Boosting CV Score: {gb_cv_scores.mean():.4f} (+/- {gb_cv_scores.std() * 2:.4f})")
        
        # Save models
        self._save_models(rf_model, gb_model, scaler)
        
        # Generate detailed reports
        self._generate_training_reports(rf_model, gb_model, X_test_scaled, y_test)
        
        return {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'scaler': scaler,
            'feature_names': self.feature_names,
            'rf_accuracy': rf_score,
            'gb_accuracy': gb_score
        }
    
    def _save_models(self, rf_model, gb_model, scaler):
        """Save trained models and scaler."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create models directory
        models_dir = Path("fixed_enhanced_models")
        models_dir.mkdir(exist_ok=True)
        
        # Save models
        with open(models_dir / f"fixed_enhanced_rf_model_{timestamp}.pkl", 'wb') as f:
            pickle.dump(rf_model, f)
        
        with open(models_dir / f"fixed_enhanced_gb_model_{timestamp}.pkl", 'wb') as f:
            pickle.dump(gb_model, f)
        
        with open(models_dir / f"fixed_enhanced_scaler_{timestamp}.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names
        with open(models_dir / f"fixed_enhanced_feature_names_{timestamp}.json", 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Save training data
        with open(models_dir / f"fixed_enhanced_training_data_{timestamp}.json", 'w') as f:
            json.dump(self.training_data, f, indent=2, default=str)
        
        self.logger.info(f"Models saved to {models_dir}")
    
    def _generate_training_reports(self, rf_model, gb_model, X_test, y_test):
        """Generate detailed training reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create reports directory
        reports_dir = Path("fixed_enhanced_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Feature importance analysis
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        gb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        rf_importance.to_csv(reports_dir / f"fixed_rf_feature_importance_{timestamp}.csv", index=False)
        gb_importance.to_csv(reports_dir / f"fixed_gb_feature_importance_{timestamp}.csv", index=False)
        
        # Generate classification reports
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        
        rf_report = classification_report(y_test, rf_pred, output_dict=True)
        gb_report = classification_report(y_test, gb_pred, output_dict=True)
        
        # Save reports
        with open(reports_dir / f"fixed_rf_classification_report_{timestamp}.json", 'w') as f:
            json.dump(rf_report, f, indent=2)
        
        with open(reports_dir / f"fixed_gb_classification_report_{timestamp}.json", 'w') as f:
            json.dump(gb_report, f, indent=2)
        
        # Create visualization
        self._create_training_visualization(rf_importance, gb_importance, timestamp)
        
        self.logger.info(f"Training reports saved to {reports_dir}")
    
    def _create_training_visualization(self, rf_importance, gb_importance, timestamp):
        """Create training visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # RF Feature Importance
        top_rf_features = rf_importance.head(10)
        ax1.barh(range(len(top_rf_features)), top_rf_features['importance'])
        ax1.set_yticks(range(len(top_rf_features)))
        ax1.set_yticklabels(top_rf_features['feature'])
        ax1.set_xlabel('Importance')
        ax1.set_title('Random Forest - Top 10 Feature Importance')
        ax1.invert_yaxis()
        
        # GB Feature Importance
        top_gb_features = gb_importance.head(10)
        ax2.barh(range(len(top_gb_features)), top_gb_features['importance'])
        ax2.set_yticks(range(len(top_gb_features)))
        ax2.set_yticklabels(top_gb_features['feature'])
        ax2.set_xlabel('Importance')
        ax2.set_title('Gradient Boosting - Top 10 Feature Importance')
        ax2.invert_yaxis()
        
        # Training data distribution
        labels = [sample['label'] for sample in self.training_data]
        label_counts = pd.Series(labels).value_counts()
        ax3.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        ax3.set_title('Training Data Distribution')
        
        # Feature correlation heatmap
        feature_df = pd.DataFrame([sample['features'] for sample in self.training_data])
        correlation_matrix = feature_df.corr()
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        ax4.set_xticks(range(len(correlation_matrix.columns)))
        ax4.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax4.set_yticks(range(len(correlation_matrix.columns)))
        ax4.set_yticklabels(correlation_matrix.columns)
        ax4.set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        # Save visualization
        reports_dir = Path("fixed_enhanced_reports")
        fig.savefig(reports_dir / f"fixed_enhanced_training_visualization_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    """Main retraining pipeline."""
    print("🔄 Fixed Enhanced Model Retraining Pipeline")
    print("=" * 60)
    
    # Initialize retrainer
    retrainer = FixedEnhancedRetrainer()
    
    # Create training dataset
    print("\n📊 Creating enhanced training dataset...")
    training_data = retrainer.create_training_dataset(max_samples=500)  # Limit for faster processing
    
    if not training_data:
        print("❌ Failed to create training dataset")
        return
    
    # Train enhanced models
    print("\n🤖 Training enhanced models...")
    models = retrainer.train_enhanced_models()
    
    if models:
        print(f"\n✅ Enhanced model training complete!")
        print(f"📈 Random Forest Accuracy: {models['rf_accuracy']:.4f}")
        print(f"📈 Gradient Boosting Accuracy: {models['gb_accuracy']:.4f}")
        print(f"🔧 Features used: {len(models['feature_names'])}")
        print(f"📁 Models saved to: fixed_enhanced_models/")
        print(f"📊 Reports saved to: fixed_enhanced_reports/")
    else:
        print("❌ Model training failed")


if __name__ == "__main__":
    main()
