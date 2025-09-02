#!/usr/bin/env python3
"""
Enhanced Model Retraining Pipeline

Integrates roadway detection capabilities with existing ML models:
- Foundation elements (north arrow, scale, legend, notes)
- Discipline classification
- Index symbol recognition
- Roadway infrastructure detection
- Multi-page reference handling
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

# Import our custom modules
from src.core.discipline_classifier import DisciplineClassifier
from src.foundation_elements.foundation_orchestrator import FoundationOrchestrator
from src.core.enhanced_ml_pipeline import EnhancedMLPipeline

class EnhancedModelRetrainer:
    """
    Comprehensive model retraining system that integrates all detection capabilities.
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.foundation_orchestrator = FoundationOrchestrator()
        self.discipline_classifier = DisciplineClassifier()
        self.enhanced_pipeline = EnhancedMLPipeline()
        
        # Roadway detection parameters
        self.roadway_features = {
            'striping_lines': 0,
            'conduit_bundles': 0,
            'barriers_guardrails': 0,
            'illumination_signals': 0,
            'index_patterns': 0,
            'multi_page_references': 0,
            'total_roadway_elements': 0
        }
        
        # Training data storage
        self.training_data = []
        self.feature_names = []
        
        print("🔄 Enhanced Model Retrainer Initialized")
        print("✅ Foundation elements integration")
        print("✅ Discipline classification")
        print("✅ Roadway infrastructure detection")
        print("✅ Multi-page index recognition")
    
    def _setup_logging(self):
        """Setup logging for the retraining process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_retraining.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def extract_enhanced_features(self, image_path):
        """
        Extract comprehensive features from an image including roadway elements.
        """
        self.logger.info(f"Extracting enhanced features from: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return None
        
        features = {}
        
        try:
            # 1. Foundation Elements Analysis
            foundation_results = self.foundation_orchestrator.analyze_drawing(image)
            features.update(self._extract_foundation_features(foundation_results))
            
            # 2. Discipline Classification
            discipline_results = self.discipline_classifier.classify_drawing(image)
            features.update(self._extract_discipline_features(discipline_results))
            
            # 3. Roadway Infrastructure Detection
            roadway_results = self._detect_roadway_elements(image)
            features.update(self._extract_roadway_features(roadway_results))
            
            # 4. Enhanced ML Pipeline Features
            pipeline_results = self.enhanced_pipeline.analyze_yolo_data(image)
            features.update(self._extract_pipeline_features(pipeline_results))
            
            # 5. Multi-page Index Analysis
            index_results = self._analyze_index_patterns(image)
            features.update(self._extract_index_features(index_results))
            
            self.logger.info(f"Successfully extracted {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def _extract_foundation_features(self, foundation_results):
        """Extract features from foundation elements analysis."""
        features = {
            'north_arrow_detected': 0,
            'scale_detected': 0,
            'legend_detected': 0,
            'notes_count': 0,
            'coordinate_system_detected': 0,
            'match_lines_detected': 0,
            'foundation_completeness_score': 0.0
        }
        
        if foundation_results:
            features['north_arrow_detected'] = 1 if foundation_results.get('north_arrow') else 0
            features['scale_detected'] = 1 if foundation_results.get('scale') else 0
            features['legend_detected'] = 1 if foundation_results.get('legend') else 0
            features['notes_count'] = len(foundation_results.get('notes', []))
            features['coordinate_system_detected'] = 1 if foundation_results.get('coordinate_system') else 0
            features['match_lines_detected'] = len(foundation_results.get('match_lines', []))
            features['foundation_completeness_score'] = foundation_results.get('completeness_score', 0.0)
        
        return features
    
    def _extract_discipline_features(self, discipline_results):
        """Extract features from discipline classification."""
        features = {
            'primary_discipline_confidence': 0.0,
            'sub_discipline_confidence': 0.0,
            'drawing_type_confidence': 0.0,
            'electrical_elements': 0,
            'structural_elements': 0,
            'civil_elements': 0,
            'mechanical_elements': 0
        }
        
        if discipline_results:
            features['primary_discipline_confidence'] = discipline_results.get('primary_confidence', 0.0)
            features['sub_discipline_confidence'] = discipline_results.get('sub_discipline_confidence', 0.0)
            features['drawing_type_confidence'] = discipline_results.get('drawing_type_confidence', 0.0)
            
            # Count discipline-specific elements
            primary_discipline = discipline_results.get('primary_discipline', '').lower()
            if 'electrical' in primary_discipline:
                features['electrical_elements'] = 1
            elif 'structural' in primary_discipline:
                features['structural_elements'] = 1
            elif 'civil' in primary_discipline:
                features['civil_elements'] = 1
            elif 'mechanical' in primary_discipline:
                features['mechanical_elements'] = 1
        
        return features
    
    def _detect_roadway_elements(self, image):
        """Detect roadway infrastructure elements."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 20, 60)
            
            # Find lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=20,
                minLineLength=30,
                maxLineGap=10
            )
            
            results = {
                'total_lines': len(lines) if lines is not None else 0,
                'striping_lines': 0,
                'conduit_bundles': 0,
                'barriers_guardrails': 0,
                'illumination_signals': 0,
                'index_patterns': 0
            }
            
            if lines is not None:
                # Analyze line patterns for roadway elements
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    if abs(angle) <= 15 or abs(angle - 180) <= 15:
                        horizontal_lines.append(line[0])
                    elif abs(angle - 90) <= 15:
                        vertical_lines.append(line[0])
                
                # Estimate roadway elements based on line patterns
                results['striping_lines'] = len(horizontal_lines) // 10  # Estimate
                results['conduit_bundles'] = len(horizontal_lines) // 20  # Estimate
                results['barriers_guardrails'] = len(horizontal_lines) // 30  # Estimate
                results['illumination_signals'] = len(vertical_lines) // 50  # Estimate
                results['index_patterns'] = len(vertical_lines) // 100  # Estimate
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in roadway detection: {e}")
            return {'total_lines': 0, 'striping_lines': 0, 'conduit_bundles': 0, 
                   'barriers_guardrails': 0, 'illumination_signals': 0, 'index_patterns': 0}
    
    def _extract_roadway_features(self, roadway_results):
        """Extract features from roadway detection."""
        features = {
            'total_lines': roadway_results.get('total_lines', 0),
            'striping_lines': roadway_results.get('striping_lines', 0),
            'conduit_bundles': roadway_results.get('conduit_bundles', 0),
            'barriers_guardrails': roadway_results.get('barriers_guardrails', 0),
            'illumination_signals': roadway_results.get('illumination_signals', 0),
            'index_patterns': roadway_results.get('index_patterns', 0),
            'roadway_complexity_score': 0.0
        }
        
        # Calculate roadway complexity score
        total_elements = (features['striping_lines'] + features['conduit_bundles'] + 
                         features['barriers_guardrails'] + features['illumination_signals'] + 
                         features['index_patterns'])
        features['roadway_complexity_score'] = min(total_elements / 100.0, 1.0)
        
        return features
    
    def _extract_pipeline_features(self, pipeline_results):
        """Extract features from enhanced ML pipeline."""
        features = {
            'yolo_features_count': 0,
            'enhanced_features_count': 0,
            'synthetic_labels_count': 0,
            'model_confidence': 0.0
        }
        
        if pipeline_results:
            features['yolo_features_count'] = len(pipeline_results.get('yolo_features', []))
            features['enhanced_features_count'] = len(pipeline_results.get('enhanced_features', []))
            features['synthetic_labels_count'] = len(pipeline_results.get('synthetic_labels', []))
            features['model_confidence'] = pipeline_results.get('confidence', 0.0)
        
        return features
    
    def _analyze_index_patterns(self, image):
        """Analyze index patterns and multi-page references."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use MSER for text region detection
            mser = cv2.MSER_create(_min_area=100, _max_area=5000)
            regions, _ = mser.detectRegions(gray)
            
            results = {
                'text_regions': len(regions) if regions is not None else 0,
                'index_patterns': 0,
                'sheet_references': 0,
                'grid_coordinates': 0,
                'multi_page_references': 0
            }
            
            if regions is not None:
                # Estimate index patterns based on text regions
                results['index_patterns'] = len(regions) // 20
                results['sheet_references'] = len(regions) // 30
                results['grid_coordinates'] = len(regions) // 40
                results['multi_page_references'] = len(regions) // 50
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in index pattern analysis: {e}")
            return {'text_regions': 0, 'index_patterns': 0, 'sheet_references': 0, 
                   'grid_coordinates': 0, 'multi_page_references': 0}
    
    def _extract_index_features(self, index_results):
        """Extract features from index pattern analysis."""
        features = {
            'text_regions': index_results.get('text_regions', 0),
            'index_patterns': index_results.get('index_patterns', 0),
            'sheet_references': index_results.get('sheet_references', 0),
            'grid_coordinates': index_results.get('grid_coordinates', 0),
            'multi_page_references': index_results.get('multi_page_references', 0),
            'index_completeness_score': 0.0
        }
        
        # Calculate index completeness score
        total_index_elements = (features['index_patterns'] + features['sheet_references'] + 
                              features['grid_coordinates'] + features['multi_page_references'])
        features['index_completeness_score'] = min(total_index_elements / 50.0, 1.0)
        
        return features
    
    def create_training_dataset(self, image_dir="yolo_processed_data_local/images"):
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
        
        training_data = []
        
        for i, image_file in enumerate(image_files):
            if i % 100 == 0:
                self.logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
            
            # Extract features
            features = self.extract_enhanced_features(image_file)
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
        
        return training_data
    
    def _create_synthetic_label(self, filename, features):
        """
        Create synthetic labels based on filename and extracted features.
        """
        filename_lower = filename.lower()
        
        # Discipline-based labeling
        if 'electrical' in filename_lower or features.get('electrical_elements', 0) > 0:
            return 'electrical'
        elif 'structural' in filename_lower or features.get('structural_elements', 0) > 0:
            return 'structural'
        elif 'civil' in filename_lower or features.get('civil_elements', 0) > 0:
            return 'civil'
        elif 'mechanical' in filename_lower or features.get('mechanical_elements', 0) > 0:
            return 'mechanical'
        
        # Roadway-based labeling
        elif features.get('roadway_complexity_score', 0) > 0.5:
            return 'roadway_infrastructure'
        elif features.get('striping_lines', 0) > 5:
            return 'roadway_striping'
        elif features.get('conduit_bundles', 0) > 3:
            return 'electrical_conduits'
        
        # Index-based labeling
        elif features.get('index_completeness_score', 0) > 0.7:
            return 'index_reference'
        elif features.get('multi_page_references', 0) > 2:
            return 'multi_page_index'
        
        # Foundation-based labeling
        elif features.get('foundation_completeness_score', 0) > 0.8:
            return 'foundation_complete'
        
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
        models_dir = Path("enhanced_models")
        models_dir.mkdir(exist_ok=True)
        
        # Save models
        with open(models_dir / f"enhanced_rf_model_{timestamp}.pkl", 'wb') as f:
            pickle.dump(rf_model, f)
        
        with open(models_dir / f"enhanced_gb_model_{timestamp}.pkl", 'wb') as f:
            pickle.dump(gb_model, f)
        
        with open(models_dir / f"enhanced_scaler_{timestamp}.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names
        with open(models_dir / f"enhanced_feature_names_{timestamp}.json", 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Save training data
        with open(models_dir / f"enhanced_training_data_{timestamp}.json", 'w') as f:
            json.dump(self.training_data, f, indent=2, default=str)
        
        self.logger.info(f"Models saved to {models_dir}")
    
    def _generate_training_reports(self, rf_model, gb_model, X_test, y_test):
        """Generate detailed training reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create reports directory
        reports_dir = Path("enhanced_reports")
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
        rf_importance.to_csv(reports_dir / f"rf_feature_importance_{timestamp}.csv", index=False)
        gb_importance.to_csv(reports_dir / f"gb_feature_importance_{timestamp}.csv", index=False)
        
        # Generate classification reports
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        
        rf_report = classification_report(y_test, rf_pred, output_dict=True)
        gb_report = classification_report(y_test, gb_pred, output_dict=True)
        
        # Save reports
        with open(reports_dir / f"rf_classification_report_{timestamp}.json", 'w') as f:
            json.dump(rf_report, f, indent=2)
        
        with open(reports_dir / f"gb_classification_report_{timestamp}.json", 'w') as f:
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
        reports_dir = Path("enhanced_reports")
        fig.savefig(reports_dir / f"enhanced_training_visualization_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    """Main retraining pipeline."""
    print("🔄 Enhanced Model Retraining Pipeline")
    print("=" * 60)
    
    # Initialize retrainer
    retrainer = EnhancedModelRetrainer()
    
    # Create training dataset
    print("\n📊 Creating enhanced training dataset...")
    training_data = retrainer.create_training_dataset()
    
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
        print(f"📁 Models saved to: enhanced_models/")
        print(f"📊 Reports saved to: enhanced_reports/")
    else:
        print("❌ Model training failed")


if __name__ == "__main__":
    main()
