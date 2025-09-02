#!/usr/bin/env python3
"""
Enhanced ML Training with Processed As-Built Data
Advanced training script using real engineering data for improved ML capabilities.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import defaultdict

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ml_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EnhancedMLTrainer:
    """Enhanced ML trainer using processed as-built data."""
    
    def __init__(self):
        self.data_dir = Path("enhanced_training_data")
        self.models_dir = Path("enhanced_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.features_data = []
        self.metadata_data = []
        self.training_features = []
        self.training_labels = []
        
        # Models
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Performance tracking
        self.training_history = []
        
    def load_processed_data(self):
        """Load all processed as-built data."""
        logger.info("Loading processed as-built data...")
        
        # Load features
        features_dir = self.data_dir / "features"
        for feature_file in features_dir.glob("*.json"):
            try:
                with open(feature_file, 'r') as f:
                    features = json.load(f)
                    self.features_data.extend(features)
                logger.info(f"Loaded features from {feature_file.name}")
            except Exception as e:
                logger.error(f"Error loading {feature_file}: {e}")
        
        # Load metadata
        metadata_dir = self.data_dir / "metadata"
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.metadata_data.append(metadata)
                logger.info(f"Loaded metadata from {metadata_file.name}")
            except Exception as e:
                logger.error(f"Error loading {metadata_file}: {e}")
        
        logger.info(f"Loaded {len(self.features_data)} feature sets and {len(self.metadata_data)} metadata records")
    
    def extract_advanced_features(self) -> List[Dict[str, Any]]:
        """Extract advanced engineering features from processed data."""
        logger.info("Extracting advanced engineering features...")
        
        advanced_features = []
        
        for feature_set in self.features_data:
            features = {}
            
            # Basic image features
            if 'image_size' in feature_set:
                features['width'] = feature_set['image_size'][1]
                features['height'] = feature_set['image_size'][0]
                features['pixel_count'] = feature_set['image_size'][0] * feature_set['image_size'][1]
            
            if 'aspect_ratio' in feature_set:
                features['aspect_ratio'] = feature_set['aspect_ratio']
            
            # Color features
            if 'mean_color' in feature_set:
                features['mean_red'] = feature_set['mean_color'][2]  # BGR format
                features['mean_green'] = feature_set['mean_color'][1]
                features['mean_blue'] = feature_set['mean_color'][0]
                features['color_variance'] = np.var(feature_set['mean_color'])
            
            if 'std_color' in feature_set:
                features['color_std_red'] = feature_set['std_color'][2]
                features['color_std_green'] = feature_set['std_color'][1]
                features['color_std_blue'] = feature_set['std_color'][0]
            
            # Engineering drawing features
            if 'edge_density' in feature_set:
                features['edge_density'] = feature_set['edge_density']
                features['edge_complexity'] = feature_set['edge_density'] * features.get('pixel_count', 1000000)
            
            if 'line_count' in feature_set:
                features['line_count'] = feature_set['line_count']
                features['line_density'] = feature_set['line_count'] / features.get('pixel_count', 1000000) * 1000000
            
            if 'contour_count' in feature_set:
                features['contour_count'] = feature_set['contour_count']
                features['contour_density'] = feature_set['contour_count'] / features.get('pixel_count', 1000000) * 1000000
            
            if 'text_regions' in feature_set:
                features['text_regions'] = feature_set['text_regions']
                features['text_density'] = feature_set['text_regions'] / features.get('pixel_count', 1000000) * 1000000
            
            # Project metadata features
            source_pdf = feature_set.get('source_pdf', '')
            project_metadata = self.get_project_metadata(source_pdf)
            
            if project_metadata:
                features['file_size_mb'] = project_metadata.get('file_size_mb', 0)
                features['page_count'] = project_metadata.get('page_count', 1)
                features['project_category'] = self.encode_category(project_metadata.get('project_category', 'unknown'))
                
                # Text analysis features
                text_sample = project_metadata.get('text_sample', '')
                features['text_length'] = len(text_sample)
                features['word_count'] = len(text_sample.split())
                features['text_density'] = features['text_length'] / features.get('pixel_count', 1000000) * 1000000
            
            # Engineering complexity features
            features['drawing_complexity'] = self.calculate_drawing_complexity(features)
            features['engineering_density'] = self.calculate_engineering_density(features)
            
            advanced_features.append(features)
        
        logger.info(f"Extracted {len(advanced_features)} advanced feature sets")
        return advanced_features
    
    def get_project_metadata(self, source_pdf: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific PDF source."""
        for metadata in self.metadata_data:
            if metadata.get('filename') == source_pdf:
                return metadata
        return None
    
    def encode_category(self, category: str) -> int:
        """Encode project category as integer."""
        category_mapping = {
            'traffic_signal': 0,
            'its': 1,
            'electrical': 2,
            'structural': 3,
            'congestion': 4,
            'special': 5,
            'unknown': 6
        }
        return category_mapping.get(category, 6)
    
    def calculate_drawing_complexity(self, features: Dict[str, Any]) -> float:
        """Calculate overall drawing complexity score."""
        complexity_factors = [
            features.get('edge_density', 0) * 100,
            features.get('line_count', 0) / 100,
            features.get('contour_count', 0) / 50,
            features.get('text_regions', 0) / 10,
            features.get('file_size_mb', 0) / 10
        ]
        return np.mean(complexity_factors)
    
    def calculate_engineering_density(self, features: Dict[str, Any]) -> float:
        """Calculate engineering element density."""
        density_factors = [
            features.get('line_density', 0),
            features.get('contour_density', 0),
            features.get('text_density', 0)
        ]
        return np.mean(density_factors)
    
    def create_training_labels(self, advanced_features: List[Dict[str, Any]]) -> np.ndarray:
        """Create multi-output training labels."""
        logger.info("Creating training labels...")
        
        labels = []
        
        for features in advanced_features:
            # Multi-output labels for engineering analysis
            label_vector = [
                features.get('drawing_complexity', 0),      # Complexity score
                features.get('engineering_density', 0),     # Engineering density
                features.get('edge_density', 0),           # Edge complexity
                features.get('line_density', 0),           # Line density
                features.get('text_density', 0),           # Text density
                features.get('aspect_ratio', 1.0),         # Aspect ratio
                features.get('color_variance', 0),         # Color variance
                features.get('file_size_mb', 0),           # File size
                features.get('page_count', 1),             # Page count
                features.get('project_category', 6)        # Project category
            ]
            labels.append(label_vector)
        
        return np.array(labels)
    
    def prepare_training_data(self):
        """Prepare training data from processed as-built files."""
        logger.info("Preparing training data...")
        
        # Load processed data
        self.load_processed_data()
        
        # Extract advanced features
        advanced_features = self.extract_advanced_features()
        
        if not advanced_features:
            logger.error("No features extracted from processed data!")
            return False
        
        # Convert features to training matrix
        feature_names = list(advanced_features[0].keys())
        X = []
        
        for features in advanced_features:
            feature_vector = [features.get(name, 0.0) for name in feature_names]
            X.append(feature_vector)
        
        X = np.array(X)
        
        # Create labels
        y = self.create_training_labels(advanced_features)
        
        # Store training data
        self.training_features = X
        self.training_labels = y
        self.feature_names = feature_names
        
        logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Feature names: {feature_names}")
        
        return True
    
    def train_enhanced_models(self):
        """Train enhanced ML models with advanced features."""
        logger.info("Training enhanced ML models...")
        
        if len(self.training_features) == 0:
            logger.error("No training data available!")
            return False
        
        X = self.training_features
        y = self.training_labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['feature_scaler'] = scaler
        
        # Define models
        models = {
            'enhanced_random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'enhanced_gradient_boosting': MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=8,
                    min_samples_split=5,
                    random_state=42
                )
            ),
            'enhanced_xgboost': MultiOutputRegressor(
                xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            )
        }
        
        # Train models
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                
                # Store results
                self.models[model_name] = model
                
                training_result = {
                    'model_name': model_name,
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_date': datetime.now().isoformat()
                }
                
                self.training_history.append(training_result)
                
                logger.info(f"[OK] {model_name} trained successfully:")
                logger.info(f"   R² Score: {r2:.4f}")
                logger.info(f"   MSE: {mse:.4f}")
                logger.info(f"   MAE: {mae:.4f}")
                logger.info(f"   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    top_features = np.argsort(importances)[-10:]  # Top 10 features
                    logger.info(f"   Top 10 features: {[self.feature_names[i] for i in top_features]}")
                elif hasattr(model, 'estimators_'):
                    # For MultiOutputRegressor
                    base_model = model.estimators_[0]
                    if hasattr(base_model, 'feature_importances_'):
                        importances = base_model.feature_importances_
                        top_features = np.argsort(importances)[-10:]
                        logger.info(f"   Top 10 features: {[self.feature_names[i] for i in top_features]}")
                
            except Exception as e:
                logger.error(f"[ERROR] Error training {model_name}: {e}")
        
        return len(self.models) > 0
    
    def save_enhanced_models(self):
        """Save trained models and metadata."""
        logger.info("Saving enhanced models...")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = self.models_dir / f"{scaler_name}.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {scaler_name} to {scaler_path}")
        
        # Save training metadata
        metadata = {
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'data_shape': {
                'X_shape': self.training_features.shape,
                'y_shape': self.training_labels.shape
            },
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = self.models_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved training metadata to {metadata_path}")
    
    def generate_training_report(self):
        """Generate comprehensive training report."""
        logger.info("Generating training report...")
        
        report = {
            'training_summary': {
                'total_models_trained': len(self.models),
                'total_training_samples': len(self.training_features),
                'feature_count': len(self.feature_names),
                'label_dimensions': self.training_labels.shape[1],
                'training_date': datetime.now().isoformat()
            },
            'model_performance': self.training_history,
            'feature_analysis': {
                'feature_names': self.feature_names,
                'feature_count': len(self.feature_names)
            },
            'data_quality': {
                'missing_values': np.isnan(self.training_features).sum(),
                'feature_ranges': {
                    name: {
                        'min': float(np.min(self.training_features[:, i])),
                        'max': float(np.max(self.training_features[:, i])),
                        'mean': float(np.mean(self.training_features[:, i])),
                        'std': float(np.std(self.training_features[:, i]))
                    }
                    for i, name in enumerate(self.feature_names)
                }
            }
        }
        
        report_path = self.models_dir / "enhanced_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")
        
        # Print summary
        print("\nEnhanced ML Training Summary:")
        print(f"   Models Trained: {len(self.models)}")
        print(f"   Training Samples: {len(self.training_features)}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Label Dimensions: {self.training_labels.shape[1]}")
        
        if self.training_history:
            best_model = max(self.training_history, key=lambda x: x['r2'])
            print(f"   Best Model: {best_model['model_name']} (R²: {best_model['r2']:.4f})")

def main():
    """Main function for enhanced ML training."""
    print("Enhanced ML Training with Processed As-Built Data")
    print("=" * 70)
    
    trainer = EnhancedMLTrainer()
    
    # Prepare training data
    if not trainer.prepare_training_data():
        print("Failed to prepare training data!")
        return
    
    # Train models
    if not trainer.train_enhanced_models():
        print("Failed to train models!")
        return
    
    # Save models
    trainer.save_enhanced_models()
    
    # Generate report
    trainer.generate_training_report()
    
    print("\nEnhanced ML Training Complete!")
    print("Check enhanced_models/ directory for trained models")
    print("Review enhanced_ml_training.log for detailed information")

if __name__ == "__main__":
    main()
