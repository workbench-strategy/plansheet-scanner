"""
Machine Learning Pipeline System for Plansheet Scanner

This module provides comprehensive ML pipeline capabilities including automated training,
model versioning, deployment, and monitoring.

Features:
- Automated model training and evaluation
- Model versioning and registry management
- Feature engineering and selection
- Hyperparameter optimization
- Model deployment and monitoring
- A/B testing capabilities
"""

import asyncio
import json
import os
import pickle
import shutil
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import optuna
import pandas as pd

# ML imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Database imports
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


@dataclass
class ModelConfig:
    """Model configuration data structure"""

    model_id: str
    name: str
    type: str  # 'classification', 'regression', 'clustering'
    algorithm: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class TrainingJob:
    """Training job data structure"""

    job_id: str
    model_config_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float  # 0.0 to 1.0
    current_step: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    metrics: Dict[str, float]
    model_path: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class ModelVersion:
    """Model version data structure"""

    version_id: str
    model_id: str
    version_number: int
    training_job_id: str
    model_path: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    created_at: datetime
    is_active: bool
    metadata: Dict[str, Any]


@dataclass
class FeatureSet:
    """Feature set data structure"""

    feature_set_id: str
    name: str
    description: str
    features: List[str]
    feature_engineering_steps: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class Experiment:
    """ML experiment data structure"""

    experiment_id: str
    name: str
    description: str
    model_configs: List[str]
    hyperparameter_ranges: Dict[str, List[Any]]
    optimization_metric: str
    max_trials: int
    status: str  # 'pending', 'running', 'completed', 'failed'
    best_model_id: Optional[str]
    results: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


class MLPipeline:
    """Machine Learning Pipeline for automated model training and deployment"""

    def __init__(
        self, models_dir: str = "models", db_config: Optional[Dict[str, Any]] = None
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Storage
        self.model_configs: Dict[str, ModelConfig] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.model_versions: Dict[str, List[ModelVersion]] = defaultdict(list)
        self.feature_sets: Dict[str, FeatureSet] = {}
        self.experiments: Dict[str, Experiment] = {}

        # Active models
        self.active_models: Dict[str, Any] = {}
        self.model_pipelines: Dict[str, Pipeline] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

        # Training queue
        self.training_queue: deque = deque()
        self.training_thread: Optional[threading.Thread] = None
        self.is_training = False

        # Database connection
        self.db_engine = None
        if db_config:
            self._setup_database(db_config)

        # Initialize default models
        self._setup_default_models()

        # Start training worker
        self.start_training_worker()

    def _setup_database(self, db_config: Dict[str, Any]):
        """Setup database connection for ML pipeline storage"""
        try:
            connection_string = (
                f"postgresql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            self.db_engine = create_engine(connection_string)

            # Create ML pipeline tables
            self._create_ml_tables()

        except Exception as e:
            print(f"Warning: Database setup failed: {e}")

    def _create_ml_tables(self):
        """Create ML pipeline tables in database"""
        if not self.db_engine:
            return

        try:
            with self.db_engine.connect() as conn:
                # Model configurations table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS model_configs (
                        model_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        type VARCHAR(50) NOT NULL,
                        algorithm VARCHAR(100) NOT NULL,
                        hyperparameters JSONB,
                        feature_columns TEXT[],
                        target_column VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Training jobs table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS training_jobs (
                        job_id VARCHAR(255) PRIMARY KEY,
                        model_config_id VARCHAR(255) NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        progress DOUBLE PRECISION NOT NULL,
                        current_step VARCHAR(255),
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        metrics JSONB,
                        model_path VARCHAR(500),
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Model versions table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS model_versions (
                        version_id VARCHAR(255) PRIMARY KEY,
                        model_id VARCHAR(255) NOT NULL,
                        version_number INTEGER NOT NULL,
                        training_job_id VARCHAR(255) NOT NULL,
                        model_path VARCHAR(500) NOT NULL,
                        metrics JSONB,
                        hyperparameters JSONB,
                        feature_importance JSONB,
                        created_at TIMESTAMP NOT NULL,
                        is_active BOOLEAN NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Feature sets table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS feature_sets (
                        feature_set_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        features TEXT[],
                        feature_engineering_steps JSONB,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Experiments table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS experiments (
                        experiment_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        model_configs TEXT[],
                        hyperparameter_ranges JSONB,
                        optimization_metric VARCHAR(100) NOT NULL,
                        max_trials INTEGER NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        best_model_id VARCHAR(255),
                        results JSONB,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                conn.commit()

        except Exception as e:
            print(f"Warning: Failed to create ML tables: {e}")

    def _setup_default_models(self):
        """Setup default model configurations"""
        # Symbol Recognition Model
        symbol_model = ModelConfig(
            model_id="symbol_recognition",
            name="Symbol Recognition",
            type="classification",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 100, "max_depth": 10, "random_state": 42},
            feature_columns=["area", "perimeter", "circularity", "aspect_ratio"],
            target_column="symbol_type",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"description": "Classifies symbols in plan sheets"},
        )
        self.model_configs["symbol_recognition"] = symbol_model

        # Quality Prediction Model
        quality_model = ModelConfig(
            model_id="quality_prediction",
            name="Quality Prediction",
            type="regression",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 100, "max_depth": 15, "random_state": 42},
            feature_columns=["processing_time", "file_size", "complexity_score"],
            target_column="quality_score",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"description": "Predicts quality scores for processing jobs"},
        )
        self.model_configs["quality_prediction"] = quality_model

        # Performance Prediction Model
        performance_model = ModelConfig(
            model_id="performance_prediction",
            name="Performance Prediction",
            type="regression",
            algorithm="linear_regression",
            hyperparameters={"fit_intercept": True, "normalize": False},
            feature_columns=["cpu_usage", "memory_usage", "file_size"],
            target_column="processing_time",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                "description": "Predicts processing time based on system metrics"
            },
        )
        self.model_configs["performance_prediction"] = performance_model

    def start_training_worker(self):
        """Start the training worker thread"""
        if self.is_training:
            return

        self.is_training = True
        self.training_thread = threading.Thread(
            target=self._training_worker_loop, daemon=True
        )
        self.training_thread.start()

    def stop_training_worker(self):
        """Stop the training worker thread"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join()

    def _training_worker_loop(self):
        """Training worker loop"""
        while self.is_training:
            try:
                if self.training_queue:
                    job_id = self.training_queue.popleft()
                    self._execute_training_job(job_id)
                else:
                    time.sleep(5)  # Wait for new jobs

            except Exception as e:
                print(f"Error in training worker: {e}")
                time.sleep(10)

    def create_model_config(
        self,
        name: str,
        model_type: str,
        algorithm: str,
        feature_columns: List[str],
        target_column: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelConfig:
        """Create a new model configuration"""
        model_config = ModelConfig(
            model_id=str(uuid.uuid4()),
            name=name,
            type=model_type,
            algorithm=algorithm,
            hyperparameters=hyperparameters or {},
            feature_columns=feature_columns,
            target_column=target_column,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {},
        )

        self.model_configs[model_config.model_id] = model_config

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        INSERT INTO model_configs 
                        (model_id, name, type, algorithm, hyperparameters, feature_columns, target_column, created_at, updated_at, metadata)
                        VALUES (:model_id, :name, :type, :algorithm, :hyperparameters, :feature_columns, :target_column, :created_at, :updated_at, :metadata)
                    """
                        ),
                        asdict(model_config),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing model config: {e}")

        return model_config

    def create_feature_set(
        self,
        name: str,
        description: str,
        features: List[str],
        feature_engineering_steps: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeatureSet:
        """Create a new feature set"""
        feature_set = FeatureSet(
            feature_set_id=str(uuid.uuid4()),
            name=name,
            description=description,
            features=features,
            feature_engineering_steps=feature_engineering_steps or [],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {},
        )

        self.feature_sets[feature_set.feature_set_id] = feature_set

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        INSERT INTO feature_sets 
                        (feature_set_id, name, description, features, feature_engineering_steps, created_at, updated_at, metadata)
                        VALUES (:feature_set_id, :name, :description, :features, :feature_engineering_steps, :created_at, :updated_at, :metadata)
                    """
                        ),
                        asdict(feature_set),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing feature set: {e}")

        return feature_set

    def submit_training_job(
        self,
        model_config_id: str,
        training_data: pd.DataFrame,
        validation_split: float = 0.2,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a training job"""
        if model_config_id not in self.model_configs:
            raise ValueError(f"Model config {model_config_id} not found")

        # Create training job
        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            model_config_id=model_config_id,
            status="pending",
            progress=0.0,
            current_step="Queued",
            start_time=None,
            end_time=None,
            metrics={},
            model_path=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {},
        )

        self.training_jobs[job.job_id] = job

        # Store training data temporarily
        job.metadata["training_data"] = training_data.to_dict()
        job.metadata["validation_split"] = validation_split

        # Add to training queue
        self.training_queue.append(job.job_id)

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        INSERT INTO training_jobs 
                        (job_id, model_config_id, status, progress, current_step, start_time, end_time, metrics, model_path, created_at, updated_at, metadata)
                        VALUES (:job_id, :model_config_id, :status, :progress, :current_step, :start_time, :end_time, :metrics, :model_path, :created_at, :updated_at, :metadata)
                    """
                        ),
                        asdict(job),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing training job: {e}")

        return job.job_id

    def _execute_training_job(self, job_id: str):
        """Execute a training job"""
        if job_id not in self.training_jobs:
            return

        job = self.training_jobs[job_id]
        model_config = self.model_configs[job.model_config_id]

        try:
            # Update job status
            job.status = "running"
            job.start_time = datetime.now()
            job.current_step = "Loading data"
            job.progress = 0.1
            self._update_training_job(job)

            # Load training data
            training_data = pd.DataFrame.from_dict(job.metadata["training_data"])
            validation_split = job.metadata.get("validation_split", 0.2)

            # Prepare features and target
            X = training_data[model_config.feature_columns]
            y = training_data[model_config.target_column]

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )

            job.current_step = "Feature engineering"
            job.progress = 0.2
            self._update_training_job(job)

            # Feature engineering
            X_train_processed, X_val_processed = self._apply_feature_engineering(
                X_train, X_val, model_config
            )

            job.current_step = "Training model"
            job.progress = 0.4
            self._update_training_job(job)

            # Train model
            model, pipeline = self._train_model(
                X_train_processed, y_train, model_config
            )

            job.current_step = "Evaluating model"
            job.progress = 0.7
            self._update_training_job(job)

            # Evaluate model
            metrics = self._evaluate_model(
                model, X_val_processed, y_val, model_config.type
            )

            job.current_step = "Saving model"
            job.progress = 0.9
            self._update_training_job(job)

            # Save model
            model_path = self._save_model(model, pipeline, model_config, job)

            # Create model version
            version = self._create_model_version(model_config, job, model_path, metrics)

            # Update job
            job.status = "completed"
            job.end_time = datetime.now()
            job.progress = 1.0
            job.current_step = "Completed"
            job.metrics = metrics
            job.model_path = model_path
            self._update_training_job(job)

            # Load model if it's the first version or better than current
            if version.is_active:
                self._load_model(version)

        except Exception as e:
            # Update job with error
            job.status = "failed"
            job.end_time = datetime.now()
            job.current_step = f"Failed: {str(e)}"
            self._update_training_job(job)
            print(f"Training job {job_id} failed: {e}")

    def _apply_feature_engineering(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, model_config: ModelConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply feature engineering to training and validation data"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Store scaler
        self.scalers[model_config.model_id] = scaler

        return X_train_scaled, X_val_scaled

    def _train_model(
        self, X_train: np.ndarray, y_train: np.ndarray, model_config: ModelConfig
    ) -> Tuple[Any, Pipeline]:
        """Train a model based on configuration"""
        # Create model based on algorithm
        if model_config.algorithm == "random_forest":
            if model_config.type == "classification":
                model = RandomForestClassifier(**model_config.hyperparameters)
            else:
                model = RandomForestRegressor(**model_config.hyperparameters)
        elif model_config.algorithm == "logistic_regression":
            model = LogisticRegression(**model_config.hyperparameters)
        elif model_config.algorithm == "linear_regression":
            model = LinearRegression(**model_config.hyperparameters)
        elif model_config.algorithm == "svm":
            if model_config.type == "classification":
                model = SVC(**model_config.hyperparameters)
            else:
                model = SVR(**model_config.hyperparameters)
        elif model_config.algorithm == "neural_network":
            if model_config.type == "classification":
                model = MLPClassifier(**model_config.hyperparameters)
            else:
                model = MLPRegressor(**model_config.hyperparameters)
        else:
            raise ValueError(f"Unsupported algorithm: {model_config.algorithm}")

        # Create pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

        # Train pipeline
        pipeline.fit(X_train, y_train)

        return pipeline.named_steps["model"], pipeline

    def _evaluate_model(
        self, model: Any, X_val: np.ndarray, y_val: np.ndarray, model_type: str
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = model.predict(X_val)

        metrics = {}

        if model_type == "classification":
            metrics["accuracy"] = accuracy_score(y_val, y_pred)
            metrics["precision"] = precision_score(y_val, y_pred, average="weighted")
            metrics["recall"] = recall_score(y_val, y_pred, average="weighted")
            metrics["f1_score"] = f1_score(y_val, y_pred, average="weighted")
        else:
            metrics["mae"] = mean_absolute_error(y_val, y_pred)
            metrics["mse"] = mean_squared_error(y_val, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["r2_score"] = r2_score(y_val, y_pred)

        # Add feature importance for tree-based models
        if hasattr(model, "feature_importances_"):
            metrics["feature_importance"] = dict(
                zip(range(len(model.feature_importances_)), model.feature_importances_)
            )

        return metrics

    def _save_model(
        self,
        model: Any,
        pipeline: Pipeline,
        model_config: ModelConfig,
        job: TrainingJob,
    ) -> str:
        """Save trained model"""
        # Create model directory
        model_dir = self.models_dir / model_config.model_id
        model_dir.mkdir(exist_ok=True)

        # Save model files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}.joblib"
        pipeline_filename = f"pipeline_{timestamp}.joblib"

        model_path = model_dir / model_filename
        pipeline_path = model_dir / pipeline_filename

        # Save model and pipeline
        joblib.dump(model, model_path)
        joblib.dump(pipeline, pipeline_path)

        return str(model_path)

    def _create_model_version(
        self,
        model_config: ModelConfig,
        job: TrainingJob,
        model_path: str,
        metrics: Dict[str, float],
    ) -> ModelVersion:
        """Create a new model version"""
        # Get next version number
        existing_versions = self.model_versions[model_config.model_id]
        version_number = len(existing_versions) + 1

        # Determine if this version should be active
        is_active = version_number == 1  # First version is active by default

        # If not first version, compare with current active version
        if not is_active and existing_versions:
            current_active = next((v for v in existing_versions if v.is_active), None)
            if current_active:
                # Compare metrics (assuming higher is better for most metrics)
                current_score = self._calculate_model_score(
                    current_active.metrics, model_config.type
                )
                new_score = self._calculate_model_score(metrics, model_config.type)
                is_active = new_score > current_score

                # Deactivate current version if new one is better
                if is_active:
                    current_active.is_active = False

        # Create version
        version = ModelVersion(
            version_id=str(uuid.uuid4()),
            model_id=model_config.model_id,
            version_number=version_number,
            training_job_id=job.job_id,
            model_path=model_path,
            metrics=metrics,
            hyperparameters=model_config.hyperparameters,
            feature_importance=metrics.get("feature_importance", {}),
            created_at=datetime.now(),
            is_active=is_active,
            metadata={},
        )

        # Add to versions
        self.model_versions[model_config.model_id].append(version)

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        INSERT INTO model_versions 
                        (version_id, model_id, version_number, training_job_id, model_path, metrics, hyperparameters, feature_importance, created_at, is_active, metadata)
                        VALUES (:version_id, :model_id, :version_number, :training_job_id, :model_path, :metrics, :hyperparameters, :feature_importance, :created_at, :is_active, :metadata)
                    """
                        ),
                        asdict(version),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing model version: {e}")

        return version

    def _calculate_model_score(
        self, metrics: Dict[str, float], model_type: str
    ) -> float:
        """Calculate a single score for model comparison"""
        if model_type == "classification":
            # Weighted combination of classification metrics
            return (
                metrics.get("accuracy", 0) * 0.4
                + metrics.get("precision", 0) * 0.3
                + metrics.get("recall", 0) * 0.3
            )
        else:
            # For regression, use R² score (higher is better)
            return metrics.get("r2_score", 0)

    def _load_model(self, version: ModelVersion):
        """Load a model version into memory"""
        try:
            model_path = Path(version.model_path)
            if model_path.exists():
                model = joblib.load(model_path)
                self.active_models[version.model_id] = model
                print(
                    f"Loaded model {version.model_id} version {version.version_number}"
                )
            else:
                print(f"Model file not found: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def _update_training_job(self, job: TrainingJob):
        """Update training job in database"""
        job.updated_at = datetime.now()

        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        UPDATE training_jobs 
                        SET status = :status, progress = :progress, current_step = :current_step,
                            start_time = :start_time, end_time = :end_time, metrics = :metrics,
                            model_path = :model_path, updated_at = :updated_at
                        WHERE job_id = :job_id
                    """
                        ),
                        asdict(job),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error updating training job: {e}")

    def get_training_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training job status"""
        if job_id not in self.training_jobs:
            return None

        job = self.training_jobs[job_id]
        return asdict(job)

    def predict(self, model_id: str, features: np.ndarray) -> np.ndarray:
        """Make predictions using an active model"""
        if model_id not in self.active_models:
            raise ValueError(f"Model {model_id} not loaded")

        model = self.active_models[model_id]

        # Apply scaling if available
        if model_id in self.scalers:
            features = self.scalers[model_id].transform(features)

        return model.predict(features)

    def create_experiment(
        self,
        name: str,
        description: str,
        model_configs: List[str],
        hyperparameter_ranges: Dict[str, List[Any]],
        optimization_metric: str,
        max_trials: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create a new ML experiment for hyperparameter optimization"""
        experiment = Experiment(
            experiment_id=str(uuid.uuid4()),
            name=name,
            description=description,
            model_configs=model_configs,
            hyperparameter_ranges=hyperparameter_ranges,
            optimization_metric=optimization_metric,
            max_trials=max_trials,
            status="pending",
            best_model_id=None,
            results={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {},
        )

        self.experiments[experiment.experiment_id] = experiment

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        INSERT INTO experiments 
                        (experiment_id, name, description, model_configs, hyperparameter_ranges, optimization_metric, max_trials, status, best_model_id, results, created_at, updated_at, metadata)
                        VALUES (:experiment_id, :name, :description, :model_configs, :hyperparameter_ranges, :optimization_metric, :max_trials, :status, :best_model_id, :results, :created_at, :updated_at, :metadata)
                    """
                        ),
                        asdict(experiment),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing experiment: {e}")

        return experiment

    def run_experiment(self, experiment_id: str, training_data: pd.DataFrame) -> str:
        """Run a hyperparameter optimization experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        experiment.status = "running"
        experiment.updated_at = datetime.now()

        # Start optimization in background
        threading.Thread(
            target=self._run_optimization,
            args=(experiment_id, training_data),
            daemon=True,
        ).start()

        return experiment_id

    def _run_optimization(self, experiment_id: str, training_data: pd.DataFrame):
        """Run hyperparameter optimization"""
        experiment = self.experiments[experiment_id]

        try:
            # Prepare data
            X = training_data[
                experiment.model_configs[0]
            ]  # Use first model config for now
            y = training_data["target"]  # Assuming target column is 'target'

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            def objective(trial):
                # Sample hyperparameters
                params = {}
                for param, values in experiment.hyperparameter_ranges.items():
                    if isinstance(values[0], int):
                        params[param] = trial.suggest_int(param, values[0], values[1])
                    elif isinstance(values[0], float):
                        params[param] = trial.suggest_float(param, values[0], values[1])
                    else:
                        params[param] = trial.suggest_categorical(param, values)

                # Train model with sampled parameters
                model = RandomForestRegressor(**params, random_state=42)
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)

                return score

            # Run optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=experiment.max_trials)

            # Store results
            experiment.results = {
                "best_params": study.best_params,
                "best_score": study.best_value,
                "trials": len(study.trials),
            }
            experiment.status = "completed"
            experiment.updated_at = datetime.now()

            # Create best model config
            best_model_config = self.create_model_config(
                name=f"{experiment.name}_optimized",
                model_type="regression",
                algorithm="random_forest",
                feature_columns=list(X.columns),
                target_column="target",
                hyperparameters=study.best_params,
                metadata={"experiment_id": experiment_id},
            )

            experiment.best_model_id = best_model_config.model_id

        except Exception as e:
            experiment.status = "failed"
            experiment.results = {"error": str(e)}
            experiment.updated_at = datetime.now()
            print(f"Experiment {experiment_id} failed: {e}")

        # Update in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        UPDATE experiments 
                        SET status = :status, best_model_id = :best_model_id, results = :results, updated_at = :updated_at
                        WHERE experiment_id = :experiment_id
                    """
                        ),
                        asdict(experiment),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error updating experiment: {e}")

    def get_model_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a model"""
        if model_id not in self.model_versions:
            return []

        return [asdict(version) for version in self.model_versions[model_id]]

    def activate_model_version(self, model_id: str, version_number: int) -> bool:
        """Activate a specific model version"""
        if model_id not in self.model_versions:
            return False

        versions = self.model_versions[model_id]
        target_version = next(
            (v for v in versions if v.version_number == version_number), None
        )

        if not target_version:
            return False

        # Deactivate current active version
        for version in versions:
            if version.is_active:
                version.is_active = False

        # Activate target version
        target_version.is_active = True

        # Load the model
        self._load_model(target_version)

        return True

    def export_model(self, model_id: str, version_number: Optional[int] = None) -> str:
        """Export a model to a file"""
        if model_id not in self.model_versions:
            raise ValueError(f"Model {model_id} not found")

        versions = self.model_versions[model_id]

        if version_number is None:
            # Export latest version
            version = versions[-1]
        else:
            version = next(
                (v for v in versions if v.version_number == version_number), None
            )
            if not version:
                raise ValueError(f"Version {version_number} not found")

        # Create export directory
        export_dir = Path("exports") / "models"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"{model_id}_v{version.version_number}_{timestamp}.zip"
        export_path = export_dir / export_filename

        # Create zip file with model and metadata
        import zipfile

        with zipfile.ZipFile(export_path, "w") as zipf:
            # Add model file
            zipf.write(version.model_path, "model.joblib")

            # Add metadata
            metadata = {
                "model_id": model_id,
                "version_number": version.version_number,
                "metrics": version.metrics,
                "hyperparameters": version.hyperparameters,
                "created_at": version.created_at.isoformat(),
            }

            import json

            zipf.writestr("metadata.json", json.dumps(metadata, indent=2))

        return str(export_path)


def main():
    """Main function to run the ML pipeline system"""
    # Initialize ML pipeline
    pipeline = MLPipeline()

    # Create sample training data
    np.random.seed(42)
    n_samples = 1000

    # Symbol recognition data
    symbol_data = pd.DataFrame(
        {
            "area": np.random.uniform(10, 1000, n_samples),
            "perimeter": np.random.uniform(20, 200, n_samples),
            "circularity": np.random.uniform(0.1, 1.0, n_samples),
            "aspect_ratio": np.random.uniform(0.5, 2.0, n_samples),
            "symbol_type": np.random.choice(
                ["circle", "square", "triangle"], n_samples
            ),
        }
    )

    # Submit training job
    job_id = pipeline.submit_training_job(
        model_config_id="symbol_recognition",
        training_data=symbol_data,
        validation_split=0.2,
    )

    print(f"Submitted training job: {job_id}")

    # Wait for training to complete
    while True:
        status = pipeline.get_training_job_status(job_id)
        if status and status["status"] in ["completed", "failed"]:
            print(f"Training completed with status: {status['status']}")
            if status["status"] == "completed":
                print(f"Model metrics: {status['metrics']}")
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
