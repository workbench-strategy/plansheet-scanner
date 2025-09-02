"""
Intelligent Quality Assurance System for Plansheet Scanner

This module provides intelligent quality assurance capabilities including automated quality control,
intelligent validation, adaptive learning, and quality improvement recommendations.

Features:
- Automated quality control and validation
- Intelligent error detection and classification
- Adaptive learning for quality improvement
- Quality metrics and scoring
- Automated quality reports
- Quality improvement recommendations
"""

import asyncio
import json
import os
import re
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
import pandas as pd

# ML imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Database imports
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


@dataclass
class QualityMetric:
    """Quality metric data structure"""

    metric_id: str
    name: str
    value: float
    threshold: float
    weight: float
    category: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class QualityIssue:
    """Quality issue data structure"""

    issue_id: str
    type: str  # 'error', 'warning', 'info'
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str
    description: str
    location: Optional[str]
    detected_at: datetime
    resolved_at: Optional[datetime]
    resolution: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class QualityScore:
    """Quality score data structure"""

    score_id: str
    overall_score: float
    category_scores: Dict[str, float]
    weighted_score: float
    issues_count: int
    critical_issues: int
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class ValidationRule:
    """Validation rule data structure"""

    rule_id: str
    name: str
    description: str
    category: str
    rule_type: str  # 'regex', 'range', 'custom', 'ml'
    parameters: Dict[str, Any]
    weight: float
    is_active: bool
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class QualityReport:
    """Quality report data structure"""

    report_id: str
    job_id: str
    overall_score: float
    category_scores: Dict[str, float]
    issues: List[QualityIssue]
    recommendations: List[str]
    generated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class QualityModel:
    """Quality prediction model data structure"""

    model_id: str
    name: str
    type: str  # 'classification', 'regression'
    features: List[str]
    target: str
    accuracy: float
    trained_at: datetime
    model_path: str
    metadata: Dict[str, Any]


class IntelligentQA:
    """Intelligent Quality Assurance System"""

    def __init__(
        self,
        models_dir: str = "quality_models",
        db_config: Optional[Dict[str, Any]] = None,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Storage
        self.quality_metrics: Dict[str, List[QualityMetric]] = defaultdict(list)
        self.quality_issues: Dict[str, List[QualityIssue]] = defaultdict(list)
        self.quality_scores: Dict[str, List[QualityScore]] = defaultdict(list)
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.quality_models: Dict[str, QualityModel] = {}

        # Active models
        self.active_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "acceptable": 0.7,
            "poor": 0.6,
        }

        # Category weights
        self.category_weights = {
            "accuracy": 0.3,
            "completeness": 0.25,
            "consistency": 0.2,
            "timeliness": 0.15,
            "reliability": 0.1,
        }

        # Database connection
        self.db_engine = None
        if db_config:
            self._setup_database(db_config)

        # Initialize default validation rules
        self._setup_default_rules()

        # Start quality monitoring
        self.start_quality_monitoring()

    def _setup_database(self, db_config: Dict[str, Any]):
        """Setup database connection for quality assurance storage"""
        try:
            connection_string = (
                f"postgresql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            self.db_engine = create_engine(connection_string)

            # Create quality assurance tables
            self._create_qa_tables()

        except Exception as e:
            print(f"Warning: Database setup failed: {e}")

    def _create_qa_tables(self):
        """Create quality assurance tables in database"""
        if not self.db_engine:
            return

        try:
            with self.db_engine.connect() as conn:
                # Quality metrics table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        metric_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        threshold DOUBLE PRECISION NOT NULL,
                        weight DOUBLE PRECISION NOT NULL,
                        category VARCHAR(100) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Quality issues table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS quality_issues (
                        issue_id VARCHAR(255) PRIMARY KEY,
                        type VARCHAR(50) NOT NULL,
                        severity VARCHAR(50) NOT NULL,
                        category VARCHAR(100) NOT NULL,
                        description TEXT NOT NULL,
                        location VARCHAR(500),
                        detected_at TIMESTAMP NOT NULL,
                        resolved_at TIMESTAMP,
                        resolution TEXT,
                        metadata JSONB
                    )
                """
                    )
                )

                # Quality scores table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS quality_scores (
                        score_id VARCHAR(255) PRIMARY KEY,
                        overall_score DOUBLE PRECISION NOT NULL,
                        category_scores JSONB NOT NULL,
                        weighted_score DOUBLE PRECISION NOT NULL,
                        issues_count INTEGER NOT NULL,
                        critical_issues INTEGER NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Validation rules table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS validation_rules (
                        rule_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        category VARCHAR(100) NOT NULL,
                        rule_type VARCHAR(50) NOT NULL,
                        parameters JSONB NOT NULL,
                        weight DOUBLE PRECISION NOT NULL,
                        is_active BOOLEAN NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Quality models table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS quality_models (
                        model_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        type VARCHAR(50) NOT NULL,
                        features TEXT[] NOT NULL,
                        target VARCHAR(255) NOT NULL,
                        accuracy DOUBLE PRECISION NOT NULL,
                        trained_at TIMESTAMP NOT NULL,
                        model_path VARCHAR(500) NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                conn.commit()

        except Exception as e:
            print(f"Warning: Failed to create QA tables: {e}")

    def _setup_default_rules(self):
        """Setup default validation rules"""
        # File format validation
        file_format_rule = ValidationRule(
            rule_id="file_format_validation",
            name="File Format Validation",
            description="Validates that files are in supported formats",
            category="format",
            rule_type="regex",
            parameters={
                "pattern": r"\.(pdf|jpg|jpeg|png|tiff|tif)$",
                "case_sensitive": False,
            },
            weight=0.2,
            is_active=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )
        self.validation_rules["file_format_validation"] = file_format_rule

        # File size validation
        file_size_rule = ValidationRule(
            rule_id="file_size_validation",
            name="File Size Validation",
            description="Validates that files are within acceptable size limits",
            category="size",
            rule_type="range",
            parameters={
                "min_size": 1024,  # 1KB
                "max_size": 100 * 1024 * 1024,  # 100MB
            },
            weight=0.15,
            is_active=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )
        self.validation_rules["file_size_validation"] = file_size_rule

        # Image resolution validation
        resolution_rule = ValidationRule(
            rule_id="resolution_validation",
            name="Image Resolution Validation",
            description="Validates that images meet minimum resolution requirements",
            category="quality",
            rule_type="range",
            parameters={"min_width": 800, "min_height": 600, "min_dpi": 150},
            weight=0.25,
            is_active=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )
        self.validation_rules["resolution_validation"] = resolution_rule

        # Content validation
        content_rule = ValidationRule(
            rule_id="content_validation",
            name="Content Validation",
            description="Validates that content meets basic requirements",
            category="content",
            rule_type="custom",
            parameters={
                "min_text_length": 100,
                "required_elements": ["title", "scale", "legend"],
                "forbidden_patterns": [r"confidential", r"secret"],
            },
            weight=0.3,
            is_active=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )
        self.validation_rules["content_validation"] = content_rule

    def start_quality_monitoring(self):
        """Start quality monitoring thread"""
        self.monitoring_thread = threading.Thread(
            target=self._quality_monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

    def _quality_monitoring_loop(self):
        """Quality monitoring loop"""
        while True:
            try:
                # Monitor quality metrics
                self._monitor_quality_metrics()

                # Update quality scores
                self._update_quality_scores()

                # Generate quality insights
                self._generate_quality_insights()

                # Wait for next cycle
                time.sleep(60)  # Check every minute

            except Exception as e:
                print(f"Error in quality monitoring: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def add_quality_metric(
        self,
        name: str,
        value: float,
        category: str,
        threshold: Optional[float] = None,
        weight: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QualityMetric:
        """Add a quality metric"""
        # Use default threshold and weight if not provided
        if threshold is None:
            threshold = self._get_default_threshold(name, category)
        if weight is None:
            weight = self.category_weights.get(category, 0.1)

        metric = QualityMetric(
            metric_id=str(uuid.uuid4()),
            name=name,
            value=value,
            threshold=threshold,
            weight=weight,
            category=category,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        # Store in memory
        self.quality_metrics[name].append(metric)

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        INSERT INTO quality_metrics
                        (metric_id, name, value, threshold, weight, category, timestamp, metadata)
                        VALUES (:metric_id, :name, :value, :threshold, :weight, :category, :timestamp, :metadata)
                    """
                        ),
                        asdict(metric),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing quality metric: {e}")

        return metric

    def _get_default_threshold(self, name: str, category: str) -> float:
        """Get default threshold for a metric"""
        default_thresholds = {
            "accuracy": 0.8,
            "completeness": 0.9,
            "consistency": 0.85,
            "timeliness": 300.0,  # seconds
            "reliability": 0.95,
        }
        return default_thresholds.get(category, 0.8)

    def validate_file(
        self, file_path: str, file_metadata: Optional[Dict[str, Any]] = None
    ) -> List[QualityIssue]:
        """Validate a file using all active validation rules"""
        issues = []
        file_path = Path(file_path)

        if not file_path.exists():
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                type="error",
                severity="critical",
                category="file",
                description=f"File not found: {file_path}",
                location=str(file_path),
                detected_at=datetime.now(),
                resolved_at=None,
                resolution=None,
                metadata={},
            )
            issues.append(issue)
            return issues

        # Apply validation rules
        for rule_id, rule in self.validation_rules.items():
            if not rule.is_active:
                continue

            try:
                if rule.rule_type == "regex":
                    issue = self._validate_regex_rule(file_path, rule)
                elif rule.rule_type == "range":
                    issue = self._validate_range_rule(file_path, rule, file_metadata)
                elif rule.rule_type == "custom":
                    issue = self._validate_custom_rule(file_path, rule, file_metadata)
                else:
                    continue

                if issue:
                    issues.append(issue)

            except Exception as e:
                print(f"Error applying rule {rule_id}: {e}")

        return issues

    def _validate_regex_rule(
        self, file_path: Path, rule: ValidationRule
    ) -> Optional[QualityIssue]:
        """Validate using regex rule"""
        pattern = rule.parameters.get("pattern")
        case_sensitive = rule.parameters.get("case_sensitive", True)

        if not pattern:
            return None

        flags = 0 if case_sensitive else re.IGNORECASE
        match = re.search(pattern, str(file_path), flags)

        if not match:
            return QualityIssue(
                issue_id=str(uuid.uuid4()),
                type="error",
                severity="high",
                category=rule.category,
                description=f"File does not match required pattern: {pattern}",
                location=str(file_path),
                detected_at=datetime.now(),
                resolved_at=None,
                resolution=None,
                metadata={"rule_id": rule.rule_id, "pattern": pattern},
            )

        return None

    def _validate_range_rule(
        self,
        file_path: Path,
        rule: ValidationRule,
        file_metadata: Optional[Dict[str, Any]],
    ) -> Optional[QualityIssue]:
        """Validate using range rule"""
        if rule.category == "size":
            file_size = file_path.stat().st_size
            min_size = rule.parameters.get("min_size", 0)
            max_size = rule.parameters.get("max_size", float("inf"))

            if file_size < min_size or file_size > max_size:
                return QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    type="warning",
                    severity="medium",
                    category=rule.category,
                    description=f"File size {file_size} is outside acceptable range [{min_size}, {max_size}]",
                    location=str(file_path),
                    detected_at=datetime.now(),
                    resolved_at=None,
                    resolution=None,
                    metadata={
                        "rule_id": rule.rule_id,
                        "file_size": file_size,
                        "min_size": min_size,
                        "max_size": max_size,
                    },
                )

        elif rule.category == "quality" and file_metadata:
            # Image resolution validation
            width = file_metadata.get("width", 0)
            height = file_metadata.get("height", 0)
            dpi = file_metadata.get("dpi", 0)

            min_width = rule.parameters.get("min_width", 0)
            min_height = rule.parameters.get("min_height", 0)
            min_dpi = rule.parameters.get("min_dpi", 0)

            if width < min_width or height < min_height or dpi < min_dpi:
                return QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    type="warning",
                    severity="medium",
                    category=rule.category,
                    description=f"Image resolution ({width}x{height}, {dpi} DPI) below minimum requirements",
                    location=str(file_path),
                    detected_at=datetime.now(),
                    resolved_at=None,
                    resolution=None,
                    metadata={
                        "rule_id": rule.rule_id,
                        "width": width,
                        "height": height,
                        "dpi": dpi,
                    },
                )

        return None

    def _validate_custom_rule(
        self,
        file_path: Path,
        rule: ValidationRule,
        file_metadata: Optional[Dict[str, Any]],
    ) -> Optional[QualityIssue]:
        """Validate using custom rule"""
        if rule.category == "content" and file_metadata:
            content = file_metadata.get("content", "")
            min_length = rule.parameters.get("min_text_length", 0)
            required_elements = rule.parameters.get("required_elements", [])
            forbidden_patterns = rule.parameters.get("forbidden_patterns", [])

            issues = []

            # Check minimum text length
            if len(content) < min_length:
                issues.append(
                    f"Content length ({len(content)}) below minimum ({min_length})"
                )

            # Check required elements
            for element in required_elements:
                if element.lower() not in content.lower():
                    issues.append(f"Required element '{element}' not found")

            # Check forbidden patterns
            for pattern in forbidden_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append(f"Forbidden pattern '{pattern}' found")

            if issues:
                return QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    type="warning",
                    severity="medium",
                    category=rule.category,
                    description=f"Content validation failed: {'; '.join(issues)}",
                    location=str(file_path),
                    detected_at=datetime.now(),
                    resolved_at=None,
                    resolution=None,
                    metadata={"rule_id": rule.rule_id, "issues": issues},
                )

        return None

    def calculate_quality_score(
        self, job_id: str, metrics: Optional[Dict[str, float]] = None
    ) -> QualityScore:
        """Calculate overall quality score"""
        if metrics is None:
            # Use recent metrics
            metrics = {}
            for metric_name, metric_list in self.quality_metrics.items():
                if metric_list:
                    metrics[metric_name] = metric_list[-1].value

        if not metrics:
            return QualityScore(
                score_id=str(uuid.uuid4()),
                overall_score=0.0,
                category_scores={},
                weighted_score=0.0,
                issues_count=0,
                critical_issues=0,
                timestamp=datetime.now(),
                metadata={},
            )

        # Calculate category scores
        category_scores = defaultdict(list)
        for metric_name, value in metrics.items():
            # Find metric info
            metric_info = None
            for metric_list in self.quality_metrics.values():
                for metric in metric_list:
                    if metric.name == metric_name:
                        metric_info = metric
                        break
                if metric_info:
                    break

            if metric_info:
                category = metric_info.category
                threshold = metric_info.threshold

                # Calculate normalized score (0-1)
                if threshold > 0:
                    normalized_score = min(value / threshold, 1.0)
                else:
                    normalized_score = 1.0 if value >= 0 else 0.0

                category_scores[category].append(normalized_score)

        # Calculate average scores per category
        final_category_scores = {}
        for category, scores in category_scores.items():
            final_category_scores[category] = np.mean(scores)

        # Calculate weighted overall score
        weighted_score = 0.0
        total_weight = 0.0

        for category, score in final_category_scores.items():
            weight = self.category_weights.get(category, 0.1)
            weighted_score += score * weight
            total_weight += weight

        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Count issues
        issues_count = len(self.quality_issues.get(job_id, []))
        critical_issues = len(
            [
                issue
                for issue in self.quality_issues.get(job_id, [])
                if issue.severity == "critical"
            ]
        )

        quality_score = QualityScore(
            score_id=str(uuid.uuid4()),
            overall_score=overall_score,
            category_scores=dict(final_category_scores),
            weighted_score=weighted_score,
            issues_count=issues_count,
            critical_issues=critical_issues,
            timestamp=datetime.now(),
            metadata={"job_id": job_id},
        )

        # Store quality score
        self.quality_scores[job_id].append(quality_score)

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        INSERT INTO quality_scores
                        (score_id, overall_score, category_scores, weighted_score, issues_count, critical_issues, timestamp, metadata)
                        VALUES (:score_id, :overall_score, :category_scores, :weighted_score, :issues_count, :critical_issues, :timestamp, :metadata)
                    """
                        ),
                        asdict(quality_score),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing quality score: {e}")

        return quality_score

    def create_validation_rule(
        self,
        name: str,
        description: str,
        category: str,
        rule_type: str,
        parameters: Dict[str, Any],
        weight: float = 0.1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationRule:
        """Create a new validation rule"""
        rule = ValidationRule(
            rule_id=str(uuid.uuid4()),
            name=name,
            description=description,
            category=category,
            rule_type=rule_type,
            parameters=parameters,
            weight=weight,
            is_active=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {},
        )

        self.validation_rules[rule.rule_id] = rule

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        INSERT INTO validation_rules
                        (rule_id, name, description, category, rule_type, parameters, weight, is_active, created_at, updated_at, metadata)
                        VALUES (:rule_id, :name, :description, :category, :rule_type, :parameters, :weight, :is_active, :created_at, :updated_at, :metadata)
                    """
                        ),
                        asdict(rule),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing validation rule: {e}")

        return rule

    def train_quality_model(
        self,
        model_name: str,
        training_data: pd.DataFrame,
        target_column: str,
        model_type: str = "classification",
    ) -> QualityModel:
        """Train a quality prediction model"""
        try:
            # Prepare features and target
            feature_columns = [
                col for col in training_data.columns if col != target_column
            ]
            X = training_data[feature_columns]
            y = training_data[target_column]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            if model_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )  # Default to classification

            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = model.score(X_test_scaled, y_test)

            # Save model
            model_path = (
                self.models_dir
                / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
            joblib.dump(model, model_path)

            # Save scaler
            scaler_path = (
                self.models_dir
                / f"{model_name}_scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
            joblib.dump(scaler, scaler_path)

            # Create quality model record
            quality_model = QualityModel(
                model_id=str(uuid.uuid4()),
                name=model_name,
                type=model_type,
                features=feature_columns,
                target=target_column,
                accuracy=accuracy,
                trained_at=datetime.now(),
                model_path=str(model_path),
                metadata={"scaler_path": str(scaler_path)},
            )

            self.quality_models[quality_model.model_id] = quality_model
            self.active_models[model_name] = model
            self.scalers[model_name] = scaler

            # Store in database
            if self.db_engine:
                try:
                    with self.db_engine.connect() as conn:
                        conn.execute(
                            text(
                                """
                            INSERT INTO quality_models
                            (model_id, name, type, features, target, accuracy, trained_at, model_path, metadata)
                            VALUES (:model_id, :name, :type, :features, :target, :accuracy, :trained_at, :model_path, :metadata)
                        """
                            ),
                            asdict(quality_model),
                        )
                        conn.commit()
                except Exception as e:
                    print(f"Error storing quality model: {e}")

            return quality_model

        except Exception as e:
            print(f"Error training quality model: {e}")
            raise

    def predict_quality(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """Predict quality using a trained model"""
        if model_name not in self.active_models:
            raise ValueError(f"Model {model_name} not found")

        model = self.active_models[model_name]
        scaler = self.scalers.get(model_name)

        if scaler:
            features = scaler.transform(features)

        return model.predict(features)

    def generate_quality_report(self, job_id: str) -> QualityReport:
        """Generate a comprehensive quality report"""
        # Get quality score
        quality_score = self.calculate_quality_score(job_id)

        # Get issues
        issues = self.quality_issues.get(job_id, [])

        # Generate recommendations
        recommendations = self._generate_quality_recommendations(quality_score, issues)

        report = QualityReport(
            report_id=str(uuid.uuid4()),
            job_id=job_id,
            overall_score=quality_score.overall_score,
            category_scores=quality_score.category_scores,
            issues=issues,
            recommendations=recommendations,
            generated_at=datetime.now(),
            metadata={},
        )

        return report

    def _generate_quality_recommendations(
        self, quality_score: QualityScore, issues: List[QualityIssue]
    ) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []

        # Overall score recommendations
        if quality_score.overall_score < self.quality_thresholds["acceptable"]:
            recommendations.append(
                "Overall quality score is below acceptable threshold. Review processing parameters and data quality."
            )

        # Category-specific recommendations
        for category, score in quality_score.category_scores.items():
            if score < 0.7:
                if category == "accuracy":
                    recommendations.append(
                        "Improve accuracy by reviewing detection algorithms and training data quality."
                    )
                elif category == "completeness":
                    recommendations.append(
                        "Enhance completeness by ensuring all required elements are captured."
                    )
                elif category == "consistency":
                    recommendations.append(
                        "Improve consistency by standardizing processing workflows."
                    )
                elif category == "timeliness":
                    recommendations.append(
                        "Optimize processing time by reviewing performance bottlenecks."
                    )
                elif category == "reliability":
                    recommendations.append(
                        "Enhance reliability by implementing better error handling and validation."
                    )

        # Issue-specific recommendations
        for issue in issues:
            if issue.severity in ["high", "critical"]:
                if issue.category == "format":
                    recommendations.append(
                        "Review file format requirements and implement better format validation."
                    )
                elif issue.category == "size":
                    recommendations.append(
                        "Optimize file sizes or implement compression strategies."
                    )
                elif issue.category == "quality":
                    recommendations.append(
                        "Improve input data quality or enhance processing algorithms."
                    )
                elif issue.category == "content":
                    recommendations.append(
                        "Implement content validation and quality checks."
                    )

        return list(set(recommendations))  # Remove duplicates

    def _monitor_quality_metrics(self):
        """Monitor quality metrics for anomalies"""
        for metric_name, metrics in self.quality_metrics.items():
            if len(metrics) < 10:
                continue

            # Get recent values
            recent_values = [m.value for m in metrics[-10:]]

            # Simple anomaly detection (basic statistical approach)
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values)

            if std_val > 0:
                latest_value = recent_values[-1]
                z_score = abs(latest_value - mean_val) / std_val

                if z_score > 2.0:  # Anomaly threshold
                    issue = QualityIssue(
                        issue_id=str(uuid.uuid4()),
                        type="warning",
                        severity="medium",
                        category="monitoring",
                        description=f"Anomaly detected in {metric_name}: value {latest_value} (z-score: {z_score:.2f})",
                        location=None,
                        detected_at=datetime.now(),
                        resolved_at=None,
                        resolution=None,
                        metadata={"metric_name": metric_name, "z_score": z_score},
                    )

                    # Store issue
                    self.quality_issues["monitoring"].append(issue)

    def _update_quality_scores(self):
        """Update quality scores for recent jobs"""
        # This would typically update scores for recent processing jobs
        # For now, we'll just ensure the scoring system is working
        pass

    def _generate_quality_insights(self):
        """Generate quality insights and trends"""
        # This would analyze quality trends and generate insights
        # For now, we'll just ensure the system is monitoring
        pass

    def get_quality_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get quality trends for a specific metric"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Get recent metrics
        recent_metrics = [
            m for m in self.quality_metrics[metric_name] if m.timestamp >= cutoff_time
        ]

        if len(recent_metrics) < 2:
            return {"trend": "insufficient_data", "change_rate": 0.0}

        # Calculate trend
        values = [m.value for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]

        # Simple linear trend
        time_deltas = [(t - timestamps[0]).total_seconds() for t in timestamps]

        if len(time_deltas) > 1:
            coeffs = np.polyfit(time_deltas, values, 1)
            slope = coeffs[0]

            # Calculate change rate
            change_rate = (
                ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
            )

            if slope > 0.001:
                trend = "improving"
            elif slope < -0.001:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
            change_rate = 0.0

        return {
            "trend": trend,
            "change_rate": change_rate,
            "current_value": values[-1],
            "average_value": np.mean(values),
            "data_points": len(values),
        }

    def export_quality_data(
        self, format: str = "json", time_period: str = "24h"
    ) -> str:
        """Export quality data"""
        try:
            # Parse time period
            if time_period.endswith("h"):
                hours = int(time_period[:-1])
            else:
                hours = 24

            cutoff_time = datetime.now() - timedelta(hours=hours)

            export_data = {
                "export_id": str(uuid.uuid4()),
                "format": format,
                "time_period": time_period,
                "exported_at": datetime.now().isoformat(),
                "metrics": {},
                "scores": {},
                "issues": {},
                "trends": {},
            }

            # Export metrics
            for metric_name, metrics in self.quality_metrics.items():
                recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                export_data["metrics"][metric_name] = [
                    asdict(m) for m in recent_metrics
                ]

            # Export scores
            for job_id, scores in self.quality_scores.items():
                recent_scores = [s for s in scores if s.timestamp >= cutoff_time]
                export_data["scores"][job_id] = [asdict(s) for s in recent_scores]

            # Export issues
            for job_id, issues in self.quality_issues.items():
                recent_issues = [i for i in issues if i.detected_at >= cutoff_time]
                export_data["issues"][job_id] = [asdict(i) for i in recent_issues]

            # Export trends
            for metric_name in self.quality_metrics.keys():
                export_data["trends"][metric_name] = self.get_quality_trends(
                    metric_name, hours
                )

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_export_{timestamp}.{format}"

            if format == "json":
                with open(filename, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)

            return filename

        except Exception as e:
            print(f"Error exporting quality data: {e}")
            return ""


def main():
    """Main function to run the intelligent quality assurance system"""
    # Initialize QA system
    qa = IntelligentQA()

    # Add some sample quality metrics
    qa.add_quality_metric("accuracy", 0.85, "accuracy")
    qa.add_quality_metric("completeness", 0.92, "completeness")
    qa.add_quality_metric("processing_time", 45.2, "timeliness")

    # Validate a sample file
    sample_file = "sample_plan.pdf"
    issues = qa.validate_file(
        sample_file,
        {
            "width": 1200,
            "height": 800,
            "dpi": 200,
            "content": "Sample plan sheet with title and scale information",
        },
    )

    print(f"Validation issues found: {len(issues)}")
    for issue in issues:
        print(f"- {issue.severity}: {issue.description}")

    # Calculate quality score
    score = qa.calculate_quality_score("sample_job")
    print(f"Quality score: {score.overall_score:.3f}")

    # Generate quality report
    report = qa.generate_quality_report("sample_job")
    print(
        f"Quality report generated with {len(report.recommendations)} recommendations"
    )


if __name__ == "__main__":
    main()
