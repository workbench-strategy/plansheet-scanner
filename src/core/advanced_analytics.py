"""
Advanced Analytics Dashboard System for Plansheet Scanner

This module provides comprehensive analytics capabilities including real-time insights,
predictive modeling, trend analysis, and automated insights generation.

Features:
- Real-time analytics processing and visualization
- Predictive modeling for processing times and quality
- Trend analysis and pattern detection
- Anomaly detection and alerting
- Custom dashboard generation
- Automated report generation
"""

import asyncio
import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Analytics and ML imports
import plotly.graph_objects as go
import psutil
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Database and API imports
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


@dataclass
class AnalyticsMetric:
    """Analytics metric data structure"""

    metric_id: str
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class PredictionResult:
    """Prediction result data structure"""

    prediction_id: str
    model_name: str
    target: str
    predicted_value: float
    confidence: float
    timestamp: datetime
    features: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class TrendAnalysis:
    """Trend analysis result"""

    trend_id: str
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0.0 to 1.0
    change_rate: float
    period: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""

    anomaly_id: str
    metric_name: str
    anomaly_score: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    description: str
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class Insight:
    """Automated insight data structure"""

    insight_id: str
    title: str
    description: str
    category: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    metrics: List[str]
    recommendations: List[str]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class DashboardConfig:
    """Dashboard configuration"""

    dashboard_id: str
    name: str
    description: str
    layout: Dict[str, Any]
    metrics: List[str]
    refresh_interval: int  # seconds
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


class AdvancedAnalytics:
    """Advanced analytics engine for real-time insights and predictions"""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.predictions: Dict[str, List[PredictionResult]] = defaultdict(list)
        self.trends: Dict[str, List[TrendAnalysis]] = defaultdict(list)
        self.anomalies: Dict[str, List[AnomalyDetection]] = defaultdict(list)
        self.insights: List[Insight] = []
        self.dashboards: Dict[str, DashboardConfig] = {}

        # ML models
        self.prediction_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Real-time processing
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False

        # Database connection
        self.db_engine = None
        if db_config:
            self._setup_database(db_config)

        # Initialize default dashboards
        self._setup_default_dashboards()

        # Start real-time processing
        self.start_real_time_processing()

    def _setup_database(self, db_config: Dict[str, Any]):
        """Setup database connection for analytics storage"""
        try:
            connection_string = (
                f"postgresql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            self.db_engine = create_engine(connection_string)

            # Create analytics tables if they don't exist
            self._create_analytics_tables()

        except Exception as e:
            print(f"Warning: Database setup failed: {e}")

    def _create_analytics_tables(self):
        """Create analytics tables in database"""
        if not self.db_engine:
            return

        try:
            with self.db_engine.connect() as conn:
                # Analytics metrics table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS analytics_metrics (
                        metric_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        unit VARCHAR(50),
                        timestamp TIMESTAMP NOT NULL,
                        category VARCHAR(100),
                        tags TEXT[],
                        metadata JSONB
                    )
                """
                    )
                )

                # Predictions table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS predictions (
                        prediction_id VARCHAR(255) PRIMARY KEY,
                        model_name VARCHAR(255) NOT NULL,
                        target VARCHAR(255) NOT NULL,
                        predicted_value DOUBLE PRECISION NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        features JSONB,
                        metadata JSONB
                    )
                """
                    )
                )

                # Trends table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS trends (
                        trend_id VARCHAR(255) PRIMARY KEY,
                        metric_name VARCHAR(255) NOT NULL,
                        trend_direction VARCHAR(50) NOT NULL,
                        trend_strength DOUBLE PRECISION NOT NULL,
                        change_rate DOUBLE PRECISION NOT NULL,
                        period VARCHAR(50) NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Anomalies table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS anomalies (
                        anomaly_id VARCHAR(255) PRIMARY KEY,
                        metric_name VARCHAR(255) NOT NULL,
                        anomaly_score DOUBLE PRECISION NOT NULL,
                        severity VARCHAR(50) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        description TEXT,
                        recommendations TEXT[],
                        metadata JSONB
                    )
                """
                    )
                )

                # Insights table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS insights (
                        insight_id VARCHAR(255) PRIMARY KEY,
                        title VARCHAR(255) NOT NULL,
                        description TEXT,
                        category VARCHAR(100),
                        priority VARCHAR(50),
                        timestamp TIMESTAMP NOT NULL,
                        metrics TEXT[],
                        recommendations TEXT[],
                        confidence DOUBLE PRECISION NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                conn.commit()

        except Exception as e:
            print(f"Warning: Failed to create analytics tables: {e}")

    def _setup_default_dashboards(self):
        """Setup default analytics dashboards"""
        # System Performance Dashboard
        system_dashboard = DashboardConfig(
            dashboard_id="system_performance",
            name="System Performance",
            description="Real-time system performance metrics",
            layout={
                "rows": 2,
                "cols": 2,
                "widgets": [
                    {"type": "metric", "title": "CPU Usage", "position": [0, 0]},
                    {"type": "metric", "title": "Memory Usage", "position": [0, 1]},
                    {
                        "type": "chart",
                        "title": "Processing Time Trend",
                        "position": [1, 0],
                    },
                    {"type": "chart", "title": "Throughput", "position": [1, 1]},
                ],
            },
            metrics=["cpu_usage", "memory_usage", "processing_time", "throughput"],
            refresh_interval=30,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )
        self.dashboards["system_performance"] = system_dashboard

        # Quality Analytics Dashboard
        quality_dashboard = DashboardConfig(
            dashboard_id="quality_analytics",
            name="Quality Analytics",
            description="Quality metrics and trends",
            layout={
                "rows": 2,
                "cols": 2,
                "widgets": [
                    {"type": "metric", "title": "Quality Score", "position": [0, 0]},
                    {"type": "metric", "title": "Error Rate", "position": [0, 1]},
                    {"type": "chart", "title": "Quality Trend", "position": [1, 0]},
                    {
                        "type": "chart",
                        "title": "Error Distribution",
                        "position": [1, 1],
                    },
                ],
            },
            metrics=["quality_score", "error_rate", "accuracy", "precision"],
            refresh_interval=60,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )
        self.dashboards["quality_analytics"] = quality_dashboard

    def start_real_time_processing(self):
        """Start real-time analytics processing"""
        if self.is_running:
            return

        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._real_time_processing_loop, daemon=True
        )
        self.processing_thread.start()

    def stop_real_time_processing(self):
        """Stop real-time analytics processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()

    def _real_time_processing_loop(self):
        """Real-time processing loop"""
        while self.is_running:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Update predictions
                self._update_predictions()

                # Detect anomalies
                self._detect_anomalies()

                # Generate insights
                self._generate_insights()

                # Store to database
                self._store_analytics_data()

                # Wait for next cycle
                time.sleep(30)  # 30-second intervals

            except Exception as e:
                print(f"Error in real-time processing: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_system_metrics(self):
        """Collect real-time system metrics"""
        try:
            # CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage = (disk.used / disk.total) * 100

            # Network I/O
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv

            # Add metrics
            self.add_metric("cpu_usage", cpu_usage, "%", "system")
            self.add_metric("memory_usage", memory_usage, "%", "system")
            self.add_metric("disk_usage", disk_usage, "%", "system")
            self.add_metric("network_sent", network_bytes_sent, "bytes", "system")
            self.add_metric("network_recv", network_bytes_recv, "bytes", "system")

        except Exception as e:
            print(f"Error collecting system metrics: {e}")

    def add_metric(
        self,
        name: str,
        value: float,
        unit: str,
        category: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a new analytics metric"""
        metric = AnalyticsMetric(
            metric_id=str(uuid.uuid4()),
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Store in memory
        self.metrics_history[name].append(metric)

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        INSERT INTO analytics_metrics 
                        (metric_id, name, value, unit, timestamp, category, tags, metadata)
                        VALUES (:metric_id, :name, :value, :unit, :timestamp, :category, :tags, :metadata)
                    """
                        ),
                        asdict(metric),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing metric to database: {e}")

    def get_metric_history(
        self, metric_name: str, hours: int = 24
    ) -> List[AnalyticsMetric]:
        """Get metric history for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Get from memory
        history = [
            m for m in self.metrics_history[metric_name] if m.timestamp >= cutoff_time
        ]

        # Get from database if needed
        if self.db_engine and len(history) < 100:  # If we need more data
            try:
                with self.db_engine.connect() as conn:
                    result = conn.execute(
                        text(
                            """
                        SELECT * FROM analytics_metrics 
                        WHERE name = :name AND timestamp >= :cutoff_time
                        ORDER BY timestamp DESC
                    """
                        ),
                        {"name": metric_name, "cutoff_time": cutoff_time},
                    )

                    for row in result:
                        metric = AnalyticsMetric(
                            metric_id=row.metric_id,
                            name=row.name,
                            value=row.value,
                            unit=row.unit,
                            timestamp=row.timestamp,
                            category=row.category,
                            tags=row.tags or [],
                            metadata=row.metadata or {},
                        )
                        history.append(metric)

            except Exception as e:
                print(f"Error retrieving metric history: {e}")

        return sorted(history, key=lambda x: x.timestamp)

    def train_prediction_model(self, target_metric: str, feature_metrics: List[str]):
        """Train a prediction model for the target metric"""
        try:
            # Get historical data
            target_history = self.get_metric_history(target_metric, hours=168)  # 1 week
            if len(target_history) < 50:
                print(f"Insufficient data for {target_metric}")
                return

            # Prepare features
            feature_data = {}
            for metric in feature_metrics:
                feature_history = self.get_metric_history(metric, hours=168)
                feature_data[metric] = {m.timestamp: m.value for m in feature_history}

            # Create training dataset
            X, y = [], []
            for target_point in target_history:
                features = []
                for metric in feature_metrics:
                    # Find closest feature value in time
                    closest_time = min(
                        feature_data[metric].keys(),
                        key=lambda t: abs((t - target_point.timestamp).total_seconds()),
                    )
                    if (
                        abs((closest_time - target_point.timestamp).total_seconds())
                        < 300
                    ):  # 5 minutes
                        features.append(feature_data[metric][closest_time])
                    else:
                        features.append(0.0)  # Default value

                X.append(features)
                y.append(target_point.value)

            if len(X) < 20:
                print(f"Insufficient training data for {target_metric}")
                return

            # Train model
            X = np.array(X)
            y = np.array(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store model
            self.prediction_models[target_metric] = {
                "model": model,
                "scaler": scaler,
                "features": feature_metrics,
                "mae": mae,
                "r2": r2,
                "trained_at": datetime.now(),
            }

            print(
                f"Trained prediction model for {target_metric}: MAE={mae:.3f}, R²={r2:.3f}"
            )

        except Exception as e:
            print(f"Error training prediction model: {e}")

    def predict_metric(self, target_metric: str) -> Optional[PredictionResult]:
        """Predict the next value for a metric"""
        if target_metric not in self.prediction_models:
            return None

        try:
            model_info = self.prediction_models[target_metric]
            model = model_info["model"]
            scaler = model_info["scaler"]
            features = model_info["features"]

            # Get current feature values
            feature_values = []
            for feature in features:
                history = self.get_metric_history(feature, hours=1)
                if history:
                    feature_values.append(history[-1].value)
                else:
                    feature_values.append(0.0)

            # Make prediction
            X = np.array([feature_values])
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]

            # Calculate confidence (using model's feature importances)
            confidence = np.mean(model.feature_importances_)

            # Create prediction result
            result = PredictionResult(
                prediction_id=str(uuid.uuid4()),
                model_name=f"{target_metric}_predictor",
                target=target_metric,
                predicted_value=prediction,
                confidence=confidence,
                timestamp=datetime.now(),
                features=dict(zip(features, feature_values)),
                metadata={"mae": model_info["mae"], "r2": model_info["r2"]},
            )

            # Store prediction
            self.predictions[target_metric].append(result)

            return result

        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    def _update_predictions(self):
        """Update predictions for all trained models"""
        for target_metric in self.prediction_models.keys():
            self.predict_metric(target_metric)

    def detect_anomaly(
        self, metric_name: str, window_hours: int = 24
    ) -> Optional[AnomalyDetection]:
        """Detect anomalies in a metric"""
        try:
            # Get recent data
            history = self.get_metric_history(metric_name, hours=window_hours)
            if len(history) < 10:
                return None

            # Extract values
            values = np.array([m.value for m in history])

            # Initialize anomaly detector if not exists
            if metric_name not in self.anomaly_detectors:
                self.anomaly_detectors[metric_name] = IsolationForest(
                    contamination=0.1, random_state=42
                )
                self.anomaly_detectors[metric_name].fit(values.reshape(-1, 1))

            # Detect anomalies
            detector = self.anomaly_detectors[metric_name]
            anomaly_scores = detector.decision_function(values.reshape(-1, 1))
            latest_score = anomaly_scores[-1]

            # Determine severity
            if latest_score < -0.5:
                severity = "critical"
            elif latest_score < -0.3:
                severity = "high"
            elif latest_score < -0.1:
                severity = "medium"
            else:
                severity = "low"

            # Create anomaly detection result
            anomaly = AnomalyDetection(
                anomaly_id=str(uuid.uuid4()),
                metric_name=metric_name,
                anomaly_score=latest_score,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Anomaly detected in {metric_name} with score {latest_score:.3f}",
                recommendations=self._generate_anomaly_recommendations(
                    metric_name, latest_score
                ),
                metadata={"window_hours": window_hours},
            )

            # Store anomaly
            self.anomalies[metric_name].append(anomaly)

            return anomaly

        except Exception as e:
            print(f"Error detecting anomaly: {e}")
            return None

    def _detect_anomalies(self):
        """Detect anomalies for all tracked metrics"""
        for metric_name in self.metrics_history.keys():
            self.detect_anomaly(metric_name)

    def _generate_anomaly_recommendations(
        self, metric_name: str, score: float
    ) -> List[str]:
        """Generate recommendations for anomaly"""
        recommendations = []

        if metric_name == "cpu_usage" and score < -0.3:
            recommendations.extend(
                [
                    "Check for CPU-intensive processes",
                    "Consider scaling up resources",
                    "Review system load patterns",
                ]
            )
        elif metric_name == "memory_usage" and score < -0.3:
            recommendations.extend(
                [
                    "Check for memory leaks",
                    "Consider increasing memory allocation",
                    "Review memory-intensive operations",
                ]
            )
        elif metric_name == "processing_time" and score < -0.3:
            recommendations.extend(
                [
                    "Check for performance bottlenecks",
                    "Review processing algorithms",
                    "Consider optimization strategies",
                ]
            )

        return recommendations

    def analyze_trends(
        self, metric_name: str, period_hours: int = 24
    ) -> Optional[TrendAnalysis]:
        """Analyze trends in a metric"""
        try:
            # Get historical data
            history = self.get_metric_history(metric_name, hours=period_hours)
            if len(history) < 10:
                return None

            # Extract values and timestamps
            values = np.array([m.value for m in history])
            timestamps = np.array([m.timestamp for m in history])

            # Calculate trend
            time_deltas = np.array(
                [(t - timestamps[0]).total_seconds() for t in timestamps]
            )

            # Linear regression
            coeffs = np.polyfit(time_deltas, values, 1)
            slope = coeffs[0]

            # Calculate trend strength (R-squared)
            y_pred = np.polyval(coeffs, time_deltas)
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Determine trend direction
            if slope > 0.001:
                direction = "increasing"
            elif slope < -0.001:
                direction = "decreasing"
            else:
                direction = "stable"

            # Calculate change rate
            if len(values) > 1:
                change_rate = (
                    ((values[-1] - values[0]) / values[0]) * 100
                    if values[0] != 0
                    else 0
                )
            else:
                change_rate = 0

            # Create trend analysis
            trend = TrendAnalysis(
                trend_id=str(uuid.uuid4()),
                metric_name=metric_name,
                trend_direction=direction,
                trend_strength=r_squared,
                change_rate=change_rate,
                period=f"{period_hours}h",
                confidence=r_squared,
                timestamp=datetime.now(),
                metadata={"slope": slope, "data_points": len(values)},
            )

            # Store trend
            self.trends[metric_name].append(trend)

            return trend

        except Exception as e:
            print(f"Error analyzing trends: {e}")
            return None

    def generate_insight(self, category: str = "general") -> Optional[Insight]:
        """Generate automated insights"""
        try:
            insights = []

            if category == "performance":
                # Performance insights
                cpu_trend = self.analyze_trends("cpu_usage", 6)
                if cpu_trend and cpu_trend.trend_direction == "increasing":
                    insights.append(
                        Insight(
                            insight_id=str(uuid.uuid4()),
                            title="CPU Usage Increasing",
                            description=f"CPU usage has been increasing by {cpu_trend.change_rate:.1f}% over the last 6 hours",
                            category="performance",
                            priority="medium",
                            timestamp=datetime.now(),
                            metrics=["cpu_usage"],
                            recommendations=[
                                "Monitor CPU-intensive processes",
                                "Consider resource scaling",
                                "Review system optimization",
                            ],
                            confidence=cpu_trend.confidence,
                            metadata={"trend_id": cpu_trend.trend_id},
                        )
                    )

            elif category == "quality":
                # Quality insights
                quality_history = self.get_metric_history("quality_score", 24)
                if len(quality_history) > 10:
                    recent_avg = np.mean([m.value for m in quality_history[-5:]])
                    overall_avg = np.mean([m.value for m in quality_history])

                    if recent_avg < overall_avg * 0.9:  # 10% drop
                        insights.append(
                            Insight(
                                insight_id=str(uuid.uuid4()),
                                title="Quality Score Declining",
                                description=f"Recent quality score ({recent_avg:.2f}) is below average ({overall_avg:.2f})",
                                category="quality",
                                priority="high",
                                timestamp=datetime.now(),
                                metrics=["quality_score"],
                                recommendations=[
                                    "Review recent processing jobs",
                                    "Check for data quality issues",
                                    "Validate processing parameters",
                                ],
                                confidence=0.8,
                                metadata={
                                    "recent_avg": recent_avg,
                                    "overall_avg": overall_avg,
                                },
                            )
                        )

            # Add insights to list
            self.insights.extend(insights)

            return insights[0] if insights else None

        except Exception as e:
            print(f"Error generating insights: {e}")
            return None

    def _generate_insights(self):
        """Generate insights for all categories"""
        self.generate_insight("performance")
        self.generate_insight("quality")

    def _store_analytics_data(self):
        """Store analytics data to database"""
        if not self.db_engine:
            return

        try:
            with self.db_engine.connect() as conn:
                # Store predictions
                for predictions in self.predictions.values():
                    for pred in predictions[-10:]:  # Store last 10 predictions
                        conn.execute(
                            text(
                                """
                            INSERT INTO predictions 
                            (prediction_id, model_name, target, predicted_value, confidence, timestamp, features, metadata)
                            VALUES (:prediction_id, :model_name, :target, :predicted_value, :confidence, :timestamp, :features, :metadata)
                            ON CONFLICT (prediction_id) DO NOTHING
                        """
                            ),
                            asdict(pred),
                        )

                # Store trends
                for trends in self.trends.values():
                    for trend in trends[-5:]:  # Store last 5 trends
                        conn.execute(
                            text(
                                """
                            INSERT INTO trends 
                            (trend_id, metric_name, trend_direction, trend_strength, change_rate, period, confidence, timestamp, metadata)
                            VALUES (:trend_id, :metric_name, :trend_direction, :trend_strength, :change_rate, :period, :confidence, :timestamp, :metadata)
                            ON CONFLICT (trend_id) DO NOTHING
                        """
                            ),
                            asdict(trend),
                        )

                # Store anomalies
                for anomalies in self.anomalies.values():
                    for anomaly in anomalies[-10:]:  # Store last 10 anomalies
                        conn.execute(
                            text(
                                """
                            INSERT INTO anomalies 
                            (anomaly_id, metric_name, anomaly_score, severity, timestamp, description, recommendations, metadata)
                            VALUES (:anomaly_id, :metric_name, :anomaly_score, :severity, :timestamp, :description, :recommendations, :metadata)
                            ON CONFLICT (anomaly_id) DO NOTHING
                        """
                            ),
                            asdict(anomaly),
                        )

                # Store insights
                for insight in self.insights[-20:]:  # Store last 20 insights
                    conn.execute(
                        text(
                            """
                        INSERT INTO insights 
                        (insight_id, title, description, category, priority, timestamp, metrics, recommendations, confidence, metadata)
                        VALUES (:insight_id, :title, :description, :category, :priority, :timestamp, :metrics, :recommendations, :confidence, :metadata)
                        ON CONFLICT (insight_id) DO NOTHING
                    """
                        ),
                        asdict(insight),
                    )

                conn.commit()

        except Exception as e:
            print(f"Error storing analytics data: {e}")

    def create_dashboard(
        self,
        name: str,
        description: str,
        metrics: List[str],
        layout: Dict[str, Any],
        refresh_interval: int = 60,
    ) -> DashboardConfig:
        """Create a new analytics dashboard"""
        dashboard = DashboardConfig(
            dashboard_id=str(uuid.uuid4()),
            name=name,
            description=description,
            layout=layout,
            metrics=metrics,
            refresh_interval=refresh_interval,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )

        self.dashboards[dashboard.dashboard_id] = dashboard
        return dashboard

    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get data for a specific dashboard"""
        if dashboard_id not in self.dashboards:
            return {}

        dashboard = self.dashboards[dashboard_id]
        data = {
            "dashboard": asdict(dashboard),
            "metrics": {},
            "predictions": {},
            "trends": {},
            "anomalies": {},
            "insights": [],
        }

        # Get current metric values
        for metric_name in dashboard.metrics:
            history = self.get_metric_history(metric_name, hours=1)
            if history:
                data["metrics"][metric_name] = {
                    "current": history[-1].value,
                    "unit": history[-1].unit,
                    "history": [
                        {"timestamp": m.timestamp.isoformat(), "value": m.value}
                        for m in history[-50:]
                    ],  # Last 50 points
                }

        # Get recent predictions
        for metric_name in dashboard.metrics:
            if metric_name in self.predictions and self.predictions[metric_name]:
                latest_pred = self.predictions[metric_name][-1]
                data["predictions"][metric_name] = asdict(latest_pred)

        # Get recent trends
        for metric_name in dashboard.metrics:
            if metric_name in self.trends and self.trends[metric_name]:
                latest_trend = self.trends[metric_name][-1]
                data["trends"][metric_name] = asdict(latest_trend)

        # Get recent anomalies
        for metric_name in dashboard.metrics:
            if metric_name in self.anomalies and self.anomalies[metric_name]:
                recent_anomalies = [asdict(a) for a in self.anomalies[metric_name][-5:]]
                data["anomalies"][metric_name] = recent_anomalies

        # Get recent insights
        data["insights"] = [asdict(i) for i in self.insights[-10:]]

        return data

    def generate_report(
        self, report_type: str = "comprehensive", time_period: str = "24h"
    ) -> Dict[str, Any]:
        """Generate an analytics report"""
        try:
            report = {
                "report_id": str(uuid.uuid4()),
                "type": report_type,
                "time_period": time_period,
                "generated_at": datetime.now().isoformat(),
                "summary": {},
                "metrics": {},
                "predictions": {},
                "trends": {},
                "anomalies": {},
                "insights": [],
                "recommendations": [],
            }

            # Parse time period
            if time_period.endswith("h"):
                hours = int(time_period[:-1])
            elif time_period.endswith("d"):
                hours = int(time_period[:-1]) * 24
            else:
                hours = 24

            # Generate summary
            total_metrics = len(self.metrics_history)
            total_predictions = sum(len(preds) for preds in self.predictions.values())
            total_anomalies = sum(len(anoms) for anoms in self.anomalies.values())
            total_insights = len(self.insights)

            report["summary"] = {
                "total_metrics": total_metrics,
                "total_predictions": total_predictions,
                "total_anomalies": total_anomalies,
                "total_insights": total_insights,
                "period_hours": hours,
            }

            # Add metric summaries
            for metric_name in self.metrics_history.keys():
                history = self.get_metric_history(metric_name, hours=hours)
                if history:
                    values = [m.value for m in history]
                    report["metrics"][metric_name] = {
                        "current": values[-1],
                        "average": np.mean(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "trend": self.analyze_trends(metric_name, hours),
                    }

            # Add predictions
            for metric_name, predictions in self.predictions.items():
                recent_predictions = [
                    p
                    for p in predictions
                    if p.timestamp >= datetime.now() - timedelta(hours=hours)
                ]
                if recent_predictions:
                    report["predictions"][metric_name] = [
                        asdict(p) for p in recent_predictions
                    ]

            # Add trends
            for metric_name in self.metrics_history.keys():
                trend = self.analyze_trends(metric_name, hours)
                if trend:
                    report["trends"][metric_name] = asdict(trend)

            # Add anomalies
            for metric_name, anomalies in self.anomalies.items():
                recent_anomalies = [
                    a
                    for a in anomalies
                    if a.timestamp >= datetime.now() - timedelta(hours=hours)
                ]
                if recent_anomalies:
                    report["anomalies"][metric_name] = [
                        asdict(a) for a in recent_anomalies
                    ]

            # Add insights
            recent_insights = [
                i
                for i in self.insights
                if i.timestamp >= datetime.now() - timedelta(hours=hours)
            ]
            report["insights"] = [asdict(i) for i in recent_insights]

            # Generate recommendations
            recommendations = []

            # Performance recommendations
            cpu_trend = report["trends"].get("cpu_usage")
            if cpu_trend and cpu_trend["trend_direction"] == "increasing":
                recommendations.append("Consider scaling up CPU resources")

            # Quality recommendations
            quality_trend = report["trends"].get("quality_score")
            if quality_trend and quality_trend["trend_direction"] == "decreasing":
                recommendations.append("Review quality control processes")

            # Anomaly recommendations
            for metric_anomalies in report["anomalies"].values():
                for anomaly in metric_anomalies:
                    if anomaly["severity"] in ["high", "critical"]:
                        recommendations.extend(anomaly["recommendations"])

            report["recommendations"] = list(set(recommendations))  # Remove duplicates

            return report

        except Exception as e:
            print(f"Error generating report: {e}")
            return {}

    def export_data(self, format: str = "json", time_period: str = "24h") -> str:
        """Export analytics data"""
        try:
            # Parse time period
            if time_period.endswith("h"):
                hours = int(time_period[:-1])
            else:
                hours = 24

            export_data = {
                "export_id": str(uuid.uuid4()),
                "format": format,
                "time_period": time_period,
                "exported_at": datetime.now().isoformat(),
                "metrics": {},
                "predictions": {},
                "trends": {},
                "anomalies": {},
                "insights": [],
            }

            # Export metrics
            for metric_name in self.metrics_history.keys():
                history = self.get_metric_history(metric_name, hours=hours)
                export_data["metrics"][metric_name] = [asdict(m) for m in history]

            # Export predictions
            for metric_name, predictions in self.predictions.items():
                recent_predictions = [
                    p
                    for p in predictions
                    if p.timestamp >= datetime.now() - timedelta(hours=hours)
                ]
                export_data["predictions"][metric_name] = [
                    asdict(p) for p in recent_predictions
                ]

            # Export trends
            for metric_name in self.metrics_history.keys():
                trend = self.analyze_trends(metric_name, hours=hours)
                if trend:
                    export_data["trends"][metric_name] = asdict(trend)

            # Export anomalies
            for metric_name, anomalies in self.anomalies.items():
                recent_anomalies = [
                    a
                    for a in anomalies
                    if a.timestamp >= datetime.now() - timedelta(hours=hours)
                ]
                export_data["anomalies"][metric_name] = [
                    asdict(a) for a in recent_anomalies
                ]

            # Export insights
            recent_insights = [
                i
                for i in self.insights
                if i.timestamp >= datetime.now() - timedelta(hours=hours)
            ]
            export_data["insights"] = [asdict(i) for i in recent_insights]

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_export_{timestamp}.{format}"

            if format == "json":
                with open(filename, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format == "csv":
                # Convert to CSV format
                import csv

                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    # Write metrics
                    for metric_name, metrics in export_data["metrics"].items():
                        writer.writerow([f"Metric: {metric_name}"])
                        writer.writerow(["timestamp", "value", "unit", "category"])
                        for metric in metrics:
                            writer.writerow(
                                [
                                    metric["timestamp"],
                                    metric["value"],
                                    metric["unit"],
                                    metric["category"],
                                ]
                            )
                        writer.writerow([])

            return filename

        except Exception as e:
            print(f"Error exporting data: {e}")
            return ""


def main():
    """Main function to run the advanced analytics system"""
    # Initialize analytics
    analytics = AdvancedAnalytics()

    # Add some sample metrics
    analytics.add_metric("processing_time", 45.2, "seconds", "performance")
    analytics.add_metric("quality_score", 0.92, "score", "quality")
    analytics.add_metric("throughput", 150, "jobs/hour", "performance")

    # Train prediction models
    analytics.train_prediction_model("processing_time", ["cpu_usage", "memory_usage"])
    analytics.train_prediction_model("quality_score", ["processing_time", "throughput"])

    # Generate report
    report = analytics.generate_report("comprehensive", "24h")
    print("Analytics Report:", json.dumps(report, indent=2, default=str))

    # Export data
    export_file = analytics.export_data("json", "24h")
    print(f"Data exported to: {export_file}")


if __name__ == "__main__":
    main()
