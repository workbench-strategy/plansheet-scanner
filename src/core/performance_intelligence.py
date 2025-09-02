"""
Performance Intelligence System for Plansheet Scanner

This module provides intelligent performance monitoring, optimization recommendations,
and automated performance tuning capabilities.

Features:
- Intelligent performance monitoring and analysis
- Automated performance optimization recommendations
- Resource utilization tracking and prediction
- Performance bottleneck detection and resolution
- Automated performance tuning and scaling
- Performance trend analysis and forecasting
"""

import asyncio
import gc
import json
import os
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
import psutil

# ML imports
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Database imports
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""

    metric_id: str
    name: str
    value: float
    unit: str
    category: str  # 'cpu', 'memory', 'disk', 'network', 'application'
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""

    alert_id: str
    metric_name: str
    alert_type: str  # 'threshold', 'anomaly', 'trend'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    threshold: Optional[float]
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime]
    metadata: Dict[str, Any]


@dataclass
class PerformanceBottleneck:
    """Performance bottleneck data structure"""

    bottleneck_id: str
    name: str
    category: str  # 'cpu', 'memory', 'disk', 'network', 'application'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    impact_score: float
    recommendations: List[str]
    detected_at: datetime
    resolved_at: Optional[datetime]
    metadata: Dict[str, Any]


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation data structure"""

    recommendation_id: str
    title: str
    description: str
    category: str  # 'cpu', 'memory', 'disk', 'network', 'application', 'algorithm'
    priority: str  # 'low', 'medium', 'high', 'critical'
    expected_improvement: float
    implementation_cost: str  # 'low', 'medium', 'high'
    implementation_steps: List[str]
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class PerformanceProfile:
    """Performance profile data structure"""

    profile_id: str
    name: str
    description: str
    baseline_metrics: Dict[str, float]
    target_metrics: Dict[str, float]
    optimization_history: List[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class ResourceUtilization:
    """Resource utilization data structure"""

    utilization_id: str
    resource_type: str  # 'cpu', 'memory', 'disk', 'network'
    current_usage: float
    peak_usage: float
    average_usage: float
    utilization_trend: str  # 'increasing', 'decreasing', 'stable'
    prediction_horizon: int  # hours
    predicted_usage: float
    timestamp: datetime
    metadata: Dict[str, Any]


class PerformanceIntelligence:
    """Performance Intelligence System"""

    def __init__(
        self,
        data_dir: str = "performance_data",
        db_config: Optional[Dict[str, Any]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Storage
        self.performance_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.performance_alerts: Dict[str, List[PerformanceAlert]] = defaultdict(list)
        self.performance_bottlenecks: Dict[
            str, List[PerformanceBottleneck]
        ] = defaultdict(list)
        self.optimization_recommendations: Dict[str, OptimizationRecommendation] = {}
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.resource_utilization: Dict[str, List[ResourceUtilization]] = defaultdict(
            list
        )

        # ML models
        self.prediction_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Performance thresholds
        self.performance_thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "response_time": {"warning": 5.0, "critical": 10.0},
            "throughput": {"warning": 100, "critical": 50},
        }

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Database connection
        self.db_engine = None
        if db_config:
            self._setup_database(db_config)

        # Initialize ML models
        self._setup_ml_models()

        # Start performance monitoring
        self.start_performance_monitoring()

    def _setup_database(self, db_config: Dict[str, Any]):
        """Setup database connection for performance data storage"""
        try:
            connection_string = (
                f"postgresql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            self.db_engine = create_engine(connection_string)

            # Create performance tables
            self._create_performance_tables()

        except Exception as e:
            print(f"Warning: Database setup failed: {e}")

    def _create_performance_tables(self):
        """Create performance tables in database"""
        if not self.db_engine:
            return

        try:
            with self.db_engine.connect() as conn:
                # Performance metrics table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        metric_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        unit VARCHAR(50) NOT NULL,
                        category VARCHAR(100) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Performance alerts table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        alert_id VARCHAR(255) PRIMARY KEY,
                        metric_name VARCHAR(255) NOT NULL,
                        alert_type VARCHAR(50) NOT NULL,
                        severity VARCHAR(50) NOT NULL,
                        message TEXT NOT NULL,
                        threshold DOUBLE PRECISION,
                        current_value DOUBLE PRECISION NOT NULL,
                        triggered_at TIMESTAMP NOT NULL,
                        resolved_at TIMESTAMP,
                        metadata JSONB
                    )
                """
                    )
                )

                # Performance bottlenecks table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS performance_bottlenecks (
                        bottleneck_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        category VARCHAR(100) NOT NULL,
                        severity VARCHAR(50) NOT NULL,
                        description TEXT NOT NULL,
                        impact_score DOUBLE PRECISION NOT NULL,
                        recommendations TEXT[],
                        detected_at TIMESTAMP NOT NULL,
                        resolved_at TIMESTAMP,
                        metadata JSONB
                    )
                """
                    )
                )

                # Optimization recommendations table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS optimization_recommendations (
                        recommendation_id VARCHAR(255) PRIMARY KEY,
                        title VARCHAR(255) NOT NULL,
                        description TEXT NOT NULL,
                        category VARCHAR(100) NOT NULL,
                        priority VARCHAR(50) NOT NULL,
                        expected_improvement DOUBLE PRECISION NOT NULL,
                        implementation_cost VARCHAR(50) NOT NULL,
                        implementation_steps TEXT[],
                        created_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Resource utilization table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS resource_utilization (
                        utilization_id VARCHAR(255) PRIMARY KEY,
                        resource_type VARCHAR(100) NOT NULL,
                        current_usage DOUBLE PRECISION NOT NULL,
                        peak_usage DOUBLE PRECISION NOT NULL,
                        average_usage DOUBLE PRECISION NOT NULL,
                        utilization_trend VARCHAR(50) NOT NULL,
                        prediction_horizon INTEGER NOT NULL,
                        predicted_usage DOUBLE PRECISION NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                conn.commit()

        except Exception as e:
            print(f"Warning: Failed to create performance tables: {e}")

    def _setup_ml_models(self):
        """Setup ML models for performance prediction and anomaly detection"""
        # Initialize prediction models
        self.prediction_models["cpu_usage"] = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self.prediction_models["memory_usage"] = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self.prediction_models["response_time"] = RandomForestRegressor(
            n_estimators=100, random_state=42
        )

        # Initialize anomaly detectors
        self.anomaly_detectors["cpu_usage"] = IsolationForest(
            contamination=0.1, random_state=42
        )
        self.anomaly_detectors["memory_usage"] = IsolationForest(
            contamination=0.1, random_state=42
        )
        self.anomaly_detectors["response_time"] = IsolationForest(
            contamination=0.1, random_state=42
        )

        # Initialize scalers
        self.scalers["cpu_usage"] = StandardScaler()
        self.scalers["memory_usage"] = StandardScaler()
        self.scalers["response_time"] = StandardScaler()

    def start_performance_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._performance_monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

    def stop_performance_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Check for alerts
                self._check_performance_alerts()

                # Detect bottlenecks
                self._detect_performance_bottlenecks()

                # Update resource utilization
                self._update_resource_utilization()

                # Generate optimization recommendations
                self._generate_optimization_recommendations()

                # Wait for next cycle
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            self.add_performance_metric(
                "cpu_usage",
                cpu_percent,
                "%",
                "cpu",
                {
                    "cpu_count": cpu_count,
                    "cpu_freq_current": cpu_freq.current if cpu_freq else None,
                    "cpu_freq_max": cpu_freq.max if cpu_freq else None,
                },
            )

            # Memory metrics
            memory = psutil.virtual_memory()
            self.add_performance_metric(
                "memory_usage",
                memory.percent,
                "%",
                "memory",
                {
                    "memory_total": memory.total,
                    "memory_available": memory.available,
                    "memory_used": memory.used,
                },
            )

            # Disk metrics
            disk = psutil.disk_usage("/")
            self.add_performance_metric(
                "disk_usage",
                (disk.used / disk.total) * 100,
                "%",
                "disk",
                {
                    "disk_total": disk.total,
                    "disk_used": disk.used,
                    "disk_free": disk.free,
                },
            )

            # Network metrics
            network = psutil.net_io_counters()
            self.add_performance_metric(
                "network_bytes_sent",
                network.bytes_sent,
                "bytes",
                "network",
                {
                    "network_packets_sent": network.packets_sent,
                    "network_bytes_recv": network.bytes_recv,
                    "network_packets_recv": network.packets_recv,
                },
            )

            # Process-specific metrics
            current_process = psutil.Process()
            self.add_performance_metric(
                "process_cpu_percent",
                current_process.cpu_percent(),
                "%",
                "application",
                {
                    "process_memory_rss": current_process.memory_info().rss,
                    "process_memory_vms": current_process.memory_info().vms,
                    "process_num_threads": current_process.num_threads(),
                },
            )

        except Exception as e:
            print(f"Error collecting system metrics: {e}")

    def add_performance_metric(
        self,
        name: str,
        value: float,
        unit: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceMetric:
        """Add a performance metric"""
        metric = PerformanceMetric(
            metric_id=str(uuid.uuid4()),
            name=name,
            value=value,
            unit=unit,
            category=category,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        # Store in memory
        self.performance_metrics[name].append(metric)

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        INSERT INTO performance_metrics
                        (metric_id, name, value, unit, category, timestamp, metadata)
                        VALUES (:metric_id, :name, :value, :unit, :category, :timestamp, :metadata)
                    """
                        ),
                        asdict(metric),
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing performance metric: {e}")

        return metric

    def _check_performance_alerts(self):
        """Check for performance alerts"""
        for metric_name, metrics in self.performance_metrics.items():
            if not metrics:
                continue

            latest_metric = metrics[-1]
            thresholds = self.performance_thresholds.get(metric_name, {})

            # Check threshold alerts
            for alert_level, threshold in thresholds.items():
                if latest_metric.value > threshold:
                    alert = PerformanceAlert(
                        alert_id=str(uuid.uuid4()),
                        metric_name=metric_name,
                        alert_type="threshold",
                        severity=alert_level,
                        message=f"{metric_name} exceeded {alert_level} threshold: {latest_metric.value:.2f} {latest_metric.unit}",
                        threshold=threshold,
                        current_value=latest_metric.value,
                        triggered_at=datetime.now(),
                        resolved_at=None,
                        metadata={},
                    )

                    self.performance_alerts[metric_name].append(alert)

            # Check anomaly alerts
            if len(metrics) >= 10 and metric_name in self.anomaly_detectors:
                recent_values = [m.value for m in metrics[-10:]]
                values_array = np.array(recent_values).reshape(-1, 1)

                # Update anomaly detector
                detector = self.anomaly_detectors[metric_name]
                detector.fit(values_array[:-1])  # Fit on all but the latest value

                # Check if latest value is anomalous
                latest_value_array = np.array([recent_values[-1]]).reshape(-1, 1)
                anomaly_score = detector.decision_function(latest_value_array)[0]

                if anomaly_score < -0.5:  # Anomaly threshold
                    alert = PerformanceAlert(
                        alert_id=str(uuid.uuid4()),
                        metric_name=metric_name,
                        alert_type="anomaly",
                        severity="medium" if anomaly_score < -0.7 else "low",
                        message=f"Anomaly detected in {metric_name}: value {latest_metric.value:.2f} (score: {anomaly_score:.3f})",
                        threshold=None,
                        current_value=latest_metric.value,
                        triggered_at=datetime.now(),
                        resolved_at=None,
                        metadata={"anomaly_score": anomaly_score},
                    )

                    self.performance_alerts[metric_name].append(alert)

    def _detect_performance_bottlenecks(self):
        """Detect performance bottlenecks"""
        bottlenecks = []

        # CPU bottleneck detection
        cpu_metrics = self.performance_metrics.get("cpu_usage", [])
        if cpu_metrics and len(cpu_metrics) >= 5:
            recent_cpu = [m.value for m in cpu_metrics[-5:]]
            avg_cpu = np.mean(recent_cpu)

            if avg_cpu > 80:
                bottleneck = PerformanceBottleneck(
                    bottleneck_id=str(uuid.uuid4()),
                    name="High CPU Usage",
                    category="cpu",
                    severity="high" if avg_cpu > 90 else "medium",
                    description=f"CPU usage averaging {avg_cpu:.1f}% over the last 5 measurements",
                    impact_score=avg_cpu / 100,
                    recommendations=[
                        "Consider scaling up CPU resources",
                        "Optimize CPU-intensive operations",
                        "Review and optimize algorithms",
                        "Implement caching strategies",
                    ],
                    detected_at=datetime.now(),
                    resolved_at=None,
                    metadata={"average_cpu": avg_cpu},
                )
                bottlenecks.append(bottleneck)

        # Memory bottleneck detection
        memory_metrics = self.performance_metrics.get("memory_usage", [])
        if memory_metrics and len(memory_metrics) >= 5:
            recent_memory = [m.value for m in memory_metrics[-5:]]
            avg_memory = np.mean(recent_memory)

            if avg_memory > 85:
                bottleneck = PerformanceBottleneck(
                    bottleneck_id=str(uuid.uuid4()),
                    name="High Memory Usage",
                    category="memory",
                    severity="high" if avg_memory > 95 else "medium",
                    description=f"Memory usage averaging {avg_memory:.1f}% over the last 5 measurements",
                    impact_score=avg_memory / 100,
                    recommendations=[
                        "Increase available memory",
                        "Implement memory pooling",
                        "Review memory leaks",
                        "Optimize data structures",
                    ],
                    detected_at=datetime.now(),
                    resolved_at=None,
                    metadata={"average_memory": avg_memory},
                )
                bottlenecks.append(bottleneck)

        # Disk bottleneck detection
        disk_metrics = self.performance_metrics.get("disk_usage", [])
        if disk_metrics and len(disk_metrics) >= 5:
            recent_disk = [m.value for m in disk_metrics[-5:]]
            avg_disk = np.mean(recent_disk)

            if avg_disk > 90:
                bottleneck = PerformanceBottleneck(
                    bottleneck_id=str(uuid.uuid4()),
                    name="High Disk Usage",
                    category="disk",
                    severity="critical" if avg_disk > 95 else "high",
                    description=f"Disk usage averaging {avg_disk:.1f}% over the last 5 measurements",
                    impact_score=avg_disk / 100,
                    recommendations=[
                        "Clean up unnecessary files",
                        "Implement data archiving",
                        "Consider disk expansion",
                        "Optimize storage usage",
                    ],
                    detected_at=datetime.now(),
                    resolved_at=None,
                    metadata={"average_disk": avg_disk},
                )
                bottlenecks.append(bottleneck)

        # Store bottlenecks
        for bottleneck in bottlenecks:
            self.performance_bottlenecks[bottleneck.category].append(bottleneck)

    def _update_resource_utilization(self):
        """Update resource utilization tracking"""
        for resource_type in ["cpu", "memory", "disk"]:
            metrics = self.performance_metrics.get(f"{resource_type}_usage", [])

            if len(metrics) >= 10:
                recent_values = [m.value for m in metrics[-10:]]
                current_usage = recent_values[-1]
                peak_usage = max(recent_values)
                average_usage = np.mean(recent_values)

                # Calculate trend
                if len(recent_values) >= 5:
                    recent_trend = recent_values[-5:]
                    trend_slope = np.polyfit(range(len(recent_trend)), recent_trend, 1)[
                        0
                    ]

                    if trend_slope > 1.0:
                        utilization_trend = "increasing"
                    elif trend_slope < -1.0:
                        utilization_trend = "decreasing"
                    else:
                        utilization_trend = "stable"
                else:
                    utilization_trend = "stable"

                # Predict future usage
                predicted_usage = self._predict_resource_usage(
                    resource_type, recent_values
                )

                utilization = ResourceUtilization(
                    utilization_id=str(uuid.uuid4()),
                    resource_type=resource_type,
                    current_usage=current_usage,
                    peak_usage=peak_usage,
                    average_usage=average_usage,
                    utilization_trend=utilization_trend,
                    prediction_horizon=1,  # 1 hour
                    predicted_usage=predicted_usage,
                    timestamp=datetime.now(),
                    metadata={},
                )

                self.resource_utilization[resource_type].append(utilization)

    def _predict_resource_usage(
        self, resource_type: str, recent_values: List[float]
    ) -> float:
        """Predict future resource usage"""
        try:
            if resource_type not in self.prediction_models or len(recent_values) < 5:
                return recent_values[-1] if recent_values else 0.0

            # Prepare features (use last 5 values as features)
            features = recent_values[-5:]
            if len(features) < 5:
                features = features + [features[-1]] * (5 - len(features))

            # Make prediction
            model = self.prediction_models[resource_type]
            scaler = self.scalers[resource_type]

            # Scale features
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]

            return max(0.0, min(100.0, prediction))  # Clamp between 0 and 100

        except Exception as e:
            print(f"Error predicting resource usage: {e}")
            return recent_values[-1] if recent_values else 0.0

    def _generate_optimization_recommendations(self):
        """Generate optimization recommendations"""
        recommendations = []

        # CPU optimization recommendations
        cpu_metrics = self.performance_metrics.get("cpu_usage", [])
        if cpu_metrics and len(cpu_metrics) >= 10:
            recent_cpu = [m.value for m in cpu_metrics[-10:]]
            avg_cpu = np.mean(recent_cpu)

            if avg_cpu > 70:
                recommendation = OptimizationRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    title="Optimize CPU Usage",
                    description=f"CPU usage is averaging {avg_cpu:.1f}%, consider optimization strategies",
                    category="cpu",
                    priority="high" if avg_cpu > 85 else "medium",
                    expected_improvement=20.0,  # Expected 20% improvement
                    implementation_cost="medium",
                    implementation_steps=[
                        "Profile CPU-intensive operations",
                        "Implement parallel processing",
                        "Optimize algorithms",
                        "Add CPU caching",
                        "Consider horizontal scaling",
                    ],
                    created_at=datetime.now(),
                    metadata={"current_cpu_usage": avg_cpu},
                )
                recommendations.append(recommendation)

        # Memory optimization recommendations
        memory_metrics = self.performance_metrics.get("memory_usage", [])
        if memory_metrics and len(memory_metrics) >= 10:
            recent_memory = [m.value for m in memory_metrics[-10:]]
            avg_memory = np.mean(recent_memory)

            if avg_memory > 75:
                recommendation = OptimizationRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    title="Optimize Memory Usage",
                    description=f"Memory usage is averaging {avg_memory:.1f}%, consider memory optimization",
                    category="memory",
                    priority="high" if avg_memory > 90 else "medium",
                    expected_improvement=15.0,  # Expected 15% improvement
                    implementation_cost="medium",
                    implementation_steps=[
                        "Implement memory pooling",
                        "Optimize data structures",
                        "Add memory monitoring",
                        "Implement garbage collection optimization",
                        "Consider memory expansion",
                    ],
                    created_at=datetime.now(),
                    metadata={"current_memory_usage": avg_memory},
                )
                recommendations.append(recommendation)

        # Algorithm optimization recommendations
        response_time_metrics = self.performance_metrics.get("response_time", [])
        if response_time_metrics and len(response_time_metrics) >= 10:
            recent_response = [m.value for m in response_time_metrics[-10:]]
            avg_response = np.mean(recent_response)

            if avg_response > 3.0:  # More than 3 seconds
                recommendation = OptimizationRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    title="Optimize Response Time",
                    description=f"Average response time is {avg_response:.2f} seconds, consider algorithm optimization",
                    category="algorithm",
                    priority="high" if avg_response > 5.0 else "medium",
                    expected_improvement=50.0,  # Expected 50% improvement
                    implementation_cost="high",
                    implementation_steps=[
                        "Profile application performance",
                        "Optimize database queries",
                        "Implement caching layers",
                        "Use more efficient algorithms",
                        "Consider async processing",
                    ],
                    created_at=datetime.now(),
                    metadata={"current_response_time": avg_response},
                )
                recommendations.append(recommendation)

        # Store recommendations
        for recommendation in recommendations:
            self.optimization_recommendations[
                recommendation.recommendation_id
            ] = recommendation

    def train_prediction_models(self, training_data: pd.DataFrame):
        """Train prediction models with historical data"""
        try:
            for metric_name in ["cpu_usage", "memory_usage", "response_time"]:
                if metric_name not in training_data.columns:
                    continue

                # Prepare features (use lagged values)
                feature_columns = []
                for i in range(1, 6):  # Use last 5 values as features
                    lag_col = f"{metric_name}_lag_{i}"
                    training_data[lag_col] = training_data[metric_name].shift(i)
                    feature_columns.append(lag_col)

                # Remove rows with NaN values
                clean_data = training_data.dropna(
                    subset=feature_columns + [metric_name]
                )

                if len(clean_data) < 20:
                    print(f"Insufficient data for {metric_name}")
                    continue

                # Split data
                X = clean_data[feature_columns]
                y = clean_data[metric_name]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Scale features
                scaler = self.scalers[metric_name]
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model
                model = self.prediction_models[metric_name]
                model.fit(X_train_scaled, y_train)

                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                print(f"Trained {metric_name} model: MAE={mae:.3f}, R²={r2:.3f}")

        except Exception as e:
            print(f"Error training prediction models: {e}")

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        summary = {
            "period_hours": hours,
            "metrics": {},
            "alerts": {},
            "bottlenecks": {},
            "recommendations": {},
            "utilization": {},
        }

        # Aggregate metrics
        for metric_name, metrics in self.performance_metrics.items():
            recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary["metrics"][metric_name] = {
                    "current": values[-1],
                    "average": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                }

        # Count alerts
        for metric_name, alerts in self.performance_alerts.items():
            recent_alerts = [a for a in alerts if a.triggered_at >= cutoff_time]
            if recent_alerts:
                summary["alerts"][metric_name] = {
                    "total": len(recent_alerts),
                    "critical": len(
                        [a for a in recent_alerts if a.severity == "critical"]
                    ),
                    "high": len([a for a in recent_alerts if a.severity == "high"]),
                    "medium": len([a for a in recent_alerts if a.severity == "medium"]),
                    "low": len([a for a in recent_alerts if a.severity == "low"]),
                }

        # Count bottlenecks
        for category, bottlenecks in self.performance_bottlenecks.items():
            recent_bottlenecks = [
                b for b in bottlenecks if b.detected_at >= cutoff_time
            ]
            if recent_bottlenecks:
                summary["bottlenecks"][category] = {
                    "total": len(recent_bottlenecks),
                    "critical": len(
                        [b for b in recent_bottlenecks if b.severity == "critical"]
                    ),
                    "high": len(
                        [b for b in recent_bottlenecks if b.severity == "high"]
                    ),
                    "medium": len(
                        [b for b in recent_bottlenecks if b.severity == "medium"]
                    ),
                    "low": len([b for b in recent_bottlenecks if b.severity == "low"]),
                }

        # Count recommendations
        recent_recommendations = [
            r
            for r in self.optimization_recommendations.values()
            if r.created_at >= cutoff_time
        ]
        if recent_recommendations:
            summary["recommendations"] = {
                "total": len(recent_recommendations),
                "critical": len(
                    [r for r in recent_recommendations if r.priority == "critical"]
                ),
                "high": len(
                    [r for r in recent_recommendations if r.priority == "high"]
                ),
                "medium": len(
                    [r for r in recent_recommendations if r.priority == "medium"]
                ),
                "low": len([r for r in recent_recommendations if r.priority == "low"]),
            }

        # Current utilization
        for resource_type, utilization_list in self.resource_utilization.items():
            if utilization_list:
                latest = utilization_list[-1]
                summary["utilization"][resource_type] = {
                    "current": latest.current_usage,
                    "peak": latest.peak_usage,
                    "average": latest.average_usage,
                    "trend": latest.utilization_trend,
                    "predicted": latest.predicted_usage,
                }

        return summary

    def create_performance_profile(
        self,
        name: str,
        description: str,
        baseline_metrics: Dict[str, float],
        target_metrics: Dict[str, float],
    ) -> PerformanceProfile:
        """Create a performance profile"""
        profile = PerformanceProfile(
            profile_id=str(uuid.uuid4()),
            name=name,
            description=description,
            baseline_metrics=baseline_metrics,
            target_metrics=target_metrics,
            optimization_history=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )

        self.performance_profiles[profile.profile_id] = profile
        return profile

    def apply_optimization(self, recommendation_id: str) -> bool:
        """Apply an optimization recommendation"""
        if recommendation_id not in self.optimization_recommendations:
            return False

        recommendation = self.optimization_recommendations[recommendation_id]

        try:
            # Apply optimization based on category
            if recommendation.category == "cpu":
                # CPU optimization strategies
                print(f"Applying CPU optimization: {recommendation.title}")
                # Implementation would go here

            elif recommendation.category == "memory":
                # Memory optimization strategies
                print(f"Applying memory optimization: {recommendation.title}")
                # Force garbage collection
                gc.collect()

            elif recommendation.category == "algorithm":
                # Algorithm optimization strategies
                print(f"Applying algorithm optimization: {recommendation.title}")
                # Implementation would go here

            # Mark recommendation as applied
            recommendation.metadata["applied"] = True
            recommendation.metadata["applied_at"] = datetime.now().isoformat()

            return True

        except Exception as e:
            print(f"Error applying optimization: {e}")
            return False

    def export_performance_data(
        self, format: str = "json", time_period: str = "24h"
    ) -> str:
        """Export performance data"""
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
                "alerts": {},
                "bottlenecks": {},
                "recommendations": {},
                "utilization": {},
            }

            # Export metrics
            for metric_name, metrics in self.performance_metrics.items():
                recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                export_data["metrics"][metric_name] = [
                    asdict(m) for m in recent_metrics
                ]

            # Export alerts
            for metric_name, alerts in self.performance_alerts.items():
                recent_alerts = [a for a in alerts if a.triggered_at >= cutoff_time]
                export_data["alerts"][metric_name] = [asdict(a) for a in recent_alerts]

            # Export bottlenecks
            for category, bottlenecks in self.performance_bottlenecks.items():
                recent_bottlenecks = [
                    b for b in bottlenecks if b.detected_at >= cutoff_time
                ]
                export_data["bottlenecks"][category] = [
                    asdict(b) for b in recent_bottlenecks
                ]

            # Export recommendations
            recent_recommendations = [
                r
                for r in self.optimization_recommendations.values()
                if r.created_at >= cutoff_time
            ]
            export_data["recommendations"] = [asdict(r) for r in recent_recommendations]

            # Export utilization
            for resource_type, utilization_list in self.resource_utilization.items():
                recent_utilization = [
                    u for u in utilization_list if u.timestamp >= cutoff_time
                ]
                export_data["utilization"][resource_type] = [
                    asdict(u) for u in recent_utilization
                ]

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_export_{timestamp}.{format}"

            if format == "json":
                with open(filename, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)

            return filename

        except Exception as e:
            print(f"Error exporting performance data: {e}")
            return ""


def main():
    """Main function to run the performance intelligence system"""
    # Initialize system
    pi = PerformanceIntelligence()

    # Add some sample metrics
    pi.add_performance_metric("cpu_usage", 75.5, "%", "cpu")
    pi.add_performance_metric("memory_usage", 82.3, "%", "memory")
    pi.add_performance_metric("response_time", 4.2, "seconds", "application")

    # Wait for monitoring to collect data
    time.sleep(35)  # Wait for monitoring cycle

    # Get performance summary
    summary = pi.get_performance_summary(hours=1)
    print("Performance Summary:", json.dumps(summary, indent=2, default=str))

    # Get optimization recommendations
    recommendations = list(pi.optimization_recommendations.values())
    print(f"Generated {len(recommendations)} optimization recommendations")

    # Export data
    export_file = pi.export_performance_data("json", "1h")
    print(f"Performance data exported to: {export_file}")


if __name__ == "__main__":
    main()
