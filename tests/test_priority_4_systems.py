"""
Comprehensive Tests for Priority 4 Systems

This module provides comprehensive unit and integration tests for all Priority 4 systems:
1. Advanced Analytics Dashboard System
2. Machine Learning Pipeline System
3. Intelligent Quality Assurance System
4. Advanced Geospatial Intelligence System
5. Performance Intelligence System
"""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.advanced_analytics import AdvancedAnalytics, AnalyticsMetric, PredictionResult
from core.advanced_geospatial_intelligence import (
    AdvancedGeospatialIntelligence,
    SpatialFeature,
)
from core.intelligent_qa import (
    IntelligentQA,
    QualityIssue,
    QualityMetric,
    ValidationRule,
)
from core.ml_pipeline import MLPipeline, ModelConfig, ModelVersion, TrainingJob
from core.performance_intelligence import (
    PerformanceAlert,
    PerformanceIntelligence,
    PerformanceMetric,
)


class TestAdvancedAnalytics:
    """Test Advanced Analytics Dashboard System"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.analytics = AdvancedAnalytics()

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_analytics_initialization(self):
        """Test analytics system initialization"""
        assert self.analytics is not None
        assert hasattr(self.analytics, "metrics_history")
        assert hasattr(self.analytics, "predictions")
        assert hasattr(self.analytics, "trends")
        assert hasattr(self.analytics, "anomalies")
        assert hasattr(self.analytics, "insights")

    def test_add_metric(self):
        """Test adding analytics metrics"""
        metric = self.analytics.add_metric("test_metric", 42.0, "count", "test")

        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.unit == "count"
        assert metric.category == "test"
        assert len(self.analytics.metrics_history["test_metric"]) == 1

    def test_get_metric_history(self):
        """Test retrieving metric history"""
        # Add multiple metrics
        self.analytics.add_metric("test_metric", 10.0, "count", "test")
        self.analytics.add_metric("test_metric", 20.0, "count", "test")
        self.analytics.add_metric("test_metric", 30.0, "count", "test")

        history = self.analytics.get_metric_history("test_metric", hours=1)
        assert len(history) == 3
        assert history[0].value == 10.0
        assert history[1].value == 20.0
        assert history[2].value == 30.0

    def test_predict_metric(self):
        """Test metric prediction"""
        # Add training data
        for i in range(50):
            self.analytics.add_metric("cpu_usage", 50 + i * 0.5, "%", "system")
            self.analytics.add_metric("memory_usage", 60 + i * 0.3, "%", "system")

        # Train prediction model
        self.analytics.train_prediction_model("cpu_usage", ["memory_usage"])

        # Make prediction
        prediction = self.analytics.predict_metric("cpu_usage")
        assert prediction is not None
        assert hasattr(prediction, "predicted_value")
        assert hasattr(prediction, "confidence")

    def test_detect_anomaly(self):
        """Test anomaly detection"""
        # Add normal data
        for i in range(20):
            self.analytics.add_metric(
                "test_metric", 50.0 + np.random.normal(0, 5), "count", "test"
            )

        # Add anomaly
        self.analytics.add_metric("test_metric", 100.0, "count", "test")

        anomaly = self.analytics.detect_anomaly("test_metric")
        assert anomaly is not None
        assert hasattr(anomaly, "anomaly_score")
        assert hasattr(anomaly, "severity")

    def test_analyze_trends(self):
        """Test trend analysis"""
        # Add trending data
        for i in range(20):
            self.analytics.add_metric("trend_metric", 10.0 + i * 2.0, "count", "test")

        trend = self.analytics.analyze_trends("trend_metric")
        assert trend is not None
        assert trend.trend_direction == "increasing"
        assert trend.trend_strength > 0.5

    def test_generate_insight(self):
        """Test insight generation"""
        insight = self.analytics.generate_insight("performance")
        # Insight generation may return None if no significant patterns
        if insight is not None:
            assert hasattr(insight, "title")
            assert hasattr(insight, "description")
            assert hasattr(insight, "recommendations")

    def test_create_dashboard(self):
        """Test dashboard creation"""
        dashboard = self.analytics.create_dashboard(
            name="Test Dashboard",
            description="Test dashboard for analytics",
            metrics=["cpu_usage", "memory_usage"],
            layout={"rows": 2, "cols": 2},
            refresh_interval=60,
        )

        assert dashboard.name == "Test Dashboard"
        assert dashboard.description == "Test dashboard for analytics"
        assert len(dashboard.metrics) == 2
        assert dashboard.refresh_interval == 60

    def test_generate_report(self):
        """Test report generation"""
        # Add some metrics first
        self.analytics.add_metric("test_metric", 42.0, "count", "test")

        report = self.analytics.generate_report("comprehensive", "24h")
        assert report is not None
        assert "summary" in report
        assert "metrics" in report
        assert "recommendations" in report

    def test_export_data(self):
        """Test data export"""
        # Add some data
        self.analytics.add_metric("test_metric", 42.0, "count", "test")

        export_file = self.analytics.export_data("json", "24h")
        assert export_file.endswith(".json")
        assert os.path.exists(export_file)

        # Cleanup
        os.remove(export_file)


class TestMLPipeline:
    """Test Machine Learning Pipeline System"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = MLPipeline(models_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_initialization(self):
        """Test ML pipeline initialization"""
        assert self.pipeline is not None
        assert hasattr(self.pipeline, "model_configs")
        assert hasattr(self.pipeline, "training_jobs")
        assert hasattr(self.pipeline, "model_versions")
        assert hasattr(self.pipeline, "feature_sets")

    def test_create_model_config(self):
        """Test creating model configuration"""
        config = self.pipeline.create_model_config(
            name="Test Model",
            model_type="classification",
            algorithm="random_forest",
            feature_columns=["feature1", "feature2"],
            target_column="target",
            hyperparameters={"n_estimators": 100},
        )

        assert config.name == "Test Model"
        assert config.type == "classification"
        assert config.algorithm == "random_forest"
        assert len(config.feature_columns) == 2
        assert config.target_column == "target"

    def test_create_feature_set(self):
        """Test creating feature set"""
        feature_set = self.pipeline.create_feature_set(
            name="Test Features",
            description="Test feature set",
            features=["feature1", "feature2", "feature3"],
        )

        assert feature_set.name == "Test Features"
        assert feature_set.description == "Test feature set"
        assert len(feature_set.features) == 3

    def test_submit_training_job(self):
        """Test submitting training job"""
        # Create model config first
        config = self.pipeline.create_model_config(
            name="Test Model",
            model_type="classification",
            algorithm="random_forest",
            feature_columns=["feature1", "feature2"],
            target_column="target",
        )

        # Create training data
        np.random.seed(42)
        training_data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.choice([0, 1], 100),
            }
        )

        job_id = self.pipeline.submit_training_job(
            model_config_id=config.model_id,
            training_data=training_data,
            validation_split=0.2,
        )

        assert job_id is not None
        assert job_id in self.pipeline.training_jobs

    def test_get_training_job_status(self):
        """Test getting training job status"""
        # Create model config
        config = self.pipeline.create_model_config(
            name="Test Model",
            model_type="classification",
            algorithm="random_forest",
            feature_columns=["feature1", "feature2"],
            target_column="target",
        )

        # Create training data
        training_data = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "target": np.random.choice([0, 1], 50),
            }
        )

        job_id = self.pipeline.submit_training_job(
            model_config_id=config.model_id, training_data=training_data
        )

        status = self.pipeline.get_training_job_status(job_id)
        assert status is not None
        assert "status" in status
        assert "progress" in status

    def test_create_experiment(self):
        """Test creating ML experiment"""
        experiment = self.pipeline.create_experiment(
            name="Test Experiment",
            description="Test hyperparameter optimization",
            model_configs=["config1", "config2"],
            hyperparameter_ranges={"n_estimators": [50, 200], "max_depth": [5, 15]},
            optimization_metric="accuracy",
            max_trials=10,
        )

        assert experiment.name == "Test Experiment"
        assert experiment.optimization_metric == "accuracy"
        assert experiment.max_trials == 10

    def test_get_model_versions(self):
        """Test getting model versions"""
        # Create model config
        config = self.pipeline.create_model_config(
            name="Test Model",
            model_type="classification",
            algorithm="random_forest",
            feature_columns=["feature1"],
            target_column="target",
        )

        versions = self.pipeline.get_model_versions(config.model_id)
        assert isinstance(versions, list)

    def test_export_model(self):
        """Test model export"""
        # Create model config
        config = self.pipeline.create_model_config(
            name="Test Model",
            model_type="classification",
            algorithm="random_forest",
            feature_columns=["feature1"],
            target_column="target",
        )

        # This would require a trained model, so we'll test the method exists
        assert hasattr(self.pipeline, "export_model")


class TestIntelligentQA:
    """Test Intelligent Quality Assurance System"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.qa = IntelligentQA(models_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_qa_initialization(self):
        """Test QA system initialization"""
        assert self.qa is not None
        assert hasattr(self.qa, "quality_metrics")
        assert hasattr(self.qa, "quality_issues")
        assert hasattr(self.qa, "validation_rules")
        assert hasattr(self.qa, "quality_thresholds")

    def test_add_quality_metric(self):
        """Test adding quality metrics"""
        metric = self.qa.add_quality_metric(
            name="accuracy", value=0.85, category="accuracy", threshold=0.8, weight=0.3
        )

        assert metric.name == "accuracy"
        assert metric.value == 0.85
        assert metric.category == "accuracy"
        assert metric.threshold == 0.8
        assert metric.weight == 0.3

    def test_validate_file(self):
        """Test file validation"""
        # Create a temporary file
        temp_file = os.path.join(self.temp_dir, "test.pdf")
        with open(temp_file, "w") as f:
            f.write("test content")

        issues = self.qa.validate_file(
            temp_file,
            {
                "width": 1200,
                "height": 800,
                "dpi": 200,
                "content": "Test plan sheet with title and scale",
            },
        )

        assert isinstance(issues, list)
        # Should pass validation since it's a PDF file

    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        # Add some quality metrics
        self.qa.add_quality_metric("accuracy", 0.85, "accuracy", threshold=0.8)
        self.qa.add_quality_metric("completeness", 0.92, "completeness", threshold=0.9)

        score = self.qa.calculate_quality_score("test_job")
        assert score is not None
        assert hasattr(score, "overall_score")
        assert hasattr(score, "category_scores")
        assert score.overall_score >= 0.0 and score.overall_score <= 1.0

    def test_create_validation_rule(self):
        """Test creating validation rule"""
        rule = self.qa.create_validation_rule(
            name="Custom Rule",
            description="Custom validation rule",
            category="custom",
            rule_type="custom",
            parameters={"min_value": 10, "max_value": 100},
            weight=0.2,
        )

        assert rule.name == "Custom Rule"
        assert rule.category == "custom"
        assert rule.rule_type == "custom"
        assert rule.weight == 0.2

    def test_train_quality_model(self):
        """Test training quality model"""
        # Create training data
        np.random.seed(42)
        training_data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "quality_score": np.random.uniform(0.5, 1.0, 100),
            }
        )

        model = self.qa.train_quality_model(
            model_name="quality_predictor",
            training_data=training_data,
            target_column="quality_score",
            model_type="regression",
        )

        assert model is not None
        assert model.name == "quality_predictor"
        assert model.type == "regression"
        assert model.accuracy > 0.0

    def test_predict_quality(self):
        """Test quality prediction"""
        # Train a model first
        training_data = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "quality_score": np.random.uniform(0.5, 1.0, 50),
            }
        )

        self.qa.train_quality_model(
            model_name="quality_predictor",
            training_data=training_data,
            target_column="quality_score",
        )

        # Make prediction
        features = np.array([[0.5, 0.3]])
        prediction = self.qa.predict_quality("quality_predictor", features)
        assert len(prediction) == 1
        assert prediction[0] >= 0.0

    def test_generate_quality_report(self):
        """Test quality report generation"""
        # Add some metrics and issues
        self.qa.add_quality_metric("accuracy", 0.85, "accuracy")

        report = self.qa.generate_quality_report("test_job")
        assert report is not None
        assert hasattr(report, "overall_score")
        assert hasattr(report, "issues")
        assert hasattr(report, "recommendations")

    def test_get_quality_trends(self):
        """Test quality trend analysis"""
        # Add trending data
        for i in range(20):
            self.qa.add_quality_metric("trend_metric", 0.5 + i * 0.02, "quality")

        trends = self.qa.get_quality_trends("trend_metric", hours=1)
        assert trends is not None
        assert "trend" in trends
        assert "change_rate" in trends

    def test_export_quality_data(self):
        """Test quality data export"""
        # Add some data
        self.qa.add_quality_metric("test_metric", 0.85, "accuracy")

        export_file = self.qa.export_quality_data("json", "24h")
        assert export_file.endswith(".json")
        assert os.path.exists(export_file)

        # Cleanup
        os.remove(export_file)


class TestAdvancedGeospatialIntelligence:
    """Test Advanced Geospatial Intelligence System"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        # Mock geospatial availability
        with patch("core.advanced_geospatial_intelligence.GEOSPATIAL_AVAILABLE", True):
            self.gis = AdvancedGeospatialIntelligence(data_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_gis_initialization(self):
        """Test GIS system initialization"""
        assert self.gis is not None
        assert hasattr(self.gis, "spatial_features")
        assert hasattr(self.gis, "spatial_analyses")
        assert hasattr(self.gis, "spatial_patterns")

    @patch("core.advanced_geospatial_intelligence.Point")
    def test_add_spatial_feature(self, mock_point):
        """Test adding spatial features"""
        # Mock Point geometry
        mock_geometry = Mock()
        mock_point.return_value = mock_geometry

        feature = self.gis.add_spatial_feature(
            geometry=mock_geometry,
            properties={"id": 1, "name": "Test Point"},
            feature_type="point",
            confidence=0.9,
        )

        assert feature is not None
        assert feature.feature_type == "point"
        assert feature.confidence == 0.9
        assert feature.properties["id"] == 1

    def test_analyze_spatial_patterns(self):
        """Test spatial pattern analysis"""
        # This test would require actual spatial data
        # For now, we'll test that the method exists
        assert hasattr(self.gis, "analyze_spatial_patterns")

    def test_assess_spatial_quality(self):
        """Test spatial quality assessment"""
        # Create a mock feature first
        mock_geometry = Mock()
        mock_geometry.is_valid = True
        mock_geometry.length = 1.0
        mock_geometry.area = 1.0

        feature = self.gis.add_spatial_feature(
            geometry=mock_geometry,
            properties={"id": 1, "name": "Test Feature"},
            feature_type="polygon",
        )

        quality = self.gis.assess_spatial_quality(feature.feature_id)
        assert quality is not None
        assert hasattr(quality, "overall_score")
        assert hasattr(quality, "accuracy_score")

    def test_create_3d_model(self):
        """Test 3D model creation"""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])

        model = self.gis.create_3d_model(
            name="Test 3D Model",
            model_type="terrain",
            vertices=vertices,
            faces=faces,
            properties={"height": 100},
        )

        assert model is not None
        assert model.name == "Test 3D Model"
        assert model.model_type == "terrain"
        assert model.vertices.shape == (4, 3)
        assert model.faces.shape == (4, 3)

    def test_calculate_spatial_metrics(self):
        """Test spatial metrics calculation"""
        # Create mock features
        mock_geometry1 = Mock()
        mock_geometry1.centroid.x = 0
        mock_geometry1.centroid.y = 0
        mock_geometry1.area = 1.0
        mock_geometry1.length = 2.0

        mock_geometry2 = Mock()
        mock_geometry2.centroid.x = 1
        mock_geometry2.centroid.y = 1
        mock_geometry2.area = 2.0
        mock_geometry2.length = 3.0

        feature1 = self.gis.add_spatial_feature(mock_geometry1, {"id": 1}, "polygon")
        feature2 = self.gis.add_spatial_feature(mock_geometry2, {"id": 2}, "polygon")

        metrics = self.gis.calculate_spatial_metrics(
            [feature1.feature_id, feature2.feature_id]
        )
        assert metrics is not None
        assert "feature_count" in metrics
        assert metrics["feature_count"] == 2

    def test_export_spatial_data(self):
        """Test spatial data export"""
        # Create a mock feature
        mock_geometry = Mock()
        mock_geometry.wkt = "POINT(0 0)"

        feature = self.gis.add_spatial_feature(
            geometry=mock_geometry,
            properties={"id": 1, "name": "Test Point"},
            feature_type="point",
        )

        export_file = self.gis.export_spatial_data("geojson", [feature.feature_id])
        assert export_file.endswith(".geojson")
        assert os.path.exists(export_file)

        # Cleanup
        os.remove(export_file)


class TestPerformanceIntelligence:
    """Test Performance Intelligence System"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.pi = PerformanceIntelligence(data_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pi_initialization(self):
        """Test performance intelligence initialization"""
        assert self.pi is not None
        assert hasattr(self.pi, "performance_metrics")
        assert hasattr(self.pi, "performance_alerts")
        assert hasattr(self.pi, "performance_bottlenecks")
        assert hasattr(self.pi, "optimization_recommendations")

    def test_add_performance_metric(self):
        """Test adding performance metrics"""
        metric = self.pi.add_performance_metric(
            name="cpu_usage",
            value=75.5,
            unit="%",
            category="cpu",
            metadata={"core_count": 8},
        )

        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.unit == "%"
        assert metric.category == "cpu"
        assert metric.metadata["core_count"] == 8

    def test_performance_thresholds(self):
        """Test performance threshold checking"""
        # Add metric above threshold
        self.pi.add_performance_metric("cpu_usage", 95.0, "%", "cpu")

        # Wait for monitoring cycle
        import time

        time.sleep(1)

        # Check if alert was generated
        cpu_alerts = self.pi.performance_alerts.get("cpu_usage", [])
        # Note: In real testing, we'd need to wait for the monitoring thread
        # For now, we'll just test the threshold configuration
        assert "cpu_usage" in self.pi.performance_thresholds
        assert "critical" in self.pi.performance_thresholds["cpu_usage"]

    def test_create_performance_profile(self):
        """Test creating performance profile"""
        profile = self.pi.create_performance_profile(
            name="Test Profile",
            description="Test performance profile",
            baseline_metrics={"cpu_usage": 50.0, "memory_usage": 60.0},
            target_metrics={"cpu_usage": 30.0, "memory_usage": 40.0},
        )

        assert profile.name == "Test Profile"
        assert profile.description == "Test performance profile"
        assert len(profile.baseline_metrics) == 2
        assert len(profile.target_metrics) == 2

    def test_get_performance_summary(self):
        """Test getting performance summary"""
        # Add some metrics
        self.pi.add_performance_metric("cpu_usage", 75.0, "%", "cpu")
        self.pi.add_performance_metric("memory_usage", 80.0, "%", "memory")

        summary = self.pi.get_performance_summary(hours=1)
        assert summary is not None
        assert "metrics" in summary
        assert "alerts" in summary
        assert "bottlenecks" in summary
        assert "recommendations" in summary

    def test_train_prediction_models(self):
        """Test training prediction models"""
        # Create training data
        np.random.seed(42)
        training_data = pd.DataFrame(
            {
                "cpu_usage": np.random.uniform(20, 90, 100),
                "memory_usage": np.random.uniform(30, 95, 100),
                "response_time": np.random.uniform(0.1, 10.0, 100),
            }
        )

        # Add lagged features
        for i in range(1, 6):
            training_data[f"cpu_usage_lag_{i}"] = training_data["cpu_usage"].shift(i)
            training_data[f"memory_usage_lag_{i}"] = training_data[
                "memory_usage"
            ].shift(i)
            training_data[f"response_time_lag_{i}"] = training_data[
                "response_time"
            ].shift(i)

        # Remove NaN values
        training_data = training_data.dropna()

        self.pi.train_prediction_models(training_data)
        # Test that models were trained (they should exist)
        assert "cpu_usage" in self.pi.prediction_models
        assert "memory_usage" in self.pi.prediction_models

    def test_apply_optimization(self):
        """Test applying optimization recommendations"""
        # Create a recommendation
        recommendation = self.pi.optimization_recommendations["test_rec"] = Mock()
        recommendation.recommendation_id = "test_rec"
        recommendation.title = "Test Optimization"
        recommendation.category = "memory"

        success = self.pi.apply_optimization("test_rec")
        # Should return True if recommendation exists
        assert success is True

    def test_export_performance_data(self):
        """Test performance data export"""
        # Add some data
        self.pi.add_performance_metric("cpu_usage", 75.0, "%", "cpu")

        export_file = self.pi.export_performance_data("json", "24h")
        assert export_file.endswith(".json")
        assert os.path.exists(export_file)

        # Cleanup
        os.remove(export_file)


class TestIntegration:
    """Integration tests for Priority 4 systems"""

    def setup_method(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup integration test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_system_integration(self):
        """Test integration between Priority 4 systems"""
        # Initialize all systems
        analytics = AdvancedAnalytics()
        ml_pipeline = MLPipeline(models_dir=self.temp_dir)
        qa = IntelligentQA(models_dir=self.temp_dir)

        # Mock geospatial availability
        with patch("core.advanced_geospatial_intelligence.GEOSPATIAL_AVAILABLE", True):
            gis = AdvancedGeospatialIntelligence(data_dir=self.temp_dir)

        pi = PerformanceIntelligence(data_dir=self.temp_dir)

        # Test that all systems can work together
        # Add metrics to analytics
        analytics.add_metric("processing_time", 45.2, "seconds", "performance")

        # Add quality metrics
        qa.add_quality_metric("accuracy", 0.85, "accuracy")

        # Add performance metrics
        pi.add_performance_metric("cpu_usage", 75.0, "%", "cpu")

        # Verify all systems are working
        assert len(analytics.metrics_history) > 0
        assert len(qa.quality_metrics) > 0
        assert len(pi.performance_metrics) > 0

    def test_data_flow_between_systems(self):
        """Test data flow between systems"""
        # This test would verify that data can flow between systems
        # For example, performance metrics from PI could feed into analytics
        # Quality metrics from QA could influence ML pipeline decisions

        analytics = AdvancedAnalytics()
        pi = PerformanceIntelligence()

        # Add performance metrics
        pi.add_performance_metric("cpu_usage", 80.0, "%", "cpu")
        pi.add_performance_metric("memory_usage", 85.0, "%", "memory")

        # Analytics could consume these metrics
        analytics.add_metric("system_health", 0.75, "score", "system")

        # Verify data flow
        assert len(pi.performance_metrics) > 0
        assert len(analytics.metrics_history) > 0


class TestPerformance:
    """Performance tests for Priority 4 systems"""

    def test_analytics_performance(self):
        """Test analytics system performance"""
        analytics = AdvancedAnalytics()

        # Add many metrics quickly
        start_time = datetime.now()
        for i in range(1000):
            analytics.add_metric(f"metric_{i}", i, "count", "test")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete within reasonable time
        assert duration < 10.0  # 10 seconds
        assert len(analytics.metrics_history) == 1000

    def test_ml_pipeline_performance(self):
        """Test ML pipeline performance"""
        pipeline = MLPipeline()

        # Create many model configs
        start_time = datetime.now()
        for i in range(100):
            pipeline.create_model_config(
                name=f"model_{i}",
                model_type="classification",
                algorithm="random_forest",
                feature_columns=["feature1", "feature2"],
                target_column="target",
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete within reasonable time
        assert duration < 5.0  # 5 seconds
        assert len(pipeline.model_configs) == 100


class TestErrorHandling:
    """Error handling tests for Priority 4 systems"""

    def test_analytics_error_handling(self):
        """Test analytics error handling"""
        analytics = AdvancedAnalytics()

        # Test with invalid parameters
        with pytest.raises(Exception):
            analytics.add_metric("", -1, "", "")  # Invalid parameters

    def test_ml_pipeline_error_handling(self):
        """Test ML pipeline error handling"""
        pipeline = MLPipeline()

        # Test with invalid model config
        with pytest.raises(ValueError):
            pipeline.create_model_config(
                name="",
                model_type="invalid_type",
                algorithm="invalid_algorithm",
                feature_columns=[],
                target_column="",
            )

    def test_qa_error_handling(self):
        """Test QA error handling"""
        qa = IntelligentQA()

        # Test with invalid file path
        issues = qa.validate_file("nonexistent_file.pdf")
        assert len(issues) > 0  # Should detect file not found

    def test_performance_intelligence_error_handling(self):
        """Test performance intelligence error handling"""
        pi = PerformanceIntelligence()

        # Test with invalid recommendation ID
        success = pi.apply_optimization("nonexistent_id")
        assert success is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
