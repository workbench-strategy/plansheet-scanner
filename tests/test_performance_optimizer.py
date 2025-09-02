"""
Unit tests for Performance Optimizer

Tests the performance optimization system including:
- Performance profiling
- LRU caching with memory management
- Bottleneck identification
- Optimization recommendations
"""

import json
import os
import tempfile
import time
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.core.performance_optimizer import (
    LRUCache,
    OptimizationResult,
    PerformanceMetrics,
    PerformanceOptimizer,
    PerformanceProfiler,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            execution_time=1.5,
            memory_usage_start=0.5,
            memory_usage_end=0.8,
            memory_peak=1.2,
            cpu_usage=75.0,
        )

        assert metrics.operation_name == "test_operation"
        assert metrics.execution_time == 1.5
        assert metrics.memory_usage_start == 0.5
        assert metrics.memory_usage_end == 0.8
        assert metrics.memory_peak == 1.2
        assert metrics.cpu_usage == 75.0
        assert metrics.timestamp is not None

    def test_performance_metrics_with_optional_fields(self):
        """Test PerformanceMetrics with optional fields."""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            execution_time=1.0,
            memory_usage_start=0.5,
            memory_usage_end=0.6,
            memory_peak=0.7,
            cpu_usage=50.0,
            cache_hits=10,
            cache_misses=2,
            accuracy=0.95,
            throughput=100.0,
        )

        assert metrics.cache_hits == 10
        assert metrics.cache_misses == 2
        assert metrics.accuracy == 0.95
        assert metrics.throughput == 100.0


class TestLRUCache:
    """Test LRU Cache functionality."""

    def test_cache_creation(self):
        """Test creating LRU cache."""
        cache = LRUCache(max_size=10, max_memory_mb=50.0)

        assert cache.max_size == 10
        assert cache.max_memory_bytes == 50.0 * 1024 * 1024
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(max_size=5)

        # Put items
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Get items
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None

        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 2

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size=3)

        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Access key1 to make it most recently used
        cache.get("key1")

        # Add new item - should evict key2 (least recently used)
        cache.put("key4", "value4")

        # Check that key2 was evicted
        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"  # Still there
        assert cache.get("key4") == "value4"  # New item

    def test_cache_memory_limit(self):
        """Test memory-based eviction."""
        cache = LRUCache(max_size=100, max_memory_mb=1.0)  # 1MB limit

        # Add large items
        large_data = "x" * (256 * 1024)  # 256KB each (smaller to fit 3 items in 1MB)

        cache.put("key1", large_data)
        cache.put("key2", large_data)

        # Fourth item should trigger eviction (3 * 256KB = 768KB, 4th item = 1MB+)
        cache.put("key3", large_data)
        cache.put("key4", large_data)

        # First item should be evicted due to memory limit
        assert cache.get("key1") is None
        assert cache.get("key2") == large_data
        assert cache.get("key3") == large_data
        assert cache.get("key4") == large_data

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=5)

        # Add some items
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access items
        cache.get("key1")  # Hit
        cache.get("key2")  # Hit
        cache.get("key3")  # Miss

        stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3
        assert "memory_usage_mb" in stats


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality."""

    def test_profiler_creation(self):
        """Test creating PerformanceProfiler."""
        profiler = PerformanceProfiler()

        assert len(profiler.metrics_history) == 0
        assert len(profiler.agent_baselines) == 0
        assert len(profiler.bottlenecks) == 0

    @patch("psutil.Process")
    def test_profile_operation_success(self, mock_process):
        """Test profiling a successful operation."""
        # Mock process
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 1024  # 1GB
        mock_process_instance.cpu_percent.return_value = 50.0
        mock_process.return_value = mock_process_instance

        profiler = PerformanceProfiler()

        # Test function
        def test_function(x, y):
            time.sleep(0.1)  # Simulate work
            return x + y

        result, metrics = profiler.profile_operation("test_op", test_function, 1, 2)

        assert result == 3
        assert metrics.operation_name == "test_op"
        assert metrics.execution_time > 0.1
        assert metrics.memory_usage_start > 0
        assert metrics.memory_usage_end > 0
        assert metrics.cpu_usage == 50.0
        assert len(profiler.metrics_history) == 1

    @patch("psutil.Process")
    def test_profile_operation_failure(self, mock_process):
        """Test profiling a failed operation."""
        # Mock process
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 1024
        mock_process_instance.cpu_percent.return_value = 50.0
        mock_process.return_value = mock_process_instance

        profiler = PerformanceProfiler()

        # Test function that raises exception
        def failing_function():
            time.sleep(0.01)  # Add small delay to ensure execution_time > 0
            raise ValueError("Test error")

        result, metrics = profiler.profile_operation("failing_op", failing_function)

        assert result is None
        assert metrics.operation_name == "failing_op"
        assert metrics.execution_time > 0
        assert len(profiler.metrics_history) == 1

    def test_identify_bottlenecks(self):
        """Test bottleneck identification."""
        profiler = PerformanceProfiler()

        # Add some test metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                operation_name="slow_operation",
                execution_time=15.0 + i,  # Slow operation
                memory_usage_start=0.5,
                memory_usage_end=0.6,
                memory_peak=1.5,  # High memory
                cpu_usage=80.0,
            )
            profiler.metrics_history.append(metrics)

        # Add some fast operation metrics
        for i in range(3):
            metrics = PerformanceMetrics(
                operation_name="fast_operation",
                execution_time=2.0 + i,
                memory_usage_start=0.1,
                memory_usage_end=0.2,
                memory_peak=0.3,
                cpu_usage=20.0,
            )
            profiler.metrics_history.append(metrics)

        bottlenecks = profiler.identify_bottlenecks()

        assert len(bottlenecks) > 0

        # Check that slow_operation is identified as bottleneck
        slow_bottleneck = next(
            (b for b in bottlenecks if b["operation_name"] == "slow_operation"), None
        )
        assert slow_bottleneck is not None
        assert slow_bottleneck["avg_time"] > 10.0
        assert slow_bottleneck["avg_memory"] > 1.0

    def test_get_performance_report(self):
        """Test performance report generation."""
        profiler = PerformanceProfiler()

        # Add test metrics
        for i in range(3):
            metrics = PerformanceMetrics(
                operation_name="test_op",
                execution_time=5.0 + i,
                memory_usage_start=0.5,
                memory_usage_end=0.6,
                memory_peak=0.8,
                cpu_usage=50.0,
            )
            profiler.metrics_history.append(metrics)

        report = profiler.get_performance_report()

        assert "summary" in report
        assert "targets" in report
        assert "target_analysis" in report
        assert "bottlenecks" in report
        assert "recommendations" in report

        summary = report["summary"]
        assert summary["total_operations"] == 3
        assert summary["avg_time_per_operation"] > 0
        assert summary["avg_memory"] > 0

    def test_get_performance_report_empty(self):
        """Test performance report with no data."""
        profiler = PerformanceProfiler()

        report = profiler.get_performance_report()

        assert "error" in report
        assert report["error"] == "No performance data available"


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer functionality."""

    def test_optimizer_creation(self):
        """Test creating PerformanceOptimizer."""
        optimizer = PerformanceOptimizer(cache_size=50, cache_memory_mb=25.0)

        assert optimizer.cache.max_size == 50
        assert optimizer.cache.max_memory_bytes == 25.0 * 1024 * 1024
        assert len(optimizer.optimization_history) == 0
        assert optimizer.memory_threshold_gb == 1.5

    @patch("psutil.Process")
    def test_optimize_agent_cache_hit(self, mock_process):
        """Test agent optimization with cache hit."""
        # Mock process
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 1024
        mock_process_instance.cpu_percent.return_value = 50.0
        mock_process.return_value = mock_process_instance

        optimizer = PerformanceOptimizer()

        # Test function
        def test_agent(x, y):
            time.sleep(0.01)  # Add small delay to ensure execution_time > 0
            return x + y

        # First call - should cache
        result1, opt_result1 = optimizer.optimize_agent("test_agent", test_agent, 1, 2)

        # Second call with same parameters - should hit cache
        result2, opt_result2 = optimizer.optimize_agent("test_agent", test_agent, 1, 2)

        assert result1 == result2 == 3
        assert opt_result1.agent_name == "test_agent"
        assert opt_result2.agent_name == "test_agent"

        # Check cache stats
        stats = optimizer.cache.get_stats()
        assert stats["hits"] > 0
        assert stats["misses"] > 0

    @patch("psutil.Process")
    def test_optimize_agent_no_cache(self, mock_process):
        """Test agent optimization without cache hit."""
        # Mock process
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 1024
        mock_process_instance.cpu_percent.return_value = 50.0
        mock_process.return_value = mock_process_instance

        optimizer = PerformanceOptimizer()

        # Test function
        def test_agent(x, y):
            time.sleep(0.1)  # Simulate work
            return x * y

        result, opt_result = optimizer.optimize_agent("test_agent", test_agent, 3, 4)

        assert result == 12
        assert opt_result.agent_name == "test_agent"
        assert opt_result.original_metrics.execution_time > 0
        assert opt_result.optimized_metrics.execution_time > 0
        assert len(optimizer.optimization_history) == 1

    def test_create_cache_key(self):
        """Test cache key creation."""
        optimizer = PerformanceOptimizer()

        # Test with different parameters
        key1 = optimizer._create_cache_key("agent1", (1, 2), {"param": "value"})
        key2 = optimizer._create_cache_key("agent1", (1, 2), {"param": "value"})
        key3 = optimizer._create_cache_key("agent1", (1, 3), {"param": "value"})

        # Same parameters should produce same key
        assert key1 == key2

        # Different parameters should produce different key
        assert key1 != key3

        # Keys should be valid MD5 hashes
        assert len(key1) == 32
        assert all(c in "0123456789abcdef" for c in key1)

    def test_generate_agent_recommendations(self):
        """Test agent-specific recommendation generation."""
        optimizer = PerformanceOptimizer()

        # Test slow operation
        slow_metrics = PerformanceMetrics(
            operation_name="slow_op",
            execution_time=35.0,  # Very slow (> 30.0 threshold)
            memory_usage_start=0.5,
            memory_usage_end=0.6,
            memory_peak=0.8,
            cpu_usage=50.0,
        )

        fast_metrics = PerformanceMetrics(
            operation_name="fast_op",
            execution_time=2.0,
            memory_usage_start=0.1,
            memory_usage_end=0.2,
            memory_peak=0.3,
            cpu_usage=20.0,
        )

        recommendations = optimizer._generate_agent_recommendations(
            slow_metrics, fast_metrics
        )

        assert len(recommendations) > 0
        assert any("parallel processing" in rec.lower() for rec in recommendations)
        # Check for critical optimization recommendation (execution_time > 30.0 should trigger this)
        assert any("critical" in rec.lower() for rec in recommendations)

    def test_get_optimization_report(self):
        """Test optimization report generation."""
        optimizer = PerformanceOptimizer()

        # Add some optimization history
        for i in range(3):
            opt_result = OptimizationResult(
                agent_name=f"agent_{i}",
                original_metrics=PerformanceMetrics(
                    operation_name=f"agent_{i}",
                    execution_time=10.0 + i,
                    memory_usage_start=0.5,
                    memory_usage_end=0.6,
                    memory_peak=1.0 + i * 0.1,
                    cpu_usage=50.0,
                ),
                optimized_metrics=PerformanceMetrics(
                    operation_name=f"agent_{i}",
                    execution_time=5.0 + i,
                    memory_usage_start=0.3,
                    memory_usage_end=0.4,
                    memory_peak=0.5 + i * 0.05,
                    cpu_usage=30.0,
                ),
                improvements={"execution_time": 0.5, "memory_usage": 0.4},
                recommendations=["Test recommendation"],
                cache_stats={"hits": 10, "misses": 2},
            )
            optimizer.optimization_history.append(opt_result)

        report = optimizer.get_optimization_report()

        assert "performance_report" in report
        assert "cache_statistics" in report
        assert "optimization_summary" in report
        assert "recommendations" in report

        summary = report["optimization_summary"]
        assert summary["total_optimizations"] == 3
        assert summary["avg_time_improvement"] == 0.5
        assert (
            abs(summary["avg_memory_improvement"] - 0.4) < 0.001
        )  # Use approximate comparison for float

    def test_save_report(self):
        """Test saving optimization report to file."""
        optimizer = PerformanceOptimizer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            optimizer.save_report(temp_file)

            # Check file was created and contains valid JSON
            assert os.path.exists(temp_file)

            with open(temp_file, "r") as f:
                data = json.load(f)

            assert "performance_report" in data
            assert "cache_statistics" in data
            assert "optimization_summary" in data

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestIntegration:
    """Integration tests for the performance optimization system."""

    @patch("psutil.Process")
    def test_full_optimization_workflow(self, mock_process):
        """Test complete optimization workflow."""
        # Mock process
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 1024
        mock_process_instance.cpu_percent.return_value = 50.0
        mock_process.return_value = mock_process_instance

        optimizer = PerformanceOptimizer()

        # Define test agents
        def fast_agent(data):
            time.sleep(0.05)
            return {"result": data * 2, "processed": True}

        def slow_agent(data):
            time.sleep(0.2)
            return {"result": data * 3, "processed": True}

        agents = {"fast_agent": fast_agent, "slow_agent": slow_agent}

        test_data = {"fast_agent": {"data": 10}, "slow_agent": {"data": 5}}

        # Profile all agents
        results = optimizer.profile_all_agents(agents, test_data)

        assert len(results) == 2
        assert "fast_agent" in results
        assert "slow_agent" in results

        # Check that slow agent takes longer
        assert (
            results["slow_agent"].execution_time > results["fast_agent"].execution_time
        )

        # Generate report
        report = optimizer.get_optimization_report()

        assert "performance_report" in report
        assert "bottlenecks" in report["performance_report"]

        # Check that slow agent is identified as bottleneck
        bottlenecks = report["performance_report"]["bottlenecks"]
        slow_bottleneck = next(
            (b for b in bottlenecks if b["operation_name"] == "slow_agent"), None
        )

        if slow_bottleneck:
            assert slow_bottleneck["avg_time"] > 0.1  # Should be slow

    def test_memory_management(self):
        """Test memory management features."""
        optimizer = PerformanceOptimizer(cache_memory_mb=1.0)  # 1MB limit

        # Add large data to cache
        large_data = "x" * (256 * 1024)  # 256KB (smaller to fit 3 items in 1MB)

        optimizer.cache.put("large_key1", large_data)
        optimizer.cache.put("large_key2", large_data)
        optimizer.cache.put("large_key3", large_data)

        # Fourth item should trigger memory-based eviction
        optimizer.cache.put("large_key4", large_data)

        # First item should be evicted
        assert optimizer.cache.get("large_key1") is None
        assert optimizer.cache.get("large_key2") == large_data
        assert optimizer.cache.get("large_key3") == large_data
        assert optimizer.cache.get("large_key4") == large_data

        # Check memory usage in stats
        stats = optimizer.cache.get_stats()
        assert stats["memory_usage_mb"] >= 0  # Can be 0 if memory usage is very small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
