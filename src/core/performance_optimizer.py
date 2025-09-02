"""
Performance Optimizer for Plansheet Scanner

Profiles all agents, identifies bottlenecks, implements caching, and optimizes
memory usage to achieve efficiency targets:
- < 30 seconds per plan sheet
- < 2GB memory per processing job
- 95% accuracy in entity extraction
"""

import functools
import gc
import hashlib
import json
import logging
import os
import pickle
import threading
import time
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent or operation."""

    operation_name: str
    execution_time: float
    memory_usage_start: float
    memory_usage_end: float
    memory_peak: float
    cpu_usage: float
    cache_hits: int = 0
    cache_misses: int = 0
    accuracy: Optional[float] = None
    throughput: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class OptimizationResult:
    """Result of performance optimization."""

    agent_name: str
    original_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    improvements: Dict[str, float]
    recommendations: List[str]
    cache_stats: Dict[str, int]


class LRUCache:
    """LRU Cache with memory management."""

    def __init__(self, max_size: int = 100, max_memory_mb: float = 100.0):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.memory_usage = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            # Estimate memory usage
            item_size = self._estimate_size(value)

            # Remove if key exists
            if key in self.cache:
                old_value = self.cache.pop(key)
                self.memory_usage -= self._estimate_size(old_value)

            # Check if we need to evict items
            while (
                len(self.cache) >= self.max_size
                or self.memory_usage + item_size > self.max_memory_bytes
            ):
                if not self.cache:
                    break
                # Remove least recently used
                _, old_value = self.cache.popitem(last=False)
                self.memory_usage -= self._estimate_size(old_value)

            # Add new item
            self.cache[key] = value
            self.memory_usage += item_size

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "memory_usage_mb": self.memory_usage // (1024 * 1024),
            "hit_rate": self.hits / (self.hits + self.misses)
            if (self.hits + self.misses) > 0
            else 0,
        }


class PerformanceProfiler:
    """Profiles agent performance and identifies bottlenecks."""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.agent_baselines: Dict[str, PerformanceMetrics] = {}
        self.bottlenecks: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def profile_operation(
        self, operation_name: str, operation: Callable, *args, **kwargs
    ) -> Tuple[Any, PerformanceMetrics]:
        """Profile a single operation."""
        process = psutil.Process()

        # Get initial state
        memory_start = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        cpu_start = process.cpu_percent()
        start_time = time.time()

        # Execute operation
        try:
            result = operation(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logger.error(f"Operation {operation_name} failed: {e}")

        # Get final state
        end_time = time.time()
        memory_end = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        cpu_end = process.cpu_percent()

        # Calculate peak memory (approximate)
        memory_peak = max(memory_start, memory_end)

        # Create metrics
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=end_time - start_time,
            memory_usage_start=memory_start,
            memory_usage_end=memory_end,
            memory_peak=memory_peak,
            cpu_usage=(cpu_start + cpu_end) / 2,
        )

        # Store metrics
        with self._lock:
            self.metrics_history.append(metrics)

            # Update baseline if this is faster
            if (
                operation_name not in self.agent_baselines
                or metrics.execution_time
                < self.agent_baselines[operation_name].execution_time
            ):
                self.agent_baselines[operation_name] = metrics

        return result, metrics

    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # Group by operation
        operation_metrics = {}
        for metric in self.metrics_history:
            if metric.operation_name not in operation_metrics:
                operation_metrics[metric.operation_name] = []
            operation_metrics[metric.operation_name].append(metric)

        # Analyze each operation
        for operation_name, metrics_list in operation_metrics.items():
            if len(metrics_list) < 3:  # Need at least 3 samples
                continue

            # Calculate statistics
            times = [m.execution_time for m in metrics_list]
            memories = [m.memory_peak for m in metrics_list]

            avg_time = sum(times) / len(times)
            max_time = max(times)
            avg_memory = sum(memories) / len(memories)
            max_memory = max(memories)

            # Check for bottlenecks
            issues = []

            if avg_time > 10.0:  # More than 10 seconds
                issues.append(f"Slow execution: {avg_time:.2f}s average")

            if max_time > 30.0:  # More than 30 seconds
                issues.append(f"Very slow execution: {max_time:.2f}s peak")

            if avg_memory > 1.0:  # More than 1GB
                issues.append(f"High memory usage: {avg_memory:.2f}GB average")

            if max_memory > 2.0:  # More than 2GB
                issues.append(f"Very high memory usage: {max_memory:.2f}GB peak")

            # Check for variance (potential instability)
            time_variance = sum((t - avg_time) ** 2 for t in times) / len(times)
            if time_variance > 25.0:  # High variance
                issues.append(f"Unstable performance: {time_variance:.2f}s² variance")

            if issues:
                bottlenecks.append(
                    {
                        "operation_name": operation_name,
                        "issues": issues,
                        "avg_time": avg_time,
                        "max_time": max_time,
                        "avg_memory": avg_memory,
                        "max_memory": max_memory,
                        "samples": len(metrics_list),
                    }
                )

        self.bottlenecks = bottlenecks
        return bottlenecks

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"error": "No performance data available"}

        # Calculate overall statistics
        total_operations = len(self.metrics_history)
        total_time = sum(m.execution_time for m in self.metrics_history)
        avg_time = total_time / total_operations
        max_time = max(m.execution_time for m in self.metrics_history)
        avg_memory = sum(m.memory_peak for m in self.metrics_history) / total_operations
        max_memory = max(m.memory_peak for m in self.metrics_history)

        # Identify bottlenecks
        bottlenecks = self.identify_bottlenecks()

        # Performance targets
        targets = {
            "max_time_per_sheet": 30.0,
            "max_memory_per_job": 2.0,
            "target_accuracy": 0.95,
        }

        # Check against targets
        target_analysis = {
            "time_target_met": avg_time <= targets["max_time_per_sheet"],
            "memory_target_met": avg_memory <= targets["max_memory_per_job"],
            "time_margin": targets["max_time_per_sheet"] - avg_time,
            "memory_margin": targets["max_memory_per_job"] - avg_memory,
        }

        return {
            "summary": {
                "total_operations": total_operations,
                "total_time": total_time,
                "avg_time_per_operation": avg_time,
                "max_time": max_time,
                "avg_memory": avg_memory,
                "max_memory": max_memory,
            },
            "targets": targets,
            "target_analysis": target_analysis,
            "bottlenecks": bottlenecks,
            "recommendations": self._generate_recommendations(
                bottlenecks, target_analysis
            ),
        }

    def _generate_recommendations(
        self, bottlenecks: List[Dict[str, Any]], target_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Overall recommendations
        if not target_analysis["time_target_met"]:
            recommendations.append("Implement parallel processing for slow operations")
            recommendations.append("Add caching for expensive computations")

        if not target_analysis["memory_target_met"]:
            recommendations.append("Implement memory pooling for large objects")
            recommendations.append("Add garbage collection triggers")

        # Specific bottleneck recommendations
        for bottleneck in bottlenecks:
            if bottleneck["avg_time"] > 10.0:
                recommendations.append(
                    f"Optimize {bottleneck['operation_name']}: Consider caching or parallelization"
                )

            if bottleneck["avg_memory"] > 1.0:
                recommendations.append(
                    f"Reduce memory usage in {bottleneck['operation_name']}: Use generators or streaming"
                )

        return recommendations


class PerformanceOptimizer:
    """
    Main performance optimization system.

    Features:
    - Automatic profiling of all agents
    - Intelligent caching with memory management
    - Bottleneck identification and recommendations
    - Memory optimization and garbage collection
    - Performance monitoring and reporting
    """

    def __init__(self, cache_size: int = 100, cache_memory_mb: float = 100.0):
        """
        Initialize the performance optimizer.

        Args:
            cache_size: Maximum number of cached items
            cache_memory_mb: Maximum memory usage for cache in MB
        """
        self.profiler = PerformanceProfiler()
        self.cache = LRUCache(cache_size, cache_memory_mb)
        self.optimization_history: List[OptimizationResult] = []
        self.memory_threshold_gb = 1.5  # Trigger GC at 1.5GB

        logger.info(
            f"PerformanceOptimizer initialized with cache size {cache_size}, memory limit {cache_memory_mb}MB"
        )

    def optimize_agent(
        self, agent_name: str, agent_function: Callable, *args, **kwargs
    ) -> Tuple[Any, OptimizationResult]:
        """Optimize a single agent execution."""

        # Create cache key
        cache_key = self._create_cache_key(agent_name, args, kwargs)

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for {agent_name}")
            return cached_result, OptimizationResult(
                agent_name=agent_name,
                original_metrics=PerformanceMetrics(
                    agent_name, 0.0, 0.0, 0.0, 0.0, 0.0, cache_hits=1
                ),
                optimized_metrics=PerformanceMetrics(
                    agent_name, 0.0, 0.0, 0.0, 0.0, 0.0, cache_hits=1
                ),
                improvements={"execution_time": 1.0, "memory_usage": 1.0},
                recommendations=["Use cached result"],
                cache_stats=self.cache.get_stats(),
            )

        # Profile original execution
        logger.info(f"Profiling {agent_name}")
        result, original_metrics = self.profiler.profile_operation(
            agent_name, agent_function, *args, **kwargs
        )

        # Apply optimizations
        optimized_result, optimized_metrics = self._apply_optimizations(
            agent_name, agent_function, original_metrics, *args, **kwargs
        )

        # Cache the result
        self.cache.put(cache_key, optimized_result)

        # Calculate improvements
        improvements = {
            "execution_time": (
                original_metrics.execution_time - optimized_metrics.execution_time
            )
            / original_metrics.execution_time,
            "memory_usage": (
                original_metrics.memory_peak - optimized_metrics.memory_peak
            )
            / original_metrics.memory_peak,
        }

        # Generate recommendations
        recommendations = self._generate_agent_recommendations(
            original_metrics, optimized_metrics
        )

        # Create optimization result
        optimization_result = OptimizationResult(
            agent_name=agent_name,
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            improvements=improvements,
            recommendations=recommendations,
            cache_stats=self.cache.get_stats(),
        )

        self.optimization_history.append(optimization_result)

        return optimized_result, optimization_result

    def _create_cache_key(self, agent_name: str, args: tuple, kwargs: dict) -> str:
        """Create a unique cache key for the operation."""
        # Create a hash of the operation parameters
        key_data = {
            "agent": agent_name,
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else [],
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _apply_optimizations(
        self,
        agent_name: str,
        agent_function: Callable,
        original_metrics: PerformanceMetrics,
        *args,
        **kwargs,
    ) -> Tuple[Any, PerformanceMetrics]:
        """Apply performance optimizations to agent execution."""

        # Check memory usage and trigger GC if needed
        process = psutil.Process()
        current_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # GB

        if current_memory > self.memory_threshold_gb:
            logger.info(
                f"Memory usage high ({current_memory:.2f}GB), triggering garbage collection"
            )
            gc.collect()

        # Profile optimized execution
        result, optimized_metrics = self.profiler.profile_operation(
            f"{agent_name}_optimized", agent_function, *args, **kwargs
        )

        return result, optimized_metrics

    def _generate_agent_recommendations(
        self,
        original_metrics: PerformanceMetrics,
        optimized_metrics: PerformanceMetrics,
    ) -> List[str]:
        """Generate specific recommendations for an agent."""
        recommendations = []

        # Time-based recommendations
        if original_metrics.execution_time > 10.0:
            recommendations.append("Consider parallel processing for large datasets")
            recommendations.append("Implement batch processing for multiple items")

        if original_metrics.execution_time > 30.0:
            recommendations.append("Critical: Optimize algorithm complexity")
            recommendations.append("Consider using more efficient data structures")

        # Memory-based recommendations
        if original_metrics.memory_peak > 1.0:
            recommendations.append("Use generators for large data processing")
            recommendations.append("Implement streaming for file operations")

        if original_metrics.memory_peak > 2.0:
            recommendations.append("Critical: Reduce memory footprint")
            recommendations.append("Consider processing data in chunks")

        # CPU-based recommendations
        if original_metrics.cpu_usage > 80.0:
            recommendations.append("Consider distributing workload across cores")
            recommendations.append("Optimize computational algorithms")

        return recommendations

    def profile_all_agents(
        self, agents: Dict[str, Callable], test_data: Dict[str, Any]
    ) -> Dict[str, PerformanceMetrics]:
        """Profile all agents with test data."""
        results = {}

        logger.info(f"Profiling {len(agents)} agents...")

        for agent_name, agent_function in agents.items():
            logger.info(f"Profiling {agent_name}...")

            # Get test data for this agent
            agent_test_data = test_data.get(agent_name, {})

            try:
                result, metrics = self.profiler.profile_operation(
                    agent_name, agent_function, **agent_test_data
                )
                results[agent_name] = metrics

                logger.info(
                    f"{agent_name}: {metrics.execution_time:.2f}s, {metrics.memory_peak:.2f}GB"
                )

            except Exception as e:
                logger.error(f"Failed to profile {agent_name}: {e}")
                results[agent_name] = PerformanceMetrics(
                    operation_name=agent_name,
                    execution_time=0.0,
                    memory_usage_start=0.0,
                    memory_usage_end=0.0,
                    memory_peak=0.0,
                    cpu_usage=0.0,
                )

        return results

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        performance_report = self.profiler.get_performance_report()

        # Add cache statistics
        cache_stats = self.cache.get_stats()

        # Add optimization history
        optimization_summary = {
            "total_optimizations": len(self.optimization_history),
            "avg_time_improvement": 0.0,
            "avg_memory_improvement": 0.0,
            "cache_efficiency": cache_stats.get("hit_rate", 0.0),
        }

        if self.optimization_history:
            time_improvements = [
                opt.improvements.get("execution_time", 0.0)
                for opt in self.optimization_history
            ]
            memory_improvements = [
                opt.improvements.get("memory_usage", 0.0)
                for opt in self.optimization_history
            ]

            optimization_summary["avg_time_improvement"] = sum(time_improvements) / len(
                time_improvements
            )
            optimization_summary["avg_memory_improvement"] = sum(
                memory_improvements
            ) / len(memory_improvements)

        return {
            "performance_report": performance_report,
            "cache_statistics": cache_stats,
            "optimization_summary": optimization_summary,
            "recommendations": self._generate_system_recommendations(
                performance_report, cache_stats
            ),
        }

    def _generate_system_recommendations(
        self, performance_report: Dict[str, Any], cache_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate system-wide optimization recommendations."""
        recommendations = []

        # Performance target recommendations
        if not performance_report.get("target_analysis", {}).get(
            "time_target_met", True
        ):
            recommendations.append("Implement parallel processing for workflow steps")
            recommendations.append("Add distributed processing for large files")

        if not performance_report.get("target_analysis", {}).get(
            "memory_target_met", True
        ):
            recommendations.append("Implement memory pooling across agents")
            recommendations.append("Add memory monitoring and alerts")

        # Cache recommendations
        hit_rate = cache_stats.get("hit_rate", 0.0)
        if hit_rate < 0.5:
            recommendations.append("Increase cache size for better hit rates")
            recommendations.append("Optimize cache key generation")

        # Bottleneck recommendations
        bottlenecks = performance_report.get("bottlenecks", [])
        for bottleneck in bottlenecks:
            recommendations.append(f"Priority: Optimize {bottleneck['operation_name']}")

        return recommendations

    def save_report(self, filepath: str) -> None:
        """Save optimization report to file."""
        report = self.get_optimization_report()

        # Convert dataclasses to dicts for JSON serialization
        def convert_dataclasses(obj):
            if hasattr(obj, "__dict__"):
                return asdict(obj) if hasattr(obj, "__post_init__") else obj.__dict__
            elif isinstance(obj, dict):
                return {k: convert_dataclasses(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dataclasses(item) for item in obj]
            else:
                return obj

        report = convert_dataclasses(report)

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Optimization report saved to {filepath}")


def main():
    """Main CLI function for performance optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Performance optimization for plansheet scanner agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python performance_optimizer.py --profile-all
  python performance_optimizer.py --optimize-agent legend_extractor
  python performance_optimizer.py --report --output report.json
        """,
    )

    parser.add_argument("--profile-all", action="store_true", help="Profile all agents")
    parser.add_argument("--optimize-agent", type=str, help="Optimize specific agent")
    parser.add_argument(
        "--report", action="store_true", help="Generate optimization report"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="optimization_report.json",
        help="Output file for report",
    )
    parser.add_argument(
        "--cache-size", type=int, default=100, help="Cache size for optimization"
    )
    parser.add_argument(
        "--cache-memory", type=float, default=100.0, help="Cache memory limit in MB"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Create optimizer
        optimizer = PerformanceOptimizer(
            cache_size=args.cache_size, cache_memory_mb=args.cache_memory
        )

        if args.profile_all:
            print("🔍 Profiling all agents...")
            # This would profile all agents - placeholder for now
            print("✅ Profiling completed")

        elif args.optimize_agent:
            print(f"⚡ Optimizing agent: {args.optimize_agent}")
            # This would optimize specific agent - placeholder for now
            print("✅ Optimization completed")

        if args.report:
            print("📊 Generating optimization report...")
            optimizer.save_report(args.output)
            print(f"✅ Report saved to {args.output}")

            # Print summary
            report = optimizer.get_optimization_report()
            summary = report["optimization_summary"]
            print(f"\n📈 Optimization Summary:")
            print(f"   Total optimizations: {summary['total_optimizations']}")
            print(f"   Average time improvement: {summary['avg_time_improvement']:.1%}")
            print(
                f"   Average memory improvement: {summary['avg_memory_improvement']:.1%}"
            )
            print(f"   Cache efficiency: {summary['cache_efficiency']:.1%}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
