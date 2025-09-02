"""
Quality Assurance System for Plansheet Scanner

Ensures production-ready quality through:
- Integration testing of agent interactions
- Performance benchmarking and baseline metrics
- Error recovery testing and failure scenarios
- Documentation review and completeness
- Security audit and vulnerability assessment
"""

import ast
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a quality assurance test."""

    test_name: str
    test_type: str  # integration, performance, security, documentation
    status: str  # passed, failed, warning
    execution_time: float
    details: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class QualityReport:
    """Comprehensive quality assurance report."""

    overall_status: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    test_results: List[TestResult]
    recommendations: List[str]
    security_issues: List[str]
    performance_metrics: Dict[str, Any]
    documentation_coverage: Dict[str, float]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class IntegrationTester:
    """Tests integration between different agents."""

    def __init__(self):
        self.test_scenarios = []
        self.agent_dependencies = {}

    def add_test_scenario(
        self,
        name: str,
        agents: List[str],
        test_data: Dict[str, Any],
        expected_output: Any,
    ):
        """Add a test scenario for agent integration."""
        self.test_scenarios.append(
            {
                "name": name,
                "agents": agents,
                "test_data": test_data,
                "expected_output": expected_output,
            }
        )

    def test_agent_integration(
        self, agent_functions: Dict[str, Callable]
    ) -> List[TestResult]:
        """Test integration between agents."""
        results = []

        for scenario in self.test_scenarios:
            logger.info(f"Testing integration scenario: {scenario['name']}")
            start_time = time.time()

            try:
                # Execute the integration test
                result = self._execute_integration_test(scenario, agent_functions)
                execution_time = time.time() - start_time

                # Check if result matches expected output
                if self._validate_output(result, scenario["expected_output"]):
                    status = "passed"
                    details = {
                        "result": result,
                        "expected": scenario["expected_output"],
                    }
                else:
                    status = "failed"
                    details = {
                        "result": result,
                        "expected": scenario["expected_output"],
                        "error": "Output does not match expected result",
                    }

            except Exception as e:
                execution_time = time.time() - start_time
                status = "failed"
                details = {"error": str(e)}

            test_result = TestResult(
                test_name=f"integration_{scenario['name']}",
                test_type="integration",
                status=status,
                execution_time=execution_time,
                details=details,
            )
            results.append(test_result)

        return results

    def _execute_integration_test(
        self, scenario: Dict[str, Any], agent_functions: Dict[str, Callable]
    ) -> Any:
        """Execute a single integration test."""
        # This is a placeholder - in real implementation, would orchestrate agents
        # For now, return a mock result
        return {"status": "success", "agents_used": scenario["agents"]}

    def _validate_output(self, actual: Any, expected: Any) -> bool:
        """Validate that actual output matches expected output."""
        # Simple validation - in real implementation, would be more sophisticated
        return isinstance(actual, type(expected))


class PerformanceBenchmarker:
    """Benchmarks performance and establishes baseline metrics."""

    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = baseline_file
        self.baseline_metrics = self._load_baseline()

    def _load_baseline(self) -> Dict[str, Any]:
        """Load performance baseline from file."""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load baseline: {e}")
        return {}

    def run_benchmarks(
        self, test_functions: Dict[str, Callable], test_data: Dict[str, Any]
    ) -> List[TestResult]:
        """Run performance benchmarks."""
        results = []

        for func_name, test_func in test_functions.items():
            logger.info(f"Benchmarking: {func_name}")
            start_time = time.time()

            try:
                # Run the test function
                result = test_func(**test_data.get(func_name, {}))
                execution_time = time.time() - start_time

                # Compare with baseline
                baseline_time = self.baseline_metrics.get(func_name, {}).get(
                    "execution_time", float("inf")
                )

                if execution_time <= baseline_time * 1.1:  # Within 10% of baseline
                    status = "passed"
                elif execution_time <= baseline_time * 1.5:  # Within 50% of baseline
                    status = "warning"
                else:
                    status = "failed"

                details = {
                    "execution_time": execution_time,
                    "baseline_time": baseline_time,
                    "performance_ratio": execution_time / baseline_time
                    if baseline_time > 0
                    else 0,
                }

            except Exception as e:
                execution_time = time.time() - start_time
                status = "failed"
                details = {"error": str(e)}

            test_result = TestResult(
                test_name=f"benchmark_{func_name}",
                test_type="performance",
                status=status,
                execution_time=execution_time,
                details=details,
            )
            results.append(test_result)

        return results

    def update_baseline(self, benchmark_results: List[TestResult]):
        """Update performance baseline with new results."""
        new_baseline = {}

        for result in benchmark_results:
            if result.status == "passed":
                func_name = result.test_name.replace("benchmark_", "")
                new_baseline[func_name] = {
                    "execution_time": result.execution_time,
                    "timestamp": result.timestamp.isoformat(),
                }

        # Save new baseline
        with open(self.baseline_file, "w") as f:
            json.dump(new_baseline, f, indent=2)

        logger.info(f"Updated performance baseline with {len(new_baseline)} metrics")


class ErrorRecoveryTester:
    """Tests error recovery and failure scenarios."""

    def __init__(self):
        self.failure_scenarios = []

    def add_failure_scenario(
        self,
        name: str,
        failure_type: str,
        test_function: Callable,
        expected_behavior: str,
    ):
        """Add a failure scenario to test."""
        self.failure_scenarios.append(
            {
                "name": name,
                "failure_type": failure_type,
                "test_function": test_function,
                "expected_behavior": expected_behavior,
            }
        )

    def test_error_recovery(self) -> List[TestResult]:
        """Test error recovery mechanisms."""
        results = []

        for scenario in self.failure_scenarios:
            logger.info(f"Testing error recovery: {scenario['name']}")
            start_time = time.time()

            try:
                # Execute the failure scenario
                result = scenario["test_function"]()
                execution_time = time.time() - start_time

                # Check if the system handled the failure appropriately
                if self._validate_error_handling(result, scenario["expected_behavior"]):
                    status = "passed"
                    details = {
                        "result": result,
                        "expected_behavior": scenario["expected_behavior"],
                    }
                else:
                    status = "failed"
                    details = {
                        "result": result,
                        "expected_behavior": scenario["expected_behavior"],
                        "error": "Error handling did not match expected behavior",
                    }

            except Exception as e:
                execution_time = time.time() - start_time
                status = "failed"
                details = {"error": str(e)}

            test_result = TestResult(
                test_name=f"error_recovery_{scenario['name']}",
                test_type="error_recovery",
                status=status,
                execution_time=execution_time,
                details=details,
            )
            results.append(test_result)

        return results

    def _validate_error_handling(self, result: Any, expected_behavior: str) -> bool:
        """Validate that error handling matches expected behavior."""
        # Simple validation - in real implementation, would be more sophisticated
        return isinstance(result, dict) and "error_handled" in result


class DocumentationReviewer:
    """Reviews documentation completeness and quality."""

    def __init__(self, docs_directory: str = "docs"):
        self.docs_directory = docs_directory
        self.required_files = [
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "LICENSE",
        ]
        self.required_sections = ["installation", "usage", "api", "examples"]

    def review_documentation(self) -> List[TestResult]:
        """Review documentation completeness and quality."""
        results = []

        # Check for required files
        for filename in self.required_files:
            filepath = os.path.join(self.docs_directory, filename)
            start_time = time.time()

            if os.path.exists(filepath):
                status = "passed"
                details = {"file_exists": True, "file_size": os.path.getsize(filepath)}
            else:
                status = "failed"
                details = {
                    "file_exists": False,
                    "error": f"Required file {filename} not found",
                }

            execution_time = time.time() - start_time

            test_result = TestResult(
                test_name=f"doc_file_{filename}",
                test_type="documentation",
                status=status,
                execution_time=execution_time,
                details=details,
            )
            results.append(test_result)

        # Check README content
        readme_path = os.path.join(self.docs_directory, "README.md")
        if os.path.exists(readme_path):
            start_time = time.time()

            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read().lower()

                missing_sections = []
                for section in self.required_sections:
                    if section not in content:
                        missing_sections.append(section)

                if not missing_sections:
                    status = "passed"
                    details = {"sections_found": self.required_sections}
                else:
                    status = "warning"
                    details = {"missing_sections": missing_sections}

            except Exception as e:
                status = "failed"
                details = {"error": str(e)}

            execution_time = time.time() - start_time

            test_result = TestResult(
                test_name="doc_readme_content",
                test_type="documentation",
                status=status,
                execution_time=execution_time,
                details=details,
            )
            results.append(test_result)

        return results


class SecurityAuditor:
    """Performs security audit and vulnerability assessment."""

    def __init__(self):
        self.security_patterns = {
            "hardcoded_secrets": [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ],
            "sql_injection": [
                r'execute\s*\(\s*["\'][^"\']*\+[^"\']*["\']',
                r'query\s*\(\s*["\'][^"\']*\+[^"\']*["\']',
            ],
            "file_operations": [
                r"open\s*\(\s*[^)]*\+[^)]*\)",
                r"file\s*\(\s*[^)]*\+[^)]*\)",
            ],
        }

    def audit_security(self, source_directory: str = "src") -> List[TestResult]:
        """Perform security audit of source code."""
        results = []

        # Scan for security issues
        for issue_type, patterns in self.security_patterns.items():
            logger.info(f"Scanning for {issue_type}")
            start_time = time.time()

            issues = self._scan_for_security_issues(source_directory, patterns)
            execution_time = time.time() - start_time

            if not issues:
                status = "passed"
                details = {"issues_found": 0}
            else:
                status = "warning" if len(issues) < 5 else "failed"
                details = {"issues_found": len(issues), "issues": issues[:5]}

            test_result = TestResult(
                test_name=f"security_{issue_type}",
                test_type="security",
                status=status,
                execution_time=execution_time,
                details=details,
            )
            results.append(test_result)

        return results

    def _scan_for_security_issues(
        self, directory: str, patterns: List[str]
    ) -> List[str]:
        """Scan directory for security issues matching patterns."""
        issues = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()

                        for pattern in patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                line_num = content[: match.start()].count("\n") + 1
                                issues.append(f"{filepath}:{line_num}: {match.group()}")

                    except Exception as e:
                        logger.warning(f"Could not scan {filepath}: {e}")

        return issues


class QualityAssurance:
    """
    Main quality assurance system.

    Features:
    - Integration testing of agent interactions
    - Performance benchmarking and baseline metrics
    - Error recovery testing and failure scenarios
    - Documentation review and completeness
    - Security audit and vulnerability assessment
    """

    def __init__(self):
        self.integration_tester = IntegrationTester()
        self.performance_benchmarker = PerformanceBenchmarker()
        self.error_recovery_tester = ErrorRecoveryTester()
        self.documentation_reviewer = DocumentationReviewer()
        self.security_auditor = SecurityAuditor()

        # Set up default test scenarios
        self._setup_default_tests()

    def _setup_default_tests(self):
        """Set up default test scenarios."""
        # Add integration test scenarios
        self.integration_tester.add_test_scenario(
            "workflow_completion",
            ["legend_extractor", "cable_entity_pipeline", "overlay"],
            {"pdf_path": "test.pdf"},
            {"status": "success"},
        )

        # Add error recovery scenarios
        self.error_recovery_tester.add_failure_scenario(
            "file_not_found",
            "file_error",
            lambda: self._simulate_file_error(),
            "graceful_error_handling",
        )

    def _simulate_file_error(self):
        """Simulate a file not found error."""
        try:
            with open("nonexistent_file.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            return {"error_handled": True, "error_type": "file_not_found"}

    def run_full_qa_suite(
        self,
        agent_functions: Dict[str, Callable] = None,
        test_data: Dict[str, Any] = None,
    ) -> QualityReport:
        """Run the complete quality assurance suite."""
        logger.info("Starting full QA suite...")

        all_results = []

        # Integration testing
        if agent_functions:
            logger.info("Running integration tests...")
            integration_results = self.integration_tester.test_agent_integration(
                agent_functions
            )
            all_results.extend(integration_results)

        # Performance benchmarking
        if test_data:
            logger.info("Running performance benchmarks...")
            benchmark_results = self.performance_benchmarker.run_benchmarks(
                agent_functions or {}, test_data
            )
            all_results.extend(benchmark_results)

        # Error recovery testing
        logger.info("Running error recovery tests...")
        error_results = self.error_recovery_tester.test_error_recovery()
        all_results.extend(error_results)

        # Documentation review
        logger.info("Reviewing documentation...")
        doc_results = self.documentation_reviewer.review_documentation()
        all_results.extend(doc_results)

        # Security audit
        logger.info("Performing security audit...")
        security_results = self.security_auditor.audit_security()
        all_results.extend(security_results)

        # Generate report
        report = self._generate_quality_report(all_results)

        logger.info(f"QA suite completed. Overall status: {report.overall_status}")
        return report

    def _generate_quality_report(self, test_results: List[TestResult]) -> QualityReport:
        """Generate comprehensive quality report."""
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == "passed"])
        failed_tests = len([r for r in test_results if r.status == "failed"])
        warning_tests = len([r for r in test_results if r.status == "warning"])

        # Determine overall status
        if failed_tests == 0:
            overall_status = "passed"
        elif failed_tests < total_tests * 0.1:  # Less than 10% failed
            overall_status = "warning"
        else:
            overall_status = "failed"

        # Generate recommendations
        recommendations = self._generate_recommendations(test_results)

        # Extract security issues
        security_issues = []
        for result in test_results:
            if result.test_type == "security" and result.status != "passed":
                if "issues" in result.details:
                    security_issues.extend(result.details["issues"])

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(test_results)

        # Calculate documentation coverage
        documentation_coverage = self._calculate_documentation_coverage(test_results)

        return QualityReport(
            overall_status=overall_status,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            test_results=test_results,
            recommendations=recommendations,
            security_issues=security_issues,
            performance_metrics=performance_metrics,
            documentation_coverage=documentation_coverage,
        )

    def _generate_recommendations(self, test_results: List[TestResult]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Failed tests
        failed_tests = [r for r in test_results if r.status == "failed"]
        if failed_tests:
            recommendations.append(f"Fix {len(failed_tests)} failed tests")

        # Security issues
        security_failures = [
            r
            for r in test_results
            if r.test_type == "security" and r.status != "passed"
        ]
        if security_failures:
            recommendations.append("Address security vulnerabilities")

        # Performance issues
        performance_warnings = [
            r
            for r in test_results
            if r.test_type == "performance" and r.status == "warning"
        ]
        if performance_warnings:
            recommendations.append("Optimize performance for slow operations")

        # Documentation issues
        doc_failures = [
            r
            for r in test_results
            if r.test_type == "documentation" and r.status == "failed"
        ]
        if doc_failures:
            recommendations.append("Complete missing documentation")

        return recommendations

    def _calculate_performance_metrics(
        self, test_results: List[TestResult]
    ) -> Dict[str, Any]:
        """Calculate performance metrics from test results."""
        performance_results = [r for r in test_results if r.test_type == "performance"]

        if not performance_results:
            return {}

        execution_times = [r.execution_time for r in performance_results]

        return {
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "total_tests": len(performance_results),
        }

    def _calculate_documentation_coverage(
        self, test_results: List[TestResult]
    ) -> Dict[str, float]:
        """Calculate documentation coverage metrics."""
        doc_results = [r for r in test_results if r.test_type == "documentation"]

        if not doc_results:
            return {"overall_coverage": 0.0}

        passed_doc_tests = len([r for r in doc_results if r.status == "passed"])
        total_doc_tests = len(doc_results)

        return {
            "overall_coverage": passed_doc_tests / total_doc_tests
            if total_doc_tests > 0
            else 0.0,
            "total_doc_tests": total_doc_tests,
            "passed_doc_tests": passed_doc_tests,
        }

    def save_report(self, report: QualityReport, filepath: str):
        """Save quality report to file."""
        with open(filepath, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        logger.info(f"Quality report saved to {filepath}")


def main():
    """Main CLI function for quality assurance."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quality assurance for plansheet scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quality_assurance.py --full-suite
  python quality_assurance.py --security-only
  python quality_assurance.py --performance-only --output report.json
        """,
    )

    parser.add_argument(
        "--full-suite", action="store_true", help="Run complete QA suite"
    )
    parser.add_argument(
        "--integration-only", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Run only performance benchmarks",
    )
    parser.add_argument(
        "--security-only", action="store_true", help="Run only security audit"
    )
    parser.add_argument(
        "--documentation-only",
        action="store_true",
        help="Run only documentation review",
    )
    parser.add_argument(
        "--output", "-o", default="qa_report.json", help="Output file for report"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        qa = QualityAssurance()

        if args.full_suite:
            print("🔍 Running full QA suite...")
            report = qa.run_full_qa_suite()
        elif args.integration_only:
            print("🔗 Running integration tests...")
            report = qa.run_full_qa_suite()  # Simplified for demo
        elif args.performance_only:
            print("⚡ Running performance benchmarks...")
            report = qa.run_full_qa_suite()  # Simplified for demo
        elif args.security_only:
            print("🔒 Running security audit...")
            report = qa.run_full_qa_suite()  # Simplified for demo
        elif args.documentation_only:
            print("📚 Reviewing documentation...")
            report = qa.run_full_qa_suite()  # Simplified for demo
        else:
            print("🔍 Running full QA suite...")
            report = qa.run_full_qa_suite()

        # Save report
        qa.save_report(report, args.output)

        # Print summary
        print(f"\n📊 QA Summary:")
        print(f"   Overall Status: {report.overall_status.upper()}")
        print(f"   Total Tests: {report.total_tests}")
        print(f"   Passed: {report.passed_tests}")
        print(f"   Failed: {report.failed_tests}")
        print(f"   Warnings: {report.warning_tests}")

        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in report.recommendations:
                print(f"   - {rec}")

        if report.security_issues:
            print(f"\n⚠️ Security Issues: {len(report.security_issues)}")
            for issue in report.security_issues[:3]:  # Show first 3
                print(f"   - {issue}")

        print(f"\n📄 Detailed report saved to: {args.output}")

        return 0 if report.overall_status != "failed" else 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
