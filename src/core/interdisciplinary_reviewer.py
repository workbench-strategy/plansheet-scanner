"""
Interdisciplinary Review (IDR) Module
Provides comprehensive code analysis from multiple disciplinary perspectives.
"""

import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import astroid
import numpy as np
from astroid import nodes

# ML/AI imports for advanced analysis
try:
    import openai
    import torch
    from transformers import AutoModel, AutoTokenizer, pipeline
except ImportError:
    print("Warning: Some ML dependencies not installed for IDR")


@dataclass
class ReviewPerspective:
    """Represents a review perspective with its criteria and findings."""

    name: str
    description: str
    criteria: List[str]
    findings: List[Dict[str, Any]]
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    confidence: float


@dataclass
class InterdisciplinaryReview:
    """Complete interdisciplinary review results."""

    file_path: str
    perspectives: Dict[str, ReviewPerspective]
    overall_risk_score: float
    recommendations: List[str]
    compliance_status: Dict[str, bool]
    business_impact: Dict[str, Any]


class SecurityReviewer:
    """Security-focused code analysis."""

    def __init__(self):
        self.security_patterns = {
            "sql_injection": [
                r'execute\s*\(\s*[\'"][^\'"]*\+.*[\'"]\s*\)',
                r'cursor\.execute\s*\(\s*[\'"][^\'"]*\+.*[\'"]\s*\)',
                r'\.query\s*\(\s*[\'"][^\'"]*\+.*[\'"]\s*\)',
            ],
            "xss": [
                r"innerHTML\s*=",
                r"outerHTML\s*=",
                r"document\.write\s*\(",
                r"eval\s*\(",
            ],
            "path_traversal": [
                r'open\s*\(\s*[\'"][^\'"]*\.\./',
                r'Path\s*\(\s*[\'"][^\'"]*\.\./',
                r"os\.path\.join\s*\(.*\.\./",
            ],
            "hardcoded_secrets": [
                r'password\s*=\s*[\'"][^\'"]+[\'"]',
                r'api_key\s*=\s*[\'"][^\'"]+[\'"]',
                r'secret\s*=\s*[\'"][^\'"]+[\'"]',
                r'token\s*=\s*[\'"][^\'"]+[\'"]',
            ],
            "weak_crypto": [
                r"md5\s*\(",
                r"sha1\s*\(",
                r"base64\.b64encode\s*\(",
                r"hashlib\.md5\s*\(",
            ],
        }

        self.security_frameworks = {
            "owasp_top_10": [
                "Injection",
                "Broken Authentication",
                "Sensitive Data Exposure",
                "XML External Entities",
                "Broken Access Control",
                "Security Misconfiguration",
                "Cross-Site Scripting",
                "Insecure Deserialization",
                "Using Components with Known Vulnerabilities",
                "Insufficient Logging & Monitoring",
            ]
        }

    def review_code(self, code: str, language: str = "python") -> ReviewPerspective:
        """Perform security review of code."""
        findings = []
        risk_level = "low"

        # Check for security patterns
        for vuln_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    findings.append(
                        {
                            "type": "security_vulnerability",
                            "category": vuln_type,
                            "severity": self._get_severity(vuln_type),
                            "line": self._get_line_number(code, match.start()),
                            "code": match.group(),
                            "description": self._get_vulnerability_description(
                                vuln_type
                            ),
                            "recommendation": self._get_security_recommendation(
                                vuln_type
                            ),
                        }
                    )

        # Check for OWASP Top 10 indicators
        owasp_findings = self._check_owasp_compliance(code)
        findings.extend(owasp_findings)

        # Determine overall risk level
        if any(f["severity"] == "critical" for f in findings):
            risk_level = "critical"
        elif any(f["severity"] == "high" for f in findings):
            risk_level = "high"
        elif any(f["severity"] == "medium" for f in findings):
            risk_level = "medium"

        return ReviewPerspective(
            name="Security Review",
            description="Analysis of security vulnerabilities and compliance with security best practices",
            criteria=[
                "OWASP Top 10 compliance",
                "Input validation and sanitization",
                "Authentication and authorization",
                "Data protection and encryption",
                "Secure coding practices",
            ],
            findings=findings,
            risk_level=risk_level,
            confidence=0.85,
        )

    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        severity_map = {
            "sql_injection": "critical",
            "xss": "high",
            "path_traversal": "high",
            "hardcoded_secrets": "high",
            "weak_crypto": "medium",
        }
        return severity_map.get(vuln_type, "medium")

    def _get_vulnerability_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type."""
        descriptions = {
            "sql_injection": "Potential SQL injection vulnerability detected",
            "xss": "Cross-site scripting vulnerability detected",
            "path_traversal": "Path traversal vulnerability detected",
            "hardcoded_secrets": "Hardcoded secrets or credentials detected",
            "weak_crypto": "Weak cryptographic implementation detected",
        }
        return descriptions.get(vuln_type, "Security issue detected")

    def _get_security_recommendation(self, vuln_type: str) -> str:
        """Get security recommendation for vulnerability type."""
        recommendations = {
            "sql_injection": "Use parameterized queries or ORM",
            "xss": "Sanitize user input and use proper output encoding",
            "path_traversal": "Validate and sanitize file paths",
            "hardcoded_secrets": "Use environment variables or secure secret management",
            "weak_crypto": "Use strong cryptographic algorithms and libraries",
        }
        return recommendations.get(vuln_type, "Review and fix security issue")

    def _get_line_number(self, code: str, position: int) -> int:
        """Get line number for a position in code."""
        return code[:position].count("\n") + 1

    def _check_owasp_compliance(self, code: str) -> List[Dict[str, Any]]:
        """Check compliance with OWASP Top 10."""
        findings = []

        # Check for injection patterns
        if re.search(r"execute\s*\(|query\s*\(|eval\s*\(", code, re.IGNORECASE):
            findings.append(
                {
                    "type": "owasp_compliance",
                    "category": "A01:2021 - Injection",
                    "severity": "high",
                    "description": "Potential injection vulnerability",
                    "recommendation": "Use parameterized queries and input validation",
                }
            )

        # Check for authentication issues
        if re.search(r'password\s*=\s*[\'"][^\'"]+[\'"]', code, re.IGNORECASE):
            findings.append(
                {
                    "type": "owasp_compliance",
                    "category": "A02:2021 - Broken Authentication",
                    "severity": "high",
                    "description": "Hardcoded credentials detected",
                    "recommendation": "Use secure authentication mechanisms",
                }
            )

        return findings


class ComplianceReviewer:
    """Compliance and regulatory review."""

    def __init__(self):
        self.compliance_frameworks = {
            "gdpr": {
                "data_processing": [
                    r"personal_data|personal_data|pii|personally_identifiable",
                    r"data\.process|process_data|data_processing",
                ],
                "consent": [
                    r"consent|permission|authorization",
                    r"opt_in|opt_out|unsubscribe",
                ],
                "data_retention": [
                    r"delete.*data|remove.*data|data.*retention",
                    r"archive|backup|storage.*period",
                ],
            },
            "sox": {
                "financial_controls": [
                    r"financial|accounting|audit|balance",
                    r"revenue|expense|profit|loss",
                ],
                "access_controls": [
                    r"access.*control|permission|authorization",
                    r"role.*based|rbac|privilege",
                ],
            },
            "hipaa": {
                "phi_handling": [
                    r"medical|health|patient|diagnosis",
                    r"phi|protected.*health|healthcare",
                ],
                "privacy": [
                    r"privacy|confidential|secure.*transmission",
                    r"encryption|encrypt|decrypt",
                ],
            },
        }

    def review_code(self, code: str, frameworks: List[str] = None) -> ReviewPerspective:
        """Perform compliance review."""
        if frameworks is None:
            frameworks = ["gdpr", "sox", "hipaa"]

        findings = []
        compliance_status = {}

        for framework in frameworks:
            if framework in self.compliance_frameworks:
                framework_findings = self._check_framework_compliance(code, framework)
                findings.extend(framework_findings)
                compliance_status[framework] = (
                    len([f for f in framework_findings if f["severity"] == "critical"])
                    == 0
                )

        risk_level = "low"
        if any(f["severity"] == "critical" for f in findings):
            risk_level = "critical"
        elif any(f["severity"] == "high" for f in findings):
            risk_level = "high"

        return ReviewPerspective(
            name="Compliance Review",
            description="Analysis of regulatory compliance and governance requirements",
            criteria=[
                "GDPR compliance",
                "SOX compliance",
                "HIPAA compliance",
                "Data protection requirements",
                "Audit trail requirements",
            ],
            findings=findings,
            risk_level=risk_level,
            confidence=0.80,
        )

    def _check_framework_compliance(
        self, code: str, framework: str
    ) -> List[Dict[str, Any]]:
        """Check compliance with specific framework."""
        findings = []
        framework_rules = self.compliance_frameworks[framework]

        for category, patterns in framework_rules.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    findings.append(
                        {
                            "type": "compliance_issue",
                            "framework": framework.upper(),
                            "category": category,
                            "severity": "medium",
                            "line": code[: match.start()].count("\n") + 1,
                            "code": match.group(),
                            "description": f"{framework.upper()} compliance consideration: {category}",
                            "recommendation": f"Review {framework.upper()} requirements for {category}",
                        }
                    )

        return findings


class BusinessReviewer:
    """Business and domain-specific review."""

    def __init__(self):
        self.business_metrics = {
            "maintainability": {
                "complexity_threshold": 10,
                "function_length_threshold": 50,
                "comment_ratio_threshold": 0.1,
            },
            "scalability": {
                "hardcoded_limits": [
                    r"limit\s*=\s*\d+",
                    r"max.*=.*\d+",
                    r"capacity\s*=\s*\d+",
                ],
                "performance_indicators": [
                    r"O\s*\(\s*n\s*\*\s*n\s*\)",
                    r"nested.*loop",
                    r"for.*for",
                ],
            },
            "business_logic": {
                "domain_terms": [
                    r"customer|client|user",
                    r"order|purchase|transaction",
                    r"product|item|service",
                    r"payment|billing|invoice",
                ]
            },
        }

    def review_code(self, code: str, domain: str = "general") -> ReviewPerspective:
        """Perform business review."""
        findings = []

        # Analyze maintainability
        maintainability_findings = self._analyze_maintainability(code)
        findings.extend(maintainability_findings)

        # Analyze scalability
        scalability_findings = self._analyze_scalability(code)
        findings.extend(scalability_findings)

        # Analyze business logic
        business_logic_findings = self._analyze_business_logic(code, domain)
        findings.extend(business_logic_findings)

        # Calculate business impact
        business_impact = self._calculate_business_impact(findings)

        risk_level = "low"
        if any(f["severity"] == "high" for f in findings):
            risk_level = "high"
        elif any(f["severity"] == "medium" for f in findings):
            risk_level = "medium"

        return ReviewPerspective(
            name="Business Review",
            description="Analysis of business impact, maintainability, and domain alignment",
            criteria=[
                "Code maintainability",
                "Scalability considerations",
                "Business logic alignment",
                "Domain-specific requirements",
                "Performance implications",
            ],
            findings=findings,
            risk_level=risk_level,
            confidence=0.75,
        )

    def _analyze_maintainability(self, code: str) -> List[Dict[str, Any]]:
        """Analyze code maintainability."""
        findings = []

        try:
            tree = ast.parse(code)

            # Check cyclomatic complexity
            complexity = self._calculate_cyclomatic_complexity(tree)
            if (
                complexity
                > self.business_metrics["maintainability"]["complexity_threshold"]
            ):
                findings.append(
                    {
                        "type": "maintainability_issue",
                        "category": "high_complexity",
                        "severity": "medium",
                        "description": f"High cyclomatic complexity ({complexity})",
                        "recommendation": "Consider breaking down complex functions",
                    }
                )

            # Check function length
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    lines = len(node.body)
                    if (
                        lines
                        > self.business_metrics["maintainability"][
                            "function_length_threshold"
                        ]
                    ):
                        findings.append(
                            {
                                "type": "maintainability_issue",
                                "category": "long_function",
                                "severity": "medium",
                                "description": f"Long function ({lines} lines)",
                                "recommendation": "Consider splitting into smaller functions",
                            }
                        )
        except:
            pass

        return findings

    def _analyze_scalability(self, code: str) -> List[Dict[str, Any]]:
        """Analyze code scalability."""
        findings = []

        # Check for hardcoded limits
        for pattern in self.business_metrics["scalability"]["hardcoded_limits"]:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                findings.append(
                    {
                        "type": "scalability_issue",
                        "category": "hardcoded_limit",
                        "severity": "medium",
                        "description": "Hardcoded limit detected",
                        "recommendation": "Consider making limits configurable",
                    }
                )

        # Check for performance issues
        for pattern in self.business_metrics["scalability"]["performance_indicators"]:
            if re.search(pattern, code, re.IGNORECASE):
                findings.append(
                    {
                        "type": "scalability_issue",
                        "category": "performance_concern",
                        "severity": "medium",
                        "description": "Potential performance issue detected",
                        "recommendation": "Review algorithm complexity and optimization",
                    }
                )

        return findings

    def _analyze_business_logic(self, code: str, domain: str) -> List[Dict[str, Any]]:
        """Analyze business logic alignment."""
        findings = []

        # Check for domain-specific terms
        for term in self.business_metrics["business_logic"]["domain_terms"]:
            if re.search(term, code, re.IGNORECASE):
                findings.append(
                    {
                        "type": "business_logic",
                        "category": "domain_alignment",
                        "severity": "low",
                        "description": f"Domain term detected: {term}",
                        "recommendation": "Ensure business logic aligns with domain requirements",
                    }
                )

        return findings

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_business_impact(
        self, findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate business impact of findings."""
        return {
            "maintainability_score": max(
                0,
                100
                - len([f for f in findings if f["category"] == "maintainability_issue"])
                * 10,
            ),
            "scalability_score": max(
                0,
                100
                - len([f for f in findings if f["category"] == "scalability_issue"])
                * 10,
            ),
            "risk_factors": [
                f["category"] for f in findings if f["severity"] in ["high", "critical"]
            ],
        }


class PerformanceReviewer:
    """Performance-focused code analysis for detecting complexity issues."""

    def __init__(self):
        self.complexity_patterns = {
            "nested_loops": [
                r"for\s+.*:\s*\n.*for\s+.*:",
                r"while\s+.*:\s*\n.*while\s+.*:",
                r"for\s+.*:\s*\n.*while\s+.*:",
                r"while\s+.*:\s*\n.*for\s+.*:",
            ],
            "quadratic_operations": [
                r"\.append\s*\(\s*\[\s*\]\s*\)",  # List of lists
                r"\.extend\s*\(\s*\[\s*\]\s*\)",  # Extending with empty list
                r"list\s*\(\s*range\s*\(\s*\w+\s*,\s*\w+\s*\)\s*\)",  # Range to list
                r"\[\s*\[\s*\]\s*for\s+.*\s+in\s+.*\s*\]",  # List comprehension with nested lists
            ],
            "inefficient_operations": [
                r"\.insert\s*\(\s*0\s*,",  # Insert at beginning
                r"\.pop\s*\(\s*0\s*\)",  # Pop from beginning
                r"\.remove\s*\(",  # Remove by value
                r"\.index\s*\(",  # Find index
                r"in\s+list\s*\(",  # Membership in list
                r"\.sort\s*\(\s*\)",  # In-place sort
                r"sorted\s*\(\s*\)",  # Sort function
            ],
            "memory_intensive": [
                r"deepcopy\s*\(",
                r"copy\.deepcopy\s*\(",
                r"pickle\.dumps\s*\(",
                r"json\.dumps\s*\(",
                r"xml\.etree\.ElementTree\.tostring\s*\(",
            ],
        }

        self.optimization_patterns = {
            "efficient_alternatives": {
                "list_operations": {
                    "inefficient": r"\.insert\s*\(\s*0\s*,",
                    "efficient": "collections.deque for O(1) operations",
                    "description": "Use deque for frequent insertions/deletions at ends",
                },
                "search_operations": {
                    "inefficient": r"in\s+list\s*\(",
                    "efficient": "set or dict for O(1) lookups",
                    "description": "Use set/dict for membership testing",
                },
                "sorting": {
                    "inefficient": r"\.sort\s*\(\s*\)",
                    "efficient": "heapq for partial sorting",
                    "description": "Use heapq for top-k operations",
                },
            }
        }

    def review_code(self, code: str, language: str = "python") -> ReviewPerspective:
        """Perform performance review."""
        findings = []

        # Analyze code structure for complexity
        complexity_findings = self._analyze_complexity(code)
        findings.extend(complexity_findings)

        # Check for inefficient operations
        inefficiency_findings = self._check_inefficient_operations(code)
        findings.extend(inefficiency_findings)

        # Analyze time complexity
        time_complexity_findings = self._analyze_time_complexity(code)
        findings.extend(time_complexity_findings)

        # Check for memory issues
        memory_findings = self._check_memory_usage(code)
        findings.extend(memory_findings)

        # Generate optimization recommendations
        optimization_findings = self._generate_optimization_recommendations(findings)
        findings.extend(optimization_findings)

        # Determine risk level based on findings
        risk_level = self._determine_risk_level(findings)

        return ReviewPerspective(
            name="Performance Review",
            description="Analysis of algorithmic complexity, performance bottlenecks, and optimization opportunities",
            criteria=[
                "Time complexity analysis",
                "Space complexity analysis",
                "Inefficient operation detection",
                "Memory usage optimization",
                "Algorithm optimization recommendations",
            ],
            findings=findings,
            risk_level=risk_level,
            confidence=0.85,
        )

    def _analyze_complexity(self, code: str) -> List[Dict[str, Any]]:
        """Analyze code for complexity issues."""
        findings = []

        # Check for nested loops
        for pattern in self.complexity_patterns["nested_loops"]:
            matches = re.finditer(pattern, code, re.MULTILINE | re.DOTALL)
            for match in matches:
                findings.append(
                    {
                        "type": "complexity_issue",
                        "category": "nested_loops",
                        "severity": "high",
                        "line": code[: match.start()].count("\n") + 1,
                        "code": match.group()[:100] + "..."
                        if len(match.group()) > 100
                        else match.group(),
                        "description": "Nested loop detected - potential O(n^2) or worse complexity",
                        "recommendation": "Consider using more efficient algorithms or data structures",
                        "complexity": "O(n^2) or worse",
                    }
                )

        # Check for quadratic operations
        for pattern in self.complexity_patterns["quadratic_operations"]:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                findings.append(
                    {
                        "type": "complexity_issue",
                        "category": "quadratic_operation",
                        "severity": "medium",
                        "line": code[: match.start()].count("\n") + 1,
                        "code": match.group(),
                        "description": "Potentially quadratic operation detected",
                        "recommendation": "Consider vectorized operations or more efficient data structures",
                        "complexity": "O(n^2)",
                    }
                )

        return findings

    def _check_inefficient_operations(self, code: str) -> List[Dict[str, Any]]:
        """Check for inefficient operations."""
        findings = []

        for pattern in self.complexity_patterns["inefficient_operations"]:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                operation = match.group()
                severity = (
                    "high"
                    if "insert(0" in operation or "pop(0" in operation
                    else "medium"
                )

                findings.append(
                    {
                        "type": "inefficient_operation",
                        "category": "list_operation",
                        "severity": severity,
                        "line": code[: match.start()].count("\n") + 1,
                        "code": operation,
                        "description": f"Inefficient operation detected: {operation}",
                        "recommendation": self._get_optimization_recommendation(
                            operation
                        ),
                        "complexity": "O(n) for list operations",
                    }
                )

        return findings

    def _analyze_time_complexity(self, code: str) -> List[Dict[str, Any]]:
        """Analyze time complexity of code patterns."""
        findings = []

        # Count nested structures
        lines = code.split("\n")
        for i, line in enumerate(lines):
            indent_level = len(line) - len(line.lstrip())

            # Check for deeply nested structures
            if indent_level > 8:  # More than 4 levels of nesting
                findings.append(
                    {
                        "type": "complexity_issue",
                        "category": "deep_nesting",
                        "severity": "medium",
                        "line": i + 1,
                        "code": line.strip(),
                        "description": f"Deep nesting detected ({indent_level//2} levels)",
                        "recommendation": "Consider extracting functions or using early returns to reduce nesting",
                        "complexity": "O(n^k) where k is nesting depth",
                    }
                )

        # Check for recursive patterns without memoization
        # Look for function definitions followed by calls to the same function
        function_pattern = r"def\s+(\w+)\s*\([^)]*\):"
        function_matches = re.finditer(function_pattern, code, re.MULTILINE)

        for match in function_matches:
            function_name = match.group(1)
            # Look for calls to the same function after its definition
            call_pattern = rf"{function_name}\s*\([^)]*\)"
            if re.search(call_pattern, code[match.end() :], re.MULTILINE):
                findings.append(
                    {
                        "type": "complexity_issue",
                        "category": "recursion",
                        "severity": "medium",
                        "description": "Recursive function detected without apparent memoization",
                        "recommendation": "Consider adding memoization or converting to iterative approach",
                        "complexity": "O(2^n) without memoization",
                    }
                )

        return findings

    def _check_memory_usage(self, code: str) -> List[Dict[str, Any]]:
        """Check for memory-intensive operations."""
        findings = []

        for pattern in self.complexity_patterns["memory_intensive"]:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                findings.append(
                    {
                        "type": "memory_issue",
                        "category": "memory_intensive",
                        "severity": "medium",
                        "line": code[: match.start()].count("\n") + 1,
                        "code": match.group(),
                        "description": "Memory-intensive operation detected",
                        "recommendation": "Consider streaming or chunked processing for large data",
                        "complexity": "High memory usage",
                    }
                )

        return findings

    def _generate_optimization_recommendations(
        self, findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations."""
        recommendations = []

        # Count findings by category
        category_counts = {}
        for finding in findings:
            category = finding.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1

        # Generate recommendations based on findings
        if category_counts.get("nested_loops", 0) > 0:
            recommendations.append(
                {
                    "type": "optimization_recommendation",
                    "category": "algorithm_optimization",
                    "severity": "high",
                    "description": f"Found {category_counts['nested_loops']} nested loop(s)",
                    "recommendation": "Consider using vectorized operations, hash maps, or divide-and-conquer algorithms",
                    "impact": "Can reduce complexity from O(n^2) to O(n log n) or O(n)",
                }
            )

        if category_counts.get("list_operation", 0) > 0:
            recommendations.append(
                {
                    "type": "optimization_recommendation",
                    "category": "data_structure_optimization",
                    "severity": "medium",
                    "description": f"Found {category_counts['list_operation']} inefficient list operation(s)",
                    "recommendation": "Use collections.deque for frequent insertions/deletions, sets for membership testing",
                    "impact": "Can reduce complexity from O(n) to O(1) for many operations",
                }
            )

        return recommendations

    def _get_optimization_recommendation(self, operation: str) -> str:
        """Get specific optimization recommendation for an operation."""
        if "insert(0" in operation:
            return "Use collections.deque for O(1) insertions at beginning"
        elif "pop(0" in operation:
            return "Use collections.deque for O(1) removals from beginning"
        elif "remove(" in operation:
            return "Use set or dict for O(1) removals by value"
        elif "index(" in operation:
            return "Use dict for O(1) lookups by key"
        elif "in list(" in operation:
            return "Use set for O(1) membership testing"
        else:
            return "Consider more efficient data structure or algorithm"

    def _determine_risk_level(self, findings: List[Dict[str, Any]]) -> str:
        """Determine overall risk level based on findings."""
        if any(f["severity"] == "critical" for f in findings):
            return "critical"
        elif any(f["severity"] == "high" for f in findings):
            return "high"
        elif any(f["severity"] == "medium" for f in findings):
            return "medium"
        else:
            return "low"


class TechnicalReviewer:
    """Technical architecture and best practices review."""

    def __init__(self):
        self.technical_patterns = {
            "code_smells": [
                r"if\s+True:",
                r"if\s+False:",
                r"while\s+True:",
                r"except\s*:",
                r"import\s*\*",
                r"global\s+\w+",
            ],
            "anti_patterns": [
                r"goto\s+",
                r"goto\s+",
                r"spaghetti\s+code",
                r"god\s+object",
                r"big\s+bang\s+integration",
            ],
            "design_patterns": [
                r"class\s+\w+Factory",
                r"class\s+\w+Singleton",
                r"class\s+\w+Observer",
                r"class\s+\w+Strategy",
            ],
        }

    def review_code(self, code: str, language: str = "python") -> ReviewPerspective:
        """Perform technical review."""
        findings = []

        # Check for code smells
        code_smell_findings = self._check_code_smells(code)
        findings.extend(code_smell_findings)

        # Check for anti-patterns
        anti_pattern_findings = self._check_anti_patterns(code)
        findings.extend(anti_pattern_findings)

        # Check for design patterns
        design_pattern_findings = self._check_design_patterns(code)
        findings.extend(design_pattern_findings)

        # Analyze architecture
        architecture_findings = self._analyze_architecture(code, language)
        findings.extend(architecture_findings)

        risk_level = "low"
        if any(f["severity"] == "high" for f in findings):
            risk_level = "high"
        elif any(f["severity"] == "medium" for f in findings):
            risk_level = "medium"

        return ReviewPerspective(
            name="Technical Review",
            description="Analysis of technical architecture, design patterns, and code quality",
            criteria=[
                "Code quality and maintainability",
                "Design patterns and architecture",
                "Best practices adherence",
                "Technical debt identification",
                "Performance considerations",
            ],
            findings=findings,
            risk_level=risk_level,
            confidence=0.90,
        )

    def _check_code_smells(self, code: str) -> List[Dict[str, Any]]:
        """Check for code smells."""
        findings = []

        for pattern in self.technical_patterns["code_smells"]:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                findings.append(
                    {
                        "type": "code_smell",
                        "category": "poor_practice",
                        "severity": "medium",
                        "line": code[: match.start()].count("\n") + 1,
                        "code": match.group(),
                        "description": "Code smell detected",
                        "recommendation": "Consider refactoring to improve code quality",
                    }
                )

        return findings

    def _check_anti_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Check for anti-patterns."""
        findings = []

        for pattern in self.technical_patterns["anti_patterns"]:
            if re.search(pattern, code, re.IGNORECASE):
                findings.append(
                    {
                        "type": "anti_pattern",
                        "category": "design_issue",
                        "severity": "high",
                        "description": "Anti-pattern detected",
                        "recommendation": "Consider alternative design approaches",
                    }
                )

        return findings

    def _check_design_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Check for design patterns."""
        findings = []

        for pattern in self.technical_patterns["design_patterns"]:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                findings.append(
                    {
                        "type": "design_pattern",
                        "category": "good_practice",
                        "severity": "low",
                        "code": match.group(),
                        "description": "Design pattern detected",
                        "recommendation": "Good use of design pattern",
                    }
                )

        return findings

    def _analyze_architecture(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Analyze code architecture."""
        findings = []

        try:
            tree = ast.parse(code)

            # Check for large classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = len(
                        [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    )
                    if methods > 10:
                        findings.append(
                            {
                                "type": "architecture_issue",
                                "category": "large_class",
                                "severity": "medium",
                                "description": f"Large class with {methods} methods",
                                "recommendation": "Consider breaking down large class",
                            }
                        )
        except:
            pass

        return findings


class InterdisciplinaryReviewer:
    """Main interdisciplinary review orchestrator."""

    def __init__(self):
        self.security_reviewer = SecurityReviewer()
        self.compliance_reviewer = ComplianceReviewer()
        self.business_reviewer = BusinessReviewer()
        self.technical_reviewer = TechnicalReviewer()
        self.performance_reviewer = PerformanceReviewer()

    def perform_review(
        self, file_path: str, frameworks: List[str] = None, domain: str = "general"
    ) -> InterdisciplinaryReview:
        """Perform comprehensive interdisciplinary review."""

        # Read code
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Perform reviews from different perspectives
        perspectives = {}

        # Security review
        security_perspective = self.security_reviewer.review_code(code)
        perspectives["security"] = security_perspective

        # Compliance review
        compliance_perspective = self.compliance_reviewer.review_code(code, frameworks)
        perspectives["compliance"] = compliance_perspective

        # Business review
        business_perspective = self.business_reviewer.review_code(code, domain)
        perspectives["business"] = business_perspective

        # Technical review
        technical_perspective = self.technical_reviewer.review_code(code)
        perspectives["technical"] = technical_perspective

        # Performance review
        performance_perspective = self.performance_reviewer.review_code(code)
        perspectives["performance"] = performance_perspective

        # Calculate overall risk score
        overall_risk_score = self._calculate_overall_risk(perspectives)

        # Generate recommendations
        recommendations = self._generate_recommendations(perspectives)

        # Determine compliance status
        compliance_status = {
            "security": security_perspective.risk_level in ["low", "medium"],
            "compliance": compliance_perspective.risk_level in ["low", "medium"],
            "business": business_perspective.risk_level in ["low", "medium"],
            "technical": technical_perspective.risk_level in ["low", "medium"],
            "performance": performance_perspective.risk_level in ["low", "medium"],
        }

        # Calculate business impact
        business_impact = (
            business_perspective.findings[0] if business_perspective.findings else {}
        )

        return InterdisciplinaryReview(
            file_path=file_path,
            perspectives=perspectives,
            overall_risk_score=overall_risk_score,
            recommendations=recommendations,
            compliance_status=compliance_status,
            business_impact=business_impact,
        )

    def _calculate_overall_risk(
        self, perspectives: Dict[str, ReviewPerspective]
    ) -> float:
        """Calculate overall risk score."""
        risk_scores = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}

        total_score = 0
        for perspective in perspectives.values():
            total_score += risk_scores.get(perspective.risk_level, 0.5)

        return total_score / len(perspectives)

    def _generate_recommendations(
        self, perspectives: Dict[str, ReviewPerspective]
    ) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []

        # Collect all findings
        all_findings = []
        for perspective_name, perspective in perspectives.items():
            for finding in perspective.findings:
                all_findings.append(
                    {
                        "perspective": perspective_name,
                        "finding": finding,
                        "priority": self._get_priority(finding["severity"]),
                    }
                )

        # Sort by priority
        all_findings.sort(key=lambda x: x["priority"], reverse=True)

        # Generate recommendations
        for finding_data in all_findings[:10]:  # Top 10 recommendations
            finding = finding_data["finding"]
            perspective = finding_data["perspective"]

            if "recommendation" in finding:
                recommendations.append(
                    f"[{perspective.upper()}] {finding['recommendation']}"
                )

        return recommendations

    def _get_priority(self, severity: str) -> int:
        """Get priority score for severity."""
        priority_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return priority_map.get(severity, 1)


def main():
    """Example usage of the Interdisciplinary Reviewer."""
    reviewer = InterdisciplinaryReviewer()

    # Example review
    result = reviewer.perform_review("example.py")

    print(f"Interdisciplinary Review Results for {result.file_path}")
    print(f"Overall Risk Score: {result.overall_risk_score:.2f}")
    print(f"Compliance Status: {result.compliance_status}")
    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"- {rec}")


if __name__ == "__main__":
    main()
