"""
AI Code Companion - Core Module
Provides intelligent code highlighting, citation, and document retrieval capabilities.
"""

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ML/AI imports
try:
    import openai
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoModel,
        AutoTokenizer,
        CodeT5ForConditionalGeneration,
        CodeT5Tokenizer,
        pipeline,
    )
except ImportError:
    print(
        "Warning: Some ML dependencies not installed. Install with: pip install torch transformers sentence-transformers openai"
    )

# Code analysis imports
try:
    import ast

    import astroid
    import jedi
    from astroid import nodes
except ImportError:
    print(
        "Warning: Code analysis dependencies not installed. Install with: pip install astroid jedi"
    )


@dataclass
class CodeSnippet:
    """Represents a code snippet with metadata."""

    content: str
    start_line: int
    end_line: int
    file_path: str
    language: str
    context: str
    confidence: float = 1.0


@dataclass
class CodeCitation:
    """Represents a code citation with source information."""

    snippet: CodeSnippet
    source_file: str
    source_line: int
    similarity_score: float
    citation_type: str  # 'exact', 'similar', 'reference'


@dataclass
class HighlightedCode:
    """Represents highlighted code with annotations."""

    code: str
    highlights: List[Dict[str, Any]]
    suggestions: List[str]
    warnings: List[str]
    complexity_score: float


class CodeEmbeddingModel:
    """Neural network model for code embedding and similarity."""

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embedding_dim = 768

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            logging.warning(f"Could not load {model_name}: {e}")

    def encode_code(self, code: str) -> np.ndarray:
        """Encode code into embedding vector."""
        if not self.tokenizer or not self.model:
            return np.zeros(self.embedding_dim)

        try:
            inputs = self.tokenizer(
                code, return_tensors="pt", truncation=True, max_length=512, padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                return embedding.flatten()
        except Exception as e:
            logging.error(f"Error encoding code: {e}")
            return np.zeros(self.embedding_dim)


class CodeAnalyzer:
    """Advanced code analysis using AST and static analysis."""

    def __init__(self):
        self.supported_languages = ["python", "javascript", "java", "cpp", "csharp"]

    def analyze_complexity(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze code complexity and provide metrics."""
        if language != "python":
            return self._analyze_generic(code, language)

        try:
            tree = ast.parse(code)
            return {
                "cyclomatic_complexity": self._calculate_cyclomatic_complexity(tree),
                "lines_of_code": len(code.split("\n")),
                "function_count": len(
                    [
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                    ]
                ),
                "class_count": len(
                    [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                ),
                "import_count": len(
                    [
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, ast.Import)
                        or isinstance(node, ast.ImportFrom)
                    ]
                ),
                "nested_depth": self._calculate_nested_depth(tree),
            }
        except Exception as e:
            logging.error(f"Error analyzing Python code: {e}")
            return {}

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of Python code."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_nested_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        current_depth = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                current_depth -= 1

        return max_depth

    def _analyze_generic(self, code: str, language: str) -> Dict[str, Any]:
        """Generic analysis for non-Python languages."""
        lines = code.split("\n")
        return {
            "lines_of_code": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "comment_lines": len(
                [
                    line
                    for line in lines
                    if line.strip().startswith(("//", "/*", "*", "#"))
                ]
            ),
            "language": language,
        }


class IntelligentDocumentRetriever:
    """Neural network-based document retrieval system."""

    def __init__(self, embedding_model: CodeEmbeddingModel):
        self.embedding_model = embedding_model
        self.document_index = {}
        self.code_snippets = []
        self.similarity_threshold = 0.7

    def index_document(self, file_path: str, code: str, language: str = "python"):
        """Index a document for retrieval."""
        embedding = self.embedding_model.encode_code(code)

        snippet = CodeSnippet(
            content=code,
            start_line=1,
            end_line=len(code.split("\n")),
            file_path=file_path,
            language=language,
            context=self._extract_context(code),
        )

        self.code_snippets.append(snippet)
        self.document_index[file_path] = {
            "embedding": embedding,
            "snippet": snippet,
            "metadata": {
                "language": language,
                "size": len(code),
                "lines": len(code.split("\n")),
            },
        }

    def search_similar_code(self, query: str, top_k: int = 5) -> List[CodeCitation]:
        """Search for similar code using neural embeddings."""
        query_embedding = self.embedding_model.encode_code(query)
        results = []

        for file_path, doc_data in self.document_index.items():
            similarity = self._cosine_similarity(query_embedding, doc_data["embedding"])

            if similarity >= self.similarity_threshold:
                citation = CodeCitation(
                    snippet=doc_data["snippet"],
                    source_file=file_path,
                    source_line=1,
                    similarity_score=similarity,
                    citation_type="similar" if similarity < 0.95 else "exact",
                )
                results.append(citation)

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _extract_context(self, code: str) -> str:
        """Extract contextual information from code."""
        lines = code.split("\n")
        if len(lines) <= 10:
            return code

        # Return first and last few lines for context
        return "\n".join(lines[:5] + ["..."] + lines[-5:])


class CodeHighlighter:
    """Intelligent code highlighting with ML-powered suggestions."""

    def __init__(self, analyzer: CodeAnalyzer):
        self.analyzer = analyzer
        self.highlight_patterns = {
            "python": {
                "keywords": r"\b(def|class|import|from|if|else|elif|for|while|try|except|finally|with|as|return|yield|break|continue|pass|raise|assert|del|global|nonlocal|lambda|True|False|None)\b",
                "strings": r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"]*"|\'[^\']*\')',
                "comments": r'(#.*$|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
                "numbers": r"\b\d+\.?\d*\b",
                "functions": r"\b\w+(?=\()",
                "classes": r"\b[A-Z][a-zA-Z0-9_]*\b",
            }
        }

        # Deprecated Python functions patterns
        self.deprecated_patterns = {
            "python": {
                "asyncio_get_event_loop": {
                    "pattern": r"asyncio\.get_event_loop\s*\(",
                    "message": "asyncio.get_event_loop() is deprecated since Python 3.7. Use asyncio.get_running_loop() or asyncio.new_event_loop() instead.",
                    "severity": "high",
                    "replacement": "asyncio.get_running_loop()",
                },
                "collections_abc": {
                    "pattern": r"from collections import (MutableMapping|MutableSequence|MutableSet|Mapping|Sequence|Set)",
                    "message": "collections.abc classes are deprecated. Import from collections.abc instead.",
                    "severity": "medium",
                    "replacement": "from collections.abc import {class_name}",
                },
                "collections_abc_usage": {
                    "pattern": r"collections\.(MutableMapping|MutableSequence|MutableSet|Mapping|Sequence|Set)",
                    "message": "collections.abc classes are deprecated. Use collections.abc.{class_name} instead.",
                    "severity": "medium",
                    "replacement": "collections.abc.{class_name}",
                },
                "urllib_parse_quote": {
                    "pattern": r"urllib\.quote\s*\(",
                    "message": "urllib.quote() is deprecated. Use urllib.parse.quote() instead.",
                    "severity": "high",
                    "replacement": "urllib.parse.quote()",
                },
                "urllib_parse_unquote": {
                    "pattern": r"urllib\.unquote\s*\(",
                    "message": "urllib.unquote() is deprecated. Use urllib.parse.unquote() instead.",
                    "severity": "high",
                    "replacement": "urllib.parse.unquote()",
                },
                "urllib_parse_urlencode": {
                    "pattern": r"urllib\.urlencode\s*\(",
                    "message": "urllib.urlencode() is deprecated. Use urllib.parse.urlencode() instead.",
                    "severity": "high",
                    "replacement": "urllib.parse.urlencode()",
                },
                "urllib2": {
                    "pattern": r"import urllib2|from urllib2 import",
                    "message": "urllib2 is deprecated. Use urllib.request instead.",
                    "severity": "high",
                    "replacement": "urllib.request",
                },
                "configparser_has_option": {
                    "pattern": r"\.has_option\s*\(",
                    "message": "configparser.has_option() is deprecated. Use the in operator instead.",
                    "severity": "medium",
                    "replacement": 'Use "option_name in config_section" instead',
                },
                "threading_current_thread": {
                    "pattern": r"threading\.current_thread\s*\(",
                    "message": "threading.current_thread() is deprecated. Use threading.current_thread() instead.",
                    "severity": "medium",
                    "replacement": "threading.current_thread()",
                },
                "unittest_assertRaisesRegexp": {
                    "pattern": r"assertRaisesRegexp\s*\(",
                    "message": "assertRaisesRegexp is deprecated. Use assertRaisesRegex instead.",
                    "severity": "medium",
                    "replacement": "assertRaisesRegex",
                },
                "os_path_walk": {
                    "pattern": r"os\.path\.walk\s*\(",
                    "message": "os.path.walk() is deprecated. Use os.walk() instead.",
                    "severity": "high",
                    "replacement": "os.walk()",
                },
                "string_letters": {
                    "pattern": r"string\.letters",
                    "message": "string.letters is deprecated. Use string.ascii_letters instead.",
                    "severity": "medium",
                    "replacement": "string.ascii_letters",
                },
                "string_lowercase": {
                    "pattern": r"string\.lowercase",
                    "message": "string.lowercase is deprecated. Use string.ascii_lowercase instead.",
                    "severity": "medium",
                    "replacement": "string.ascii_lowercase",
                },
                "string_uppercase": {
                    "pattern": r"string\.uppercase",
                    "message": "string.uppercase is deprecated. Use string.ascii_uppercase instead.",
                    "severity": "medium",
                    "replacement": "string.ascii_uppercase",
                },
                "string_digits": {
                    "pattern": r"string\.digits",
                    "message": "string.digits is deprecated. Use string.digits (still valid) or string.ascii_digits instead.",
                    "severity": "low",
                    "replacement": "string.ascii_digits",
                },
                "string_whitespace": {
                    "pattern": r"string\.whitespace",
                    "message": "string.whitespace is deprecated. Use string.whitespace (still valid) or string.ascii_whitespace instead.",
                    "severity": "low",
                    "replacement": "string.ascii_whitespace",
                },
                "imp_module": {
                    "pattern": r"import imp|from imp import",
                    "message": "imp module is deprecated. Use importlib instead.",
                    "severity": "high",
                    "replacement": "importlib",
                },
                "imp_load_source": {
                    "pattern": r"imp\.load_source\s*\(",
                    "message": "imp.load_source() is deprecated. Use importlib.util.spec_from_file_location() instead.",
                    "severity": "high",
                    "replacement": "importlib.util.spec_from_file_location()",
                },
                "imp_load_module": {
                    "pattern": r"imp\.load_module\s*\(",
                    "message": "imp.load_module() is deprecated. Use importlib.util.module_from_spec() instead.",
                    "severity": "high",
                    "replacement": "importlib.util.module_from_spec()",
                },
            }
        }

    def highlight_code(self, code: str, language: str = "python") -> HighlightedCode:
        """Highlight code with intelligent annotations."""
        highlights = []
        suggestions = []
        warnings = []

        # Analyze complexity
        complexity_metrics = self.analyzer.analyze_complexity(code, language)

        # Generate highlights
        highlights.extend(self._generate_syntax_highlights(code, language))
        highlights.extend(self._generate_semantic_highlights(code, language))
        highlights.extend(self._generate_deprecated_highlights(code, language))

        # Generate suggestions
        suggestions.extend(self._generate_suggestions(code, complexity_metrics))

        # Generate warnings
        warnings.extend(self._generate_warnings(code, complexity_metrics))
        warnings.extend(self._generate_deprecated_warnings(code, language))

        return HighlightedCode(
            code=code,
            highlights=highlights,
            suggestions=suggestions,
            warnings=warnings,
            complexity_score=complexity_metrics.get("cyclomatic_complexity", 1),
        )

    def _generate_syntax_highlights(
        self, code: str, language: str
    ) -> List[Dict[str, Any]]:
        """Generate syntax-based highlights."""
        highlights = []
        patterns = self.highlight_patterns.get(language, {})

        for highlight_type, pattern in patterns.items():
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                highlights.append(
                    {
                        "type": highlight_type,
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "color": self._get_highlight_color(highlight_type),
                    }
                )

        return highlights

    def _generate_semantic_highlights(
        self, code: str, language: str
    ) -> List[Dict[str, Any]]:
        """Generate semantic highlights using ML insights."""
        highlights = []

        # Highlight potential issues
        if language == "python":
            # Highlight unused imports
            if "import" in code and "from" in code:
                highlights.append(
                    {
                        "type": "potential_issue",
                        "start": code.find("import"),
                        "end": code.find("import") + 6,
                        "text": "import",
                        "color": "orange",
                        "message": "Consider using specific imports",
                    }
                )

            # Highlight long functions
            lines = code.split("\n")
            if len(lines) > 20:
                highlights.append(
                    {
                        "type": "complexity_warning",
                        "start": 0,
                        "end": len(code),
                        "text": code,
                        "color": "yellow",
                        "message": "Function is quite long, consider breaking it down",
                    }
                )

        return highlights

    def _generate_suggestions(self, code: str, metrics: Dict[str, Any]) -> List[str]:
        """Generate intelligent suggestions based on code analysis."""
        suggestions = []

        if metrics.get("cyclomatic_complexity", 0) > 10:
            suggestions.append(
                "Consider breaking down complex logic into smaller functions"
            )

        if metrics.get("nested_depth", 0) > 4:
            suggestions.append(
                "Deep nesting detected. Consider using early returns or guard clauses"
            )

        if metrics.get("lines_of_code", 0) > 50:
            suggestions.append(
                "Large function detected. Consider splitting into smaller, focused functions"
            )

        if "import *" in code:
            suggestions.append(
                "Avoid wildcard imports. Import specific modules for better clarity"
            )

        return suggestions

    def _generate_warnings(self, code: str, metrics: Dict[str, Any]) -> List[str]:
        """Generate warnings for potential issues."""
        warnings = []

        if metrics.get("cyclomatic_complexity", 0) > 15:
            warnings.append("Very high cyclomatic complexity detected")

        if "TODO" in code or "FIXME" in code:
            warnings.append("TODO/FIXME comments found - consider addressing these")

        if "print(" in code:
            warnings.append("Consider using proper logging instead of print statements")

        return warnings

    def _generate_deprecated_highlights(
        self, code: str, language: str
    ) -> List[Dict[str, Any]]:
        """Generate highlights for deprecated functions."""
        highlights = []

        if language == "python":
            deprecated_patterns = self.deprecated_patterns.get("python", {})

            for deprecation_type, config in deprecated_patterns.items():
                pattern = config["pattern"]
                matches = re.finditer(pattern, code, re.MULTILINE)

                for match in matches:
                    highlights.append(
                        {
                            "type": "deprecated_function",
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(),
                            "color": "red",
                            "message": config["message"],
                            "severity": config["severity"],
                            "replacement": config["replacement"],
                            "deprecation_type": deprecation_type,
                        }
                    )

        return highlights

    def _generate_deprecated_warnings(self, code: str, language: str) -> List[str]:
        """Generate warnings for deprecated functions."""
        warnings = []

        if language == "python":
            deprecated_patterns = self.deprecated_patterns.get("python", {})

            for deprecation_type, config in deprecated_patterns.items():
                pattern = config["pattern"]
                matches = list(re.finditer(pattern, code, re.MULTILINE))

                if matches:
                    count = len(matches)
                    severity = config["severity"]
                    message = config["message"]

                    if count == 1:
                        warnings.append(f"Deprecated function detected: {message}")
                    else:
                        warnings.append(
                            f"Found {count} instances of deprecated function: {message}"
                        )

        return warnings

    def _get_highlight_color(self, highlight_type: str) -> str:
        """Get color for highlight type."""
        color_map = {
            "keywords": "blue",
            "strings": "green",
            "comments": "gray",
            "numbers": "purple",
            "functions": "orange",
            "classes": "cyan",
            "deprecated_function": "red",
        }
        return color_map.get(highlight_type, "black")


class AICodeCompanion:
    """Main AI Code Companion class that orchestrates all functionality."""

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.embedding_model = CodeEmbeddingModel(model_name)
        self.analyzer = CodeAnalyzer()
        self.highlighter = CodeHighlighter(self.analyzer)
        self.retriever = IntelligentDocumentRetriever(self.embedding_model)

        # Initialize interdisciplinary reviewer
        try:
            from src.core.interdisciplinary_reviewer import InterdisciplinaryReviewer

            self.idr_reviewer = InterdisciplinaryReviewer()
        except ImportError:
            logging.warning("Interdisciplinary reviewer not available")
            self.idr_reviewer = None

        # Initialize code generation model
        try:
            self.code_generator = CodeT5ForConditionalGeneration.from_pretrained(
                "Salesforce/codet5-base"
            )
            self.code_tokenizer = CodeT5Tokenizer.from_pretrained(
                "Salesforce/codet5-base"
            )
        except Exception as e:
            logging.warning(f"Could not load code generation model: {e}")
            self.code_generator = None
            self.code_tokenizer = None

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive analysis of a code file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            # Determine language from file extension
            language = self._detect_language(file_path)

            # Index for retrieval
            self.retriever.index_document(file_path, code, language)

            # Analyze and highlight
            complexity_metrics = self.analyzer.analyze_complexity(code, language)
            highlighted_code = self.highlighter.highlight_code(code, language)

            return {
                "file_path": file_path,
                "language": language,
                "metrics": complexity_metrics,
                "highlights": highlighted_code.highlights,
                "suggestions": highlighted_code.suggestions,
                "warnings": highlighted_code.warnings,
                "complexity_score": highlighted_code.complexity_score,
            }
        except Exception as e:
            logging.error(f"Error analyzing file {file_path}: {e}")
            return {"error": str(e)}

    def search_codebase(self, query: str, top_k: int = 5) -> List[CodeCitation]:
        """Search codebase for similar code."""
        return self.retriever.search_similar_code(query, top_k)

    def generate_code_suggestions(
        self, context: str, language: str = "python"
    ) -> List[str]:
        """Generate code suggestions using neural models."""
        if not self.code_generator or not self.code_tokenizer:
            return ["Code generation model not available"]

        try:
            inputs = self.code_tokenizer(
                context, return_tensors="pt", truncation=True, max_length=512
            )

            with torch.no_grad():
                generated_ids = self.code_generator.generate(
                    inputs.input_ids,
                    max_length=100,
                    num_return_sequences=3,
                    temperature=0.7,
                    do_sample=True,
                )

            suggestions = []
            for ids in generated_ids:
                suggestion = self.code_tokenizer.decode(ids, skip_special_tokens=True)
                suggestions.append(suggestion)

            return suggestions
        except Exception as e:
            logging.error(f"Error generating code suggestions: {e}")
            return ["Error generating suggestions"]

    def get_code_citations(self, code: str) -> List[CodeCitation]:
        """Find citations for given code in the indexed codebase."""
        return self.retriever.search_similar_code(code)

    def perform_interdisciplinary_review(
        self, file_path: str, frameworks: List[str] = None, domain: str = "general"
    ) -> Dict[str, Any]:
        """Perform comprehensive interdisciplinary review (IDR) of code."""
        if not self.idr_reviewer:
            return {"error": "Interdisciplinary reviewer not available"}

        try:
            result = self.idr_reviewer.perform_review(file_path, frameworks, domain)
            return {
                "file_path": result.file_path,
                "overall_risk_score": result.overall_risk_score,
                "perspectives": {
                    name: {
                        "name": perspective.name,
                        "description": perspective.description,
                        "risk_level": perspective.risk_level,
                        "confidence": perspective.confidence,
                        "findings": perspective.findings,
                    }
                    for name, perspective in result.perspectives.items()
                },
                "recommendations": result.recommendations,
                "compliance_status": result.compliance_status,
                "business_impact": result.business_impact,
            }
        except Exception as e:
            logging.error(f"Error performing IDR: {e}")
            return {"error": str(e)}

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".swift": "swift",
            ".kt": "kotlin",
        }
        return language_map.get(ext, "unknown")


def main():
    """Example usage of the AI Code Companion."""
    companion = AICodeCompanion()

    # Example: Analyze a Python file
    result = companion.analyze_file("example.py")
    print(json.dumps(result, indent=2))

    # Example: Search for similar code
    citations = companion.search_codebase("def calculate_complexity")
    for citation in citations:
        print(
            f"Similar code found in {citation.source_file} (similarity: {citation.similarity_score:.2f})"
        )


if __name__ == "__main__":
    main()
