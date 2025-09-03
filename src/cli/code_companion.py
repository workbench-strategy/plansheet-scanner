#!/usr/bin/env python3
"""
AI Code Companion CLI
Provides command-line interface for intelligent code analysis, highlighting, and retrieval.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.code_companion import AICodeCompanion, CodeCitation


def print_highlighted_code(code: str, highlights: List[Dict[str, Any]]):
    """Print code with ANSI color highlighting."""
    if not highlights:
        print(code)
        return

    # Sort highlights by start position
    highlights.sort(key=lambda x: x["start"])

    last_end = 0
    for highlight in highlights:
        # Print text before highlight
        if highlight["start"] > last_end:
            print(code[last_end : highlight["start"]], end="")

        # Print highlighted text with color
        color_code = get_ansi_color(highlight.get("color", "white"))
        print(f"{color_code}{code[highlight['start']:highlight['end']]}\033[0m", end="")

        last_end = highlight["end"]

    # Print remaining text
    if last_end < len(code):
        print(code[last_end:], end="")


def get_ansi_color(color: str) -> str:
    """Get ANSI color code."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "gray": "\033[90m",
        "orange": "\033[33m",
    }
    return colors.get(color, "\033[0m")


def print_citations(citations: List[CodeCitation]):
    """Print code citations in a formatted way."""
    if not citations:
        print("No similar code found.")
        return

    print(f"\n📚 Found {len(citations)} similar code snippets:")
    print("=" * 60)

    for i, citation in enumerate(citations, 1):
        print(f"\n{i}. File: {citation.source_file}")
        print(f"   Similarity: {citation.similarity_score:.2f}")
        print(f"   Type: {citation.citation_type}")
        print(f"   Lines: {citation.snippet.start_line}-{citation.snippet.end_line}")
        print(f"   Language: {citation.snippet.language}")

        # Show context
        context_lines = citation.snippet.context.split("\n")
        if len(context_lines) > 6:
            context_lines = context_lines[:3] + ["..."] + context_lines[-3:]

        print("   Context:")
        for line in context_lines:
            print(f"     {line}")
        print("-" * 40)


def analyze_command(args):
    """Handle the analyze command."""
    companion = AICodeCompanion()

    if not os.path.exists(args.file):
        print(f"❌ Error: File {args.file} not found.")
        return 1

    print(f"🔍 Analyzing {args.file}...")
    result = companion.analyze_file(args.file)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return 1

    # Print analysis results
    print(f"\n📊 Analysis Results for {result['file_path']}")
    print("=" * 50)
    print(f"Language: {result['language']}")
    print(f"Complexity Score: {result['complexity_score']}")

    # Print metrics
    if "metrics" in result:
        metrics = result["metrics"]
        print(f"\n📈 Metrics:")
        for key, value in metrics.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

    # Print warnings
    if result["warnings"]:
        print(f"\n⚠️  Warnings:")
        for warning in result["warnings"]:
            print(f"  • {warning}")

    # Print suggestions
    if result["suggestions"]:
        print(f"\n💡 Suggestions:")
        for suggestion in result["suggestions"]:
            print(f"  • {suggestion}")

    # Print highlighted code if requested
    if args.show_code:
        print(f"\n🔍 Highlighted Code:")
        print("-" * 50)
        with open(args.file, "r") as f:
            code = f.read()
        print_highlighted_code(code, result["highlights"])
        print()

    # Save results to JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"💾 Results saved to {args.output}")

    return 0


def search_command(args):
    """Handle the search command."""
    companion = AICodeCompanion()

    # Index files if directory provided
    if args.directory:
        print(f"📚 Indexing files in {args.directory}...")
        for file_path in Path(args.directory).rglob("*"):
            if file_path.is_file() and file_path.suffix in [
                ".py",
                ".js",
                ".java",
                ".cpp",
                ".c",
                ".cs",
            ]:
                try:
                    companion.analyze_file(str(file_path))
                except Exception as e:
                    print(f"Warning: Could not index {file_path}: {e}")

    # Index specific files
    if args.files:
        for file_path in args.files:
            if os.path.exists(file_path):
                companion.analyze_file(file_path)
            else:
                print(f"Warning: File {file_path} not found.")

    print(f"🔍 Searching for: {args.query}")
    citations = companion.search_codebase(args.query, args.top_k)
    print_citations(citations)

    return 0


def suggest_command(args):
    """Handle the suggest command."""
    companion = AICodeCompanion()

    if args.context:
        context = args.context
    elif args.file:
        with open(args.file, "r") as f:
            context = f.read()
    else:
        print("❌ Error: Must provide either --context or --file")
        return 1

    print(f"🤖 Generating code suggestions...")
    suggestions = companion.generate_code_suggestions(context, args.language)

    print(f"\n💡 Code Suggestions:")
    print("=" * 40)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion}")

    return 0


def cite_command(args):
    """Handle the cite command."""
    companion = AICodeCompanion()

    # Index files if directory provided
    if args.directory:
        print(f"📚 Indexing files in {args.directory}...")
        for file_path in Path(args.directory).rglob("*"):
            if file_path.is_file() and file_path.suffix in [
                ".py",
                ".js",
                ".java",
                ".cpp",
                ".c",
                ".cs",
            ]:
                try:
                    companion.analyze_file(str(file_path))
                except Exception as e:
                    print(f"Warning: Could not index {file_path}: {e}")

    if args.code:
        code = args.code
    elif args.file:
        with open(args.file, "r") as f:
            code = f.read()
    else:
        print("❌ Error: Must provide either --code or --file")
        return 1

    print(f"🔍 Finding citations for code...")
    citations = companion.get_code_citations(code)
    print_citations(citations)

    return 0


def idr_command(args):
    """Handle the interdisciplinary review command."""
    companion = AICodeCompanion()

    if not os.path.exists(args.file):
        print(f"❌ Error: File {args.file} not found.")
        return 1

    print(f"🔍 Performing interdisciplinary review of {args.file}...")
    print(f"📋 Frameworks: {', '.join(args.frameworks)}")
    print(f"🏢 Domain: {args.domain}")

    result = companion.perform_interdisciplinary_review(
        args.file, args.frameworks, args.domain
    )

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return 1

    # Display results
    print(f"\n📊 Interdisciplinary Review Results")
    print("=" * 60)
    print(f"File: {result['file_path']}")
    print(f"Overall Risk Score: {result['overall_risk_score']:.2f}")

    # Display perspectives
    print(f"\n🔍 Review Perspectives:")
    for perspective_name, perspective in result["perspectives"].items():
        print(f"\n{perspective['name']} ({perspective['risk_level'].upper()} risk)")
        print(f"  Description: {perspective['description']}")
        print(f"  Confidence: {perspective['confidence']:.2f}")

        if perspective["findings"]:
            print(f"  Findings ({len(perspective['findings'])}):")
            for i, finding in enumerate(perspective["findings"][:3], 1):  # Show top 3
                print(f"    {i}. {finding.get('description', 'Finding')}")
                if "recommendation" in finding:
                    print(f"       Recommendation: {finding['recommendation']}")
        else:
            print("  No issues found")

    # Display compliance status
    print(f"\n✅ Compliance Status:")
    for framework, compliant in result["compliance_status"].items():
        status = "✅ Compliant" if compliant else "❌ Non-compliant"
        print(f"  {framework.title()}: {status}")

    # Display recommendations
    if result["recommendations"]:
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(result["recommendations"][:5], 1):  # Show top 5
            print(f"  {i}. {rec}")

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Code Companion - Intelligent code analysis and retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a Python file
  python code_companion.py analyze example.py --show-code

  # Search for similar code in a directory
  python code_companion.py search "def calculate_complexity" --directory ./src

  # Generate code suggestions
  python code_companion.py suggest --context "def process_data(data):"

  # Find citations for code
  python code_companion.py cite --code "def calculate_complexity" --directory ./src
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a code file")
    analyze_parser.add_argument("file", help="File to analyze")
    analyze_parser.add_argument(
        "--show-code", action="store_true", help="Show highlighted code"
    )
    analyze_parser.add_argument("--output", help="Save results to JSON file")
    analyze_parser.set_defaults(func=analyze_command)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar code")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--directory", help="Directory to search in")
    search_parser.add_argument("--files", nargs="+", help="Specific files to index")
    search_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )
    search_parser.set_defaults(func=search_command)

    # Suggest command
    suggest_parser = subparsers.add_parser("suggest", help="Generate code suggestions")
    suggest_parser.add_argument("--context", help="Code context")
    suggest_parser.add_argument("--file", help="File containing context")
    suggest_parser.add_argument(
        "--language", default="python", help="Programming language"
    )
    suggest_parser.set_defaults(func=suggest_command)

    # Cite command
    cite_parser = subparsers.add_parser("cite", help="Find citations for code")
    cite_parser.add_argument("--code", help="Code to find citations for")
    cite_parser.add_argument("--file", help="File containing code")
    cite_parser.add_argument("--directory", help="Directory to search in")
    cite_parser.set_defaults(func=cite_command)

    # IDR command
    idr_parser = subparsers.add_parser("idr", help="Perform interdisciplinary review")
    idr_parser.add_argument("file", help="File to review")
    idr_parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["gdpr", "sox", "hipaa"],
        help="Compliance frameworks to check",
    )
    idr_parser.add_argument("--domain", default="general", help="Business domain")
    idr_parser.add_argument("--output", help="Save results to JSON file")
    idr_parser.set_defaults(func=idr_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
