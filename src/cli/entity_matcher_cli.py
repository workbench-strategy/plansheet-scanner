#!/usr/bin/env python
"""
Engineering Document Entity Matcher CLI Tool

This script provides a command-line interface for matching and highlighting
entities (like cable types, equipment IDs, etc.) across engineering documents.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from core.document_entity_matcher import (
    DataValidator, 
    EntityExtractor,
    EntityMatcher,
    OutputGenerator,
    configure_parser,
    main as core_main
)

def main():
    """Main entry point for the CLI tool."""
    exit_code = core_main()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
