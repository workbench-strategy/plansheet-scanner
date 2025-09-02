"""
Foundation Elements Package

Core foundation elements for engineering drawing analysis:
- North Arrow Detection
- Scale Detection  
- Legend Extraction
- Notes Extraction
- Coordinate System Analysis
- Drawing Set Analysis

This package provides the fundamental building blocks for understanding
engineering drawings before moving to discipline-specific analysis.
"""

from .coordinate_system_analyzer import CoordinateSystemAnalyzer
from .drawing_set_analyzer import DrawingSetAnalyzer
from .foundation_orchestrator import FoundationOrchestrator
from .legend_extractor import LegendExtractor
from .north_arrow_detector import NorthArrowDetector
from .notes_extractor import NotesExtractor
from .scale_detector import ScaleDetector

__all__ = [
    "NorthArrowDetector",
    "ScaleDetector",
    "LegendExtractor",
    "NotesExtractor",
    "CoordinateSystemAnalyzer",
    "DrawingSetAnalyzer",
    "FoundationOrchestrator",
]

__version__ = "1.0.0"
__author__ = "Plansheet Scanner Team"
