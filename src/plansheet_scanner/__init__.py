"""
PlanSheet Scanner - ML-powered plansheet scanner for engineering drawings and traffic plans.

A comprehensive tool for extracting, processing, and analyzing engineering plan sheets
using machine learning and computer vision techniques.
"""

__version__ = "1.0.0"
__author__ = "HNTB DIS SEA_DTS Python Working Group"
__email__ = "your-email@example.com"
__description__ = (
    "ML-powered plansheet scanner for engineering drawings and traffic plans"
)

# Core imports
from .core.adaptive_reviewer import AdaptiveReviewer
from .core.code_companion import CodeCompanion
from .core.foundation_trainer import FoundationTrainer
from .core.interdisciplinary_reviewer import InterdisciplinaryReviewer
from .core.traffic_plan_reviewer import TrafficPlanReviewer

# Main classes
__all__ = [
    "AdaptiveReviewer",
    "TrafficPlanReviewer",
    "InterdisciplinaryReviewer",
    "FoundationTrainer",
    "CodeCompanion",
]
