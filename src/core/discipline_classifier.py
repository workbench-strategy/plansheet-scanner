"""
Discipline Classification System

Uses foundation elements and index symbol recognition to classify engineering drawings
by discipline (Electrical, Structural, Civil, Traffic, etc.) with high accuracy.

This system integrates:
- Foundation elements (legend, notes, coordinate systems)
- Index symbol recognition from existing symbol training
- Multi-stage classification (primary discipline, sub-discipline, drawing type)
- Confidence scoring and validation
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Import foundation elements
from ..foundation_elements import (
    CoordinateSystemAnalyzer,
    FoundationOrchestrator,
    LegendExtractor,
    NotesExtractor,
)
from .line_matcher import LineMatcher

# Import existing symbol recognition
from .ml_enhanced_symbol_recognition import MLSymbolRecognizer, SymbolDetection

logger = logging.getLogger(__name__)


@dataclass
class DisciplineClassification:
    """Result of discipline classification analysis."""

    primary_discipline: str
    sub_discipline: str
    drawing_type: str
    confidence: float
    supporting_evidence: List[str]
    index_symbols: List[str]
    foundation_score: float
    classification_method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IndexSymbol:
    """Represents an index symbol found in the drawing."""

    symbol_id: str
    symbol_name: str
    discipline: str
    confidence: float
    position: Tuple[int, int]
    context: str
    variations: List[str] = field(default_factory=list)


class DisciplineClassifier:
    """
    Multi-stage discipline classification system using foundation elements and index symbols.

    Classification stages:
    1. Primary discipline (Electrical, Structural, Civil, Traffic, etc.)
    2. Sub-discipline (Power, Lighting, Concrete, Steel, Drainage, etc.)
    3. Drawing type (Plan, Section, Detail, Schedule, etc.)
    """

    def __init__(self, symbol_model_path: Optional[str] = None):
        """
        Initialize the discipline classifier.

        Args:
            symbol_model_path: Path to trained symbol recognition model
        """
        # Initialize foundation elements
        self.foundation_orchestrator = FoundationOrchestrator()
        self.legend_extractor = LegendExtractor()
        self.notes_extractor = NotesExtractor()
        self.coordinate_analyzer = CoordinateSystemAnalyzer()

        # Initialize symbol recognition
        self.symbol_recognizer = MLSymbolRecognizer(symbol_model_path)
        self.line_matcher = LineMatcher()

        # Discipline definitions with index symbols
        self.discipline_definitions = {
            "electrical": {
                "primary_keywords": [
                    "electrical",
                    "power",
                    "lighting",
                    "conduit",
                    "cable",
                ],
                "index_symbols": {
                    "conduit": ["COND", "EMT", "PVC", "RMC", "CONDUIT"],
                    "junction_box": ["JB", "JBOX", "JUNCTION", "BOX"],
                    "transformer": ["XFMR", "TRANS", "TRANSFORMER"],
                    "lighting": ["LIGHT", "LAMP", "FIXTURE", "POLE"],
                    "panel": ["PANEL", "SWBD", "DISTRIBUTION"],
                    "grounding": ["GND", "GROUND", "EARTH"],
                },
                "sub_disciplines": ["power", "lighting", "communications", "controls"],
                "drawing_types": [
                    "power_plan",
                    "lighting_plan",
                    "single_line",
                    "panel_schedule",
                ],
            },
            "structural": {
                "primary_keywords": [
                    "structural",
                    "concrete",
                    "steel",
                    "reinforcement",
                    "beam",
                ],
                "index_symbols": {
                    "beam": ["BEAM", "GIRDER", "JOIST", "TRUSS"],
                    "column": ["COL", "COLUMN", "POST", "PIER"],
                    "foundation": ["FOOTING", "PILE", "CAISSON", "SLAB"],
                    "reinforcement": ["REBAR", "REINF", "STEEL", "BAR"],
                    "connection": ["BOLT", "WELD", "PLATE", "ANGLE"],
                    "expansion_joint": ["EXP", "JOINT", "EXPANSION"],
                },
                "sub_disciplines": ["concrete", "steel", "timber", "masonry"],
                "drawing_types": [
                    "framing_plan",
                    "foundation_plan",
                    "section",
                    "detail",
                ],
            },
            "civil": {
                "primary_keywords": [
                    "civil",
                    "drainage",
                    "grading",
                    "earthwork",
                    "utilities",
                ],
                "index_symbols": {
                    "catch_basin": ["CB", "CATCH", "BASIN", "INLET"],
                    "manhole": ["MH", "MANHOLE", "VAULT"],
                    "pipe": ["PIPE", "CULVERT", "DRAIN", "SEWER"],
                    "grade": ["GRADE", "SLOPE", "ELEV", "BENCHMARK"],
                    "curb": ["CURB", "GUTTER", "EDGE"],
                    "pavement": ["PAVEMENT", "ASPHALT", "CONCRETE"],
                },
                "sub_disciplines": ["drainage", "grading", "utilities", "pavement"],
                "drawing_types": [
                    "drainage_plan",
                    "grading_plan",
                    "utility_plan",
                    "profile",
                ],
            },
            "traffic": {
                "primary_keywords": [
                    "traffic",
                    "signal",
                    "sign",
                    "marking",
                    "detector",
                ],
                "index_symbols": {
                    "traffic_signal": ["TS", "SIGNAL", "LIGHT", "TRAFFIC"],
                    "detector": ["DET", "LOOP", "SENSOR", "CAMERA"],
                    "sign": ["SIGN", "STOP", "YIELD", "WARNING"],
                    "marking": ["MARK", "STRIPE", "CROSSWALK", "STOP_BAR"],
                    "pedestrian": ["PED", "CROSSWALK", "RAMP", "BUTTON"],
                    "controller": ["CTRL", "CONTROLLER", "CABINET"],
                },
                "sub_disciplines": ["signals", "signs", "markings", "detection"],
                "drawing_types": [
                    "signal_plan",
                    "sign_plan",
                    "marking_plan",
                    "detection_plan",
                ],
            },
            "mechanical": {
                "primary_keywords": [
                    "mechanical",
                    "hvac",
                    "ventilation",
                    "heating",
                    "cooling",
                ],
                "index_symbols": {
                    "duct": ["DUCT", "AIR", "VENT", "RETURN"],
                    "equipment": ["AHU", "RTU", "UNIT", "HANDLER"],
                    "diffuser": ["DIFF", "REG", "GRILLE", "VENT"],
                    "pump": ["PUMP", "MOTOR", "FAN", "BLOWER"],
                    "valve": ["VALVE", "DAMPER", "VAV", "CONTROL"],
                },
                "sub_disciplines": ["hvac", "plumbing", "fire_protection"],
                "drawing_types": ["hvac_plan", "ductwork", "equipment_schedule"],
            },
            "landscape": {
                "primary_keywords": [
                    "landscape",
                    "irrigation",
                    "planting",
                    "tree",
                    "shrub",
                ],
                "index_symbols": {
                    "tree": ["TREE", "SHRUB", "PLANT", "VEGETATION"],
                    "irrigation": ["IRR", "SPRINKLER", "HEAD", "VALVE"],
                    "hardscape": ["PAVER", "WALL", "FENCE", "SEAT"],
                    "lighting": ["LIGHT", "PATH", "ACCENT", "FLOOD"],
                },
                "sub_disciplines": ["planting", "irrigation", "hardscape", "lighting"],
                "drawing_types": ["planting_plan", "irrigation_plan", "hardscape_plan"],
            },
        }

        # Load trained models if available
        self._load_trained_models()

        logger.info(
            "DisciplineClassifier initialized with foundation elements and symbol recognition"
        )

    def _load_trained_models(self):
        """Load any pre-trained models for discipline classification."""
        # Check for trained symbol recognition model
        model_paths = [
            "symbol_models/best_model.pth",
            "models/symbol_recognition_model.pth",
            "trained_models/symbol_classifier.pkl",
        ]

        for path in model_paths:
            if Path(path).exists():
                try:
                    self.symbol_recognizer.load_model(path)
                    logger.info(f"Loaded symbol recognition model from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model from {path}: {e}")

    def classify_discipline(
        self, drawing_path: str, page_number: int = 0
    ) -> DisciplineClassification:
        """
        Classify the discipline of an engineering drawing.

        Args:
            drawing_path: Path to the drawing file
            page_number: Page number to analyze (0-indexed)

        Returns:
            DisciplineClassification with primary discipline, sub-discipline, and confidence
        """
        logger.info(f"Classifying discipline for {drawing_path} page {page_number}")

        # Step 1: Analyze foundation elements
        foundation_analysis = self.foundation_orchestrator.analyze_drawing(
            drawing_path, page_number
        )

        # Step 2: Extract index symbols
        index_symbols = self._extract_index_symbols(drawing_path, page_number)

        # Step 3: Analyze legend for discipline indicators
        legend_evidence = self._analyze_legend_for_discipline(
            foundation_analysis.legend
        )

        # Step 4: Analyze notes for discipline terminology
        notes_evidence = self._analyze_notes_for_discipline(foundation_analysis.notes)

        # Step 5: Perform multi-stage classification
        primary_discipline = self._classify_primary_discipline(
            index_symbols, legend_evidence, notes_evidence
        )

        sub_discipline = self._classify_sub_discipline(
            primary_discipline, index_symbols, notes_evidence
        )

        drawing_type = self._classify_drawing_type(
            primary_discipline, foundation_analysis
        )

        # Step 6: Calculate confidence and supporting evidence
        confidence = self._calculate_classification_confidence(
            primary_discipline, index_symbols, legend_evidence, notes_evidence
        )

        supporting_evidence = self._compile_supporting_evidence(
            index_symbols, legend_evidence, notes_evidence
        )

        return DisciplineClassification(
            primary_discipline=primary_discipline,
            sub_discipline=sub_discipline,
            drawing_type=drawing_type,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            index_symbols=[sym.symbol_name for sym in index_symbols],
            foundation_score=foundation_analysis.foundation_score,
            classification_method="foundation_elements_with_index_symbols",
        )

    def _extract_index_symbols(
        self, drawing_path: str, page_number: int
    ) -> List[IndexSymbol]:
        """
        Extract index symbols from the drawing using symbol recognition.

        Args:
            drawing_path: Path to the drawing file
            page_number: Page number to analyze

        Returns:
            List of detected index symbols
        """
        index_symbols = []

        try:
            # Use symbol recognizer to detect symbols
            symbol_results = self.symbol_recognizer.recognize_symbols(
                drawing_path, page_number
            )

            for detection in symbol_results.detections:
                # Check if this symbol matches any discipline index symbols
                for discipline, definition in self.discipline_definitions.items():
                    for symbol_type, variations in definition["index_symbols"].items():
                        for variation in variations:
                            if self._symbol_matches_variation(detection, variation):
                                index_symbol = IndexSymbol(
                                    symbol_id=f"{discipline}_{symbol_type}_{len(index_symbols)}",
                                    symbol_name=symbol_type,
                                    discipline=discipline,
                                    confidence=detection.confidence,
                                    position=detection.center,
                                    context=f"Found in {discipline} context",
                                    variations=variations,
                                )
                                index_symbols.append(index_symbol)
                                break

                        if any(sym.symbol_name == symbol_type for sym in index_symbols):
                            break

        except Exception as e:
            logger.warning(f"Error extracting index symbols: {e}")

        # Also check for text-based index symbols in notes and legends
        text_symbols = self._extract_text_based_symbols(drawing_path, page_number)
        index_symbols.extend(text_symbols)

        return index_symbols

    def _symbol_matches_variation(
        self, detection: SymbolDetection, variation: str
    ) -> bool:
        """Check if a detected symbol matches a variation pattern."""
        # This would use the symbol recognition model's classification
        # For now, we'll use a simple text matching approach
        symbol_text = detection.metadata.get("text", "").upper()
        return variation.upper() in symbol_text

    def _extract_text_based_symbols(
        self, drawing_path: str, page_number: int
    ) -> List[IndexSymbol]:
        """Extract index symbols from text in the drawing."""
        text_symbols = []

        try:
            # Use notes extractor to find text
            notes_detection = self.notes_extractor.detect_from_pdf(
                drawing_path, page_number
            )

            for note in notes_detection.notes:
                note_text = note.text.upper()

                # Check for discipline-specific patterns
                for discipline, definition in self.discipline_definitions.items():
                    for symbol_type, variations in definition["index_symbols"].items():
                        for variation in variations:
                            if variation.upper() in note_text:
                                text_symbol = IndexSymbol(
                                    symbol_id=f"text_{discipline}_{symbol_type}_{len(text_symbols)}",
                                    symbol_name=symbol_type,
                                    discipline=discipline,
                                    confidence=0.8,  # Text-based detection confidence
                                    position=(0, 0),  # Unknown position for text
                                    context=f"Found in text: {note.text[:50]}...",
                                    variations=variations,
                                )
                                text_symbols.append(text_symbol)
                                break

                        if any(sym.symbol_name == symbol_type for sym in text_symbols):
                            break

        except Exception as e:
            logger.warning(f"Error extracting text-based symbols: {e}")

        return text_symbols

    def _analyze_legend_for_discipline(self, legend_detection) -> Dict[str, float]:
        """Analyze legend content for discipline indicators."""
        discipline_scores = {
            discipline: 0.0 for discipline in self.discipline_definitions.keys()
        }

        if not legend_detection or not legend_detection.symbols:
            return discipline_scores

        for symbol in legend_detection.symbols:
            symbol_text = symbol.symbol_name.upper()

            for discipline, definition in self.discipline_definitions.items():
                # Check primary keywords
                for keyword in definition["primary_keywords"]:
                    if keyword.upper() in symbol_text:
                        discipline_scores[discipline] += symbol.confidence * 0.5

                # Check index symbols
                for symbol_type, variations in definition["index_symbols"].items():
                    for variation in variations:
                        if variation.upper() in symbol_text:
                            discipline_scores[discipline] += symbol.confidence * 1.0

        return discipline_scores

    def _analyze_notes_for_discipline(self, notes_detection) -> Dict[str, float]:
        """Analyze notes content for discipline terminology."""
        discipline_scores = {
            discipline: 0.0 for discipline in self.discipline_definitions.keys()
        }

        if not notes_detection or not notes_detection.notes:
            return discipline_scores

        for note in notes_detection.notes:
            note_text = note.text.upper()

            for discipline, definition in self.discipline_definitions.items():
                # Check primary keywords
                for keyword in definition["primary_keywords"]:
                    if keyword.upper() in note_text:
                        discipline_scores[discipline] += note.confidence * 0.3

                # Check index symbols
                for symbol_type, variations in definition["index_symbols"].items():
                    for variation in variations:
                        if variation.upper() in note_text:
                            discipline_scores[discipline] += note.confidence * 0.7

        return discipline_scores

    def _classify_primary_discipline(
        self,
        index_symbols: List[IndexSymbol],
        legend_evidence: Dict[str, float],
        notes_evidence: Dict[str, float],
    ) -> str:
        """Classify the primary discipline based on all evidence."""
        discipline_scores = {
            discipline: 0.0 for discipline in self.discipline_definitions.keys()
        }

        # Score based on index symbols
        for symbol in index_symbols:
            discipline_scores[symbol.discipline] += symbol.confidence * 2.0

        # Score based on legend evidence
        for discipline, score in legend_evidence.items():
            discipline_scores[discipline] += score * 1.5

        # Score based on notes evidence
        for discipline, score in notes_evidence.items():
            discipline_scores[discipline] += score * 1.0

        # Return discipline with highest score
        if discipline_scores:
            primary_discipline = max(discipline_scores, key=discipline_scores.get)
            if discipline_scores[primary_discipline] > 0:
                return primary_discipline

        return "unknown"

    def _classify_sub_discipline(
        self,
        primary_discipline: str,
        index_symbols: List[IndexSymbol],
        notes_evidence: Dict[str, float],
    ) -> str:
        """Classify the sub-discipline within the primary discipline."""
        if primary_discipline not in self.discipline_definitions:
            return "unknown"

        definition = self.discipline_definitions[primary_discipline]
        sub_discipline_scores = {sub: 0.0 for sub in definition["sub_disciplines"]}

        # Analyze index symbols for sub-discipline indicators
        for symbol in index_symbols:
            if symbol.discipline == primary_discipline:
                # Add logic to map symbols to sub-disciplines
                # This would be more sophisticated in practice
                sub_discipline_scores[
                    definition["sub_disciplines"][0]
                ] += symbol.confidence

        if sub_discipline_scores:
            return max(sub_discipline_scores, key=sub_discipline_scores.get)

        return (
            definition["sub_disciplines"][0]
            if definition["sub_disciplines"]
            else "unknown"
        )

    def _classify_drawing_type(
        self, primary_discipline: str, foundation_analysis
    ) -> str:
        """Classify the type of drawing (plan, section, detail, etc.)."""
        if primary_discipline not in self.discipline_definitions:
            return "unknown"

        definition = self.discipline_definitions[primary_discipline]

        # Simple logic based on foundation elements
        if foundation_analysis.scale and foundation_analysis.scale.confidence > 0.7:
            return (
                definition["drawing_types"][0]
                if definition["drawing_types"]
                else "plan"
            )

        return "unknown"

    def _calculate_classification_confidence(
        self,
        primary_discipline: str,
        index_symbols: List[IndexSymbol],
        legend_evidence: Dict[str, float],
        notes_evidence: Dict[str, float],
    ) -> float:
        """Calculate confidence in the classification."""
        if primary_discipline == "unknown":
            return 0.0

        confidence_factors = []

        # Factor 1: Number of supporting index symbols
        supporting_symbols = [
            s for s in index_symbols if s.discipline == primary_discipline
        ]
        if supporting_symbols:
            symbol_confidence = min(len(supporting_symbols) * 0.2, 1.0)
            confidence_factors.append(symbol_confidence)

        # Factor 2: Legend evidence strength
        legend_score = legend_evidence.get(primary_discipline, 0.0)
        legend_confidence = min(legend_score * 0.3, 1.0)
        confidence_factors.append(legend_confidence)

        # Factor 3: Notes evidence strength
        notes_score = notes_evidence.get(primary_discipline, 0.0)
        notes_confidence = min(notes_score * 0.2, 1.0)
        confidence_factors.append(notes_confidence)

        # Factor 4: Foundation completeness
        foundation_confidence = 0.5  # This would come from foundation analysis

        if confidence_factors:
            return min(
                sum(confidence_factors) / len(confidence_factors)
                + foundation_confidence,
                1.0,
            )

        return 0.5

    def _compile_supporting_evidence(
        self,
        index_symbols: List[IndexSymbol],
        legend_evidence: Dict[str, float],
        notes_evidence: Dict[str, float],
    ) -> List[str]:
        """Compile list of supporting evidence for the classification."""
        evidence = []

        # Add index symbol evidence
        for symbol in index_symbols:
            evidence.append(
                f"Index symbol '{symbol.symbol_name}' ({symbol.confidence:.2f})"
            )

        # Add legend evidence
        for discipline, score in legend_evidence.items():
            if score > 0:
                evidence.append(f"Legend evidence for {discipline}: {score:.2f}")

        # Add notes evidence
        for discipline, score in notes_evidence.items():
            if score > 0:
                evidence.append(f"Notes evidence for {discipline}: {score:.2f}")

        return evidence

    def batch_classify(
        self, drawing_paths: List[str]
    ) -> List[DisciplineClassification]:
        """
        Classify multiple drawings in batch.

        Args:
            drawing_paths: List of drawing file paths

        Returns:
            List of discipline classifications
        """
        classifications = []

        for i, path in enumerate(drawing_paths):
            try:
                classification = self.classify_discipline(path)
                classifications.append(classification)
                logger.info(
                    f"Classified {path}: {classification.primary_discipline} ({classification.confidence:.2f})"
                )
            except Exception as e:
                logger.error(f"Error classifying {path}: {e}")
                # Add error classification
                error_classification = DisciplineClassification(
                    primary_discipline="error",
                    sub_discipline="error",
                    drawing_type="error",
                    confidence=0.0,
                    supporting_evidence=[f"Error: {str(e)}"],
                    index_symbols=[],
                    foundation_score=0.0,
                    classification_method="error",
                )
                classifications.append(error_classification)

        return classifications

    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get statistics about the classification system."""
        return {
            "disciplines_supported": list(self.discipline_definitions.keys()),
            "total_index_symbols": sum(
                len(defn["index_symbols"])
                for defn in self.discipline_definitions.values()
            ),
            "symbol_recognizer_loaded": self.symbol_recognizer.model is not None,
            "foundation_elements_available": True,
            "classification_methods": ["foundation_elements_with_index_symbols"],
        }


def main():
    """Test the discipline classifier."""
    import argparse

    parser = argparse.ArgumentParser(description="Discipline Classification System")
    parser.add_argument("drawing_path", help="Path to drawing file or directory")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    classifier = DisciplineClassifier()

    drawing_path = Path(args.drawing_path)

    if drawing_path.is_file():
        # Single file classification
        classification = classifier.classify_discipline(str(drawing_path))

        result = {
            "drawing_path": str(drawing_path),
            "classification": {
                "primary_discipline": classification.primary_discipline,
                "sub_discipline": classification.sub_discipline,
                "drawing_type": classification.drawing_type,
                "confidence": classification.confidence,
                "supporting_evidence": classification.supporting_evidence,
                "index_symbols": classification.index_symbols,
                "foundation_score": classification.foundation_score,
                "classification_method": classification.classification_method,
                "timestamp": classification.timestamp,
            },
        }

        print(f"Primary Discipline: {classification.primary_discipline}")
        print(f"Sub-Discipline: {classification.sub_discipline}")
        print(f"Drawing Type: {classification.drawing_type}")
        print(f"Confidence: {classification.confidence:.2f}")
        print(f"Foundation Score: {classification.foundation_score:.2f}")
        print(f"Index Symbols: {classification.index_symbols}")
        print(f"Supporting Evidence: {classification.supporting_evidence}")

    elif drawing_path.is_dir():
        # Batch classification
        drawing_files = list(drawing_path.glob("*.pdf")) + list(
            drawing_path.glob("*.dwg")
        )
        classifications = classifier.batch_classify([str(f) for f in drawing_files])

        result = {
            "batch_classification": {
                "total_drawings": len(drawing_files),
                "successful_classifications": len(
                    [c for c in classifications if c.primary_discipline != "error"]
                ),
                "classifications": [
                    {
                        "drawing_path": str(drawing_files[i]),
                        "primary_discipline": c.primary_discipline,
                        "sub_discipline": c.sub_discipline,
                        "drawing_type": c.drawing_type,
                        "confidence": c.confidence,
                        "foundation_score": c.foundation_score,
                    }
                    for i, c in enumerate(classifications)
                ],
            }
        }

        # Print summary
        discipline_counts = {}
        for c in classifications:
            if c.primary_discipline != "error":
                discipline_counts[c.primary_discipline] = (
                    discipline_counts.get(c.primary_discipline, 0) + 1
                )

        print(f"Batch Classification Results:")
        print(f"Total Drawings: {len(drawing_files)}")
        print(
            f"Successful Classifications: {len([c for c in classifications if c.primary_discipline != 'error'])}"
        )
        print(f"Discipline Distribution: {discipline_counts}")

    else:
        print(f"Error: {drawing_path} does not exist")
        return

    # Save results if output file specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
