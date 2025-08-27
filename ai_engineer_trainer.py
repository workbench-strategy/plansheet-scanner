#!/usr/bin/env python3
"""
AI Engineer Trainer - Comprehensive Training System
Trains an AI engineer to understand code and review engineering drawings
using thousands of as-built drawings as training data.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
import logging
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Warning: ML dependencies not installed. Install with: pip install scikit-learn joblib torch transformers")

@dataclass
class AsBuiltDrawing:
    """Represents an as-built drawing with all relevant information."""
    drawing_id: str
    project_name: str
    sheet_number: str
    sheet_title: str
    discipline: str
    original_design: Dict[str, Any]  # Original design elements
    as_built_changes: List[Dict[str, Any]]  # Changes made during construction
    code_references: List[str]  # Building codes, standards, specifications
    review_notes: List[str]  # Review comments and findings
    approval_status: str  # approved, rejected, conditional
    reviewer_feedback: Dict[str, Any]  # Detailed reviewer feedback
    construction_notes: str  # Field notes from construction
    file_path: str
    confidence: float = 1.0

@dataclass
class CodeKnowledge:
    """Represents knowledge about building codes and standards."""
    code_section: str
    code_title: str
    requirements: List[str]
    examples: List[str]
    common_violations: List[str]
    review_checklist: List[str]

@dataclass
class ReviewPattern:
    """Represents a pattern the AI engineer learns for reviewing."""
    pattern_type: str  # code_violation, design_error, constructability, etc.
    description: str
    indicators: List[str]  # What to look for
    severity: str  # critical, major, minor
    examples: List[str]
    confidence: float

class AIEngineerTrainer:
    """Comprehensive trainer for AI engineer to understand code and review drawings."""
    
    def __init__(self, data_dir: str = "ai_engineer_data", model_dir: str = "ai_engineer_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.as_built_drawings = []
        self.code_knowledge = {}
        self.review_patterns = []
        self.training_examples = []
        
        # Models
        self.code_classifier = None
        self.review_classifier = None
        self.violation_detector = None
        self.text_analyzer = None
        
        # Code knowledge base for highway projects
        self.code_knowledge_base = {
            "traffic_control": {
                "MUTCD": {
                    "sections": ["2A", "2B", "2C", "2D", "2E", "2F", "2G", "2H"],
                    "requirements": [
                        "Signal heads must be visible from 100 feet",
                        "Pedestrian signals required at crosswalks",
                        "Detector loops must be properly sized",
                        "Sign placement must meet visibility requirements"
                    ],
                    "common_violations": [
                        "Signal heads too low",
                        "Missing pedestrian signals",
                        "Inadequate detector coverage",
                        "Signs not visible from required distance"
                    ]
                },
                "WSDOT_Standards": {
                    "sections": ["Traffic Signals", "Signing", "Markings"],
                    "requirements": [
                        "All traffic control devices must meet WSDOT standards",
                        "Signal timing must be coordinated",
                        "Signs must be retroreflective",
                        "Markings must be durable"
                    ]
                }
            },
            "electrical": {
                "NEC": {
                    "sections": ["Article 300", "Article 400", "Article 600"],
                    "requirements": [
                        "Conduit must be properly sized",
                        "Grounding must be continuous",
                        "Junction boxes must be accessible",
                        "Wire fill must not exceed limits"
                    ],
                    "common_violations": [
                        "Undersized conduit",
                        "Missing grounding",
                        "Inaccessible junction boxes",
                        "Overfilled conduit"
                    ]
                }
            },
            "structural": {
                "AASHTO": {
                    "sections": ["LRFD Bridge Design", "Standard Specifications"],
                    "requirements": [
                        "Load ratings must meet design standards",
                        "Reinforcement must be properly detailed",
                        "Concrete cover must meet requirements",
                        "Expansion joints must be properly sized"
                    ]
                }
            },
            "drainage": {
                "WSDOT_Hydraulics": {
                    "sections": ["Storm Drainage", "Culvert Design"],
                    "requirements": [
                        "Drainage capacity must meet 10-year storm",
                        "Inlets must be properly spaced",
                        "Pipe slopes must meet minimum requirements",
                        "Culverts must be properly sized"
                    ]
                }
            }
        }
        
        # Load existing data
        self._load_training_data()
    
    def add_as_built_drawing(self, drawing: AsBuiltDrawing):
        """Add an as-built drawing for training."""
        self.as_built_drawings.append(drawing)
        self._extract_learning_patterns(drawing)
        self._save_as_built_drawing(drawing)
        
        print(f"✅ Added as-built drawing: {drawing.drawing_id} - {drawing.sheet_title}")
    
    def _extract_learning_patterns(self, drawing: AsBuiltDrawing):
        """Extract learning patterns from as-built drawing."""
        # Extract code violation patterns
        for change in drawing.as_built_changes:
            if change.get("code_violation"):
                pattern = ReviewPattern(
                    pattern_type="code_violation",
                    description=f"Code violation in {drawing.discipline}: {change.get('description', '')}",
                    indicators=change.get("indicators", []),
                    severity=change.get("severity", "major"),
                    examples=[f"Drawing {drawing.sheet_number}: {change.get('description', '')}"],
                    confidence=drawing.confidence
                )
                self.review_patterns.append(pattern)
        
        # Extract design error patterns
        for note in drawing.review_notes:
            if "error" in note.lower() or "incorrect" in note.lower():
                pattern = ReviewPattern(
                    pattern_type="design_error",
                    description=f"Design error found: {note}",
                    indicators=["design inconsistency", "calculation error", "missing detail"],
                    severity="major",
                    examples=[f"Drawing {drawing.sheet_number}: {note}"],
                    confidence=drawing.confidence
                )
                self.review_patterns.append(pattern)
        
        # Extract constructability patterns
        if drawing.construction_notes:
            if "difficult" in drawing.construction_notes.lower() or "impossible" in drawing.construction_notes.lower():
                pattern = ReviewPattern(
                    pattern_type="constructability_issue",
                    description=f"Constructability issue: {drawing.construction_notes}",
                    indicators=["construction difficulty", "access issues", "coordination problems"],
                    severity="major",
                    examples=[f"Drawing {drawing.sheet_number}: {drawing.construction_notes}"],
                    confidence=drawing.confidence
                )
                self.review_patterns.append(pattern)
    
    def train_code_understanding_model(self, min_examples: int = 50):
        """Train model to understand building codes and standards."""
        if len(self.as_built_drawings) < min_examples:
            print(f"Need at least {min_examples} examples, have {len(self.as_built_drawings)}")
            return
        
        print(f"🤖 Training code understanding model on {len(self.as_built_drawings)} as-built drawings...")
        
        # Prepare training data
        X = []
        y = []
        
        for drawing in self.as_built_drawings:
            # Extract features from drawing
            features = self._extract_code_features(drawing)
            X.append(features)
            
            # Create labels for different code aspects
            code_labels = self._extract_code_labels(drawing)
            y.append(code_labels)
        
        # Train multiple models for different aspects
        self._train_code_classifier(X, y)
        self._train_violation_detector(X, y)
        
        print("✅ Code understanding models trained successfully!")
    
    def _extract_code_features(self, drawing: AsBuiltDrawing) -> List[float]:
        """Extract features related to code compliance."""
        features = []
        
        # Text analysis
        all_text = f"{drawing.sheet_title} {' '.join(drawing.review_notes)} {drawing.construction_notes}"
        all_text = all_text.lower()
        
        # Code reference features
        for code_type, codes in self.code_knowledge_base.items():
            for code_name, code_info in codes.items():
                # Count code references
                code_count = sum(1 for ref in drawing.code_references if code_name.lower() in ref.lower())
                features.append(code_count)
                
                # Count requirement violations
                violation_count = sum(1 for req in code_info.get("requirements", []) 
                                    if any(violation in all_text for violation in code_info.get("common_violations", [])))
                features.append(violation_count)
        
        # Drawing type features
        features.append(1 if "plan" in drawing.sheet_title.lower() else 0)
        features.append(1 if "detail" in drawing.sheet_title.lower() else 0)
        features.append(1 if "section" in drawing.sheet_title.lower() else 0)
        
        # Discipline features
        for discipline in ["traffic", "electrical", "structural", "drainage", "mechanical"]:
            features.append(1 if discipline in drawing.discipline.lower() else 0)
        
        # Approval status features
        features.append(1 if drawing.approval_status == "approved" else 0)
        features.append(1 if drawing.approval_status == "rejected" else 0)
        features.append(1 if drawing.approval_status == "conditional" else 0)
        
        return features
    
    def _extract_code_labels(self, drawing: AsBuiltDrawing) -> Dict[str, Any]:
        """Extract labels for code compliance."""
        labels = {
            "has_violations": any(change.get("code_violation") for change in drawing.as_built_changes),
            "has_design_errors": any("error" in note.lower() for note in drawing.review_notes),
            "has_constructability_issues": "difficult" in drawing.construction_notes.lower() or "impossible" in drawing.construction_notes.lower(),
            "approval_status": drawing.approval_status
        }
        return labels
    
    def _train_code_classifier(self, X: List[List[float]], y: List[Dict[str, Any]]):
        """Train the code classification model."""
        # Prepare data for violation detection
        X_array = np.array(X)
        y_violations = np.array([label["has_violations"] for label in y])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_array, y_violations, test_size=0.2, random_state=42)
        
        # Train model
        self.code_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.code_classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.code_classifier.score(X_train, y_train)
        test_score = self.code_classifier.score(X_test, y_test)
        
        print(f"   Code violation detection - Train: {train_score:.3f}, Test: {test_score:.3f}")
    
    def _train_violation_detector(self, X: List[List[float]], y: List[Dict[str, Any]]):
        """Train the violation detection model."""
        # Prepare data for design error detection
        X_array = np.array(X)
        y_errors = np.array([label["has_design_errors"] for label in y])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_array, y_errors, test_size=0.2, random_state=42)
        
        # Train model
        self.violation_detector = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.violation_detector.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.violation_detector.score(X_train, y_train)
        test_score = self.violation_detector.score(X_test, y_test)
        
        print(f"   Design error detection - Train: {train_score:.3f}, Test: {test_score:.3f}")
    
    def review_drawing(self, drawing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Review a drawing using trained AI engineer."""
        if self.code_classifier is None:
            print("❌ Models not trained yet. Train the models first.")
            return {"status": "not_trained"}
        
        # Extract features
        features = self._extract_code_features_from_dict(drawing_info)
        X = np.array([features])
        
        # Make predictions
        has_violations = self.code_classifier.predict(X)[0]
        has_errors = self.violation_detector.predict(X)[0]
        
        # Generate review recommendations
        recommendations = self._generate_review_recommendations(drawing_info, has_violations, has_errors)
        
        return {
            "has_code_violations": bool(has_violations),
            "has_design_errors": bool(has_errors),
            "recommendations": recommendations,
            "confidence": max(self.code_classifier.predict_proba(X)[0].max(), 
                           self.violation_detector.predict_proba(X)[0].max())
        }
    
    def _extract_code_features_from_dict(self, drawing_info: Dict[str, Any]) -> List[float]:
        """Extract features from drawing info dictionary."""
        # Simplified feature extraction for new drawings
        features = []
        
        # Basic features
        text = f"{drawing_info.get('sheet_title', '')} {drawing_info.get('notes', '')}".lower()
        
        # Code reference counts
        for code_type, codes in self.code_knowledge_base.items():
            for code_name in codes.keys():
                features.append(text.count(code_name.lower()))
                features.append(0)  # Placeholder for violation count
        
        # Drawing type features
        features.extend([0, 0, 0])  # plan, detail, section
        
        # Discipline features
        features.extend([0, 0, 0, 0, 0])  # traffic, electrical, structural, drainage, mechanical
        
        # Approval status features
        features.extend([0, 0, 0])  # approved, rejected, conditional
        
        return features
    
    def _generate_review_recommendations(self, drawing_info: Dict[str, Any], has_violations: bool, has_errors: bool) -> List[str]:
        """Generate review recommendations based on predictions."""
        recommendations = []
        
        if has_violations:
            recommendations.append("Review for code compliance violations")
            recommendations.append("Check against applicable building codes and standards")
        
        if has_errors:
            recommendations.append("Review for design errors and inconsistencies")
            recommendations.append("Verify calculations and design assumptions")
        
        # Add discipline-specific recommendations
        discipline = drawing_info.get("discipline", "").lower()
        if "traffic" in discipline:
            recommendations.append("Verify MUTCD compliance for traffic control devices")
            recommendations.append("Check signal timing and coordination")
        elif "electrical" in discipline:
            recommendations.append("Verify NEC compliance for electrical installations")
            recommendations.append("Check conduit sizing and grounding")
        elif "structural" in discipline:
            recommendations.append("Verify AASHTO compliance for structural elements")
            recommendations.append("Check load ratings and reinforcement details")
        
        return recommendations
    
    def generate_sample_as_built_data(self, num_examples: int = 100):
        """Generate sample as-built data for testing."""
        print(f"📊 Generating {num_examples} sample as-built drawings...")
        
        # Sample as-built data
        sample_drawings = [
            AsBuiltDrawing(
                drawing_id="AB-001",
                project_name="SR-167 Interchange",
                sheet_number="T-001",
                sheet_title="Traffic Signal Plan - As Built",
                discipline="traffic",
                original_design={"signal_heads": 4, "detector_loops": 8},
                as_built_changes=[
                    {
                        "description": "Signal head relocated due to utility conflict",
                        "code_violation": True,
                        "indicators": ["utility conflict", "relocation required"],
                        "severity": "major"
                    }
                ],
                code_references=["MUTCD 2A.01", "WSDOT Traffic Standards"],
                review_notes=["Signal head visibility meets requirements", "Detector coverage adequate"],
                approval_status="approved",
                reviewer_feedback={"visibility": "good", "coverage": "adequate"},
                construction_notes="Installation completed per approved plans",
                file_path="sample_as_built_001.pdf"
            ),
            AsBuiltDrawing(
                drawing_id="AB-002",
                project_name="I-5 Bridge Rehabilitation",
                sheet_number="E-001",
                sheet_title="Electrical Plan - As Built",
                discipline="electrical",
                original_design={"conduit_size": "2 inch", "junction_boxes": 6},
                as_built_changes=[
                    {
                        "description": "Conduit size increased to 3 inch for future capacity",
                        "code_violation": False,
                        "indicators": ["capacity upgrade", "future planning"],
                        "severity": "minor"
                    }
                ],
                code_references=["NEC Article 300", "NEC Article 400"],
                review_notes=["Conduit sizing meets NEC requirements", "Grounding system properly installed"],
                approval_status="approved",
                reviewer_feedback={"conduit": "adequate", "grounding": "proper"},
                construction_notes="Electrical system installed per NEC requirements",
                file_path="sample_as_built_002.pdf"
            )
        ]
        
        # Generate variations
        for i in range(num_examples):
            base_drawing = sample_drawings[i % len(sample_drawings)]
            
            # Create variation
            variation = AsBuiltDrawing(
                drawing_id=f"AB-{i+1:03d}",
                project_name=f"{base_drawing.project_name} - Variation {i+1}",
                sheet_number=f"{base_drawing.sheet_number}-{i+1:02d}",
                sheet_title=f"{base_drawing.sheet_title} - Variation {i+1}",
                discipline=base_drawing.discipline,
                original_design=base_drawing.original_design.copy(),
                as_built_changes=base_drawing.as_built_changes.copy(),
                code_references=base_drawing.code_references.copy(),
                review_notes=base_drawing.review_notes.copy(),
                approval_status=base_drawing.approval_status,
                reviewer_feedback=base_drawing.reviewer_feedback.copy(),
                construction_notes=f"{base_drawing.construction_notes} (variation {i+1})",
                file_path=f"sample_as_built_{i+1:03d}.pdf"
            )
            
            self.add_as_built_drawing(variation)
        
        print(f"✅ Generated {num_examples} sample as-built drawings")
    
    def _save_as_built_drawing(self, drawing: AsBuiltDrawing):
        """Save as-built drawing to disk."""
        drawing_file = self.data_dir / f"as_built_{drawing.drawing_id}.json"
        
        with open(drawing_file, 'w') as f:
            json.dump({
                "drawing_id": drawing.drawing_id,
                "project_name": drawing.project_name,
                "sheet_number": drawing.sheet_number,
                "sheet_title": drawing.sheet_title,
                "discipline": drawing.discipline,
                "original_design": drawing.original_design,
                "as_built_changes": drawing.as_built_changes,
                "code_references": drawing.code_references,
                "review_notes": drawing.review_notes,
                "approval_status": drawing.approval_status,
                "reviewer_feedback": drawing.reviewer_feedback,
                "construction_notes": drawing.construction_notes,
                "file_path": drawing.file_path,
                "confidence": drawing.confidence,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    def _load_training_data(self):
        """Load existing training data."""
        drawing_files = list(self.data_dir.glob("as_built_*.json"))
        
        for drawing_file in drawing_files:
            try:
                with open(drawing_file, 'r') as f:
                    data = json.load(f)
                
                drawing = AsBuiltDrawing(
                    drawing_id=data["drawing_id"],
                    project_name=data["project_name"],
                    sheet_number=data["sheet_number"],
                    sheet_title=data["sheet_title"],
                    discipline=data["discipline"],
                    original_design=data["original_design"],
                    as_built_changes=data["as_built_changes"],
                    code_references=data["code_references"],
                    review_notes=data["review_notes"],
                    approval_status=data["approval_status"],
                    reviewer_feedback=data["reviewer_feedback"],
                    construction_notes=data["construction_notes"],
                    file_path=data["file_path"],
                    confidence=data["confidence"]
                )
                
                self.as_built_drawings.append(drawing)
                self._extract_learning_patterns(drawing)
            except Exception as e:
                print(f"Warning: Could not load {drawing_file}: {e}")
        
        print(f"✅ Loaded {len(self.as_built_drawings)} as-built drawings")
        print(f"✅ Extracted {len(self.review_patterns)} review patterns")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about the training data."""
        if not self.as_built_drawings:
            return {"total_drawings": 0}
        
        disciplines = [d.discipline for d in self.as_built_drawings]
        discipline_counts = {}
        for discipline in disciplines:
            discipline_counts[discipline] = discipline_counts.get(discipline, 0) + 1
        
        approval_statuses = [d.approval_status for d in self.as_built_drawings]
        status_counts = {}
        for status in approval_statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        violation_count = sum(1 for d in self.as_built_drawings 
                            for change in d.as_built_changes if change.get("code_violation"))
        
        return {
            "total_drawings": len(self.as_built_drawings),
            "discipline_distribution": discipline_counts,
            "approval_status_distribution": status_counts,
            "total_violations": violation_count,
            "review_patterns": len(self.review_patterns),
            "models_trained": self.code_classifier is not None
        }

def main():
    """Main function to demonstrate AI engineer training."""
    print("🤖 AI Engineer Trainer - Code Understanding & Review System")
    print("=" * 70)
    
    # Initialize trainer
    trainer = AIEngineerTrainer()
    
    # Generate sample data
    trainer.generate_sample_as_built_data(num_examples=100)
    
    # Train models
    trainer.train_code_understanding_model(min_examples=50)
    
    # Test review capability
    test_drawing = {
        "sheet_number": "T-002",
        "sheet_title": "Traffic Signal Plan",
        "discipline": "traffic",
        "notes": "Signal heads and detector loops shown. Pedestrian signals included."
    }
    
    review_result = trainer.review_drawing(test_drawing)
    print(f"\n🧪 Test Review:")
    print(f"   Drawing: {test_drawing['sheet_number']} - {test_drawing['sheet_title']}")
    print(f"   Has Code Violations: {review_result['has_code_violations']}")
    print(f"   Has Design Errors: {review_result['has_design_errors']}")
    print(f"   Confidence: {review_result['confidence']:.3f}")
    print(f"   Recommendations:")
    for rec in review_result['recommendations']:
        print(f"     - {rec}")
    
    # Show statistics
    stats = trainer.get_training_statistics()
    print(f"\n📊 Training Statistics:")
    print(f"   Total as-built drawings: {stats['total_drawings']}")
    print(f"   Review patterns learned: {stats['review_patterns']}")
    print(f"   Total code violations: {stats['total_violations']}")
    print(f"   Models trained: {stats['models_trained']}")
    
    if stats['discipline_distribution']:
        print(f"   Discipline distribution:")
        for discipline, count in stats['discipline_distribution'].items():
            print(f"     {discipline}: {count}")

if __name__ == "__main__":
    main()
