#!/usr/bin/env python3
"""
Improved AI Engineer Trainer - Better Training System
Trains an AI engineer with more diverse and realistic training data.
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
import random

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
except ImportError:
    print("Warning: ML dependencies not installed. Install with: pip install scikit-learn joblib")

@dataclass
class AsBuiltDrawing:
    """Represents an as-built drawing with all relevant information."""
    drawing_id: str
    project_name: str
    sheet_number: str
    sheet_title: str
    discipline: str
    original_design: Dict[str, Any]
    as_built_changes: List[Dict[str, Any]]
    code_references: List[str]
    review_notes: List[str]
    approval_status: str
    reviewer_feedback: Dict[str, Any]
    construction_notes: str
    file_path: str
    confidence: float = 1.0

class ImprovedAIEngineerTrainer:
    """Improved trainer for AI engineer with better data diversity."""
    
    def __init__(self, data_dir: str = "improved_ai_data", model_dir: str = "improved_ai_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.as_built_drawings = []
        self.review_patterns = []
        
        # Models
        self.code_classifier = None
        self.violation_detector = None
        self.discipline_classifier = None
        self.feature_scaler = None
        self.label_encoder = None
        
        # Code knowledge base
        self.code_knowledge_base = {
            "traffic_control": {
                "MUTCD": {
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
                }
            },
            "electrical": {
                "NEC": {
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
                    "requirements": [
                        "Load ratings must meet design standards",
                        "Reinforcement must be properly detailed",
                        "Concrete cover must meet requirements",
                        "Expansion joints must be properly sized"
                    ]
                }
            }
        }
        
        # Load existing data
        self._load_training_data()
        
        # Load existing models if available
        self._load_models()
    
    def generate_diverse_training_data(self, num_examples: int = 200):
        """Generate diverse training data with realistic variations."""
        print(f"📊 Generating {num_examples} diverse training examples...")
        
        # Base project templates
        project_templates = [
            {
                "name": "SR-167 Interchange",
                "type": "intersection",
                "agency": "WSDOT",
                "complexity": "high"
            },
            {
                "name": "I-5 Bridge Rehabilitation",
                "type": "bridge",
                "agency": "WSDOT",
                "complexity": "medium"
            },
            {
                "name": "Local Street Improvement",
                "type": "roadway",
                "agency": "City",
                "complexity": "low"
            },
            {
                "name": "Highway Widening",
                "type": "highway",
                "agency": "WSDOT",
                "complexity": "high"
            }
        ]
        
        # Discipline templates with realistic variations
        discipline_templates = {
            "traffic": {
                "sheet_patterns": ["T-", "TS-", "TM-", "TP-"],
                "titles": [
                    "Traffic Signal Plan",
                    "Traffic Signing Plan", 
                    "Traffic Marking Plan",
                    "Traffic Control Plan",
                    "Signal Timing Plan"
                ],
                "common_issues": [
                    "Signal head visibility issues",
                    "Missing pedestrian signals",
                    "Inadequate detector coverage",
                    "Sign placement conflicts",
                    "Timing coordination problems"
                ]
            },
            "electrical": {
                "sheet_patterns": ["E-", "EL-", "EP-", "EC-"],
                "titles": [
                    "Electrical Plan",
                    "Electrical Details",
                    "Power Distribution",
                    "Electrical Conduit Plan",
                    "Electrical Equipment Plan"
                ],
                "common_issues": [
                    "Undersized conduit",
                    "Missing grounding",
                    "Inaccessible junction boxes",
                    "Overfilled conduit",
                    "Voltage drop issues"
                ]
            },
            "structural": {
                "sheet_patterns": ["S-", "ST-", "B-", "BR-"],
                "titles": [
                    "Structural Plan",
                    "Bridge Plan",
                    "Structural Details",
                    "Foundation Plan",
                    "Reinforcement Details"
                ],
                "common_issues": [
                    "Insufficient reinforcement",
                    "Inadequate concrete cover",
                    "Load rating issues",
                    "Expansion joint problems",
                    "Foundation settlement"
                ]
            },
            "drainage": {
                "sheet_patterns": ["C-", "CV-", "D-", "G-"],
                "titles": [
                    "Drainage Plan",
                    "Storm Drainage",
                    "Culvert Details",
                    "Grading Plan",
                    "Erosion Control"
                ],
                "common_issues": [
                    "Inadequate drainage capacity",
                    "Poor inlet spacing",
                    "Insufficient pipe slopes",
                    "Erosion control issues",
                    "Flooding problems"
                ]
            }
        }
        
        # Generate diverse examples
        for i in range(num_examples):
            # Randomly select project and discipline
            project = random.choice(project_templates)
            discipline = random.choice(list(discipline_templates.keys()))
            template = discipline_templates[discipline]
            
            # Create realistic variations
            sheet_number = f"{random.choice(template['sheet_patterns'])}{i+1:03d}"
            sheet_title = f"{random.choice(template['titles'])} - As Built"
            
            # Randomly decide if there are violations (more realistic distribution)
            has_violations = random.random() < 0.3  # 30% chance of violations
            has_design_errors = random.random() < 0.2  # 20% chance of design errors
            has_constructability_issues = random.random() < 0.15  # 15% chance of constructability issues
            
            # Create as-built changes based on probability
            as_built_changes = []
            if has_violations:
                issue = random.choice(template["common_issues"])
                as_built_changes.append({
                    "description": f"{issue} - field modification required",
                    "code_violation": True,
                    "indicators": [issue.lower().split()[0], "field modification"],
                    "severity": random.choice(["critical", "major", "minor"])
                })
            
            # Create review notes
            review_notes = []
            if has_design_errors:
                review_notes.append(f"Design error found: {random.choice(template['common_issues'])}")
            else:
                review_notes.append(f"Design review completed - {discipline} elements meet requirements")
            
            # Determine approval status based on issues
            if has_violations and has_violations:
                approval_status = random.choice(["rejected", "conditional"])
            elif has_violations:
                approval_status = random.choice(["conditional", "approved"])
            else:
                approval_status = "approved"
            
            # Create construction notes
            if has_constructability_issues:
                construction_notes = f"Construction difficulty encountered: {random.choice(['access issues', 'coordination problems', 'material availability'])}"
            else:
                construction_notes = f"Construction completed per approved plans - {discipline} installation successful"
            
            # Create drawing
            drawing = AsBuiltDrawing(
                drawing_id=f"AB-{i+1:03d}",
                project_name=f"{project['name']} - Variation {i+1}",
                sheet_number=sheet_number,
                sheet_title=sheet_title,
                discipline=discipline,
                original_design={"elements": random.randint(3, 12), "complexity": project["complexity"]},
                as_built_changes=as_built_changes,
                code_references=[f"{discipline.upper()}_Standards", f"{project['agency']}_Requirements"],
                review_notes=review_notes,
                approval_status=approval_status,
                reviewer_feedback={"quality": random.choice(["excellent", "good", "adequate", "poor"])},
                construction_notes=construction_notes,
                file_path=f"as_built_{i+1:03d}.pdf"
            )
            
            self.add_as_built_drawing(drawing)
        
        print(f"✅ Generated {num_examples} diverse training examples")
    
    def add_as_built_drawing(self, drawing: AsBuiltDrawing):
        """Add an as-built drawing for training."""
        self.as_built_drawings.append(drawing)
        self._extract_learning_patterns(drawing)
        self._save_as_built_drawing(drawing)
    
    def _extract_learning_patterns(self, drawing: AsBuiltDrawing):
        """Extract learning patterns from as-built drawing."""
        # Extract code violation patterns
        for change in drawing.as_built_changes:
            if change.get("code_violation"):
                pattern = {
                    "type": "code_violation",
                    "discipline": drawing.discipline,
                    "description": change.get("description", ""),
                    "severity": change.get("severity", "major"),
                    "indicators": change.get("indicators", [])
                }
                self.review_patterns.append(pattern)
        
        # Extract design error patterns
        for note in drawing.review_notes:
            if "error" in note.lower():
                pattern = {
                    "type": "design_error",
                    "discipline": drawing.discipline,
                    "description": note,
                    "severity": "major",
                    "indicators": ["design error", "calculation error"]
                }
                self.review_patterns.append(pattern)
    
    def train_models(self, min_examples: int = 50):
        """Train all models with improved data handling."""
        if len(self.as_built_drawings) < min_examples:
            print(f"Need at least {min_examples} examples, have {len(self.as_built_drawings)}")
            return
        
        print(f"🤖 Training models on {len(self.as_built_drawings)} diverse examples...")
        
        # Prepare training data
        X = []
        y_violations = []
        y_errors = []
        y_disciplines = []
        
        for drawing in self.as_built_drawings:
            features = self._extract_features(drawing)
            X.append(features)
            
            # Labels for different models
            y_violations.append(any(change.get("code_violation") for change in drawing.as_built_changes))
            y_errors.append(any("error" in note.lower() for note in drawing.review_notes))
            y_disciplines.append(drawing.discipline)
        
        X = np.array(X)
        y_violations = np.array(y_violations)
        y_errors = np.array(y_errors)
        y_disciplines = np.array(y_disciplines)
        
        # Check data diversity
        print(f"   Data distribution:")
        print(f"     Code violations: {sum(y_violations)}/{len(y_violations)} ({sum(y_violations)/len(y_violations)*100:.1f}%)")
        print(f"     Design errors: {sum(y_errors)}/{len(y_errors)} ({sum(y_errors)/len(y_errors)*100:.1f}%)")
        
        # Train violation detection model
        if sum(y_violations) > 0 and sum(y_violations) < len(y_violations):
            self._train_violation_detector(X, y_violations)
        
        # Train error detection model
        if sum(y_errors) > 0 and sum(y_errors) < len(y_errors):
            self._train_error_detector(X, y_errors)
        
        # Train discipline classifier
        if len(set(y_disciplines)) > 1:
            self._train_discipline_classifier(X, y_disciplines)
        
        print("✅ All models trained successfully!")
    
    def _extract_features(self, drawing: AsBuiltDrawing) -> List[float]:
        """Extract comprehensive features from drawing."""
        features = []
        
        # Text analysis
        all_text = f"{drawing.sheet_title} {' '.join(drawing.review_notes)} {drawing.construction_notes}"
        all_text = all_text.lower()
        
        # Text-based features
        features.append(len(all_text))  # Text length
        features.append(len(all_text.split()))  # Word count
        features.append(len(set(all_text.split())))  # Unique words
        
        # Code reference features
        for discipline, codes in self.code_knowledge_base.items():
            for code_name, code_info in codes.items():
                # Count code references
                code_count = sum(1 for ref in drawing.code_references if code_name.lower() in ref.lower())
                features.append(code_count)
                
                # Count violation keywords
                violation_count = sum(1 for violation in code_info.get("common_violations", [])
                                    if violation.lower() in all_text)
                features.append(violation_count)
        
        # Drawing type features
        features.append(1 if "plan" in drawing.sheet_title.lower() else 0)
        features.append(1 if "detail" in drawing.sheet_title.lower() else 0)
        features.append(1 if "section" in drawing.sheet_title.lower() else 0)
        
        # Discipline features
        for discipline in ["traffic", "electrical", "structural", "drainage"]:
            features.append(1 if discipline in drawing.discipline.lower() else 0)
        
        # Approval status features
        features.append(1 if drawing.approval_status == "approved" else 0)
        features.append(1 if drawing.approval_status == "rejected" else 0)
        features.append(1 if drawing.approval_status == "conditional" else 0)
        
        # Issue indicators
        features.append(len(drawing.as_built_changes))
        features.append(len(drawing.review_notes))
        features.append(1 if "difficult" in drawing.construction_notes.lower() else 0)
        features.append(1 if "error" in ' '.join(drawing.review_notes).lower() else 0)
        
        return features
    
    def _train_violation_detector(self, X: np.ndarray, y: np.ndarray):
        """Train code violation detection model."""
        print("   Training code violation detector...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        self.violation_detector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.violation_detector.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.violation_detector.score(X_train_scaled, y_train)
        test_score = self.violation_detector.score(X_test_scaled, y_test)
        
        print(f"     Violation detection - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Save scaler
        self.feature_scaler = scaler
        
        # Save models to disk
        self._save_models()
    
    def _train_error_detector(self, X: np.ndarray, y: np.ndarray):
        """Train design error detection model."""
        print("   Training design error detector...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        self.error_detector = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.error_detector.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.error_detector.score(X_train_scaled, y_train)
        test_score = self.error_detector.score(X_test_scaled, y_test)
        
        print(f"     Error detection - Train: {train_score:.3f}, Test: {test_score:.3f}")
    
    def _train_discipline_classifier(self, X: np.ndarray, y: np.ndarray):
        """Train discipline classification model."""
        print("   Training discipline classifier...")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        self.discipline_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.discipline_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.discipline_classifier.score(X_train_scaled, y_train)
        test_score = self.discipline_classifier.score(X_test_scaled, y_test)
        
        print(f"     Discipline classification - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Save encoder
        self.label_encoder = label_encoder
    
    def review_drawing(self, drawing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Review a drawing using trained models."""
        if self.violation_detector is None:
            return {"status": "models_not_trained"}
        
        # Extract features
        features = self._extract_features_from_dict(drawing_info)
        X = np.array([features])
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Make predictions
        has_violations = self.violation_detector.predict(X_scaled)[0]
        violation_confidence = self.violation_detector.predict_proba(X_scaled)[0].max()
        
        has_errors = False
        error_confidence = 0.0
        if hasattr(self, 'error_detector'):
            has_errors = self.error_detector.predict(X_scaled)[0]
            error_confidence = self.error_detector.predict_proba(X_scaled)[0].max()
        
        predicted_discipline = "unknown"
        discipline_confidence = 0.0
        if hasattr(self, 'discipline_classifier') and self.label_encoder is not None:
            discipline_pred = self.discipline_classifier.predict(X_scaled)[0]
            predicted_discipline = self.label_encoder.inverse_transform([discipline_pred])[0]
            discipline_confidence = self.discipline_classifier.predict_proba(X_scaled)[0].max()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(drawing_info, has_violations, has_errors, predicted_discipline)
        
        return {
            "has_code_violations": bool(has_violations),
            "has_design_errors": bool(has_errors),
            "predicted_discipline": predicted_discipline,
            "violation_confidence": violation_confidence,
            "error_confidence": error_confidence,
            "discipline_confidence": discipline_confidence,
            "recommendations": recommendations,
            "overall_confidence": max(violation_confidence, error_confidence, discipline_confidence)
        }
    
    def _extract_features_from_dict(self, drawing_info: Dict[str, Any]) -> List[float]:
        """Extract features from drawing info dictionary."""
        # Enhanced feature extraction for new drawings
        features = []
        
        # Basic text features
        text = f"{drawing_info.get('sheet_title', '')} {drawing_info.get('construction_notes', '')}".lower()
        features.extend([len(text), len(text.split()), len(set(text.split()))])
        
        # Code reference counts
        for discipline, codes in self.code_knowledge_base.items():
            for code_name in codes.keys():
                features.extend([text.count(code_name.lower()), 0])  # code refs, violations
        
        # Drawing type features
        sheet_title = drawing_info.get('sheet_title', '').lower()
        features.append(1 if "plan" in sheet_title else 0)
        features.append(1 if "detail" in sheet_title else 0)
        features.append(1 if "section" in sheet_title else 0)
        
        # Discipline features based on sheet title and content
        discipline = drawing_info.get('discipline', '').lower()
        features.append(1 if "traffic" in discipline or "signal" in sheet_title else 0)
        features.append(1 if "electrical" in discipline or "electrical" in sheet_title else 0)
        features.append(1 if "structural" in discipline or "structural" in sheet_title else 0)
        features.append(1 if "drainage" in discipline or "drainage" in sheet_title else 0)
        
        # Approval status features
        approval_status = drawing_info.get('approval_status', '').lower()
        features.append(1 if approval_status == "approved" else 0)
        features.append(1 if approval_status == "rejected" else 0)
        features.append(1 if approval_status == "conditional" else 0)
        
        # Issue indicators
        features.append(len(drawing_info.get('as_built_changes', [])))
        features.append(len(drawing_info.get('review_notes', [])))
        features.append(1 if "difficult" in text else 0)
        features.append(1 if "error" in text else 0)
        
        return features
    
    def _generate_recommendations(self, drawing_info: Dict[str, Any], has_violations: bool, has_errors: bool, discipline: str) -> List[str]:
        """Generate review recommendations."""
        recommendations = []
        
        if has_violations:
            recommendations.append("Review for code compliance violations")
            recommendations.append("Check against applicable building codes and standards")
        
        if has_errors:
            recommendations.append("Review for design errors and inconsistencies")
            recommendations.append("Verify calculations and design assumptions")
        
        # Discipline-specific recommendations
        if discipline == "traffic":
            recommendations.append("Verify MUTCD compliance for traffic control devices")
            recommendations.append("Check signal timing and coordination")
        elif discipline == "electrical":
            recommendations.append("Verify NEC compliance for electrical installations")
            recommendations.append("Check conduit sizing and grounding")
        elif discipline == "structural":
            recommendations.append("Verify AASHTO compliance for structural elements")
            recommendations.append("Check load ratings and reinforcement details")
        elif discipline == "drainage":
            recommendations.append("Verify drainage capacity and inlet spacing")
            recommendations.append("Check pipe slopes and erosion control")
        
        return recommendations
    
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
        """Get comprehensive training statistics."""
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
        
        error_count = sum(1 for d in self.as_built_drawings 
                         for note in d.review_notes if "error" in note.lower())
        
        return {
            "total_drawings": len(self.as_built_drawings),
            "discipline_distribution": discipline_counts,
            "approval_status_distribution": status_counts,
            "total_violations": violation_count,
            "total_errors": error_count,
            "review_patterns": len(self.review_patterns),
            "models_trained": self.violation_detector is not None
        }
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            import joblib
            
            # Save violation detector
            if self.violation_detector is not None:
                joblib.dump(self.violation_detector, self.model_dir / "violation_detector.pkl")
            
            # Save error detector
            if hasattr(self, 'error_detector') and self.error_detector is not None:
                joblib.dump(self.error_detector, self.model_dir / "error_detector.pkl")
            
            # Save discipline classifier
            if hasattr(self, 'discipline_classifier') and self.discipline_classifier is not None:
                joblib.dump(self.discipline_classifier, self.model_dir / "discipline_classifier.pkl")
            
            # Save scaler
            if self.feature_scaler is not None:
                joblib.dump(self.feature_scaler, self.model_dir / "feature_scaler.pkl")
            
            # Save label encoder
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                joblib.dump(self.label_encoder, self.model_dir / "label_encoder.pkl")
            
            print(f"✅ Models saved to {self.model_dir}")
            
        except Exception as e:
            print(f"Warning: Could not save models: {e}")
    
    def _load_models(self):
        """Load trained models from disk."""
        try:
            import joblib
            
            # Load violation detector
            violation_model_path = self.model_dir / "violation_detector.pkl"
            if violation_model_path.exists():
                self.violation_detector = joblib.load(violation_model_path)
                print(f"✅ Loaded violation detector from {violation_model_path}")
            
            # Load error detector
            error_model_path = self.model_dir / "error_detector.pkl"
            if error_model_path.exists():
                self.error_detector = joblib.load(error_model_path)
                print(f"✅ Loaded error detector from {error_model_path}")
            
            # Load discipline classifier
            discipline_model_path = self.model_dir / "discipline_classifier.pkl"
            if discipline_model_path.exists():
                self.discipline_classifier = joblib.load(discipline_model_path)
                print(f"✅ Loaded discipline classifier from {discipline_model_path}")
            
            # Load scaler
            scaler_path = self.model_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                print(f"✅ Loaded feature scaler from {scaler_path}")
            
            # Load label encoder
            encoder_path = self.model_dir / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                print(f"✅ Loaded label encoder from {encoder_path}")
            
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
    
    def save_models(self):
        """Public method to save models."""
        self._save_models()
    
    def load_models(self):
        """Public method to load models."""
        self._load_models()

def main():
    """Main function to demonstrate improved AI engineer training."""
    print("🤖 Improved AI Engineer Trainer - Better Training System")
    print("=" * 70)
    
    # Initialize trainer
    trainer = ImprovedAIEngineerTrainer()
    
    # Generate diverse training data
    trainer.generate_diverse_training_data(num_examples=200)
    
    # Train models
    trainer.train_models(min_examples=50)
    
    # Test review capability
    test_drawings = [
        {
            "sheet_number": "T-002",
            "sheet_title": "Traffic Signal Plan",
            "discipline": "traffic",
            "notes": "Signal heads and detector loops shown. Pedestrian signals included."
        },
        {
            "sheet_number": "E-003",
            "sheet_title": "Electrical Plan",
            "discipline": "electrical", 
            "notes": "Power distribution and conduit routing. Grounding system shown."
        },
        {
            "sheet_number": "S-004",
            "sheet_title": "Structural Plan",
            "discipline": "structural",
            "notes": "Bridge structure and reinforcement details. Load ratings calculated."
        }
    ]
    
    print(f"\n🧪 Test Reviews:")
    for i, test_drawing in enumerate(test_drawings, 1):
        review_result = trainer.review_drawing(test_drawing)
        print(f"\n   Test {i}: {test_drawing['sheet_number']} - {test_drawing['sheet_title']}")
        print(f"     Predicted Discipline: {review_result['predicted_discipline']} (confidence: {review_result['discipline_confidence']:.3f})")
        print(f"     Has Code Violations: {review_result['has_code_violations']} (confidence: {review_result['violation_confidence']:.3f})")
        print(f"     Has Design Errors: {review_result['has_design_errors']} (confidence: {review_result['error_confidence']:.3f})")
        print(f"     Overall Confidence: {review_result['overall_confidence']:.3f}")
        print(f"     Recommendations:")
        for rec in review_result['recommendations']:
            print(f"       - {rec}")
    
    # Show comprehensive statistics
    stats = trainer.get_training_statistics()
    print(f"\n📊 Training Statistics:")
    print(f"   Total as-built drawings: {stats['total_drawings']}")
    print(f"   Review patterns learned: {stats['review_patterns']}")
    print(f"   Total code violations: {stats['total_violations']}")
    print(f"   Total design errors: {stats['total_errors']}")
    print(f"   Models trained: {stats['models_trained']}")
    
    if stats['discipline_distribution']:
        print(f"   Discipline distribution:")
        for discipline, count in stats['discipline_distribution'].items():
            print(f"     {discipline}: {count}")
    
    if stats['approval_status_distribution']:
        print(f"   Approval status distribution:")
        for status, count in stats['approval_status_distribution'].items():
            print(f"     {status}: {count}")

if __name__ == "__main__":
    main()
