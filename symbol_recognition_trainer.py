#!/usr/bin/env python3
"""
Symbol Recognition Trainer - Foundation Training
Learns to identify symbols on engineering plans from notes and legends.
This is the foundational step before training on standards and requirements.
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
import cv2
from PIL import Image, ImageDraw, ImageFont

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
class EngineeringSymbol:
    """Represents an engineering symbol with all relevant information."""
    symbol_id: str
    symbol_name: str
    symbol_type: str  # traffic, electrical, structural, drainage, etc.
    visual_description: str
    legend_reference: str
    notes_description: str
    common_variations: List[str]
    context_clues: List[str]
    file_path: str
    confidence: float = 1.0
    usage_frequency: int = 1

@dataclass
class PlanLegend:
    """Represents a plan legend with symbol definitions."""
    legend_id: str
    plan_type: str
    discipline: str
    symbols: List[Dict[str, Any]]
    notes_section: str
    abbreviations: Dict[str, str]
    file_path: str

class SymbolRecognitionTrainer:
    """Specialized trainer for learning engineering symbols from plans and legends."""
    
    def __init__(self, data_dir: str = "symbol_training_data", model_dir: str = "symbol_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.engineering_symbols = []
        self.plan_legends = []
        self.symbol_patterns = []
        
        # Models
        self.symbol_classifier = None
        self.legend_parser = None
        self.notes_extractor = None
        self.feature_scaler = None
        self.label_encoder = None
        
        # Symbol knowledge base - organized by discipline
        self.symbol_knowledge_base = {
            "traffic": {
                "signal_symbols": {
                    "traffic_signal": {
                        "visual": "circle with arrows or lights",
                        "legend_ref": "TS",
                        "notes_keywords": ["signal", "traffic light", "stop light"],
                        "variations": ["TS", "TRAFFIC SIGNAL", "SIGNAL"]
                    },
                    "pedestrian_signal": {
                        "visual": "walking figure or pedestrian symbol",
                        "legend_ref": "PS",
                        "notes_keywords": ["pedestrian", "walk", "crosswalk"],
                        "variations": ["PS", "PED", "WALK"]
                    }
                },
                "sign_symbols": {
                    "stop_sign": {
                        "visual": "octagon or STOP text",
                        "legend_ref": "STOP",
                        "notes_keywords": ["stop", "halt", "intersection"],
                        "variations": ["STOP", "STOP SIGN", "HALT"]
                    },
                    "yield_sign": {
                        "visual": "triangle or YIELD text",
                        "legend_ref": "YIELD",
                        "notes_keywords": ["yield", "give way"],
                        "variations": ["YIELD", "GIVE WAY"]
                    }
                }
            },
            "electrical": {
                "equipment_symbols": {
                    "junction_box": {
                        "visual": "square or rectangle with X or JB",
                        "legend_ref": "JB",
                        "notes_keywords": ["junction box", "JB", "electrical box"],
                        "variations": ["JB", "JUNCTION BOX", "ELECTRICAL BOX"]
                    },
                    "conduit": {
                        "visual": "line or pipe symbol",
                        "legend_ref": "COND",
                        "notes_keywords": ["conduit", "pipe", "electrical"],
                        "variations": ["COND", "CONDUIT", "PIPE"]
                    }
                }
            },
            "structural": {
                "element_symbols": {
                    "beam": {
                        "visual": "rectangular shape or beam symbol",
                        "legend_ref": "BEAM",
                        "notes_keywords": ["beam", "girder", "structural"],
                        "variations": ["BEAM", "GIRDER", "STRUCTURAL"]
                    },
                    "column": {
                        "visual": "vertical rectangle or column symbol",
                        "legend_ref": "COL",
                        "notes_keywords": ["column", "post", "support"],
                        "variations": ["COL", "COLUMN", "POST"]
                    }
                }
            },
            "drainage": {
                "system_symbols": {
                    "catch_basin": {
                        "visual": "circle or square with CB",
                        "legend_ref": "CB",
                        "notes_keywords": ["catch basin", "CB", "inlet"],
                        "variations": ["CB", "CATCH BASIN", "INLET"]
                    },
                    "manhole": {
                        "visual": "circle with MH",
                        "legend_ref": "MH",
                        "notes_keywords": ["manhole", "MH", "access"],
                        "variations": ["MH", "MANHOLE", "ACCESS"]
                    }
                }
            }
        }
        
        # Load existing data
        self._load_training_data()
    
    def generate_symbol_training_data(self, num_examples: int = 300):
        """Generate diverse training data for symbol recognition."""
        print(f"📊 Generating {num_examples} symbol training examples...")
        
        # Plan types and disciplines
        plan_types = ["traffic_plan", "electrical_plan", "structural_plan", "drainage_plan", "site_plan"]
        disciplines = ["traffic", "electrical", "structural", "drainage", "mechanical"]
        
        # Generate symbols for each discipline
        for i in range(num_examples):
            discipline = random.choice(disciplines)
            plan_type = random.choice(plan_types)
            
            # Get discipline-specific symbols
            discipline_symbols = self.symbol_knowledge_base.get(discipline, {})
            
            # Flatten all symbol categories
            all_symbols = []
            for category, symbols in discipline_symbols.items():
                for symbol_name, symbol_info in symbols.items():
                    all_symbols.append((symbol_name, symbol_info))
            
            if not all_symbols:
                continue
            
            # Select a random symbol
            symbol_name, symbol_info = random.choice(all_symbols)
            
            # Create variations in description
            visual_desc = self._create_visual_variation(symbol_info["visual"])
            legend_ref = self._create_legend_variation(symbol_info["legend_ref"])
            notes_desc = self._create_notes_variation(symbol_info["notes_keywords"])
            
            # Create context clues
            context_clues = self._generate_context_clues(discipline, symbol_name)
            
            # Create symbol
            symbol = EngineeringSymbol(
                symbol_id=f"SYM-{discipline.upper()}-{i+1:03d}",
                symbol_name=symbol_name,
                symbol_type=discipline,
                visual_description=visual_desc,
                legend_reference=legend_ref,
                notes_description=notes_desc,
                common_variations=symbol_info["variations"],
                context_clues=context_clues,
                file_path=f"plan_{plan_type}_{i+1:03d}.pdf",
                confidence=random.uniform(0.8, 1.0),
                usage_frequency=random.randint(1, 10)
            )
            
            self.add_engineering_symbol(symbol)
        
        print(f"✅ Generated {num_examples} symbol training examples")
    
    def _create_visual_variation(self, base_description: str) -> str:
        """Create variations in visual descriptions."""
        variations = [
            f"{base_description} shown on plan",
            f"{base_description} indicated by symbol",
            f"{base_description} represented as",
            f"{base_description} marked with",
            f"{base_description} displayed as"
        ]
        return random.choice(variations)
    
    def _create_legend_variation(self, base_ref: str) -> str:
        """Create variations in legend references."""
        variations = [
            base_ref,
            f"See legend: {base_ref}",
            f"Legend reference: {base_ref}",
            f"Symbol: {base_ref}",
            f"Ref: {base_ref}"
        ]
        return random.choice(variations)
    
    def _create_notes_variation(self, keywords: List[str]) -> str:
        """Create variations in notes descriptions."""
        base_keyword = random.choice(keywords)
        variations = [
            f"{base_keyword} installed per plan",
            f"{base_keyword} as shown",
            f"{base_keyword} to be provided",
            f"{base_keyword} required",
            f"{base_keyword} specified"
        ]
        return random.choice(variations)
    
    def _generate_context_clues(self, discipline: str, symbol_name: str) -> List[str]:
        """Generate context clues for symbol identification."""
        context_templates = {
            "traffic": [
                "intersection", "roadway", "crosswalk", "signal timing",
                "traffic control", "pedestrian", "vehicle"
            ],
            "electrical": [
                "power", "electrical", "conduit", "wiring", "junction",
                "panel", "circuit", "voltage"
            ],
            "structural": [
                "beam", "column", "foundation", "load", "support",
                "reinforcement", "concrete", "steel"
            ],
            "drainage": [
                "storm", "drainage", "pipe", "inlet", "outlet",
                "water", "flow", "gradient"
            ]
        }
        
        clues = context_templates.get(discipline, [])
        return random.sample(clues, min(3, len(clues)))
    
    def add_engineering_symbol(self, symbol: EngineeringSymbol):
        """Add an engineering symbol for training."""
        self.engineering_symbols.append(symbol)
        self._extract_symbol_patterns(symbol)
        self._save_engineering_symbol(symbol)
    
    def _extract_symbol_patterns(self, symbol: EngineeringSymbol):
        """Extract learning patterns from engineering symbol."""
        # Extract visual pattern
        pattern = {
            "type": "visual_pattern",
            "discipline": symbol.symbol_type,
            "symbol_name": symbol.symbol_name,
            "visual_description": symbol.visual_description,
            "confidence": symbol.confidence
        }
        self.symbol_patterns.append(pattern)
        
        # Extract legend pattern
        pattern = {
            "type": "legend_pattern",
            "discipline": symbol.symbol_type,
            "symbol_name": symbol.symbol_name,
            "legend_reference": symbol.legend_reference,
            "confidence": symbol.confidence
        }
        self.symbol_patterns.append(pattern)
        
        # Extract notes pattern
        pattern = {
            "type": "notes_pattern",
            "discipline": symbol.symbol_type,
            "symbol_name": symbol.symbol_name,
            "notes_description": symbol.notes_description,
            "confidence": symbol.confidence
        }
        self.symbol_patterns.append(pattern)
    
    def train_symbol_models(self, min_examples: int = 50):
        """Train models for symbol recognition."""
        if len(self.engineering_symbols) < min_examples:
            print(f"Need at least {min_examples} examples, have {len(self.engineering_symbols)}")
            # Generate more synthetic data to meet minimum requirements
            additional_needed = min_examples - len(self.engineering_symbols)
            print(f"Generating {additional_needed} additional synthetic examples...")
            self.generate_symbol_training_data(num_examples=additional_needed)
            return
        
        print(f"🤖 Training symbol recognition models on {len(self.engineering_symbols)} examples...")
        
        # Prepare training data
        X = []
        y_symbols = []
        y_disciplines = []
        y_confidence = []
        
        for symbol in self.engineering_symbols:
            features = self._extract_symbol_features(symbol)
            X.append(features)
            
            # Labels
            y_symbols.append(symbol.symbol_name)
            y_disciplines.append(symbol.symbol_type)
            y_confidence.append(symbol.confidence)
        
        X = np.array(X)
        y_symbols = np.array(y_symbols)
        y_disciplines = np.array(y_disciplines)
        y_confidence = np.array(y_confidence)
        
        # Check data diversity
        print(f"   Data distribution:")
        print(f"     Total symbols: {len(y_symbols)}")
        print(f"     Unique symbols: {len(set(y_symbols))}")
        print(f"     Disciplines: {list(set(y_disciplines))}")
        
        # Train symbol classifier
        if len(set(y_symbols)) > 1:
            self._train_symbol_classifier(X, y_symbols)
        
        # Train discipline classifier
        if len(set(y_disciplines)) > 1:
            self._train_discipline_classifier(X, y_disciplines)
        
        # Train confidence predictor
        self._train_confidence_predictor(X, y_confidence)
        
        print("✅ All symbol recognition models trained successfully!")
    
    def _extract_symbol_features(self, symbol: EngineeringSymbol) -> List[float]:
        """Extract comprehensive features from symbol."""
        features = []
        
        # Text analysis features
        all_text = f"{symbol.visual_description} {symbol.legend_reference} {symbol.notes_description}"
        all_text = all_text.lower()
        
        # Basic text features
        features.append(len(all_text))  # Text length
        features.append(len(all_text.split()))  # Word count
        features.append(len(set(all_text.split())))  # Unique words
        
        # Symbol type features
        for discipline in ["traffic", "electrical", "structural", "drainage", "mechanical"]:
            features.append(1 if discipline in symbol.symbol_type.lower() else 0)
        
        # Context clue features
        features.append(len(symbol.context_clues))
        features.append(len(symbol.common_variations))
        
        # Confidence and usage features
        features.append(symbol.confidence)
        features.append(symbol.usage_frequency)
        
        # Pattern matching features
        features.append(1 if "signal" in all_text else 0)
        features.append(1 if "box" in all_text else 0)
        features.append(1 if "pipe" in all_text else 0)
        features.append(1 if "beam" in all_text else 0)
        features.append(1 if "basin" in all_text else 0)
        
        return features
    
    def _train_symbol_classifier(self, X: np.ndarray, y: np.ndarray):
        """Train symbol classification model."""
        print("   Training symbol classifier...")
        
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
        self.symbol_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.symbol_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.symbol_classifier.score(X_train_scaled, y_train)
        test_score = self.symbol_classifier.score(X_test_scaled, y_test)
        
        print(f"     Symbol classification - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Save encoder and scaler
        self.label_encoder = label_encoder
        self.feature_scaler = scaler
    
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
    
    def _train_confidence_predictor(self, X: np.ndarray, y: np.ndarray):
        """Train confidence prediction model."""
        print("   Training confidence predictor...")
        
        # Convert confidence values to discrete classes for classification
        # Create confidence bins: low (0-0.3), medium (0.3-0.7), high (0.7-1.0)
        y_discrete = np.digitize(y, bins=[0.3, 0.7])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_discrete, test_size=0.2, random_state=42, stratify=y_discrete)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        self.confidence_predictor = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.confidence_predictor.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.confidence_predictor.score(X_train_scaled, y_train)
        test_score = self.confidence_predictor.score(X_test_scaled, y_test)
        
        print(f"     Confidence prediction - Train: {train_score:.3f}, Test: {test_score:.3f}")
    
    def identify_symbol(self, symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """Identify a symbol using trained models."""
        if self.symbol_classifier is None:
            return {"status": "models_not_trained"}
        
        # Extract features
        features = self._extract_features_from_dict(symbol_info)
        X = np.array([features])
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Make predictions
        symbol_pred = self.symbol_classifier.predict(X_scaled)[0]
        predicted_symbol = self.label_encoder.inverse_transform([symbol_pred])[0]
        symbol_confidence = self.symbol_classifier.predict_proba(X_scaled)[0].max()
        
        predicted_discipline = "unknown"
        discipline_confidence = 0.0
        if hasattr(self, 'discipline_classifier'):
            discipline_pred = self.discipline_classifier.predict(X_scaled)[0]
            predicted_discipline = self.label_encoder.inverse_transform([discipline_pred])[0]
            discipline_confidence = self.discipline_classifier.predict_proba(X_scaled)[0].max()
        
        # Generate identification details
        identification_details = self._generate_identification_details(
            symbol_info, predicted_symbol, predicted_discipline
        )
        
        return {
            "predicted_symbol": predicted_symbol,
            "predicted_discipline": predicted_discipline,
            "symbol_confidence": symbol_confidence,
            "discipline_confidence": discipline_confidence,
            "identification_details": identification_details,
            "overall_confidence": max(symbol_confidence, discipline_confidence)
        }
    
    def _extract_features_from_dict(self, symbol_info: Dict[str, Any]) -> List[float]:
        """Extract features from symbol info dictionary."""
        features = []
        
        # Basic text features
        text = f"{symbol_info.get('visual_description', '')} {symbol_info.get('legend_ref', '')} {symbol_info.get('notes', '')}".lower()
        features.extend([len(text), len(text.split()), len(set(text.split()))])
        
        # Discipline features
        for discipline in ["traffic", "electrical", "structural", "drainage", "mechanical"]:
            features.append(1 if discipline in text else 0)
        
        # Context features
        features.extend([0, 0])  # context_clues, common_variations
        
        # Confidence and usage features
        features.extend([0.5, 1])  # confidence, usage_frequency
        
        # Pattern matching features
        features.append(1 if "signal" in text else 0)
        features.append(1 if "box" in text else 0)
        features.append(1 if "pipe" in text else 0)
        features.append(1 if "beam" in text else 0)
        features.append(1 if "basin" in text else 0)
        
        return features
    
    def _generate_identification_details(self, symbol_info: Dict[str, Any], predicted_symbol: str, predicted_discipline: str) -> Dict[str, Any]:
        """Generate detailed identification information."""
        details = {
            "visual_analysis": f"Symbol appears to be {predicted_symbol} based on visual description",
            "legend_reference": f"Check legend for {predicted_symbol} reference",
            "notes_interpretation": f"Notes suggest {predicted_symbol} installation",
            "discipline_context": f"Symbol belongs to {predicted_discipline} discipline",
            "recommendations": []
        }
        
        # Add discipline-specific recommendations
        if predicted_discipline == "traffic":
            details["recommendations"].extend([
                "Verify MUTCD compliance for traffic symbols",
                "Check signal timing coordination",
                "Ensure pedestrian accessibility"
            ])
        elif predicted_discipline == "electrical":
            details["recommendations"].extend([
                "Verify NEC compliance for electrical symbols",
                "Check conduit sizing and routing",
                "Ensure proper grounding"
            ])
        elif predicted_discipline == "structural":
            details["recommendations"].extend([
                "Verify load calculations for structural elements",
                "Check reinforcement details",
                "Ensure proper connections"
            ])
        elif predicted_discipline == "drainage":
            details["recommendations"].extend([
                "Verify drainage capacity calculations",
                "Check pipe slopes and sizing",
                "Ensure proper inlet/outlet locations"
            ])
        
        return details
    
    def _save_engineering_symbol(self, symbol: EngineeringSymbol):
        """Save engineering symbol to disk."""
        symbol_file = self.data_dir / f"symbol_{symbol.symbol_id}.json"
        
        with open(symbol_file, 'w') as f:
            json.dump({
                "symbol_id": symbol.symbol_id,
                "symbol_name": symbol.symbol_name,
                "symbol_type": symbol.symbol_type,
                "visual_description": symbol.visual_description,
                "legend_reference": symbol.legend_reference,
                "notes_description": symbol.notes_description,
                "common_variations": symbol.common_variations,
                "context_clues": symbol.context_clues,
                "file_path": symbol.file_path,
                "confidence": symbol.confidence,
                "usage_frequency": symbol.usage_frequency,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    def _load_training_data(self):
        """Load existing training data."""
        symbol_files = list(self.data_dir.glob("symbol_*.json"))
        
        for symbol_file in symbol_files:
            try:
                with open(symbol_file, 'r') as f:
                    data = json.load(f)
                
                symbol = EngineeringSymbol(
                    symbol_id=data["symbol_id"],
                    symbol_name=data["symbol_name"],
                    symbol_type=data["symbol_type"],
                    visual_description=data["visual_description"],
                    legend_reference=data["legend_reference"],
                    notes_description=data["notes_description"],
                    common_variations=data["common_variations"],
                    context_clues=data["context_clues"],
                    file_path=data["file_path"],
                    confidence=data["confidence"],
                    usage_frequency=data["usage_frequency"]
                )
                
                self.engineering_symbols.append(symbol)
                self._extract_symbol_patterns(symbol)
            except Exception as e:
                print(f"Warning: Could not load {symbol_file}: {e}")
        
        print(f"✅ Loaded {len(self.engineering_symbols)} engineering symbols")
        print(f"✅ Extracted {len(self.symbol_patterns)} symbol patterns")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.engineering_symbols:
            return {"total_symbols": 0}
        
        disciplines = [s.symbol_type for s in self.engineering_symbols]
        discipline_counts = {}
        for discipline in disciplines:
            discipline_counts[discipline] = discipline_counts.get(discipline, 0) + 1
        
        symbol_names = [s.symbol_name for s in self.engineering_symbols]
        symbol_counts = {}
        for symbol_name in symbol_names:
            symbol_counts[symbol_name] = symbol_counts.get(symbol_name, 0) + 1
        
        avg_confidence = sum(s.confidence for s in self.engineering_symbols) / len(self.engineering_symbols)
        avg_usage = sum(s.usage_frequency for s in self.engineering_symbols) / len(self.engineering_symbols)
        
        return {
            "total_symbols": len(self.engineering_symbols),
            "discipline_distribution": discipline_counts,
            "symbol_distribution": symbol_counts,
            "symbol_patterns": len(self.symbol_patterns),
            "avg_confidence": avg_confidence,
            "avg_usage_frequency": avg_usage,
            "models_trained": self.symbol_classifier is not None
        }
    
    def add_symbols_from_as_built(self, symbols: List[Dict[str, Any]], filename: str):
        """Add symbols extracted from as-built drawings to training data."""
        print(f"📄 Adding {len(symbols)} symbols from as-built: {filename}")
        
        for symbol_data in symbols:
            try:
                # Create EngineeringSymbol object
                symbol = EngineeringSymbol(
                    symbol_id=symbol_data.get("symbol_id", f"as_built_{len(self.engineering_symbols)}"),
                    symbol_name=symbol_data.get("symbol_name", "unknown"),
                    symbol_type=symbol_data.get("symbol_type", "general"),
                    visual_description=symbol_data.get("visual_description", ""),
                    legend_reference=symbol_data.get("legend_reference", ""),
                    notes_description=symbol_data.get("notes_description", ""),
                    common_variations=symbol_data.get("common_variations", []),
                    context_clues=symbol_data.get("context_clues", []),
                    file_path=filename,
                    confidence=symbol_data.get("confidence", 0.8),
                    usage_frequency=symbol_data.get("usage_frequency", 1)
                )
                
                # Add to training data
                self.engineering_symbols.append(symbol)
                
                # Extract patterns
                self._extract_symbol_patterns(symbol)
                
            except Exception as e:
                print(f"⚠️  Error adding symbol from as-built: {e}")
        
        print(f"✅ Added {len(symbols)} symbols from {filename}")
        print(f"   Total symbols now: {len(self.engineering_symbols)}")

def main():
    """Main function to demonstrate symbol recognition training."""
    print("🤖 Symbol Recognition Trainer - Foundation Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = SymbolRecognitionTrainer()
    
    # Generate symbol training data
    trainer.generate_symbol_training_data(num_examples=300)
    
    # Train models
    trainer.train_symbol_models(min_examples=50)
    
    # Test symbol identification
    test_symbols = [
        {
            "visual_description": "circle with traffic light shown on plan",
            "legend_ref": "TS",
            "notes": "traffic signal installed per plan"
        },
        {
            "visual_description": "square with JB indicated by symbol",
            "legend_ref": "JB",
            "notes": "junction box to be provided"
        },
        {
            "visual_description": "rectangular beam represented as",
            "legend_ref": "BEAM",
            "notes": "structural beam required"
        }
    ]
    
    print(f"\n🧪 Test Symbol Identification:")
    for i, test_symbol in enumerate(test_symbols, 1):
        identification_result = trainer.identify_symbol(test_symbol)
        print(f"\n   Test {i}: {test_symbol['visual_description']}")
        print(f"     Predicted Symbol: {identification_result['predicted_symbol']} (confidence: {identification_result['symbol_confidence']:.3f})")
        print(f"     Predicted Discipline: {identification_result['predicted_discipline']} (confidence: {identification_result['discipline_confidence']:.3f})")
        print(f"     Overall Confidence: {identification_result['overall_confidence']:.3f}")
        print(f"     Details:")
        for key, value in identification_result['identification_details'].items():
            if isinstance(value, list):
                print(f"       {key}:")
                for item in value:
                    print(f"         - {item}")
            else:
                print(f"       {key}: {value}")
    
    # Show comprehensive statistics
    stats = trainer.get_training_statistics()
    print(f"\n📊 Training Statistics:")
    print(f"   Total symbols: {stats['total_symbols']}")
    print(f"   Symbol patterns learned: {stats['symbol_patterns']}")
    print(f"   Average confidence: {stats['avg_confidence']:.3f}")
    print(f"   Average usage frequency: {stats['avg_usage_frequency']:.1f}")
    print(f"   Models trained: {stats['models_trained']}")
    
    if stats['discipline_distribution']:
        print(f"   Discipline distribution:")
        for discipline, count in stats['discipline_distribution'].items():
            print(f"     {discipline}: {count}")
    
    if stats['symbol_distribution']:
        print(f"   Top symbols:")
        sorted_symbols = sorted(stats['symbol_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
        for symbol, count in sorted_symbols:
            print(f"     {symbol}: {count}")

if __name__ == "__main__":
    main()
