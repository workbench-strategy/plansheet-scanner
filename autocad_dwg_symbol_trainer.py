#!/usr/bin/env python3
"""
AutoCAD .dwg Symbol Trainer - Mini Model Training System
Specialized trainer for extracting and learning symbols directly from AutoCAD .dwg files.
Creates mini models that can recognize engineering symbols in AutoCAD drawings.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from dataclasses import dataclass
import logging
from collections import defaultdict
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import pickle

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
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
except ImportError as e:
    print(f"Warning: ML dependencies not installed. Install with: pip install scikit-learn joblib torch torchvision")
    print(f"Missing: {e}")

# AutoCAD integration
try:
    import win32com.client
    AUTOCAD_AVAILABLE = True
except ImportError:
    print("Warning: win32com not available. Install with: pip install pywin32")
    AUTOCAD_AVAILABLE = False

# Alternative DWG libraries
try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    print("Warning: ezdxf not available. Install with: pip install ezdxf")
    EZDXF_AVAILABLE = False

@dataclass
class AutoCADSymbol:
    """Represents an AutoCAD symbol with geometric and metadata information."""
    symbol_id: str
    symbol_name: str
    symbol_type: str  # traffic, electrical, structural, drainage, etc.
    entity_type: str  # AcDbCircle, AcDbPolyline, AcDbText, etc.
    geometry: Dict[str, Any]  # Coordinates, dimensions, properties
    layer_name: str
    color: str
    linetype: str
    lineweight: float
    file_path: str
    confidence: float = 1.0
    usage_frequency: int = 1
    bounding_box: Optional[Tuple[float, float, float, float]] = None

@dataclass
class MiniModelConfig:
    """Configuration for mini model training."""
    model_type: str = "random_forest"  # random_forest, gradient_boost, neural_network
    feature_extraction: str = "geometric"  # geometric, visual, hybrid
    max_depth: int = 10
    n_estimators: int = 100
    learning_rate: float = 0.1
    batch_size: int = 32
    epochs: int = 50
    validation_split: float = 0.2

class AutoCADSymbolExtractor:
    """Extracts symbols from AutoCAD .dwg files using multiple methods."""
    
    def __init__(self):
        self.extraction_methods = []
        
        if AUTOCAD_AVAILABLE:
            self.extraction_methods.append("win32com")
        
        if EZDXF_AVAILABLE:
            self.extraction_methods.append("ezdxf")
        
        if not self.extraction_methods:
            raise RuntimeError("No AutoCAD extraction methods available. Install pywin32 or ezdxf.")
    
    def extract_symbols_from_dwg(self, dwg_path: str) -> List[AutoCADSymbol]:
        """Extract symbols from a .dwg file using available methods."""
        symbols = []
        
        for method in self.extraction_methods:
            try:
                if method == "win32com":
                    symbols.extend(self._extract_with_win32com(dwg_path))
                elif method == "ezdxf":
                    symbols.extend(self._extract_with_ezdxf(dwg_path))
            except Exception as e:
                print(f"Warning: Failed to extract with {method}: {e}")
        
        return symbols
    
    def _extract_with_win32com(self, dwg_path: str) -> List[AutoCADSymbol]:
        """Extract symbols using AutoCAD COM interface."""
        symbols = []
        
        try:
            acad = win32com.client.Dispatch("AutoCAD.Application")
            doc = acad.Documents.Open(dwg_path)
            
            entity_count = 0
            for entity in doc.ModelSpace:
                entity_count += 1
                
                try:
                    symbol = self._process_autocad_entity(entity, dwg_path)
                    if symbol:
                        symbols.append(symbol)
                except Exception as e:
                    print(f"Warning: Failed to process entity {entity_count}: {e}")
            
            doc.Close()
            print(f"Extracted {len(symbols)} symbols from {dwg_path} using win32com")
            
        except Exception as e:
            print(f"Error extracting with win32com: {e}")
        
        return symbols
    
    def _extract_with_ezdxf(self, dwg_path: str) -> List[AutoCADSymbol]:
        """Extract symbols using ezdxf library."""
        symbols = []
        
        try:
            doc = ezdxf.readfile(dwg_path)
            msp = doc.modelspace()
            
            entity_count = 0
            for entity in msp:
                entity_count += 1
                
                try:
                    symbol = self._process_ezdxf_entity(entity, dwg_path)
                    if symbol:
                        symbols.append(symbol)
                except Exception as e:
                    print(f"Warning: Failed to process entity {entity_count}: {e}")
            
            print(f"Extracted {len(symbols)} symbols from {dwg_path} using ezdxf")
            
        except Exception as e:
            print(f"Error extracting with ezdxf: {e}")
        
        return symbols
    
    def _process_autocad_entity(self, entity, dwg_path: str) -> Optional[AutoCADSymbol]:
        """Process an AutoCAD entity and convert to symbol."""
        try:
            entity_type = entity.ObjectName
            
            if entity_type == "AcDbCircle":
                return self._process_circle_entity(entity, dwg_path)
            elif entity_type == "AcDbPolyline":
                return self._process_polyline_entity(entity, dwg_path)
            elif entity_type == "AcDbText":
                return self._process_text_entity(entity, dwg_path)
            elif entity_type == "AcDbLine":
                return self._process_line_entity(entity, dwg_path)
            elif entity_type == "AcDbArc":
                return self._process_arc_entity(entity, dwg_path)
            elif entity_type == "AcDbBlockReference":
                return self._process_block_entity(entity, dwg_path)
            
        except Exception as e:
            print(f"Error processing entity: {e}")
        
        return None
    
    def _process_ezdxf_entity(self, entity, dwg_path: str) -> Optional[AutoCADSymbol]:
        """Process an ezdxf entity and convert to symbol."""
        try:
            entity_type = entity.dxftype()
            
            if entity_type == "CIRCLE":
                return self._process_ezdxf_circle(entity, dwg_path)
            elif entity_type == "LWPOLYLINE":
                return self._process_ezdxf_polyline(entity, dwg_path)
            elif entity_type == "TEXT":
                return self._process_ezdxf_text(entity, dwg_path)
            elif entity_type == "LINE":
                return self._process_ezdxf_line(entity, dwg_path)
            elif entity_type == "ARC":
                return self._process_ezdxf_arc(entity, dwg_path)
            elif entity_type == "INSERT":
                return self._process_ezdxf_insert(entity, dwg_path)
            
        except Exception as e:
            print(f"Error processing ezdxf entity: {e}")
        
        return None
    
    def _process_circle_entity(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process circle entity (signal heads, manholes, etc.)."""
        center = [entity.Center[0], entity.Center[1]]
        radius = entity.Radius
        
        # Classify based on size and context
        if radius < 2.0:
            symbol_type = "signal_head"
            symbol_name = "traffic_signal"
        elif radius < 5.0:
            symbol_type = "manhole"
            symbol_name = "utility_manhole"
        else:
            symbol_type = "general"
            symbol_name = "circle_symbol"
        
        return AutoCADSymbol(
            symbol_id=f"circle_{len(str(center))}",
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            entity_type="AcDbCircle",
            geometry={
                "center": center,
                "radius": radius,
                "area": np.pi * radius ** 2
            },
            layer_name=getattr(entity, 'Layer', '0'),
            color=getattr(entity, 'Color', 'BYLAYER'),
            linetype=getattr(entity, 'Linetype', 'CONTINUOUS'),
            lineweight=getattr(entity, 'Lineweight', -1),
            file_path=dwg_path,
            bounding_box=(center[0] - radius, center[1] - radius, 
                         center[0] + radius, center[1] + radius)
        )
    
    def _process_polyline_entity(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process polyline entity (detectors, conduits, etc.)."""
        try:
            coords = entity.Coordinates
            vertices = entity.NumberOfVertices
            
            # Calculate bounding box
            x_coords = [coords[i] for i in range(0, len(coords), 2)]
            y_coords = [coords[i+1] for i in range(0, len(coords), 2)]
            
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
            # Classify based on shape and size
            if vertices == 4 and self._is_rectangular(coords):
                symbol_type = "detector"
                symbol_name = "traffic_detector"
            elif vertices > 4:
                symbol_type = "conduit"
                symbol_name = "electrical_conduit"
            else:
                symbol_type = "general"
                symbol_name = "polyline_symbol"
            
            return AutoCADSymbol(
                symbol_id=f"polyline_{len(str(coords))}",
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                entity_type="AcDbPolyline",
                geometry={
                    "coordinates": coords,
                    "vertices": vertices,
                    "length": self._calculate_polyline_length(coords)
                },
                layer_name=getattr(entity, 'Layer', '0'),
                color=getattr(entity, 'Color', 'BYLAYER'),
                linetype=getattr(entity, 'Linetype', 'CONTINUOUS'),
                lineweight=getattr(entity, 'Lineweight', -1),
                file_path=dwg_path,
                bounding_box=bbox
            )
        except Exception as e:
            print(f"Error processing polyline: {e}")
            return None
    
    def _process_text_entity(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process text entity (labels, notes, etc.)."""
        try:
            text = entity.TextString
            position = [entity.InsertionPoint[0], entity.InsertionPoint[1]]
            
            # Classify based on text content
            if any(keyword in text.lower() for keyword in ['signal', 'detector', 'loop']):
                symbol_type = "traffic"
                symbol_name = "traffic_label"
            elif any(keyword in text.lower() for keyword in ['conduit', 'cable', 'electrical']):
                symbol_type = "electrical"
                symbol_name = "electrical_label"
            else:
                symbol_type = "general"
                symbol_name = "text_label"
            
            return AutoCADSymbol(
                symbol_id=f"text_{hash(text)}",
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                entity_type="AcDbText",
                geometry={
                    "text": text,
                    "position": position,
                    "height": getattr(entity, 'Height', 0.18)
                },
                layer_name=getattr(entity, 'Layer', '0'),
                color=getattr(entity, 'Color', 'BYLAYER'),
                linetype=getattr(entity, 'Linetype', 'CONTINUOUS'),
                lineweight=getattr(entity, 'Lineweight', -1),
                file_path=dwg_path
            )
        except Exception as e:
            print(f"Error processing text: {e}")
            return None
    
    def _process_line_entity(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process line entity (wires, conduits, etc.)."""
        try:
            start_point = [entity.StartPoint[0], entity.StartPoint[1]]
            end_point = [entity.EndPoint[0], entity.EndPoint[1]]
            
            length = np.sqrt((end_point[0] - start_point[0])**2 + 
                           (end_point[1] - start_point[1])**2)
            
            # Classify based on length and layer
            layer = getattr(entity, 'Layer', '0')
            if 'conduit' in layer.lower() or 'cable' in layer.lower():
                symbol_type = "electrical"
                symbol_name = "electrical_line"
            elif length > 50:
                symbol_type = "general"
                symbol_name = "long_line"
            else:
                symbol_type = "general"
                symbol_name = "line_symbol"
            
            return AutoCADSymbol(
                symbol_id=f"line_{hash(str(start_point) + str(end_point))}",
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                entity_type="AcDbLine",
                geometry={
                    "start_point": start_point,
                    "end_point": end_point,
                    "length": length
                },
                layer_name=layer,
                color=getattr(entity, 'Color', 'BYLAYER'),
                linetype=getattr(entity, 'Linetype', 'CONTINUOUS'),
                lineweight=getattr(entity, 'Lineweight', -1),
                file_path=dwg_path
            )
        except Exception as e:
            print(f"Error processing line: {e}")
            return None
    
    def _process_arc_entity(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process arc entity (curved elements)."""
        try:
            center = [entity.Center[0], entity.Center[1]]
            radius = entity.Radius
            start_angle = entity.StartAngle
            end_angle = entity.EndAngle
            
            return AutoCADSymbol(
                symbol_id=f"arc_{hash(str(center) + str(radius))}",
                symbol_name="arc_symbol",
                symbol_type="general",
                entity_type="AcDbArc",
                geometry={
                    "center": center,
                    "radius": radius,
                    "start_angle": start_angle,
                    "end_angle": end_angle
                },
                layer_name=getattr(entity, 'Layer', '0'),
                color=getattr(entity, 'Color', 'BYLAYER'),
                linetype=getattr(entity, 'Linetype', 'CONTINUOUS'),
                lineweight=getattr(entity, 'Lineweight', -1),
                file_path=dwg_path
            )
        except Exception as e:
            print(f"Error processing arc: {e}")
            return None
    
    def _process_block_entity(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process block reference entity (complex symbols)."""
        try:
            name = entity.Name
            position = [entity.InsertionPoint[0], entity.InsertionPoint[1]]
            
            # Classify based on block name
            if any(keyword in name.lower() for keyword in ['signal', 'detector']):
                symbol_type = "traffic"
                symbol_name = "traffic_block"
            elif any(keyword in name.lower() for keyword in ['conduit', 'electrical']):
                symbol_type = "electrical"
                symbol_name = "electrical_block"
            else:
                symbol_type = "general"
                symbol_name = "block_symbol"
            
            return AutoCADSymbol(
                symbol_id=f"block_{hash(name)}",
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                entity_type="AcDbBlockReference",
                geometry={
                    "name": name,
                    "position": position,
                    "scale": getattr(entity, 'XScaleFactor', 1.0)
                },
                layer_name=getattr(entity, 'Layer', '0'),
                color=getattr(entity, 'Color', 'BYLAYER'),
                linetype=getattr(entity, 'Linetype', 'CONTINUOUS'),
                lineweight=getattr(entity, 'Lineweight', -1),
                file_path=dwg_path
            )
        except Exception as e:
            print(f"Error processing block: {e}")
            return None
    
    # Helper methods for ezdxf processing (similar structure)
    def _process_ezdxf_circle(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process ezdxf circle entity."""
        center = entity.dxf.center[:2]  # Only x, y coordinates
        radius = entity.dxf.radius
        
        return AutoCADSymbol(
            symbol_id=f"ezdxf_circle_{hash(str(center))}",
            symbol_name="circle_symbol",
            symbol_type="general",
            entity_type="CIRCLE",
            geometry={
                "center": center,
                "radius": radius,
                "area": np.pi * radius ** 2
            },
            layer_name=entity.dxf.layer,
            color="BYLAYER",
            linetype="CONTINUOUS",
            lineweight=-1,
            file_path=dwg_path
        )
    
    def _process_ezdxf_polyline(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process ezdxf polyline entity."""
        points = list(entity.get_points())
        vertices = len(points)
        
        return AutoCADSymbol(
            symbol_id=f"ezdxf_polyline_{hash(str(points))}",
            symbol_name="polyline_symbol",
            symbol_type="general",
            entity_type="LWPOLYLINE",
            geometry={
                "points": points,
                "vertices": vertices
            },
            layer_name=entity.dxf.layer,
            color="BYLAYER",
            linetype="CONTINUOUS",
            lineweight=-1,
            file_path=dwg_path
        )
    
    def _process_ezdxf_text(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process ezdxf text entity."""
        text = entity.dxf.text
        position = entity.dxf.insert[:2]
        
        return AutoCADSymbol(
            symbol_id=f"ezdxf_text_{hash(text)}",
            symbol_name="text_label",
            symbol_type="general",
            entity_type="TEXT",
            geometry={
                "text": text,
                "position": position
            },
            layer_name=entity.dxf.layer,
            color="BYLAYER",
            linetype="CONTINUOUS",
            lineweight=-1,
            file_path=dwg_path
        )
    
    def _process_ezdxf_line(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process ezdxf line entity."""
        start = entity.dxf.start[:2]
        end = entity.dxf.end[:2]
        
        return AutoCADSymbol(
            symbol_id=f"ezdxf_line_{hash(str(start) + str(end))}",
            symbol_name="line_symbol",
            symbol_type="general",
            entity_type="LINE",
            geometry={
                "start": start,
                "end": end
            },
            layer_name=entity.dxf.layer,
            color="BYLAYER",
            linetype="CONTINUOUS",
            lineweight=-1,
            file_path=dwg_path
        )
    
    def _process_ezdxf_arc(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process ezdxf arc entity."""
        center = entity.dxf.center[:2]
        radius = entity.dxf.radius
        
        return AutoCADSymbol(
            symbol_id=f"ezdxf_arc_{hash(str(center))}",
            symbol_name="arc_symbol",
            symbol_type="general",
            entity_type="ARC",
            geometry={
                "center": center,
                "radius": radius
            },
            layer_name=entity.dxf.layer,
            color="BYLAYER",
            linetype="CONTINUOUS",
            lineweight=-1,
            file_path=dwg_path
        )
    
    def _process_ezdxf_insert(self, entity, dwg_path: str) -> AutoCADSymbol:
        """Process ezdxf insert entity."""
        name = entity.dxf.name
        position = entity.dxf.insert[:2]
        
        return AutoCADSymbol(
            symbol_id=f"ezdxf_insert_{hash(name)}",
            symbol_name="block_symbol",
            symbol_type="general",
            entity_type="INSERT",
            geometry={
                "name": name,
                "position": position
            },
            layer_name=entity.dxf.layer,
            color="BYLAYER",
            linetype="CONTINUOUS",
            lineweight=-1,
            file_path=dwg_path
        )
    
    # Utility methods
    def _is_rectangular(self, coords: List[float]) -> bool:
        """Check if polyline coordinates form a rectangle."""
        if len(coords) != 8:  # 4 points * 2 coordinates
            return False
        
        x_coords = [coords[i] for i in range(0, len(coords), 2)]
        y_coords = [coords[i+1] for i in range(0, len(coords), 2)]
        
        # Check if it's roughly rectangular
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        return x_range > 0 and y_range > 0
    
    def _calculate_polyline_length(self, coords: List[float]) -> float:
        """Calculate the total length of a polyline."""
        if len(coords) < 4:
            return 0.0
        
        total_length = 0.0
        for i in range(0, len(coords) - 2, 2):
            x1, y1 = coords[i], coords[i+1]
            x2, y2 = coords[i+2], coords[i+3]
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length += segment_length
        
        return total_length

class AutoCADSymbolTrainer:
    """Main trainer for AutoCAD .dwg symbol recognition."""
    
    def __init__(self, data_dir: str = "autocad_training_data", model_dir: str = "autocad_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.symbol_extractor = AutoCADSymbolExtractor()
        
        # Data storage
        self.autocad_symbols = []
        self.training_features = []
        self.training_labels = []
        
        # Models
        self.symbol_classifier = None
        self.feature_scaler = None
        self.label_encoder = None
        
        # Configuration
        self.config = MiniModelConfig()
        
        # Load existing data
        self._load_training_data()
    
    def add_dwg_file(self, dwg_path: str) -> int:
        """Add a .dwg file and extract symbols for training."""
        print(f"📁 Processing AutoCAD file: {dwg_path}")
        
        try:
            symbols = self.symbol_extractor.extract_symbols_from_dwg(dwg_path)
            
            for symbol in symbols:
                self.autocad_symbols.append(symbol)
            
            print(f"✅ Extracted {len(symbols)} symbols from {dwg_path}")
            print(f"   Total symbols now: {len(self.autocad_symbols)}")
            
            return len(symbols)
            
        except Exception as e:
            print(f"❌ Error processing {dwg_path}: {e}")
            return 0
    
    def add_dwg_directory(self, directory_path: str) -> int:
        """Add all .dwg files from a directory."""
        directory = Path(directory_path)
        dwg_files = list(directory.glob("*.dwg"))
        
        if not dwg_files:
            print(f"⚠️  No .dwg files found in {directory_path}")
            return 0
        
        total_symbols = 0
        for dwg_file in dwg_files:
            symbols_added = self.add_dwg_file(str(dwg_file))
            total_symbols += symbols_added
        
        print(f"✅ Processed {len(dwg_files)} .dwg files, extracted {total_symbols} symbols total")
        return total_symbols
    
    def extract_features(self, symbol: AutoCADSymbol) -> List[float]:
        """Extract features from an AutoCAD symbol for training."""
        features = []
        
        # Geometric features
        geometry = symbol.geometry
        
        if symbol.entity_type == "AcDbCircle":
            features.extend([
                geometry.get("radius", 0),
                geometry.get("area", 0),
                len(str(geometry.get("center", [0, 0])))
            ])
        
        elif symbol.entity_type == "AcDbPolyline":
            features.extend([
                geometry.get("vertices", 0),
                geometry.get("length", 0),
                len(str(geometry.get("coordinates", [])))
            ])
        
        elif symbol.entity_type == "AcDbText":
            text = geometry.get("text", "")
            features.extend([
                len(text),
                geometry.get("height", 0),
                len(str(geometry.get("position", [0, 0])))
            ])
        
        elif symbol.entity_type == "AcDbLine":
            features.extend([
                geometry.get("length", 0),
                len(str(geometry.get("start_point", [0, 0]))),
                len(str(geometry.get("end_point", [0, 0])))
            ])
        
        # Layer features
        layer_features = self._extract_layer_features(symbol.layer_name)
        features.extend(layer_features)
        
        # Color features
        color_features = self._extract_color_features(symbol.color)
        features.extend(color_features)
        
        # Lineweight features
        features.append(symbol.lineweight)
        
        # Bounding box features (if available)
        if symbol.bounding_box:
            bbox = symbol.bounding_box
            features.extend([
                bbox[2] - bbox[0],  # width
                bbox[3] - bbox[1],  # height
                (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # area
            ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _extract_layer_features(self, layer_name: str) -> List[float]:
        """Extract features from layer name."""
        layer_lower = layer_name.lower()
        
        features = [
            1.0 if 'traffic' in layer_lower else 0.0,
            1.0 if 'electrical' in layer_lower else 0.0,
            1.0 if 'conduit' in layer_lower else 0.0,
            1.0 if 'signal' in layer_lower else 0.0,
            1.0 if 'detector' in layer_lower else 0.0,
            1.0 if 'manhole' in layer_lower else 0.0,
            len(layer_name),
            layer_name.count('_'),
            layer_name.count('-')
        ]
        
        return features
    
    def _extract_color_features(self, color: str) -> List[float]:
        """Extract features from color information."""
        color_lower = color.lower()
        
        features = [
            1.0 if 'red' in color_lower else 0.0,
            1.0 if 'green' in color_lower else 0.0,
            1.0 if 'blue' in color_lower else 0.0,
            1.0 if 'yellow' in color_lower else 0.0,
            1.0 if 'white' in color_lower else 0.0,
            1.0 if 'bylayer' in color_lower else 0.0,
            len(color)
        ]
        
        return features
    
    def train_mini_model(self, config: Optional[MiniModelConfig] = None) -> Dict[str, Any]:
        """Train a mini model on the extracted AutoCAD symbols."""
        if len(self.autocad_symbols) < 10:
            print(f"⚠️  Need at least 10 symbols for training, have {len(self.autocad_symbols)}")
            return {"success": False, "error": "Insufficient training data"}
        
        if config:
            self.config = config
        
        print(f"🤖 Training mini model on {len(self.autocad_symbols)} AutoCAD symbols...")
        print(f"   Model type: {self.config.model_type}")
        print(f"   Feature extraction: {self.config.feature_extraction}")
        
        # Extract features and labels
        X = []
        y = []
        
        for symbol in self.autocad_symbols:
            features = self.extract_features(symbol)
            X.append(features)
            y.append(symbol.symbol_type)
        
        X = np.array(X)
        y = np.array(y)
        
        # Check data diversity
        unique_labels = set(y)
        print(f"   Symbol types: {list(unique_labels)}")
        print(f"   Feature dimensions: {X.shape[1]}")
        
        if len(unique_labels) < 2:
            print("⚠️  Need at least 2 different symbol types for classification")
            return {"success": False, "error": "Insufficient label diversity"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model based on configuration
        if self.config.model_type == "random_forest":
            self.symbol_classifier = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42
            )
        elif self.config.model_type == "gradient_boost":
            self.symbol_classifier = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                random_state=42
            )
        elif self.config.model_type == "neural_network":
            self.symbol_classifier = self._create_neural_network(X_train_scaled.shape[1], len(unique_labels))
        
        # Train the model
        if self.config.model_type in ["random_forest", "gradient_boost"]:
            self.symbol_classifier.fit(X_train_scaled, y_train_encoded)
            
            # Evaluate
            train_score = self.symbol_classifier.score(X_train_scaled, y_train_encoded)
            test_score = self.symbol_classifier.score(X_test_scaled, y_test_encoded)
            
            # Feature importance (for tree-based models)
            feature_importance = None
            if hasattr(self.symbol_classifier, 'feature_importances_'):
                feature_importance = self.symbol_classifier.feature_importances_.tolist()
        
        elif self.config.model_type == "neural_network":
            # Neural network training
            train_score, test_score = self._train_neural_network(
                X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded
            )
            feature_importance = None
        
        # Generate report
        y_pred = self.symbol_classifier.predict(X_test_scaled)
        classification_rep = classification_report(
            y_test_encoded, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Save model
        model_path = self.model_dir / f"autocad_symbol_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.save_model(str(model_path))
        
        results = {
            "success": True,
            "model_path": str(model_path),
            "train_score": train_score,
            "test_score": test_score,
            "classification_report": classification_rep,
            "feature_importance": feature_importance,
            "symbol_types": list(unique_labels),
            "total_symbols": len(self.autocad_symbols),
            "feature_dimensions": X.shape[1]
        }
        
        print(f"✅ Mini model training completed!")
        print(f"   Train accuracy: {train_score:.3f}")
        print(f"   Test accuracy: {test_score:.3f}")
        print(f"   Model saved to: {model_path}")
        
        return results
    
    def _create_neural_network(self, input_dim: int, num_classes: int) -> nn.Module:
        """Create a simple neural network for symbol classification."""
        class SymbolNet(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(SymbolNet, self).__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return SymbolNet(input_dim, num_classes)
    
    def _train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train neural network model."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.symbol_classifier.parameters(), lr=0.001)
        
        # Training loop
        self.symbol_classifier.train()
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            outputs = self.symbol_classifier(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        self.symbol_classifier.eval()
        with torch.no_grad():
            train_outputs = self.symbol_classifier(X_train_tensor)
            test_outputs = self.symbol_classifier(X_test_tensor)
            
            train_pred = torch.argmax(train_outputs, dim=1)
            test_pred = torch.argmax(test_outputs, dim=1)
            
            train_score = (train_pred == y_train_tensor).float().mean().item()
            test_score = (test_pred == y_test_tensor).float().mean().item()
        
        return train_score, test_score
    
    def predict_symbol_type(self, dwg_path: str) -> List[Dict[str, Any]]:
        """Predict symbol types in a new .dwg file."""
        if self.symbol_classifier is None:
            print("⚠️  No trained model available. Train a model first.")
            return []
        
        print(f"🔍 Predicting symbols in: {dwg_path}")
        
        # Extract symbols
        symbols = self.symbol_extractor.extract_symbols_from_dwg(dwg_path)
        
        if not symbols:
            print("No symbols found in the file.")
            return []
        
        # Extract features
        X = []
        for symbol in symbols:
            features = self.extract_features(symbol)
            X.append(features)
        
        X = np.array(X)
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Predict
        predictions = self.symbol_classifier.predict(X_scaled)
        probabilities = self.symbol_classifier.predict_proba(X_scaled)
        
        # Format results
        results = []
        for i, symbol in enumerate(symbols):
            predicted_type = self.label_encoder.inverse_transform([predictions[i]])[0]
            confidence = np.max(probabilities[i])
            
            results.append({
                "symbol_id": symbol.symbol_id,
                "symbol_name": symbol.symbol_name,
                "entity_type": symbol.entity_type,
                "predicted_type": predicted_type,
                "confidence": confidence,
                "geometry": symbol.geometry,
                "layer_name": symbol.layer_name
            })
        
        print(f"✅ Predicted {len(results)} symbols")
        return results
    
    def save_model(self, model_path: str):
        """Save the trained model and components."""
        model_data = {
            "symbol_classifier": self.symbol_classifier,
            "feature_scaler": self.feature_scaler,
            "label_encoder": self.label_encoder,
            "config": self.config,
            "training_stats": {
                "total_symbols": len(self.autocad_symbols),
                "symbol_types": list(set(s.symbol_type for s in self.autocad_symbols)),
                "trained_at": datetime.now().isoformat()
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.symbol_classifier = model_data["symbol_classifier"]
            self.feature_scaler = model_data["feature_scaler"]
            self.label_encoder = model_data["label_encoder"]
            self.config = model_data["config"]
            
            print(f"📂 Model loaded from: {model_path}")
            print(f"   Training stats: {model_data['training_stats']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def _load_training_data(self):
        """Load existing training data from disk."""
        data_file = self.data_dir / "autocad_symbols.json"
        
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct symbols from JSON
                for symbol_data in data:
                    symbol = AutoCADSymbol(**symbol_data)
                    self.autocad_symbols.append(symbol)
                
                print(f"📂 Loaded {len(self.autocad_symbols)} existing symbols from {data_file}")
                
            except Exception as e:
                print(f"⚠️  Error loading training data: {e}")
    
    def save_training_data(self):
        """Save training data to disk."""
        data_file = self.data_dir / "autocad_symbols.json"
        
        # Convert symbols to JSON-serializable format
        symbol_data = []
        for symbol in self.autocad_symbols:
            symbol_dict = {
                "symbol_id": symbol.symbol_id,
                "symbol_name": symbol.symbol_name,
                "symbol_type": symbol.symbol_type,
                "entity_type": symbol.entity_type,
                "geometry": symbol.geometry,
                "layer_name": symbol.layer_name,
                "color": symbol.color,
                "linetype": symbol.linetype,
                "lineweight": symbol.lineweight,
                "file_path": symbol.file_path,
                "confidence": symbol.confidence,
                "usage_frequency": symbol.usage_frequency,
                "bounding_box": symbol.bounding_box
            }
            symbol_data.append(symbol_dict)
        
        with open(data_file, 'w') as f:
            json.dump(symbol_data, f, indent=2)
        
        print(f"💾 Saved {len(self.autocad_symbols)} symbols to {data_file}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about the training data."""
        if not self.autocad_symbols:
            return {"total_symbols": 0, "symbol_types": [], "entity_types": []}
        
        symbol_types = defaultdict(int)
        entity_types = defaultdict(int)
        layer_names = defaultdict(int)
        
        for symbol in self.autocad_symbols:
            symbol_types[symbol.symbol_type] += 1
            entity_types[symbol.entity_type] += 1
            layer_names[symbol.layer_name] += 1
        
        return {
            "total_symbols": len(self.autocad_symbols),
            "symbol_types": dict(symbol_types),
            "entity_types": dict(entity_types),
            "layer_names": dict(layer_names),
            "files_processed": len(set(s.file_path for s in self.autocad_symbols))
        }

def main():
    """Main function to demonstrate AutoCAD symbol training."""
    print("🚀 AutoCAD .dwg Symbol Trainer")
    print("=" * 50)
    
    # Initialize trainer
    trainer = AutoCADSymbolTrainer()
    
    # Check available extraction methods
    print(f"📋 Available extraction methods: {trainer.symbol_extractor.extraction_methods}")
    
    # Example usage
    print("\n📁 Example usage:")
    print("1. Add .dwg files:")
    print("   trainer.add_dwg_file('path/to/drawing.dwg')")
    print("   trainer.add_dwg_directory('path/to/dwg/folder')")
    print("\n2. Train mini model:")
    print("   config = MiniModelConfig(model_type='random_forest')")
    print("   results = trainer.train_mini_model(config)")
    print("\n3. Predict symbols:")
    print("   predictions = trainer.predict_symbol_type('new_drawing.dwg')")
    
    # Show current statistics
    stats = trainer.get_training_statistics()
    print(f"\n📊 Current training data:")
    print(f"   Total symbols: {stats['total_symbols']}")
    print(f"   Symbol types: {list(stats['symbol_types'].keys())}")
    print(f"   Entity types: {list(stats['entity_types'].keys())}")
    
    return trainer

if __name__ == "__main__":
    trainer = main()
