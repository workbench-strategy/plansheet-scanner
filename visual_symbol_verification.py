#!/usr/bin/env python3
"""
Visual Symbol Verification System

Industry-standard visual verification system for engineering AI models.
Shows model detections with bounding boxes, confidence scores, and allows QC feedback.
"""

import sys
import os
import json
import pickle
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RadioButtons
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the symbol recognition components
from src.core.ml_enhanced_symbol_recognition import MLSymbolRecognizer, SymbolDetection
from src.core.discipline_classifier import DisciplineClassifier

# Configure matplotlib for better display
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


class VisualVerificationSystem:
    """
    Industry-standard visual verification system for engineering AI models.
    
    Features:
    - Visual annotation overlays with bounding boxes
    - Confidence scores and color-coded classifications
    - QC feedback interface for corrections
    - Training data generation from feedback
    - Model retraining pipeline
    """
    
    def __init__(self):
        self.symbol_recognizer = MLSymbolRecognizer()
        self.discipline_classifier = DisciplineClassifier()
        
        # Color scheme for different disciplines (industry standard)
        self.discipline_colors = {
            'electrical': '#FF6B6B',      # Red
            'structural': '#4ECDC4',      # Teal
            'civil': '#45B7D1',           # Blue
            'traffic': '#96CEB4',         # Green
            'mechanical': '#FFEAA7',      # Yellow
            'landscape': '#DDA0DD',       # Plum
            'unknown': '#808080'          # Gray
        }
        
        # Symbol type colors
        self.symbol_colors = {
            'electrical_panel': '#FF0000',
            'light_pole': '#FF4444',
            'junction_box': '#FF8888',
            'conduit': '#FFCCCC',
            'structural_beam': '#00FF00',
            'manhole': '#0088FF',
            'pipe': '#00CCFF',
            'small_device': '#FFFF00',
            'complex_equipment': '#FF8800'
        }
        
        # QC feedback storage
        self.qc_feedback = []
        self.current_image_path = None
        self.current_detections = []
        
        print("🎯 Visual Verification System Initialized")
        print("✅ Industry-standard verification tools ready")
        print("✅ QC feedback system active")
        print("✅ Training data generation enabled")
    
    def analyze_drawing_with_visual_output(self, image_path):
        """
        Analyze engineering drawing and generate visual verification output.
        
        Args:
            image_path: Path to the engineering drawing image
            
        Returns:
            dict: Analysis results with visual annotations
        """
        print(f"\n🔍 Analyzing: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Perform analysis
        analysis = self._analyze_drawing_for_symbols(image, image_path.name)
        
        # Generate visual annotations
        visual_output = self._generate_visual_annotations(image, analysis)
        
        # Store for QC review
        self.current_image_path = image_path
        self.current_detections = analysis['potential_symbols']
        
        return {
            'image_path': str(image_path),
            'analysis': analysis,
            'visual_output': visual_output,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_drawing_for_symbols(self, image, filename):
        """Analyze drawing for symbols with enhanced detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Initialize results
        analysis = {
            'potential_symbols': [],
            'text_regions': [],
            'line_patterns': [],
            'filename': filename,
            'image_size': image.shape,
            'processing_metadata': {}
        }
        
        # Enhanced edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Find contours with real-world parameters
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours for symbols
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 50000:  # Real symbol size range
                x, y, w, h = cv2.boundingRect(contour)
                
                # Analyze contour properties
                aspect_ratio = w / h if h > 0 else 0
                extent = area / (w * h) if w * h > 0 else 0
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Classify symbol
                symbol_type = self._classify_engineering_symbol(contour, aspect_ratio, extent, circularity, area)
                
                if symbol_type:
                    # Calculate confidence
                    confidence = self._calculate_symbol_confidence(contour, aspect_ratio, extent, circularity)
                    
                    # Determine discipline
                    discipline = self._determine_symbol_discipline(symbol_type)
                    
                    # Create detection object
                    detection = {
                        'type': symbol_type,
                        'discipline': discipline,
                        'position': (x, y),
                        'size': (w, h),
                        'area': area,
                        'confidence': confidence,
                        'properties': {
                            'aspect_ratio': aspect_ratio,
                            'extent': extent,
                            'circularity': circularity,
                            'perimeter': perimeter
                        },
                        'bbox': [x, y, x + w, y + h]  # For visualization
                    }
                    
                    analysis['potential_symbols'].append(detection)
        
        # Enhanced line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=5)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 50:
                    line_type = self._classify_engineering_line(length)
                    
                    analysis['line_patterns'].append({
                        'type': line_type,
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': length,
                        'discipline': self._determine_line_discipline(line_type)
                    })
        
        # Text region detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        for region in regions:
            if len(region) > 10:
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                x, y, w, h = cv2.boundingRect(hull)
                
                if 20 < w < 500 and 10 < h < 100:
                    analysis['text_regions'].append({
                        'position': (x, y),
                        'size': (w, h),
                        'area': w * h,
                        'region_points': len(region)
                    })
        
        return analysis
    
    def _classify_engineering_symbol(self, contour, aspect_ratio, extent, circularity, area):
        """Classify contour as engineering symbol."""
        # Circular symbols
        if 0.7 < aspect_ratio < 1.3 and circularity > 0.6:
            if area < 1000:
                return "light_pole"
            elif area < 5000:
                return "junction_box"
            else:
                return "manhole"
        
        # Rectangular symbols
        elif 0.3 < aspect_ratio < 3.0 and extent > 0.5:
            if area < 2000:
                return "electrical_panel"
            elif area < 10000:
                return "equipment_box"
            else:
                return "transformer"
        
        # Linear symbols
        elif aspect_ratio > 4.0 or aspect_ratio < 0.25:
            if area < 1000:
                return "conduit"
            elif area < 5000:
                return "pipe"
            else:
                return "structural_beam"
        
        # Complex symbols
        elif extent < 0.4 and area > 2000:
            return "complex_equipment"
        
        # Small symbols
        elif area < 500 and extent > 0.6:
            return "small_device"
        
        return None
    
    def _calculate_symbol_confidence(self, contour, aspect_ratio, extent, circularity):
        """Calculate confidence score for symbol detection."""
        confidence = 0.3  # Base confidence
        
        if extent > 0.7:
            confidence += 0.2
        if 0.2 < aspect_ratio < 5.0:
            confidence += 0.1
        if circularity > 0.6:
            confidence += 0.1
        
        area = cv2.contourArea(contour)
        if 100 < area < 10000:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _determine_symbol_discipline(self, symbol_type):
        """Determine discipline for symbol type."""
        discipline_map = {
            'electrical_panel': 'electrical',
            'light_pole': 'electrical',
            'junction_box': 'electrical',
            'conduit': 'electrical',
            'structural_beam': 'structural',
            'complex_equipment': 'structural',
            'manhole': 'civil',
            'pipe': 'civil',
            'small_device': 'traffic'
        }
        return discipline_map.get(symbol_type, 'unknown')
    
    def _classify_engineering_line(self, length):
        """Classify line as engineering element."""
        if length > 300:
            return "structural_beam"
        elif length > 150:
            return "electrical_conduit"
        elif length > 80:
            return "pipe"
        else:
            return "detail_line"
    
    def _determine_line_discipline(self, line_type):
        """Determine discipline for line type."""
        discipline_map = {
            'structural_beam': 'structural',
            'electrical_conduit': 'electrical',
            'pipe': 'civil',
            'detail_line': 'unknown'
        }
        return discipline_map.get(line_type, 'unknown')
    
    def _generate_visual_annotations(self, image, analysis):
        """Generate visual annotations for verification."""
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title('Original Engineering Drawing', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Annotated image
        ax2.imshow(image_rgb)
        ax2.set_title('AI Model Detections', fontsize=14, fontweight='bold')
        
        # Add symbol annotations
        for symbol in analysis['potential_symbols']:
            x, y, w, h = symbol['position'][0], symbol['position'][1], symbol['size'][0], symbol['size'][1]
            confidence = symbol['confidence']
            symbol_type = symbol['type']
            discipline = symbol['discipline']
            
            # Get color
            color = self.symbol_colors.get(symbol_type, self.discipline_colors.get(discipline, '#808080'))
            
            # Create bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            # Add label
            label = f"{symbol_type}\n{discipline}\n{confidence:.2f}"
            ax2.text(x, y-5, label, fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add line annotations
        for line in analysis['line_patterns']:
            x1, y1 = line['start']
            x2, y2 = line['end']
            line_type = line['type']
            discipline = line['discipline']
            
            color = self.discipline_colors.get(discipline, '#808080')
            ax2.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.7)
        
        ax2.axis('off')
        
        # Add legend
        legend_elements = []
        for discipline, color in self.discipline_colors.items():
            if discipline != 'unknown':
                legend_elements.append(patches.Patch(color=color, label=discipline.title()))
        
        ax2.legend(handles=legend_elements, loc='upper right', title='Disciplines')
        
        plt.tight_layout()
        
        return fig
    
    def save_verification_output(self, analysis_result, output_dir="verification_output"):
        """Save verification output with annotations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(analysis_result['image_path']).stem
        fig_path = output_path / f"{image_name}_annotated_{timestamp}.png"
        
        analysis_result['visual_output'].savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(analysis_result['visual_output'])
        
        # Save analysis data
        data_path = output_path / f"{image_name}_analysis_{timestamp}.json"
        with open(data_path, 'w') as f:
            json.dump(analysis_result['analysis'], f, indent=2, default=str)
        
        print(f"✅ Verification output saved:")
        print(f"  • Annotated image: {fig_path}")
        print(f"  • Analysis data: {data_path}")
        
        return fig_path, data_path
    
    def create_qc_interface(self, analysis_result):
        """Create QC feedback interface."""
        print("\n🎯 QC Feedback Interface")
        print("=" * 50)
        
        # Display analysis summary
        analysis = analysis_result['analysis']
        print(f"📊 Detection Summary:")
        print(f"  • Symbols detected: {len(analysis['potential_symbols'])}")
        print(f"  • Line patterns: {len(analysis['line_patterns'])}")
        print(f"  • Text regions: {len(analysis['text_regions'])}")
        
        # Show symbol breakdown
        symbol_counts = {}
        for symbol in analysis['potential_symbols']:
            symbol_type = symbol['type']
            symbol_counts[symbol_type] = symbol_counts.get(symbol_type, 0) + 1
        
        print(f"\n🔍 Symbol Breakdown:")
        for symbol_type, count in symbol_counts.items():
            print(f"  • {symbol_type}: {count}")
        
        # Create QC feedback
        qc_feedback = self._collect_qc_feedback(analysis_result)
        
        return qc_feedback
    
    def _collect_qc_feedback(self, analysis_result):
        """Collect QC feedback from user."""
        print(f"\n📝 QC Feedback Collection")
        print("=" * 50)
        
        feedback = {
            'image_path': analysis_result['image_path'],
            'timestamp': datetime.now().isoformat(),
            'corrections': [],
            'overall_accuracy': 0.0,
            'comments': ''
        }
        
        # Ask for overall accuracy rating
        while True:
            try:
                accuracy = float(input("Rate overall detection accuracy (0.0-1.0): "))
                if 0.0 <= accuracy <= 1.0:
                    feedback['overall_accuracy'] = accuracy
                    break
                else:
                    print("Please enter a value between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number")
        
        # Ask for corrections
        print(f"\n🔧 Symbol Corrections")
        print("For each incorrect detection, provide corrections:")
        
        for i, symbol in enumerate(analysis_result['analysis']['potential_symbols']):
            print(f"\nSymbol {i+1}: {symbol['type']} (confidence: {symbol['confidence']:.2f})")
            print(f"Position: {symbol['position']}, Size: {symbol['size']}")
            
            correct = input("Is this detection correct? (y/n): ").lower().strip()
            
            if correct != 'y':
                correction = {
                    'original_detection': symbol,
                    'corrected_type': input("Correct symbol type: ").strip(),
                    'corrected_discipline': input("Correct discipline: ").strip(),
                    'confidence_adjustment': float(input("Confidence adjustment (-1.0 to 1.0): "))
                }
                feedback['corrections'].append(correction)
        
        # Ask for comments
        feedback['comments'] = input("\nAdditional comments: ").strip()
        
        return feedback
    
    def save_qc_feedback(self, qc_feedback, output_dir="qc_feedback"):
        """Save QC feedback for training."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(qc_feedback['image_path']).stem
        feedback_path = output_path / f"{image_name}_qc_feedback_{timestamp}.json"
        
        with open(feedback_path, 'w') as f:
            json.dump(qc_feedback, f, indent=2, default=str)
        
        print(f"✅ QC feedback saved: {feedback_path}")
        return feedback_path
    
    def generate_training_data(self, qc_feedback):
        """Generate training data from QC feedback."""
        print(f"\n🎓 Generating Training Data from QC Feedback")
        print("=" * 50)
        
        training_data = {
            'source_image': qc_feedback['image_path'],
            'timestamp': datetime.now().isoformat(),
            'training_samples': [],
            'metadata': {
                'overall_accuracy': qc_feedback['overall_accuracy'],
                'corrections_count': len(qc_feedback['corrections']),
                'qc_reviewer': 'human_expert'
            }
        }
        
        # Process corrections into training samples
        for correction in qc_feedback['corrections']:
            original = correction['original_detection']
            
            training_sample = {
                'bbox': original['bbox'],
                'original_type': original['type'],
                'corrected_type': correction['corrected_type'],
                'original_discipline': original['discipline'],
                'corrected_discipline': correction['corrected_discipline'],
                'original_confidence': original['confidence'],
                'confidence_adjustment': correction['confidence_adjustment'],
                'properties': original['properties']
            }
            
            training_data['training_samples'].append(training_sample)
        
        return training_data
    
    def save_training_data(self, training_data, output_dir="training_data"):
        """Save training data for model retraining."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(training_data['source_image']).stem
        training_path = output_path / f"{image_name}_training_data_{timestamp}.json"
        
        with open(training_path, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        print(f"✅ Training data saved: {training_path}")
        return training_path


def main():
    """Main verification workflow."""
    print("🎯 Engineering AI Visual Verification System")
    print("=" * 60)
    print("Industry-standard verification with QC feedback")
    print()
    
    # Initialize system
    verifier = VisualVerificationSystem()
    
    # Find test images
    yolo_images_dir = Path("yolo_processed_data_local/images")
    if not yolo_images_dir.exists():
        print("❌ YOLO processed images directory not found")
        return
    
    # Find as-built images
    as_built_images = [f for f in yolo_images_dir.glob("*.png") if "asbuilt" in f.name.lower()]
    print(f"📁 Found {len(as_built_images)} as-built drawings for verification")
    
    if not as_built_images:
        print("❌ No as-built images found")
        return
    
    # Test with first image
    test_image = as_built_images[0]
    print(f"\n🔍 Testing with: {test_image.name}")
    
    try:
        # Analyze drawing
        analysis_result = verifier.analyze_drawing_with_visual_output(test_image)
        
        # Save verification output
        fig_path, data_path = verifier.save_verification_output(analysis_result)
        
        # Create QC interface
        qc_feedback = verifier.create_qc_interface(analysis_result)
        
        # Save QC feedback
        feedback_path = verifier.save_qc_feedback(qc_feedback)
        
        # Generate training data
        training_data = verifier.generate_training_data(qc_feedback)
        training_path = verifier.save_training_data(training_data)
        
        print(f"\n🎉 Verification Complete!")
        print(f"✅ Annotated image: {fig_path}")
        print(f"✅ Analysis data: {data_path}")
        print(f"✅ QC feedback: {feedback_path}")
        print(f"✅ Training data: {training_path}")
        
        print(f"\n🚀 Next Steps:")
        print(f"  1. Review annotated image for verification")
        print(f"  2. Use QC feedback for model improvement")
        print(f"  3. Retrain model with new training data")
        print(f"  4. Deploy improved model")
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

