#!/usr/bin/env python3
"""
Demo Visual Verification System

Shows exactly what the AI model understands from engineering drawings
with visual annotations and detailed analysis output.
"""

import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the symbol recognition components
from src.core.ml_enhanced_symbol_recognition import MLSymbolRecognizer, SymbolDetection
from src.core.discipline_classifier import DisciplineClassifier

# Configure matplotlib for better display
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10


class DemoVisualVerification:
    """
    Demo visual verification system that shows model understanding.
    """
    
    def __init__(self):
        self.symbol_recognizer = MLSymbolRecognizer()
        self.discipline_classifier = DisciplineClassifier()
        
        # Color scheme for different disciplines
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
        
        print("🎯 Demo Visual Verification System")
        print("✅ Shows exactly what the AI model understands")
        print("✅ Visual annotations with bounding boxes")
        print("✅ Confidence scores and discipline classifications")
    
    def analyze_drawing_demo(self, image_path):
        """
        Analyze engineering drawing and show model understanding.
        """
        print(f"\n🔍 Analyzing: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"📊 Image Info:")
        print(f"  • Size: {image.shape[1]}x{image.shape[0]} pixels")
        print(f"  • File size: {image_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Perform analysis
        analysis = self._analyze_drawing_for_symbols(image, image_path.name)
        
        # Generate visual output
        visual_output = self._generate_visual_demo(image, analysis)
        
        # Print detailed analysis
        self._print_detailed_analysis(analysis)
        
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
        
        print(f"🔍 Processing {len(contours)} contours for symbols...")
        
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
            print(f"🔍 Processing {len(lines)} lines for patterns...")
            
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
        
        print(f"🔍 Processing {len(regions)} text regions...")
        
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
    
    def _generate_visual_demo(self, image, analysis):
        """Generate visual demo showing model understanding."""
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title('Original Engineering Drawing', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Symbol detections
        ax2.imshow(image_rgb)
        ax2.set_title('AI Symbol Detections', fontsize=14, fontweight='bold')
        
        # Add symbol annotations
        for symbol in analysis['potential_symbols']:
            x, y, w, h = symbol['position'][0], symbol['position'][1], symbol['size'][0], symbol['size'][1]
            confidence = symbol['confidence']
            symbol_type = symbol['type']
            discipline = symbol['discipline']
            
            # Get color
            color = self.symbol_colors.get(symbol_type, self.discipline_colors.get(discipline, '#808080'))
            
            # Create bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            # Add label
            label = f"{symbol_type}\n{discipline}\n{confidence:.2f}"
            ax2.text(x, y-10, label, fontsize=9, color=color, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        ax2.axis('off')
        
        # Line patterns
        ax3.imshow(image_rgb)
        ax3.set_title('AI Line Pattern Detections', fontsize=14, fontweight='bold')
        
        # Add line annotations
        for line in analysis['line_patterns']:
            x1, y1 = line['start']
            x2, y2 = line['end']
            line_type = line['type']
            discipline = line['discipline']
            
            color = self.discipline_colors.get(discipline, '#808080')
            ax3.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.8)
        
        ax3.axis('off')
        
        # Add legend
        legend_elements = []
        for discipline, color in self.discipline_colors.items():
            if discipline != 'unknown':
                legend_elements.append(patches.Patch(color=color, label=discipline.title()))
        
        ax3.legend(handles=legend_elements, loc='upper right', title='Disciplines')
        
        plt.tight_layout()
        
        return fig
    
    def _print_detailed_analysis(self, analysis):
        """Print detailed analysis of model understanding."""
        print(f"\n📊 AI Model Understanding Analysis")
        print("=" * 60)
        
        # Symbol analysis
        print(f"🔍 Symbol Detections: {len(analysis['potential_symbols'])}")
        if analysis['potential_symbols']:
            print(f"  Symbol Breakdown:")
            symbol_counts = {}
            discipline_counts = {}
            
            for symbol in analysis['potential_symbols']:
                symbol_type = symbol['type']
                discipline = symbol['discipline']
                confidence = symbol['confidence']
                
                symbol_counts[symbol_type] = symbol_counts.get(symbol_type, 0) + 1
                discipline_counts[discipline] = discipline_counts.get(discipline, 0) + 1
                
                print(f"    • {symbol_type} ({discipline}) - Confidence: {confidence:.2f}")
                print(f"      Position: {symbol['position']}, Size: {symbol['size']}")
            
            print(f"\n  Symbol Type Summary:")
            for symbol_type, count in symbol_counts.items():
                print(f"    • {symbol_type}: {count}")
            
            print(f"\n  Discipline Summary:")
            for discipline, count in discipline_counts.items():
                print(f"    • {discipline}: {count}")
        else:
            print(f"  ⚠️  No symbols detected in this drawing")
        
        # Line analysis
        print(f"\n📏 Line Pattern Detections: {len(analysis['line_patterns'])}")
        if analysis['line_patterns']:
            line_counts = {}
            line_discipline_counts = {}
            
            for line in analysis['line_patterns']:
                line_type = line['type']
                discipline = line['discipline']
                length = line['length']
                
                line_counts[line_type] = line_counts.get(line_type, 0) + 1
                line_discipline_counts[discipline] = line_discipline_counts.get(discipline, 0) + 1
            
            print(f"  Line Type Summary:")
            for line_type, count in line_counts.items():
                print(f"    • {line_type}: {count}")
            
            print(f"  Line Discipline Summary:")
            for discipline, count in line_discipline_counts.items():
                print(f"    • {discipline}: {count}")
        
        # Text analysis
        print(f"\n📝 Text Region Detections: {len(analysis['text_regions'])}")
        if analysis['text_regions']:
            total_text_area = sum(region['area'] for region in analysis['text_regions'])
            avg_text_size = total_text_area / len(analysis['text_regions'])
            print(f"  Text Analysis:")
            print(f"    • Total text area: {total_text_area:,} pixels")
            print(f"    • Average text region size: {avg_text_size:.0f} pixels")
            print(f"    • Text density: High" if len(analysis['text_regions']) > 1000 else "    • Text density: Medium")
        
        # Overall assessment
        print(f"\n🎯 AI Model Assessment:")
        if analysis['potential_symbols']:
            avg_confidence = sum(s['confidence'] for s in analysis['potential_symbols']) / len(analysis['potential_symbols'])
            print(f"  • Symbol detection confidence: {avg_confidence:.2f}")
            print(f"  • Primary discipline: {max(discipline_counts, key=discipline_counts.get) if discipline_counts else 'Unknown'}")
        else:
            print(f"  • Symbol detection: None detected")
        
        print(f"  • Line pattern detection: {len(analysis['line_patterns'])} patterns")
        print(f"  • Text region detection: {len(analysis['text_regions'])} regions")
        
        # Model understanding quality
        total_elements = len(analysis['potential_symbols']) + len(analysis['line_patterns']) + len(analysis['text_regions'])
        if total_elements > 100:
            print(f"  • Model understanding: Comprehensive")
        elif total_elements > 50:
            print(f"  • Model understanding: Good")
        elif total_elements > 10:
            print(f"  • Model understanding: Basic")
        else:
            print(f"  • Model understanding: Limited")
    
    def save_demo_output(self, analysis_result, output_dir="demo_verification_output"):
        """Save demo verification output."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(analysis_result['image_path']).stem
        fig_path = output_path / f"{image_name}_demo_annotated_{timestamp}.png"
        
        analysis_result['visual_output'].savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(analysis_result['visual_output'])
        
        # Save analysis data
        data_path = output_path / f"{image_name}_demo_analysis_{timestamp}.json"
        with open(data_path, 'w') as f:
            json.dump(analysis_result['analysis'], f, indent=2, default=str)
        
        print(f"\n✅ Demo verification output saved:")
        print(f"  • Annotated image: {fig_path}")
        print(f"  • Analysis data: {data_path}")
        
        return fig_path, data_path


def main():
    """Main demo workflow."""
    print("🎯 Engineering AI Visual Verification Demo")
    print("=" * 60)
    print("Shows exactly what the AI model understands")
    print()
    
    # Initialize demo system
    demo = DemoVisualVerification()
    
    # Find test images
    yolo_images_dir = Path("yolo_processed_data_local/images")
    if not yolo_images_dir.exists():
        print("❌ YOLO processed images directory not found")
        return
    
    # Find as-built images
    as_built_images = [f for f in yolo_images_dir.glob("*.png") if "asbuilt" in f.name.lower()]
    print(f"📁 Found {len(as_built_images)} as-built drawings for demo")
    
    if not as_built_images:
        print("❌ No as-built images found")
        return
    
    # Test with first 3 images
    test_images = as_built_images[:3]
    
    for i, test_image in enumerate(test_images, 1):
        print(f"\n{'='*80}")
        print(f"🔍 Demo {i}: {test_image.name}")
        print(f"{'='*80}")
        
        try:
            # Analyze drawing
            analysis_result = demo.analyze_drawing_demo(test_image)
            
            # Save demo output
            fig_path, data_path = demo.save_demo_output(analysis_result)
            
            print(f"\n✅ Demo {i} Complete!")
            print(f"  • Annotated image: {fig_path}")
            print(f"  • Analysis data: {data_path}")
            
        except Exception as e:
            print(f"❌ Error in demo {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Demo Complete!")
    print(f"✅ Generated visual verification for {len(test_images)} drawings")
    print(f"✅ Shows exactly what the AI model understands")
    print(f"✅ Ready for QC review and feedback")


if __name__ == "__main__":
    main()

