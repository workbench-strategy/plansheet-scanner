#!/usr/bin/env python3
"""
Tiny Line Detector - Specialized Expert Model

Dedicated system for detecting tiny lines, conduit, small electrical symbols,
and fine details that the main detection system misses.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TinyLineDetector:
    """
    Specialized detector for tiny lines and small electrical symbols.
    
    Features:
    - Multi-scale line detection (tiny to small)
    - Conduit pattern recognition
    - Small electrical symbol detection
    - Fine detail preservation
    - High sensitivity for small elements
    """
    
    def __init__(self):
        # Tiny line detection parameters
        self.tiny_line_params = {
            'min_length': 5,      # Very short lines
            'max_length': 200,    # Small lines
            'min_thickness': 1,   # Single pixel lines
            'max_thickness': 5,   # Thin lines
            'angle_tolerance': 15, # Degrees
            'gap_tolerance': 3    # Pixels
        }
        
        # Conduit detection parameters
        self.conduit_params = {
            'min_length': 10,
            'max_length': 500,
            'aspect_ratio_range': (3, 20),  # Long and thin
            'parallel_threshold': 10,       # Pixels
            'spacing_range': (5, 50)        # Pixels between parallel lines
        }
        
        # Small symbol parameters
        self.small_symbol_params = {
            'min_area': 5,        # Very small areas
            'max_area': 1000,     # Small symbols
            'min_dimension': 2,   # Pixels
            'max_dimension': 50   # Pixels
        }
        
        # Color scheme for tiny elements
        self.tiny_colors = {
            'conduit': '#FF0000',           # Bright red
            'electrical_line': '#FF6600',   # Orange
            'small_symbol': '#FFCC00',      # Yellow
            'detail_line': '#00FF00',       # Green
            'fine_detail': '#00CCFF'        # Light blue
        }
        
        print("🔍 Tiny Line Detector Initialized")
        print("✅ Specialized for small electrical symbols")
        print("✅ Expert at conduit detection")
        print("✅ High sensitivity for fine details")
    
    def detect_tiny_lines(self, image):
        """
        Detect tiny lines and small electrical symbols with high sensitivity.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale edge detection for tiny elements
        tiny_edges = self._detect_tiny_edges(gray)
        
        # Detect tiny lines
        tiny_lines = self._find_tiny_lines(tiny_edges)
        
        # Detect conduit patterns
        conduit_patterns = self._detect_conduit_patterns(tiny_edges)
        
        # Detect small symbols
        small_symbols = self._detect_small_symbols(gray)
        
        # Combine results
        results = {
            'tiny_lines': tiny_lines,
            'conduit_patterns': conduit_patterns,
            'small_symbols': small_symbols,
            'total_detections': len(tiny_lines) + len(conduit_patterns) + len(small_symbols)
        }
        
        return results
    
    def _detect_tiny_edges(self, gray):
        """Detect edges optimized for tiny elements."""
        # Multiple edge detection methods for tiny elements
        
        # Method 1: Very sensitive Canny
        edges1 = cv2.Canny(gray, 10, 30)  # Lower thresholds for sensitivity
        
        # Method 2: Sobel for fine details
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_magnitude = np.uint8(sobel_magnitude * 255 / sobel_magnitude.max())
        
        # Method 3: Laplacian for fine edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Combine edge detections
        combined_edges = cv2.bitwise_or(edges1, sobel_magnitude)
        combined_edges = cv2.bitwise_or(combined_edges, laplacian)
        
        # Morphological operations to connect tiny line segments
        kernel = np.ones((2, 2), np.uint8)
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        return combined_edges
    
    def _find_tiny_lines(self, edges):
        """Find tiny lines with high sensitivity."""
        tiny_lines = []
        
        # Use HoughLinesP with very sensitive parameters
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=5,  # Very low threshold
            minLineLength=self.tiny_line_params['min_length'],
            maxLineGap=self.tiny_line_params['gap_tolerance']
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Check if it's a tiny line
                if (self.tiny_line_params['min_length'] <= length <= 
                    self.tiny_line_params['max_length']):
                    
                    # Calculate line properties
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    thickness = self._estimate_line_thickness(edges, x1, y1, x2, y2)
                    
                    # Classify tiny line
                    line_type = self._classify_tiny_line(length, angle, thickness)
                    
                    tiny_line = {
                        'type': line_type,
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': length,
                        'angle': angle,
                        'thickness': thickness,
                        'confidence': self._calculate_tiny_line_confidence(length, thickness),
                        'bbox': [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    }
                    
                    tiny_lines.append(tiny_line)
        
        return tiny_lines
    
    def _detect_conduit_patterns(self, edges):
        """Detect conduit patterns (parallel lines)."""
        conduit_patterns = []
        
        # Find all lines first
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=10,
            minLineLength=self.conduit_params['min_length'],
            maxLineGap=5
        )
        
        if lines is not None:
            # Group lines by angle to find parallel patterns
            angle_groups = {}
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Normalize angle to 0-180 degrees
                if angle < 0:
                    angle += 180
                
                # Group by angle (within tolerance)
                grouped = False
                for group_angle in angle_groups:
                    if abs(angle - group_angle) <= self.conduit_params['angle_tolerance']:
                        angle_groups[group_angle].append(line[0])
                        grouped = True
                        break
                
                if not grouped:
                    angle_groups[angle] = [line[0]]
            
            # Find parallel line patterns (conduit)
            for angle, lines_in_group in angle_groups.items():
                if len(lines_in_group) >= 2:  # At least 2 parallel lines
                    # Check spacing between lines
                    parallel_pattern = self._find_parallel_pattern(lines_in_group, angle)
                    
                    if parallel_pattern:
                        conduit_patterns.append(parallel_pattern)
        
        return conduit_patterns
    
    def _find_parallel_pattern(self, lines, angle):
        """Find parallel line patterns that could be conduit."""
        if len(lines) < 2:
            return None
        
        # Sort lines by perpendicular distance from origin
        sorted_lines = sorted(lines, key=lambda l: self._perpendicular_distance(l, angle))
        
        # Check spacing between consecutive lines
        pattern_lines = [sorted_lines[0]]
        
        for i in range(1, len(sorted_lines)):
            spacing = self._line_spacing(sorted_lines[i-1], sorted_lines[i], angle)
            
            if (self.conduit_params['spacing_range'][0] <= spacing <= 
                self.conduit_params['spacing_range'][1]):
                pattern_lines.append(sorted_lines[i])
        
        if len(pattern_lines) >= 2:
            # Calculate pattern properties
            total_length = sum(np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in pattern_lines)
            avg_spacing = np.mean([self._line_spacing(pattern_lines[i], pattern_lines[i+1], angle) 
                                 for i in range(len(pattern_lines)-1)])
            
            return {
                'type': 'conduit_pattern',
                'lines': pattern_lines,
                'angle': angle,
                'line_count': len(pattern_lines),
                'total_length': total_length,
                'avg_spacing': avg_spacing,
                'confidence': min(len(pattern_lines) / 3.0, 1.0),  # Higher confidence with more lines
                'bbox': self._calculate_pattern_bbox(pattern_lines)
            }
        
        return None
    
    def _detect_small_symbols(self, gray):
        """Detect very small symbols and electrical components."""
        small_symbols = []
        
        # Use MSER for small regions
        mser = cv2.MSER_create(
            _min_area=self.small_symbol_params['min_area'],
            _max_area=self.small_symbol_params['max_area']
        )
        
        regions, _ = mser.detectRegions(gray)
        
        for region in regions:
            if len(region) >= 5:  # Minimum points for a meaningful region
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                x, y, w, h = cv2.boundingRect(hull)
                
                # Check size constraints
                if (self.small_symbol_params['min_dimension'] <= w <= 
                    self.small_symbol_params['max_dimension'] and
                    self.small_symbol_params['min_dimension'] <= h <= 
                    self.small_symbol_params['max_dimension']):
                    
                    # Analyze region properties
                    area = cv2.contourArea(hull)
                    aspect_ratio = w / h if h > 0 else 0
                    extent = area / (w * h) if w * h > 0 else 0
                    
                    # Classify small symbol
                    symbol_type = self._classify_small_symbol(area, aspect_ratio, extent, len(region))
                    
                    if symbol_type:
                        small_symbol = {
                            'type': symbol_type,
                            'position': (x, y),
                            'size': (w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'extent': extent,
                            'region_points': len(region),
                            'confidence': self._calculate_small_symbol_confidence(area, extent, len(region)),
                            'bbox': [x, y, x + w, y + h]
                        }
                        
                        small_symbols.append(small_symbol)
        
        return small_symbols
    
    def _classify_tiny_line(self, length, angle, thickness):
        """Classify tiny line based on properties."""
        # Normalize angle
        if angle < 0:
            angle += 180
        
        # Electrical lines (often horizontal or vertical)
        if (abs(angle) <= 10 or abs(angle - 90) <= 10) and thickness <= 3:
            return 'electrical_line'
        
        # Conduit (longer, often diagonal)
        elif length > 50 and thickness <= 2:
            return 'conduit'
        
        # Detail lines (shorter, various angles)
        elif length <= 30:
            return 'detail_line'
        
        # Fine details (very thin)
        elif thickness == 1:
            return 'fine_detail'
        
        return 'unknown_line'
    
    def _classify_small_symbol(self, area, aspect_ratio, extent, points):
        """Classify small symbol based on properties."""
        # Small electrical components
        if 10 <= area <= 200 and 0.5 <= aspect_ratio <= 2.0 and extent > 0.6:
            return 'small_electrical_component'
        
        # Tiny symbols
        elif area <= 50 and extent > 0.7:
            return 'tiny_symbol'
        
        # Small circles (light poles, junction boxes)
        elif 0.7 <= aspect_ratio <= 1.3 and extent > 0.5:
            return 'small_circle'
        
        # Small rectangles (panels, switches)
        elif aspect_ratio > 2.0 and extent > 0.6:
            return 'small_rectangle'
        
        return None
    
    def _estimate_line_thickness(self, edges, x1, y1, x2, y2):
        """Estimate the thickness of a line."""
        # Sample points along the line
        num_samples = 10
        thicknesses = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            # Check perpendicular direction for thickness
            if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                # Sample perpendicular to line direction
                angle = np.arctan2(y2-y1, x2-x1) + np.pi/2
                thickness = 0
                
                for d in range(-5, 6):  # Check up to 5 pixels perpendicular
                    px = int(x + d * np.cos(angle))
                    py = int(y + d * np.sin(angle))
                    
                    if (0 <= px < edges.shape[1] and 0 <= py < edges.shape[0] and 
                        edges[py, px] > 0):
                        thickness = max(thickness, abs(d))
                
                thicknesses.append(thickness)
        
        return np.mean(thicknesses) if thicknesses else 1
    
    def _perpendicular_distance(self, line, angle):
        """Calculate perpendicular distance from line to origin."""
        x1, y1, x2, y2 = line
        # Distance from point (x1, y1) to line through origin with given angle
        return abs(x1 * np.cos(np.radians(angle + 90)) + y1 * np.sin(np.radians(angle + 90)))
    
    def _line_spacing(self, line1, line2, angle):
        """Calculate spacing between two parallel lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Use midpoint of first line
        mid_x1 = (x1 + x2) / 2
        mid_y1 = (y1 + y2) / 2
        
        # Use midpoint of second line
        mid_x2 = (x3 + x4) / 2
        mid_y2 = (y3 + y4) / 2
        
        # Calculate perpendicular distance
        return abs((mid_x2 - mid_x1) * np.cos(np.radians(angle + 90)) + 
                  (mid_y2 - mid_y1) * np.sin(np.radians(angle + 90)))
    
    def _calculate_pattern_bbox(self, lines):
        """Calculate bounding box for a pattern of lines."""
        if not lines:
            return [0, 0, 0, 0]
        
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    def _calculate_tiny_line_confidence(self, length, thickness):
        """Calculate confidence for tiny line detection."""
        confidence = 0.3  # Base confidence
        
        # Length factor
        if 10 <= length <= 100:
            confidence += 0.2
        elif 5 <= length <= 200:
            confidence += 0.1
        
        # Thickness factor (thinner is better for tiny lines)
        if thickness <= 2:
            confidence += 0.2
        elif thickness <= 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_small_symbol_confidence(self, area, extent, points):
        """Calculate confidence for small symbol detection."""
        confidence = 0.3  # Base confidence
        
        # Area factor
        if 10 <= area <= 200:
            confidence += 0.2
        elif 5 <= area <= 500:
            confidence += 0.1
        
        # Extent factor (how well it fills its bounding box)
        if extent > 0.7:
            confidence += 0.2
        elif extent > 0.5:
            confidence += 0.1
        
        # Points factor (more points = more complex shape)
        if 10 <= points <= 100:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def visualize_tiny_detections(self, image, results):
        """Visualize tiny line and symbol detections."""
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Tiny detections
        ax2.imshow(image_rgb)
        ax2.set_title('Tiny Line & Symbol Detections', fontsize=14, fontweight='bold')
        
        # Draw tiny lines
        for line in results['tiny_lines']:
            x1, y1 = line['start']
            x2, y2 = line['end']
            line_type = line['type']
            confidence = line['confidence']
            
            color = self.tiny_colors.get(line_type, '#808080')
            ax2.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.8)
            
            # Add label for high-confidence detections
            if confidence > 0.6:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax2.text(mid_x, mid_y, f"{line_type}\n{confidence:.2f}", 
                        fontsize=6, color=color, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Draw conduit patterns
        for pattern in results['conduit_patterns']:
            color = self.tiny_colors['conduit']
            for line in pattern['lines']:
                x1, y1, x2, y2 = line
                ax2.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.9)
            
            # Add pattern label
            if pattern['confidence'] > 0.5:
                bbox = pattern['bbox']
                mid_x = (bbox[0] + bbox[2]) / 2
                mid_y = (bbox[1] + bbox[3]) / 2
                ax2.text(mid_x, mid_y, f"Conduit\n{pattern['line_count']} lines\n{pattern['confidence']:.2f}",
                        fontsize=8, color=color, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Draw small symbols
        for symbol in results['small_symbols']:
            x, y, w, h = symbol['position'][0], symbol['position'][1], symbol['size'][0], symbol['size'][1]
            symbol_type = symbol['type']
            confidence = symbol['confidence']
            
            color = self.tiny_colors.get(symbol_type, '#808080')
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            # Add label for high-confidence detections
            if confidence > 0.5:
                ax2.text(x, y-2, f"{symbol_type}\n{confidence:.2f}",
                        fontsize=6, color=color, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        ax2.axis('off')
        
        # Add legend
        legend_elements = []
        for line_type, color in self.tiny_colors.items():
            legend_elements.append(patches.Patch(color=color, label=line_type.replace('_', ' ').title()))
        
        ax2.legend(handles=legend_elements, loc='upper right', title='Tiny Elements')
        
        plt.tight_layout()
        
        return fig
    
    def save_tiny_detections(self, results, output_dir="tiny_detections"):
        """Save tiny detection results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results_path = output_path / f"tiny_detections_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Tiny detections saved: {results_path}")
        print(f"📊 Detection Summary:")
        print(f"  • Tiny lines: {len(results['tiny_lines'])}")
        print(f"  • Conduit patterns: {len(results['conduit_patterns'])}")
        print(f"  • Small symbols: {len(results['small_symbols'])}")
        print(f"  • Total detections: {results['total_detections']}")
        
        return results_path


def main():
    """Test the tiny line detector."""
    print("🔍 Tiny Line Detector Test")
    print("=" * 50)
    
    # Initialize detector
    detector = TinyLineDetector()
    
    # Find test images
    yolo_images_dir = Path("yolo_processed_data_local/images")
    if not yolo_images_dir.exists():
        print("❌ YOLO processed images directory not found")
        return
    
    # Find as-built images
    as_built_images = [f for f in yolo_images_dir.glob("*.png") if "asbuilt" in f.name.lower()]
    print(f"📁 Found {len(as_built_images)} as-built drawings for testing")
    
    if not as_built_images:
        print("❌ No as-built images found")
        return
    
    # Test with first image
    test_image = as_built_images[0]
    print(f"\n🔍 Testing with: {test_image.name}")
    
    # Load image
    image = cv2.imread(str(test_image))
    if image is None:
        print(f"❌ Could not load image: {test_image}")
        return
    
    # Detect tiny elements
    results = detector.detect_tiny_lines(image)
    
    # Visualize results
    fig = detector.visualize_tiny_detections(image, results)
    
    # Save results
    detector.save_tiny_detections(results)
    
    # Save visualization
    output_path = Path("tiny_detections")
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_path / f"tiny_detections_visualization_{timestamp}.png"
    
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Visualization saved: {fig_path}")
    print(f"\n🎉 Tiny line detection complete!")


if __name__ == "__main__":
    main()
