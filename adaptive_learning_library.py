#!/usr/bin/env python3
"""
Adaptive Learning Library System

Advanced system that learns to distinguish between similar visual elements:
- Borders vs Walls (both thick lines but different contexts)
- Conduits vs Pipes vs Cables
- Guardrails vs Barriers vs Handrails
- Electrical symbols vs Mechanical symbols
- Index patterns vs General text

Features:
- Expandable knowledge base
- Context-aware classification
- Feedback-driven learning
- Pattern recognition improvement
- Multi-dimensional feature analysis
"""

import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class AdaptiveLearningLibrary:
    """
    Adaptive learning system that builds and expands knowledge of engineering elements.
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Knowledge base storage
        self.knowledge_base = {
            'element_patterns': {},
            'context_rules': {},
            'feature_clusters': {},
            'classification_history': [],
            'feedback_data': [],
            'confidence_thresholds': {}
        }
        
        # Initialize with basic patterns
        self._initialize_basic_patterns()
        
        # Learning parameters
        self.learning_rate = 0.1
        self.min_confidence_threshold = 0.6
        self.max_patterns_per_element = 100
        
        print("🧠 Adaptive Learning Library Initialized")
        print("✅ Expandable knowledge base")
        print("✅ Context-aware classification")
        print("✅ Feedback-driven learning")
        print("✅ Pattern recognition improvement")
    
    def _setup_logging(self):
        """Setup logging for the adaptive learning system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('adaptive_learning.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_basic_patterns(self):
        """Initialize the knowledge base with basic engineering patterns."""
        
        # Border vs Wall patterns
        self.knowledge_base['element_patterns']['border'] = {
            'description': 'Drawing border lines',
            'visual_features': {
                'line_thickness': (3, 8),  # Pixels
                'line_style': 'solid',
                'position': 'perimeter',
                'color': 'black',
                'context': 'drawing_boundary'
            },
            'context_rules': [
                'Located at drawing edges',
                'Forms rectangular boundary',
                'Consistent thickness',
                'No breaks or interruptions',
                'Consistent across project pages',
                'Same style and thickness throughout project',
                'Always on outside edge of drawing area',
                'May have title block or project info'
            ],
            'examples': [],
            'confidence_score': 0.8
        }
        
        # Scale patterns
        self.knowledge_base['element_patterns']['scale'] = {
            'description': 'Drawing scale indicators',
            'visual_features': {
                'line_thickness': (2, 5),
                'line_style': 'solid',
                'pattern': 'scale_bar',
                'position': 'corner_or_legend',
                'context': 'measurement_reference'
            },
            'context_rules': [
                'Usually in corner or legend area',
                'Has numerical labels',
                'Shows measurement units',
                'Often has tick marks',
                'May have text like "SCALE" or "1" = 50\'"'
            ],
            'examples': [],
            'confidence_score': 0.85
        }
        
        # North arrow patterns
        self.knowledge_base['element_patterns']['north_arrow'] = {
            'description': 'North direction indicators',
            'visual_features': {
                'shape': 'arrow_or_compass',
                'size': 'small_to_medium',
                'position': 'corner_or_legend',
                'context': 'orientation_reference'
            },
            'context_rules': [
                'Usually in corner or legend area',
                'Arrow pointing upward (north)',
                'May have "N" label',
                'Often has compass rose design',
                'Can be simple arrow or elaborate compass'
            ],
            'examples': [],
            'confidence_score': 0.9
        }
        
        # Match line patterns
        self.knowledge_base['element_patterns']['match_line'] = {
            'description': 'Drawing match lines for sheet connections',
            'visual_features': {
                'line_thickness': (2, 4),
                'line_style': 'dashed_or_dotted',
                'pattern': 'match_line',
                'position': 'station_edge',
                'context': 'sheet_connection'
            },
            'context_rules': [
                'Located at station line edges',
                'Dashed or dotted line style',
                'May have "MATCH LINE" text',
                'Shows where sheets connect',
                'Often has reference numbers',
                'Appears at specific station intervals',
                'Consistent across project pages',
                'May have station numbers (e.g., STA 100+00)'
            ],
            'examples': [],
            'confidence_score': 0.8
        }
        
        self.knowledge_base['element_patterns']['wall'] = {
            'description': 'Structural wall elements',
            'visual_features': {
                'line_thickness': (2, 6),  # Pixels
                'line_style': 'solid',
                'position': 'interior',
                'color': 'black',
                'context': 'structural_element'
            },
            'context_rules': [
                'Located within drawing area',
                'May have openings (doors/windows)',
                'Connected to other walls',
                'Part of room/space definition'
            ],
            'examples': [],
            'confidence_score': 0.7
        }
        
        # Conduit vs Pipe patterns
        self.knowledge_base['element_patterns']['conduit'] = {
            'description': 'Electrical conduit systems',
            'visual_features': {
                'line_thickness': (1, 3),
                'line_style': 'solid',
                'pattern': 'parallel_lines',
                'context': 'electrical_system'
            },
            'context_rules': [
                'Often in parallel bundles',
                'Connected to electrical symbols',
                'May have junction boxes',
                'Follows electrical routing patterns'
            ],
            'examples': [],
            'confidence_score': 0.75
        }
        
        self.knowledge_base['element_patterns']['pipe'] = {
            'description': 'Mechanical pipe systems',
            'visual_features': {
                'line_thickness': (2, 4),
                'line_style': 'solid',
                'pattern': 'single_line',
                'context': 'mechanical_system'
            },
            'context_rules': [
                'Connected to mechanical symbols',
                'May have valves or fittings',
                'Follows plumbing/HVAC patterns',
                'Often labeled with pipe specs'
            ],
            'examples': [],
            'confidence_score': 0.7
        }
        
        # Guardrail vs Barrier patterns
        self.knowledge_base['element_patterns']['guardrail'] = {
            'description': 'Roadway guardrail systems',
            'visual_features': {
                'line_thickness': (2, 5),
                'line_style': 'solid',
                'pattern': 'continuous_line',
                'context': 'roadway_safety'
            },
            'context_rules': [
                'Located along roadway edges',
                'Continuous horizontal line',
                'May have post indicators',
                'Part of traffic safety system'
            ],
            'examples': [],
            'confidence_score': 0.8
        }
        
        self.knowledge_base['element_patterns']['barrier'] = {
            'description': 'Concrete or metal barriers',
            'visual_features': {
                'line_thickness': (4, 8),
                'line_style': 'solid',
                'pattern': 'thick_line',
                'context': 'traffic_control'
            },
            'context_rules': [
                'Thicker than guardrails',
                'May be segmented',
                'Often in median areas',
                'Provides physical separation'
            ],
            'examples': [],
            'confidence_score': 0.75
        }
        
        # Index vs General text patterns
        self.knowledge_base['element_patterns']['index_text'] = {
            'description': 'Drawing index and reference text',
            'visual_features': {
                'text_size': 'small',
                'font_style': 'standard',
                'position': 'organized_grid',
                'context': 'reference_system'
            },
            'context_rules': [
                'Organized in grid pattern',
                'Contains reference numbers',
                'Located in specific areas',
                'Follows consistent format'
            ],
            'examples': [],
            'confidence_score': 0.85
        }
        
        self.knowledge_base['element_patterns']['general_text'] = {
            'description': 'General drawing annotations',
            'visual_features': {
                'text_size': 'variable',
                'font_style': 'variable',
                'position': 'scattered',
                'context': 'annotation'
            },
            'context_rules': [
                'Scattered throughout drawing',
                'Variable sizes and styles',
                'Labels specific elements',
                'No consistent pattern'
            ],
            'examples': [],
            'confidence_score': 0.6
        }
    
    def analyze_element(self, image_path, region_bbox=None):
        """
        Analyze a specific element in an image and classify it using the knowledge base.
        """
        self.logger.info(f"Analyzing element in: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return None
        
        # Extract region if specified
        if region_bbox:
            x, y, w, h = region_bbox
            image = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract comprehensive features
        features = self._extract_element_features(gray, image)
        
        # Classify using knowledge base
        classification = self._classify_element(features)
        
        # Store for learning
        self._store_classification_example(features, classification)
        
        return {
            'image_path': str(image_path),
            'region_bbox': region_bbox,
            'extracted_features': features,
            'classification': classification,
            'confidence': classification['confidence'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_element_features(self, gray, color_image):
        """
        Extract comprehensive features for element classification.
        """
        features = {}
        
        # 1. Line analysis
        line_features = self._analyze_lines(gray)
        features.update(line_features)
        
        # 2. Shape analysis
        shape_features = self._analyze_shapes(gray)
        features.update(shape_features)
        
        # 3. Texture analysis
        texture_features = self._analyze_texture(gray)
        features.update(texture_features)
        
        # 4. Position analysis
        position_features = self._analyze_position(gray)
        features.update(position_features)
        
        # 5. Context analysis
        context_features = self._analyze_context(gray, color_image)
        features.update(context_features)
        
        # 6. Engineering-specific analysis
        engineering_features = self._analyze_engineering_patterns(gray, color_image)
        features.update(engineering_features)
        
        return features
    
    def _analyze_lines(self, gray):
        """Analyze line characteristics."""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, 
                               minLineLength=30, maxLineGap=10)
        
        line_features = {
            'total_lines': 0,
            'avg_line_length': 0,
            'avg_line_thickness': 0,
            'line_density': 0,
            'horizontal_lines': 0,
            'vertical_lines': 0,
            'diagonal_lines': 0,
            'parallel_line_groups': 0
        }
        
        if lines is not None:
            line_features['total_lines'] = len(lines)
            
            line_lengths = []
            line_angles = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                line_lengths.append(length)
                line_angles.append(angle)
            
            if line_lengths:
                line_features['avg_line_length'] = np.mean(line_lengths)
                line_features['line_density'] = len(lines) / (gray.shape[0] * gray.shape[1]) * 10000
                
                # Count line orientations
                for angle in line_angles:
                    if abs(angle) <= 15 or abs(angle - 180) <= 15:
                        line_features['horizontal_lines'] += 1
                    elif abs(angle - 90) <= 15:
                        line_features['vertical_lines'] += 1
                    else:
                        line_features['diagonal_lines'] += 1
                
                # Find parallel line groups
                line_features['parallel_line_groups'] = self._find_parallel_groups(line_angles)
        
        return line_features
    
    def _analyze_shapes(self, gray):
        """Analyze shape characteristics."""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_features = {
            'contour_count': len(contours),
            'total_contour_area': 0,
            'avg_contour_area': 0,
            'contour_density': 0,
            'rectangular_shapes': 0,
            'circular_shapes': 0,
            'aspect_ratios': []
        }
        
        if contours:
            areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                areas.append(area)
                
                # Analyze shape type
                if len(contour) >= 4:
                    # Check if rectangular
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    rect_area = cv2.contourArea(box)
                    
                    if area > 0 and abs(area - rect_area) / area < 0.1:
                        shape_features['rectangular_shapes'] += 1
                    
                    # Calculate aspect ratio
                    if rect[1][0] > 0 and rect[1][1] > 0:
                        aspect_ratio = max(rect[1]) / min(rect[1])
                        shape_features['aspect_ratios'].append(aspect_ratio)
            
            shape_features['total_contour_area'] = sum(areas)
            shape_features['avg_contour_area'] = np.mean(areas) if areas else 0
            shape_features['contour_density'] = len(contours) / (gray.shape[0] * gray.shape[1]) * 10000
        
        return shape_features
    
    def _analyze_texture(self, gray):
        """Analyze texture characteristics."""
        # Calculate texture features using GLCM-like approach
        texture_features = {
            'smoothness': 0,
            'uniformity': 0,
            'entropy': 0,
            'contrast': 0
        }
        
        # Simple texture measures
        texture_features['smoothness'] = 1.0 / (1.0 + cv2.Laplacian(gray, cv2.CV_64F).var())
        texture_features['uniformity'] = np.sum(gray**2) / (gray.shape[0] * gray.shape[1])
        texture_features['entropy'] = -np.sum(gray * np.log2(gray + 1e-10)) / (gray.shape[0] * gray.shape[1])
        texture_features['contrast'] = np.std(gray)
        
        return texture_features
    
    def _analyze_position(self, gray):
        """Analyze position characteristics."""
        height, width = gray.shape
        
        position_features = {
            'center_x': width / 2,
            'center_y': height / 2,
            'distance_from_center': 0,
            'distance_from_edge': 0,
            'relative_position': 'center'
        }
        
        # Calculate distances
        center_x, center_y = width / 2, height / 2
        position_features['distance_from_center'] = np.sqrt(center_x**2 + center_y**2)
        position_features['distance_from_edge'] = min(center_x, center_y, width - center_x, height - center_y)
        
        # Determine relative position
        if center_x < width * 0.2 or center_x > width * 0.8:
            position_features['relative_position'] = 'edge_horizontal'
        elif center_y < height * 0.2 or center_y > height * 0.8:
            position_features['relative_position'] = 'edge_vertical'
        else:
            position_features['relative_position'] = 'interior'
        
        return position_features
    
    def _analyze_context(self, gray, color_image):
        """Analyze contextual features."""
        context_features = {
            'surrounding_elements': 0,
            'connection_points': 0,
            'symbol_proximity': 0,
            'text_proximity': 0,
            'background_complexity': 0
        }
        
        # Analyze surrounding area
        edges = cv2.Canny(gray, 50, 150)
        context_features['surrounding_elements'] = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Background complexity
        context_features['background_complexity'] = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return context_features
    
    def _analyze_engineering_patterns(self, gray, color_image):
        """
        Analyze engineering-specific patterns like station lines, match lines, etc.
        """
        engineering_features = {
            'station_line_patterns': 0,
            'match_line_patterns': 0,
            'border_consistency': 0,
            'scale_bar_patterns': 0,
            'north_arrow_patterns': 0,
            'dashed_line_patterns': 0,
            'dotted_line_patterns': 0,
            'edge_regularity': 0
        }
        
        # Enhanced edge detection for engineering patterns
        edges_canny = cv2.Canny(gray, 30, 100)
        edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges_sobel = np.uint8(np.absolute(edges_sobel))
        
        # Detect dashed and dotted lines
        engineering_features.update(self._detect_line_patterns(gray, edges_canny))
        
        # Detect station line patterns
        engineering_features['station_line_patterns'] = self._detect_station_patterns(gray)
        
        # Detect match line patterns
        engineering_features['match_line_patterns'] = self._detect_match_line_patterns(gray)
        
        # Analyze border consistency
        engineering_features['border_consistency'] = self._analyze_border_consistency(gray)
        
        # Detect scale bar patterns
        engineering_features['scale_bar_patterns'] = self._detect_scale_patterns(gray)
        
        # Detect north arrow patterns
        engineering_features['north_arrow_patterns'] = self._detect_north_arrow_patterns(gray)
        
        # Analyze edge regularity
        engineering_features['edge_regularity'] = self._analyze_edge_regularity(edges_canny)
        
        return engineering_features
    
    def _detect_line_patterns(self, gray, edges):
        """Detect dashed and dotted line patterns."""
        patterns = {
            'dashed_line_patterns': 0,
            'dotted_line_patterns': 0
        }
        
        # Find lines using Hough Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, 
                               minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Analyze line segment for patterns
                line_segment = gray[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                
                if line_segment.size > 0:
                    # Check for dashed pattern (gaps in line)
                    if self._is_dashed_pattern(line_segment):
                        patterns['dashed_line_patterns'] += 1
                    
                    # Check for dotted pattern (discrete points)
                    if self._is_dotted_pattern(line_segment):
                        patterns['dotted_line_patterns'] += 1
        
        return patterns
    
    def _is_dashed_pattern(self, line_segment):
        """Check if line segment has dashed pattern."""
        if line_segment.size < 10:
            return False
        
        # Calculate variance in intensity along the line
        intensity_variance = np.var(line_segment)
        mean_intensity = np.mean(line_segment)
        
        # Dashed lines have high variance due to gaps
        return intensity_variance > mean_intensity * 0.3
    
    def _is_dotted_pattern(self, line_segment):
        """Check if line segment has dotted pattern."""
        if line_segment.size < 5:
            return False
        
        # Count distinct bright regions (dots)
        threshold = np.mean(line_segment) + np.std(line_segment)
        bright_regions = np.sum(line_segment > threshold)
        
        # Dotted lines have distinct bright regions
        return bright_regions > 2 and bright_regions < line_segment.size * 0.5
    
    def _detect_station_patterns(self, gray):
        """Detect station line patterns."""
        # Look for horizontal lines with station numbers
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, 
                               minLineLength=50, maxLineGap=10)
        
        station_patterns = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Check for horizontal lines (station lines are usually horizontal)
                if abs(angle) <= 10 or abs(angle - 180) <= 10:
                    # Check if line is long enough to be a station line
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > 100:  # Minimum station line length
                        station_patterns += 1
        
        return station_patterns
    
    def _detect_match_line_patterns(self, gray):
        """Detect match line patterns."""
        # Match lines are typically dashed and at edges
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, 
                               minLineLength=30, maxLineGap=8)
        
        match_patterns = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is near edge (match lines are at sheet edges)
                height, width = gray.shape
                edge_threshold = min(width, height) * 0.1
                
                if (x1 < edge_threshold or x1 > width - edge_threshold or 
                    y1 < edge_threshold or y1 > height - edge_threshold):
                    
                    # Check for dashed pattern
                    line_segment = gray[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                    if line_segment.size > 0 and self._is_dashed_pattern(line_segment):
                        match_patterns += 1
        
        return match_patterns
    
    def _analyze_border_consistency(self, gray):
        """Analyze border consistency across the image."""
        height, width = gray.shape
        
        # Check border regions
        border_regions = [
            gray[0:10, :],           # Top border
            gray[height-10:height, :], # Bottom border
            gray[:, 0:10],           # Left border
            gray[:, width-10:width]  # Right border
        ]
        
        consistency_score = 0
        
        for region in border_regions:
            if region.size > 0:
                # Calculate edge density in border region
                edges = cv2.Canny(region, 30, 100)
                edge_density = np.sum(edges > 0) / region.size
                
                # Consistent borders have similar edge density
                if edge_density > 0.1:  # Threshold for border detection
                    consistency_score += 1
        
        return consistency_score / 4.0  # Normalize to 0-1
    
    def _detect_scale_patterns(self, gray):
        """Detect scale bar patterns."""
        # Look for horizontal bars with tick marks
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        scale_patterns = 0
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum area
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Scale bars are typically horizontal and rectangular
                if w > h * 2 and w > 50:  # Wide horizontal rectangle
                    # Check for tick marks (small vertical lines)
                    roi = gray[y:y+h, x:x+w]
                    roi_edges = cv2.Canny(roi, 30, 100)
                    vertical_lines = cv2.HoughLinesP(roi_edges, rho=1, theta=np.pi/180, 
                                                   threshold=10, minLineLength=5, maxLineGap=2)
                    
                    if vertical_lines is not None and len(vertical_lines) > 2:
                        scale_patterns += 1
        
        return scale_patterns
    
    def _detect_north_arrow_patterns(self, gray):
        """Detect north arrow patterns."""
        # Look for arrow-like shapes or compass patterns
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        arrow_patterns = 0
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Minimum area
                # Get convex hull to detect arrow shapes
                hull = cv2.convexHull(contour)
                
                # Check if shape has arrow-like properties
                if len(hull) >= 5:  # Arrow has at least 5 points
                    # Calculate aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Arrows are typically taller than wide
                    if 0.5 < aspect_ratio < 2.0:
                        arrow_patterns += 1
        
        return arrow_patterns
    
    def _analyze_edge_regularity(self, edges):
        """Analyze edge regularity for engineering patterns."""
        if edges.size == 0:
            return 0
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate edge orientation consistency
        # Use Sobel operators to get gradients
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        # Calculate direction consistency
        direction_std = np.std(direction[magnitude > 0])
        regularity = 1.0 / (1.0 + direction_std)  # Higher regularity = lower std
        
        return regularity * edge_density
    
    def _find_parallel_groups(self, angles, tolerance=10):
        """Find groups of parallel lines."""
        if not angles:
            return 0
        
        # Group angles by similarity
        angle_groups = []
        used_angles = set()
        
        for i, angle1 in enumerate(angles):
            if i in used_angles:
                continue
            
            group = [i]
            used_angles.add(i)
            
            for j, angle2 in enumerate(angles):
                if j in used_angles:
                    continue
                
                # Check if angles are parallel (within tolerance)
                diff = abs(angle1 - angle2)
                if diff <= tolerance or abs(diff - 180) <= tolerance:
                    group.append(j)
                    used_angles.add(j)
            
            if len(group) > 1:
                angle_groups.append(group)
        
        return len(angle_groups)
    
    def _classify_element(self, features):
        """
        Classify element using the knowledge base and feature analysis.
        """
        classifications = []
        
        for element_type, pattern in self.knowledge_base['element_patterns'].items():
            score = self._calculate_similarity_score(features, pattern)
            classifications.append({
                'element_type': element_type,
                'score': score,
                'confidence': min(score * pattern['confidence_score'], 1.0)
            })
        
        # Sort by confidence
        classifications.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return best classification
        if classifications:
            best = classifications[0]
            return {
                'element_type': best['element_type'],
                'confidence': best['confidence'],
                'all_classifications': classifications,
                'description': self.knowledge_base['element_patterns'][best['element_type']]['description']
            }
        else:
            return {
                'element_type': 'unknown',
                'confidence': 0.0,
                'all_classifications': [],
                'description': 'Unknown element type'
            }
    
    def _calculate_similarity_score(self, features, pattern):
        """
        Calculate similarity score between features and pattern.
        """
        score = 0.0
        total_checks = 0
        
        # Check visual features
        if 'visual_features' in pattern:
            visual_pattern = pattern['visual_features']
            
            # Line thickness check
            if 'line_thickness' in visual_pattern:
                expected_min, expected_max = visual_pattern['line_thickness']
                if 'avg_line_thickness' in features:
                    thickness = features['avg_line_thickness']
                    if expected_min <= thickness <= expected_max:
                        score += 0.3
                    total_checks += 1
            
            # Line density check
            if 'line_density' in features:
                density = features['line_density']
                if density > 10:  # High line density
                    score += 0.2
                total_checks += 1
            
            # Position check
            if 'position' in visual_pattern and 'relative_position' in features:
                expected_pos = visual_pattern['position']
                actual_pos = features['relative_position']
                
                if expected_pos == 'perimeter' and 'edge' in actual_pos:
                    score += 0.3
                elif expected_pos == 'interior' and actual_pos == 'interior':
                    score += 0.3
                total_checks += 1
        
        # Check context rules
        if 'context_rules' in pattern:
            context_rules = pattern['context_rules']
            
            # Apply context-specific scoring
            if 'border' in pattern.get('description', '').lower():
                if features.get('relative_position', '') in ['edge_horizontal', 'edge_vertical']:
                    score += 0.4
                if features.get('rectangular_shapes', 0) > 0:
                    score += 0.2
                if features.get('border_consistency', 0) > 0.5:
                    score += 0.3
                total_checks += 3
            
            elif 'wall' in pattern.get('description', '').lower():
                if features.get('relative_position', '') == 'interior':
                    score += 0.4
                if features.get('horizontal_lines', 0) > 0 or features.get('vertical_lines', 0) > 0:
                    score += 0.3
                total_checks += 2
            
            elif 'conduit' in pattern.get('description', '').lower():
                if features.get('parallel_line_groups', 0) > 0:
                    score += 0.4
                if features.get('avg_line_length', 0) > 50:
                    score += 0.2
                total_checks += 2
            
            elif 'match_line' in pattern.get('description', '').lower():
                if features.get('match_line_patterns', 0) > 0:
                    score += 0.5
                if features.get('dashed_line_patterns', 0) > 0:
                    score += 0.3
                if features.get('relative_position', '') in ['edge_horizontal', 'edge_vertical']:
                    score += 0.2
                total_checks += 3
            
            elif 'scale' in pattern.get('description', '').lower():
                if features.get('scale_bar_patterns', 0) > 0:
                    score += 0.5
                if features.get('relative_position', '') in ['edge_horizontal', 'edge_vertical']:
                    score += 0.3
                total_checks += 2
            
            elif 'north_arrow' in pattern.get('description', '').lower():
                if features.get('north_arrow_patterns', 0) > 0:
                    score += 0.5
                if features.get('relative_position', '') in ['edge_horizontal', 'edge_vertical']:
                    score += 0.3
                total_checks += 2
        
        # Normalize score
        if total_checks > 0:
            score = score / total_checks
        
        return min(score, 1.0)
    
    def _store_classification_example(self, features, classification):
        """Store classification example for learning."""
        example = {
            'features': features,
            'classification': classification,
            'timestamp': datetime.now().isoformat()
        }
        
        self.knowledge_base['classification_history'].append(example)
        
        # Limit history size
        if len(self.knowledge_base['classification_history']) > 1000:
            self.knowledge_base['classification_history'] = self.knowledge_base['classification_history'][-500:]
    
    def add_feedback(self, image_path, region_bbox, correct_classification, user_confidence=1.0):
        """
        Add user feedback to improve the knowledge base.
        """
        feedback = {
            'image_path': str(image_path),
            'region_bbox': region_bbox,
            'correct_classification': correct_classification,
            'user_confidence': user_confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.knowledge_base['feedback_data'].append(feedback)
        
        # Update knowledge base based on feedback
        self._update_knowledge_base(feedback)
        
        self.logger.info(f"Added feedback for {correct_classification}")
    
    def _update_knowledge_base(self, feedback):
        """
        Update knowledge base based on user feedback.
        """
        correct_type = feedback['correct_classification']
        
        # Add to examples if element type exists
        if correct_type in self.knowledge_base['element_patterns']:
            pattern = self.knowledge_base['element_patterns'][correct_type]
            
            # Add as example
            if len(pattern['examples']) < self.max_patterns_per_element:
                pattern['examples'].append({
                    'image_path': feedback['image_path'],
                    'region_bbox': feedback['region_bbox'],
                    'user_confidence': feedback['user_confidence'],
                    'timestamp': feedback['timestamp']
                })
            
            # Adjust confidence based on feedback
            if feedback['user_confidence'] > 0.8:
                pattern['confidence_score'] = min(pattern['confidence_score'] + self.learning_rate, 1.0)
            elif feedback['user_confidence'] < 0.3:
                pattern['confidence_score'] = max(pattern['confidence_score'] - self.learning_rate, 0.1)
        
        # Create new pattern if it doesn't exist
        else:
            self.knowledge_base['element_patterns'][correct_type] = {
                'description': f'User-defined {correct_type}',
                'visual_features': {},
                'context_rules': [],
                'examples': [{
                    'image_path': feedback['image_path'],
                    'region_bbox': feedback['region_bbox'],
                    'user_confidence': feedback['user_confidence'],
                    'timestamp': feedback['timestamp']
                }],
                'confidence_score': 0.5
            }
    
    def save_knowledge_base(self, filepath="adaptive_knowledge_base.json"):
        """Save the knowledge base to file."""
        with open(filepath, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2, default=str)
        
        self.logger.info(f"Knowledge base saved to: {filepath}")
    
    def load_knowledge_base(self, filepath="adaptive_knowledge_base.json"):
        """Load the knowledge base from file."""
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                self.knowledge_base = json.load(f)
            
            self.logger.info(f"Knowledge base loaded from: {filepath}")
        else:
            self.logger.warning(f"Knowledge base file not found: {filepath}")


def main():
    """Main function to demonstrate adaptive learning."""
    print("🧠 Adaptive Learning Library Demo")
    print("=" * 50)
    
    # Initialize library
    library = AdaptiveLearningLibrary()
    
    # Load existing knowledge base if available
    library.load_knowledge_base()
    
    # Analyze some elements
    print("\n🔍 Analyzing elements...")
    sample_image = "yolo_processed_data_local/images/Asbuilts_Illumination-Power_SR509_Phase1_Stage1b_page_001.png"
    
    if Path(sample_image).exists():
        # Analyze different regions
        regions = [
            (0, 0, 100, 100),      # Top-left corner (likely border)
            (200, 200, 100, 100),  # Center region (likely content)
            (400, 400, 100, 100)   # Another region
        ]
        
        for i, region in enumerate(regions):
            result = library.analyze_element(sample_image, region)
            if result:
                print(f"Region {i+1}: {result['classification']['element_type']} "
                      f"(confidence: {result['classification']['confidence']:.3f})")
    
    # Add some feedback examples
    print("\n📝 Adding feedback examples...")
    library.add_feedback(sample_image, (0, 0, 100, 100), 'border', 0.9)
    library.add_feedback(sample_image, (200, 200, 100, 100), 'wall', 0.8)
    
    # Save updated knowledge base
    library.save_knowledge_base()
    
    print("\n✅ Adaptive learning demo complete!")
    print(f"📊 Knowledge base contains {len(library.knowledge_base['element_patterns'])} element types")
    print(f"📈 Learning history: {len(library.knowledge_base['classification_history'])} examples")
    print(f"💡 Feedback data: {len(library.knowledge_base['feedback_data'])} entries")


if __name__ == "__main__":
    main()
