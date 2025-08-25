"""
Traffic Signal & ITS Plan Reviewer
Specialized plan review system for traffic signals, ITS, and MUTCD highway signing.
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

@dataclass
class PlanElement:
    """Represents a plan element with location and metadata."""
    element_type: str  # 'signal_head', 'detector', 'sign', 'pavement_marking', etc.
    location: Tuple[float, float]  # x, y coordinates
    confidence: float
    metadata: Dict[str, Any]
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2

@dataclass
class PlanReviewResult:
    """Results of a plan review."""
    plan_type: str  # 'traffic_signal', 'its', 'mutcd_signing'
    compliance_score: float
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    elements_found: List[PlanElement]
    standards_checked: List[str]

class MUTCDReviewer:
    """MUTCD (Manual on Uniform Traffic Control Devices) compliance reviewer."""
    
    def __init__(self):
        self.mutcd_standards = {
            'sign_placement': {
                'minimum_height': 7.0,  # feet
                'maximum_height': 17.0,  # feet
                'lateral_offset': 6.0,  # feet from edge of roadway
                'spacing_requirements': {
                    'warning_signs': 250,  # feet minimum
                    'regulatory_signs': 100,  # feet minimum
                    'guide_signs': 300  # feet minimum
                }
            },
            'sign_types': {
                'regulatory': ['stop', 'yield', 'speed_limit', 'no_parking', 'one_way'],
                'warning': ['curve', 'intersection', 'pedestrian', 'school', 'work_zone'],
                'guide': ['street_name', 'route_marker', 'destination', 'exit']
            },
            'pavement_markings': {
                'lane_lines': ['solid_yellow', 'broken_white', 'double_yellow'],
                'stop_lines': ['solid_white', 'width_12_inches'],
                'crosswalks': ['parallel_lines', 'continental', 'zebra']
            }
        }
        
        # MUTCD sign recognition patterns
        self.sign_patterns = {
            'stop_sign': {
                'shape': 'octagon',
                'color': 'red',
                'text': 'STOP',
                'size': {'width': 30, 'height': 30}  # inches
            },
            'yield_sign': {
                'shape': 'triangle',
                'color': 'red_white',
                'text': 'YIELD',
                'size': {'width': 36, 'height': 36}
            },
            'speed_limit': {
                'shape': 'rectangle',
                'color': 'white_black',
                'pattern': r'SPEED\s+LIMIT\s+\d+',
                'size': {'width': 24, 'height': 30}
            }
        }
    
    def review_signing_plan(self, plan_image: np.ndarray) -> PlanReviewResult:
        """Review MUTCD signing plan for compliance."""
        issues = []
        recommendations = []
        elements_found = []
        
        # Detect signs in the plan
        signs = self._detect_signs(plan_image)
        elements_found.extend(signs)
        
        # Check sign placement compliance
        placement_issues = self._check_sign_placement(signs)
        issues.extend(placement_issues)
        
        # Check sign spacing
        spacing_issues = self._check_sign_spacing(signs)
        issues.extend(spacing_issues)
        
        # Check pavement markings
        markings = self._detect_pavement_markings(plan_image)
        elements_found.extend(markings)
        
        marking_issues = self._check_pavement_markings(markings)
        issues.extend(marking_issues)
        
        # Generate recommendations
        recommendations = self._generate_mutcd_recommendations(issues)
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(issues, len(elements_found))
        
        return PlanReviewResult(
            plan_type='mutcd_signing',
            compliance_score=compliance_score,
            issues=issues,
            recommendations=recommendations,
            elements_found=elements_found,
            standards_checked=['MUTCD 2009', 'MUTCD 2023']
        )
    
    def _detect_signs(self, image: np.ndarray) -> List[PlanElement]:
        """Detect traffic signs in the plan image."""
        signs = []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Classify shape
            shape = self._classify_shape(len(approx))
            
            if shape in ['octagon', 'triangle', 'rectangle', 'diamond']:
                # This could be a sign
                sign_type = self._classify_sign_type(shape, image[y:y+h, x:x+w])
                
                if sign_type:
                    signs.append(PlanElement(
                        element_type=f'sign_{sign_type}',
                        location=(x + w/2, y + h/2),
                        confidence=0.8,
                        metadata={
                            'shape': shape,
                            'size': (w, h),
                            'sign_type': sign_type
                        },
                        bounding_box=(x, y, x+w, y+h)
                    ))
        
        return signs
    
    def _classify_shape(self, vertices: int) -> str:
        """Classify shape based on number of vertices."""
        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            return 'rectangle'
        elif vertices == 8:
            return 'octagon'
        elif vertices == 4:  # Diamond is also 4 vertices
            return 'diamond'
        else:
            return 'unknown'
    
    def _classify_sign_type(self, shape: str, sign_image: np.ndarray) -> Optional[str]:
        """Classify the type of sign based on shape and content."""
        if shape == 'octagon':
            return 'stop'
        elif shape == 'triangle':
            return 'yield'
        elif shape == 'diamond':
            return 'warning'
        elif shape == 'rectangle':
            # Could be speed limit, guide sign, etc.
            return 'regulatory'
        return None
    
    def _check_sign_placement(self, signs: List[PlanElement]) -> List[Dict[str, Any]]:
        """Check sign placement for MUTCD compliance."""
        issues = []
        
        for sign in signs:
            x, y = sign.location
            
            # Check height (assuming y-coordinate represents height)
            if y < self.mutcd_standards['sign_placement']['minimum_height']:
                issues.append({
                    'type': 'placement_issue',
                    'element': sign.element_type,
                    'issue': 'Sign height below MUTCD minimum',
                    'severity': 'high',
                    'standard': 'MUTCD Section 2A.18',
                    'recommendation': f'Increase sign height to minimum {self.mutcd_standards["sign_placement"]["minimum_height"]} feet'
                })
            
            # Check lateral offset
            if x < self.mutcd_standards['sign_placement']['lateral_offset']:
                issues.append({
                    'type': 'placement_issue',
                    'element': sign.element_type,
                    'issue': 'Sign too close to roadway edge',
                    'severity': 'medium',
                    'standard': 'MUTCD Section 2A.19',
                    'recommendation': f'Move sign to minimum {self.mutcd_standards["sign_placement"]["lateral_offset"]} feet from edge'
                })
        
        return issues
    
    def _check_sign_spacing(self, signs: List[PlanElement]) -> List[Dict[str, Any]]:
        """Check sign spacing for compliance."""
        issues = []
        
        # Group signs by type
        sign_groups = {}
        for sign in signs:
            sign_type = sign.metadata.get('sign_type', 'unknown')
            if sign_type not in sign_groups:
                sign_groups[sign_type] = []
            sign_groups[sign_type].append(sign)
        
        # Check spacing within each group
        for sign_type, group_signs in sign_groups.items():
            if len(group_signs) < 2:
                continue
            
            # Sort by x-coordinate (assuming left-to-right reading)
            group_signs.sort(key=lambda s: s.location[0])
            
            for i in range(len(group_signs) - 1):
                sign1 = group_signs[i]
                sign2 = group_signs[i + 1]
                
                distance = abs(sign2.location[0] - sign1.location[0])
                min_spacing = self.mutcd_standards['sign_placement']['spacing_requirements'].get(sign_type, 100)
                
                if distance < min_spacing:
                    issues.append({
                        'type': 'spacing_issue',
                        'element': f'{sign1.element_type} and {sign2.element_type}',
                        'issue': f'Insufficient spacing between {sign_type} signs',
                        'severity': 'medium',
                        'standard': 'MUTCD Section 2A.20',
                        'recommendation': f'Increase spacing to minimum {min_spacing} feet'
                    })
        
        return issues
    
    def _detect_pavement_markings(self, image: np.ndarray) -> List[PlanElement]:
        """Detect pavement markings in the plan."""
        markings = []
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect white markings
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Detect yellow markings
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find contours for white markings
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in white_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Classify marking type based on shape and size
            marking_type = self._classify_marking_type(w, h, 'white')
            
            if marking_type:
                markings.append(PlanElement(
                    element_type=f'marking_{marking_type}',
                    location=(x + w/2, y + h/2),
                    confidence=0.7,
                    metadata={
                        'color': 'white',
                        'size': (w, h),
                        'marking_type': marking_type
                    },
                    bounding_box=(x, y, x+w, y+h)
                ))
        
        return markings
    
    def _classify_marking_type(self, width: int, height: int, color: str) -> Optional[str]:
        """Classify pavement marking type based on dimensions and color."""
        aspect_ratio = width / height if height > 0 else 0
        
        if color == 'white':
            if aspect_ratio > 10:  # Very long and thin
                return 'lane_line'
            elif aspect_ratio > 5:  # Long and thin
                return 'stop_line'
            else:
                return 'crosswalk'
        elif color == 'yellow':
            if aspect_ratio > 10:
                return 'center_line'
            else:
                return 'edge_line'
        
        return None
    
    def _check_pavement_markings(self, markings: List[PlanElement]) -> List[Dict[str, Any]]:
        """Check pavement markings for MUTCD compliance."""
        issues = []
        
        for marking in markings:
            marking_type = marking.metadata.get('marking_type', '')
            
            if marking_type == 'stop_line':
                # Check stop line width (should be 12-24 inches)
                width = marking.metadata['size'][0]
                if width < 12 or width > 24:
                    issues.append({
                        'type': 'marking_issue',
                        'element': marking.element_type,
                        'issue': 'Stop line width outside MUTCD specifications',
                        'severity': 'medium',
                        'standard': 'MUTCD Section 3B.16',
                        'recommendation': 'Adjust stop line width to 12-24 inches'
                    })
        
        return issues
    
    def _generate_mutcd_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate MUTCD-specific recommendations."""
        recommendations = []
        
        # Group issues by type
        placement_issues = [i for i in issues if i['type'] == 'placement_issue']
        spacing_issues = [i for i in issues if i['type'] == 'spacing_issue']
        marking_issues = [i for i in issues if i['type'] == 'marking_issue']
        
        if placement_issues:
            recommendations.append("Review sign placement for MUTCD height and offset requirements")
        
        if spacing_issues:
            recommendations.append("Verify sign spacing meets MUTCD minimum distance requirements")
        
        if marking_issues:
            recommendations.append("Check pavement marking dimensions and placement")
        
        if not issues:
            recommendations.append("Plan appears to meet MUTCD requirements")
        
        return recommendations
    
    def _calculate_compliance_score(self, issues: List[Dict[str, Any]], total_elements: int) -> float:
        """Calculate MUTCD compliance score."""
        if total_elements == 0:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        total_penalty = 0
        for issue in issues:
            severity = issue.get('severity', 'medium')
            weight = severity_weights.get(severity, 0.5)
            total_penalty += weight
        
        # Normalize by number of elements
        normalized_penalty = total_penalty / total_elements
        
        # Convert to compliance score (0-1, where 1 is fully compliant)
        compliance_score = max(0, 1 - normalized_penalty)
        
        return compliance_score

class TrafficSignalReviewer:
    """Traffic signal plan reviewer."""
    
    def __init__(self):
        self.signal_standards = {
            'signal_head_placement': {
                'minimum_height': 15.0,  # feet
                'maximum_height': 18.0,  # feet
                'visibility_angle': 45,  # degrees
                'spacing': 40  # feet between heads
            },
            'detector_placement': {
                'stop_bar_distance': 20,  # feet from stop bar
                'spacing': 30,  # feet between detectors
                'coverage_length': 40  # feet detection zone
            },
            'pedestrian_features': {
                'push_button_location': 5,  # feet from crosswalk
                'accessible_route': True,
                'pedestrian_signal': True
            }
        }
        
        self.signal_elements = {
            'signal_heads': ['red', 'yellow', 'green', 'left_turn', 'right_turn'],
            'detectors': ['loop', 'video', 'microwave', 'radar'],
            'pedestrian': ['push_button', 'pedestrian_signal', 'accessible_features'],
            'controller': ['cabinet', 'controller', 'conflict_monitor']
        }
    
    def review_signal_plan(self, plan_image: np.ndarray) -> PlanReviewResult:
        """Review traffic signal plan for compliance."""
        issues = []
        recommendations = []
        elements_found = []
        
        # Detect signal elements
        signal_heads = self._detect_signal_heads(plan_image)
        elements_found.extend(signal_heads)
        
        detectors = self._detect_detectors(plan_image)
        elements_found.extend(detectors)
        
        pedestrian_features = self._detect_pedestrian_features(plan_image)
        elements_found.extend(pedestrian_features)
        
        # Check signal head placement
        head_issues = self._check_signal_head_placement(signal_heads)
        issues.extend(head_issues)
        
        # Check detector placement
        detector_issues = self._check_detector_placement(detectors)
        issues.extend(detector_issues)
        
        # Check pedestrian accessibility
        ped_issues = self._check_pedestrian_accessibility(pedestrian_features)
        issues.extend(ped_issues)
        
        # Generate recommendations
        recommendations = self._generate_signal_recommendations(issues)
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(issues, len(elements_found))
        
        return PlanReviewResult(
            plan_type='traffic_signal',
            compliance_score=compliance_score,
            issues=issues,
            recommendations=recommendations,
            elements_found=elements_found,
            standards_checked=['ITE Signal Timing Manual', 'MUTCD Part 4', 'AASHTO Green Book']
        )
    
    def _detect_signal_heads(self, image: np.ndarray) -> List[PlanElement]:
        """Detect traffic signal heads in the plan."""
        signal_heads = []
        
        # Look for circular objects (signal heads)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Hough Circle detection
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        if circles is not None:
            for circle in circles[0]:
                x, y, radius = circle
                
                # Classify signal head type based on color
                head_type = self._classify_signal_head(image, int(x), int(y), int(radius))
                
                signal_heads.append(PlanElement(
                    element_type=f'signal_head_{head_type}',
                    location=(x, y),
                    confidence=0.8,
                    metadata={
                        'radius': radius,
                        'head_type': head_type
                    },
                    bounding_box=(int(x-radius), int(y-radius), int(x+radius), int(y+radius))
                ))
        
        return signal_heads
    
    def _classify_signal_head(self, image: np.ndarray, x: int, y: int, radius: int) -> str:
        """Classify signal head type based on color analysis."""
        # Extract region around signal head
        x1, y1 = max(0, x-radius), max(0, y-radius)
        x2, y2 = min(image.shape[1], x+radius), min(image.shape[0], y+radius)
        
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return 'unknown'
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Check for red, yellow, green
        red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
        
        # Count pixels of each color
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Determine dominant color
        total_pixels = region.shape[0] * region.shape[1]
        if red_pixels > total_pixels * 0.1:
            return 'red'
        elif yellow_pixels > total_pixels * 0.1:
            return 'yellow'
        elif green_pixels > total_pixels * 0.1:
            return 'green'
        else:
            return 'unknown'
    
    def _detect_detectors(self, image: np.ndarray) -> List[PlanElement]:
        """Detect vehicle detectors in the plan."""
        detectors = []
        
        # Look for rectangular objects (detector loops)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this looks like a detector (rectangular, reasonable size)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            if 0.5 < aspect_ratio < 2.0 and 100 < area < 5000:
                detectors.append(PlanElement(
                    element_type='detector_loop',
                    location=(x + w/2, y + h/2),
                    confidence=0.7,
                    metadata={
                        'size': (w, h),
                        'detector_type': 'loop'
                    },
                    bounding_box=(x, y, x+w, y+h)
                ))
        
        return detectors
    
    def _detect_pedestrian_features(self, image: np.ndarray) -> List[PlanElement]:
        """Detect pedestrian features in the plan."""
        features = []
        
        # Look for small circular objects (push buttons)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
            param1=50, param2=25, minRadius=2, maxRadius=15
        )
        
        if circles is not None:
            for circle in circles[0]:
                x, y, radius = circle
                
                features.append(PlanElement(
                    element_type='pedestrian_push_button',
                    location=(x, y),
                    confidence=0.6,
                    metadata={
                        'radius': radius,
                        'feature_type': 'push_button'
                    },
                    bounding_box=(int(x-radius), int(y-radius), int(x+radius), int(y+radius))
                ))
        
        return features
    
    def _check_signal_head_placement(self, signal_heads: List[PlanElement]) -> List[Dict[str, Any]]:
        """Check signal head placement for compliance."""
        issues = []
        
        for head in signal_heads:
            x, y = head.location
            
            # Check height (assuming y-coordinate represents height)
            if y < self.signal_standards['signal_head_placement']['minimum_height']:
                issues.append({
                    'type': 'placement_issue',
                    'element': head.element_type,
                    'issue': 'Signal head height below minimum requirement',
                    'severity': 'high',
                    'standard': 'ITE Signal Timing Manual',
                    'recommendation': f'Increase signal head height to minimum {self.signal_standards["signal_head_placement"]["minimum_height"]} feet'
                })
        
        return issues
    
    def _check_detector_placement(self, detectors: List[PlanElement]) -> List[Dict[str, Any]]:
        """Check detector placement for compliance."""
        issues = []
        
        for detector in detectors:
            x, y = detector.location
            
            # Check distance from stop bar
            if x < self.signal_standards['detector_placement']['stop_bar_distance']:
                issues.append({
                    'type': 'placement_issue',
                    'element': detector.element_type,
                    'issue': 'Detector too close to stop bar',
                    'severity': 'medium',
                    'standard': 'ITE Signal Timing Manual',
                    'recommendation': f'Move detector to minimum {self.signal_standards["detector_placement"]["stop_bar_distance"]} feet from stop bar'
                })
        
        return issues
    
    def _check_pedestrian_accessibility(self, features: List[PlanElement]) -> List[Dict[str, Any]]:
        """Check pedestrian accessibility features."""
        issues = []
        
        push_buttons = [f for f in features if f.element_type == 'pedestrian_push_button']
        
        if not push_buttons:
            issues.append({
                'type': 'accessibility_issue',
                'element': 'pedestrian_features',
                'issue': 'No pedestrian push buttons detected',
                'severity': 'high',
                'standard': 'ADA Standards',
                'recommendation': 'Add pedestrian push buttons for accessibility'
            })
        
        return issues
    
    def _generate_signal_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate traffic signal-specific recommendations."""
        recommendations = []
        
        placement_issues = [i for i in issues if i['type'] == 'placement_issue']
        accessibility_issues = [i for i in issues if i['type'] == 'accessibility_issue']
        
        if placement_issues:
            recommendations.append("Review signal head and detector placement for ITE standards")
        
        if accessibility_issues:
            recommendations.append("Verify pedestrian accessibility features meet ADA requirements")
        
        if not issues:
            recommendations.append("Traffic signal plan appears to meet standards")
        
        return recommendations
    
    def _calculate_compliance_score(self, issues: List[Dict[str, Any]], total_elements: int) -> float:
        """Calculate traffic signal compliance score."""
        if total_elements == 0:
            return 1.0
        
        severity_weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        total_penalty = 0
        for issue in issues:
            severity = issue.get('severity', 'medium')
            weight = severity_weights.get(severity, 0.5)
            total_penalty += weight
        
        normalized_penalty = total_penalty / total_elements
        compliance_score = max(0, 1 - normalized_penalty)
        
        return compliance_score

class ITSReviewer:
    """ITS (Intelligent Transportation Systems) plan reviewer."""
    
    def __init__(self):
        self.its_standards = {
            'communications': {
                'fiber_connectivity': True,
                'wireless_backup': True,
                'redundancy': True
            },
            'sensors': {
                'camera_coverage': 0.8,  # 80% coverage
                'radar_coverage': 0.6,   # 60% coverage
                'weather_sensors': True
            },
            'data_management': {
                'storage_capacity': 'adequate',
                'backup_systems': True,
                'cybersecurity': True
            }
        }
        
        self.its_elements = {
            'cameras': ['cctv', 'ptz', 'fixed'],
            'sensors': ['radar', 'microwave', 'infrared', 'weather'],
            'communications': ['fiber', 'wireless', 'cellular'],
            'displays': ['vms', 'dms', 'har']
        }
    
    def review_its_plan(self, plan_image: np.ndarray) -> PlanReviewResult:
        """Review ITS plan for compliance."""
        issues = []
        recommendations = []
        elements_found = []
        
        # Detect ITS elements
        cameras = self._detect_cameras(plan_image)
        elements_found.extend(cameras)
        
        sensors = self._detect_sensors(plan_image)
        elements_found.extend(sensors)
        
        communications = self._detect_communications(plan_image)
        elements_found.extend(communications)
        
        # Check coverage
        coverage_issues = self._check_coverage(cameras, sensors)
        issues.extend(coverage_issues)
        
        # Check redundancy
        redundancy_issues = self._check_redundancy(communications)
        issues.extend(redundancy_issues)
        
        # Generate recommendations
        recommendations = self._generate_its_recommendations(issues)
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(issues, len(elements_found))
        
        return PlanReviewResult(
            plan_type='its',
            compliance_score=compliance_score,
            issues=issues,
            recommendations=recommendations,
            elements_found=elements_found,
            standards_checked=['NTCIP', 'ITE ITS Standards', 'AASHTO ITS Guide']
        )
    
    def _detect_cameras(self, image: np.ndarray) -> List[PlanElement]:
        """Detect CCTV cameras in the plan."""
        cameras = []
        
        # Look for camera symbols (typically small rectangles or circles)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this looks like a camera (small, square-ish)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            if 0.5 < aspect_ratio < 2.0 and 50 < area < 1000:
                cameras.append(PlanElement(
                    element_type='cctv_camera',
                    location=(x + w/2, y + h/2),
                    confidence=0.7,
                    metadata={
                        'size': (w, h),
                        'camera_type': 'fixed'
                    },
                    bounding_box=(x, y, x+w, y+h)
                ))
        
        return cameras
    
    def _detect_sensors(self, image: np.ndarray) -> List[PlanElement]:
        """Detect ITS sensors in the plan."""
        sensors = []
        
        # Look for sensor symbols (typically small circles or diamonds)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
            param1=50, param2=25, minRadius=3, maxRadius=20
        )
        
        if circles is not None:
            for circle in circles[0]:
                x, y, radius = circle
                
                sensors.append(PlanElement(
                    element_type='its_sensor',
                    location=(x, y),
                    confidence=0.6,
                    metadata={
                        'radius': radius,
                        'sensor_type': 'radar'
                    },
                    bounding_box=(int(x-radius), int(y-radius), int(x+radius), int(y+radius))
                ))
        
        return sensors
    
    def _detect_communications(self, image: np.ndarray) -> List[PlanElement]:
        """Detect communication infrastructure in the plan."""
        communications = []
        
        # Look for communication symbols (typically lines or nodes)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Line detection
        lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                communications.append(PlanElement(
                    element_type='communication_line',
                    location=((x1+x2)/2, (y1+y2)/2),
                    confidence=0.5,
                    metadata={
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'comm_type': 'fiber'
                    },
                    bounding_box=(min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
                ))
        
        return communications
    
    def _check_coverage(self, cameras: List[PlanElement], sensors: List[PlanElement]) -> List[Dict[str, Any]]:
        """Check coverage of ITS elements."""
        issues = []
        
        # Calculate coverage based on number of elements
        total_cameras = len(cameras)
        total_sensors = len(sensors)
        
        if total_cameras < 3:  # Assuming minimum cameras needed
            issues.append({
                'type': 'coverage_issue',
                'element': 'cctv_cameras',
                'issue': 'Insufficient CCTV camera coverage',
                'severity': 'medium',
                'standard': 'ITE ITS Standards',
                'recommendation': 'Add additional CCTV cameras for adequate coverage'
            })
        
        if total_sensors < 2:  # Assuming minimum sensors needed
            issues.append({
                'type': 'coverage_issue',
                'element': 'its_sensors',
                'issue': 'Insufficient sensor coverage',
                'severity': 'medium',
                'standard': 'ITE ITS Standards',
                'recommendation': 'Add additional sensors for adequate coverage'
            })
        
        return issues
    
    def _check_redundancy(self, communications: List[PlanElement]) -> List[Dict[str, Any]]:
        """Check communication redundancy."""
        issues = []
        
        # Check for multiple communication paths
        if len(communications) < 2:
            issues.append({
                'type': 'redundancy_issue',
                'element': 'communications',
                'issue': 'Insufficient communication redundancy',
                'severity': 'high',
                'standard': 'NTCIP Standards',
                'recommendation': 'Add redundant communication paths for reliability'
            })
        
        return issues
    
    def _generate_its_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate ITS-specific recommendations."""
        recommendations = []
        
        coverage_issues = [i for i in issues if i['type'] == 'coverage_issue']
        redundancy_issues = [i for i in issues if i['type'] == 'redundancy_issue']
        
        if coverage_issues:
            recommendations.append("Review ITS element coverage for adequate system monitoring")
        
        if redundancy_issues:
            recommendations.append("Verify communication redundancy for system reliability")
        
        if not issues:
            recommendations.append("ITS plan appears to meet standards")
        
        return recommendations
    
    def _calculate_compliance_score(self, issues: List[Dict[str, Any]], total_elements: int) -> float:
        """Calculate ITS compliance score."""
        if total_elements == 0:
            return 1.0
        
        severity_weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        total_penalty = 0
        for issue in issues:
            severity = issue.get('severity', 'medium')
            weight = severity_weights.get(severity, 0.5)
            total_penalty += weight
        
        normalized_penalty = total_penalty / total_elements
        compliance_score = max(0, 1 - normalized_penalty)
        
        return compliance_score

class TrafficPlanReviewer:
    """Main traffic plan reviewer that orchestrates all review types."""
    
    def __init__(self):
        self.mutcd_reviewer = MUTCDReviewer()
        self.signal_reviewer = TrafficSignalReviewer()
        self.its_reviewer = ITSReviewer()
    
    def review_plan(self, plan_path: str, plan_type: str = 'auto') -> PlanReviewResult:
        """Review a traffic plan based on its type."""
        
        # Load plan image
        if plan_path.endswith('.pdf'):
            # Handle PDF plans
            doc = fitz.open(plan_path)
            page = doc.load_page(0)
            pix = page.get_pixmap()
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
        else:
            # Handle image files
            img_array = cv2.imread(plan_path)
        
        if img_array is None:
            raise ValueError(f"Could not load plan from {plan_path}")
        
        # Auto-detect plan type if not specified
        if plan_type == 'auto':
            plan_type = self._detect_plan_type(img_array)
        
        # Perform appropriate review
        if plan_type == 'mutcd_signing':
            return self.mutcd_reviewer.review_signing_plan(img_array)
        elif plan_type == 'traffic_signal':
            return self.signal_reviewer.review_signal_plan(img_array)
        elif plan_type == 'its':
            return self.its_reviewer.review_its_plan(img_array)
        else:
            raise ValueError(f"Unknown plan type: {plan_type}")
    
    def _detect_plan_type(self, image: np.ndarray) -> str:
        """Auto-detect the type of traffic plan."""
        # Simple heuristic-based detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Count potential signal heads (circles)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        signal_count = len(circles[0]) if circles is not None else 0
        
        # Count potential signs (polygons)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sign_count = 0
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) in [3, 4, 8]:  # Triangle, rectangle, octagon
                sign_count += 1
        
        # Determine plan type based on element counts
        if signal_count > 2:
            return 'traffic_signal'
        elif sign_count > 5:
            return 'mutcd_signing'
        else:
            return 'its'  # Default to ITS if unclear

def main():
    """Example usage of the Traffic Plan Reviewer."""
    reviewer = TrafficPlanReviewer()
    
    # Example: Review a traffic signal plan
    try:
        result = reviewer.review_plan("traffic_signal_plan.pdf", "traffic_signal")
        
        print(f"Traffic Plan Review Results")
        print(f"Plan Type: {result.plan_type}")
        print(f"Compliance Score: {result.compliance_score:.2f}")
        print(f"Standards Checked: {', '.join(result.standards_checked)}")
        
        print(f"\nIssues Found ({len(result.issues)}):")
        for issue in result.issues:
            print(f"- {issue['issue']} ({issue['severity']})")
        
        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"- {rec}")
        
        print(f"\nElements Detected ({len(result.elements_found)}):")
        for element in result.elements_found:
            print(f"- {element.element_type} at {element.location}")
    
    except Exception as e:
        print(f"Error reviewing plan: {e}")

if __name__ == "__main__":
    main()