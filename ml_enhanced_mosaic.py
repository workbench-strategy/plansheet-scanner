#!/usr/bin/env python3
"""
ML-Enhanced Mosaic System
Integrates trained ML model to improve north arrow detection, sheet classification, and layout optimization.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from dataclasses import dataclass
import cv2
import io
import math
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from improved_ai_trainer import ImprovedAIEngineerTrainer
from plan_review_and_tagging import PlanReviewer

def setup_logging():
    """Set up detailed logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ml_enhanced_mosaic.log'),
            logging.StreamHandler()
        ]
    )

@dataclass
class MLEnhancedSheet:
    """Enhanced sheet information with ML analysis."""
    page: fitz.Page
    page_index: int
    width_in: float
    height_in: float
    sheet_name: Optional[str] = None
    page_number: Optional[str] = None
    north_rotation: float = 0.0
    ml_discipline: Optional[str] = None
    ml_confidence: float = 0.0
    ml_issues: List[Dict] = None
    matchlines: Dict = None
    position: Tuple[float, float] = (0.0, 0.0)
    is_placed: bool = False
    
    def __post_init__(self):
        if self.ml_issues is None:
            self.ml_issues = []
        if self.matchlines is None:
            self.matchlines = {}

class MLEnhancedMosaicSystem:
    """ML-enhanced mosaic system with improved detection and layout."""
    
    def __init__(self, template_path: str = "templates/YetAnotherNorth/newnorth.png"):
        self.template_path = template_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML components
        self.logger.info("Initializing ML components...")
        self.ml_trainer = ImprovedAIEngineerTrainer()
        self.plan_reviewer = PlanReviewer()
        
        # Load north arrow template
        if os.path.exists(template_path):
            self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            self.logger.info(f"Loaded north arrow template: {self.template.shape}")
        else:
            self.template = None
            self.logger.warning(f"North arrow template not found: {template_path}")
    
    def load_sheets(self, pdf_path: str, max_sheets: int = 10) -> List[MLEnhancedSheet]:
        """Load and analyze sheets with ML enhancement."""
        self.logger.info(f"Loading up to {max_sheets} sheets from {pdf_path}")
        
        doc = fitz.open(pdf_path)
        sheets = []
        
        for i in range(min(len(doc), max_sheets)):
            page = doc[i]
            rect = page.rect
            width_in = rect.width / 72.0
            height_in = rect.height / 72.0
            
            sheet = MLEnhancedSheet(
                page=page,
                page_index=i,
                width_in=width_in,
                height_in=height_in
            )
            
            # Extract basic information
            self._extract_sheet_info(sheet)
            
            # Perform ML analysis
            self._analyze_sheet_with_ml(sheet)
            
            # Detect north arrow and rotation
            self._detect_north_arrow(sheet)
            
            sheets.append(sheet)
            self.logger.info(f"Loaded sheet {i+1}: {sheet.sheet_name} ({width_in:.1f}\" x {height_in:.1f}\")")
        
        doc.close()
        return sheets
    
    def _extract_sheet_info(self, sheet: MLEnhancedSheet):
        """Extract basic sheet information."""
        try:
            # Extract text from page
            text = sheet.page.get_text()
            
            # Look for sheet name patterns
            import re
            
            # Common sheet name patterns
            patterns = [
                r'Sheet\s*([A-Z]+\d*)',  # Sheet T-01, Sheet E-02
                r'([A-Z]+\d*)\s*-\s*\d+',  # T-01 - 1, E-02 - 2
                r'([A-Z]+)\s*(\d+)',  # T 01, E 02
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    sheet.sheet_name = match.group(0).strip()
                    break
            
            # Look for page numbers
            page_patterns = [
                r'Page\s*(\d+)',
                r'Sheet\s*\d+\s*of\s*(\d+)',
                r'(\d+)\s*of\s*\d+',
            ]
            
            for pattern in page_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    sheet.page_number = match.group(1)
                    break
                    
        except Exception as e:
            self.logger.warning(f"Error extracting sheet info: {e}")
    
    def _analyze_sheet_with_ml(self, sheet: MLEnhancedSheet):
        """Analyze sheet using trained ML model."""
        try:
            # Extract text content
            text_content = sheet.page.get_text()
            
            # Create plan data for ML analysis
            plan_data = {
                "sheet_title": sheet.sheet_name or f"Sheet {sheet.page_index}",
                "discipline": "unknown",
                "project_name": "Unknown Project",
                "construction_notes": text_content[:500],  # First 500 chars
                "as_built_changes": []
            }
            
            # Perform ML review
            review_result = self.plan_reviewer.review_plan_comprehensive(plan_data)
            
            # Extract ML results
            ml_analysis = review_result['ml_analysis']
            sheet.ml_discipline = ml_analysis['predicted_discipline']
            sheet.ml_confidence = ml_analysis['overall_confidence']
            sheet.ml_issues = review_result['issues']
            
            self.logger.info(f"ML Analysis for {sheet.sheet_name}: {sheet.ml_discipline} (conf: {sheet.ml_confidence:.3f})")
            
        except Exception as e:
            self.logger.warning(f"Error in ML analysis: {e}")
            sheet.ml_discipline = "unknown"
            sheet.ml_confidence = 0.0
    
    def _detect_north_arrow(self, sheet: MLEnhancedSheet):
        """Detect north arrow and determine rotation needed."""
        if self.template is None:
            self.logger.warning("No north arrow template available")
            return
        
        try:
            # Convert page to image
            pix = sheet.page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Convert to grayscale
            gray_image = pil_image.convert('L')
            image_array = np.array(gray_image)
            
            # Detect north arrow with template rotation
            result = self._detect_north_arrow_with_rotation(image_array)
            
            if result.detected:
                sheet.north_rotation = result.angle
                self.logger.info(f"North arrow detected on {sheet.sheet_name}: rotation {result.angle:.1f}° (conf: {result.confidence:.3f})")
            else:
                self.logger.debug(f"No north arrow detected on {sheet.sheet_name}")
                
        except Exception as e:
            self.logger.warning(f"Error detecting north arrow: {e}")
    
    def _detect_north_arrow_with_rotation(self, image: np.ndarray, rotation_step: float = 5.0) -> 'NorthDetectionResult':
        """Detect north arrow by rotating template and finding best match."""
        if self.template is None:
            return NorthDetectionResult(False, 0.0, 0.0, 0.0, (0, 0))
        
        best_confidence = 0.0
        best_angle = 0.0
        best_position = (0, 0)
        
        # Try different template rotations
        for angle in np.arange(0, 360, rotation_step):
            # Rotate template
            rotated_template = self._rotate_template(self.template, angle)
            
            # Template matching
            result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                best_angle = angle
                best_position = max_loc
        
        # Determine if detection was successful
        threshold = 0.3
        detected = best_confidence >= threshold
        
        if detected:
            # Calculate rotation needed to align north to top
            rotation_needed = -best_angle
            
            return NorthDetectionResult(
                detected=True,
                angle=rotation_needed,
                confidence=best_confidence,
                template_angle=best_angle,
                position=best_position
            )
        else:
            return NorthDetectionResult(False, 0.0, best_confidence, 0.0, (0, 0))
    
    def _rotate_template(self, template: np.ndarray, angle: float) -> np.ndarray:
        """Rotate template by given angle."""
        height, width = template.shape
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image size
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for new size
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(template, rotation_matrix, (new_width, new_height))
        
        return rotated
    
    def create_connectivity_map(self, sheets: List[MLEnhancedSheet]) -> Dict:
        """Create connectivity map based on sheet names and ML analysis."""
        self.logger.info("Creating connectivity map...")
        
        connectivity = {}
        
        for i, sheet in enumerate(sheets):
            sheet_name = sheet.sheet_name or f"Sheet_{i}"
            connectivity[sheet_name] = {
                'sheet_index': i,
                'discipline': sheet.ml_discipline,
                'confidence': sheet.ml_confidence,
                'targets': {},
                'targeted_by': set(),
                'issues': len(sheet.ml_issues)
            }
        
        # Find connections based on sheet naming patterns
        for i, sheet in enumerate(sheets):
            sheet_name = sheet.sheet_name
            if not sheet_name:
                continue
            
            # Look for related sheets (same discipline, sequential numbers, etc.)
            for j, other_sheet in enumerate(sheets):
                if i == j:
                    continue
                
                other_name = other_sheet.sheet_name
                if not other_name:
                    continue
                
                # Check if sheets are related
                if self._are_sheets_related(sheet, other_sheet):
                    # Determine connection direction based on sheet names
                    connection = self._determine_connection_direction(sheet_name, other_name)
                    if connection:
                        edge, direction = connection
                        connectivity[sheet_name]['targets'][edge] = [other_name]
                        connectivity[other_name]['targeted_by'].add(sheet_name)
        
        return connectivity
    
    def _are_sheets_related(self, sheet1: MLEnhancedSheet, sheet2: MLEnhancedSheet) -> bool:
        """Determine if two sheets are related."""
        # Same discipline
        if sheet1.ml_discipline == sheet2.ml_discipline and sheet1.ml_discipline != "unknown":
            return True
        
        # Sequential sheet numbers
        if sheet1.sheet_name and sheet2.sheet_name:
            import re
            num1 = re.search(r'\d+', sheet1.sheet_name)
            num2 = re.search(r'\d+', sheet2.sheet_name)
            if num1 and num2:
                try:
                    n1 = int(num1.group())
                    n2 = int(num2.group())
                    if abs(n1 - n2) == 1:  # Sequential
                        return True
                except ValueError:
                    pass
        
        return False
    
    def _determine_connection_direction(self, name1: str, name2: str) -> Optional[Tuple[str, str]]:
        """Determine connection direction between two sheets."""
        import re
        
        # Extract numbers from sheet names
        num1 = re.search(r'\d+', name1)
        num2 = re.search(r'\d+', name2)
        
        if num1 and num2:
            try:
                n1 = int(num1.group())
                n2 = int(num2.group())
                
                if n1 < n2:
                    return ('right', 'left')  # name1 -> name2
                elif n1 > n2:
                    return ('left', 'right')  # name2 -> name1
            except ValueError:
                pass
        
        return None
    
    def optimize_layout(self, sheets: List[MLEnhancedSheet], connectivity: Dict) -> Dict[int, Tuple[float, float]]:
        """Optimize sheet layout using ML insights and connectivity."""
        self.logger.info("Optimizing layout with ML insights...")
        
        positions = {}
        placed = set()
        
        # Start with sheet that has highest ML confidence and most connections
        best_start = self._find_best_starting_sheet(sheets, connectivity)
        if best_start is not None:
            positions[best_start] = (0.0, 0.0)
            placed.add(best_start)
            self.logger.info(f"Starting with sheet {best_start}: {sheets[best_start].sheet_name}")
        
        # Place remaining sheets
        while len(placed) < len(sheets):
            placed_count = len(placed)
            
            for i, sheet in enumerate(sheets):
                if i in placed:
                    continue
                
                # Try to place sheet relative to already placed sheets
                position = self._find_best_position(sheet, sheets, connectivity, positions, placed)
                if position is not None:
                    positions[i] = position
                    placed.add(i)
                    self.logger.info(f"Placed sheet {i}: {sheet.sheet_name} at {position}")
            
            # If no new sheets were placed, break to avoid infinite loop
            if len(placed) == placed_count:
                break
        
        # Place remaining unplaced sheets in a grid
        unplaced = set(range(len(sheets))) - placed
        if unplaced:
            self.logger.warning(f"Could not place {len(unplaced)} sheets automatically, using grid layout")
            self._place_in_grid(sheets, positions, unplaced)
        
        return positions
    
    def _find_best_starting_sheet(self, sheets: List[MLEnhancedSheet], connectivity: Dict) -> Optional[int]:
        """Find the best sheet to start layout with."""
        best_score = -1
        best_sheet = None
        
        for i, sheet in enumerate(sheets):
            sheet_name = sheet.sheet_name or f"Sheet_{i}"
            
            # Score based on ML confidence and connectivity
            ml_score = sheet.ml_confidence
            conn_score = len(connectivity.get(sheet_name, {}).get('targets', {}))
            total_score = ml_score + conn_score * 0.1
            
            if total_score > best_score:
                best_score = total_score
                best_sheet = i
        
        return best_sheet
    
    def _find_best_position(self, sheet: MLEnhancedSheet, all_sheets: List[MLEnhancedSheet], 
                           connectivity: Dict, positions: Dict, placed: Set) -> Optional[Tuple[float, float]]:
        """Find best position for a sheet relative to already placed sheets."""
        sheet_name = sheet.sheet_name or f"Sheet_{sheet.page_index}"
        
        # Look for connections to already placed sheets
        for placed_idx in placed:
            placed_sheet = all_sheets[placed_idx]
            placed_name = placed_sheet.sheet_name or f"Sheet_{placed_idx}"
            
            # Check if sheets are connected
            if self._are_sheets_connected(sheet_name, placed_name, connectivity):
                # Calculate position based on connection
                base_pos = positions[placed_idx]
                offset = self._calculate_connection_offset(sheet, placed_sheet)
                return (base_pos[0] + offset[0], base_pos[1] + offset[1])
        
        return None
    
    def _are_sheets_connected(self, name1: str, name2: str, connectivity: Dict) -> bool:
        """Check if two sheets are connected in the connectivity map."""
        if name1 in connectivity and name2 in connectivity[name1].get('targets', {}):
            return True
        if name2 in connectivity and name1 in connectivity[name2].get('targets', {}):
            return True
        return False
    
    def _calculate_connection_offset(self, sheet1: MLEnhancedSheet, sheet2: MLEnhancedSheet) -> Tuple[float, float]:
        """Calculate offset between two connected sheets."""
        # For now, place sheets side by side
        return (sheet2.width_in, 0.0)
    
    def _place_in_grid(self, sheets: List[MLEnhancedSheet], positions: Dict, unplaced: Set):
        """Place unplaced sheets in a grid layout."""
        grid_x = 0
        grid_y = 0
        max_width = max(s.width_in for s in sheets)
        
        for i in unplaced:
            positions[i] = (grid_x * max_width, grid_y * max_width)
            grid_x += 1
            if grid_x > 3:  # Start new row after 4 sheets
                grid_x = 0
                grid_y += 1
    
    def create_ml_enhanced_visualization(self, sheets: List[MLEnhancedSheet], connectivity: Dict, 
                                       positions: Dict, output_path: str = "ml_enhanced_mosaic.png"):
        """Create enhanced visualization with ML insights."""
        self.logger.info("Creating ML-enhanced visualization...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot 1: ML-enhanced connectivity network
        ax1.set_title("ML-Enhanced Connectivity Network", fontsize=14, fontweight='bold')
        G = nx.DiGraph()
        
        # Add nodes with ML information
        for sheet_name, info in connectivity.items():
            discipline = info.get('discipline', 'unknown')
            confidence = info.get('confidence', 0.0)
            issues = info.get('issues', 0)
            
            # Color nodes by discipline
            color_map = {
                'traffic': 'lightblue',
                'electrical': 'lightgreen', 
                'structural': 'lightcoral',
                'drainage': 'lightyellow',
                'unknown': 'lightgray'
            }
            color = color_map.get(discipline, 'lightgray')
            
            G.add_node(sheet_name, discipline=discipline, confidence=confidence, issues=issues, color=color)
        
        # Add edges
        for sheet_name, info in connectivity.items():
            for edge, targets in info['targets'].items():
                for target in targets:
                    if target in connectivity:
                        G.add_edge(sheet_name, target, edge=edge)
        
        # Draw network
        pos = nx.spring_layout(G)
        colors = [G.nodes[node]['color'] for node in G.nodes()]
        nx.draw(G, pos, ax=ax1, with_labels=True, node_color=colors,
                node_size=2000, font_size=10, font_weight='bold',
                arrows=True, edge_color='gray', arrowsize=20)
        
        # Plot 2: ML Analysis Summary
        ax2.set_title("ML Analysis Summary", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Count disciplines
        discipline_counts = {}
        confidence_sum = 0
        total_issues = 0
        
        for info in connectivity.values():
            discipline = info.get('discipline', 'unknown')
            discipline_counts[discipline] = discipline_counts.get(discipline, 0) + 1
            confidence_sum += info.get('confidence', 0.0)
            total_issues += info.get('issues', 0)
        
        avg_confidence = confidence_sum / len(connectivity) if connectivity else 0
        
        # Create summary text
        summary_text = f"ML Analysis Summary:\n\n"
        summary_text += f"Total Sheets: {len(sheets)}\n"
        summary_text += f"Average Confidence: {avg_confidence:.3f}\n"
        summary_text += f"Total Issues Detected: {total_issues}\n\n"
        summary_text += f"Discipline Distribution:\n"
        for discipline, count in discipline_counts.items():
            summary_text += f"  {discipline}: {count}\n"
        
        ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        # Plot 3: Sheet Details Table
        ax3.set_title("Sheet Details", fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        table_data = []
        for i, sheet in enumerate(sheets):
            sheet_name = sheet.sheet_name or f"Sheet_{i}"
            discipline = sheet.ml_discipline or "unknown"
            confidence = f"{sheet.ml_confidence:.3f}"
            issues = len(sheet.ml_issues)
            north_rot = f"{sheet.north_rotation:.1f}°"
            
            table_data.append([sheet_name, discipline, confidence, issues, north_rot])
        
        table = ax3.table(cellText=table_data,
                         colLabels=['Sheet', 'Discipline', 'Confidence', 'Issues', 'North Rot'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Plot 4: Optimized Layout
        ax4.set_title("ML-Optimized Layout", fontsize=14, fontweight='bold')
        
        # Calculate canvas size
        max_x = max(s.width_in for s in sheets)
        max_y = max(s.height_in for s in sheets)
        canvas_width = max_x * len(sheets) * 2
        canvas_height = max_y * len(sheets) * 2
        
        ax4.set_xlim(-canvas_width/2, canvas_width/2)
        ax4.set_ylim(-canvas_height/2, canvas_height/2)
        ax4.grid(True, alpha=0.3)
        
        # Place sheets based on optimized positions
        for i, sheet in enumerate(sheets):
            if i in positions:
                pos = positions[i]
                rect = patches.Rectangle(pos, sheet.width_in, sheet.height_in,
                                       linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7)
                ax4.add_patch(rect)
                
                # Add sheet info
                sheet_name = sheet.sheet_name or f"Sheet_{i}"
                discipline = sheet.ml_discipline or "unknown"
                ax4.text(pos[0] + sheet.width_in/2, pos[1] + sheet.height_in/2,
                        f"{sheet_name}\n{discipline}\n{sheet.ml_confidence:.3f}",
                        ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax4.set_xlabel("X (inches)")
        ax4.set_ylabel("Y (inches)")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"ML-enhanced visualization saved to: {output_path}")
        plt.show()
    
    def export_ml_analysis_report(self, sheets: List[MLEnhancedSheet], connectivity: Dict, 
                                 positions: Dict, output_path: str = "ml_mosaic_analysis.json"):
        """Export comprehensive ML analysis report."""
        self.logger.info("Exporting ML analysis report...")
        
        report = {
            "analysis_metadata": {
                "analysis_date": datetime.now().isoformat(),
                "system": "ML-Enhanced Mosaic System",
                "version": "1.0"
            },
            "sheet_analysis": [],
            "connectivity_analysis": connectivity,
            "layout_optimization": {
                "positions": {str(k): v for k, v in positions.items()},
                "placed_sheets": len(positions),
                "total_sheets": len(sheets)
            },
            "ml_insights": {
                "discipline_distribution": {},
                "average_confidence": 0.0,
                "total_issues": 0
            }
        }
        
        # Add sheet analysis
        for i, sheet in enumerate(sheets):
            sheet_analysis = {
                "sheet_index": i,
                "sheet_name": sheet.sheet_name,
                "page_number": sheet.page_number,
                "dimensions": {
                    "width_inches": sheet.width_in,
                    "height_inches": sheet.height_in
                },
                "ml_analysis": {
                    "discipline": sheet.ml_discipline,
                    "confidence": sheet.ml_confidence,
                    "issues": sheet.ml_issues
                },
                "north_detection": {
                    "rotation_angle": sheet.north_rotation,
                    "detected": abs(sheet.north_rotation) > 1.0
                },
                "layout": {
                    "position": positions.get(i, (0, 0)),
                    "is_placed": i in positions
                }
            }
            report["sheet_analysis"].append(sheet_analysis)
        
        # Calculate ML insights
        discipline_counts = {}
        confidence_sum = 0
        total_issues = 0
        
        for sheet in sheets:
            discipline = sheet.ml_discipline or "unknown"
            discipline_counts[discipline] = discipline_counts.get(discipline, 0) + 1
            confidence_sum += sheet.ml_confidence
            total_issues += len(sheet.ml_issues)
        
        report["ml_insights"]["discipline_distribution"] = discipline_counts
        report["ml_insights"]["average_confidence"] = confidence_sum / len(sheets) if sheets else 0
        report["ml_insights"]["total_issues"] = total_issues
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"ML analysis report exported to: {output_path}")
        return report

@dataclass
class NorthDetectionResult:
    """Result of north arrow detection."""
    detected: bool
    angle: float
    confidence: float
    template_angle: float
    position: Tuple[int, int]

def main():
    """Main function for ML-enhanced mosaic system."""
    parser = argparse.ArgumentParser(description='ML-Enhanced Mosaic System')
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--output', default='ml_enhanced_mosaic.png', help='Output visualization path')
    parser.add_argument('--sheets', type=int, default=10, help='Number of sheets to process')
    parser.add_argument('--report', default='ml_mosaic_analysis.json', help='Output analysis report path')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ML-Enhanced Mosaic System")
    
    try:
        # Initialize system
        mosaic_system = MLEnhancedMosaicSystem()
        
        # Load and analyze sheets
        sheets = mosaic_system.load_sheets(args.pdf, args.sheets)
        logger.info(f"Loaded and analyzed {len(sheets)} sheets")
        
        # Create connectivity map
        connectivity = mosaic_system.create_connectivity_map(sheets)
        logger.info(f"Created connectivity map with {len(connectivity)} nodes")
        
        # Optimize layout
        positions = mosaic_system.optimize_layout(sheets, connectivity)
        logger.info(f"Optimized layout for {len(positions)} sheets")
        
        # Create visualization
        mosaic_system.create_ml_enhanced_visualization(sheets, connectivity, positions, args.output)
        
        # Export analysis report
        report = mosaic_system.export_ml_analysis_report(sheets, connectivity, positions, args.report)
        
        logger.info("ML-Enhanced Mosaic analysis completed successfully!")
        logger.info(f"Visualization: {args.output}")
        logger.info(f"Analysis report: {args.report}")
        
        return 0
        
    except Exception as e:
        logger.error(f"ML-Enhanced Mosaic analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
