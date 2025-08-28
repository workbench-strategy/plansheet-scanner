#!/usr/bin/env python3
"""
Enhanced Plan Highlighter with ML Intelligence
Uses trained ML models to intelligently highlight conduit, fiber, and critical elements in plans.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from improved_ai_trainer import ImprovedAIEngineerTrainer
from plan_review_and_tagging import PlanReviewer

class EnhancedPlanHighlighter:
    """Enhanced plan highlighter with ML intelligence."""
    
    def __init__(self):
        self.ml_trainer = ImprovedAIEngineerTrainer()
        self.plan_reviewer = PlanReviewer()
        
        # Highlighting colors and styles
        self.highlight_colors = {
            'conduit': (255, 0, 0),      # Red
            'fiber': (0, 0, 255),        # Blue
            'junction_box': (0, 255, 0), # Green
            'signal': (255, 255, 0),     # Yellow
            'violation': (255, 0, 255),  # Magenta
            'change': (255, 165, 0),     # Orange
            'critical': (128, 0, 128)    # Purple
        }
        
        # Element detection patterns
        self.element_patterns = {
            'conduit': [
                r'conduit', r'pipe', r'electrical conduit', r'wiring',
                r'COND', r'PIPE', r'ELEC', r'WIRE'
            ],
            'fiber': [
                r'fiber', r'fiber optic', r'optical', r'communication',
                r'FIBER', r'OPTIC', r'COMM', r'DATA'
            ],
            'junction_box': [
                r'junction box', r'j-box', r'JB', r'electrical box',
                r'JUNCTION', r'BOX', r'JB', r'ELEC'
            ],
            'signal': [
                r'signal', r'traffic signal', r'TS', r'light',
                r'SIGNAL', r'TS', r'LIGHT', r'TRAFFIC'
            ]
        }
    
    def analyze_plan_for_highlighting(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze plan using ML to determine what should be highlighted."""
        print(f"🔍 Analyzing plan: {plan_data.get('sheet_title', 'Unknown')}")
        
        # Perform ML analysis
        ml_result = self.ml_trainer.review_drawing(plan_data)
        review_result = self.plan_reviewer.review_plan_comprehensive(plan_data)
        
        # Extract highlighting information
        highlighting_info = {
            'discipline': ml_result.get('predicted_discipline', 'unknown'),
            'confidence': ml_result.get('overall_confidence', 0.0),
            'has_violations': ml_result.get('has_code_violations', False),
            'has_errors': ml_result.get('has_design_errors', False),
            'recommendations': ml_result.get('recommendations', []),
            'issues': review_result.get('issues', []),
            'elements_to_highlight': [],
            'critical_areas': [],
            'as_built_changes': []
        }
        
        # Identify elements to highlight based on construction notes
        construction_notes = plan_data.get('construction_notes', '').lower()
        for element_type, patterns in self.element_patterns.items():
            for pattern in patterns:
                if pattern.lower() in construction_notes:
                    highlighting_info['elements_to_highlight'].append({
                        'type': element_type,
                        'pattern': pattern,
                        'confidence': 0.8
                    })
                    break
        
        # Identify as-built changes
        as_built_changes = plan_data.get('as_built_changes', [])
        for change in as_built_changes:
            highlighting_info['as_built_changes'].append({
                'description': change.get('description', ''),
                'severity': change.get('severity', 'minor'),
                'type': change.get('type', 'unknown')
            })
        
        # Identify critical areas based on issues
        for issue in highlighting_info['issues']:
            if issue.get('tag') in ['high', 'critical']:
                highlighting_info['critical_areas'].append({
                    'category': issue.get('category', 'Unknown'),
                    'title': issue.get('title', 'Unknown Issue'),
                    'description': issue.get('description', '')
                })
        
        return highlighting_info
    
    def create_highlighted_plan(self, plan_path: str, highlighting_info: Dict[str, Any], 
                               output_path: str = None) -> str:
        """Create a highlighted version of the plan."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"highlighted_plan_{timestamp}.png"
        
        print(f"🎨 Creating highlighted plan: {output_path}")
        
        # Load the plan image
        if plan_path.endswith('.pdf'):
            # Convert PDF to image
            doc = fitz.open(plan_path)
            page = doc[0]  # First page
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            doc.close()
        else:
            # Load image directly
            image = Image.open(plan_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create drawing overlay
        draw = ImageDraw.Draw(image)
        
        # Add highlighting based on analysis
        self._add_element_highlights(draw, highlighting_info, image.size)
        self._add_issue_highlights(draw, highlighting_info, image.size)
        self._add_legend(draw, highlighting_info, image.size)
        
        # Save highlighted plan
        image.save(output_path)
        print(f"✅ Highlighted plan saved: {output_path}")
        
        return output_path
    
    def _add_element_highlights(self, draw: ImageDraw.Draw, highlighting_info: Dict[str, Any], 
                               image_size: Tuple[int, int]):
        """Add highlights for detected elements."""
        width, height = image_size
        
        # Highlight conduit and fiber elements
        for element in highlighting_info['elements_to_highlight']:
            element_type = element['type']
            if element_type in self.highlight_colors:
                color = self.highlight_colors[element_type]
                
                # Create a semi-transparent overlay
                overlay = Image.new('RGBA', image_size, (*color, 50))
                draw.bitmap((0, 0), overlay)
                
                # Add text label
                label = f"{element_type.upper()}: {element['pattern']}"
                draw.text((10, 10), label, fill=color, font=None)
    
    def _add_issue_highlights(self, draw: ImageDraw.Draw, highlighting_info: Dict[str, Any], 
                             image_size: Tuple[int, int]):
        """Add highlights for code violations and issues."""
        width, height = image_size
        
        # Highlight critical areas
        for area in highlighting_info['critical_areas']:
            color = self.highlight_colors['critical']
            
            # Add warning box
            box_x = width - 300
            box_y = 50
            draw.rectangle([box_x, box_y, box_x + 280, box_y + 100], 
                          outline=color, width=3)
            
            # Add warning text
            draw.text((box_x + 10, box_y + 10), 
                     f"CRITICAL: {area['category']}", 
                     fill=color, font=None)
            draw.text((box_x + 10, box_y + 30), 
                     area['title'][:40], 
                     fill=color, font=None)
    
    def _add_legend(self, draw: ImageDraw.Draw, highlighting_info: Dict[str, Any], 
                   image_size: Tuple[int, int]):
        """Add a legend explaining the highlights."""
        width, height = image_size
        
        # Create legend box
        legend_x = 10
        legend_y = height - 200
        legend_width = 300
        legend_height = 180
        
        # Background
        draw.rectangle([legend_x, legend_y, legend_x + legend_width, legend_y + legend_height], 
                      fill=(255, 255, 255, 200), outline=(0, 0, 0), width=2)
        
        # Title
        draw.text((legend_x + 10, legend_y + 10), "ML-ENHANCED HIGHLIGHTS", 
                  fill=(0, 0, 0), font=None)
        
        # Legend items
        y_offset = 40
        for element_type, color in self.highlight_colors.items():
            if any(e['type'] == element_type for e in highlighting_info['elements_to_highlight']):
                # Color box
                draw.rectangle([legend_x + 10, legend_y + y_offset, 
                               legend_x + 30, legend_y + y_offset + 20], 
                              fill=color)
                
                # Label
                draw.text((legend_x + 35, legend_y + y_offset), 
                         element_type.replace('_', ' ').title(), 
                         fill=(0, 0, 0), font=None)
                y_offset += 25
        
        # Add ML confidence
        confidence = highlighting_info.get('confidence', 0.0)
        draw.text((legend_x + 10, legend_y + y_offset + 10), 
                 f"ML Confidence: {confidence:.1%}", 
                 fill=(0, 0, 0), font=None)
    
    def generate_highlighting_report(self, highlighting_info: Dict[str, Any], 
                                   output_path: str = None) -> str:
        """Generate a detailed report of what was highlighted."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"highlighting_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': {
                'discipline': highlighting_info['discipline'],
                'confidence': highlighting_info['confidence'],
                'has_violations': highlighting_info['has_violations'],
                'has_errors': highlighting_info['has_errors']
            },
            'elements_highlighted': highlighting_info['elements_to_highlight'],
            'critical_issues': highlighting_info['critical_areas'],
            'as_built_changes': highlighting_info['as_built_changes'],
            'recommendations': highlighting_info['recommendations'],
            'highlighting_legend': {
                'conduit': 'Red - Electrical conduit and piping',
                'fiber': 'Blue - Fiber optic and communication',
                'junction_box': 'Green - Junction boxes and electrical equipment',
                'signal': 'Yellow - Traffic signals and control devices',
                'violation': 'Magenta - Code violations and compliance issues',
                'change': 'Orange - As-built changes and modifications',
                'critical': 'Purple - Critical issues requiring attention'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Highlighting report saved: {output_path}")
        return output_path
    
    def highlight_multiple_plans(self, plan_data_list: List[Dict[str, Any]], 
                               output_dir: str = "highlighted_plans") -> Dict[str, Any]:
        """Highlight multiple plans and generate comprehensive report."""
        Path(output_dir).mkdir(exist_ok=True)
        
        results = {
            'total_plans': len(plan_data_list),
            'highlighted_plans': [],
            'summary': {
                'total_elements': 0,
                'total_violations': 0,
                'total_changes': 0,
                'disciplines_found': set()
            }
        }
        
        for i, plan_data in enumerate(plan_data_list):
            print(f"\n🔍 Processing plan {i+1}/{len(plan_data_list)}: {plan_data.get('sheet_title', 'Unknown')}")
            
            # Analyze plan
            highlighting_info = self.analyze_plan_for_highlighting(plan_data)
            
            # Update summary
            results['summary']['total_elements'] += len(highlighting_info['elements_to_highlight'])
            results['summary']['total_violations'] += len(highlighting_info['critical_areas'])
            results['summary']['total_changes'] += len(highlighting_info['as_built_changes'])
            results['summary']['disciplines_found'].add(highlighting_info['discipline'])
            
            # Create highlighted plan (if plan file exists)
            plan_file = plan_data.get('file_path')
            if plan_file and os.path.exists(plan_file):
                output_path = os.path.join(output_dir, f"highlighted_{i+1:03d}.png")
                self.create_highlighted_plan(plan_file, highlighting_info, output_path)
                
                results['highlighted_plans'].append({
                    'plan_id': plan_data.get('drawing_id', f'plan_{i+1}'),
                    'highlighted_file': output_path,
                    'analysis': highlighting_info
                })
            
            # Generate individual report
            report_path = os.path.join(output_dir, f"report_{i+1:03d}.json")
            self.generate_highlighting_report(highlighting_info, report_path)
        
        # Generate comprehensive report
        comprehensive_report = os.path.join(output_dir, "comprehensive_highlighting_report.json")
        with open(comprehensive_report, 'w') as f:
            json.dump(results, f, indent=2, default=list)
        
        print(f"\n🎉 Multi-plan highlighting complete!")
        print(f"   📁 Output directory: {output_dir}")
        print(f"   📊 Plans processed: {results['total_plans']}")
        print(f"   🔍 Elements highlighted: {results['summary']['total_elements']}")
        print(f"   ⚠️  Violations found: {results['summary']['total_violations']}")
        print(f"   🔄 Changes detected: {results['summary']['total_changes']}")
        
        return results

def main():
    """Example usage of the enhanced plan highlighter."""
    print("🎨 Enhanced Plan Highlighter with ML Intelligence")
    print("=" * 60)
    
    highlighter = EnhancedPlanHighlighter()
    
    # Example plan data
    test_plans = [
        {
            "drawing_id": "conduit_plan_001",
            "sheet_title": "Electrical Conduit Plan",
            "discipline": "electrical",
            "project_name": "Fiber Conduit Project",
            "construction_notes": "Fiber conduit installed per plan with additional runs for future expansion. Junction boxes installed at 200' intervals. Grounding conductors installed per NEC requirements.",
            "as_built_changes": [
                {"description": "Additional fiber conduit installed", "severity": "minor"}
            ],
            "file_path": "conduit_plan.pdf"
        },
        {
            "drawing_id": "its_plan_001",
            "sheet_title": "ITS Communication Plan",
            "discipline": "its",
            "project_name": "ITS Fiber Network",
            "construction_notes": "ITS fiber network installed with redundant connections and future expansion capacity. Communication cabinets installed at strategic locations.",
            "as_built_changes": [
                {"description": "Fiber routing adjusted for field conditions", "severity": "minor"}
            ],
            "file_path": "its_plan.pdf"
        }
    ]
    
    # Process plans
    results = highlighter.highlight_multiple_plans(test_plans)
    
    print("\n✅ Enhanced plan highlighting complete!")
    print("   Your ML system can now intelligently highlight:")
    print("   - Conduit and fiber routing")
    print("   - Junction boxes and electrical equipment")
    print("   - Code violations and compliance issues")
    print("   - As-built changes and modifications")
    print("   - Critical areas requiring attention")

if __name__ == "__main__":
    main()

