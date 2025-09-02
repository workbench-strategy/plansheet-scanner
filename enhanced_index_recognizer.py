#!/usr/bin/env python3
"""
Enhanced Index Symbol and Text Recognizer

Specialized system for extracting discipline information from:
- Drawing indexes and legends
- Reference symbols and codes
- Text annotations and labels
- Cross-reference patterns
- Multi-page index relationships
"""

import cv2
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
import pytesseract
from collections import defaultdict, Counter
import logging

class EnhancedIndexRecognizer:
    """
    Advanced index symbol and text recognition system for engineering drawings.
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Discipline-specific keywords and patterns
        self.discipline_patterns = {
            'electrical': {
                'keywords': ['electrical', 'power', 'lighting', 'conduit', 'cable', 'panel', 'circuit', 'voltage', 'amp', 'watt'],
                'symbols': ['COND', 'EMT', 'PVC', 'RMC', 'JB', 'JBOX', 'SW', 'CB', 'FUSE', 'TRANS', 'GEN'],
                'codes': ['NEC', 'IEEE', 'UL', 'CSA', 'IEC'],
                'file_patterns': ['*electrical*', '*power*', '*lighting*', '*conduit*']
            },
            'structural': {
                'keywords': ['structural', 'steel', 'concrete', 'beam', 'column', 'foundation', 'rebar', 'welding', 'bolt'],
                'symbols': ['W', 'HSS', 'WT', 'ST', 'PL', 'L', 'C', 'MC', 'HP', 'PIPE'],
                'codes': ['AISC', 'ACI', 'ASCE', 'IBC', 'IRC'],
                'file_patterns': ['*structural*', '*steel*', '*concrete*', '*beam*']
            },
            'civil': {
                'keywords': ['civil', 'roadway', 'drainage', 'grading', 'pavement', 'curb', 'guardrail', 'sign', 'marking'],
                'symbols': ['RD', 'DR', 'GR', 'PAV', 'CURB', 'GRL', 'SIGN', 'MARK', 'MAN', 'CATCH'],
                'codes': ['AASHTO', 'MUTCD', 'FHWA', 'DOT', 'MEPDG'],
                'file_patterns': ['*civil*', '*roadway*', '*drainage*', '*grading*']
            },
            'mechanical': {
                'keywords': ['mechanical', 'HVAC', 'plumbing', 'duct', 'pipe', 'valve', 'pump', 'fan', 'chiller'],
                'symbols': ['HVAC', 'DUCT', 'PIPE', 'VALVE', 'PUMP', 'FAN', 'CHLR', 'AHU', 'VAV', 'RTU'],
                'codes': ['ASHRAE', 'SMACNA', 'ASME', 'ANSI', 'NFPA'],
                'file_patterns': ['*mechanical*', '*hvac*', '*plumbing*', '*duct*']
            },
            'architectural': {
                'keywords': ['architectural', 'floor', 'ceiling', 'wall', 'door', 'window', 'finish', 'furniture'],
                'symbols': ['FLR', 'CLG', 'WALL', 'DR', 'WIN', 'FIN', 'FURN', 'EQ', 'PLMB', 'ELEC'],
                'codes': ['IBC', 'ADA', 'LEED', 'ASHRAE', 'NFPA'],
                'file_patterns': ['*architectural*', '*floor*', '*ceiling*', '*wall*']
            }
        }
        
        # Common engineering symbols and abbreviations
        self.engineering_symbols = {
            'dimensions': ['DIM', 'DIA', 'RAD', 'LENGTH', 'WIDTH', 'HEIGHT', 'THK', 'DEPTH'],
            'materials': ['STEEL', 'CONC', 'ALUM', 'WOOD', 'PLASTIC', 'COPPER', 'PVC', 'HDPE'],
            'specifications': ['SPEC', 'DET', 'SECT', 'ELEV', 'PLAN', 'SCHED', 'NOTE', 'REF'],
            'references': ['REF', 'SEE', 'TYP', 'SIM', 'CONT', 'MATCH', 'SHEET', 'DETAIL'],
            'status': ['EXIST', 'PROP', 'REMOVE', 'RELOC', 'MODIFY', 'NEW', 'DEMO']
        }
        
        # Multi-page index patterns
        self.index_patterns = {
            'sheet_references': [r'SHEET\s*\d+', r'PLAN\s*\d+', r'DETAIL\s*\d+', r'REF\s*\d+'],
            'grid_coordinates': [r'[A-Z]\d+', r'\d+[A-Z]', r'[A-Z]\d+[A-Z]', r'\d+[A-Z]\d+'],
            'discipline_codes': [r'[A-Z]{2,4}\s*\d+', r'\d+[A-Z]{2,4}', r'[A-Z]+\s*[A-Z]+\s*\d+'],
            'cross_references': [r'SEE\s+[A-Z]+\s*\d+', r'REF\s+[A-Z]+\s*\d+', r'CONT\s+[A-Z]+\s*\d+']
        }
        
        print("🔍 Enhanced Index Recognizer Initialized")
        print("✅ Discipline-specific pattern recognition")
        print("✅ Engineering symbol extraction")
        print("✅ Multi-page index analysis")
        print("✅ Text and code recognition")
    
    def _setup_logging(self):
        """Setup logging for the index recognition process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_index_recognition.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def analyze_drawing_index(self, image_path):
        """
        Comprehensive analysis of drawing index, symbols, and text.
        """
        self.logger.info(f"Analyzing drawing index: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return None
        
        results = {
            'filename': Path(image_path).name,
            'discipline_confidence': {},
            'extracted_symbols': [],
            'extracted_text': [],
            'index_patterns': [],
            'cross_references': [],
            'recommended_discipline': None,
            'confidence_score': 0.0
        }
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Extract text using OCR
            text_results = self._extract_text_regions(gray)
            results['extracted_text'] = text_results['text_lines']
            
            # 2. Extract symbols and patterns
            symbol_results = self._extract_symbols_and_patterns(gray)
            results['extracted_symbols'] = symbol_results['symbols']
            results['index_patterns'] = symbol_results['patterns']
            
            # 3. Analyze discipline indicators
            discipline_results = self._analyze_discipline_indicators(
                text_results['text_lines'], 
                symbol_results['symbols'],
                Path(image_path).name
            )
            results['discipline_confidence'] = discipline_results['confidence']
            results['recommended_discipline'] = discipline_results['recommended']
            results['confidence_score'] = discipline_results['score']
            
            # 4. Find cross-references
            cross_refs = self._find_cross_references(text_results['text_lines'])
            results['cross_references'] = cross_refs
            
            self.logger.info(f"Analysis complete - Recommended discipline: {results['recommended_discipline']}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing drawing index: {e}")
            return None
    
    def _extract_text_regions(self, gray):
        """
        Extract text regions using multiple methods.
        """
        text_lines = []
        
        # Method 1: MSER for text region detection
        mser = cv2.MSER_create(min_area=50, max_area=2000)
        regions, _ = mser.detectRegions(gray)
        
        if regions is not None:
            for region in regions:
                if len(region) >= 10:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(region)
                    
                    # Extract region for OCR
                    roi = gray[y:y+h, x:x+w]
                    
                    # Apply OCR
                    try:
                        text = pytesseract.image_to_string(roi, config='--psm 8 --oem 3')
                        text = text.strip()
                        
                        if text and len(text) > 1:
                            text_lines.append({
                                'text': text,
                                'bbox': (x, y, w, h),
                                'confidence': 0.8
                            })
                    except:
                        continue
        
        # Method 2: Contour-based text detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter for text-like regions
                if 0.5 < aspect_ratio < 10 and h > 10:
                    roi = gray[y:y+h, x:x+w]
                    try:
                        text = pytesseract.image_to_string(roi, config='--psm 8 --oem 3')
                        text = text.strip()
                        
                        if text and len(text) > 1:
                            text_lines.append({
                                'text': text,
                                'bbox': (x, y, w, h),
                                'confidence': 0.6
                            })
                    except:
                        continue
        
        return {'text_lines': text_lines}
    
    def _extract_symbols_and_patterns(self, gray):
        """
        Extract engineering symbols and patterns.
        """
        symbols = []
        patterns = []
        
        # Template matching for common symbols
        symbol_templates = self._create_symbol_templates()
        
        for symbol_name, template in symbol_templates.items():
            matches = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(matches >= 0.7)
            
            for pt in zip(*locations[::-1]):
                symbols.append({
                    'symbol': symbol_name,
                    'bbox': (pt[0], pt[1], template.shape[1], template.shape[0]),
                    'confidence': float(matches[pt[1], pt[0]])
                })
        
        # Pattern matching using regex on extracted text
        text_content = " ".join([line['text'] for line in self._extract_text_regions(gray)['text_lines']])
        
        # Check for index patterns
        for pattern_name, pattern_list in self.index_patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                for match in matches:
                    patterns.append({
                        'type': pattern_name,
                        'pattern': pattern,
                        'match': match,
                        'confidence': 0.9
                    })
        
        return {'symbols': symbols, 'patterns': patterns}
    
    def _create_symbol_templates(self):
        """
        Create template images for common engineering symbols.
        """
        templates = {}
        
        # Create simple text-based templates
        symbol_texts = [
            'COND', 'EMT', 'JB', 'SW', 'CB', 'W', 'HSS', 'PL', 'L', 'C',
            'RD', 'DR', 'GR', 'PAV', 'CURB', 'HVAC', 'DUCT', 'PIPE', 'VALVE'
        ]
        
        for text in symbol_texts:
            # Create a simple template (in practice, you'd use actual symbol images)
            template = np.zeros((30, 80), dtype=np.uint8)
            cv2.putText(template, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            templates[text] = template
        
        return templates
    
    def _analyze_discipline_indicators(self, text_lines, symbols, filename):
        """
        Analyze text and symbols to determine discipline.
        """
        discipline_scores = defaultdict(float)
        all_text = " ".join([line['text'] for line in text_lines]).lower()
        filename_lower = filename.lower()
        
        # Analyze each discipline
        for discipline, patterns in self.discipline_patterns.items():
            score = 0.0
            
            # Check filename patterns
            for pattern in patterns['file_patterns']:
                if pattern.replace('*', '') in filename_lower:
                    score += 2.0
            
            # Check keywords in text
            for keyword in patterns['keywords']:
                if keyword.lower() in all_text:
                    score += 1.0
            
            # Check symbols
            for symbol in patterns['symbols']:
                if symbol.lower() in all_text:
                    score += 1.5
                # Check extracted symbols
                for sym in symbols:
                    if symbol.lower() in sym['symbol'].lower():
                        score += sym['confidence']
            
            # Check codes
            for code in patterns['codes']:
                if code.lower() in all_text:
                    score += 2.0
            
            discipline_scores[discipline] = score
        
        # Find the discipline with highest score
        if discipline_scores:
            recommended = max(discipline_scores, key=discipline_scores.get)
            max_score = discipline_scores[recommended]
            total_score = sum(discipline_scores.values())
            confidence = max_score / total_score if total_score > 0 else 0
        else:
            recommended = 'unknown'
            confidence = 0.0
        
        return {
            'confidence': dict(discipline_scores),
            'recommended': recommended,
            'score': confidence
        }
    
    def _find_cross_references(self, text_lines):
        """
        Find cross-references to other sheets or details.
        """
        cross_refs = []
        
        for line in text_lines:
            text = line['text']
            
            # Look for cross-reference patterns
            patterns = [
                r'SEE\s+([A-Z]+\s*\d+)',
                r'REF\s+([A-Z]+\s*\d+)',
                r'CONT\s+([A-Z]+\s*\d+)',
                r'DETAIL\s+([A-Z]+\s*\d+)',
                r'SHEET\s+(\d+)',
                r'PLAN\s+(\d+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    cross_refs.append({
                        'type': 'cross_reference',
                        'reference': match,
                        'context': text,
                        'bbox': line['bbox']
                    })
        
        return cross_refs
    
    def batch_analyze_drawings(self, image_dir="yolo_processed_data_local/images", max_samples=100):
        """
        Analyze multiple drawings to build index recognition database.
        """
        self.logger.info(f"Starting batch analysis of drawings in: {image_dir}")
        
        image_path = Path(image_dir)
        if not image_path.exists():
            self.logger.error(f"Image directory not found: {image_dir}")
            return
        
        # Find all images
        image_files = list(image_path.glob("*.png"))
        if len(image_files) > max_samples:
            image_files = image_files[:max_samples]
        
        self.logger.info(f"Found {len(image_files)} images for analysis")
        
        analysis_results = []
        discipline_stats = defaultdict(int)
        
        for i, image_file in enumerate(image_files):
            if i % 10 == 0:
                self.logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
            
            result = self.analyze_drawing_index(image_file)
            if result:
                analysis_results.append(result)
                if result['recommended_discipline']:
                    discipline_stats[result['recommended_discipline']] += 1
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"index_analysis_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'analysis_results': analysis_results,
                'discipline_stats': dict(discipline_stats),
                'total_analyzed': len(analysis_results)
            }, f, indent=2, default=str)
        
        self.logger.info(f"Batch analysis complete!")
        self.logger.info(f"Discipline distribution: {dict(discipline_stats)}")
        self.logger.info(f"Results saved to: {results_file}")
        
        return analysis_results


def main():
    """Main function to demonstrate index recognition."""
    print("🔍 Enhanced Index Recognition Demo")
    print("=" * 50)
    
    # Initialize recognizer
    recognizer = EnhancedIndexRecognizer()
    
    # Analyze a single drawing
    print("\n📋 Analyzing single drawing...")
    sample_image = "yolo_processed_data_local/images/Asbuilts_Illumination-Power_SR509_Phase1_Stage1b_page_001.png"
    
    if Path(sample_image).exists():
        result = recognizer.analyze_drawing_index(sample_image)
        if result:
            print(f"✅ Analysis complete!")
            print(f"📊 Recommended discipline: {result['recommended_discipline']}")
            print(f"🎯 Confidence score: {result['confidence_score']:.3f}")
            print(f"📝 Extracted text lines: {len(result['extracted_text'])}")
            print(f"🔧 Extracted symbols: {len(result['extracted_symbols'])}")
            print(f"🔗 Cross-references: {len(result['cross_references'])}")
    else:
        print(f"❌ Sample image not found: {sample_image}")
    
    # Batch analysis
    print("\n📊 Starting batch analysis...")
    results = recognizer.batch_analyze_drawings(max_samples=50)
    
    if results:
        print(f"\n✅ Batch analysis complete!")
        print(f"📈 Total drawings analyzed: {len(results)}")
        
        # Show discipline distribution
        discipline_counts = Counter([r['recommended_discipline'] for r in results if r['recommended_discipline']])
        print(f"📊 Discipline distribution:")
        for discipline, count in discipline_counts.most_common():
            print(f"   {discipline}: {count}")


if __name__ == "__main__":
    main()
