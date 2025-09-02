#!/usr/bin/env python3
"""
Process All As-Built Files for Enhanced ML Training
Comprehensive script to process all 50+ as-built PDF files and enhance ML capabilities.
"""

import os
import sys
import json
import cv2
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import re
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('all_as_builts_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AsBuiltProcessor:
    """Process all as-built PDF files for enhanced ML training."""
    
    def __init__(self):
        self.as_built_dir = Path("as_built_drawings")
        self.output_dir = Path("enhanced_training_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in ["images", "metadata", "extracted_text", "features"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Project type mapping
        self.project_types = {
            'traffic_signal': ['signal', 'traffic', 'intersection'],
            'its': ['its', 'fiber', 'communication', 'atms', 'camera'],
            'electrical': ['electrical', 'lighting', 'power', 'conduit'],
            'structural': ['structural', 'bridge', 'ramp', 'interchange'],
            'congestion': ['congestion', 'management', 'traffic_flow'],
            'special': ['special', 'provisions', 'standards']
        }
    
    def analyze_as_built_collection(self):
        """Analyze the entire as-built collection."""
        logger.info("Analyzing as-built collection...")
        
        pdf_files = list(self.as_built_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        collection_stats = {
            'total_files': len(pdf_files),
            'total_size_mb': sum(f.stat().st_size / (1024*1024) for f in pdf_files),
            'file_types': defaultdict(int),
            'project_categories': defaultdict(int),
            'date_range': {'earliest': None, 'latest': None}
        }
        
        for pdf_file in pdf_files:
            # Analyze file name for project type
            filename = pdf_file.name.lower()
            
            # Determine project category
            category = self.categorize_project(filename)
            collection_stats['project_categories'][category] += 1
            
            # Extract date if present
            date_match = re.search(r'(\d{4})', filename)
            if date_match:
                year = int(date_match.group(1))
                if collection_stats['date_range']['earliest'] is None or year < collection_stats['date_range']['earliest']:
                    collection_stats['date_range']['earliest'] = year
                if collection_stats['date_range']['latest'] is None or year > collection_stats['date_range']['latest']:
                    collection_stats['date_range']['latest'] = year
        
        logger.info(f"Collection Statistics:")
        logger.info(f"  Total Files: {collection_stats['total_files']}")
        logger.info(f"  Total Size: {collection_stats['total_size_mb']:.1f} MB")
        logger.info(f"  Date Range: {collection_stats['date_range']['earliest']} - {collection_stats['date_range']['latest']}")
        logger.info(f"  Project Categories:")
        for category, count in collection_stats['project_categories'].items():
            logger.info(f"    {category}: {count} files")
        
        return collection_stats
    
    def categorize_project(self, filename: str) -> str:
        """Categorize project based on filename."""
        filename_lower = filename.lower()
        
        for category, keywords in self.project_types.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
        
        return 'unknown'
    
    def extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF file."""
        try:
            doc = fitz.open(pdf_path)
            
            metadata = {
                'filename': pdf_path.name,
                'file_size_mb': pdf_path.stat().st_size / (1024*1024),
                'page_count': len(doc),
                'project_category': self.categorize_project(pdf_path.name),
                'extraction_date': datetime.now().isoformat()
            }
            
            # Extract text from first few pages
            text_content = []
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                text_content.append(page.get_text())
            
            metadata['text_sample'] = '\n'.join(text_content)[:2000]  # First 2000 chars
            
            doc.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {'filename': pdf_path.name, 'error': str(e)}
    
    def convert_pdf_to_images(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        """Convert PDF pages to images."""
        try:
            doc = fitz.open(pdf_path)
            image_paths = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Save image
                image_filename = f"{pdf_path.stem}_page_{page_num:03d}.png"
                image_path = output_dir / image_filename
                pix.save(str(image_path))
                
                image_paths.append(image_path)
            
            doc.close()
            logger.info(f"Converted {pdf_path.name} to {len(image_paths)} images")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting {pdf_path} to images: {e}")
            return []
    
    def extract_engineering_features(self, image_path: Path) -> Dict[str, Any]:
        """Extract engineering-specific features from image."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return {}
            
            features = {}
            
            # Basic image features
            features['image_size'] = image.shape[:2]
            features['aspect_ratio'] = image.shape[1] / image.shape[0]
            features['color_channels'] = image.shape[2]
            
            # Color analysis
            features['mean_color'] = np.mean(image, axis=(0, 1)).tolist()
            features['std_color'] = np.std(image, axis=(0, 1)).tolist()
            
            # Edge detection for engineering elements
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Line detection for engineering drawings
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
            features['line_count'] = len(lines) if lines is not None else 0
            
            # Contour analysis for symbols
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features['contour_count'] = len(contours)
            
            # Text region detection (simplified)
            # This would be enhanced with OCR in production
            features['text_regions'] = self.detect_text_regions(gray)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return {}
    
    def detect_text_regions(self, gray_image: np.ndarray) -> int:
        """Detect potential text regions in image."""
        # Simple text region detection using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(morph)
        return num_labels - 1  # Subtract background
    
    def process_all_as_builts(self):
        """Process all as-built files for enhanced ML training."""
        logger.info("Starting comprehensive as-built processing...")
        
        # Analyze collection
        stats = self.analyze_as_built_collection()
        
        # Process each PDF file
        pdf_files = list(self.as_built_dir.glob("*.pdf"))
        processed_files = 0
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}...")
                
                # Extract metadata
                metadata = self.extract_pdf_metadata(pdf_file)
                metadata_path = self.output_dir / "metadata" / f"{pdf_file.stem}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Convert to images
                images_dir = self.output_dir / "images" / pdf_file.stem
                images_dir.mkdir(exist_ok=True)
                image_paths = self.convert_pdf_to_images(pdf_file, images_dir)
                
                # Extract features from each image
                features_data = []
                for image_path in image_paths:
                    features = self.extract_engineering_features(image_path)
                    features['image_path'] = str(image_path)
                    features['source_pdf'] = pdf_file.name
                    features_data.append(features)
                
                # Save features
                features_path = self.output_dir / "features" / f"{pdf_file.stem}_features.json"
                with open(features_path, 'w') as f:
                    json.dump(features_data, f, indent=2)
                
                processed_files += 1
                logger.info(f"[OK] Completed {pdf_file.name} ({len(image_paths)} pages)")
                
            except Exception as e:
                logger.error(f"[ERROR] Error processing {pdf_file.name}: {e}")
        
        # Generate summary report
        self.generate_processing_report(stats, processed_files)
        
        logger.info(f"Processing complete! {processed_files}/{len(pdf_files)} files processed")
    
    def generate_processing_report(self, stats: Dict, processed_files: int):
        """Generate comprehensive processing report."""
        report = {
            'processing_date': datetime.now().isoformat(),
            'collection_stats': stats,
            'processing_results': {
                'total_files': stats['total_files'],
                'processed_files': processed_files,
                'success_rate': processed_files / stats['total_files'] * 100
            },
            'output_structure': {
                'images_dir': str(self.output_dir / "images"),
                'metadata_dir': str(self.output_dir / "metadata"),
                'features_dir': str(self.output_dir / "features"),
                'extracted_text_dir': str(self.output_dir / "extracted_text")
            },
            'ml_training_ready': {
                'total_images': len(list((self.output_dir / "images").rglob("*.png"))),
                'total_features': len(list((self.output_dir / "features").glob("*.json"))),
                'total_metadata': len(list((self.output_dir / "metadata").glob("*.json")))
            }
        }
        
        report_path = self.output_dir / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Processing Report Generated:")
        logger.info(f"  Success Rate: {report['processing_results']['success_rate']:.1f}%")
        logger.info(f"  Total Images: {report['ml_training_ready']['total_images']}")
        logger.info(f"  Total Features: {report['ml_training_ready']['total_features']}")
        logger.info(f"  Report saved to: {report_path}")

def main():
    """Main function to process all as-built files."""
    print("Processing All As-Built Files for Enhanced ML Training")
    print("=" * 70)
    
    processor = AsBuiltProcessor()
    processor.process_all_as_builts()
    
    print("\nProcessing Complete!")
    print("Check enhanced_training_data/ directory for results")
    print("Review all_as_builts_processing.log for detailed information")

if __name__ == "__main__":
    main()
