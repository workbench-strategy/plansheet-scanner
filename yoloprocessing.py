#!/usr/bin/env python3
"""
YOLO Processing Script for GitHub Codespaces - 32GB Machine
Optimized for maximum speed processing of all 57 PDF files with cloud resources.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import fitz
from pathlib import Path
from datetime import datetime
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_single_pdf_worker(pdf_path_str):
    """Standalone function for multiprocessing - optimized for 32GB RAM."""
    pdf_path = Path(pdf_path_str)
    try:
        logger.info(f"Processing {pdf_path.name}...")
        
        doc = fitz.open(pdf_path)
        
        # Extract metadata
        metadata = {
            'filename': pdf_path.name,
            'file_size_mb': pdf_path.stat().st_size / (1024*1024),
            'page_count': len(doc),
            'processing_date': datetime.now().isoformat(),
            'processing_environment': 'github_codespaces_32gb'
        }
        
        # Convert all pages to images with maximum quality
        image_paths = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Use maximum quality for 32GB machine
            mat = fitz.Matrix(4.0, 4.0)  # Increased from 3.0 to 4.0 for better quality
            pix = page.get_pixmap(matrix=mat)
            
            image_filename = f"{pdf_path.stem}_page_{page_num:03d}.png"
            image_path = Path("yolo_processed_data/images") / image_filename
            image_path.parent.mkdir(exist_ok=True)
            pix.save(str(image_path))
            image_paths.append(image_path)
            
            # Extract features immediately with enhanced processing
            features = extract_features_from_image_worker(pix)
            features['source_pdf'] = pdf_path.name
            features['page_number'] = page_num
            features['processing_quality'] = '4x_ultra_high'
            
            # Save features
            features_filename = f"{pdf_path.stem}_page_{page_num:03d}_features.json"
            features_path = Path("yolo_processed_data/features") / features_filename
            features_path.parent.mkdir(exist_ok=True)
            with open(features_path, 'w') as f:
                json.dump(features, f, indent=2)
        
        doc.close()
        
        # Save metadata
        metadata_path = Path("yolo_processed_data/metadata") / f"{pdf_path.stem}_metadata.json"
        metadata_path.parent.mkdir(exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Completed {pdf_path.name}: {len(image_paths)} pages")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error processing {pdf_path.name}: {e}")
        return False

def extract_features_from_image_worker(pix):
    """Standalone function for feature extraction - enhanced for 32GB RAM."""
    # Convert pixmap to numpy array
    img_data = pix.tobytes("ppm")
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    features = {
        'width': img.shape[1],
        'height': img.shape[0],
        'aspect_ratio': img.shape[1] / img.shape[0],
        'mean_color': np.mean(img, axis=(0, 1)).tolist(),
        'color_variance': np.var(img),
        'total_pixels': img.shape[0] * img.shape[1],
    }
    
    # Enhanced features for 32GB machine
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection with multiple thresholds
    edges_low = cv2.Canny(gray, 30, 100)
    edges_high = cv2.Canny(gray, 50, 150)
    features['edge_density_low'] = np.sum(edges_low > 0) / (edges_low.shape[0] * edges_low.shape[1])
    features['edge_density_high'] = np.sum(edges_high > 0) / (edges_high.shape[0] * edges_high.shape[1])
    
    # Line detection with multiple parameters
    lines_short = cv2.HoughLinesP(edges_high, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
    lines_long = cv2.HoughLinesP(edges_high, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    features['line_count_short'] = len(lines_short) if lines_short is not None else 0
    features['line_count_long'] = len(lines_long) if lines_long is not None else 0
    
    # Contour analysis
    contours, _ = cv2.findContours(edges_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features['contour_count'] = len(contours)
    
    # Additional texture features
    features['texture_variance'] = np.var(gray)
    features['brightness_mean'] = np.mean(gray)
    features['brightness_std'] = np.std(gray)
    
    return features

class YOLOProcessor:
    """YOLO processor optimized for GitHub Codespaces 32GB machine."""
    
    def __init__(self):
        self.as_built_dir = Path("as_built_drawings")
        self.output_dir = Path("yolo_processed_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # YOLO settings - optimized for 32GB RAM
        self.max_workers = min(mp.cpu_count(), 16)  # Cap at 16 workers for stability
        self.image_quality = 4.0  # Maximum quality for 32GB
        self.max_pages_per_pdf = 200  # Increased page limit
        self.batch_size = 20  # Process 20 PDFs simultaneously
        
        # Create subdirectories
        for subdir in ["images", "metadata", "features", "reports"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def get_system_info(self):
        """Get system information for 32GB machine."""
        memory = psutil.virtual_memory()
        cpu_count = mp.cpu_count()
        
        logger.info(f"🚀 YOLO Processing Environment - GitHub Codespaces 32GB:")
        logger.info(f"  CPU Cores: {cpu_count}")
        logger.info(f"  Total RAM: {memory.total / (1024**3):.1f} GB")
        logger.info(f"  Available RAM: {memory.available / (1024**3):.1f} GB")
        logger.info(f"  Memory Usage: {memory.percent:.1f}%")
        logger.info(f"  Workers: {self.max_workers}")
        logger.info(f"  Image Quality: {self.image_quality}x (Ultra High)")
        logger.info(f"  Batch Size: {self.batch_size}")
        logger.info(f"  Max Pages per PDF: {self.max_pages_per_pdf}")
        
        # Check if we're in GitHub Codespaces
        if os.environ.get('CODESPACES'):
            logger.info(f"  Environment: GitHub Codespaces")
            logger.info(f"  Machine Type: 32GB RAM")
        else:
            logger.info(f"  Environment: Local Machine")
    
    def run_yolo_processing(self):
        """Run YOLO processing on all files with 32GB optimization."""
        logger.info("🚀 Starting YOLO processing for GitHub Codespaces 32GB...")
        self.get_system_info()
        
        pdf_files = list(self.as_built_dir.glob("*.pdf"))
        logger.info(f"📁 Found {len(pdf_files)} PDF files for YOLO processing")
        
        # Sort files by size for better resource management
        pdf_files.sort(key=lambda f: f.stat().st_size, reverse=True)
        
        start_time = time.time()
        processed = 0
        failed = 0
        
        # Process in parallel with optimized workers
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs using string paths
            future_to_pdf = {executor.submit(process_single_pdf_worker, str(pdf)): pdf for pdf in pdf_files}
            
            # Process results as they complete
            for future in as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    result = future.result()
                    if result:
                        processed += 1
                        logger.info(f"✅ Progress: {processed}/{len(pdf_files)} files completed")
                    else:
                        failed += 1
                        logger.warning(f"❌ Progress: {failed} files failed so far")
                except Exception as e:
                    logger.error(f"💥 Exception processing {pdf.name}: {e}")
                    failed += 1
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        self.generate_yolo_report(processed, failed, total_time, len(pdf_files))
        
        logger.info(f"🎉 YOLO Processing Complete!")
        logger.info(f"  ✅ Processed: {processed}")
        logger.info(f"  ❌ Failed: {failed}")
        logger.info(f"  ⏱️  Total Time: {total_time/60:.1f} minutes")
        logger.info(f"  📊 Average: {total_time/len(pdf_files):.1f} seconds per file")
        logger.info(f"  🚀 Success Rate: {processed/len(pdf_files)*100:.1f}%")
    
    def generate_yolo_report(self, processed, failed, total_time, total_files):
        """Generate comprehensive YOLO processing report."""
        report = {
            'yolo_processing_summary': {
                'total_files': total_files,
                'processed_files': processed,
                'failed_files': failed,
                'success_rate': processed / total_files * 100,
                'total_time_minutes': total_time / 60,
                'average_time_per_file': total_time / total_files,
                'processing_date': datetime.now().isoformat(),
                'processing_environment': 'github_codespaces_32gb',
                'image_quality': '4x_ultra_high'
            },
            'system_performance': {
                'cpu_cores': mp.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'workers_used': self.max_workers,
                'batch_size': self.batch_size,
                'max_pages_per_pdf': self.max_pages_per_pdf
            },
            'performance_metrics': {
                'files_per_minute': processed / (total_time / 60),
                'pages_per_second': 'calculated_from_metadata',
                'memory_efficiency': 'optimized_for_32gb'
            }
        }
        
        report_path = self.output_dir / "reports" / "yolo_processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 YOLO report saved to {report_path}")

def main():
    """Main function for YOLO processing on GitHub Codespaces."""
    print("🚀 YOLO Processing - GitHub Codespaces 32GB Machine")
    print("=" * 60)
    print("Optimized for maximum speed processing with cloud resources")
    print("=" * 60)
    
    processor = YOLOProcessor()
    processor.run_yolo_processing()

if __name__ == "__main__":
    main()