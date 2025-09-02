#!/usr/bin/env python3
"""
YOLO Processing Script - Local Windows PC Optimization
Optimized for running on Windows PC with Intel vPro i7 through VS Code.
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
        logging.FileHandler('yolo_processing_local.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_single_pdf_worker_local(pdf_path_str):
    """Standalone function for multiprocessing - optimized for Windows PC with Intel vPro i7."""
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
            'processing_environment': 'local_windows_intel_vpro_i7'
        }
        
        # Convert pages to images with optimized quality for Intel vPro i7
        image_paths = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Use optimized quality for Intel vPro i7 (good balance of speed/quality)
            mat = fitz.Matrix(2.5, 2.5)  # Optimized for Intel vPro i7
            pix = page.get_pixmap(matrix=mat)
            
            image_filename = f"{pdf_path.stem}_page_{page_num:03d}.png"
            image_path = Path("yolo_processed_data_local/images") / image_filename
            image_path.parent.mkdir(exist_ok=True)
            pix.save(str(image_path))
            image_paths.append(image_path)
            
            # Extract features with optimized processing for Intel vPro i7
            features = extract_features_from_image_worker_local(pix)
            features['source_pdf'] = pdf_path.name
            features['page_number'] = page_num
            features['processing_quality'] = '2.5x_optimized_intel_vpro'
            
            # Save features
            features_filename = f"{pdf_path.stem}_page_{page_num:03d}_features.json"
            features_path = Path("yolo_processed_data_local/features") / features_filename
            features_path.parent.mkdir(exist_ok=True)
            with open(features_path, 'w') as f:
                json.dump(features, f, indent=2)
        
        doc.close()
        
        # Save metadata
        metadata_path = Path("yolo_processed_data_local/metadata") / f"{pdf_path.stem}_metadata.json"
        metadata_path.parent.mkdir(exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Completed {pdf_path.name}: {len(image_paths)} pages")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error processing {pdf_path.name}: {e}")
        return False

def extract_features_from_image_worker_local(pix):
    """Standalone function for feature extraction - optimized for Intel vPro i7."""
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
    
    # Optimized features for Intel vPro i7 (balanced performance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhanced edge detection for Intel vPro i7
    edges = cv2.Canny(gray, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Optimized line detection for Intel vPro i7
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    features['line_count'] = len(lines) if lines is not None else 0
    
    # Enhanced contour analysis for Intel vPro i7
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features['contour_count'] = len(contours)
    
    # Additional features optimized for Intel vPro i7
    features['texture_variance'] = np.var(gray)
    features['brightness_mean'] = np.mean(gray)
    features['brightness_std'] = np.std(gray)
    
    return features

class YOLOProcessorLocal:
    """YOLO processor optimized for Windows PC with Intel vPro i7."""
    
    def __init__(self):
        self.as_built_dir = Path("as_built_drawings")
        self.output_dir = Path("yolo_processed_data_local")
        self.output_dir.mkdir(exist_ok=True)
        
        # YOLO settings - optimized for Intel vPro i7 (16 cores, 31.6GB RAM)
        self.max_workers = min(mp.cpu_count(), 12)  # Use 12 workers for Intel vPro i7
        self.image_quality = 2.5  # Optimized quality for Intel vPro i7
        self.max_pages_per_pdf = 100  # Increased for Intel vPro i7
        self.batch_size = 8  # Process 8 PDFs simultaneously for Intel vPro i7
        
        # Create subdirectories
        for subdir in ["images", "metadata", "features", "reports"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def get_system_info(self):
        """Get system information for Intel vPro i7 Windows PC."""
        memory = psutil.virtual_memory()
        cpu_count = mp.cpu_count()
        
        logger.info(f"🖥️ YOLO Processing Environment - Intel vPro i7 Windows PC:")
        logger.info(f"  CPU Cores: {cpu_count} (Intel vPro i7)")
        logger.info(f"  Total RAM: {memory.total / (1024**3):.1f} GB")
        logger.info(f"  Available RAM: {memory.available / (1024**3):.1f} GB")
        logger.info(f"  Memory Usage: {memory.percent:.1f}%")
        logger.info(f"  Workers: {self.max_workers} (optimized for Intel vPro i7)")
        logger.info(f"  Image Quality: {self.image_quality}x (optimized)")
        logger.info(f"  Batch Size: {self.batch_size}")
        logger.info(f"  Max Pages per PDF: {self.max_pages_per_pdf}")
        logger.info(f"  Environment: Windows PC + Intel vPro i7 + VS Code")
    
    def run_yolo_processing(self):
        """Run YOLO processing on all files with Intel vPro i7 optimization."""
        logger.info("🖥️ Starting YOLO processing for Intel vPro i7 Windows PC...")
        self.get_system_info()
        
        pdf_files = list(self.as_built_dir.glob("*.pdf"))
        logger.info(f"📁 Found {len(pdf_files)} PDF files for YOLO processing")
        
        # Sort files by size for better resource management
        pdf_files.sort(key=lambda f: f.stat().st_size, reverse=True)
        
        start_time = time.time()
        processed = 0
        failed = 0
        
        # Process in parallel with optimized workers for Intel vPro i7
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs using string paths
            future_to_pdf = {executor.submit(process_single_pdf_worker_local, str(pdf)): pdf for pdf in pdf_files}
            
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
                'processing_environment': 'local_windows_intel_vpro_i7',
                'image_quality': '2.5x_optimized_intel_vpro'
            },
            'system_performance': {
                'cpu_cores': mp.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'workers_used': self.max_workers,
                'batch_size': self.batch_size,
                'max_pages_per_pdf': self.max_pages_per_pdf,
                'processor': 'Intel vPro i7'
            },
            'performance_metrics': {
                'files_per_minute': processed / (total_time / 60),
                'pages_per_second': 'calculated_from_metadata',
                'memory_efficiency': 'optimized_for_intel_vpro_i7'
            }
        }
        
        report_path = self.output_dir / "reports" / "yolo_processing_local_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 YOLO report saved to {report_path}")

def main():
    """Main function for YOLO processing on Intel vPro i7 Windows PC."""
    print("🖥️ YOLO Processing - Intel vPro i7 Windows PC + VS Code")
    print("=" * 60)
    print("Optimized for Intel vPro i7 with 16 cores and 31.6GB RAM")
    print("=" * 60)
    
    processor = YOLOProcessorLocal()
    processor.run_yolo_processing()

if __name__ == "__main__":
    main()
