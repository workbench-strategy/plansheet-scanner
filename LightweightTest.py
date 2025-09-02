#!/usr/bin/env python3
"""
Local Test Script for Enhanced ML Pipeline
Lightweight version for laptop testing with 2-3 sample files.
"""

import os
import sys
import time
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_dependencies():
    """Test if all required dependencies are available."""
    logger.info("Testing dependencies...")
    
    required_packages = [
        'cv2', 'numpy', 'sklearn', 'xgboost', 'joblib', 
        'fitz', 'pandas', 'pathlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'fitz':
                import fitz
            elif package == 'sklearn':
                import sklearn
            elif package == 'xgboost':
                import xgboost
            elif package == 'joblib':
                import joblib
            elif package == 'pandas':
                import pandas
            elif package == 'pathlib':
                from pathlib import Path
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install: pip install opencv-python numpy scikit-learn xgboost joblib PyMuPDF pandas")
        return False
    
    logger.info("All dependencies available")
    return True

def test_data_availability():
    """Test if as-built data is available."""
    logger.info("Testing data availability...")
    
    as_built_dir = Path("as_built_drawings")
    
    if not as_built_dir.exists():
        logger.error(f"Directory {as_built_dir} does not exist!")
        return False
    
    pdf_files = list(as_built_dir.glob("*.pdf"))
    
    if len(pdf_files) == 0:
        logger.error("No PDF files found!")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Show file sizes
    total_size = sum(f.stat().st_size for f in pdf_files)
    logger.info(f"Total size: {total_size / (1024*1024):.1f} MB")
    
    # Show largest files
    largest_files = sorted(pdf_files, key=lambda f: f.stat().st_size, reverse=True)[:5]
    logger.info("Largest files:")
    for f in largest_files:
        size_mb = f.stat().st_size / (1024*1024)
        logger.info(f"  {f.name}: {size_mb:.1f} MB")
    
    return True

def test_system_resources():
    """Test system resources."""
    logger.info("Testing system resources...")
    
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        logger.info(f"CPU cores: {cpu_count}")
        logger.info(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        logger.info(f"Memory usage: {memory.percent:.1f}%")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        logger.info(f"Disk space: {disk.free / (1024**3):.1f} GB available")
        
        return True
        
    except ImportError:
        logger.warning("psutil not available, skipping system resource check")
        return True

def test_small_processing():
    """Test processing with 2-3 small files."""
    logger.info("Testing small processing...")
    
    as_built_dir = Path("as_built_drawings")
    pdf_files = list(as_built_dir.glob("*.pdf"))
    
    # Select 2-3 smallest files for testing
    small_files = sorted(pdf_files, key=lambda f: f.stat().st_size)[:3]
    
    logger.info(f"Testing with {len(small_files)} small files:")
    for f in small_files:
        size_mb = f.stat().st_size / (1024*1024)
        logger.info(f"  {f.name}: {size_mb:.1f} MB")
    
    # Test basic PDF reading
    for pdf_file in small_files:
        try:
            import fitz
            doc = fitz.open(pdf_file)
            page_count = len(doc)
            doc.close()
            logger.info(f"✓ {pdf_file.name}: {page_count} pages")
        except Exception as e:
            logger.error(f"✗ {pdf_file.name}: {e}")
    
    return True

def main():
    """Run local tests."""
    print("Local Test for Enhanced ML Pipeline")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Availability", test_data_availability),
        ("System Resources", test_system_resources),
        ("Small Processing", test_small_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            if test_func():
                print(f"✓ {test_name}: PASSED")
                passed += 1
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for local processing.")
        print("\nNext steps:")
        print("1. Run: python process_laptop_optimized.py")
        print("2. Or run: python run_enhanced_ml_pipeline.py --check-only")
    else:
        print("\n❌ Some tests failed. Please fix issues before proceeding.")

if __name__ == "__main__":
    main()