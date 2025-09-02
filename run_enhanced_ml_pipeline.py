#!/usr/bin/env python3
"""
Enhanced ML Pipeline Runner
Comprehensive script to run the entire enhanced ML pipeline from data processing to model training.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ml_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EnhancedMLPipeline:
    """Enhanced ML pipeline runner."""
    
    def __init__(self):
        self.start_time = time.time()
        self.pipeline_steps = []
        self.step_results = {}
        
    def log_step(self, step_name: str, status: str, details: str = ""):
        """Log pipeline step with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        duration = time.time() - self.start_time
        
        step_info = {
            'step': step_name,
            'status': status,
            'timestamp': timestamp,
            'duration': duration,
            'details': details
        }
        
        self.pipeline_steps.append(step_info)
        self.step_results[step_name] = status
        
        logger.info(f"[{timestamp}] {step_name}: {status}")
        if details:
            logger.info(f"  Details: {details}")
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        logger.info("Checking dependencies...")
        
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
            logger.error("Please install missing packages:")
            logger.error("pip install opencv-python numpy scikit-learn xgboost joblib PyMuPDF pandas")
            return False
        
        logger.info("All dependencies available")
        return True
    
    def check_data_availability(self) -> bool:
        """Check if as-built data is available."""
        logger.info("Checking data availability...")
        
        as_built_dir = Path("as_built_drawings")
        
        if not as_built_dir.exists():
            logger.error(f"Directory {as_built_dir} does not exist!")
            logger.error("Please ensure as-built PDF files are in the as_built_drawings/ directory")
            return False
        
        pdf_files = list(as_built_dir.glob("*.pdf"))
        
        if len(pdf_files) == 0:
            logger.error("No PDF files found in as_built_drawings/ directory!")
            logger.error("Please add your as-built PDF files to the directory")
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF files in {as_built_dir}")
        return True
    
    def run_data_processing(self) -> bool:
        """Run the data processing step."""
        logger.info("Starting data processing...")
        
        try:
            # Run the data processing script
            result = subprocess.run(
                [sys.executable, "process_all_as_builts.py"],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("Data processing completed successfully")
                return True
            else:
                logger.error(f"Data processing failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Data processing timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Error running data processing: {e}")
            return False
    
    def run_enhanced_training(self) -> bool:
        """Run the enhanced ML training step."""
        logger.info("Starting enhanced ML training...")
        
        try:
            # Run the enhanced training script
            result = subprocess.run(
                [sys.executable, "enhanced_ml_training.py"],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("Enhanced ML training completed successfully")
                return True
            else:
                logger.error(f"Enhanced ML training failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Enhanced ML training timed out after 30 minutes")
            return False
        except Exception as e:
            logger.error(f"Error running enhanced ML training: {e}")
            return False
    
    def validate_results(self) -> bool:
        """Validate the pipeline results."""
        logger.info("Validating pipeline results...")
        
        # Check for processed data
        enhanced_data_dir = Path("enhanced_training_data")
        if not enhanced_data_dir.exists():
            logger.error("Enhanced training data directory not found!")
            return False
        
        # Check for images
        images_dir = enhanced_data_dir / "images"
        if not images_dir.exists():
            logger.error("Images directory not found!")
            return False
        
        image_count = len(list(images_dir.rglob("*.png")))
        if image_count == 0:
            logger.error("No processed images found!")
            return False
        
        # Check for features
        features_dir = enhanced_data_dir / "features"
        if not features_dir.exists():
            logger.error("Features directory not found!")
            return False
        
        feature_count = len(list(features_dir.glob("*.json")))
        if feature_count == 0:
            logger.error("No feature files found!")
            return False
        
        # Check for models
        models_dir = Path("enhanced_models")
        if not models_dir.exists():
            logger.error("Enhanced models directory not found!")
            return False
        
        model_count = len(list(models_dir.glob("*.joblib")))
        if model_count == 0:
            logger.error("No trained models found!")
            return False
        
        logger.info(f"Validation successful:")
        logger.info(f"  Processed images: {image_count}")
        logger.info(f"  Feature files: {feature_count}")
        logger.info(f"  Trained models: {model_count}")
        
        return True
    
    def generate_pipeline_report(self):
        """Generate comprehensive pipeline report."""
        logger.info("Generating pipeline report...")
        
        total_duration = time.time() - self.start_time
        
        report = {
            'pipeline_summary': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'total_steps': len(self.pipeline_steps),
                'successful_steps': sum(1 for step in self.pipeline_steps if step['status'] == 'SUCCESS'),
                'failed_steps': sum(1 for step in self.pipeline_steps if step['status'] == 'FAILED')
            },
            'step_details': self.pipeline_steps,
            'overall_status': 'SUCCESS' if all(status == 'SUCCESS' for status in self.step_results.values()) else 'FAILED'
        }
        
        # Save report
        report_path = Path("enhanced_ml_pipeline_report.json")
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Pipeline report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("ENHANCED ML PIPELINE SUMMARY")
        print("="*80)
        print(f"Total Duration: {total_duration/60:.1f} minutes")
        print(f"Steps Completed: {report['pipeline_summary']['successful_steps']}/{report['pipeline_summary']['total_steps']}")
        print(f"Overall Status: {report['overall_status']}")
        
        print("\nStep Details:")
        for step in self.pipeline_steps:
            status_icon = "[OK]" if step['status'] == 'SUCCESS' else "[ERROR]"
            print(f"  {status_icon} {step['step']} ({step['timestamp']})")
            if step['details']:
                print(f"     {step['details']}")
        
        if report['overall_status'] == 'SUCCESS':
            print("\nPipeline completed successfully!")
            print("Check enhanced_training_data/ for processed data")
            print("Check enhanced_models/ for trained models")
            print("Review enhanced_ml_pipeline_report.json for detailed results")
        else:
            print("\nPipeline completed with errors!")
            print("Check enhanced_ml_pipeline.log for error details")
    
    def run_pipeline(self, skip_processing: bool = False, skip_training: bool = False):
        """Run the complete enhanced ML pipeline."""
        print("Enhanced ML Pipeline Starting...")
        print("="*80)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            self.log_step("Dependency Check", "FAILED", "Missing required packages")
            return False
        self.log_step("Dependency Check", "SUCCESS")
        
        # Step 2: Check data availability
        if not self.check_data_availability():
            self.log_step("Data Availability Check", "FAILED", "No as-built PDF files found")
            return False
        self.log_step("Data Availability Check", "SUCCESS")
        
        # Step 3: Data processing (optional)
        if not skip_processing:
            if not self.run_data_processing():
                self.log_step("Data Processing", "FAILED", "Error during PDF processing")
                return False
            self.log_step("Data Processing", "SUCCESS")
        else:
            self.log_step("Data Processing", "SKIPPED", "User requested skip")
        
        # Step 4: Enhanced ML training (optional)
        if not skip_training:
            if not self.run_enhanced_training():
                self.log_step("Enhanced ML Training", "FAILED", "Error during model training")
                return False
            self.log_step("Enhanced ML Training", "SUCCESS")
        else:
            self.log_step("Enhanced ML Training", "SKIPPED", "User requested skip")
        
        # Step 5: Validate results
        if not self.validate_results():
            self.log_step("Results Validation", "FAILED", "Pipeline outputs not found")
            return False
        self.log_step("Results Validation", "SUCCESS")
        
        # Step 6: Generate report
        self.generate_pipeline_report()
        self.log_step("Report Generation", "SUCCESS")
        
        return True

def main():
    """Main function for the enhanced ML pipeline."""
    parser = argparse.ArgumentParser(description="Enhanced ML Pipeline Runner")
    parser.add_argument("--skip-processing", action="store_true", 
                       help="Skip data processing step (use existing processed data)")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip ML training step")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check dependencies and data availability")
    
    args = parser.parse_args()
    
    pipeline = EnhancedMLPipeline()
    
    if args.check_only:
        print("Checking pipeline prerequisites...")
        deps_ok = pipeline.check_dependencies()
        data_ok = pipeline.check_data_availability()
        
        if deps_ok and data_ok:
            print("All prerequisites met - ready to run pipeline")
        else:
            print("Prerequisites not met - please fix issues above")
        return
    
    # Run the pipeline
    success = pipeline.run_pipeline(
        skip_processing=args.skip_processing,
        skip_training=args.skip_training
    )
    
    if success:
        print("\nEnhanced ML Pipeline completed successfully!")
    else:
        print("\nEnhanced ML Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
