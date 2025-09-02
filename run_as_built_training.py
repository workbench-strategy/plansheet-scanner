#!/usr/bin/env python3
"""
Quick CLI to run as-built training
Simple interface to start training with different options.
"""

import argparse
import sys
import subprocess
from pathlib import Path

def run_training(use_real_data: bool = True, min_examples: int = 3, verbose: bool = False):
    """Run the as-built training process."""
    
    print("🚀 Starting As-Built Training")
    print("=" * 50)
    
    if use_real_data:
        print("📊 Using real as-built data")
        script = "start_as_built_training.py"
    else:
        print("🧪 Using sample data for testing")
        script = "feed_ml_system.py"
    
    # Check if script exists
    if not Path(script).exists():
        print(f"❌ Error: {script} not found")
        return False
    
    # Build command
    cmd = [sys.executable, script]
    
    if verbose:
        print(f"🔧 Running: {' '.join(cmd)}")
    
    try:
        # Run the training script
        result = subprocess.run(cmd, check=True, capture_output=not verbose, text=True)
        
        if not verbose and result.stdout:
            print("📋 Output:")
            print(result.stdout)
        
        print("✅ Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code {e.returncode}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ Error: Could not find {script}")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "cv2",
        "numpy", 
        "sklearn",
        "joblib"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies available")
    return True

def show_status():
    """Show current training status."""
    print("📊 Training Status")
    print("=" * 30)
    
    # Check directories
    directories = [
        "real_as_built_data",
        "training_data", 
        "models"
    ]
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            files = len(list(path.rglob("*")))
            print(f"✅ {directory}: {files} files")
        else:
            print(f"❌ {directory}: not found")
    
    # Check for log file
    log_file = Path("as_built_training.log")
    if log_file.exists():
        print(f"📋 Training log: {log_file}")
    else:
        print("📋 Training log: not found")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Run as-built training")
    parser.add_argument("--sample", action="store_true", help="Use sample data instead of real data")
    parser.add_argument("--min-examples", type=int, default=3, help="Minimum examples for training")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--check", action="store_true", help="Check dependencies")
    parser.add_argument("--status", action="store_true", help="Show training status")
    
    args = parser.parse_args()
    
    if args.check:
        return 0 if check_dependencies() else 1
    
    if args.status:
        show_status()
        return 0
    
    # Check dependencies before running
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return 1
    
    # Run training
    success = run_training(
        use_real_data=not args.sample,
        min_examples=args.min_examples,
        verbose=args.verbose
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
