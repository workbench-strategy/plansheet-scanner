#!/usr/bin/env python3
"""
Quick Log Cleanup Script
Cleans up logs when there are failures or to start fresh.
"""

import os
import shutil
from pathlib import Path

def clean_logs():
    """Clean up all log files and status files."""
    print("🧹 Cleaning up logs...")
    
    # Files to clean
    log_files = [
        "logs/symbol_training.log",
        "logs/symbol_performance.log", 
        "logs/as_built_processing.log",
        "logs/symbol_training_status.json"
    ]
    
    cleaned_count = 0
    for log_file in log_files:
        file_path = Path(log_file)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"   ✅ Deleted: {log_file}")
                cleaned_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {log_file}: {e}")
    
    # Clean up processed/failed directories
    dirs_to_clean = [
        "processed_drawings",
        "failed_drawings", 
        "extracted_symbols"
    ]
    
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ Deleted directory: {dir_name}")
                cleaned_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete directory {dir_name}: {e}")
    
    print(f"\n🎉 Cleanup complete! Deleted {cleaned_count} files/directories")
    print("💡 You can now restart the symbol training service fresh")

def reset_training_data():
    """Reset all training data to start completely fresh."""
    print("🔄 Resetting all training data...")
    
    # Directories to reset
    dirs_to_reset = [
        "symbol_training_data",
        "symbol_models"
    ]
    
    reset_count = 0
    for dir_name in dirs_to_reset:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ Reset: {dir_name}")
                reset_count += 1
            except Exception as e:
                print(f"   ❌ Failed to reset {dir_name}: {e}")
    
    print(f"\n🎉 Training data reset complete! Reset {reset_count} directories")
    print("💡 The system will regenerate all training data from scratch")

def quick_clean():
    """Quick clean - just logs and status, keep training data."""
    clean_logs()
    print("\n🚀 Ready to restart! Run: python symbol_training_control.py start")

def full_reset():
    """Full reset - everything including training data."""
    clean_logs()
    reset_training_data()
    print("\n🚀 Complete reset! Run: python symbol_training_control.py start")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "logs":
            clean_logs()
        elif command == "reset":
            reset_training_data()
        elif command == "quick":
            quick_clean()
        elif command == "full":
            full_reset()
        else:
            print("❌ Unknown command. Use: logs, reset, quick, or full")
    else:
        print("🧹 Symbol Training Log Cleanup")
        print("=" * 40)
        print("Commands:")
        print("  python clean_logs.py logs   - Clean just log files")
        print("  python clean_logs.py reset  - Reset training data")
        print("  python clean_logs.py quick  - Quick clean (logs only)")
        print("  python clean_logs.py full   - Full reset (everything)")
        print()
        print("💡 Use 'quick' for most failures, 'full' for complete restart")

