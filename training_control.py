#!/usr/bin/env python3
"""
Training Control Interface
Simple interface to control and monitor the background training service.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class TrainingController:
    """Control interface for background training service."""
    
    def __init__(self):
        self.service_script = "background_training_service.py"
        self.status_file = "logs/training_status.json"
        self.config_file = "config/training_config.json"
        self.process = None
    
    def start_service(self, background: bool = True):
        """Start the background training service."""
        if self.is_running():
            print("⚠️  Service is already running")
            return False
        
        try:
            if background:
                # Start in background (Windows)
                if os.name == 'nt':
                    self.process = subprocess.Popen(
                        [sys.executable, self.service_script],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                else:
                    # Start in background (Unix/Linux)
                    self.process = subprocess.Popen(
                        [sys.executable, self.service_script],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                
                print("✅ Background training service started")
                print("   - Service will run automatically when system is idle")
                print("   - Check logs/background_training.log for details")
                return True
            else:
                # Start in foreground
                print("🤖 Starting training service in foreground...")
                subprocess.run([sys.executable, self.service_script])
                return True
                
        except Exception as e:
            print(f"❌ Error starting service: {e}")
            return False
    
    def stop_service(self):
        """Stop the background training service."""
        if not self.is_running():
            print("⚠️  Service is not running")
            return False
        
        try:
            # Try graceful shutdown first
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                    print("✅ Service stopped gracefully")
                    return True
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    print("⚠️  Service force-killed")
                    return True
            
            # If no process handle, try to find and kill by script name
            for proc in self._find_service_processes():
                proc.terminate()
                print(f"✅ Terminated process {proc.pid}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error stopping service: {e}")
            return False
    
    def _find_service_processes(self):
        """Find running service processes."""
        import psutil
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any(self.service_script in arg for arg in cmdline):
                    processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def is_running(self) -> bool:
        """Check if the service is running."""
        # Check status file first
        if Path(self.status_file).exists():
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                if status.get("is_running", False):
                    return True
            except:
                pass
        
        # Check for running processes
        return len(self._find_service_processes()) > 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        if not Path(self.status_file).exists():
            return {"is_running": False, "error": "Status file not found"}
        
        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)
            return status
        except Exception as e:
            return {"is_running": False, "error": str(e)}
    
    def show_status(self):
        """Display current training status."""
        status = self.get_status()
        
        print("🤖 Background Training Service Status")
        print("=" * 50)
        
        if status.get("error"):
            print(f"❌ Error: {status['error']}")
            return
        
        # Service status
        is_running = status.get("is_running", False)
        print(f"Service Status: {'🟢 Running' if is_running else '🔴 Stopped'}")
        
        # Last training time
        last_training = status.get("last_training_time")
        if last_training:
            try:
                last_time = datetime.fromisoformat(last_training)
                time_diff = datetime.now() - last_time
                print(f"Last Training: {last_time.strftime('%Y-%m-%d %H:%M:%S')} ({time_diff.days}d {time_diff.seconds//3600}h ago)")
            except:
                print(f"Last Training: {last_training}")
        else:
            print("Last Training: Never")
        
        # Training statistics
        training_stats = status.get("training_stats", {})
        if training_stats:
            print(f"\n📊 Training Statistics:")
            print(f"   Total Drawings: {training_stats.get('total_drawings', 0)}")
            print(f"   Review Patterns: {training_stats.get('review_patterns', 0)}")
            print(f"   Code Violations: {training_stats.get('total_violations', 0)}")
            print(f"   Design Errors: {training_stats.get('total_errors', 0)}")
            print(f"   Models Trained: {training_stats.get('models_trained', False)}")
            
            # Discipline distribution
            discipline_dist = training_stats.get('discipline_distribution', {})
            if discipline_dist:
                print(f"   Discipline Distribution:")
                for discipline, count in discipline_dist.items():
                    print(f"     {discipline}: {count}")
        
        # Performance statistics
        perf_stats = status.get("performance_stats", {})
        if perf_stats:
            print(f"\n⚡ Performance Statistics:")
            print(f"   Avg CPU Usage: {perf_stats.get('avg_cpu', 0):.1f}%")
            print(f"   Avg Memory Usage: {perf_stats.get('avg_memory', 0):.1f}%")
            print(f"   Avg Training Time: {perf_stats.get('avg_training_time', 0):.1f}s")
    
    def show_logs(self, lines: int = 20):
        """Show recent training logs."""
        log_file = "logs/background_training.log"
        
        if not Path(log_file).exists():
            print("❌ No log file found")
            return
        
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
            
            print(f"📋 Recent Training Logs (last {lines} lines):")
            print("=" * 60)
            
            for line in all_lines[-lines:]:
                print(line.rstrip())
                
        except Exception as e:
            print(f"❌ Error reading logs: {e}")
    
    def force_training(self):
        """Force an immediate training session."""
        if not self.is_running():
            print("❌ Service is not running. Start it first.")
            return False
        
        print("🔄 Forcing immediate training session...")
        
        # This would require implementing a way to communicate with the running service
        # For now, we'll just show a message
        print("⚠️  Force training feature requires service communication implementation")
        print("   The service will automatically train when the system is idle")
        return False
    
    def show_config(self):
        """Show current training configuration."""
        if not Path(self.config_file).exists():
            print("❌ Configuration file not found")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            print("⚙️  Training Configuration:")
            print("=" * 40)
            
            # Training schedule
            schedule = config.get("training_schedule", {})
            print(f"Training Schedule:")
            print(f"   Enabled: {schedule.get('enabled', False)}")
            print(f"   Interval: {schedule.get('interval_hours', 6)} hours")
            print(f"   Max Time: {schedule.get('max_training_time_minutes', 120)} minutes")
            print(f"   Idle Threshold: {schedule.get('idle_threshold_percent', 30)}%")
            
            # Model config
            model_config = config.get("model_config", {})
            print(f"\nModel Configuration:")
            print(f"   Min Examples: {model_config.get('min_examples', 50)}")
            print(f"   Max Examples: {model_config.get('max_examples', 1000)}")
            print(f"   Validation Split: {model_config.get('validation_split', 0.2)}")
            
            # Data management
            data_mgmt = config.get("data_management", {})
            print(f"\nData Management:")
            print(f"   Data Directory: {data_mgmt.get('data_dir', 'improved_ai_data')}")
            print(f"   Model Directory: {data_mgmt.get('model_dir', 'improved_ai_models')}")
            print(f"   Max Files: {data_mgmt.get('max_data_files', 1000)}")
            
        except Exception as e:
            print(f"❌ Error reading configuration: {e}")

def main():
    """Main function for the training control interface."""
    parser = argparse.ArgumentParser(description="Control background training service")
    parser.add_argument("command", choices=["start", "stop", "status", "logs", "force", "config"],
                       help="Command to execute")
    parser.add_argument("--foreground", "-f", action="store_true",
                       help="Start service in foreground (for debugging)")
    parser.add_argument("--lines", "-n", type=int, default=20,
                       help="Number of log lines to show (default: 20)")
    
    args = parser.parse_args()
    
    controller = TrainingController()
    
    if args.command == "start":
        controller.start_service(background=not args.foreground)
    elif args.command == "stop":
        controller.stop_service()
    elif args.command == "status":
        controller.show_status()
    elif args.command == "logs":
        controller.show_logs(args.lines)
    elif args.command == "force":
        controller.force_training()
    elif args.command == "config":
        controller.show_config()

if __name__ == "__main__":
    main()
