#!/usr/bin/env python3
"""
Symbol Training Control Interface
Simple interface to control and monitor the symbol recognition background training service.
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

class SymbolTrainingController:
    """Control interface for symbol recognition background training service."""
    
    def __init__(self):
        self.service_script = "symbol_background_trainer.py"
        self.status_file = "logs/symbol_training_status.json"
        self.config_file = "config/symbol_training_config.json"
        self.process = None
    
    def start_service(self, background: bool = True):
        """Start the symbol background training service."""
        if self.is_running():
            print("⚠️  Symbol training service is already running")
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
                
                print("✅ Symbol background training service started")
                print("   - Service will learn symbols from plans, legends, and notes")
                print("   - Training runs automatically when system is idle")
                print("   - Check logs/symbol_training.log for details")
                return True
            else:
                # Start in foreground
                print("🤖 Starting symbol training service in foreground...")
                subprocess.run([sys.executable, self.service_script])
                return True
                
        except Exception as e:
            print(f"❌ Error starting symbol training service: {e}")
            return False
    
    def stop_service(self):
        """Stop the symbol background training service."""
        if not self.is_running():
            print("⚠️  Symbol training service is not running")
            return False
        
        try:
            # Try graceful shutdown first
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                    print("✅ Symbol training service stopped gracefully")
                    return True
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    print("⚠️  Symbol training service force-killed")
                    return True
            
            # If no process handle, try to find and kill by script name
            for proc in self._find_service_processes():
                proc.terminate()
                print(f"✅ Terminated symbol training process {proc.pid}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error stopping symbol training service: {e}")
            return False
    
    def _find_service_processes(self):
        """Find running symbol training service processes."""
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
        """Check if the symbol training service is running."""
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
        """Get current symbol training status."""
        if not Path(self.status_file).exists():
            return {"is_running": False, "error": "Status file not found"}
        
        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)
            return status
        except Exception as e:
            return {"is_running": False, "error": str(e)}
    
    def show_status(self):
        """Display current symbol training status."""
        status = self.get_status()
        
        print("�� Symbol Recognition Training Service Status")
        print("=" * 60)
        
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
            print(f"\n📊 Symbol Training Statistics:")
            print(f"   Total Symbols: {training_stats.get('total_symbols', 0)}")
            print(f"   Symbol Patterns: {training_stats.get('symbol_patterns', 0)}")
            print(f"   Average Confidence: {training_stats.get('avg_confidence', 0):.3f}")
            print(f"   Average Usage Frequency: {training_stats.get('avg_usage_frequency', 0):.1f}")
            print(f"   Models Trained: {training_stats.get('models_trained', False)}")
            
            # Discipline distribution
            discipline_dist = training_stats.get('discipline_distribution', {})
            if discipline_dist:
                print(f"   Discipline Distribution:")
                for discipline, count in discipline_dist.items():
                    print(f"     {discipline}: {count}")
            
            # Symbol distribution
            symbol_dist = training_stats.get('symbol_distribution', {})
            if symbol_dist:
                print(f"   Top Symbols:")
                sorted_symbols = sorted(symbol_dist.items(), key=lambda x: x[1], reverse=True)[:5]
                for symbol, count in sorted_symbols:
                    print(f"     {symbol}: {count}")
        
        # Performance statistics
        perf_stats = status.get("performance_stats", {})
        if perf_stats:
            print(f"\n⚡ Performance Statistics:")
            print(f"   Avg CPU Usage: {perf_stats.get('avg_cpu', 0):.1f}%")
            print(f"   Avg Memory Usage: {perf_stats.get('avg_memory', 0):.1f}%")
            print(f"   Avg Training Time: {perf_stats.get('avg_training_time', 0):.1f}s")
    
    def show_logs(self, lines: int = 20):
        """Show recent symbol training logs."""
        log_file = "logs/symbol_training.log"
        
        if not Path(log_file).exists():
            print("❌ No symbol training log file found")
            return
        
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
            
            print(f"📋 Recent Symbol Training Logs (last {lines} lines):")
            print("=" * 70)
            
            for line in all_lines[-lines:]:
                print(line.rstrip())
                
        except Exception as e:
            print(f"❌ Error reading symbol training logs: {e}")
    
    def force_training(self):
        """Force an immediate symbol training session."""
        if not self.is_running():
            print("❌ Symbol training service is not running. Start it first.")
            return False
        
        print("🔄 Forcing immediate symbol training session...")
        
        # This would require implementing a way to communicate with the running service
        # For now, we'll just show a message
        print("⚠️  Force training feature requires service communication implementation")
        print("   The service will automatically train when the system is idle")
        return False
    
    def show_config(self):
        """Show current symbol training configuration."""
        if not Path(self.config_file).exists():
            print("❌ Symbol training configuration file not found")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            print("⚙️  Symbol Training Configuration:")
            print("=" * 50)
            
            # Training schedule
            schedule = config.get("training_schedule", {})
            print(f"Training Schedule:")
            print(f"   Enabled: {schedule.get('enabled', False)}")
            print(f"   Interval: {schedule.get('interval_hours', 4)} hours")
            print(f"   Max Time: {schedule.get('max_training_time_minutes', 60)} minutes")
            print(f"   Idle Threshold: {schedule.get('idle_threshold_percent', 25)}%")
            
            # Model config
            model_config = config.get("model_config", {})
            print(f"\nModel Configuration:")
            print(f"   Min Examples: {model_config.get('min_examples', 30)}")
            print(f"   Max Examples: {model_config.get('max_examples', 500)}")
            print(f"   Validation Split: {model_config.get('validation_split', 0.2)}")
            
            # Symbol specific
            symbol_specific = config.get("symbol_specific", {})
            print(f"\nSymbol-Specific Configuration:")
            print(f"   Disciplines: {', '.join(symbol_specific.get('disciplines', []))}")
            print(f"   Focus Areas: {', '.join(symbol_specific.get('focus_areas', []))}")
            print(f"   Confidence Threshold: {symbol_specific.get('confidence_threshold', 0.8)}")
            
            # Data management
            data_mgmt = config.get("data_management", {})
            print(f"\nData Management:")
            print(f"   Data Directory: {data_mgmt.get('data_dir', 'symbol_training_data')}")
            print(f"   Model Directory: {data_mgmt.get('model_dir', 'symbol_models')}")
            print(f"   Max Files: {data_mgmt.get('max_data_files', 500)}")
            
        except Exception as e:
            print(f"❌ Error reading symbol training configuration: {e}")
    
    def test_symbol_identification(self, symbol_info: Dict[str, Any]):
        """Test symbol identification with current models."""
        if not self.is_running():
            print("❌ Symbol training service is not running. Start it first.")
            return False
        
        print("🧪 Testing symbol identification...")
        print(f"   Symbol Info: {symbol_info}")
        
        # This would require implementing communication with the running service
        # For now, we'll just show a message
        print("⚠️  Symbol identification test requires service communication implementation")
        print("   The service will automatically identify symbols during training")
        return False

def main():
    """Main function for the symbol training control interface."""
    parser = argparse.ArgumentParser(description="Control symbol recognition background training service")
    parser.add_argument("command", choices=["start", "stop", "status", "logs", "force", "config", "test"],
                       help="Command to execute")
    parser.add_argument("--foreground", "-f", action="store_true",
                       help="Start service in foreground (for debugging)")
    parser.add_argument("--lines", "-n", type=int, default=20,
                       help="Number of log lines to show (default: 20)")
    parser.add_argument("--symbol-info", type=str, default='{"visual_description": "traffic signal", "legend_ref": "TS", "notes": "signal installed"}',
                       help="Symbol info for testing (JSON string)")
    
    args = parser.parse_args()
    
    controller = SymbolTrainingController()
    
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
    elif args.command == "test":
        try:
            symbol_info = json.loads(args.symbol_info)
            controller.test_symbol_identification(symbol_info)
        except json.JSONDecodeError:
            print("❌ Invalid JSON in --symbol-info argument")

if __name__ == "__main__":
    main()
