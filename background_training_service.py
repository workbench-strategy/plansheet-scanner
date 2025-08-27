#!/usr/bin/env python3
"""
Background Training Service for Engineering Model
Runs training in the background without interrupting user work.
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import queue
import subprocess
import psutil
import schedule

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the improved trainer
from improved_ai_trainer import ImprovedAIEngineerTrainer

class BackgroundTrainingService:
    """Background service for continuous model training."""
    
    def __init__(self, config_file: str = "config/training_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.is_running = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.last_training_time = None
        self.training_stats = {}
        
        # Threading
        self.training_thread = None
        self.stop_event = threading.Event()
        self.status_queue = queue.Queue()
        
        # Initialize trainer
        self.trainer = ImprovedAIEngineerTrainer(
            data_dir=self.config.get("data_dir", "improved_ai_data"),
            model_dir=self.config.get("model_dir", "improved_ai_models")
        )
        
        # Performance monitoring
        self.performance_stats = {
            "cpu_usage": [],
            "memory_usage": [],
            "training_times": [],
            "model_accuracy": []
        }
        
        self.logger.info("Background Training Service initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        default_config = {
            "training_schedule": {
                "enabled": True,
                "interval_hours": 6,
                "max_training_time_minutes": 120,
                "idle_threshold_percent": 30
            },
            "data_management": {
                "data_dir": "improved_ai_data",
                "model_dir": "improved_ai_models",
                "backup_interval_hours": 24,
                "max_data_files": 1000
            },
            "model_config": {
                "min_examples": 50,
                "max_examples": 1000,
                "validation_split": 0.2,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "monitoring": {
                "log_level": "INFO",
                "log_file": "logs/background_training.log",
                "status_file": "logs/training_status.json",
                "performance_log": "logs/performance.log"
            },
            "notifications": {
                "enabled": False,
                "email": "",
                "webhook_url": ""
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                self._merge_configs(default_config, user_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        return default_config
    
    def _merge_configs(self, default: Dict, user: Dict):
        """Recursively merge user config with defaults."""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        log_config = self.config["monitoring"]
        log_file = Path(log_config["log_file"])
        log_file.parent.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config["log_level"]),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("BackgroundTraining")
        
        # Create performance logger
        perf_log_file = Path(log_config["performance_log"])
        perf_log_file.parent.mkdir(exist_ok=True)
        
        self.perf_logger = logging.getLogger("Performance")
        perf_handler = logging.FileHandler(perf_log_file)
        perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
    
    def start(self):
        """Start the background training service."""
        if self.is_running:
            self.logger.warning("Service is already running")
            return
        
        self.logger.info("Starting background training service...")
        self.is_running = True
        self.stop_event.clear()
        
        # Start training thread
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        # Schedule regular training
        self._setup_schedule()
        
        self.logger.info("Background training service started successfully")
        self._save_status()
    
    def stop(self):
        """Stop the background training service."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping background training service...")
        self.is_running = False
        self.stop_event.set()
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=30)
        
        self.logger.info("Background training service stopped")
        self._save_status()
    
    def _training_loop(self):
        """Main training loop that runs in background."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Check if system is idle enough for training
                if self._should_train():
                    self.logger.info("Starting training session...")
                    self._run_training_session()
                else:
                    self.logger.debug("System busy, skipping training session")
                
                # Wait before next check
                time.sleep(self.config["training_schedule"]["interval_hours"] * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _should_train(self) -> bool:
        """Check if system is idle enough for training."""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Check if any intensive processes are running
            intensive_processes = self._check_intensive_processes()
            
            idle_threshold = self.config["training_schedule"]["idle_threshold_percent"]
            
            should_train = (
                cpu_percent < idle_threshold and
                memory_percent < 80 and
                not intensive_processes
            )
            
            self.logger.debug(f"System check - CPU: {cpu_percent}%, Memory: {memory_percent}%, Intensive processes: {intensive_processes}, Should train: {should_train}")
            
            return should_train
            
        except Exception as e:
            self.logger.error(f"Error checking system status: {e}")
            return False
    
    def _check_intensive_processes(self) -> bool:
        """Check if any intensive processes are running."""
        intensive_keywords = [
            "photoshop", "illustrator", "autocad", "revit", "civil3d",
            "microsoft word", "excel", "powerpoint", "outlook",
            "chrome", "firefox", "edge", "safari",
            "visual studio", "pycharm", "vscode"
        ]
        
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                proc_name = proc.info['name'].lower()
                if any(keyword in proc_name for keyword in intensive_keywords):
                    if proc.cpu_percent() > 10:  # Only consider if using significant CPU
                        return True
        except Exception as e:
            self.logger.error(f"Error checking processes: {e}")
        
        return False
    
    def _run_training_session(self):
        """Run a single training session."""
        start_time = time.time()
        max_time = self.config["training_schedule"]["max_training_time_minutes"] * 60
        
        try:
            # Generate training data if needed
            current_count = len(self.trainer.as_built_drawings)
            target_count = self.config["model_config"]["max_examples"]
            
            if current_count < target_count:
                self.logger.info(f"Generating additional training data ({current_count} -> {target_count})")
                additional_examples = min(target_count - current_count, 100)  # Generate in batches
                self.trainer.generate_diverse_training_data(num_examples=additional_examples)
            
            # Train models
            self.logger.info("Training models...")
            self.trainer.train_models(min_examples=self.config["model_config"]["min_examples"])
            
            # Record training statistics
            training_time = time.time() - start_time
            self.training_stats = self.trainer.get_training_statistics()
            self.training_stats["last_training_time"] = training_time
            self.training_stats["timestamp"] = datetime.now().isoformat()
            
            self.last_training_time = datetime.now()
            
            # Log performance metrics
            self._log_performance_metrics(training_time)
            
            self.logger.info(f"Training session completed in {training_time:.2f} seconds")
            self._save_status()
            
        except Exception as e:
            self.logger.error(f"Error during training session: {e}")
            training_time = time.time() - start_time
            self.logger.error(f"Training session failed after {training_time:.2f} seconds")
    
    def _log_performance_metrics(self, training_time: float):
        """Log performance metrics for monitoring."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Record metrics
            self.performance_stats["cpu_usage"].append(cpu_percent)
            self.performance_stats["memory_usage"].append(memory.percent)
            self.performance_stats["training_times"].append(training_time)
            
            # Keep only last 100 entries
            for key in self.performance_stats:
                if len(self.performance_stats[key]) > 100:
                    self.performance_stats[key] = self.performance_stats[key][-100:]
            
            # Log to performance file
            self.perf_logger.info(
                f"Training completed - Time: {training_time:.2f}s, "
                f"CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, "
                f"Drawings: {self.training_stats.get('total_drawings', 0)}"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {e}")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Update status
                self._save_status()
                
                # Check for data cleanup
                self._cleanup_old_data()
                
                # Wait before next check
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _cleanup_old_data(self):
        """Clean up old training data files."""
        try:
            data_dir = Path(self.config["data_management"]["data_dir"])
            max_files = self.config["data_management"]["max_data_files"]
            
            if data_dir.exists():
                files = list(data_dir.glob("*.json"))
                if len(files) > max_files:
                    # Sort by modification time and remove oldest
                    files.sort(key=lambda x: x.stat().st_mtime)
                    files_to_remove = files[:-max_files]
                    
                    for file in files_to_remove:
                        file.unlink()
                        self.logger.info(f"Removed old data file: {file.name}")
        
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {e}")
    
    def _setup_schedule(self):
        """Setup scheduled training tasks."""
        interval_hours = self.config["training_schedule"]["interval_hours"]
        
        # Schedule regular training
        schedule.every(interval_hours).hours.do(self._scheduled_training)
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        scheduler_thread.start()
    
    def _scheduler_loop(self):
        """Background scheduler loop."""
        while self.is_running and not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _scheduled_training(self):
        """Scheduled training callback."""
        if self.is_running and not self.stop_event.is_set():
            self.logger.info("Scheduled training triggered")
            # The training loop will handle the actual training
    
    def _save_status(self):
        """Save current training status to file."""
        try:
            status_file = Path(self.config["monitoring"]["status_file"])
            status_file.parent.mkdir(exist_ok=True)
            
            status = {
                "is_running": self.is_running,
                "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
                "training_stats": self.training_stats,
                "performance_stats": {
                    "avg_cpu": sum(self.performance_stats["cpu_usage"][-10:]) / max(len(self.performance_stats["cpu_usage"][-10:]), 1),
                    "avg_memory": sum(self.performance_stats["memory_usage"][-10:]) / max(len(self.performance_stats["memory_usage"][-10:]), 1),
                    "avg_training_time": sum(self.performance_stats["training_times"][-10:]) / max(len(self.performance_stats["training_times"][-10:]), 1)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving status: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "is_running": self.is_running,
            "last_training_time": self.last_training_time,
            "training_stats": self.training_stats,
            "performance_stats": self.performance_stats
        }
    
    def force_training(self):
        """Force an immediate training session."""
        if self.is_running:
            self.logger.info("Forcing immediate training session...")
            self._run_training_session()
        else:
            self.logger.warning("Service is not running")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\nReceived shutdown signal. Stopping background training service...")
    if hasattr(signal_handler, 'service'):
        signal_handler.service.stop()
    sys.exit(0)

def main():
    """Main function to run the background training service."""
    print("🤖 Background Training Service for Engineering Model")
    print("=" * 60)
    
    # Create service
    service = BackgroundTrainingService()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal_handler.service = service
    
    try:
        # Start service
        service.start()
        
        print("✅ Background training service is running...")
        print("   - Training will run automatically when system is idle")
        print("   - Check logs/background_training.log for details")
        print("   - Press Ctrl+C to stop")
        
        # Keep main thread alive
        while service.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping service...")
    finally:
        service.stop()
        print("Service stopped.")

if __name__ == "__main__":
    main()
