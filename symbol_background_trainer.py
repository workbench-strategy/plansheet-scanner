#!/usr/bin/env python3
"""
Symbol Background Training Service
Runs symbol recognition training in the background without interrupting user work.
This is the foundational training step before moving to standards and requirements.
"""

import os
import sys
import json
import time
import signal
import logging
import threading
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import queue
import subprocess
import psutil
import schedule

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the symbol recognition trainer
from symbol_recognition_trainer import SymbolRecognitionTrainer

class SymbolBackgroundTrainingService:
    """Background service for continuous symbol recognition training."""
    
    def __init__(self, config_file: str = "config/symbol_training_config.json"):
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
        self.trainer = SymbolRecognitionTrainer(
            data_dir=self.config.get("data_dir", "symbol_training_data"),
            model_dir=self.config.get("model_dir", "symbol_models")
        )
        
        # Performance monitoring
        self.performance_stats = {
            "cpu_usage": [],
            "memory_usage": [],
            "training_times": [],
            "symbol_accuracy": []
        }
        
        self.logger.info("Symbol Background Training Service initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load symbol training configuration."""
        default_config = {
            "training_schedule": {
                "enabled": True,
                "interval_hours": 4,
                "max_training_time_minutes": 60,
                "idle_threshold_percent": 25
            },
            "data_management": {
                "data_dir": "symbol_training_data",
                "model_dir": "symbol_models",
                "backup_interval_hours": 12,
                "max_data_files": 500,
                "auto_cleanup": True
            },
            "model_config": {
                "min_examples": 30,
                "max_examples": 500,
                "validation_split": 0.2,
                "batch_size": 16,
                "learning_rate": 0.001,
                "epochs_per_session": 5
            },
            "monitoring": {
                "log_level": "INFO",
                "log_file": "logs/symbol_training.log",
                "status_file": "logs/symbol_training_status.json",
                "performance_log": "logs/symbol_performance.log",
                "save_models": True,
                "save_checkpoints": True
            },
            "symbol_specific": {
                "disciplines": ["traffic", "electrical", "structural", "drainage", "mechanical"],
                "focus_areas": ["visual_recognition", "legend_parsing", "notes_extraction"],
                "confidence_threshold": 0.8,
                "min_symbols_per_discipline": 10
            },
            "notifications": {
                "enabled": False,
                "email": "",
                "webhook_url": "",
                "training_complete": True,
                "training_failed": True,
                "system_issues": True
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
        """Setup comprehensive logging system with rotation."""
        log_config = self.config["monitoring"]
        log_file = Path(log_config["log_file"])
        log_file.parent.mkdir(exist_ok=True)
        
        # Configure logging with rotation
        from logging.handlers import RotatingFileHandler
        
        # Main log handler with rotation (max 5MB, keep 3 backup files)
        main_handler = RotatingFileHandler(
            log_file, 
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        main_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config["log_level"]),
            handlers=[main_handler, console_handler]
        )
        
        self.logger = logging.getLogger("SymbolTraining")
        
        # Create performance logger with rotation
        perf_log_file = Path(log_config["performance_log"])
        perf_log_file.parent.mkdir(exist_ok=True)
        
        self.perf_logger = logging.getLogger("SymbolPerformance")
        perf_handler = RotatingFileHandler(
            perf_log_file,
            maxBytes=1*1024*1024,  # 1MB
            backupCount=2
        )
        perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
    
    def start(self):
        """Start the background symbol training service."""
        if self.is_running:
            self.logger.warning("Service is already running")
            return
        
        self.logger.info("Starting symbol background training service...")
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
        
        self.logger.info("Symbol background training service started successfully")
        self._save_status()
    
    def stop(self):
        """Stop the background symbol training service."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping symbol background training service...")
        self.is_running = False
        self.stop_event.set()
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=30)
        
        self.logger.info("Symbol background training service stopped")
        self._save_status()
    
    def _training_loop(self):
        """Main training loop that runs in background."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Check if system is idle enough for training
                if self._should_train():
                    self.logger.info("Starting symbol training session...")
                    self._run_symbol_training_session()
                else:
                    self.logger.debug("System busy, skipping symbol training session")
                
                # Wait before next check
                time.sleep(self.config["training_schedule"]["interval_hours"] * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in symbol training loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _should_train(self) -> bool:
        """Check if system is idle enough for symbol training."""
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
                memory_percent < 75 and  # Lower threshold for symbol training
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
    
    def _run_symbol_training_session(self):
        """Run a single symbol training session."""
        start_time = time.time()
        max_time = self.config["training_schedule"]["max_training_time_minutes"] * 60
        
        try:
            # Process as-built drawings first
            self._process_as_built_drawings()
            
            # Generate symbol training data if needed
            current_count = len(self.trainer.engineering_symbols)
            target_count = self.config["model_config"]["max_examples"]
            
            if current_count < target_count:
                self.logger.info(f"Generating additional symbol training data ({current_count} -> {target_count})")
                additional_examples = min(target_count - current_count, 50)  # Generate in smaller batches
                self.trainer.generate_symbol_training_data(num_examples=additional_examples)
            
            # Train symbol models
            self.logger.info("Training symbol recognition models...")
            self.trainer.train_symbol_models(min_examples=self.config["model_config"]["min_examples"])
            
            # Record training statistics
            training_time = time.time() - start_time
            self.training_stats = self.trainer.get_training_statistics()
            self.training_stats["last_training_time"] = training_time
            self.training_stats["timestamp"] = datetime.now().isoformat()
            
            self.last_training_time = datetime.now()
            
            # Log performance metrics
            self._log_performance_metrics(training_time)
            
            self.logger.info(f"Symbol training session completed in {training_time:.2f} seconds")
            self._save_status()
            
        except Exception as e:
            self.logger.error(f"Error during symbol training session: {e}")
            training_time = time.time() - start_time
            self.logger.error(f"Symbol training session failed after {training_time:.2f} seconds")
            # Still save status even if training failed
            self._save_status()
    
    def _process_as_built_drawings(self):
        """Process as-built drawings to extract symbols."""
        try:
            as_built_config = self.config["data_management"].get("as_built_processing", {})
            if not as_built_config.get("enabled", False):
                return
            
            input_dir = Path(as_built_config["input_dir"])
            if not input_dir.exists():
                self.logger.warning(f"As-built input directory not found: {input_dir}")
                return
            
            # Find unprocessed PDF files
            pdf_files = list(input_dir.glob("*.pdf"))
            if not pdf_files:
                self.logger.info("No new as-built PDF files to process")
                return
            
            max_files = as_built_config.get("max_files_per_session", 5)
            files_to_process = pdf_files[:max_files]
            
            self.logger.info(f"Processing {len(files_to_process)} as-built drawings...")
            
            for pdf_file in files_to_process:
                try:
                    self.logger.info(f"Processing as-built: {pdf_file.name}")
                    
                    # Extract symbols from PDF
                    symbols = self._extract_symbols_from_pdf(pdf_file)
                    
                    if symbols:
                        # Add to training data
                        self.trainer.add_symbols_from_as_built(symbols, pdf_file.name)
                        self.logger.info(f"Extracted {len(symbols)} symbols from {pdf_file.name}")
                        
                        # Move to processed directory
                        processed_dir = Path(as_built_config["processed_dir"])
                        processed_dir.mkdir(exist_ok=True)
                        shutil.move(str(pdf_file), str(processed_dir / pdf_file.name))
                    else:
                        # Move to failed directory
                        failed_dir = Path(as_built_config["failed_dir"])
                        failed_dir.mkdir(exist_ok=True)
                        shutil.move(str(pdf_file), str(failed_dir / pdf_file.name))
                        self.logger.warning(f"No symbols found in {pdf_file.name}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {pdf_file.name}: {e}")
                    # Move to failed directory
                    failed_dir = Path(as_built_config["failed_dir"])
                    failed_dir.mkdir(exist_ok=True)
                    shutil.move(str(pdf_file), str(failed_dir / pdf_file.name))
            
            self.logger.info(f"As-built processing completed. Processed: {len(files_to_process)} files")
            
            # If no symbols were found, generate some synthetic ones based on the file names
            if len(self.trainer.engineering_symbols) < 10:
                self.logger.info("No symbols found in as-built files, generating synthetic symbols based on file content...")
                self._generate_synthetic_symbols_from_filenames(files_to_process)
            
        except Exception as e:
            self.logger.error(f"Error in as-built processing: {e}")
    
    def _generate_synthetic_symbols_from_filenames(self, files_processed: List[Path]):
        """Generate synthetic symbols based on file names when no real symbols are found."""
        for pdf_file in files_processed:
            filename = pdf_file.name.lower()
            
            # Determine discipline and symbols based on filename
            symbols_to_add = []
            
            if any(word in filename for word in ["signal", "traffic", "its"]):
                symbols_to_add.extend([
                    {"symbol_name": "traffic_signal", "symbol_type": "traffic"},
                    {"symbol_name": "detector_loop", "symbol_type": "traffic"},
                    {"symbol_name": "sign", "symbol_type": "traffic"}
                ])
            
            if any(word in filename for word in ["electrical", "power", "light"]):
                symbols_to_add.extend([
                    {"symbol_name": "junction_box", "symbol_type": "electrical"},
                    {"symbol_name": "conduit", "symbol_type": "electrical"},
                    {"symbol_name": "light", "symbol_type": "electrical"},
                    {"symbol_name": "panel", "symbol_type": "electrical"}
                ])
            
            if any(word in filename for word in ["structural", "beam", "foundation"]):
                symbols_to_add.extend([
                    {"symbol_name": "beam", "symbol_type": "structural"},
                    {"symbol_name": "column", "symbol_type": "structural"},
                    {"symbol_name": "foundation", "symbol_type": "structural"}
                ])
            
            if any(word in filename for word in ["drainage", "storm", "pipe"]):
                symbols_to_add.extend([
                    {"symbol_name": "catch_basin", "symbol_type": "drainage"},
                    {"symbol_name": "manhole", "symbol_type": "drainage"},
                    {"symbol_name": "pipe", "symbol_type": "drainage"}
                ])
            
            # Add synthetic symbols
            for symbol_info in symbols_to_add:
                symbol = {
                    "symbol_id": f"synthetic_{symbol_info['symbol_name']}_{len(self.trainer.engineering_symbols)}",
                    "symbol_name": symbol_info["symbol_name"],
                    "symbol_type": symbol_info["symbol_type"],
                    "visual_description": f"synthetic {symbol_info['symbol_name']} from {pdf_file.name}",
                    "legend_reference": symbol_info["symbol_name"].upper(),
                    "notes_description": f"Generated from filename analysis of {pdf_file.name}",
                    "common_variations": [symbol_info["symbol_name"].upper()],
                    "context_clues": ["synthetic", "filename_analysis"],
                    "file_path": pdf_file.name,
                    "confidence": 0.6,  # Lower confidence for synthetic symbols
                    "usage_frequency": 1,
                    "source": "synthetic_from_filename",
                    "page": 1
                }
                self.trainer.add_symbols_from_as_built([symbol], pdf_file.name)
    
    def _extract_symbols_from_pdf(self, pdf_file: Path) -> List[Dict[str, Any]]:
        """Extract symbols from a PDF file."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(str(pdf_file))
            symbols = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                
                # Look for symbol patterns in text
                page_symbols = self._find_symbols_in_text(text, page_num + 1, pdf_file.name)
                symbols.extend(page_symbols)
                
                # Extract images and look for visual symbols
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    img_symbols = self._find_symbols_in_image(img, page_num + 1, pdf_file.name)
                    symbols.extend(img_symbols)
            
            doc.close()
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error extracting symbols from {pdf_file.name}: {e}")
            return []
    
    def _find_symbols_in_text(self, text: str, page_num: int, filename: str) -> List[Dict[str, Any]]:
        """Find symbols in text content."""
        symbols = []
        
        # Define symbol patterns to look for - expanded for better coverage
        symbol_patterns = {
            "traffic_signal": [r"TS", r"traffic signal", r"signal", r"traffic light", r"stop light", r"traffic control"],
            "junction_box": [r"JB", r"junction box", r"J-box", r"electrical box", r"junction"],
            "catch_basin": [r"CB", r"catch basin", r"inlet", r"storm drain", r"drainage"],
            "manhole": [r"MH", r"manhole", r"access", r"sewer", r"utility access"],
            "detector_loop": [r"DL", r"detector", r"loop", r"vehicle detector", r"traffic detector"],
            "conduit": [r"conduit", r"pipe", r"cable", r"electrical conduit", r"wiring"],
            "light": [r"light", r"lamp", r"fixture", r"street light", r"lighting"],
            "panel": [r"panel", r"electrical panel", r"breaker", r"electrical", r"power panel"],
            "sign": [r"sign", r"stop sign", r"yield sign", r"speed limit", r"traffic sign"],
            "pole": [r"pole", r"utility pole", r"light pole", r"signal pole"],
            "cable": [r"cable", r"fiber", r"communication", r"data cable"],
            "pipe": [r"pipe", r"storm pipe", r"water pipe", r"utility pipe"],
            "foundation": [r"foundation", r"footing", r"base", r"structural"],
            "beam": [r"beam", r"girder", r"structural beam", r"support"],
            "column": [r"column", r"post", r"structural column", r"support"]
        }
        
        import re
        
        for symbol_type, patterns in symbol_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    symbol = {
                        "symbol_id": f"{symbol_type}_{len(symbols)}",
                        "symbol_name": symbol_type,
                        "symbol_type": self._get_discipline_for_symbol(symbol_type),
                        "visual_description": f"text reference: {match.group()}",
                        "legend_reference": match.group(),
                        "notes_description": f"Found in text on page {page_num}",
                        "common_variations": [match.group()],
                        "context_clues": ["text", "drawing", "plan"],
                        "file_path": filename,
                        "confidence": 0.8,
                        "usage_frequency": 1,
                        "source": "as_built_drawing",
                        "page": page_num
                    }
                    symbols.append(symbol)
        
        return symbols
    
    def _find_symbols_in_image(self, img, page_num: int, filename: str) -> List[Dict[str, Any]]:
        """Find symbols in image content."""
        # This is a placeholder - would need more sophisticated image analysis
        # For now, return empty list
        return []
    
    def _get_discipline_for_symbol(self, symbol_type: str) -> str:
        """Get discipline for a symbol type."""
        discipline_map = {
            "traffic_signal": "traffic",
            "detector_loop": "traffic",
            "junction_box": "electrical",
            "conduit": "electrical",
            "light": "electrical",
            "panel": "electrical",
            "catch_basin": "drainage",
            "manhole": "drainage"
        }
        return discipline_map.get(symbol_type, "general")
    
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
                f"Symbol training completed - Time: {training_time:.2f}s, "
                f"CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, "
                f"Symbols: {self.training_stats.get('total_symbols', 0)}"
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
        """Clean up old symbol training data files."""
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
                        self.logger.info(f"Removed old symbol data file: {file.name}")
        
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {e}")
    
    def _setup_schedule(self):
        """Setup scheduled symbol training tasks."""
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
            self.logger.info("Scheduled symbol training triggered")
            # The training loop will handle the actual training
    
    def _save_status(self):
        """Save current symbol training status to file."""
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
    
    def stop(self):
        """Stop the background symbol training service."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping symbol background training service...")
        self.is_running = False
        self.stop_event.set()
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=30)
        
        self.logger.info("Symbol background training service stopped")
        # Force update status to show stopped
        self._save_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current symbol training status."""
        return {
            "is_running": self.is_running,
            "last_training_time": self.last_training_time,
            "training_stats": self.training_stats,
            "performance_stats": self.performance_stats
        }
    
    def force_training(self):
        """Force an immediate symbol training session."""
        if self.is_running:
            self.logger.info("Forcing immediate symbol training session...")
            self._run_symbol_training_session()
        else:
            self.logger.warning("Service is not running")
    
    def test_symbol_identification(self, symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test symbol identification with current models."""
        if not self.is_running:
            return {"status": "service_not_running"}
        
        try:
            result = self.trainer.identify_symbol(symbol_info)
            self.logger.info(f"Symbol identification test completed: {result.get('predicted_symbol', 'unknown')}")
            return result
        except Exception as e:
            self.logger.error(f"Error in symbol identification test: {e}")
            return {"status": "error", "message": str(e)}

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\nReceived shutdown signal. Stopping symbol background training service...")
    if hasattr(signal_handler, 'service'):
        signal_handler.service.stop()
    sys.exit(0)

def main():
    """Main function to run the symbol background training service."""
    print("🤖 Symbol Background Training Service - Foundation Training")
    print("=" * 70)
    
    # Create service
    service = SymbolBackgroundTrainingService()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal_handler.service = service
    
    try:
        # Start service
        service.start()
        
        print("✅ Symbol background training service is running...")
        print("   - Symbol training will run automatically when system is idle")
        print("   - Focus: Learning symbols from plans, legends, and notes")
        print("   - Check logs/symbol_training.log for details")
        print("   - Press Ctrl+C to stop")
        
        # Keep main thread alive
        while service.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping symbol training service...")
    finally:
        service.stop()
        print("Symbol training service stopped.")

if __name__ == "__main__":
    main()
