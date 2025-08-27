#!/usr/bin/env python3
"""
As-Built Drawing Processor
Shows where as-built drawings go during and after processing.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

class AsBuiltProcessor:
    """Process as-built drawings and organize the data flow."""
    
    def __init__(self):
        # Directory structure for as-built processing
        self.directories = {
            "input": "as_built_drawings",           # Where you put original as-builts
            "processing": "processing_queue",       # Temporary processing area
            "processed": "processed_drawings",      # Successfully processed drawings
            "failed": "failed_drawings",           # Failed processing attempts
            "extracted_data": "extracted_symbols",  # Extracted symbol data
            "training_data": "symbol_training_data", # Training data for AI
            "models": "symbol_models",             # Trained models
            "logs": "logs"                         # Processing logs
        }
        
        # Create all directories
        self._create_directories()
    
    def _create_directories(self):
        """Create the complete directory structure."""
        for dir_name, dir_path in self.directories.items():
            Path(dir_path).mkdir(exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
    
    def show_data_flow(self):
        """Show the complete data flow for as-built processing."""
        print("🔄 As-Built Drawing Data Flow")
        print("=" * 50)
        
        print("\n📁 Directory Structure:")
        for dir_name, dir_path in self.directories.items():
            print(f"   {dir_name:15} → {dir_path}/")
        
        print("\n🔄 Processing Flow:")
        print("   1. Input:        as_built_drawings/     ← You put original PDFs here")
        print("   2. Processing:   processing_queue/      ← Temporary processing area")
        print("   3. Success:      processed_drawings/    ← Successfully processed")
        print("   4. Failed:       failed_drawings/       ← Failed processing")
        print("   5. Extracted:    extracted_symbols/     ← Symbol data extracted")
        print("   6. Training:     symbol_training_data/  ← AI training data")
        print("   7. Models:       symbol_models/         ← Trained AI models")
        print("   8. Logs:         logs/                  ← Processing logs")
    
    def process_as_built(self, drawing_path: str) -> Dict[str, Any]:
        """Process a single as-built drawing and show where it goes."""
        drawing_name = Path(drawing_path).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n🔄 Processing: {drawing_name}")
        print("-" * 40)
        
        # Step 1: Copy to processing queue
        processing_path = f"{self.directories['processing']}/{timestamp}_{drawing_name}"
        print(f"   1. → processing_queue/{timestamp}_{drawing_name}")
        
        # Step 2: Simulate processing (extract symbols)
        extracted_symbols = self._extract_symbols_from_drawing(drawing_path)
        
        # Step 3: Save extracted data
        extracted_file = f"{self.directories['extracted_data']}/{timestamp}_{drawing_name.replace('.pdf', '_symbols.json')}"
        with open(extracted_file, 'w') as f:
            json.dump(extracted_symbols, f, indent=2)
        print(f"   2. → extracted_symbols/{timestamp}_{drawing_name.replace('.pdf', '_symbols.json')}")
        
        # Step 4: Move to processed directory
        processed_path = f"{self.directories['processed']}/{drawing_name}"
        print(f"   3. → processed_drawings/{drawing_name}")
        
        # Step 5: Add to training data
        training_file = f"{self.directories['training_data']}/as_built_{timestamp}_{drawing_name.replace('.pdf', '.json')}"
        training_data = self._create_training_data(extracted_symbols, drawing_name)
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"   4. → symbol_training_data/as_built_{timestamp}_{drawing_name.replace('.pdf', '.json')}")
        
        # Step 6: Log processing
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "drawing": drawing_name,
            "status": "processed",
            "symbols_found": len(extracted_symbols.get("symbols", [])),
            "processing_time": "2.3s",
            "files_created": [
                extracted_file,
                training_file
            ]
        }
        
        log_file = f"{self.directories['logs']}/as_built_processing.log"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        
        print(f"   5. → logs/as_built_processing.log (logged)")
        
        return {
            "status": "success",
            "drawing": drawing_name,
            "symbols_extracted": len(extracted_symbols.get("symbols", [])),
            "files_created": [extracted_file, training_file]
        }
    
    def _extract_symbols_from_drawing(self, drawing_path: str) -> Dict[str, Any]:
        """Simulate extracting symbols from a drawing."""
        # This would be replaced with actual symbol extraction logic
        return {
            "drawing_file": drawing_path,
            "extraction_timestamp": datetime.now().isoformat(),
            "symbols": [
                {
                    "symbol_id": "TS_001",
                    "symbol_type": "traffic_signal",
                    "discipline": "traffic",
                    "location": {"x": 150, "y": 200},
                    "confidence": 0.95,
                    "visual_description": "circle with traffic light",
                    "legend_reference": "TS",
                    "notes": "traffic signal installed per plan"
                },
                {
                    "symbol_id": "JB_001",
                    "symbol_type": "junction_box",
                    "discipline": "electrical",
                    "location": {"x": 300, "y": 150},
                    "confidence": 0.88,
                    "visual_description": "square with JB",
                    "legend_reference": "JB",
                    "notes": "junction box to be provided"
                }
            ],
            "metadata": {
                "total_symbols": 2,
                "disciplines_found": ["traffic", "electrical"],
                "processing_notes": "Successfully extracted symbols from drawing"
            }
        }
    
    def _create_training_data(self, extracted_symbols: Dict[str, Any], drawing_name: str) -> Dict[str, Any]:
        """Create training data from extracted symbols."""
        training_examples = []
        
        for symbol in extracted_symbols.get("symbols", []):
            training_example = {
                "symbol_id": symbol["symbol_id"],
                "symbol_name": symbol["symbol_type"],
                "symbol_type": symbol["discipline"],
                "visual_description": symbol["visual_description"],
                "legend_reference": symbol["legend_reference"],
                "notes_description": symbol["notes"],
                "common_variations": [symbol["legend_reference"], symbol["symbol_type"].upper()],
                "context_clues": ["intersection", "roadway", "electrical"],
                "file_path": drawing_name,
                "confidence": symbol["confidence"],
                "usage_frequency": 1,
                "source": "as_built_drawing",
                "extraction_timestamp": datetime.now().isoformat()
            }
            training_examples.append(training_example)
        
        return {
            "drawing_source": drawing_name,
            "training_examples": training_examples,
            "total_examples": len(training_examples),
            "created_timestamp": datetime.now().isoformat()
        }
    
    def show_file_locations(self):
        """Show where files are located after processing."""
        print("\n📂 File Locations After Processing:")
        print("=" * 50)
        
        print("\n🎯 Original As-Builts:")
        print(f"   → {self.directories['input']}/")
        print("   (Your original PDF files stay here)")
        
        print("\n✅ Successfully Processed:")
        print(f"   → {self.directories['processed']}/")
        print("   (Copies of successfully processed drawings)")
        
        print("\n❌ Failed Processing:")
        print(f"   → {self.directories['failed']}/")
        print("   (Drawings that couldn't be processed)")
        
        print("\n🔍 Extracted Symbol Data:")
        print(f"   → {self.directories['extracted_data']}/")
        print("   (JSON files with extracted symbols from each drawing)")
        
        print("\n🤖 AI Training Data:")
        print(f"   → {self.directories['training_data']}/")
        print("   (Formatted data ready for AI training)")
        
        print("\n🧠 Trained Models:")
        print(f"   → {self.directories['models']}/")
        print("   (AI models trained on your as-built data)")
        
        print("\n📋 Processing Logs:")
        print(f"   → {self.directories['logs']}/")
        print("   (Detailed logs of all processing activities)")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed drawings."""
        stats = {}
        
        for dir_name, dir_path in self.directories.items():
            if Path(dir_path).exists():
                files = list(Path(dir_path).glob("*"))
                stats[dir_name] = {
                    "directory": dir_path,
                    "file_count": len(files),
                    "file_types": list(set(f.suffix for f in files if f.suffix))
                }
        
        return stats
    
    def show_processing_stats(self):
        """Show current processing statistics."""
        stats = self.get_processing_stats()
        
        print("\n📊 Current Processing Statistics:")
        print("=" * 40)
        
        for dir_name, dir_stats in stats.items():
            print(f"\n{dir_name.upper()}:")
            print(f"   Directory: {dir_stats['directory']}")
            print(f"   Files: {dir_stats['file_count']}")
            if dir_stats['file_types']:
                print(f"   Types: {', '.join(dir_stats['file_types'])}")

def main():
    """Main function to demonstrate as-built processing flow."""
    print("🏗️  As-Built Drawing Processor")
    print("=" * 50)
    
    processor = AsBuiltProcessor()
    
    # Show the complete data flow
    processor.show_data_flow()
    
    # Show file locations
    processor.show_file_locations()
    
    # Show current stats
    processor.show_processing_stats()
    
    # Example processing
    print("\n🔄 Example Processing:")
    print("-" * 30)
    
    # Simulate processing an as-built drawing
    example_drawing = "traffic_plan_001.pdf"
    result = processor.process_as_built(example_drawing)
    
    print(f"\n✅ Processing Complete!")
    print(f"   Drawing: {result['drawing']}")
    print(f"   Symbols Extracted: {result['symbols_extracted']}")
    print(f"   Files Created: {len(result['files_created'])}")
    
    print("\n🎯 Next Steps:")
    print("   1. Put your as-built PDFs in: as_built_drawings/")
    print("   2. Run the symbol training service")
    print("   3. Check logs/ for processing results")
    print("   4. Monitor symbol_training_data/ for AI training data")

if __name__ == "__main__":
    main()
