#!/usr/bin/env python3
"""
Start As-Built Training with Real Data - Fixed Version
Comprehensive script to begin training with real as-built drawings and data.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging without Unicode characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('as_built_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for as-built training."""
    directories = [
        "real_as_built_data",
        "real_as_built_data/images",
        "real_as_built_data/images/traffic_signals",
        "real_as_built_data/images/its",
        "real_as_built_data/images/electrical",
        "real_as_built_data/images/structural",
        "real_as_built_data/metadata",
        "real_as_built_data/reviews",
        "training_data",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"[OK] Created directory: {directory}")

def load_existing_real_world_data() -> List[Dict[str, Any]]:
    """Load existing real-world data from JSON files."""
    real_world_dir = Path("real_world_data")
    data = []
    
    if not real_world_dir.exists():
        logger.warning("real_world_data directory not found")
        return data
    
    for json_file in real_world_dir.glob("drawing_*.json"):
        try:
            with open(json_file, 'r') as f:
                drawing_data = json.load(f)
                data.append(drawing_data)
                logger.info(f"[LOADED] {drawing_data['sheet_number']} - {drawing_data['discipline']}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"[DATA] Loaded {len(data)} existing drawing records")
    return data

def create_metadata_from_real_data(drawing_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive metadata from real drawing data."""
    
    # Map discipline to plan type
    discipline_mapping = {
        "traffic": "traffic_signal",
        "electrical": "electrical",
        "structural": "structural",
        "its": "its"
    }
    
    plan_type = discipline_mapping.get(drawing_data.get("discipline", "").lower(), "unknown")
    
    # Create final elements based on discipline
    final_elements = []
    if plan_type == "traffic_signal":
        final_elements = [
            {
                "type": "signal_head_red",
                "location": [100, 200],
                "approved": drawing_data.get("approval_status") == "approved",
                "specifications": "ITE Type 3, 12-inch"
            },
            {
                "type": "signal_head_green",
                "location": [100, 220],
                "approved": drawing_data.get("approval_status") == "approved",
                "specifications": "ITE Type 3, 12-inch"
            },
            {
                "type": "detector_loop",
                "location": [80, 180],
                "approved": drawing_data.get("approval_status") == "approved",
                "specifications": "6x6 foot inductive loop"
            }
        ]
    elif plan_type == "electrical":
        final_elements = [
            {
                "type": "conduit",
                "location": [150, 250],
                "approved": drawing_data.get("approval_status") == "approved",
                "specifications": "2-inch PVC conduit"
            },
            {
                "type": "junction_box",
                "location": [200, 300],
                "approved": drawing_data.get("approval_status") == "approved",
                "specifications": "12x12x6 inch junction box"
            }
        ]
    elif plan_type == "its":
        final_elements = [
            {
                "type": "camera",
                "location": [120, 180],
                "approved": drawing_data.get("approval_status") == "approved",
                "specifications": "PTZ traffic camera"
            },
            {
                "type": "detector_station",
                "location": [90, 160],
                "approved": drawing_data.get("approval_status") == "approved",
                "specifications": "Microwave vehicle detector"
            }
        ]
    
    # Create project info
    project_info = {
        "budget": 500000,  # Default values - should be updated with real data
        "duration_months": 6,
        "complexity_score": 0.7,
        "project_type": "intersection",
        "contractor": "ABC Construction",
        "engineer": "XYZ Engineering",
        "code_references": drawing_data.get("code_references", []),
        "approval_status": drawing_data.get("approval_status", "pending")
    }
    
    metadata = {
        "plan_id": drawing_data.get("drawing_id", "unknown"),
        "plan_type": plan_type,
        "project_name": f"{drawing_data.get('sheet_title', 'Unknown Project')}",
        "sheet_number": drawing_data.get("sheet_number", "Unknown"),
        "sheet_title": drawing_data.get("sheet_title", "Unknown"),
        "discipline": drawing_data.get("discipline", "unknown"),
        "final_elements": final_elements,
        "construction_notes": drawing_data.get("notes", "No construction notes available"),
        "project_info": project_info,
        "approval_date": drawing_data.get("timestamp", datetime.now().isoformat()),
        "reviewer": "System Generated",
        "compliance_score": 0.95 if drawing_data.get("approval_status") == "approved" else 0.7,
        "review_comments": drawing_data.get("review_comments", [])
    }
    
    return metadata

def create_sample_image_for_discipline(discipline: str) -> np.ndarray:
    """Create a sample image for the given discipline."""
    # Create a 1000x1000 image with different colors based on discipline
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    if discipline.lower() == "traffic":
        # Traffic signal colors (red, yellow, green)
        image[200:300, 100:200] = [0, 0, 255]  # Red
        image[300:400, 100:200] = [0, 255, 255]  # Yellow
        image[400:500, 100:200] = [0, 255, 0]  # Green
        # Add detector loop
        cv2.rectangle(image, (80, 180), (120, 220), (255, 255, 255), 2)
        
    elif discipline.lower() == "electrical":
        # Electrical colors (blue for conduit, gray for boxes)
        cv2.rectangle(image, (150, 250), (250, 350), (255, 0, 0), 3)  # Blue conduit
        cv2.rectangle(image, (200, 300), (300, 400), (128, 128, 128), 2)  # Gray junction box
        
    elif discipline.lower() == "its":
        # ITS colors (purple for cameras, orange for detectors)
        cv2.circle(image, (120, 180), 30, (255, 0, 255), -1)  # Purple camera
        cv2.rectangle(image, (90, 160), (150, 200), (0, 165, 255), 2)  # Orange detector
        
    else:
        # Default gray
        image[:] = [128, 128, 128]
    
    # Add text label
    cv2.putText(image, discipline.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    return image

def process_real_world_data():
    """Process existing real-world data and create training-ready files."""
    logger.info("[PROCESSING] Processing real-world data...")
    
    # Load existing data
    drawing_data = load_existing_real_world_data()
    
    if not drawing_data:
        logger.warning("No real-world data found. Creating sample data...")
        # Create sample data if none exists
        drawing_data = [
            {
                "drawing_id": "sample_001",
                "sheet_number": "T-001",
                "sheet_title": "Traffic Signal Plan",
                "discipline": "traffic",
                "notes": "Signal heads and detector loops shown",
                "code_references": ["MUTCD", "WSDOT Standards"],
                "review_comments": ["Design meets requirements", "No violations found"],
                "approval_status": "approved",
                "file_path": "traffic_001.pdf"
            },
            {
                "drawing_id": "sample_002",
                "sheet_number": "E-001",
                "sheet_title": "Electrical Plan",
                "discipline": "electrical",
                "notes": "Power distribution and conduit routing",
                "code_references": ["NEC", "WSDOT Electrical"],
                "review_comments": ["Conduit sizing adequate", "Grounding system proper"],
                "approval_status": "approved",
                "file_path": "electrical_001.pdf"
            },
            {
                "drawing_id": "sample_003",
                "sheet_number": "ITS-001",
                "sheet_title": "ITS Plan",
                "discipline": "its",
                "notes": "Traffic monitoring and detection systems",
                "code_references": ["WSDOT ITS Standards"],
                "review_comments": ["Camera placement optimal", "Detection coverage adequate"],
                "approval_status": "approved",
                "file_path": "its_001.pdf"
            }
        ]
    
    # Process each drawing
    for drawing in drawing_data:
        try:
            # Create metadata
            metadata = create_metadata_from_real_data(drawing)
            
            # Create sample image
            discipline = drawing.get("discipline", "unknown")
            image = create_sample_image_for_discipline(discipline)
            
            # Save image
            image_filename = f"{drawing['drawing_id']}.png"
            image_path = f"real_as_built_data/images/{discipline.lower()}/{image_filename}"
            cv2.imwrite(image_path, image)
            
            # Save metadata
            metadata_filename = f"{drawing['drawing_id']}.json"
            metadata_path = f"real_as_built_data/metadata/{metadata_filename}"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create review data
            review_data = {
                "plan_id": drawing['drawing_id'],
                "milestone": "final",
                "compliance_score": metadata['compliance_score'],
                "approved_elements": metadata['final_elements'],
                "rejected_elements": [],
                "comments": drawing.get("review_comments", []),
                "review_date": drawing.get("timestamp", datetime.now().isoformat())
            }
            
            review_filename = f"review_{drawing['drawing_id']}.json"
            review_path = f"real_as_built_data/reviews/{review_filename}"
            with open(review_path, 'w') as f:
                json.dump(review_data, f, indent=2)
            
            logger.info(f"[OK] Processed: {drawing['sheet_number']} - {discipline}")
            
        except Exception as e:
            logger.error(f"Error processing {drawing.get('drawing_id', 'unknown')}: {e}")
    
    logger.info(f"[COMPLETE] Processed {len(drawing_data)} drawings")

def start_foundation_training():
    """Start the foundation training process."""
    logger.info("[TRAINING] Starting Foundation Training...")
    
    try:
        from src.core.foundation_trainer import FoundationTrainer
        
        # Initialize trainer
        trainer = FoundationTrainer("training_data", "models")
        
        # Import batch data
        import_as_built_batch("real_as_built_data", trainer)
        
        # Extract features and train models
        logger.info("[FEATURES] Extracting training features...")
        trainer.extract_training_features()
        
        logger.info("[MODELS] Training foundation models...")
        trainer.train_foundation_models(min_examples=3)  # Lower threshold for initial training
        
        # Show statistics
        stats = trainer.get_training_statistics()
        logger.info(f"[STATS] Training Statistics:")
        logger.info(f"   Total as-built records: {stats['total_as_built']}")
        logger.info(f"   Total training examples: {stats['total_training_examples']}")
        logger.info(f"   Models trained: {stats['models_trained']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during foundation training: {e}")
        return False

def import_as_built_batch(data_dir: str, trainer):
    """Import all as-built data from a directory."""
    logger.info(f"[IMPORT] Importing as-built data from {data_dir}...")
    
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    metadata_dir = data_path / "metadata"
    
    if not images_dir.exists() or not metadata_dir.exists():
        logger.error("Required directories not found")
        return
    
    # Process each plan type
    for plan_type_dir in images_dir.iterdir():
        if not plan_type_dir.is_dir():
            continue
            
        plan_type = plan_type_dir.name
        logger.info(f"Processing {plan_type} as-builts...")
        
        # Process each image in the plan type directory
        for image_file in plan_type_dir.glob("*.png"):
            plan_id = image_file.stem
            
            # Load image
            as_built_image = cv2.imread(str(image_file))
            if as_built_image is None:
                logger.warning(f"Could not load {image_file}")
                continue
            
            # Load metadata
            metadata_file = metadata_dir / f"{plan_id}.json"
            if not metadata_file.exists():
                logger.warning(f"No metadata for {plan_id}")
                continue
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add to trainer
            trainer.add_as_built_data(
                plan_id=plan_id,
                plan_type=plan_type,
                as_built_image=as_built_image,
                final_elements=metadata.get('final_elements', []),
                construction_notes=metadata.get('construction_notes', ''),
                approval_date=datetime.fromisoformat(metadata.get('approval_date', datetime.now().isoformat())),
                project_info=metadata.get('project_info', {})
            )
            
            logger.info(f"[OK] Added {plan_id}")

def main():
    """Main function to start as-built training."""
    print("Starting As-Built Training with Real Data")
    print("=" * 60)
    
    try:
        # Step 1: Setup directories
        logger.info("[SETUP] Setting up directories...")
        setup_directories()
        
        # Step 2: Process real-world data
        logger.info("[DATA] Processing real-world data...")
        process_real_world_data()
        
        # Step 3: Start foundation training
        logger.info("[TRAIN] Starting foundation training...")
        success = start_foundation_training()
        
        if success:
            print("\n[SUCCESS] As-built training completed successfully!")
            print("[INFO] Check the training_data/ and models/ directories for results")
            print("[INFO] Review as_built_training.log for detailed information")
        else:
            print("\n[ERROR] Training encountered errors. Check the log file for details.")
            
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"\n[ERROR] Critical error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
