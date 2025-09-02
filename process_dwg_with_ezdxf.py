#!/usr/bin/env python3
"""
Process .dwg Files with ezdxf
This script uses ezdxf to extract symbols from your AutoCAD files.
"""

import os
import sys
from pathlib import Path
import ezdxf
import numpy as np
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from autocad_dwg_symbol_trainer import AutoCADSymbol, AutoCADSymbolTrainer, MiniModelConfig

def extract_symbols_with_ezdxf(dwg_file_path):
    """Extract symbols from a .dwg file using ezdxf."""
    symbols = []
    
    try:
        # Read the DWG file
        doc = ezdxf.readfile(dwg_file_path)
        msp = doc.modelspace()
        
        print(f"   Processing: {Path(dwg_file_path).name}")
        print(f"   Entities found: {len(msp)}")
        
        entity_count = 0
        for entity in msp:
            entity_count += 1
            
            try:
                symbol = process_ezdxf_entity(entity, dwg_file_path)
                if symbol:
                    symbols.append(symbol)
            except Exception as e:
                # Skip problematic entities
                continue
        
        print(f"   ✅ Extracted {len(symbols)} symbols from {entity_count} entities")
        
    except Exception as e:
        print(f"   ❌ Error processing {Path(dwg_file_path).name}: {e}")
    
    return symbols

def process_ezdxf_entity(entity, dwg_file_path):
    """Process an ezdxf entity and convert to symbol."""
    entity_type = entity.dxftype()
    
    try:
        if entity_type == "CIRCLE":
            return process_ezdxf_circle(entity, dwg_file_path)
        elif entity_type == "LWPOLYLINE":
            return process_ezdxf_polyline(entity, dwg_file_path)
        elif entity_type == "TEXT":
            return process_ezdxf_text(entity, dwg_file_path)
        elif entity_type == "LINE":
            return process_ezdxf_line(entity, dwg_file_path)
        elif entity_type == "ARC":
            return process_ezdxf_arc(entity, dwg_file_path)
        elif entity_type == "INSERT":
            return process_ezdxf_insert(entity, dwg_file_path)
        else:
            # Create a generic symbol for other entity types
            return create_generic_symbol(entity, dwg_file_path)
            
    except Exception as e:
        return None

def process_ezdxf_circle(entity, dwg_file_path):
    """Process ezdxf circle entity."""
    center = entity.dxf.center[:2]  # Only x, y coordinates
    radius = entity.dxf.radius
    
    # Classify based on size and context
    if radius < 2.0:
        symbol_type = "traffic"
        symbol_name = "traffic_signal"
    elif radius < 5.0:
        symbol_type = "electrical"
        symbol_name = "electrical_junction"
    else:
        symbol_type = "general"
        symbol_name = "circle_symbol"
    
    return AutoCADSymbol(
        symbol_id=f"circle_{hash(str(center))}",
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        entity_type="CIRCLE",
        geometry={
            "center": center,
            "radius": radius,
            "area": np.pi * radius ** 2
        },
        layer_name=entity.dxf.layer,
        color="BYLAYER",
        linetype="CONTINUOUS",
        lineweight=-1,
        file_path=dwg_file_path,
        bounding_box=(center[0] - radius, center[1] - radius, 
                     center[0] + radius, center[1] + radius)
    )

def process_ezdxf_polyline(entity, dwg_file_path):
    """Process ezdxf polyline entity."""
    points = list(entity.get_points())
    vertices = len(points)
    
    # Calculate bounding box
    if points:
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    else:
        bbox = None
    
    # Classify based on shape and size
    if vertices == 4:
        symbol_type = "traffic"
        symbol_name = "traffic_detector"
    elif vertices > 4:
        symbol_type = "electrical"
        symbol_name = "electrical_conduit"
    else:
        symbol_type = "general"
        symbol_name = "polyline_symbol"
    
    return AutoCADSymbol(
        symbol_id=f"polyline_{hash(str(points))}",
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        entity_type="LWPOLYLINE",
        geometry={
            "points": points,
            "vertices": vertices
        },
        layer_name=entity.dxf.layer,
        color="BYLAYER",
        linetype="CONTINUOUS",
        lineweight=-1,
        file_path=dwg_file_path,
        bounding_box=bbox
    )

def process_ezdxf_text(entity, dwg_file_path):
    """Process ezdxf text entity."""
    text = entity.dxf.text
    position = entity.dxf.insert[:2]
    
    # Classify based on text content
    if any(keyword in text.lower() for keyword in ['signal', 'detector', 'loop']):
        symbol_type = "traffic"
        symbol_name = "traffic_label"
    elif any(keyword in text.lower() for keyword in ['conduit', 'cable', 'electrical']):
        symbol_type = "electrical"
        symbol_name = "electrical_label"
    else:
        symbol_type = "general"
        symbol_name = "text_label"
    
    return AutoCADSymbol(
        symbol_id=f"text_{hash(text)}",
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        entity_type="TEXT",
        geometry={
            "text": text,
            "position": position
        },
        layer_name=entity.dxf.layer,
        color="BYLAYER",
        linetype="CONTINUOUS",
        lineweight=-1,
        file_path=dwg_file_path
    )

def process_ezdxf_line(entity, dwg_file_path):
    """Process ezdxf line entity."""
    start = entity.dxf.start[:2]
    end = entity.dxf.end[:2]
    
    # Calculate length
    length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    # Classify based on length and layer
    layer = entity.dxf.layer.lower()
    if 'conduit' in layer or 'cable' in layer:
        symbol_type = "electrical"
        symbol_name = "electrical_line"
    elif length > 50:
        symbol_type = "general"
        symbol_name = "long_line"
    else:
        symbol_type = "general"
        symbol_name = "line_symbol"
    
    return AutoCADSymbol(
        symbol_id=f"line_{hash(str(start) + str(end))}",
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        entity_type="LINE",
        geometry={
            "start": start,
            "end": end,
            "length": length
        },
        layer_name=entity.dxf.layer,
        color="BYLAYER",
        linetype="CONTINUOUS",
        lineweight=-1,
        file_path=dwg_file_path
    )

def process_ezdxf_arc(entity, dwg_file_path):
    """Process ezdxf arc entity."""
    center = entity.dxf.center[:2]
    radius = entity.dxf.radius
    
    return AutoCADSymbol(
        symbol_id=f"arc_{hash(str(center))}",
        symbol_name="arc_symbol",
        symbol_type="general",
        entity_type="ARC",
        geometry={
            "center": center,
            "radius": radius
        },
        layer_name=entity.dxf.layer,
        color="BYLAYER",
        linetype="CONTINUOUS",
        lineweight=-1,
        file_path=dwg_file_path
    )

def process_ezdxf_insert(entity, dwg_file_path):
    """Process ezdxf insert entity."""
    name = entity.dxf.name
    position = entity.dxf.insert[:2]
    
    # Classify based on block name
    if any(keyword in name.lower() for keyword in ['signal', 'detector']):
        symbol_type = "traffic"
        symbol_name = "traffic_block"
    elif any(keyword in name.lower() for keyword in ['conduit', 'electrical']):
        symbol_type = "electrical"
        symbol_name = "electrical_block"
    else:
        symbol_type = "general"
        symbol_name = "block_symbol"
    
    return AutoCADSymbol(
        symbol_id=f"insert_{hash(name)}",
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        entity_type="INSERT",
        geometry={
            "name": name,
            "position": position
        },
        layer_name=entity.dxf.layer,
        color="BYLAYER",
        linetype="CONTINUOUS",
        lineweight=-1,
        file_path=dwg_file_path
    )

def create_generic_symbol(entity, dwg_file_path):
    """Create a generic symbol for unknown entity types."""
    return AutoCADSymbol(
        symbol_id=f"generic_{hash(str(entity))}",
        symbol_name="generic_symbol",
        symbol_type="general",
        entity_type=entity.dxftype(),
        geometry={
            "entity_type": entity.dxftype(),
            "layer": entity.dxf.layer
        },
        layer_name=entity.dxf.layer,
        color="BYLAYER",
        linetype="CONTINUOUS",
        lineweight=-1,
        file_path=dwg_file_path
    )

def process_all_dwg_files():
    """Process all .dwg files in the dwg_files directory."""
    print("🚀 Processing .dwg Files with ezdxf")
    print("=" * 50)
    
    # Find all .dwg files
    dwg_files_dir = Path.cwd() / "dwg_files"
    all_dwg_files = []
    
    if dwg_files_dir.exists():
        for dwg_file in dwg_files_dir.rglob("*.dwg"):
            if dwg_file.stat().st_size > 0:  # Skip empty files
                all_dwg_files.append(dwg_file)
    
    print(f"📁 Found {len(all_dwg_files)} .dwg files")
    
    if not all_dwg_files:
        print("❌ No .dwg files found!")
        return
    
    # Initialize trainer
    trainer = AutoCADSymbolTrainer()
    
    # Process files
    total_symbols = 0
    successful_files = 0
    
    print(f"\n📥 Processing .dwg files...")
    
    for dwg_file in all_dwg_files:
        try:
            symbols = extract_symbols_with_ezdxf(str(dwg_file))
            
            if symbols:
                # Add symbols to trainer
                for symbol in symbols:
                    trainer.autocad_symbols.append(symbol)
                
                total_symbols += len(symbols)
                successful_files += 1
                print(f"   ✅ {dwg_file.name}: {len(symbols)} symbols")
            else:
                print(f"   ⚠️  {dwg_file.name}: No symbols extracted")
                
        except Exception as e:
            print(f"   ❌ {dwg_file.name}: {e}")
    
    print(f"\n📊 Processing Summary:")
    print(f"   Files processed: {successful_files}/{len(all_dwg_files)}")
    print(f"   Total symbols extracted: {total_symbols}")
    
    # Show statistics
    stats = trainer.get_training_statistics()
    print(f"\n📈 Training data statistics:")
    print(f"   Total symbols: {stats['total_symbols']}")
    print(f"   Symbol types: {stats['symbol_types']}")
    print(f"   Entity types: {stats['entity_types']}")
    
    # Train model if we have enough data
    if stats['total_symbols'] >= 10:
        print(f"\n🤖 Training model...")
        try:
            config = MiniModelConfig(
                model_type="random_forest",
                n_estimators=100,
                max_depth=10
            )
            
            results = trainer.train_mini_model(config)
            
            if results["success"]:
                print(f"✅ Training successful!")
                print(f"   Train accuracy: {results['train_score']:.3f}")
                print(f"   Test accuracy: {results['test_score']:.3f}")
                print(f"   Model saved to: {results['model_path']}")
                print(f"   Symbol types: {results['symbol_types']}")
                
                # Save training data
                trainer.save_training_data()
                print(f"💾 Training data saved")
                
            else:
                print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during training: {e}")
    else:
        print(f"\n⚠️  Need at least 10 symbols for training, have {stats['total_symbols']}")
        print(f"   Try processing more .dwg files")

def main():
    """Main function."""
    try:
        process_all_dwg_files()
        
        print(f"\n🎉 Process completed!")
        print(f"\n💡 Next steps:")
        print(f"   1. Check the autocad_models/ directory for trained models")
        print(f"   2. Use the trained model for predictions on new drawings")
        print(f"   3. The system now uses ezdxf instead of AutoCAD COM")
        
    except Exception as e:
        print(f"❌ Process failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
