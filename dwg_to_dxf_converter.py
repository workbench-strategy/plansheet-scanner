#!/usr/bin/env python3
"""
DWG to DXF Converter Helper
This script helps convert DWG files to DXF format for processing.
"""

import os
import sys
from pathlib import Path
import subprocess
import shutil

def check_autocad_installation():
    """Check if AutoCAD is installed and accessible."""
    print("🔍 Checking AutoCAD installation...")
    
    # Common AutoCAD installation paths
    autocad_paths = [
        r"C:\Program Files\Autodesk\AutoCAD 2024\acad.exe",
        r"C:\Program Files\Autodesk\AutoCAD 2023\acad.exe",
        r"C:\Program Files\Autodesk\AutoCAD 2022\acad.exe",
        r"C:\Program Files\Autodesk\AutoCAD 2021\acad.exe",
        r"C:\Program Files\Autodesk\AutoCAD 2020\acad.exe",
        r"C:\Program Files\Autodesk\AutoCAD 2019\acad.exe",
        r"C:\Program Files\Autodesk\AutoCAD 2018\acad.exe",
        r"C:\Program Files\Autodesk\AutoCAD 2017\acad.exe",
        r"C:\Program Files\Autodesk\AutoCAD 2016\acad.exe",
        r"C:\Program Files\Autodesk\AutoCAD 2015\acad.exe",
    ]
    
    for path in autocad_paths:
        if os.path.exists(path):
            print(f"   ✅ AutoCAD found at: {path}")
            return path
    
    print("   ❌ AutoCAD not found in common locations")
    return None

def convert_dwg_to_dxf_using_autocad(autocad_path, dwg_file, output_dir):
    """Convert DWG to DXF using AutoCAD command line."""
    try:
        # Create a script file for AutoCAD
        script_content = f"""
(command "._open" "{dwg_file}")
(command "._dxfout" "{output_dir}\\{Path(dwg_file).stem}.dxf")
(command "._close")
(command "._quit")
"""
        
        script_file = output_dir / "convert_script.scr"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Run AutoCAD with the script
        cmd = [autocad_path, "/s", str(script_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"   ✅ Converted: {Path(dwg_file).name}")
            return True
        else:
            print(f"   ❌ Failed to convert: {Path(dwg_file).name}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error converting {Path(dwg_file).name}: {e}")
        return False

def convert_dwg_to_dxf_manual():
    """Provide manual conversion instructions."""
    print("\n📋 Manual DWG to DXF Conversion Instructions:")
    print("=" * 50)
    print("Since we can't automatically convert DWG files, here are your options:")
    print()
    print("Option 1: Use AutoCAD (if you have it installed)")
    print("   1. Open AutoCAD")
    print("   2. Open each .dwg file")
    print("   3. Use SAVEAS command")
    print("   4. Choose 'AutoCAD DXF (*.dxf)' as file type")
    print("   5. Save in the 'dwg_files' directory")
    print()
    print("Option 2: Use Online Converters")
    print("   1. Go to https://cloudconvert.com/dwg-to-dxf")
    print("   2. Upload your .dwg files")
    print("   3. Download the converted .dxf files")
    print("   4. Place them in the 'dwg_files' directory")
    print()
    print("Option 3: Use Free CAD Software")
    print("   1. Download LibreCAD (free)")
    print("   2. Open .dwg files")
    print("   3. Export as .dxf")
    print()
    print("Option 4: Use the Existing Training Data")
    print("   The system already has 105 synthetic symbols trained!")
    print("   You can use the existing model for predictions.")

def create_sample_dxf_files():
    """Create sample DXF files for testing."""
    print("\n🧪 Creating sample DXF files for testing...")
    
    output_dir = Path.cwd() / "dwg_files" / "sample_dxf"
    output_dir.mkdir(exist_ok=True)
    
    # Create a simple DXF file with basic entities
    dxf_content = """0
SECTION
2
HEADER
9
$ACADVER
1
AC1014
9
$DWGCODEPAGE
3
ANSI_1252
0
ENDSEC
0
SECTION
2
ENTITIES
0
CIRCLE
8
TRAFFIC_SIGNALS
10
100.0
20
200.0
30
0.0
40
5.0
0
TEXT
8
TRAFFIC_LABELS
10
100.0
20
190.0
30
0.0
40
2.5
1
SIGNAL_1
0
LINE
8
ELECTRICAL_CONDUIT
10
50.0
20
100.0
30
0.0
11
150.0
21
100.0
31
0.0
0
ENDSEC
0
EOF"""
    
    # Create multiple sample files
    for i in range(5):
        filename = output_dir / f"sample_traffic_{i+1}.dxf"
        with open(filename, 'w') as f:
            f.write(dxf_content)
        print(f"   ✅ Created: {filename.name}")
    
    print(f"\n📁 Sample DXF files created in: {output_dir}")
    print("   You can now test the system with these files!")

def main():
    """Main function."""
    print("🚀 DWG to DXF Converter Helper")
    print("=" * 40)
    
    # Check AutoCAD installation
    autocad_path = check_autocad_installation()
    
    if autocad_path:
        print(f"\n✅ AutoCAD found! You can use it to convert your DWG files.")
        print(f"   AutoCAD path: {autocad_path}")
    else:
        print(f"\n❌ AutoCAD not found. You'll need to convert files manually.")
    
    # Show manual conversion instructions
    convert_dwg_to_dxf_manual()
    
    # Create sample DXF files
    create_sample_dxf_files()
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Convert your .dwg files to .dxf format")
    print(f"   2. Place the .dxf files in the dwg_files directory")
    print(f"   3. Run: python process_dwg_with_ezdxf.py")
    print(f"   4. Or test with the sample DXF files first")
    
    print(f"\n🎯 Alternative: Use the existing trained model!")
    print(f"   The system already has a trained model with 105 symbols.")
    print(f"   You can use it for predictions without converting files.")

if __name__ == "__main__":
    main()
