# PlanSheet Scanner - Current State & Next Steps

## 🎯 **CURRENT STATUS**

This is the **plansheet-scanner-new** workspace, which contains the most up-to-date version of the PlanSheet Scanner project. The user has been working on creating mosaics of plan sheets and has encountered an issue with PDF highlighting not appearing on actual words.

## 📁 **PROJECT STRUCTURE**

### **Key Files Transferred from Original Workspace:**
- `efficient_north_detector.py` - Working north arrow detection with two-stage scanning
- `mosaic_improved.py` - Advanced mosaic creation logic
- `confirm_north_detection.py` - North arrow detection verification
- `debug_mosaic_layout.py` - Mosaic layout debugging
- `src/core/smart_plan_seamer.py` - Core seaming logic (creates qa_report.csv)
- `src/cli/smart_seamer.py` - CLI interface
- `templates/YetAnotherNorth/newnorth.png` - User's custom north arrow template

### **User's Original Work:**
- `verify_highlights.py` - Checks for highlights/annotations in PDFs
- `streamlit_legend_extractor.py` - Interactive legend symbol extraction
- `src/core/document_entity_matcher.py` - Main entity matching and highlighting
- `Tables/highlight_cables_variations.py` - Cable highlighting functionality
- `run_cable_matcher.py` - Cable matching pipeline

### **Debug Scripts Created:**
- `debug_highlight_coordinates.py` - Debug coordinate system issues
- `fix_highlight_coordinates.py` - Attempt to fix coordinate problems

## 🚨 **CURRENT ISSUE: Highlighting Not Appearing on Words**

### **Problem Description:**
The user reports that "the highlights are not appearing on the actual words." This means that when the system creates PDF highlights/annotations, they are not properly positioned over the text content.

### **Root Cause Analysis:**
The issue is likely in the coordinate system handling between:
1. **Text detection** (`page.search_for()`) 
2. **Highlight placement** (`page.add_highlight_annot()`)
3. **Page coordinate systems** (PDF points vs display coordinates)

### **Files Involved:**
- `src/core/document_entity_matcher.py` (lines 691, 986) - Uses `page.add_highlight_annot(rect)`
- `Tables/highlight_cables_variations.py` (line 141) - Uses `page.add_highlight_annot(inst)`
- `streamlit_legend_extractor.py` - Uses canvas drawing for visual highlighting

## 🔧 **IMMEDIATE NEXT STEPS**

### **1. Debug the Highlighting Issue**
```bash
# Run the debug script to see what's happening with coordinates
python debug_highlight_coordinates.py your_pdf.pdf "CCTV"

# Run the fix script to attempt coordinate corrections
python fix_highlight_coordinates.py your_pdf.pdf "CCTV"
```

### **2. Check Coordinate System Issues**
The debug scripts will show:
- Exact coordinates being found by `page.search_for()`
- Whether coordinates are within page bounds
- Whether highlights are being created successfully
- Any errors in the highlighting process

### **3. Potential Fixes to Implement**
Based on the analysis, the issue might be:
- **Page rotation** - Coordinates need transformation for rotated pages
- **Coordinate system mismatch** - PDF points vs display coordinates
- **DPI/scale issues** - Coordinate scaling problems

## 🎯 **PROJECT GOALS**

### **Primary Objectives:**
1. **Fix PDF highlighting** - Ensure highlights appear on actual words
2. **Complete mosaic creation** - Finish the plan sheet mosaicking functionality
3. **North arrow detection** - Ensure proper north alignment for sheets
4. **Symbol extraction** - Improve legend symbol detection and extraction

### **Secondary Objectives:**
1. **Cable matching** - Improve cable entity detection and highlighting
2. **QA reporting** - Generate comprehensive quality assurance reports
3. **User interface** - Improve Streamlit interface for symbol extraction

## 📋 **WORKFLOW STATUS**

### **✅ Completed:**
- North arrow detection with efficient two-stage scanning
- Basic mosaic layout algorithm
- Text extraction and entity matching
- QA report generation (qa_report.csv)
- Debug tools for coordinate issues

### **🔄 In Progress:**
- PDF highlighting coordinate system fix
- Mosaic creation with north alignment
- Symbol extraction improvements

### **❌ Blocked:**
- PDF highlighting until coordinate issue is resolved

## 🛠️ **TECHNICAL DETAILS**

### **Key Dependencies:**
- PyMuPDF (fitz) - PDF processing and highlighting
- OpenCV (cv2) - Image processing and template matching
- NumPy - Numerical operations
- Streamlit - Web interface for symbol extraction

### **Coordinate Systems:**
- **PDF Points** - 72 points per inch (PyMuPDF default)
- **Display Coordinates** - Pixels or scaled coordinates
- **Page Coordinates** - Relative to page origin

### **Highlighting Methods:**
1. `page.add_highlight_annot(rect)` - Creates PDF highlight annotations
2. `page.draw_rect(rect, color, width)` - Draws visual rectangles
3. Canvas drawing (Streamlit) - Interactive visual highlighting

## 🎯 **RECOMMENDED APPROACH**

### **Step 1: Debug Current Issue**
1. Run `debug_highlight_coordinates.py` on a test PDF
2. Analyze the output to understand coordinate behavior
3. Identify specific coordinate system issues

### **Step 2: Implement Fix**
1. Modify `document_entity_matcher.py` to use corrected coordinates
2. Test with the fix script
3. Verify highlights appear on actual words

### **Step 3: Continue Mosaic Work**
1. Integrate working north arrow detection into mosaic pipeline
2. Test mosaic creation with corrected highlighting
3. Improve layout algorithm for better sheet placement

### **Step 4: Enhance Symbol Extraction**
1. Improve Streamlit interface
2. Add batch processing capabilities
3. Enhance symbol matching accuracy

## 📞 **USER CONTEXT**

The user has been working on this project for several sessions, focusing on:
- Creating mosaics of plan sheets
- Detecting and aligning north arrows
- Extracting and highlighting symbols/text
- Building a comprehensive plan sheet processing pipeline

The user is experienced with the project and understands the technical challenges. They want to resolve the highlighting issue first, then continue with mosaic creation and symbol extraction improvements.

## 🔍 **DEBUGGING TIPS**

1. **Check page rotation** - Rotated pages may need coordinate transformation
2. **Verify coordinate bounds** - Ensure coordinates are within page dimensions
3. **Test with simple text** - Use common words like "CCTV" for testing
4. **Compare coordinate systems** - Check if PDF points vs display coordinates are the issue
5. **Use visual confirmation** - Draw rectangles to see where highlights are being placed

## 📝 **NOTES FOR NEXT AGENT**

- The user has been very helpful and responsive to debugging efforts
- They understand the technical challenges and want systematic solutions
- Focus on the highlighting issue first, as it's blocking other progress
- The debug scripts should provide clear information about what's happening
- Coordinate system issues are common in PDF processing - check for rotation, scaling, and bounds

---

**Last Updated:** August 15, 2025  
**Current Focus:** Fixing PDF highlighting coordinate system  
**Next Priority:** Complete mosaic creation with north alignment

