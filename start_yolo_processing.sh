#!/bin/bash

# YOLO Processing Startup Script for GitHub Codespaces
# Optimized for 32GB RAM machine

echo "🚀 YOLO Processing Startup - GitHub Codespaces 32GB"
echo "=================================================="

# Check if we're in GitHub Codespaces
if [ -n "$CODESPACES" ]; then
    echo "✅ Running in GitHub Codespaces"
    echo "🖥️  Machine Type: 32GB RAM"
else
    echo "⚠️  Running locally (not in Codespaces)"
fi

# Check system resources
echo ""
echo "📊 System Resources:"
echo "  CPU Cores: $(nproc)"
echo "  Total RAM: $(free -h | awk 'NR==2{print $2}')"
echo "  Available RAM: $(free -h | awk 'NR==2{print $7}')"

# Verify Python and dependencies
echo ""
echo "🔍 Verifying Python Environment:"
python --version
pip list | grep -E "(opencv|numpy|fitz|psutil)"

# Check if as_built_drawings directory exists
if [ -d "as_built_drawings" ]; then
    PDF_COUNT=$(find as_built_drawings -name "*.pdf" | wc -l)
    echo "✅ Found $PDF_COUNT PDF files in as_built_drawings/"
else
    echo "❌ as_built_drawings/ directory not found!"
    echo "Please ensure your PDF files are in the as_built_drawings/ directory"
    exit 1
fi

# Create output directories
echo ""
echo "📁 Creating output directories..."
mkdir -p yolo_processed_data/{images,metadata,features,reports}

# Run lightweight test first
echo ""
echo "🧪 Running system tests..."
python LightweightTest.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed! Ready for YOLO processing."
    echo ""
    echo "🚀 Starting YOLO processing..."
    echo "   This will process all $PDF_COUNT PDF files with maximum quality."
    echo "   Expected time: 20-30 minutes"
    echo "   Press Ctrl+C to stop at any time"
    echo ""
    
    # Start YOLO processing
    python yoloprocessing.py
    
    echo ""
    echo "🎉 Processing complete!"
    echo "📊 Check yolo_processed_data/reports/ for detailed results"
else
    echo ""
    echo "❌ System tests failed. Please check the errors above."
    exit 1
fi
