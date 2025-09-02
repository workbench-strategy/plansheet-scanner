#!/usr/bin/env python3
"""
Test script for Enhanced Roadway Detector
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def test_roadway_detection():
    """Test the enhanced roadway detector with as-built images."""
    print("🛣️ Testing Enhanced Roadway Detector")
    print("=" * 50)
    
    # Find test images
    yolo_images_dir = Path("yolo_processed_data_local/images")
    if not yolo_images_dir.exists():
        print("❌ YOLO processed images directory not found")
        return
    
    # Find as-built images
    as_built_images = [f for f in yolo_images_dir.glob("*.png") if "asbuilt" in f.name.lower()]
    print(f"📁 Found {len(as_built_images)} as-built drawings for testing")
    
    if not as_built_images:
        print("❌ No as-built images found")
        return
    
    # Test with first image
    test_image = as_built_images[0]
    print(f"\n🔍 Testing with: {test_image.name}")
    
    # Load image
    image = cv2.imread(str(test_image))
    if image is None:
        print(f"❌ Could not load image: {test_image}")
        return
    
    print(f"✅ Image loaded successfully: {image.shape}")
    
    # Basic edge detection test
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 60)
    
    # Find lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=30,
        maxLineGap=10
    )
    
    print(f"✅ Found {len(lines) if lines is not None else 0} lines")
    
    # Create visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Lines detected
    ax2.imshow(image_rgb)
    ax2.set_title('Lines Detected', fontsize=14, fontweight='bold')
    
    if lines is not None:
        for line in lines[:50]:  # Show first 50 lines
            x1, y1, x2, y2 = line[0]
            ax2.plot([x1, x2], [y1, y2], color='red', linewidth=1, alpha=0.7)
    
    ax2.axis('off')
    
    # Save visualization
    output_path = Path("roadway_detections")
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_path / f"roadway_test_{timestamp}.png"
    
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Visualization saved: {fig_path}")
    print(f"\n🎉 Basic roadway detection test complete!")

if __name__ == "__main__":
    test_roadway_detection()
