import cv2
import numpy as np
import fitz  # PyMuPDF

# CONFIG
PDF_FILE = "WA13_90%_IH69 CTMS.pdf"
PAGE_NUM = 0
DPI = 300
OUTPUT_IMAGE = "matched_lines.png"

# Load and render the page
doc = fitz.open(PDF_FILE)
pix = doc.load_page(PAGE_NUM).get_pixmap(dpi=DPI)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
if pix.n == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# Convert to grayscale and detect edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Draw lines for visualization
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite(OUTPUT_IMAGE, img)
print(f"✅ Saved line-matched image as {OUTPUT_IMAGE}")
