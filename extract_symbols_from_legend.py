import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, simpledialog

# Constants
OUTPUT_DIR = "templates"
os.makedirs(OUTPUT_DIR + "/existing", exist_ok=True)
os.makedirs(OUTPUT_DIR + "/proposed", exist_ok=True)

# Step 1: Load PDF and show selected page
pdf_path = filedialog.askopenfilename(title="Select the CTMS Plan PDF", filetypes=[("PDF Files", "*.pdf")])
page_number = simpledialog.askinteger("Page #", "Enter the page number with the legend (starts at 0):")

doc = fitz.open(pdf_path)
pix = doc.load_page(page_number).get_pixmap(dpi=300)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
if pix.n == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

clone = img.copy()
roi_list = []

# Step 2: OpenCV rectangle selector
def click_and_crop(event, x, y, flags, param):
    global ref_pt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        ref_pt.append((x, y))
        cropping = False
        cv2.rectangle(img, ref_pt[0], ref_pt[1], (0, 255, 0), 2)
        roi = clone[ref_pt[0][1]:ref_pt[1][1], ref_pt[0][0]:ref_pt[1][0]]
        roi_list.append(roi)
        cv2.imshow("Legend Sheet", img)

cv2.namedWindow("Legend Sheet")
cv2.setMouseCallback("Legend Sheet", click_and_crop)
print("🖱️ Select each symbol from the legend. Press ESC when done.")

while True:
    cv2.imshow("Legend Sheet", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

# Step 3: Save each selected symbol
for i, roi in enumerate(roi_list):
    name = simpledialog.askstring("Symbol Name", f"Name for symbol #{i+1} (e.g., CCTV)?")
    category = simpledialog.askstring("Category", "Type 'existing' or 'proposed'")
    category_folder = os.path.join(OUTPUT_DIR, category.lower())
    cv2.imwrite(os.path.join(category_folder, f"{name}.png"), roi)

print("✅ Symbol templates saved.")
