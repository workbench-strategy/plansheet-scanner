import fitz  # PyMuPDF
import cv2
import numpy as np
import os

# Constants
OUTPUT_DIR = "templates"
os.makedirs(OUTPUT_DIR + "/existing", exist_ok=True)
os.makedirs(OUTPUT_DIR + "/proposed", exist_ok=True)

# Step 1: Load PDF and show selected page
while True:
    pdf_path = input("Enter the path to the CTMS Plan PDF: ")
    if os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
        break
    print("Invalid PDF path or file does not exist. Please try again.")

while True:
    try:
        page_number_str = input("Enter the page number with the legend (starts at 0): ")
        page_number = int(page_number_str)
        break
    except ValueError:
        print("Invalid page number. Please enter an integer.")

doc = fitz.open(pdf_path)
if page_number < 0 or page_number >= doc.page_count:
    print(f"Error: Page number {page_number} is out of range for PDF with {doc.page_count} pages.")
    exit()

pix = doc.load_page(page_number).get_pixmap(dpi=300)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
if pix.n == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

clone = img.copy()
roi_list = []
ref_pt = []  # Initialize ref_pt
cropping = False  # Initialize cropping

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
    while True:
        name = input(f"Name for symbol #{i+1} (e.g., CCTV -- press Enter to skip this symbol): ")
        if not name.strip() and roi_list:  # Allow skipping if a name is not provided
            print(f"Skipping symbol #{i+1}")
            break
        if name.strip():  # Proceed if name is not empty
            break
        print("Symbol name cannot be empty.")
    if not name.strip():  # If skipped, continue to next ROI
        continue

    while True:
        category = input("Type 'existing' or 'proposed': ").lower()
        if category in ["existing", "proposed"]:
            break
        print("Invalid category. Please type 'existing' or 'proposed'.")

    category_folder = os.path.join(OUTPUT_DIR, category)  # category is already lower
    os.makedirs(category_folder, exist_ok=True)  # Ensure the specific category folder exists
    cv2.imwrite(os.path.join(category_folder, f"{name}.png"), roi)

print("✅ Symbol templates saved.")
