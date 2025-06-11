import fitz
import cv2
import pytesseract
import numpy as np
import xml.etree.ElementTree as ET
import zipfile
import os
import logging
import tkinter as tk  # For file dialog
from tkinter import filedialog

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions for Folder/File Paths ---
def ensure_dir_exists(dir_path):
    """Creates a directory if it doesn't exist."""
    try:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Directory created (or already exists): {dir_path}")
    except Exception as e:
        logging.error(f"Error creating directory {dir_path}: {e}")
        return False
    return True

def get_pdf_path():
    """Prompts the user to select a PDF file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        logging.info(f"User selected PDF: {file_path}")
        return file_path
    else:
        logging.warning("No PDF file selected.")
        return None

# --- 1. PDF to Images ---
def pdf_to_images(pdf_path, output_folder="images", dpi=300):
    """Converts each PDF page to a high-resolution image."""
    if not ensure_dir_exists(output_folder):
        return None

    try:
        pdf_document = fitz.open(pdf_path)
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            output_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
            pix.save(output_path)
            logging.info(f"Saved page {page_number + 1} to {output_path}")
        pdf_document.close()
        return output_folder
    except Exception as e:
        logging.error(f"Error converting PDF to images: {e}")
        return None

# --- 2. Legend Extraction (Placeholder - Complex) ---
def extract_legend_data(image_path):
    """
    Placeholder for legend extraction. Needs image processing and OCR.
    """
    logging.warning("Legend extraction is a placeholder. Needs implementation.")
    #  ---  Example (VERY simplified) ---
    #  In reality, you'd need contour detection, cropping, OCR, etc.
    legend_data = {
        "symbol_1": "Description for Symbol 1",
        "symbol_2": "Description for Symbol 2",
        # ... more symbols
    }
    return legend_data

# --- 3. Symbol Recognition (Placeholder - Complex) ---
def find_symbols_in_image(image_path, legend_data, symbol_folder="symbols"):
    """Finds symbols from the legend in the image."""

    image = cv2.imread(image_path)
    found_units = []

    for symbol, description in legend_data.items():
        symbol_image_path = os.path.join(symbol_folder, f"{symbol}.png")
        try:
            symbol_image = cv2.imread(symbol_image_path)
            result = cv2.matchTemplate(image, symbol_image, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.8)  # Threshold for match
            h, w = symbol_image.shape[:2]
            for pt in zip(*locations[::-1]):  # Reverse x, y
                found_units.append({
                    "symbol": symbol,
                    "description": description,
                    "x": pt[0],
                    "y": pt[1],
                    "height": h,  # Add height and width
                    "width": w,
                    "image_path": image_path  # Keep track of the source image
                })
            logging.debug(f"Found {len(locations[0])} instances of symbol '{symbol}' in {image_path}")
        except FileNotFoundError:
            logging.error(f"Symbol image not found: {symbol_image_path}")
        except Exception as e:
            logging.error(f"Error finding symbol '{symbol}' in {image_path}: {e}")
    return found_units

# --- 4. Unit Assembly (Using Matchlines - Placeholder) ---
def assemble_units_using_matchlines(all_units):
    """
    Assembles units from different plan sheets using matchlines.
    This is a placeholder; matchline detection is complex.
    """
    logging.warning("Unit assembly using matchlines is a placeholder.")

    assembled_data = {}
    #  This is where you'd implement the logic to:
    #  1. Detect matchlines in the images (OpenCV).
    #  2. Determine the transformation between coordinate systems of adjacent sheets.
    #  3. Transform the coordinates of units from each sheet to a common coordinate system.
    #  For now, let's just group by page:
    for unit in all_units:
        page_num = os.path.splitext(os.path.basename(unit["image_path"]))[0].split("_")[1] # Extract page number from image name
        if page_num not in assembled_data:
            assembled_data[page_num] = []
        assembled_data[page_num].append(unit)
    return assembled_data

# --- 5. KMZ Export ---
def create_kml(assembled_data, output_folder, output_path="output.kml", symbol_folder="symbols"):
    """Creates a KMZ file from the assembled data."""

    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")
    ET.SubElement(document, "name").text = "Extracted Features"

    for page, units in assembled_data.items():
        folder = ET.SubElement(document, "Folder")
        ET.SubElement(folder, "name").text = page  # Group by page
        for unit in units:
            placemark = ET.SubElement(folder, "Placemark")
            ET.SubElement(placemark, "name").text = unit["symbol"]
            ET.SubElement(placemark, "description").text = unit["description"]
            point = ET.SubElement(placemark, "Point")
            ET.SubElement(point, "coordinates").text = f"{unit['x']},{unit['y']}"  # Pixel coords

            # Add Style with icon scaling
            style = ET.SubElement(placemark, "Style")
            icon_style = ET.SubElement(style, "IconStyle")
            scale = ET.SubElement(icon_style, "scale")
            avg_size = (unit['height'] + unit['width']) / 20  # You might need to tweak this
            scale.text = str(max(0.5, min(avg_size, 2.0)))  # Clamp scale
            icon = ET.SubElement(icon_style, "Icon")
            icon_href = os.path.join(symbol_folder, f"{unit['symbol']}.png").replace("\\", "/") # For KML compatibility
            ET.SubElement(icon, "href").text = icon_href

    tree = ET.ElementTree(kml)
    tree.write(output_path)

    #  Package into KMZ
    kmz_path = output_path.replace(".kml", ".kmz")
    with zipfile.ZipFile(kmz_path, "w", zipfile.ZIP_DEFLATED) as kmz:
        kmz.write(output_path, "doc.kml", "doc.kml")  # Standard name
        for image_file in os.listdir(output_folder):
            if image_file.endswith(".png"):
                kmz.write(os.path.join(output_folder, image_file), image_file)
    logging.info(f"KMZ file created: {kmz_path}")

# --- 6. MicroStation Export (Conceptual) ---
def create_dgn(assembled_data, output_path="output.dgn"):
    """
    Conceptual:  Creates a MicroStation DGN file.
    This requires a MicroStation library or API (e.g., PythonMicroStation).
    """
    logging.warning("MicroStation export is not implemented.")
    print("MicroStation export is not implemented in this basic example.")

# --- Main Execution ---
if __name__ == "__main__":
    # Define default folder names
    image_folder = "images"
    output_folder = "output"
    symbol_folder = "symbols"  # Folder to store symbol images (if you extract them)

    # Ensure output directories exist
    ensure_dir_exists(image_folder)
    ensure_dir_exists(output_folder)
    ensure_dir_exists(symbol_folder)

    # 0. Get PDF Path
    pdf_path = None
    while not pdf_path:
        pdf_path = get_pdf_path()
        if not pdf_path:
            logging.error("No PDF file provided. Exiting.")
            exit()

    # 1. Convert PDF to Images
    image_folder = pdf_to_images(pdf_path, image_folder)
    if not image_folder:
        logging.error("PDF to images conversion failed. Exiting.")
        exit()

    all_units = []
    # Process each image
    for image_file in sorted(os.listdir(image_folder)):
        if image_file.endswith(".png"):
            image_path = os.path.join(image_folder, image_file)
            logging.info(f"Processing image: {image_path}")

            # 2. Extract Legend Data (Placeholder)
            legend_data = extract_legend_data(image_path)  #  <--  IMPLEMENT THIS

            # 3. Find Symbols
            units = find_symbols_in_image(image_path, legend_data, symbol_folder)  #  <-- IMPLEMENT THIS
            all_units.extend(units)

    # 4. Assemble Units using Matchlines (Placeholder)
    assembled_data = assemble_units_using_matchlines(all_units)

    # 5. KMZ Export
    create_kml(assembled_data, output_folder, os.path.join(output_folder, "its_data.kml"), symbol_folder)

    # 6. MicroStation Export (Conceptual)
    create_dgn(assembled_data, os.path.join(output_folder, "its_data.dgn"))