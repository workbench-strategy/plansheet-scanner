import fitz  # PyMuPDF
import simplekml
import os

# CONFIG
PDF_FILE = "WA13_90%_IH69 CTMS.pdf"
PAGES = [0, 1, 2]  # Add more page numbers as needed
DPI = 300
CONTROL_BOX = {
    # Define pixel coordinates of 4 corners (top-left, top-right, bottom-right, bottom-left)
    "pixels": [(100, 100), (1500, 100), (1500, 2000), (100, 2000)],
    # Corresponding real-world coordinates (lon, lat)
    "geo": [(-95.3922, 29.7361), (-95.3815, 29.7361), (-95.3815, 29.7280), (-95.3922, 29.7280)]
}

def generate_kmz_overlay(pdf_path, pages, dpi, control_box, output_kmz):
    doc = fitz.open(pdf_path)
    kml = simplekml.Kml()

    for i in pages:
        print(f"Processing overlay for page {i}")
        pix = doc.load_page(i).get_pixmap(dpi=dpi)
        image_path = f"page_{i}.png"
        pix.save(image_path)

        # Use four corners to create GroundOverlay
        overlay = kml.newgroundoverlay(name=f"Sheet {i}")
        overlay.icon.href = image_path
        overlay.latlonbox.north = control_box["geo"][0][1]
        overlay.latlonbox.south = control_box["geo"][2][1]
        overlay.latlonbox.east = control_box["geo"][1][0]
        overlay.latlonbox.west = control_box["geo"][0][0]
        overlay.latlonbox.rotation = 0

    kml.savekmz(output_kmz)
    print(f"✅ KMZ with image overlays saved: {output_kmz}")

generate_kmz_overlay(PDF_FILE, PAGES, DPI, CONTROL_BOX, "plan_sheets_overlay.kmz")
