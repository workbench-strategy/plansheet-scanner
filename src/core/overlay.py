import fitz  # PyMuPDF
import simplekml
import os

def generate_kmz_overlay(pdf_path, pages, dpi, control_box, output_kmz):
    doc = fitz.open(pdf_path)
    kml = simplekml.Kml()
    for i in pages:
        print(f"Processing overlay for page {i}")
        pix = doc.load_page(i).get_pixmap(dpi=dpi)
        image_path = f"page_{i}.png"
        pix.save(image_path)
        overlay = kml.newgroundoverlay(name=f"Sheet {i}")
        overlay.icon.href = image_path
        overlay.latlonbox.north = control_box["geo"][0][1]
        overlay.latlonbox.south = control_box["geo"][2][1]
        overlay.latlonbox.east = control_box["geo"][1][0]
        overlay.latlonbox.west = control_box["geo"][0][0]
        overlay.latlonbox.rotation = 0
    kml.savekmz(output_kmz)
    print(f"✅ KMZ with image overlays saved: {output_kmz}")
