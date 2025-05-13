import os
import cv2
import numpy as np
import fitz  # PyMuPDF
import simplekml
from glob import glob

# Constants
TEMPLATE_DIR = "templates"
PDF_FILE = "WA13_90%_IH69 CTMS.pdf"  # <-- You can change this
OUTPUT_KMZ = "matched_symbols.kmz"
CONTROL_POINTS = {
    # Manually define control points (pixel_x, pixel_y): (lon, lat)
    # These should be for a single page and reused if sheets have the same layout
    (100, 100): (-95.3922, 29.7361),  # Montrose
    (1500, 100): (-95.3815, 29.7361),  # Main
    (100, 2000): (-95.3922, 29.7280),  # South point 1
}

def affine_transform(pixels):
    """Compute affine transform based on CONTROL_POINTS and apply it."""
    import numpy as np
    from numpy.linalg import lstsq

    src = np.array(list(CONTROL_POINTS.keys()), dtype=np.float32)
    dst = np.array([CONTROL_POINTS[k] for k in CONTROL_POINTS], dtype=np.float32)

    A = []
    B = []
    for (x, y), (lon, lat) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.extend([lon, lat])
    A = np.array(A)
    B = np.array(B)
    transform, _, _, _ = lstsq(A, B, rcond=None)

    def apply(pt):
        x, y = pt
        lon = transform[0]*x + transform[1]*y + transform[2]
        lat = transform[3]*x + transform[4]*y + transform[5]
        return (lon, lat)

    return [apply(p) for p in pixels]

def match_templates(page_img, templates):
    """Returns matched locations for each template."""
    matches = {}
    for category in ["existing", "proposed"]:
        for path in glob(f"{TEMPLATE_DIR}/{category}/*.png"):
            name = os.path.splitext(os.path.basename(path))[0]
            template = cv2.imread(path, 0)
            gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.82
            loc = np.where(result >= threshold)
            for pt in zip(*loc[::-1]):
                matches.setdefault((category, name), []).append(pt)
    return matches

def main():
    kml = simplekml.Kml()
    doc = fitz.open(PDF_FILE)

    for i, page in enumerate(doc):
        print(f"Processing page {i + 1}")
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        matches = match_templates(img, TEMPLATE_DIR)
        for (category, name), points in matches.items():
            folder = kml.newfolder(name=f"{category.title()} - {name}")
            for pt in points:
                lon, lat = affine_transform([pt])[0]
                folder.newpoint(name=f"{name}", coords=[(lon, lat)])

    kml.savekmz(OUTPUT_KMZ)
    print(f"✅ KMZ saved to {OUTPUT_KMZ}")

if __name__ == "__main__":
    main()
