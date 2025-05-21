import os
import cv2
import numpy as np
import fitz  # PyMuPDF
import simplekml
from glob import glob
import argparse
import json

# Constants
TEMPLATE_DIR = "templates"

def load_control_points(json_path):
    """Loads control points from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    control_points = {}
    points_data = data.get("pixel_to_geo", data)

    for k, v in points_data.items():
        try:
            x_str, y_str = k.split(',')
            control_points[(int(x_str), int(y_str))] = tuple(v)
        except ValueError:
            print(f"Warning: Could not parse control point key: {k}. Skipping.")
            continue
    if not control_points:
        raise ValueError("No valid control points loaded. Check the JSON file format.")
    return control_points

def affine_transform(pixels, control_points):
    """Compute affine transform based on control_points and apply it."""
    import numpy as np
    from numpy.linalg import lstsq

    if not control_points or len(control_points) < 3:
        raise ValueError("At least 3 control points are required for affine transformation.")

    src = np.array(list(control_points.keys()), dtype=np.float32)
    dst = np.array([control_points[k] for k in control_points], dtype=np.float32)

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

def match_templates(page_img, template_dir):
    """Returns matched locations for each template."""
    matches = {}
    for category in ["existing", "proposed"]:
        for path in glob(f"{template_dir}/{category}/*.png"):
            name = os.path.splitext(os.path.basename(path))[0]
            template = cv2.imread(path, 0)
            if template is None:
                print(f"Warning: Could not read template image: {path}. Skipping.")
                continue
            gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.82
            loc = np.where(result >= threshold)
            for pt in zip(*loc[::-1]):
                matches.setdefault((category, name), []).append(pt)
    return matches

def main():
    parser = argparse.ArgumentParser(description="Match symbols in a PDF and generate a KMZ file.")
    parser.add_argument("--pdf-file", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output-kmz", required=True, help="Path for the generated KMZ file.")
    parser.add_argument("--control-points-file", required=True, help="Path to the JSON file containing control points.")
    args = parser.parse_args()

    try:
        control_points = load_control_points(args.control_points_file)
    except Exception as e:
        print(f"Error loading control points: {e}")
        return

    kml = simplekml.Kml()
    try:
        doc = fitz.open(args.pdf_file)
    except Exception as e:
        print(f"Error opening PDF file {args.pdf_file}: {e}")
        return

    for i, page in enumerate(doc):
        print(f"Processing page {i + 1}")
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        matches = match_templates(img, TEMPLATE_DIR)
        if not matches:
            print(f"No matches found on page {i+1}.")
            continue
            
        for (category, name), points in matches.items():
            folder = kml.newfolder(name=f"{category.title()} - {name} (Page {i+1})")
            try:
                transformed_points = affine_transform([pt for pt in points], control_points)
                for pt_orig, (lon, lat) in zip(points, transformed_points):
                    folder.newpoint(name=f"{name}", coords=[(lon, lat)])
            except ValueError as e:
                print(f"Skipping points for {name} on page {i+1} due to affine transform error: {e}")
                continue

    try:
        kml.savekmz(args.output_kmz)
        print(f"✅ KMZ saved to {args.output_kmz}")
    except Exception as e:
        print(f"Error saving KMZ file {args.output_kmz}: {e}")

if __name__ == "__main__":
    main()
