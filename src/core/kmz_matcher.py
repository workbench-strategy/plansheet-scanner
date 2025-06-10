import os
import cv2
import numpy as np
import fitz  # PyMuPDF
import simplekml
from glob import glob
import argparse
import json
import math
from numpy.linalg import lstsq

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
    params, _, _, _ = lstsq(A, B, rcond=None)
    return params

# Core logic for matching and generating KMZ
# ...existing code from match_and_generate_kmz.py will be moved here...
