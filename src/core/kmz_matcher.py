import argparse
import json
import math
import os
from glob import glob

import cv2
import fitz  # PyMuPDF
import numpy as np
import simplekml
from numpy.linalg import lstsq

TEMPLATE_DIR = "templates"


def load_control_points(json_path):
    """Loads control points from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    control_points = {}
    points_data = data.get("pixel_to_geo", data)
    for k, v in points_data.items():
        try:
            x_str, y_str = k.split(",")
            control_points[(int(x_str), int(y_str))] = tuple(v)
        except ValueError:
            print(f"Warning: Could not parse control point key: {k}. Skipping.")
            continue
    if not control_points:
        raise ValueError("No valid control points loaded. Check the JSON file format.")
    return control_points


def affine_transform(pixels, control_points):
    """
    Compute affine transformation parameters from control points.

    Args:
        pixels: List of pixel coordinates to transform
        control_points: Dictionary mapping pixel coordinates to geographic coordinates

    Returns:
        params: 6-element array of affine transformation parameters

    Raises:
        ValueError: If control points are insufficient or degenerate (collinear)
    """
    if not control_points or len(control_points) < 3:
        raise ValueError(
            "At least 3 control points are required for affine transformation."
        )

    # Extract source and destination points
    src = np.array(list(control_points.keys()), dtype=np.float32)
    dst = np.array([control_points[k] for k in control_points], dtype=np.float32)

    # Check for degenerate cases (collinear points)
    if len(src) >= 3:
        # Check if points are collinear by computing the area of the triangle
        # formed by the first three points
        p1, p2, p3 = src[:3]

        # Compute area using cross product (2D equivalent)
        # Area = |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)| / 2
        area = (
            abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
            / 2
        )

        # If area is very small, points are approximately collinear
        if area < 1e-6:  # Threshold for numerical precision
            raise ValueError(
                f"Control points are collinear (degenerate case). "
                f"Area of triangle formed by first 3 points: {area:.6f}. "
                f"Please provide at least 3 non-collinear control points for accurate transformation."
            )

        # For more than 3 points, check if all points lie on a line
        if len(src) > 3:
            # Use linear regression to check if all points lie on a line
            x_coords = src[:, 0]
            y_coords = src[:, 1]

            # Fit a line through all points
            A_line = np.column_stack([x_coords, np.ones_like(x_coords)])
            try:
                slope, intercept = np.linalg.lstsq(A_line, y_coords, rcond=None)[0]

                # Calculate residuals (distance from points to the fitted line)
                predicted_y = slope * x_coords + intercept
                residuals = np.abs(y_coords - predicted_y)
                max_residual = np.max(residuals)

                # If all points are very close to the line, they're collinear
                if max_residual < 1e-3:  # Threshold for collinearity
                    raise ValueError(
                        f"All control points are approximately collinear. "
                        f"Maximum deviation from fitted line: {max_residual:.6f}. "
                        f"Please provide control points that form a non-degenerate triangle or quadrilateral."
                    )
            except np.linalg.LinAlgError:
                # This can happen if all x-coordinates are the same (vertical line)
                if np.allclose(x_coords, x_coords[0]):
                    raise ValueError(
                        "All control points have the same x-coordinate (vertical line). "
                        "Please provide control points that form a non-degenerate triangle or quadrilateral."
                    )

    # Build the system of equations for affine transformation
    A = []
    B = []
    for (x, y), (lon, lat) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.extend([lon, lat])

    A = np.array(A)
    B = np.array(B)

    # Solve the system using least squares
    try:
        params, residuals, rank, s = lstsq(A, B, rcond=None)

        # Check if the solution is well-conditioned
        if rank < 6:
            raise ValueError(
                f"Affine transformation matrix is rank-deficient (rank={rank}). "
                f"This indicates the control points do not provide sufficient constraints. "
                f"Please provide more diverse control point locations."
            )

        # Check residuals to see how well the transformation fits
        if len(residuals) > 0 and np.any(residuals > 1e-3):
            print(
                f"Warning: Large residuals detected in affine transformation. "
                f"Maximum residual: {np.max(residuals):.6f}. "
                f"Consider checking control point accuracy."
            )

        return params

    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Failed to compute affine transformation: {str(e)}. "
            f"This may indicate degenerate control points or numerical instability."
        )


# Core logic for matching and generating KMZ
# ...existing code from match_and_generate_kmz.py will be moved here...
