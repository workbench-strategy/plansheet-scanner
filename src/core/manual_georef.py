import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def export_control_points_to_geojson(
    control_points: Dict[str, Any],
    output_path: str,
    description: str = "Control Points",
    crs: str = "EPSG:4326",
) -> str:
    """
    Export selected control points to GeoJSON format.

    Args:
        control_points: Dictionary containing control point data
        output_path: Path for the output GeoJSON file
        description: Description for the GeoJSON feature collection
        crs: Coordinate reference system (default: EPSG:4326)

    Returns:
        Path to the created GeoJSON file

    Raises:
        ValueError: If control_points format is invalid
        IOError: If file cannot be written
    """
    if not isinstance(control_points, dict):
        raise ValueError("control_points must be a dictionary")

    if "geo" not in control_points or "pixel" not in control_points:
        raise ValueError("control_points must contain both 'geo' and 'pixel' keys")

    geo_coords = control_points["geo"]
    pixel_coords = control_points["pixel"]

    if not isinstance(geo_coords, list) or not isinstance(pixel_coords, list):
        raise ValueError("'geo' and 'pixel' must be lists")

    if len(geo_coords) != len(pixel_coords):
        raise ValueError("'geo' and 'pixel' lists must have the same length")

    if len(geo_coords) == 0:
        raise ValueError("No control points provided")

    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "properties": {
            "description": description,
            "created": datetime.now().isoformat(),
            "control_point_count": len(geo_coords),
            "source": "manual_georef.py",
        },
        "features": [],
    }

    # Add each control point as a Point feature
    for i, (geo_coord, pixel_coord) in enumerate(zip(geo_coords, pixel_coords)):
        if len(geo_coord) != 2 or len(pixel_coord) != 2:
            raise ValueError(
                f"Control point {i} must have exactly 2 coordinates (lon, lat) and (x, y)"
            )

        lon, lat = geo_coord
        x, y = pixel_coord

        # Validate coordinates
        if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
            raise ValueError(f"Geographic coordinates at index {i} must be numeric")

        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"Pixel coordinates at index {i} must be numeric")

        # Validate longitude and latitude ranges
        if not -180 <= lon <= 180:
            raise ValueError(
                f"Longitude {lon} at index {i} is out of valid range (-180 to 180)"
            )

        if not -90 <= lat <= 90:
            raise ValueError(
                f"Latitude {lat} at index {i} is out of valid range (-90 to 90)"
            )

        feature = {
            "type": "Feature",
            "properties": {
                "id": i,
                "point_id": f"CP_{i:03d}",
                "pixel_x": x,
                "pixel_y": y,
                "longitude": lon,
                "latitude": lat,
                "description": f"Control Point {i+1}",
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],  # GeoJSON uses [longitude, latitude] order
            },
        }

        geojson["features"].append(feature)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Write GeoJSON file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        logger.info(f"GeoJSON file created successfully: {output_path}")
        logger.info(f"Exported {len(geo_coords)} control points")

        return output_path

    except Exception as e:
        raise IOError(f"Failed to write GeoJSON file: {e}")


def validate_control_points(control_points: Dict[str, Any]) -> bool:
    """
    Validate control points data structure.

    Args:
        control_points: Dictionary containing control point data

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not isinstance(control_points, dict):
        raise ValueError("control_points must be a dictionary")

    required_keys = ["geo", "pixel"]
    for key in required_keys:
        if key not in control_points:
            raise ValueError(f"control_points must contain '{key}' key")

    geo_coords = control_points["geo"]
    pixel_coords = control_points["pixel"]

    if not isinstance(geo_coords, list) or not isinstance(pixel_coords, list):
        raise ValueError("'geo' and 'pixel' must be lists")

    if len(geo_coords) != len(pixel_coords):
        raise ValueError("'geo' and 'pixel' lists must have the same length")

    if len(geo_coords) == 0:
        raise ValueError("No control points provided")

    # Validate each coordinate pair
    for i, (geo_coord, pixel_coord) in enumerate(zip(geo_coords, pixel_coords)):
        if len(geo_coord) != 2 or len(pixel_coord) != 2:
            raise ValueError(f"Control point {i} must have exactly 2 coordinates")

        lon, lat = geo_coord
        x, y = pixel_coord

        # Check coordinate types
        if not all(isinstance(val, (int, float)) for val in [lon, lat, x, y]):
            raise ValueError(f"All coordinates at index {i} must be numeric")

        # Check geographic coordinate ranges
        if not -180 <= lon <= 180:
            raise ValueError(f"Longitude {lon} at index {i} is out of valid range")

        if not -90 <= lat <= 90:
            raise ValueError(f"Latitude {lat} at index {i} is out of valid range")

    logger.info(f"Control points validation passed: {len(geo_coords)} points")
    return True


def create_sample_control_points() -> Dict[str, Any]:
    """
    Create sample control points for testing and demonstration.

    Returns:
        Dictionary with sample control points
    """
    return {
        "geo": [
            [-122.4194, 37.7749],  # San Francisco coordinates
            [-122.4000, 37.7749],
            [-122.4000, 37.7600],
            [-122.4194, 37.7600],
        ],
        "pixel": [[100, 100], [500, 100], [500, 400], [100, 400]],  # Pixel coordinates
    }


def main():
    """Main CLI function for manual georeferencing and GeoJSON export."""
    parser = argparse.ArgumentParser(
        description="Manual pixel-to-GPS calibration with GeoJSON export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manual_georef.py --export sample_control_points.json --output control_points.geojson
  python manual_georef.py --sample --output sample.geojson --description "Sample Control Points"
  python manual_georef.py --validate control_points.json
        """,
    )

    parser.add_argument("--export", help="Path to JSON file containing control points")
    parser.add_argument(
        "--output",
        "-o",
        default="control_points.geojson",
        help="Output GeoJSON file path (default: control_points.geojson)",
    )
    parser.add_argument(
        "--description",
        default="Control Points",
        help="Description for the GeoJSON feature collection",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:4326",
        help="Coordinate reference system (default: EPSG:4326)",
    )
    parser.add_argument(
        "--sample", action="store_true", help="Create sample control points for testing"
    )
    parser.add_argument("--validate", help="Validate control points JSON file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.validate:
            # Validate control points file
            logger.info(f"Validating control points file: {args.validate}")

            if not os.path.exists(args.validate):
                print(f"❌ Error: File not found: {args.validate}")
                return 1

            with open(args.validate, "r") as f:
                control_points = json.load(f)

            validate_control_points(control_points)
            print(
                f"✅ Control points validation passed: {len(control_points['geo'])} points"
            )

        elif args.sample:
            # Create sample control points
            logger.info("Creating sample control points")
            control_points = create_sample_control_points()

            output_path = export_control_points_to_geojson(
                control_points=control_points,
                output_path=args.output,
                description=args.description,
                crs=args.crs,
            )

            print(f"✅ Sample GeoJSON created: {output_path}")

        elif args.export:
            # Export control points from JSON file
            logger.info(f"Exporting control points from: {args.export}")

            if not os.path.exists(args.export):
                print(f"❌ Error: File not found: {args.export}")
                return 1

            with open(args.export, "r") as f:
                control_points = json.load(f)

            # Validate control points
            validate_control_points(control_points)

            # Export to GeoJSON
            output_path = export_control_points_to_geojson(
                control_points=control_points,
                output_path=args.output,
                description=args.description,
                crs=args.crs,
            )

            print(f"✅ GeoJSON exported successfully: {output_path}")
            print(f"📊 Control points: {len(control_points['geo'])}")

        else:
            # Interactive mode
            print("��️ Manual Pixel-to-GPS Calibration Tool")
            print("=" * 50)
            print("This tool helps you create control points for georeferencing.")
            print(
                "Use OpenCV or another viewer to get pixel coordinates of known features."
            )
            print("Then match them to real GPS coordinates (lon, lat).")
            print("\nExample CONTROL_POINTS format:")
            print(json.dumps(create_sample_control_points(), indent=2))
            print("\nUsage:")
            print("  --export input.json --output output.geojson")
            print("  --sample --output sample.geojson")
            print("  --validate control_points.json")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
