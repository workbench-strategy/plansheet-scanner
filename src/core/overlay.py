import argparse
import logging
import os
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import simplekml

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_control_box(control_box: Dict[str, Any]) -> bool:
    """
    Validate control_box coordinates before processing.

    Args:
        control_box: Dictionary containing geographic coordinates

    Returns:
        True if control_box is valid, False otherwise

    Raises:
        ValueError: If control_box is invalid with specific error message
    """
    if not isinstance(control_box, dict):
        raise ValueError("control_box must be a dictionary")

    if "geo" not in control_box:
        raise ValueError("control_box must contain 'geo' key")

    geo_coords = control_box["geo"]
    if not isinstance(geo_coords, list) or len(geo_coords) < 4:
        raise ValueError(
            "control_box['geo'] must be a list with at least 4 coordinate pairs"
        )

    # Check that each coordinate pair has 2 values (lon, lat)
    for i, coord in enumerate(geo_coords):
        if not isinstance(coord, (list, tuple)) or len(coord) != 2:
            raise ValueError(
                f"Coordinate {i} must be a list/tuple with 2 values (lon, lat)"
            )

        lon, lat = coord
        if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
            raise ValueError(f"Coordinate {i} values must be numeric")

        # Validate longitude range (-180 to 180)
        if not -180 <= lon <= 180:
            raise ValueError(
                f"Longitude {lon} at coordinate {i} is out of valid range (-180 to 180)"
            )

        # Validate latitude range (-90 to 90)
        if not -90 <= lat <= 90:
            raise ValueError(
                f"Latitude {lat} at coordinate {i} is out of valid range (-90 to 90)"
            )

    # Check for degenerate cases (all coordinates the same)
    coords_array = np.array(geo_coords)
    if len(coords_array) >= 3:
        # Check if first 3 points form a triangle with zero area
        area = 0.5 * abs(
            coords_array[0][0] * (coords_array[1][1] - coords_array[2][1])
            + coords_array[1][0] * (coords_array[2][1] - coords_array[0][1])
            + coords_array[2][0] * (coords_array[0][1] - coords_array[1][1])
        )
        if area < 1e-10:
            raise ValueError("Control points form a degenerate (collinear) shape")

    logger.info("Control box validation passed")
    return True


def get_kmz_size_mb(kmz_path: str) -> float:
    """
    Get the size of a KMZ file in megabytes.

    Args:
        kmz_path: Path to the KMZ file

    Returns:
        Size in megabytes
    """
    if not os.path.exists(kmz_path):
        return 0.0

    size_bytes = os.path.getsize(kmz_path)
    return size_bytes / (1024 * 1024)


def compress_images_for_kmz(
    image_paths: List[str], target_size_mb: float = 10.0
) -> List[str]:
    """
    Compress images to ensure KMZ output stays under target size.

    Args:
        image_paths: List of image file paths to compress
        target_size_mb: Target size in megabytes

    Returns:
        List of compressed image paths
    """
    import tempfile

    import cv2
    from PIL import Image

    compressed_paths = []
    total_size = 0.0

    for img_path in image_paths:
        if not os.path.exists(img_path):
            logger.warning(f"Image file not found: {img_path}")
            continue

        # Get original image size
        original_size = os.path.getsize(img_path) / (1024 * 1024)
        total_size += original_size

        # If total size exceeds target, compress the image
        if total_size > target_size_mb:
            logger.info(f"Compressing {img_path} to reduce size")

            # Read image with PIL for better compression control
            with Image.open(img_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Create temporary file for compressed image
                temp_fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(temp_fd)

                # Save with compression
                img.save(temp_path, "JPEG", quality=70, optimize=True)

                compressed_paths.append(temp_path)
                compressed_size = os.path.getsize(temp_path) / (1024 * 1024)
                total_size = total_size - original_size + compressed_size

                logger.info(
                    f"Compressed {img_path}: {original_size:.2f}MB -> {compressed_size:.2f}MB"
                )
        else:
            compressed_paths.append(img_path)

    logger.info(f"Total image size: {total_size:.2f}MB")
    return compressed_paths


def generate_kmz_overlay(
    pdf_path: str,
    pages: List[int],
    dpi: int,
    control_box: Dict[str, Any],
    output_kmz: str,
    validate_coords: bool = True,
    max_size_mb: float = 10.0,
) -> str:
    """
    Generate KMZ overlay from PDF pages with enhanced features.

    Args:
        pdf_path: Path to the PDF file
        pages: List of page numbers to process
        dpi: DPI for image generation
        control_box: Dictionary containing geographic coordinates
        output_kmz: Output KMZ file path
        validate_coords: Whether to validate control_box coordinates
        max_size_mb: Maximum allowed KMZ file size in megabytes

    Returns:
        Path to the generated KMZ file

    Raises:
        ValueError: If control_box validation fails
        FileNotFoundError: If PDF file doesn't exist
        RuntimeError: If KMZ generation fails
    """
    # Validate input parameters
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pages:
        raise ValueError("No pages specified for processing")

    if dpi <= 0:
        raise ValueError("DPI must be positive")

    if max_size_mb <= 0:
        raise ValueError("Maximum size must be positive")

    # Validate control box coordinates if requested
    if validate_coords:
        validate_control_box(control_box)

    logger.info(f"Starting KMZ overlay generation for {len(pages)} pages")
    logger.info(f"PDF: {pdf_path}")
    logger.info(f"DPI: {dpi}")
    logger.info(f"Output: {output_kmz}")
    logger.info(f"Max size: {max_size_mb}MB")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_kmz)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    try:
        # Open PDF document
        doc = fitz.open(pdf_path)
        logger.info(f"PDF opened successfully. Total pages: {doc.page_count}")

        # Validate page numbers
        for page_num in pages:
            if page_num < 0 or page_num >= doc.page_count:
                raise ValueError(
                    f"Page {page_num} is out of range (0-{doc.page_count-1})"
                )

        # Create KML document
        kml = simplekml.Kml()
        kml.document.name = f"PDF Overlay - {os.path.basename(pdf_path)}"
        kml.document.description = f"Generated from {pdf_path} at {dpi} DPI"

        image_paths = []

        # Process each page
        for i, page_num in enumerate(pages):
            logger.info(f"Processing page {page_num} ({i+1}/{len(pages)})")

            try:
                # Load page and create pixmap
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=dpi)

                # Generate image filename
                image_path = f"page_{page_num:03d}.png"
                pix.save(image_path)
                image_paths.append(image_path)

                # Create ground overlay
                overlay = kml.newgroundoverlay(name=f"Sheet {page_num}")
                overlay.description = (
                    f"Page {page_num} from {os.path.basename(pdf_path)}"
                )
                overlay.icon.href = image_path

                # Set geographic bounds
                geo_coords = control_box["geo"]
                overlay.latlonbox.north = geo_coords[0][1]  # Top latitude
                overlay.latlonbox.south = geo_coords[2][1]  # Bottom latitude
                overlay.latlonbox.east = geo_coords[1][0]  # Right longitude
                overlay.latlonbox.west = geo_coords[0][0]  # Left longitude
                overlay.latlonbox.rotation = 0

                logger.info(f"Page {page_num} processed successfully")

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                raise RuntimeError(f"Failed to process page {page_num}: {e}")

        # Compress images if needed
        logger.info("Checking image sizes for compression")
        compressed_paths = compress_images_for_kmz(image_paths, max_size_mb)

        # Update KML with compressed image paths
        for i, (overlay, original_path) in enumerate(zip(kml.features, image_paths)):
            if original_path in compressed_paths:
                compressed_path = compressed_paths[
                    compressed_paths.index(original_path)
                ]
                overlay.icon.href = os.path.basename(compressed_path)

        # Save KMZ file
        logger.info(f"Saving KMZ file: {output_kmz}")
        kml.savekmz(output_kmz)

        # Check final file size
        final_size = get_kmz_size_mb(output_kmz)
        logger.info(f"KMZ file size: {final_size:.2f}MB")

        if final_size > max_size_mb:
            logger.warning(
                f"KMZ file size ({final_size:.2f}MB) exceeds target ({max_size_mb}MB)"
            )
        else:
            logger.info(f"KMZ file size is within target limit")

        # Clean up temporary image files
        for img_path in image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
                logger.debug(f"Cleaned up temporary file: {img_path}")

        doc.close()
        logger.info(f"✅ KMZ overlay generation completed: {output_kmz}")
        return output_kmz

    except Exception as e:
        logger.error(f"KMZ generation failed: {e}")
        # Clean up any partial files
        if os.path.exists(output_kmz):
            os.remove(output_kmz)
        raise RuntimeError(f"KMZ generation failed: {e}")


def main():
    """Main CLI function for KMZ overlay generation."""
    parser = argparse.ArgumentParser(
        description="Generate KMZ overlay from PDF pages with multiple DPI options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python overlay.py input.pdf --pages 0 1 2 --dpi 300 --output output.kmz
  python overlay.py input.pdf --pages 0 --dpi 150 --output low_res.kmz --max-size 5
  python overlay.py input.pdf --all-pages --dpi 600 --output high_res.kmz --no-validate
        """,
    )

    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--pages", type=int, nargs="+", help="Page numbers to process (0-based)"
    )
    parser.add_argument(
        "--all-pages", action="store_true", help="Process all pages in the PDF"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for image generation (default: 300)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.kmz",
        help="Output KMZ file path (default: output.kmz)",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=10.0,
        help="Maximum KMZ file size in MB (default: 10.0)",
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip coordinate validation"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.pages and not args.all_pages:
        parser.error("Must specify either --pages or --all-pages")

    if args.dpi <= 0:
        parser.error("DPI must be positive")

    if args.max_size <= 0:
        parser.error("Maximum size must be positive")

    try:
        # Open PDF to get page count
        doc = fitz.open(args.pdf_path)
        total_pages = doc.page_count
        doc.close()

        # Determine pages to process
        if args.all_pages:
            pages = list(range(total_pages))
        else:
            pages = args.pages

        # Example control box (in real usage, this would come from user input or file)
        # This is a placeholder - in practice, users would provide their own coordinates
        control_box = {
            "geo": [
                [-122.4194, 37.7749],  # Top-left (lon, lat)
                [-122.4000, 37.7749],  # Top-right
                [-122.4000, 37.7600],  # Bottom-right
                [-122.4194, 37.7600],  # Bottom-left
            ]
        }

        # Generate KMZ overlay
        output_path = generate_kmz_overlay(
            pdf_path=args.pdf_path,
            pages=pages,
            dpi=args.dpi,
            control_box=control_box,
            output_kmz=args.output,
            validate_coords=not args.no_validate,
            max_size_mb=args.max_size,
        )

        print(f"\n✅ KMZ overlay generated successfully!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Size: {get_kmz_size_mb(output_path):.2f}MB")
        print(f"�� Pages processed: {len(pages)}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
