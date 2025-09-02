import os

import cv2
import fitz  # PyMuPDF
import numpy as np


def extract_symbols_from_legend(pdf_path, page_number, output_dir="templates/existing"):
    """
    Extracts symbols from the legend of a PDF page using OpenCV rectangle selection.
    Saves each selected ROI as an image in the output_dir.
    Returns a list of file paths to the saved images.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    if page_number < 0 or page_number >= doc.page_count:
        print(
            f"Error: Page number {page_number} is out of range for PDF with {doc.page_count} pages."
        )
        return []

    pix = doc.load_page(page_number).get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    clone = img.copy()
    ref_pt = []
    cropping = False
    roi_count = 0
    saved_files = []

    def click_and_crop(event, x, y, flags, param):
        nonlocal cropping, roi_count
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_pt.clear()
            ref_pt.append((x, y))
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            ref_pt.append((x, y))
            cropping = False
            cv2.rectangle(img, ref_pt[0], ref_pt[1], (0, 255, 0), 2)
            roi = clone[ref_pt[0][1] : ref_pt[1][1], ref_pt[0][0] : ref_pt[1][0]]
            if roi.size > 0:
                filename = os.path.join(output_dir, f"legend_symbol_{roi_count}.png")
                cv2.imwrite(filename, roi)
                saved_files.append(filename)
                roi_count += 1
                print(f"Saved: {filename}")
            cv2.imshow("Legend Sheet", img)

    cv2.namedWindow("Legend Sheet")
    cv2.setMouseCallback("Legend Sheet", click_and_crop)
    print(
        "🖱️ Select each symbol from the legend. Drag a rectangle, release mouse. Press ESC when done."
    )
    while True:
        cv2.imshow("Legend Sheet", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()
    return saved_files


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract legend symbols from a PDF page."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--page", type=int, default=0, help="Page number (0-based index)"
    )
    parser.add_argument(
        "--output_dir",
        default="templates/existing",
        help="Directory to save extracted symbols",
    )
    args = parser.parse_args()
    saved = extract_symbols_from_legend(args.pdf_path, args.page, args.output_dir)
    print(f"\nExtracted {len(saved)} symbols. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
