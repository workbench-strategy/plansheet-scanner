import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class LineSegment:
    """Represents a line segment with start and end points."""

    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    length: float
    angle: float
    confidence: float = 1.0


@dataclass
class MatchedPair:
    """Represents a matched pair of line segments."""

    line1: LineSegment
    line2: LineSegment
    confidence: float
    distance: float
    angle_diff: float


class LineMatcher:
    """
    Core logic for detecting and matching line segments between two vector layers.
    Uses OpenCV for line detection and provides confidence scores for matches.
    """

    def __init__(
        self,
        min_line_length: float = 30.0,
        max_angle_diff: float = 15.0,
        distance_threshold: float = 50.0,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize LineMatcher with detection and matching parameters.

        Args:
            min_line_length: Minimum length for detected lines
            max_angle_diff: Maximum angle difference for matching (degrees)
            distance_threshold: Maximum distance for matching (pixels)
            confidence_threshold: Minimum confidence for valid matches
        """
        self.min_line_length = min_line_length
        self.max_angle_diff = max_angle_diff
        self.distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold

        logger.info(
            f"LineMatcher initialized with: min_length={min_line_length}, "
            f"max_angle_diff={max_angle_diff}, distance_threshold={distance_threshold}"
        )

    def detect_lines(
        self, image: np.ndarray, method: str = "hough"
    ) -> List[LineSegment]:
        """
        Detect line segments in an image using OpenCV.

        Args:
            image: Input image (grayscale or BGR)
            method: Detection method ("hough", "contour", or "combined")

        Returns:
            List of detected LineSegment objects
        """
        if image is None or image.size == 0:
            logger.warning("Empty or None image provided")
            return []

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        lines = []

        if method == "hough" or method == "combined":
            lines.extend(self._detect_hough_lines(gray))

        if method == "contour" or method == "combined":
            lines.extend(self._detect_contour_lines(gray))

        # Remove duplicates and filter by length
        unique_lines = self._remove_duplicate_lines(lines)
        filtered_lines = [
            line for line in unique_lines if line.length >= self.min_line_length
        ]

        logger.info(
            f"Detected {len(filtered_lines)} line segments using {method} method"
        )
        return filtered_lines

    def _detect_hough_lines(self, gray_image: np.ndarray) -> List[LineSegment]:
        """Detect lines using Hough Line Transform."""
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=self.min_line_length,
            maxLineGap=10,
        )

        line_segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line properties
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                # Normalize angle to 0-180 degrees
                angle = angle % 180
                if angle < 0:
                    angle += 180

                line_segment = LineSegment(
                    start_point=(float(x1), float(y1)),
                    end_point=(float(x2), float(y2)),
                    length=float(length),
                    angle=float(angle),
                )
                line_segments.append(line_segment)

        return line_segments

    def _detect_contour_lines(self, gray_image: np.ndarray) -> List[LineSegment]:
        """Detect lines using contour analysis."""
        # Apply threshold
        _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        line_segments = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Extract line segments from polygon
            for i in range(len(approx)):
                pt1 = approx[i][0]
                pt2 = approx[(i + 1) % len(approx)][0]

                x1, y1 = pt1
                x2, y2 = pt2

                # Calculate line properties
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                # Normalize angle
                angle = angle % 180
                if angle < 0:
                    angle += 180

                if length >= self.min_line_length:
                    line_segment = LineSegment(
                        start_point=(float(x1), float(y1)),
                        end_point=(float(x2), float(y2)),
                        length=float(length),
                        angle=float(angle),
                    )
                    line_segments.append(line_segment)

        return line_segments

    def _remove_duplicate_lines(self, lines: List[LineSegment]) -> List[LineSegment]:
        """Remove duplicate or very similar line segments."""
        if not lines:
            return []

        unique_lines = []
        for line in lines:
            is_duplicate = False

            for existing_line in unique_lines:
                # Check if lines are very similar
                if self._are_lines_similar(line, existing_line):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_lines.append(line)

        return unique_lines

    def _are_lines_similar(
        self, line1: LineSegment, line2: LineSegment, tolerance: float = 5.0
    ) -> bool:
        """Check if two line segments are very similar."""
        # Check angle similarity
        angle_diff = abs(line1.angle - line2.angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        if angle_diff > tolerance:
            return False

        # Check length similarity
        length_diff = abs(line1.length - line2.length)
        if length_diff > tolerance:
            return False

        # Check spatial proximity
        center1 = (
            (line1.start_point[0] + line1.end_point[0]) / 2,
            (line1.start_point[1] + line1.end_point[1]) / 2,
        )
        center2 = (
            (line2.start_point[0] + line2.end_point[0]) / 2,
            (line2.start_point[1] + line2.end_point[1]) / 2,
        )

        distance = np.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        )
        return distance <= tolerance

    def match_lines(
        self, lines1: List[LineSegment], lines2: List[LineSegment]
    ) -> List[MatchedPair]:
        """
        Match line segments between two sets using geometric similarity.

        Args:
            lines1: First set of line segments
            lines2: Second set of line segments

        Returns:
            List of MatchedPair objects with confidence scores
        """
        if not lines1 or not lines2:
            logger.warning("Empty line sets provided for matching")
            return []

        matches = []

        for line1 in lines1:
            best_match = None
            best_confidence = 0.0

            for line2 in lines2:
                confidence, distance, angle_diff = self._calculate_match_confidence(
                    line1, line2
                )

                if (
                    confidence > best_confidence
                    and confidence >= self.confidence_threshold
                ):
                    best_confidence = confidence
                    best_match = MatchedPair(
                        line1=line1,
                        line2=line2,
                        confidence=confidence,
                        distance=distance,
                        angle_diff=angle_diff,
                    )

            if best_match:
                matches.append(best_match)

        # Sort matches by confidence (highest first)
        matches.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(
            f"Found {len(matches)} line matches between {len(lines1)} and {len(lines2)} lines"
        )
        return matches

    def _calculate_match_confidence(
        self, line1: LineSegment, line2: LineSegment
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence score for matching two line segments.

        Returns:
            Tuple of (confidence, distance, angle_difference)
        """
        # Calculate angle difference
        angle_diff = abs(line1.angle - line2.angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # Calculate length similarity
        length_ratio = min(line1.length, line2.length) / max(line1.length, line2.length)

        # Calculate spatial distance between line centers
        center1 = (
            (line1.start_point[0] + line1.end_point[0]) / 2,
            (line1.start_point[1] + line1.end_point[1]) / 2,
        )
        center2 = (
            (line2.start_point[0] + line2.end_point[0]) / 2,
            (line2.start_point[1] + line2.end_point[1]) / 2,
        )

        distance = np.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        )

        # Calculate confidence based on multiple factors
        angle_score = max(0, 1 - (angle_diff / self.max_angle_diff))
        length_score = length_ratio
        distance_score = max(0, 1 - (distance / self.distance_threshold))

        # Weighted combination
        confidence = 0.4 * angle_score + 0.3 * length_score + 0.3 * distance_score

        return confidence, distance, angle_diff

    def detect_and_match(
        self, image1: np.ndarray, image2: np.ndarray, method: str = "combined"
    ) -> List[MatchedPair]:
        """
        Detect lines in both images and match them.

        Args:
            image1: First image
            image2: Second image
            method: Line detection method

        Returns:
            List of matched line pairs
        """
        logger.info("Starting line detection and matching")

        # Detect lines in both images
        lines1 = self.detect_lines(image1, method)
        lines2 = self.detect_lines(image2, method)

        # Match lines
        matches = self.match_lines(lines1, lines2)

        return matches

    def visualize_matches(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        matches: List[MatchedPair],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize line matches between two images.

        Args:
            image1: First image
            image2: Second image
            matches: List of matched pairs
            output_path: Optional path to save visualization

        Returns:
            Visualization image
        """
        # Create side-by-side visualization
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        # Ensure both images have same height
        max_h = max(h1, h2)
        vis_width = w1 + w2 + 50  # Add gap between images

        # Create visualization canvas
        if len(image1.shape) == 3:
            vis_image = np.zeros((max_h, vis_width, 3), dtype=np.uint8)
        else:
            vis_image = np.zeros((max_h, vis_width), dtype=np.uint8)

        # Copy images to visualization
        vis_image[:h1, :w1] = image1
        vis_image[:h2, w1 + 50 : w1 + 50 + w2] = image2

        # Draw lines and matches
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for i, match in enumerate(
            matches[:10]
        ):  # Limit to first 10 matches for clarity
            color = colors[i % len(colors)]

            # Draw line in first image
            cv2.line(
                vis_image,
                (int(match.line1.start_point[0]), int(match.line1.start_point[1])),
                (int(match.line1.end_point[0]), int(match.line1.end_point[1])),
                color,
                2,
            )

            # Draw line in second image
            cv2.line(
                vis_image,
                (
                    int(match.line2.start_point[0]) + w1 + 50,
                    int(match.line2.start_point[1]),
                ),
                (
                    int(match.line2.end_point[0]) + w1 + 50,
                    int(match.line2.end_point[1]),
                ),
                color,
                2,
            )

            # Draw connection line
            center1 = (
                (match.line1.start_point[0] + match.line1.end_point[0]) / 2,
                (match.line1.start_point[1] + match.line1.end_point[1]) / 2,
            )
            center2 = (
                (match.line2.start_point[0] + match.line2.end_point[0]) / 2 + w1 + 50,
                (match.line2.start_point[1] + match.line2.end_point[1]) / 2,
            )

            cv2.line(
                vis_image,
                (int(center1[0]), int(center1[1])),
                (int(center2[0]), int(center2[1])),
                color,
                1,
            )

        # Add text information
        cv2.putText(
            vis_image,
            f"Matches: {len(matches)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Visualization saved to: {output_path}")

        return vis_image


def main():
    """Main CLI function for line matching."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect and match line segments between two images using OpenCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python line_matcher.py image1.jpg image2.jpg --output matches.png
  python line_matcher.py image1.jpg image2.jpg --method hough --min-length 50
  python line_matcher.py image1.jpg image2.jpg --confidence 0.8 --visualize
        """,
    )

    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument("--output", "-o", help="Output path for visualization")
    parser.add_argument(
        "--method",
        choices=["hough", "contour", "combined"],
        default="combined",
        help="Line detection method",
    )
    parser.add_argument(
        "--min-length", type=float, default=30.0, help="Minimum line length"
    )
    parser.add_argument(
        "--max-angle-diff",
        type=float,
        default=15.0,
        help="Maximum angle difference for matching (degrees)",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=50.0,
        help="Maximum distance for matching (pixels)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Minimum confidence for valid matches",
    )
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load images
        image1 = cv2.imread(args.image1)
        image2 = cv2.imread(args.image2)

        if image1 is None:
            raise FileNotFoundError(f"Could not load image: {args.image1}")
        if image2 is None:
            raise FileNotFoundError(f"Could not load image: {args.image2}")

        # Create line matcher
        matcher = LineMatcher(
            min_line_length=args.min_length,
            max_angle_diff=args.max_angle_diff,
            distance_threshold=args.distance_threshold,
            confidence_threshold=args.confidence,
        )

        # Detect and match lines
        matches = matcher.detect_and_match(image1, image2, args.method)

        # Print results
        print(f"✅ Found {len(matches)} line matches")
        for i, match in enumerate(matches[:5]):  # Show top 5 matches
            print(
                f"Match {i+1}: Confidence={match.confidence:.3f}, "
                f"Distance={match.distance:.1f}, Angle_diff={match.angle_diff:.1f}"
            )

        # Create visualization
        if args.visualize or args.output:
            vis_image = matcher.visualize_matches(image1, image2, matches, args.output)

            if args.visualize:
                cv2.imshow("Line Matches", vis_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
