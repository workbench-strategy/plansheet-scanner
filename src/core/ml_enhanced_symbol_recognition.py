"""
ML-Enhanced Symbol Recognition System

This module provides intelligent symbol recognition capabilities that can be integrated
across all agents in the plansheet scanner system. It uses computer vision and ML
techniques to detect, classify, and validate symbols with confidence scoring.

Author: Plansheet Scanner Team
Date: 2024
"""

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SymbolDetection:
    """Represents a detected symbol with metadata."""

    symbol_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    area: float
    color: Tuple[int, int, int]
    features: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolClassification:
    """Represents a symbol classification result."""

    symbol_class: str
    confidence: float
    alternative_classes: List[Tuple[str, float]] = field(default_factory=list)
    features_used: List[str] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class MLSymbolRecognitionResult:
    """Complete result from ML symbol recognition."""

    detections: List[SymbolDetection]
    classifications: List[SymbolClassification]
    total_symbols: int
    processing_time: float
    confidence_threshold: float
    model_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SymbolFeatureExtractor:
    """Extracts features from detected symbols for classification."""

    def __init__(self):
        self.feature_names = [
            "area",
            "perimeter",
            "aspect_ratio",
            "circularity",
            "solidity",
            "extent",
            "color_mean_r",
            "color_mean_g",
            "color_mean_b",
            "color_std_r",
            "color_std_g",
            "color_std_b",
            "edge_density",
            "corner_count",
            "line_count",
            "texture_variance",
        ]

    def extract_features(self, image: np.ndarray, mask: np.ndarray) -> List[float]:
        """
        Extract comprehensive features from a symbol image.

        Args:
            image: RGB image array
            mask: Binary mask of the symbol

        Returns:
            List of feature values
        """
        features = []

        # Basic geometric features
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [0.0] * len(self.feature_names)

        contour = max(contours, key=cv2.contourArea)

        # Area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features.extend([area, perimeter])

        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        features.append(aspect_ratio)

        # Circularity
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        features.append(circularity)

        # Solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        features.append(solidity)

        # Extent (area / bounding rectangle area)
        extent = area / (w * h) if w * h > 0 else 0
        features.append(extent)

        # Color features
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        color_mean = cv2.mean(masked_image, mask=mask)[:3]
        color_std = np.std(masked_image.reshape(-1, 3), axis=0)
        features.extend(color_mean)
        features.extend(color_std)

        # Edge density
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)

        # Corner count
        corners = cv2.goodFeaturesToTrack(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
        )
        corner_count = len(corners) if corners is not None else 0
        features.append(corner_count)

        # Line count using Hough transform
        lines = cv2.HoughLinesP(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=20,
            maxLineGap=10,
        )
        line_count = len(lines) if lines is not None else 0
        features.append(line_count)

        # Texture variance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        texture_variance = np.var(gray)
        features.append(texture_variance)

        return features


class SymbolCNN(nn.Module):
    """Convolutional Neural Network for symbol classification."""

    def __init__(self, num_classes: int, input_channels: int = 3):
        super(SymbolCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class SymbolDataset(Dataset):
    """Dataset for symbol training data."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Load samples
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)

                for img_path in class_dir.glob("*.png"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class MLSymbolRecognizer:
    """Main ML-enhanced symbol recognition system."""

    def __init__(
        self, model_path: Optional[str] = None, confidence_threshold: float = 0.7
    ):
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = SymbolFeatureExtractor()
        self.model = None
        self.class_names = []
        self.model_version = "1.0.0"

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No pre-trained model found. Initialize with default settings.")

    def train_model(
        self, data_dir: str, output_path: str, epochs: int = 50, batch_size: int = 32
    ):
        """
        Train the symbol recognition model.

        Args:
            data_dir: Directory containing training data organized by class
            output_path: Path to save the trained model
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        logger.info(f"Training symbol recognition model with data from {data_dir}")

        # Load dataset
        dataset = SymbolDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        self.class_names = list(dataset.class_to_idx.keys())
        self.model = SymbolCNN(num_classes=len(self.class_names))

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            accuracy = 100.0 * correct / total
            logger.info(
                f"Epoch {epoch+1}/{epochs}: Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%"
            )

        # Save model
        self.save_model(output_path)
        logger.info(f"Model saved to {output_path}")

    def save_model(self, model_path: str):
        """Save the trained model and metadata."""
        model_data = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "class_names": self.class_names,
            "model_version": self.model_version,
            "feature_names": self.feature_extractor.feature_names,
        }
        torch.save(model_data, model_path)

    def load_model(self, model_path: str):
        """Load a trained model and metadata."""
        model_data = torch.load(model_path, map_location="cpu")
        self.class_names = model_data["class_names"]
        self.model_version = model_data.get("model_version", "1.0.0")

        if model_data["model_state_dict"]:
            self.model = SymbolCNN(num_classes=len(self.class_names))
            self.model.load_state_dict(model_data["model_state_dict"])
            self.model.eval()

        logger.info(
            f"Loaded model with {len(self.class_names)} classes: {self.class_names}"
        )

    def detect_symbols(self, image: np.ndarray) -> List[SymbolDetection]:
        """
        Detect potential symbols in an image using computer vision techniques.

        Args:
            image: Input image array

        Returns:
            List of detected symbols
        """
        detections = []

        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Multiple detection methods
        methods = [
            self._detect_by_contours,
            self._detect_by_color_clustering,
            self._detect_by_edge_detection,
        ]

        for method in methods:
            try:
                method_detections = method(image, gray)
                detections.extend(method_detections)
            except Exception as e:
                logger.warning(f"Detection method failed: {e}")

        # Remove duplicates and filter by confidence
        detections = self._filter_detections(detections)

        return detections

    def _detect_by_contours(
        self, image: np.ndarray, gray: np.ndarray
    ) -> List[SymbolDetection]:
        """Detect symbols using contour analysis."""
        detections = []

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > 10000:  # Filter by size
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Calculate confidence based on shape properties
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

            # Higher confidence for more regular shapes
            confidence = min(0.9, 0.3 + circularity * 0.6)

            # Skip if confidence is too low (likely noise)
            if confidence < 0.4:
                continue

            # Extract color information
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            color_mean = cv2.mean(image, mask=mask)[:3]

            detection = SymbolDetection(
                symbol_type="unknown",
                confidence=confidence,
                bbox=(x, y, w, h),
                center=(x + w // 2, y + h // 2),
                area=area,
                color=tuple(map(int, color_mean)),
            )
            detections.append(detection)

        return detections

    def _detect_by_color_clustering(
        self, image: np.ndarray, gray: np.ndarray
    ) -> List[SymbolDetection]:
        """Detect symbols using color clustering."""
        detections = []

        # Reshape image for clustering
        pixels = image.reshape(-1, 3)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        labels = kmeans.fit_predict(pixels)

        # Create masks for each cluster
        for i in range(kmeans.n_clusters):
            mask = (labels == i).reshape(image.shape[:2])
            mask = mask.astype(np.uint8) * 255

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200 or area > 15000:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                confidence = 0.6  # Medium confidence for color-based detection

                detection = SymbolDetection(
                    symbol_type="unknown",
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    center=(x + w // 2, y + h // 2),
                    area=area,
                    color=tuple(map(int, kmeans.cluster_centers_[i])),
                )
                detections.append(detection)

        return detections

    def _detect_by_edge_detection(
        self, image: np.ndarray, gray: np.ndarray
    ) -> List[SymbolDetection]:
        """Detect symbols using edge detection."""
        detections = []

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in edges
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 150 or area > 12000:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Calculate edge density
            roi_edges = edges[y : y + h, x : x + w]
            edge_density = np.sum(roi_edges > 0) / (w * h)

            # Higher confidence for higher edge density
            confidence = min(0.8, 0.2 + edge_density * 0.6)

            # Extract color
            roi = image[y : y + h, x : x + w]
            color_mean = np.mean(roi, axis=(0, 1))

            detection = SymbolDetection(
                symbol_type="unknown",
                confidence=confidence,
                bbox=(x, y, w, h),
                center=(x + w // 2, y + h // 2),
                area=area,
                color=tuple(map(int, color_mean)),
            )
            detections.append(detection)

        return detections

    def _filter_detections(
        self, detections: List[SymbolDetection]
    ) -> List[SymbolDetection]:
        """Filter and deduplicate detections."""
        if not detections:
            return []

        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)

        # Remove overlapping detections
        filtered = []
        for detection in detections:
            if detection.confidence < self.confidence_threshold:
                continue

            # Check for overlap with existing detections
            overlap = False
            for existing in filtered:
                if (
                    self._calculate_overlap(detection.bbox, existing.bbox) > 0.3
                ):  # Lower threshold for better filtering
                    overlap = True
                    break

            if not overlap:
                filtered.append(detection)

        return filtered

    def _calculate_overlap(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0

    def classify_symbols(
        self, image: np.ndarray, detections: List[SymbolDetection]
    ) -> List[SymbolClassification]:
        """
        Classify detected symbols using the trained model.

        Args:
            image: Input image
            detections: List of detected symbols

        Returns:
            List of classification results
        """
        classifications = []

        if not self.model or not detections:
            return classifications

        for detection in detections:
            start_time = time.time()

            # Extract ROI
            x, y, w, h = detection.bbox
            roi = image[y : y + h, x : x + w]

            # Resize to model input size
            roi_resized = cv2.resize(roi, (64, 64))
            roi_tensor = (
                torch.FloatTensor(roi_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
            )

            # Get prediction
            with torch.no_grad():
                output = self.model(roi_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                # Get top 3 predictions
                top3_prob, top3_indices = torch.topk(probabilities, 3)

                alternative_classes = []
                for i in range(3):
                    if i < len(top3_indices[0]):
                        class_idx = top3_indices[0][i].item()
                        prob = top3_prob[0][i].item()
                        if class_idx < len(self.class_names):
                            alternative_classes.append(
                                (self.class_names[class_idx], prob)
                            )

            processing_time = time.time() - start_time

            classification = SymbolClassification(
                symbol_class=self.class_names[predicted.item()]
                if predicted.item() < len(self.class_names)
                else "unknown",
                confidence=confidence.item(),
                alternative_classes=alternative_classes,
                features_used=self.feature_extractor.feature_names,
                processing_time=processing_time,
            )
            classifications.append(classification)

        return classifications

    def recognize_symbols(self, image: np.ndarray) -> MLSymbolRecognitionResult:
        """
        Complete symbol recognition pipeline.

        Args:
            image: Input image array

        Returns:
            Complete recognition result
        """
        start_time = time.time()

        # Detect symbols
        detections = self.detect_symbols(image)

        # Classify symbols
        classifications = self.classify_symbols(image, detections)

        # Update detection types with classifications
        for detection, classification in zip(detections, classifications):
            detection.symbol_type = classification.symbol_class
            detection.confidence = classification.confidence
            detection.metadata = {
                "alternative_classes": classification.alternative_classes,
                "processing_time": classification.processing_time,
            }

        processing_time = time.time() - start_time

        result = MLSymbolRecognitionResult(
            detections=detections,
            classifications=classifications,
            total_symbols=len(detections),
            processing_time=processing_time,
            confidence_threshold=self.confidence_threshold,
            model_version=self.model_version,
            metadata={
                "image_shape": image.shape,
                "detection_methods": ["contours", "color_clustering", "edge_detection"],
            },
        )

        return result

    def generate_report(self, result: MLSymbolRecognitionResult, output_path: str):
        """Generate a detailed report of symbol recognition results."""
        report = {
            "summary": {
                "total_symbols": result.total_symbols,
                "processing_time": result.processing_time,
                "model_version": result.model_version,
                "confidence_threshold": result.confidence_threshold,
            },
            "detections": [
                {
                    "symbol_type": d.symbol_type,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "center": d.center,
                    "area": d.area,
                    "color": d.color,
                    "metadata": d.metadata,
                }
                for d in result.detections
            ],
            "statistics": {
                "symbol_types": {},
                "confidence_distribution": {},
                "size_distribution": {},
            },
        }

        # Calculate statistics
        for detection in result.detections:
            # Symbol type distribution
            symbol_type = detection.symbol_type
            report["statistics"]["symbol_types"][symbol_type] = (
                report["statistics"]["symbol_types"].get(symbol_type, 0) + 1
            )

            # Confidence distribution
            conf_range = f"{int(detection.confidence * 10) * 10}-{(int(detection.confidence * 10) + 1) * 10}%"
            report["statistics"]["confidence_distribution"][conf_range] = (
                report["statistics"]["confidence_distribution"].get(conf_range, 0) + 1
            )

            # Size distribution
            size_range = f"{int(detection.area / 100) * 100}-{(int(detection.area / 100) + 1) * 100}"
            report["statistics"]["size_distribution"][size_range] = (
                report["statistics"]["size_distribution"].get(size_range, 0) + 1
            )

        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Symbol recognition report saved to {output_path}")
        return report


def main():
    """Main function for CLI usage."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="ML-Enhanced Symbol Recognition")
    parser.add_argument(
        "--mode", choices=["train", "recognize"], required=True, help="Operation mode"
    )
    parser.add_argument(
        "--input", required=True, help="Input image or training data directory"
    )
    parser.add_argument("--output", help="Output path for results or model")
    parser.add_argument("--model", help="Path to trained model")
    parser.add_argument(
        "--confidence", type=float, default=0.7, help="Confidence threshold"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")

    args = parser.parse_args()

    recognizer = MLSymbolRecognizer(
        model_path=args.model, confidence_threshold=args.confidence
    )

    if args.mode == "train":
        if not args.output:
            args.output = "symbol_model.pth"
        recognizer.train_model(args.input, args.output, args.epochs)

    elif args.mode == "recognize":
        # Load image
        image = cv2.imread(args.input)
        if image is None:
            logger.error(f"Could not load image: {args.input}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Recognize symbols
        result = recognizer.recognize_symbols(image)

        # Generate report
        if args.output:
            recognizer.generate_report(result, args.output)

        # Print summary
        print(
            f"Detected {result.total_symbols} symbols in {result.processing_time:.2f} seconds"
        )
        for detection in result.detections:
            print(f"- {detection.symbol_type}: {detection.confidence:.2f} confidence")


if __name__ == "__main__":
    main()
