#!/usr/bin/env python3
"""
Command-line interface for PlanSheet Scanner.

This module provides the main CLI entry point for the plansheet scanner tool.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..core.adaptive_reviewer import AdaptiveReviewer
from ..core.code_companion import CodeCompanion
from ..core.foundation_trainer import FoundationTrainer
from ..core.interdisciplinary_reviewer import InterdisciplinaryReviewer
from ..core.traffic_plan_reviewer import TrafficPlanReviewer


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("plansheet_scanner.log"),
        ],
    )


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="ML-powered plansheet scanner for engineering drawings and traffic plans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a single plansheet
  plansheet-scanner scan path/to/plansheet.pdf

  # Batch process multiple plansheets
  plansheet-scanner batch path/to/plansheets/ --output results/

  # Train a new model
  plansheet-scanner train --data path/to/training/data --model output/model.pkl

  # Review traffic plans
  plansheet-scanner review-traffic path/to/traffic/plans/ --output analysis.json
        """,
    )

    # Global options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a single plansheet")
    scan_parser.add_argument("input", type=Path, help="Input plansheet file")
    scan_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    scan_parser.add_argument(
        "--format",
        choices=["json", "csv", "xml"],
        default="json",
        help="Output format (default: json)",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Batch process multiple plansheets"
    )
    batch_parser.add_argument("input_dir", type=Path, help="Input directory")
    batch_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    batch_parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes (default: 4)"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--data", type=Path, required=True, help="Training data directory"
    )
    train_parser.add_argument(
        "--model", type=Path, required=True, help="Output model file"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    # Review traffic command
    review_parser = subparsers.add_parser("review-traffic", help="Review traffic plans")
    review_parser.add_argument(
        "input", type=Path, help="Input traffic plan file or directory"
    )
    review_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("traffic_review.json"),
        help="Output file (default: traffic_review.json)",
    )

    return parser


def scan_plansheet(input_path: Path, output_dir: Path, output_format: str) -> None:
    """Scan a single plansheet."""
    logging.info(f"Scanning plansheet: {input_path}")

    # Initialize the adaptive reviewer
    reviewer = AdaptiveReviewer()

    # Process the plansheet
    result = reviewer.process_plansheet(input_path)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_path.stem}_analysis.{output_format}"

    if output_format == "json":
        import json

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
    elif output_format == "csv":
        import pandas as pd

        df = pd.DataFrame([result])
        df.to_csv(output_file, index=False)

    logging.info(f"Results saved to: {output_file}")


def batch_process(input_dir: Path, output_dir: Path, workers: int) -> None:
    """Batch process multiple plansheets."""
    logging.info(f"Batch processing directory: {input_dir}")

    # Find all plansheet files
    plansheet_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.png"))

    if not plansheet_files:
        logging.warning(f"No plansheet files found in {input_dir}")
        return

    logging.info(f"Found {len(plansheet_files)} plansheet files")

    # Initialize the traffic plan reviewer for batch processing
    reviewer = TrafficPlanReviewer()

    # Process files
    results = []
    for i, file_path in enumerate(plansheet_files, 1):
        logging.info(f"Processing {i}/{len(plansheet_files)}: {file_path.name}")
        try:
            result = reviewer.process_plansheet(file_path)
            result["file"] = file_path.name
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

    # Save batch results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "batch_results.json"

    import json

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Batch results saved to: {output_file}")


def train_model(data_dir: Path, model_path: Path, epochs: int) -> None:
    """Train a new model."""
    logging.info(f"Training model with data from: {data_dir}")

    # Initialize the foundation trainer
    trainer = FoundationTrainer()

    # Train the model
    trainer.train(data_dir, epochs=epochs)

    # Save the model
    trainer.save_model(model_path)

    logging.info(f"Model saved to: {model_path}")


def review_traffic_plans(input_path: Path, output_path: Path) -> None:
    """Review traffic plans."""
    logging.info(f"Reviewing traffic plans: {input_path}")

    # Initialize the traffic plan reviewer
    reviewer = TrafficPlanReviewer()

    if input_path.is_file():
        # Single file
        result = reviewer.review_plansheet(input_path)
        results = [result]
    else:
        # Directory
        results = reviewer.batch_review(input_path)

    # Save results
    import json

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Traffic review results saved to: {output_path}")


def main(args: Optional[list] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Set up logging
    setup_logging(parsed_args.verbose)

    try:
        if parsed_args.command == "scan":
            scan_plansheet(parsed_args.input, parsed_args.output, parsed_args.format)
        elif parsed_args.command == "batch":
            batch_process(
                parsed_args.input_dir, parsed_args.output, parsed_args.workers
            )
        elif parsed_args.command == "train":
            train_model(parsed_args.data, parsed_args.model, parsed_args.epochs)
        elif parsed_args.command == "review-traffic":
            review_traffic_plans(parsed_args.input, parsed_args.output)
        else:
            parser.print_help()
            return 1

        return 0

    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logging.error(f"Error: {e}")
        if parsed_args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
