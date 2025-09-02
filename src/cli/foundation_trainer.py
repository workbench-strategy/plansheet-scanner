#!/usr/bin/env python3
"""
Foundation Trainer CLI
Manages foundation model training using as-builts and past reviewed plans.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.foundation_trainer import FoundationTrainer


def add_as_built_command(args):
    """Handle the add-as-built command."""
    trainer = FoundationTrainer(args.data_dir, args.model_dir)

    # Load as-built image
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file {args.image} not found.")
        return 1

    as_built_image = cv2.imread(args.image)
    if as_built_image is None:
        print(f"❌ Error: Could not load image {args.image}")
        return 1

    # Load final elements if provided
    final_elements = []
    if args.elements:
        with open(args.elements, "r") as f:
            final_elements = json.load(f)

    # Load project info if provided
    project_info = {}
    if args.project_info:
        with open(args.project_info, "r") as f:
            project_info = json.load(f)

    # Add as-built data
    trainer.add_as_built_data(
        plan_id=args.plan_id,
        plan_type=args.plan_type,
        as_built_image=as_built_image,
        final_elements=final_elements,
        construction_notes=args.notes,
        approval_date=datetime.now(),
        project_info=project_info,
    )

    print(f"✅ Added as-built data for plan {args.plan_id}")
    return 0


def add_milestone_command(args):
    """Handle the add-milestone command."""
    trainer = FoundationTrainer(args.data_dir, args.model_dir)

    # Load approved and rejected elements
    approved_elements = []
    rejected_elements = []

    if args.approved_elements:
        with open(args.approved_elements, "r") as f:
            approved_elements = json.load(f)

    if args.rejected_elements:
        with open(args.rejected_elements, "r") as f:
            rejected_elements = json.load(f)

    # Load reviewer comments
    reviewer_comments = []
    if args.comments:
        with open(args.comments, "r") as f:
            reviewer_comments = json.load(f)

    # Add review milestone
    trainer.add_review_milestone(
        plan_id=args.plan_id,
        milestone=args.milestone,
        reviewer_comments=reviewer_comments,
        approved_elements=approved_elements,
        rejected_elements=rejected_elements,
        compliance_score=args.compliance_score,
        review_date=datetime.now(),
    )

    print(f"✅ Added review milestone for plan {args.plan_id} ({args.milestone})")
    return 0


def train_models_command(args):
    """Handle the train-models command."""
    trainer = FoundationTrainer(args.data_dir, args.model_dir)

    print("🔍 Extracting training features...")
    trainer.extract_training_features()

    print("🤖 Training foundation models...")
    trainer.train_foundation_models(min_examples=args.min_examples)

    # Show training statistics
    stats = trainer.get_training_statistics()
    print(f"\n📊 Training Statistics:")
    print(f"   Total as-built records: {stats['total_as_built']}")
    print(f"   Total milestone records: {stats['total_milestones']}")
    print(f"   Total training examples: {stats['total_training_examples']}")
    print(f"   Models trained: {stats['models_trained']}")

    if stats["as_built_by_type"]:
        print(f"   As-built by type:")
        for plan_type, count in stats["as_built_by_type"].items():
            print(f"     {plan_type}: {count}")

    if stats["milestones_by_type"]:
        print(f"   Milestones by type:")
        for milestone, count in stats["milestones_by_type"].items():
            print(f"     {milestone}: {count}")

    return 0


def show_statistics_command(args):
    """Handle the show-statistics command."""
    trainer = FoundationTrainer(args.data_dir, args.model_dir)
    stats = trainer.get_training_statistics()

    print("📊 Foundation Training Statistics")
    print("=" * 50)
    print(f"Total As-Built Records: {stats['total_as_built']}")
    print(f"Total Milestone Records: {stats['total_milestones']}")
    print(f"Total Training Examples: {stats['total_training_examples']}")
    print(f"Models Trained: {stats['models_trained']}")

    if stats["as_built_by_type"]:
        print(f"\nAs-Built Records by Plan Type:")
        for plan_type, count in stats["as_built_by_type"].items():
            print(f"  {plan_type}: {count}")

    if stats["milestones_by_type"]:
        print(f"\nReview Milestones by Type:")
        for milestone, count in stats["milestones_by_type"].items():
            print(f"  {milestone}: {count}")

    # Show data timeline
    if stats["data_timeline"]:
        print(f"\nData Timeline (Last 10 entries):")
        timeline = sorted(stats["data_timeline"], key=lambda x: x["date"], reverse=True)
        for entry in timeline[:10]:
            date = entry["date"][:10]  # Just the date part
            if entry["type"] == "as_built":
                print(f"  {date}: As-built ({entry['plan_type']})")
            else:
                print(f"  {date}: Milestone ({entry['milestone']})")

    return 0


def predict_command(args):
    """Handle the predict command."""
    trainer = FoundationTrainer(args.data_dir, args.model_dir)

    # Load plan image
    if not os.path.exists(args.plan_image):
        print(f"❌ Error: Plan image {args.plan_image} not found.")
        return 1

    plan_image = cv2.imread(args.plan_image)
    if plan_image is None:
        print(f"❌ Error: Could not load plan image {args.plan_image}")
        return 1

    # Load detected elements if provided
    detected_elements = []
    if args.detected_elements:
        with open(args.detected_elements, "r") as f:
            detected_elements = json.load(f)

    # Make prediction
    prediction = trainer.predict_with_foundation_models(plan_image, detected_elements)

    if not prediction:
        print("❌ No foundation models available for prediction")
        return 1

    # Display results
    print(f"🤖 Foundation Model Prediction Results")
    print("=" * 50)
    print(f"Model Type: {prediction['model_type']}")

    if "final_prediction" in prediction:
        final_pred = prediction["final_prediction"]
        print(f"Final Prediction: {final_pred}")

    if "confidence_scores" in prediction:
        print(f"\nConfidence Scores:")
        for model, confidence in prediction["confidence_scores"].items():
            print(f"  {model}: {confidence:.3f}")

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(prediction, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")

    return 0


def export_data_command(args):
    """Handle the export-data command."""
    trainer = FoundationTrainer(args.data_dir, args.model_dir)

    # Export as-built data
    as_built_data = []
    for as_built in trainer.as_built_data:
        as_built_data.append(
            {
                "plan_id": as_built.plan_id,
                "plan_type": as_built.plan_type,
                "construction_notes": as_built.construction_notes,
                "approval_date": as_built.approval_date.isoformat(),
                "project_info": as_built.project_info,
                "final_elements": as_built.final_elements,
            }
        )

    # Export milestone data
    milestone_data = []
    for milestone in trainer.review_milestones:
        milestone_data.append(
            {
                "plan_id": milestone.plan_id,
                "milestone": milestone.milestone,
                "reviewer_comments": milestone.reviewer_comments,
                "approved_elements": milestone.approved_elements,
                "rejected_elements": milestone.rejected_elements,
                "compliance_score": milestone.compliance_score,
                "review_date": milestone.review_date.isoformat(),
            }
        )

    # Save to files
    if args.format == "json":
        if as_built_data:
            with open(f"{args.output}_as_built.json", "w") as f:
                json.dump(as_built_data, f, indent=2, default=str)
            print(
                f"✅ Exported {len(as_built_data)} as-built records to {args.output}_as_built.json"
            )

        if milestone_data:
            with open(f"{args.output}_milestones.json", "w") as f:
                json.dump(milestone_data, f, indent=2, default=str)
            print(
                f"✅ Exported {len(milestone_data)} milestone records to {args.output}_milestones.json"
            )

    elif args.format == "csv":
        import pandas as pd

        if as_built_data:
            df_as_built = pd.DataFrame(as_built_data)
            df_as_built.to_csv(f"{args.output}_as_built.csv", index=False)
            print(
                f"✅ Exported {len(as_built_data)} as-built records to {args.output}_as_built.csv"
            )

        if milestone_data:
            df_milestones = pd.DataFrame(milestone_data)
            df_milestones.to_csv(f"{args.output}_milestones.csv", index=False)
            print(
                f"✅ Exported {len(milestone_data)} milestone records to {args.output}_milestones.csv"
            )

    return 0


def batch_import_command(args):
    """Handle the batch-import command."""
    trainer = FoundationTrainer(args.data_dir, args.model_dir)

    data_dir = Path(args.data_directory)
    if not data_dir.exists():
        print(f"❌ Error: Data directory {args.data_directory} not found.")
        return 1

    imported_count = 0

    # Import as-built data
    for as_built_file in data_dir.glob("as_built_*.json"):
        try:
            with open(as_built_file, "r") as f:
                data = json.load(f)

            # Load image
            image_path = data_dir / f"{data['plan_id']}.png"
            if image_path.exists():
                as_built_image = cv2.imread(str(image_path))

                trainer.add_as_built_data(
                    plan_id=data["plan_id"],
                    plan_type=data["plan_type"],
                    as_built_image=as_built_image,
                    final_elements=data.get("final_elements", []),
                    construction_notes=data.get("construction_notes", ""),
                    approval_date=datetime.fromisoformat(data["approval_date"]),
                    project_info=data.get("project_info", {}),
                )
                imported_count += 1
        except Exception as e:
            print(f"⚠️ Error importing {as_built_file}: {e}")

    # Import milestone data
    for milestone_file in data_dir.glob("milestone_*.json"):
        try:
            with open(milestone_file, "r") as f:
                data = json.load(f)

            trainer.add_review_milestone(
                plan_id=data["plan_id"],
                milestone=data["milestone"],
                reviewer_comments=data.get("reviewer_comments", []),
                approved_elements=data.get("approved_elements", []),
                rejected_elements=data.get("rejected_elements", []),
                compliance_score=data.get("compliance_score", 0.5),
                review_date=datetime.fromisoformat(data["review_date"]),
            )
            imported_count += 1
        except Exception as e:
            print(f"⚠️ Error importing {milestone_file}: {e}")

    print(f"✅ Imported {imported_count} records from {args.data_directory}")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Foundation Trainer - Train models using as-builts and past reviewed plans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add as-built data
  python foundation_trainer.py add-as-built as_built_001 traffic_signal as_built_image.png --elements final_elements.json --project-info project.json

  # Add review milestone
  python foundation_trainer.py add-milestone plan_001 final --compliance-score 0.95 --approved-elements approved.json --comments reviewer_comments.json

  # Train foundation models
  python foundation_trainer.py train-models --min-examples 20

  # Show training statistics
  python foundation_trainer.py show-statistics

  # Make prediction with foundation models
  python foundation_trainer.py predict plan_image.png --detected-elements elements.json --output prediction.json

  # Export training data
  python foundation_trainer.py export-data --format json --output training_data

  # Batch import from directory
  python foundation_trainer.py batch-import /path/to/training/data
        """,
    )

    parser.add_argument(
        "--data-dir", default="training_data", help="Directory for training data"
    )
    parser.add_argument(
        "--model-dir", default="models", help="Directory for trained models"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add as-built command
    as_built_parser = subparsers.add_parser(
        "add-as-built", help="Add as-built data for training"
    )
    as_built_parser.add_argument("plan_id", help="Unique plan identifier")
    as_built_parser.add_argument(
        "plan_type", help="Type of plan (traffic_signal, its, mutcd_signing)"
    )
    as_built_parser.add_argument("image", help="Path to as-built image")
    as_built_parser.add_argument("--elements", help="Path to final elements JSON file")
    as_built_parser.add_argument(
        "--project-info", help="Path to project info JSON file"
    )
    as_built_parser.add_argument("--notes", default="", help="Construction notes")
    as_built_parser.set_defaults(func=add_as_built_command)

    # Add milestone command
    milestone_parser = subparsers.add_parser(
        "add-milestone", help="Add review milestone data"
    )
    milestone_parser.add_argument("plan_id", help="Unique plan identifier")
    milestone_parser.add_argument(
        "milestone",
        choices=["preliminary", "final", "construction", "as_built"],
        help="Review milestone type",
    )
    milestone_parser.add_argument(
        "--approved-elements", help="Path to approved elements JSON file"
    )
    milestone_parser.add_argument(
        "--rejected-elements", help="Path to rejected elements JSON file"
    )
    milestone_parser.add_argument(
        "--comments", help="Path to reviewer comments JSON file"
    )
    milestone_parser.add_argument(
        "--compliance-score", type=float, default=0.5, help="Compliance score (0-1)"
    )
    milestone_parser.set_defaults(func=add_milestone_command)

    # Train models command
    train_parser = subparsers.add_parser("train-models", help="Train foundation models")
    train_parser.add_argument(
        "--min-examples",
        type=int,
        default=20,
        help="Minimum examples required for training",
    )
    train_parser.set_defaults(func=train_models_command)

    # Show statistics command
    stats_parser = subparsers.add_parser(
        "show-statistics", help="Show training statistics"
    )
    stats_parser.set_defaults(func=show_statistics_command)

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Make prediction with foundation models"
    )
    predict_parser.add_argument("plan_image", help="Path to plan image")
    predict_parser.add_argument(
        "--detected-elements", help="Path to detected elements JSON file"
    )
    predict_parser.add_argument("--output", help="Output file for prediction results")
    predict_parser.set_defaults(func=predict_command)

    # Export data command
    export_parser = subparsers.add_parser("export-data", help="Export training data")
    export_parser.add_argument(
        "--format", choices=["json", "csv"], default="json", help="Export format"
    )
    export_parser.add_argument("--output", required=True, help="Output file prefix")
    export_parser.set_defaults(func=export_data_command)

    # Batch import command
    import_parser = subparsers.add_parser(
        "batch-import", help="Batch import from directory"
    )
    import_parser.add_argument(
        "data_directory", help="Directory containing training data files"
    )
    import_parser.set_defaults(func=batch_import_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
