#!/usr/bin/env python3
"""
Learning Manager CLI
Manages the adaptive learning system for traffic plan review.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.adaptive_reviewer import AdaptiveReviewer, FeedbackCollector

def collect_feedback_command(args):
    """Handle the collect-feedback command."""
    adaptive_reviewer = AdaptiveReviewer(args.model_dir)
    feedback_collector = FeedbackCollector(adaptive_reviewer)
    
    # Load original prediction if provided
    original_prediction = {}
    if args.original_prediction:
        with open(args.original_prediction, 'r') as f:
            original_prediction = json.load(f)
    
    # Load human corrections if provided
    human_corrections = {}
    if args.human_corrections:
        with open(args.human_corrections, 'r') as f:
            human_corrections = json.load(f)
    
    # Collect feedback
    feedback_collector.collect_feedback(
        plan_id=args.plan_id,
        plan_type=args.plan_type,
        reviewer_id=args.reviewer_id,
        original_prediction=original_prediction,
        human_corrections=human_corrections,
        confidence_score=args.confidence,
        review_time=args.review_time,
        notes=args.notes
    )
    
    print(f"✅ Feedback collected for plan {args.plan_id}")
    
    # Show learning statistics
    stats = feedback_collector.get_feedback_summary()
    print(f"\n📊 Learning Statistics:")
    print(f"   Total feedback: {stats['total_feedback']}")
    print(f"   Models trained: {stats['models_trained']}")
    print(f"   Average confidence: {stats['average_confidence']:.2f}")
    print(f"   Average review time: {stats['average_review_time']:.1f} seconds")

def train_models_command(args):
    """Handle the train-models command."""
    adaptive_reviewer = AdaptiveReviewer(args.model_dir)
    
    print("🤖 Training models on accumulated feedback...")
    adaptive_reviewer.train_models(min_examples=args.min_examples)
    
    # Show training results
    stats = adaptive_reviewer.get_learning_statistics()
    print(f"\n📈 Training Results:")
    print(f"   Models trained: {stats['models_trained']}")
    print(f"   Total feedback used: {stats['total_feedback']}")
    
    if stats['feedback_by_plan_type']:
        print(f"   Feedback by plan type:")
        for plan_type, count in stats['feedback_by_plan_type'].items():
            print(f"     {plan_type}: {count}")

def show_statistics_command(args):
    """Handle the show-statistics command."""
    adaptive_reviewer = AdaptiveReviewer(args.model_dir)
    stats = adaptive_reviewer.get_learning_statistics()
    
    print("📊 Learning System Statistics")
    print("=" * 50)
    print(f"Total Feedback: {stats['total_feedback']}")
    print(f"Models Trained: {stats['models_trained']}")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")
    print(f"Average Review Time: {stats['average_review_time']:.1f} seconds")
    
    if stats['feedback_by_plan_type']:
        print(f"\nFeedback by Plan Type:")
        for plan_type, count in stats['feedback_by_plan_type'].items():
            print(f"  {plan_type}: {count}")
    
    # Show model performance if available
    if stats['models_trained'] > 0:
        print(f"\n🤖 Trained Models:")
        for model_name in ['random_forest', 'gradient_boosting']:
            model_path = Path(args.model_dir) / f"{model_name}.joblib"
            if model_path.exists():
                print(f"  ✅ {model_name}")
            else:
                print(f"  ❌ {model_name} (not found)")

def export_feedback_command(args):
    """Handle the export-feedback command."""
    adaptive_reviewer = AdaptiveReviewer(args.model_dir)
    
    if not adaptive_reviewer.feedback_file.exists():
        print("❌ No feedback data found")
        return 1
    
    # Read all feedback
    feedback_data = []
    with open(adaptive_reviewer.feedback_file, 'r') as f:
        for line in f:
            if line.strip():
                feedback_data.append(json.loads(line))
    
    # Export to specified format
    if args.format == 'json':
        with open(args.output, 'w') as f:
            json.dump(feedback_data, f, indent=2, default=str)
    elif args.format == 'csv':
        import pandas as pd
        df = pd.DataFrame(feedback_data)
        df.to_csv(args.output, index=False)
    else:
        print(f"❌ Unsupported format: {args.format}")
        return 1
    
    print(f"✅ Exported {len(feedback_data)} feedback records to {args.output}")

def import_feedback_command(args):
    """Handle the import-feedback command."""
    adaptive_reviewer = AdaptiveReviewer(args.model_dir)
    
    # Read feedback from file
    if args.format == 'json':
        with open(args.input, 'r') as f:
            feedback_data = json.load(f)
    elif args.format == 'csv':
        import pandas as pd
        df = pd.read_csv(args.input)
        feedback_data = df.to_dict('records')
    else:
        print(f"❌ Unsupported format: {args.format}")
        return 1
    
    # Import feedback
    imported_count = 0
    for feedback in feedback_data:
        try:
            adaptive_reviewer.record_feedback(
                plan_id=feedback['plan_id'],
                plan_type=feedback['plan_type'],
                reviewer_id=feedback['reviewer_id'],
                original_prediction=feedback['original_prediction'],
                human_corrections=feedback['human_corrections'],
                confidence_score=feedback['confidence_score'],
                review_time=feedback['review_time'],
                notes=feedback.get('notes', '')
            )
            imported_count += 1
        except Exception as e:
            print(f"⚠️ Error importing feedback {feedback.get('plan_id', 'unknown')}: {e}")
    
    print(f"✅ Imported {imported_count} feedback records")

def reset_models_command(args):
    """Handle the reset-models command."""
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print("❌ Model directory not found")
        return 1
    
    # List files to be deleted
    files_to_delete = []
    for file in model_dir.glob("*.joblib"):
        files_to_delete.append(file)
    
    if args.feedback:
        feedback_file = model_dir / "review_feedback.jsonl"
        if feedback_file.exists():
            files_to_delete.append(feedback_file)
        
        training_file = model_dir / "training_data.pkl"
        if training_file.exists():
            files_to_delete.append(training_file)
    
    if not files_to_delete:
        print("ℹ️ No files to delete")
        return 0
    
    # Confirm deletion
    print("🗑️ Files to be deleted:")
    for file in files_to_delete:
        print(f"  {file}")
    
    if not args.force:
        confirm = input("\nAre you sure? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Operation cancelled")
            return 1
    
    # Delete files
    deleted_count = 0
    for file in files_to_delete:
        try:
            file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ Error deleting {file}: {e}")
    
    print(f"✅ Deleted {deleted_count} files")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Learning Manager - Manage adaptive learning for traffic plan review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect feedback from a review
  python learning_manager.py collect-feedback plan_001 traffic_signal reviewer_001 --confidence 0.8 --review-time 120

  # Train models on accumulated feedback
  python learning_manager.py train-models --min-examples 10

  # Show learning statistics
  python learning_manager.py show-statistics

  # Export feedback data
  python learning_manager.py export-feedback --format json --output feedback.json

  # Import feedback data
  python learning_manager.py import-feedback --format json --input feedback.json

  # Reset models and start fresh
  python learning_manager.py reset-models --feedback --force
        """
    )
    
    parser.add_argument('--model-dir', default='models', help='Directory for models and feedback')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect feedback command
    collect_parser = subparsers.add_parser('collect-feedback', help='Collect feedback from human review')
    collect_parser.add_argument('plan_id', help='Unique plan identifier')
    collect_parser.add_argument('plan_type', help='Type of plan (traffic_signal, its, mutcd_signing)')
    collect_parser.add_argument('reviewer_id', help='Reviewer identifier')
    collect_parser.add_argument('--confidence', type=float, default=0.5, help='Confidence score (0-1)')
    collect_parser.add_argument('--review-time', type=float, default=0.0, help='Review time in seconds')
    collect_parser.add_argument('--notes', default='', help='Additional notes')
    collect_parser.add_argument('--original-prediction', help='Path to original prediction JSON file')
    collect_parser.add_argument('--human-corrections', help='Path to human corrections JSON file')
    collect_parser.set_defaults(func=collect_feedback_command)
    
    # Train models command
    train_parser = subparsers.add_parser('train-models', help='Train models on accumulated feedback')
    train_parser.add_argument('--min-examples', type=int, default=10, help='Minimum examples required for training')
    train_parser.set_defaults(func=train_models_command)
    
    # Show statistics command
    stats_parser = subparsers.add_parser('show-statistics', help='Show learning system statistics')
    stats_parser.set_defaults(func=show_statistics_command)
    
    # Export feedback command
    export_parser = subparsers.add_parser('export-feedback', help='Export feedback data')
    export_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Export format')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.set_defaults(func=export_feedback_command)
    
    # Import feedback command
    import_parser = subparsers.add_parser('import-feedback', help='Import feedback data')
    import_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Import format')
    import_parser.add_argument('--input', required=True, help='Input file path')
    import_parser.set_defaults(func=import_feedback_command)
    
    # Reset models command
    reset_parser = subparsers.add_parser('reset-models', help='Reset models and feedback data')
    reset_parser.add_argument('--feedback', action='store_true', help='Also delete feedback data')
    reset_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    reset_parser.set_defaults(func=reset_models_command)
    
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