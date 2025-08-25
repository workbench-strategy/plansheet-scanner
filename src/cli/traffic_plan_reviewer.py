#!/usr/bin/env python3
"""
Traffic Plan Reviewer CLI
Specialized command-line interface for traffic signal, ITS, and MUTCD signing plan review.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.traffic_plan_reviewer import TrafficPlanReviewer, PlanReviewResult

def print_review_results(result: PlanReviewResult):
    """Print plan review results in a formatted way."""
    
    print(f"\n🚦 Traffic Plan Review Results")
    print("=" * 60)
    print(f"📋 Plan Type: {result.plan_type.replace('_', ' ').title()}")
    print(f"📊 Compliance Score: {result.compliance_score:.2f} ({result.compliance_score*100:.1f}%)")
    print(f"📚 Standards Checked: {', '.join(result.standards_checked)}")
    
    # Compliance status
    if result.compliance_score >= 0.9:
        status = "✅ EXCELLENT"
    elif result.compliance_score >= 0.7:
        status = "🟡 GOOD"
    elif result.compliance_score >= 0.5:
        status = "🟠 FAIR"
    else:
        status = "🔴 POOR"
    
    print(f"📈 Overall Status: {status}")
    
    # Issues summary
    if result.issues:
        print(f"\n⚠️  Issues Found ({len(result.issues)}):")
        print("-" * 40)
        
        # Group issues by severity
        critical_issues = [i for i in result.issues if i['severity'] == 'critical']
        high_issues = [i for i in result.issues if i['severity'] == 'high']
        medium_issues = [i for i in result.issues if i['severity'] == 'medium']
        low_issues = [i for i in result.issues if i['severity'] == 'low']
        
        if critical_issues:
            print(f"🔴 Critical Issues ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"   • {issue['issue']}")
                print(f"     Standard: {issue['standard']}")
                print(f"     Recommendation: {issue['recommendation']}")
                print()
        
        if high_issues:
            print(f"🟠 High Priority Issues ({len(high_issues)}):")
            for issue in high_issues:
                print(f"   • {issue['issue']}")
                print(f"     Standard: {issue['standard']}")
                print(f"     Recommendation: {issue['recommendation']}")
                print()
        
        if medium_issues:
            print(f"🟡 Medium Priority Issues ({len(medium_issues)}):")
            for issue in medium_issues:
                print(f"   • {issue['issue']}")
                print(f"     Standard: {issue['standard']}")
                print(f"     Recommendation: {issue['recommendation']}")
                print()
        
        if low_issues:
            print(f"🟢 Low Priority Issues ({len(low_issues)}):")
            for issue in low_issues:
                print(f"   • {issue['issue']}")
                print(f"     Standard: {issue['standard']}")
                print(f"     Recommendation: {issue['recommendation']}")
                print()
    else:
        print(f"\n✅ No issues found! Plan appears to meet all standards.")
    
    # Recommendations
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        print("-" * 40)
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Elements detected
    if result.elements_found:
        print(f"\n🔍 Elements Detected ({len(result.elements_found)}):")
        print("-" * 40)
        
        # Group elements by type
        element_types = {}
        for element in result.elements_found:
            element_type = element.element_type.split('_')[0]  # Get base type
            if element_type not in element_types:
                element_types[element_type] = []
            element_types[element_type].append(element)
        
        for element_type, elements in element_types.items():
            print(f"   📍 {element_type.title()} ({len(elements)}):")
            for element in elements:
                print(f"      • {element.element_type} at ({element.location[0]:.1f}, {element.location[1]:.1f})")
                if 'confidence' in element.__dict__:
                    print(f"        Confidence: {element.confidence:.2f}")

def review_command(args):
    """Handle the review command."""
    reviewer = TrafficPlanReviewer()
    
    if not os.path.exists(args.plan):
        print(f"❌ Error: Plan file {args.plan} not found.")
        return 1
    
    print(f"🔍 Reviewing traffic plan: {args.plan}")
    print(f"📋 Plan Type: {args.plan_type}")
    
    try:
        result = reviewer.review_plan(args.plan, args.plan_type)
        print_review_results(result)
        
        # Save results if requested
        if args.output:
            # Convert result to dictionary for JSON serialization
            result_dict = {
                'plan_type': result.plan_type,
                'compliance_score': result.compliance_score,
                'issues': result.issues,
                'recommendations': result.recommendations,
                'standards_checked': result.standards_checked,
                'elements_found': [
                    {
                        'element_type': e.element_type,
                        'location': e.location,
                        'confidence': e.confidence,
                        'metadata': e.metadata,
                        'bounding_box': e.bounding_box
                    }
                    for e in result.elements_found
                ]
            }
            
            with open(args.output, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.output}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error reviewing plan: {e}")
        return 1

def batch_review_command(args):
    """Handle the batch review command."""
    reviewer = TrafficPlanReviewer()
    
    if not os.path.exists(args.directory):
        print(f"❌ Error: Directory {args.directory} not found.")
        return 1
    
    # Find all plan files
    plan_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    plan_files = []
    
    for ext in plan_extensions:
        plan_files.extend(Path(args.directory).glob(f"*{ext}"))
        plan_files.extend(Path(args.directory).glob(f"*{ext.upper()}"))
    
    if not plan_files:
        print(f"❌ No plan files found in {args.directory}")
        return 1
    
    print(f"🔍 Found {len(plan_files)} plan files for batch review")
    print(f"📋 Plan Type: {args.plan_type}")
    
    results = []
    
    for plan_file in plan_files:
        print(f"\n📄 Reviewing: {plan_file.name}")
        
        try:
            result = reviewer.review_plan(str(plan_file), args.plan_type)
            results.append({
                'file': plan_file.name,
                'result': result
            })
            
            # Print summary
            status = "✅" if result.compliance_score >= 0.7 else "⚠️" if result.compliance_score >= 0.5 else "❌"
            print(f"   {status} Compliance: {result.compliance_score:.2f} ({len(result.issues)} issues)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'file': plan_file.name,
                'error': str(e)
            })
    
    # Summary report
    print(f"\n📊 Batch Review Summary")
    print("=" * 50)
    
    successful_reviews = [r for r in results if 'result' in r]
    failed_reviews = [r for r in results if 'error' in r]
    
    print(f"✅ Successful Reviews: {len(successful_reviews)}")
    print(f"❌ Failed Reviews: {len(failed_reviews)}")
    
    if successful_reviews:
        avg_compliance = sum(r['result'].compliance_score for r in successful_reviews) / len(successful_reviews)
        print(f"📊 Average Compliance Score: {avg_compliance:.2f}")
        
        # Best and worst plans
        best_plan = max(successful_reviews, key=lambda x: x['result'].compliance_score)
        worst_plan = min(successful_reviews, key=lambda x: x['result'].compliance_score)
        
        print(f"🏆 Best Plan: {best_plan['file']} ({best_plan['result'].compliance_score:.2f})")
        print(f"⚠️  Worst Plan: {worst_plan['file']} ({worst_plan['result'].compliance_score:.2f})")
    
    # Save batch results if requested
    if args.output:
        batch_results = {
            'summary': {
                'total_plans': len(results),
                'successful_reviews': len(successful_reviews),
                'failed_reviews': len(failed_reviews),
                'average_compliance': avg_compliance if successful_reviews else 0
            },
            'results': []
        }
        
        for result in results:
            if 'result' in result:
                batch_results['results'].append({
                    'file': result['file'],
                    'plan_type': result['result'].plan_type,
                    'compliance_score': result['result'].compliance_score,
                    'issues_count': len(result['result'].issues),
                    'standards_checked': result['result'].standards_checked
                })
            else:
                batch_results['results'].append({
                    'file': result['file'],
                    'error': result['error']
                })
        
        with open(args.output, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        print(f"\n💾 Batch results saved to {args.output}")
    
    return 0

def standards_command(args):
    """Handle the standards command - show available standards."""
    print("📚 Available Traffic Engineering Standards")
    print("=" * 50)
    
    standards = {
        'MUTCD (Manual on Uniform Traffic Control Devices)': {
            'description': 'Federal standards for traffic control devices',
            'sections': ['Part 1 - General', 'Part 2 - Signs', 'Part 3 - Markings', 'Part 4 - Signals'],
            'applicable_to': ['mutcd_signing', 'traffic_signal']
        },
        'ITE Signal Timing Manual': {
            'description': 'Institute of Transportation Engineers signal timing guidelines',
            'sections': ['Signal Design', 'Timing Parameters', 'Coordination', 'Pedestrian Features'],
            'applicable_to': ['traffic_signal']
        },
        'AASHTO Green Book': {
            'description': 'American Association of State Highway and Transportation Officials geometric design',
            'sections': ['Geometric Design', 'Intersection Design', 'Safety Considerations'],
            'applicable_to': ['traffic_signal', 'mutcd_signing']
        },
        'NTCIP (National Transportation Communications for ITS Protocol)': {
            'description': 'Standards for ITS communications and protocols',
            'sections': ['Device Communications', 'Data Exchange', 'System Integration'],
            'applicable_to': ['its']
        },
        'ADA Standards': {
            'description': 'Americans with Disabilities Act accessibility requirements',
            'sections': ['Pedestrian Access', 'Signal Accessibility', 'Crossing Design'],
            'applicable_to': ['traffic_signal', 'mutcd_signing']
        }
    }
    
    for standard, info in standards.items():
        print(f"\n📖 {standard}")
        print(f"   Description: {info['description']}")
        print(f"   Applicable to: {', '.join(info['applicable_to'])}")
        print(f"   Key Sections: {', '.join(info['sections'])}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Traffic Plan Reviewer - Specialized review for traffic signals, ITS, and MUTCD signing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review a traffic signal plan
  python traffic_plan_reviewer.py review signal_plan.pdf --type traffic_signal

  # Review MUTCD signing plan
  python traffic_plan_reviewer.py review signing_plan.pdf --type mutcd_signing

  # Review ITS plan with auto-detection
  python traffic_plan_reviewer.py review its_plan.pdf --type auto

  # Batch review all plans in a directory
  python traffic_plan_reviewer.py batch /path/to/plans --type traffic_signal

  # Show available standards
  python traffic_plan_reviewer.py standards
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Review command
    review_parser = subparsers.add_parser('review', help='Review a single traffic plan')
    review_parser.add_argument('plan', help='Path to the plan file (PDF or image)')
    review_parser.add_argument('--type', '--plan-type', dest='plan_type', 
                              choices=['auto', 'traffic_signal', 'its', 'mutcd_signing'],
                              default='auto', help='Type of plan to review')
    review_parser.add_argument('--output', help='Save results to JSON file')
    review_parser.set_defaults(func=review_command)
    
    # Batch review command
    batch_parser = subparsers.add_parser('batch', help='Review multiple plans in a directory')
    batch_parser.add_argument('directory', help='Directory containing plan files')
    batch_parser.add_argument('--type', '--plan-type', dest='plan_type',
                             choices=['auto', 'traffic_signal', 'its', 'mutcd_signing'],
                             default='auto', help='Type of plans to review')
    batch_parser.add_argument('--output', help='Save batch results to JSON file')
    batch_parser.set_defaults(func=batch_review_command)
    
    # Standards command
    standards_parser = subparsers.add_parser('standards', help='Show available traffic engineering standards')
    standards_parser.set_defaults(func=standards_command)
    
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