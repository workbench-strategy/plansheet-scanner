#!/usr/bin/env python3
"""
Plan Review and Issue Tagging System
Demonstrates how to use the ML system to review plans and tag issues.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import time
from improved_ai_trainer import ImprovedAIEngineerTrainer

class PlanReviewer:
    """Comprehensive plan review and issue tagging system."""
    
    def __init__(self):
        """Initialize the plan reviewer with trained ML models."""
        print("🤖 Initializing Plan Review System...")
        self.trainer = ImprovedAIEngineerTrainer()
        self.issue_tags = {
            "critical": "🔴 Critical - Must be addressed immediately",
            "high": "🟠 High - Significant impact on safety/functionality", 
            "medium": "🟡 Medium - Moderate impact, should be reviewed",
            "low": "🟢 Low - Minor issue, consider for future improvements",
            "info": "🔵 Info - Informational note"
        }
        
        print(f"✅ Loaded {len(self.trainer.as_built_drawings)} training examples")
        print(f"✅ Models ready for plan review")
    
    def review_plan_comprehensive(self, plan_data):
        """Comprehensive plan review with detailed issue tagging."""
        print(f"\n🔍 COMPREHENSIVE PLAN REVIEW")
        print("=" * 60)
        
        # Basic plan info
        print(f"📄 Plan: {plan_data.get('sheet_title', 'Unknown Plan')}")
        print(f"🏗️  Discipline: {plan_data.get('discipline', 'Unknown')}")
        print(f"📋 Project: {plan_data.get('project_name', 'Unknown Project')}")
        print()
        
        # ML Analysis
        print("🤖 ML ANALYSIS RESULTS:")
        print("-" * 40)
        
        ml_result = self.trainer.review_drawing(plan_data)
        
        print(f"🎯 Predicted Discipline: {ml_result['predicted_discipline']}")
        print(f"⚠️  Code Violations Detected: {ml_result['has_code_violations']}")
        print(f"❌ Design Errors Found: {ml_result['has_design_errors']}")
        print(f"📊 Confidence Level: {ml_result['overall_confidence']:.1%}")
        
        # Generate detailed issues
        issues = self._generate_detailed_issues(plan_data, ml_result)
        
        return {
            "plan_info": plan_data,
            "ml_analysis": ml_result,
            "issues": issues,
            "review_summary": self._create_review_summary(issues)
        }
    
    def _generate_detailed_issues(self, plan_data, ml_result):
        """Generate detailed issue tags based on plan content and ML analysis."""
        issues = []
        
        # Discipline-specific issues
        discipline = ml_result['predicted_discipline']
        
        if discipline == "traffic":
            issues.extend(self._check_traffic_issues(plan_data))
        elif discipline == "electrical":
            issues.extend(self._check_electrical_issues(plan_data))
        elif discipline == "structural":
            issues.extend(self._check_structural_issues(plan_data))
        elif discipline == "drainage":
            issues.extend(self._check_drainage_issues(plan_data))
        
        # General compliance issues
        if ml_result['has_code_violations']:
            issues.extend(self._check_compliance_issues(plan_data))
        
        # Design quality issues
        if ml_result['has_design_errors']:
            issues.extend(self._check_design_issues(plan_data))
        
        # Confidence-based issues
        if ml_result['overall_confidence'] < 0.7:
            issues.append({
                "tag": "medium",
                "category": "Review Quality",
                "title": "Low Confidence Analysis",
                "description": f"ML analysis confidence is {ml_result['overall_confidence']:.1%}. Manual review recommended.",
                "recommendation": "Have experienced engineer review this plan manually."
            })
        
        return issues
    
    def _check_traffic_issues(self, plan_data):
        """Check for traffic-specific issues."""
        issues = []
        
        # MUTCD compliance checks
        issues.append({
            "tag": "high",
            "category": "Traffic Control",
            "title": "MUTCD Compliance Review Required",
            "description": "Traffic control devices must comply with Manual on Uniform Traffic Control Devices.",
            "recommendation": "Verify signal timing, signing, and pavement markings meet MUTCD standards."
        })
        
        # Signal coordination
        issues.append({
            "tag": "medium",
            "category": "Signal Systems",
            "title": "Signal Coordination Analysis",
            "description": "Check signal coordination with adjacent intersections.",
            "recommendation": "Review signal timing and coordination parameters."
        })
        
        # ITS systems
        if "its" in plan_data.get('sheet_title', '').lower():
            issues.append({
                "tag": "high",
                "category": "ITS Systems",
                "title": "ITS System Integration",
                "description": "Intelligent Transportation Systems require proper integration.",
                "recommendation": "Verify ITS equipment compatibility and communication protocols."
            })
        
        return issues
    
    def _check_electrical_issues(self, plan_data):
        """Check for electrical-specific issues."""
        issues = []
        
        # NEC compliance
        issues.append({
            "tag": "critical",
            "category": "Electrical Safety",
            "title": "NEC Compliance Verification",
            "description": "Electrical installations must comply with National Electrical Code.",
            "recommendation": "Review electrical design for NEC compliance, especially grounding and bonding."
        })
        
        # Power distribution
        issues.append({
            "tag": "high",
            "category": "Power Systems",
            "title": "Power Distribution Analysis",
            "description": "Verify adequate power capacity and distribution.",
            "recommendation": "Check load calculations and power distribution design."
        })
        
        # Illumination systems
        if "illumination" in plan_data.get('sheet_title', '').lower():
            issues.append({
                "tag": "medium",
                "category": "Illumination",
                "title": "Lighting Design Review",
                "description": "Verify lighting levels meet roadway illumination standards.",
                "recommendation": "Check lighting calculations and fixture specifications."
            })
        
        return issues
    
    def _check_structural_issues(self, plan_data):
        """Check for structural-specific issues."""
        issues = []
        
        # AASHTO compliance
        issues.append({
            "tag": "critical",
            "category": "Structural Safety",
            "title": "AASHTO Design Standards",
            "description": "Structural elements must meet AASHTO design standards.",
            "recommendation": "Verify load ratings, design loads, and structural calculations."
        })
        
        # Foundation design
        issues.append({
            "tag": "high",
            "category": "Foundation",
            "title": "Foundation Design Review",
            "description": "Check foundation design for soil conditions and loads.",
            "recommendation": "Review geotechnical reports and foundation design."
        })
        
        return issues
    
    def _check_drainage_issues(self, plan_data):
        """Check for drainage-specific issues."""
        issues = []
        
        # Storm water management
        issues.append({
            "tag": "high",
            "category": "Storm Water",
            "title": "Storm Water Management",
            "description": "Verify storm water collection and conveyance systems.",
            "recommendation": "Check drainage calculations and pipe sizing."
        })
        
        # Culvert design
        if "culvert" in plan_data.get('sheet_title', '').lower():
            issues.append({
                "tag": "medium",
                "category": "Culvert Design",
                "title": "Culvert Capacity Verification",
                "description": "Verify culvert capacity for design storm events.",
                "recommendation": "Review hydraulic calculations and culvert sizing."
            })
        
        return issues
    
    def _check_compliance_issues(self, plan_data):
        """Check for general compliance issues."""
        return [{
            "tag": "high",
            "category": "Code Compliance",
            "title": "Code Compliance Review Required",
            "description": "ML analysis detected potential code compliance issues.",
            "recommendation": "Review plan against applicable building codes and standards."
        }]
    
    def _check_design_issues(self, plan_data):
        """Check for general design issues."""
        return [{
            "tag": "medium",
            "category": "Design Quality",
            "title": "Design Quality Review Required",
            "description": "ML analysis detected potential design issues.",
            "recommendation": "Review design for technical accuracy and constructability."
        }]
    
    def _create_review_summary(self, issues):
        """Create a summary of the review results."""
        if not issues:
            return "✅ No issues detected - Plan appears to meet standards."
        
        critical_count = len([i for i in issues if i['tag'] == 'critical'])
        high_count = len([i for i in issues if i['tag'] == 'high'])
        medium_count = len([i for i in issues if i['tag'] == 'medium'])
        low_count = len([i for i in issues if i['tag'] == 'low'])
        
        summary = f"📊 REVIEW SUMMARY:\n"
        summary += f"   🔴 Critical Issues: {critical_count}\n"
        summary += f"   🟠 High Priority: {high_count}\n"
        summary += f"   🟡 Medium Priority: {medium_count}\n"
        summary += f"   🟢 Low Priority: {low_count}\n"
        summary += f"   📋 Total Issues: {len(issues)}"
        
        return summary
    
    def print_review_results(self, review_results):
        """Print comprehensive review results with issue tags."""
        print(f"\n📋 DETAILED ISSUE ANALYSIS:")
        print("=" * 60)
        
        if not review_results['issues']:
            print("✅ No issues detected!")
            return
        
        # Group issues by priority
        by_priority = {}
        for issue in review_results['issues']:
            priority = issue['tag']
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append(issue)
        
        # Print issues by priority
        for priority in ['critical', 'high', 'medium', 'low', 'info']:
            if priority in by_priority:
                print(f"\n{self.issue_tags[priority]}:")
                print("-" * 50)
                
                for i, issue in enumerate(by_priority[priority], 1):
                    print(f"{i}. {issue['title']}")
                    print(f"   Category: {issue['category']}")
                    print(f"   Description: {issue['description']}")
                    print(f"   Recommendation: {issue['recommendation']}")
                    print()
        
        # Print summary
        print(review_results['review_summary'])
    
    def export_review_report(self, review_results, filename=None):
        """Export review results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plan_review_report_{timestamp}.json"
        
        # Add metadata
        report_data = {
            "review_metadata": {
                "review_date": datetime.now().isoformat(),
                "reviewer": "ML-Powered Plan Review System",
                "version": "1.0"
            },
            "review_results": review_results
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Review report exported to: {filename}")
        return filename

def demonstrate_plan_review():
    """Demonstrate the plan review system with sample plans."""
    print("🏗️ PLAN REVIEW AND ISSUE TAGGING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize reviewer
    reviewer = PlanReviewer()
    
    # Sample plans to review
    sample_plans = [
        {
            "sheet_title": "Traffic Signal Plan - I-5 at 272nd Ave",
            "discipline": "traffic",
            "project_name": "I-5 Corridor Improvements",
            "construction_notes": "New traffic signal installation with ITS integration",
            "as_built_changes": [
                {"description": "Signal timing modified during construction", "severity": "minor"}
            ]
        },
        {
            "sheet_title": "Electrical Power Distribution - SR 509",
            "discipline": "electrical", 
            "project_name": "SR 509 Widening Project",
            "construction_notes": "Power distribution for roadway lighting and signals",
            "as_built_changes": [
                {"description": "Additional conduit installed for future expansion", "severity": "minor"}
            ]
        },
        {
            "sheet_title": "Bridge Foundation Design - I-90",
            "discipline": "structural",
            "project_name": "I-90 Bridge Rehabilitation",
            "construction_notes": "Foundation design for bridge pier rehabilitation",
            "as_built_changes": []
        },
        {
            "sheet_title": "Storm Drainage System - L200",
            "discipline": "drainage",
            "project_name": "L200 Local Road Improvements",
            "construction_notes": "Storm water collection and conveyance system",
            "as_built_changes": [
                {"description": "Culvert size increased due to field conditions", "severity": "moderate"}
            ]
        }
    ]
    
    # Review each plan
    for i, plan in enumerate(sample_plans, 1):
        print(f"\n{'='*80}")
        print(f"PLAN REVIEW #{i}")
        print(f"{'='*80}")
        
        # Perform comprehensive review
        review_results = reviewer.review_plan_comprehensive(plan)
        
        # Print results
        reviewer.print_review_results(review_results)
        
        # Export report
        report_filename = reviewer.export_review_report(review_results, f"plan_review_{i}_{plan['discipline']}.json")
        
        print(f"\n✅ Plan #{i} review completed!")
        print(f"📄 Report saved: {report_filename}")
    
    print(f"\n🎉 ALL PLAN REVIEWS COMPLETED!")
    print("=" * 80)
    print("The ML system has successfully reviewed all sample plans and tagged issues.")
    print("Each review includes:")
    print("  • ML-powered discipline classification")
    print("  • Code compliance checking")
    print("  • Design error detection")
    print("  • Priority-based issue tagging")
    print("  • Specific recommendations")
    print("  • Exportable reports")

if __name__ == "__main__":
    demonstrate_plan_review()
