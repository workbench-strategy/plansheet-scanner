#!/usr/bin/env python3
"""
Test ML System - Simple verification script
"""

import time
from improved_ai_trainer import ImprovedAIEngineerTrainer

def test_ml_system():
    """Test the ML system functionality."""
    print("🤖 Testing ML System")
    print("=" * 50)
    
    # Initialize trainer
    print("📊 Initializing AI Trainer...")
    trainer = ImprovedAIEngineerTrainer()
    
    # Check if models are trained
    stats = trainer.get_training_statistics()
    print(f"✅ Models trained: {stats.get('models_trained', False)}")
    print(f"📊 Training examples: {stats.get('total_drawings', 0)}")
    
    # Test review functionality
    print("\n🧪 Testing Review Functionality...")
    
    test_cases = [
        {
            "drawing_id": "test_001",
            "project_name": "SR-167 Interchange",
            "sheet_number": "T-01",
            "sheet_title": "Traffic Signal Plan",
            "discipline": "traffic",
            "original_design": {
                "signal_heads": 4,
                "detector_loops": 8,
                "pedestrian_signals": 2
            },
            "as_built_changes": [
                {"type": "signal_head_added", "location": [100, 200]}
            ],
            "code_references": ["MUTCD", "WSDOT Standards"],
            "review_notes": ["Good design", "Meets requirements"],
            "approval_status": "approved",
            "reviewer_feedback": {"compliance_score": 0.95},
            "construction_notes": "All elements installed per plan",
            "file_path": "traffic_plan_001.pdf"
        },
        {
            "drawing_id": "test_002",
            "project_name": "I-5 Bridge Rehabilitation",
            "sheet_number": "E-01",
            "sheet_title": "Electrical Plan",
            "discipline": "electrical",
            "original_design": {
                "conduit_runs": 12,
                "junction_boxes": 6,
                "grounding_points": 4
            },
            "as_built_changes": [
                {"type": "conduit_modified", "location": [300, 150]}
            ],
            "code_references": ["NEC", "WSDOT Standards"],
            "review_notes": ["Minor modifications required"],
            "approval_status": "conditional",
            "reviewer_feedback": {"compliance_score": 0.88},
            "construction_notes": "Conduit routing adjusted for field conditions",
            "file_path": "electrical_plan_001.pdf"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['sheet_number']} - {test_case['sheet_title']}")
        
        start_time = time.time()
        result = trainer.review_drawing(test_case)
        review_time = time.time() - start_time
        
        print(f"   ⏱️  Review time: {review_time:.3f} seconds")
        print(f"   🎯 Predicted discipline: {result.get('predicted_discipline', 'unknown')}")
        print(f"   ⚠️  Code violations: {result.get('has_code_violations', False)}")
        print(f"   ❌ Design errors: {result.get('has_design_errors', False)}")
        print(f"   📊 Overall confidence: {result.get('overall_confidence', 0):.3f}")
        
        if 'recommendations' in result:
            print(f"   💡 Recommendations:")
            for rec in result['recommendations'][:3]:  # Show first 3
                print(f"     - {rec}")
    
    # Performance summary
    print(f"\n📈 Performance Summary:")
    print(f"   - Models available: {stats.get('models_trained', False)}")
    print(f"   - Training examples: {stats.get('total_drawings', 0)}")
    print(f"   - Review patterns: {stats.get('review_patterns', 0)}")
    print(f"   - Violations detected: {stats.get('total_violations', 0)}")
    print(f"   - Errors found: {stats.get('total_errors', 0)}")
    
    return True

if __name__ == "__main__":
    success = test_ml_system()
    if success:
        print("\n✅ ML System Test Completed Successfully!")
    else:
        print("\n❌ ML System Test Failed!")
