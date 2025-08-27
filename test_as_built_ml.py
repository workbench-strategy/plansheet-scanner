#!/usr/bin/env python3
"""
Test As-Built Pipeline ML Functionality
Comprehensive evaluation of the ML system with verbose output.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Setup verbose logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_as_built_ml.log')
    ]
)
logger = logging.getLogger(__name__)

def test_ml_imports():
    """Test that all ML dependencies are properly installed."""
    print("=" * 60)
    print("🧪 TESTING ML DEPENDENCIES")
    print("=" * 60)
    
    ml_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('sklearn', 'sklearn'),
        ('torch', 'torch'),
        ('joblib', 'joblib'),
        ('cv2', 'cv2'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn')
    ]
    
    results = {}
    for package_name, import_name in ml_packages:
        try:
            __import__(import_name)
            version = getattr(__import__(import_name), '__version__', 'unknown')
            print(f"✅ {package_name}: {version}")
            results[package_name] = {'status': 'success', 'version': version}
        except ImportError as e:
            print(f"❌ {package_name}: {e}")
            results[package_name] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_improved_ai_trainer():
    """Test the improved AI trainer functionality."""
    print("\n" + "=" * 60)
    print("🤖 TESTING IMPROVED AI TRAINER")
    print("=" * 60)
    
    try:
        from improved_ai_trainer import ImprovedAIEngineerTrainer
        
        print("📊 Initializing Improved AI Trainer...")
        trainer = ImprovedAIEngineerTrainer(
            data_dir="test_improved_ai_data",
            model_dir="test_improved_ai_models"
        )
        
        print(f"✅ Trainer initialized successfully")
        print(f"   - Data directory: {trainer.data_dir}")
        print(f"   - Model directory: {trainer.model_dir}")
        
        # Test data generation
        print("\n📈 Generating training data...")
        start_time = time.time()
        trainer.generate_diverse_training_data(num_examples=50)  # Small test set
        generation_time = time.time() - start_time
        
        print(f"✅ Generated {len(trainer.as_built_drawings)} training examples")
        print(f"   - Generation time: {generation_time:.2f} seconds")
        print(f"   - Average time per example: {generation_time/len(trainer.as_built_drawings):.3f} seconds")
        
        # Test model training
        print("\n🎯 Training ML models...")
        start_time = time.time()
        trainer.train_models(min_examples=10)  # Lower threshold for testing
        training_time = time.time() - start_time
        
        print(f"✅ Model training completed")
        print(f"   - Training time: {training_time:.2f} seconds")
        
        # Get training statistics
        stats = trainer.get_training_statistics()
        print(f"\n📊 Training Statistics:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        return {
            'status': 'success',
            'training_examples': len(trainer.as_built_drawings),
            'generation_time': generation_time,
            'training_time': training_time,
            'statistics': stats
        }
        
    except Exception as e:
        print(f"❌ Error testing Improved AI Trainer: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}

def test_foundation_trainer():
    """Test the foundation trainer functionality."""
    print("\n" + "=" * 60)
    print("🏗️ TESTING FOUNDATION TRAINER")
    print("=" * 60)
    
    try:
        from src.core.foundation_trainer import FoundationTrainer
        
        print("📊 Initializing Foundation Trainer...")
        trainer = FoundationTrainer(
            data_dir="test_foundation_data",
            model_dir="test_foundation_models"
        )
        
        print(f"✅ Foundation trainer initialized successfully")
        print(f"   - Data directory: {trainer.data_dir}")
        print(f"   - Model directory: {trainer.model_dir}")
        
        # Test adding as-built data
        print("\n📋 Adding sample as-built data...")
        
        # Create sample as-built data
        sample_as_built = {
            'plan_id': 'test_001',
            'plan_type': 'traffic_signal',
            'as_built_image': np.random.rand(100, 100, 3),  # Random image
            'final_elements': [
                {'type': 'signal_head', 'location': [100, 200], 'approved': True},
                {'type': 'detector_loop', 'location': [80, 180], 'approved': True}
            ],
            'construction_notes': 'All elements installed per approved plans',
            'approval_date': datetime.now(),
            'project_info': {
                'budget': 500000,
                'duration_months': 6,
                'complexity_score': 0.7
            }
        }
        
        trainer.add_as_built_data(**sample_as_built)
        print(f"✅ Added sample as-built data")
        print(f"   - Total as-built records: {len(trainer.as_built_data)}")
        
        # Test adding review milestone
        print("\n📝 Adding sample review milestone...")
        trainer.add_review_milestone(
            plan_id='test_001',
            milestone='final',
            reviewer_comments=['Good design', 'Meets all requirements'],
            approved_elements=[{'type': 'signal_head', 'location': [100, 200]}],
            rejected_elements=[],
            compliance_score=0.95,
            review_date=datetime.now()
        )
        print(f"✅ Added sample review milestone")
        print(f"   - Total review milestones: {len(trainer.review_milestones)}")
        
        return {
            'status': 'success',
            'as_built_count': len(trainer.as_built_data),
            'milestone_count': len(trainer.review_milestones)
        }
        
    except Exception as e:
        print(f"❌ Error testing Foundation Trainer: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}

def test_adaptive_reviewer():
    """Test the adaptive reviewer functionality."""
    print("\n" + "=" * 60)
    print("🔄 TESTING ADAPTIVE REVIEWER")
    print("=" * 60)
    
    try:
        from src.core.adaptive_reviewer import AdaptiveReviewer
        
        print("📊 Initializing Adaptive Reviewer...")
        reviewer = AdaptiveReviewer(model_dir="test_adaptive_models")
        
        print(f"✅ Adaptive reviewer initialized successfully")
        print(f"   - Model directory: {reviewer.model_dir}")
        
        # Test plan review
        print("\n🔍 Testing plan review functionality...")
        
        # Create sample plan data with proper object structure
        class MockElement:
            def __init__(self, element_type, confidence, location):
                self.element_type = element_type
                self.confidence = confidence
                self.location = location
                self.metadata = {'size': [20, 20]}
        
        sample_plan = {
            'plan_image': np.random.rand(200, 300, 3),  # Random image
            'detected_elements': [
                MockElement('signal_head', 0.9, [100, 200]),
                MockElement('detector_loop', 0.8, [80, 180])
            ],
            'plan_type': 'traffic_signal'
        }
        
        review_result = reviewer.review_plan(**sample_plan)
        print(f"✅ Plan review completed")
        print(f"   - Review status: {review_result.get('status', 'unknown')}")
        print(f"   - Features extracted: {len(review_result.get('features', []))}")
        
        # Test feedback recording
        print("\n📝 Testing feedback recording...")
        reviewer.record_feedback(
            plan_id='test_plan_001',
            plan_type='traffic_signal',
            reviewer_id='test_reviewer',
            original_prediction=review_result,
            human_corrections={'signal_head': 'approved', 'detector_loop': 'rejected'},
            confidence_score=0.85,
            review_time=120.5,
            notes='Good overall design, but detector placement needs adjustment'
        )
        print(f"✅ Feedback recorded successfully")
        
        return {
            'status': 'success',
            'review_result': review_result
        }
        
    except Exception as e:
        print(f"❌ Error testing Adaptive Reviewer: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}

def test_as_built_processor():
    """Test the as-built processor functionality."""
    print("\n" + "=" * 60)
    print("🏗️ TESTING AS-BUILT PROCESSOR")
    print("=" * 60)
    
    try:
        from as_built_processor import AsBuiltProcessor
        
        print("📊 Initializing As-Built Processor...")
        processor = AsBuiltProcessor()
        
        print(f"✅ As-built processor initialized successfully")
        
        # Show data flow
        print("\n🔄 Data Flow Structure:")
        processor.show_data_flow()
        
        # Show current stats
        print("\n📊 Current Processing Statistics:")
        processor.show_processing_stats()
        
        # Test processing (simulated)
        print("\n🔄 Testing processing simulation...")
        result = processor.process_as_built("test_drawing.pdf")
        
        print(f"✅ Processing simulation completed")
        print(f"   - Status: {result['status']}")
        print(f"   - Symbols extracted: {result['symbols_extracted']}")
        print(f"   - Files created: {len(result['files_created'])}")
        
        return {
            'status': 'success',
            'processing_result': result
        }
        
    except Exception as e:
        print(f"❌ Error testing As-Built Processor: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}

def test_performance_metrics():
    """Test and report performance metrics."""
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE METRICS")
    print("=" * 60)
    
    import psutil
    import time
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    print(f"💻 System Performance:")
    print(f"   - CPU Usage: {cpu_percent:.1f}%")
    print(f"   - Memory Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f} GB / {memory.total / 1024**3:.1f} GB)")
    print(f"   - Disk Usage: {disk.percent:.1f}% ({disk.used / 1024**3:.1f} GB / {disk.total / 1024**3:.1f} GB)")
    
    # Test ML model loading time
    print(f"\n⚡ ML Model Performance:")
    start_time = time.time()
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        
        # Create and train a simple model
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test prediction speed
        X_test = np.random.rand(10, 10)
        pred_start = time.time()
        predictions = model.predict(X_test)
        pred_time = time.time() - pred_start
        
        print(f"   - Model training time: {time.time() - start_time:.3f} seconds")
        print(f"   - Prediction time (10 samples): {pred_time:.3f} seconds")
        if pred_time > 0:
            print(f"   - Prediction speed: {10/pred_time:.1f} samples/second")
        else:
            print(f"   - Prediction speed: Very fast (< 0.001 seconds)")
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'model_training_time': time.time() - start_time,
            'prediction_time': pred_time,
            'prediction_speed': 10/pred_time
        }
        
    except Exception as e:
        print(f"   - Model performance test failed: {e}")
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'model_test_error': str(e)
        }
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': disk.percent
    }

def main():
    """Run all ML functionality tests."""
    print("🚀 AS-BUILT PIPELINE ML FUNCTIONALITY TEST")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run all tests
    test_results = {}
    
    # Test 1: ML Dependencies
    test_results['ml_imports'] = test_ml_imports()
    
    # Test 2: Improved AI Trainer
    test_results['improved_ai_trainer'] = test_improved_ai_trainer()
    
    # Test 3: Foundation Trainer
    test_results['foundation_trainer'] = test_foundation_trainer()
    
    # Test 4: Adaptive Reviewer
    test_results['adaptive_reviewer'] = test_adaptive_reviewer()
    
    # Test 5: As-Built Processor
    test_results['as_built_processor'] = test_as_built_processor()
    
    # Test 6: Performance Metrics
    test_results['performance'] = test_performance_metrics()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = result.get('status', 'unknown')
        if status == 'success':
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            if 'error' in result:
                print(f"   Error: {result['error']}")
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    
    # Save detailed results
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: test_results.json")
    print(f"📝 Log file: test_as_built_ml.log")
    
    if passed == total:
        print("\n🎉 All tests passed! The ML functionality is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
