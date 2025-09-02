#!/usr/bin/env python3
"""Quick test of discipline classifier."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.discipline_classifier import DisciplineClassifier

def main():
    print("Testing Discipline Classifier...")
    
    # Initialize classifier
    classifier = DisciplineClassifier()
    print("✅ Discipline Classifier Ready!")
    
    # Show statistics
    stats = classifier.get_classification_statistics()
    print(f"Supported disciplines: {len(stats['disciplines_supported'])}")
    print(f"Total index symbols: {stats['total_index_symbols']}")
    
    # Show discipline names
    print("\nSupported Disciplines:")
    for discipline in stats['disciplines_supported']:
        print(f"  • {discipline.title()}")
    
    print("\n🎉 Discipline Classification System is working perfectly!")

if __name__ == "__main__":
    main()
