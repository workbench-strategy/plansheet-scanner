#!/usr/bin/env python3
"""
Training Requirements Analysis for Plansheet Symbol Recognition
Analyzes how many plansheets are needed to train models for comprehensive symbol recognition.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class SymbolCategory:
    """Represents a category of symbols in plansheets."""
    name: str
    discipline: str
    common_variations: int
    complexity: str  # low, medium, high
    training_examples_needed: int

@dataclass
class DisciplineRequirements:
    """Training requirements for a specific discipline."""
    discipline: str
    total_symbols: int
    unique_symbol_types: int
    min_plansheets: int
    recommended_plansheets: int
    symbol_categories: List[SymbolCategory]

class TrainingRequirementsAnalyzer:
    """Analyzes training requirements for comprehensive plansheet symbol recognition."""
    
    def __init__(self):
        # Define symbol categories by discipline
        self.discipline_symbols = {
            "traffic": {
                "signal_heads": {"variations": 8, "complexity": "medium"},
                "detector_loops": {"variations": 6, "complexity": "low"},
                "signs": {"variations": 25, "complexity": "high"},
                "markings": {"variations": 15, "complexity": "medium"},
                "pedestrian_facilities": {"variations": 10, "complexity": "medium"},
                "traffic_control_devices": {"variations": 12, "complexity": "high"}
            },
            "electrical": {
                "conduit": {"variations": 8, "complexity": "low"},
                "junction_boxes": {"variations": 6, "complexity": "medium"},
                "transformers": {"variations": 4, "complexity": "high"},
                "power_poles": {"variations": 5, "complexity": "medium"},
                "electrical_cables": {"variations": 10, "complexity": "medium"},
                "grounding_systems": {"variations": 7, "complexity": "high"}
            },
            "structural": {
                "reinforcement": {"variations": 12, "complexity": "high"},
                "concrete_elements": {"variations": 8, "complexity": "medium"},
                "steel_elements": {"variations": 10, "complexity": "high"},
                "foundations": {"variations": 6, "complexity": "high"},
                "expansion_joints": {"variations": 4, "complexity": "medium"},
                "bridge_elements": {"variations": 15, "complexity": "high"}
            },
            "drainage": {
                "catch_basins": {"variations": 6, "complexity": "low"},
                "manholes": {"variations": 8, "complexity": "medium"},
                "culverts": {"variations": 5, "complexity": "medium"},
                "drainage_pipes": {"variations": 10, "complexity": "low"},
                "inlets": {"variations": 7, "complexity": "medium"},
                "erosion_control": {"variations": 8, "complexity": "medium"}
            },
            "mechanical": {
                "ductwork": {"variations": 8, "complexity": "medium"},
                "equipment": {"variations": 12, "complexity": "high"},
                "ventilation": {"variations": 6, "complexity": "medium"},
                "piping": {"variations": 10, "complexity": "medium"}
            },
            "landscape": {
                "trees": {"variations": 8, "complexity": "low"},
                "shrubs": {"variations": 6, "complexity": "low"},
                "irrigation": {"variations": 7, "complexity": "medium"},
                "hardscape": {"variations": 10, "complexity": "medium"}
            },
            "utilities": {
                "water_lines": {"variations": 6, "complexity": "medium"},
                "sewer_lines": {"variations": 6, "complexity": "medium"},
                "gas_lines": {"variations": 4, "complexity": "medium"},
                "telecommunications": {"variations": 8, "complexity": "medium"},
                "fiber_optics": {"variations": 5, "complexity": "medium"}
            }
        }
    
    def calculate_training_requirements(self) -> Dict[str, Any]:
        """Calculate comprehensive training requirements."""
        print("🔍 Analyzing Training Requirements for Plansheet Symbol Recognition")
        print("=" * 70)
        
        total_symbols = 0
        total_unique_symbols = 0
        discipline_requirements = []
        
        for discipline, symbols in self.discipline_symbols.items():
            discipline_total = sum(symbol["variations"] for symbol in symbols.values())
            unique_symbols = len(symbols)
            total_symbols += discipline_total
            total_unique_symbols += unique_symbols
            
            # Calculate training requirements for this discipline
            min_plansheets = self._calculate_min_plansheets(discipline, symbols)
            recommended_plansheets = self._calculate_recommended_plansheets(discipline, symbols)
            
            # Create symbol categories
            symbol_categories = []
            for symbol_name, symbol_info in symbols.items():
                training_examples = self._calculate_symbol_training_examples(
                    symbol_info["variations"], symbol_info["complexity"]
                )
                category = SymbolCategory(
                    name=symbol_name,
                    discipline=discipline,
                    common_variations=symbol_info["variations"],
                    complexity=symbol_info["complexity"],
                    training_examples_needed=training_examples
                )
                symbol_categories.append(category)
            
            discipline_req = DisciplineRequirements(
                discipline=discipline,
                total_symbols=discipline_total,
                unique_symbol_types=unique_symbols,
                min_plansheets=min_plansheets,
                recommended_plansheets=recommended_plansheets,
                symbol_categories=symbol_categories
            )
            discipline_requirements.append(discipline_req)
            
            print(f"\n📊 {discipline.upper()} DISCIPLINE:")
            print(f"   Total symbols: {discipline_total}")
            print(f"   Unique symbol types: {unique_symbols}")
            print(f"   Minimum plansheets needed: {min_plansheets}")
            print(f"   Recommended plansheets: {recommended_plansheets}")
            print(f"   Symbol categories:")
            for category in symbol_categories:
                print(f"     - {category.name}: {category.common_variations} variations, "
                      f"{category.complexity} complexity, {category.training_examples_needed} examples needed")
        
        # Calculate overall requirements
        overall_min_plansheets = max(req.min_plansheets for req in discipline_requirements)
        overall_recommended_plansheets = sum(req.recommended_plansheets for req in discipline_requirements)
        
        # Account for cross-discipline learning and symbol overlap
        cross_discipline_efficiency = 0.3  # 30% efficiency gain from cross-learning
        adjusted_min_plansheets = int(overall_min_plansheets * (1 - cross_discipline_efficiency))
        adjusted_recommended_plansheets = int(overall_recommended_plansheets * (1 - cross_discipline_efficiency))
        
        print(f"\n🎯 OVERALL TRAINING REQUIREMENTS:")
        print(f"   Total unique symbols across all disciplines: {total_unique_symbols}")
        print(f"   Total symbol variations: {total_symbols}")
        print(f"   Minimum plansheets needed (per discipline): {overall_min_plansheets}")
        print(f"   Recommended plansheets (per discipline): {overall_recommended_plansheets}")
        print(f"   Adjusted minimum (with cross-learning): {adjusted_min_plansheets}")
        print(f"   Adjusted recommended (with cross-learning): {adjusted_recommended_plansheets}")
        
        # Calculate dataset requirements
        dataset_requirements = self._calculate_dataset_requirements(
            total_symbols, total_unique_symbols, adjusted_recommended_plansheets
        )
        
        return {
            "total_symbols": total_symbols,
            "total_unique_symbols": total_unique_symbols,
            "discipline_requirements": discipline_requirements,
            "overall_min_plansheets": adjusted_min_plansheets,
            "overall_recommended_plansheets": adjusted_recommended_plansheets,
            "dataset_requirements": dataset_requirements
        }
    
    def _calculate_min_plansheets(self, discipline: str, symbols: Dict[str, Any]) -> int:
        """Calculate minimum plansheets needed for a discipline."""
        total_variations = sum(symbol["variations"] for symbol in symbols.values())
        avg_complexity = np.mean([self._complexity_to_numeric(symbol["complexity"]) 
                                for symbol in symbols.values()])
        
        # Base calculation: 3 examples per symbol variation for basic recognition
        base_requirement = total_variations * 3
        
        # Adjust for complexity
        complexity_factor = 1 + (avg_complexity - 1) * 0.5
        
        # Minimum 5 plansheets per discipline for statistical validity
        min_plansheets = max(5, int(base_requirement * complexity_factor / 10))
        
        return min_plansheets
    
    def _calculate_recommended_plansheets(self, discipline: str, symbols: Dict[str, Any]) -> int:
        """Calculate recommended plansheets for robust training."""
        min_plansheets = self._calculate_min_plansheets(discipline, symbols)
        
        # Recommended is 3x minimum for robust training
        recommended = min_plansheets * 3
        
        # Cap at reasonable maximum
        return min(recommended, 50)
    
    def _calculate_symbol_training_examples(self, variations: int, complexity: str) -> int:
        """Calculate training examples needed for a symbol category."""
        complexity_factor = self._complexity_to_numeric(complexity)
        
        # Base: 5 examples per variation
        base_examples = variations * 5
        
        # Adjust for complexity
        adjusted_examples = int(base_examples * complexity_factor)
        
        # Minimum 10 examples, maximum 200
        return max(10, min(200, adjusted_examples))
    
    def _complexity_to_numeric(self, complexity: str) -> float:
        """Convert complexity string to numeric factor."""
        complexity_map = {
            "low": 1.0,
            "medium": 1.5,
            "high": 2.0
        }
        return complexity_map.get(complexity, 1.0)
    
    def _calculate_dataset_requirements(self, total_symbols: int, unique_symbols: int, 
                                      recommended_plansheets: int) -> Dict[str, Any]:
        """Calculate dataset requirements for comprehensive training."""
        
        # Symbol recognition model requirements
        symbol_recognition_examples = unique_symbols * 50  # 50 examples per unique symbol
        
        # Legend recognition model requirements
        legend_recognition_examples = recommended_plansheets * 3  # 3 legend examples per plansheet
        
        # Cross-discipline learning examples
        cross_discipline_examples = unique_symbols * 20  # 20 cross-discipline examples per symbol
        
        # Total dataset size
        total_dataset_size = symbol_recognition_examples + legend_recognition_examples + cross_discipline_examples
        
        # Quality requirements
        quality_requirements = {
            "high_quality_annotations": int(total_dataset_size * 0.7),  # 70% high quality
            "medium_quality_annotations": int(total_dataset_size * 0.2),  # 20% medium quality
            "low_quality_annotations": int(total_dataset_size * 0.1),   # 10% low quality
        }
        
        return {
            "symbol_recognition_examples": symbol_recognition_examples,
            "legend_recognition_examples": legend_recognition_examples,
            "cross_discipline_examples": cross_discipline_examples,
            "total_dataset_size": total_dataset_size,
            "quality_requirements": quality_requirements
        }
    
    def generate_training_plan(self, available_plansheets: int) -> Dict[str, Any]:
        """Generate a training plan based on available plansheets."""
        requirements = self.calculate_training_requirements()
        
        print(f"\n📋 TRAINING PLAN FOR {available_plansheets} AVAILABLE PLANSHEETS:")
        print("=" * 70)
        
        if available_plansheets < requirements["overall_min_plansheets"]:
            print(f"⚠️  WARNING: Only {available_plansheets} plansheets available, "
                  f"but {requirements['overall_min_plansheets']} minimum needed")
            print("   Consider:")
            print("   - Starting with high-priority disciplines")
            print("   - Using data augmentation techniques")
            print("   - Collecting more plansheets")
        
        # Calculate coverage
        coverage_percentage = (available_plansheets / requirements["overall_recommended_plansheets"]) * 100
        
        # Prioritize disciplines
        discipline_priority = self._prioritize_disciplines(requirements["discipline_requirements"])
        
        # Allocate plansheets
        allocation = self._allocate_plansheets(available_plansheets, discipline_priority)
        
        training_plan = {
            "available_plansheets": available_plansheets,
            "coverage_percentage": coverage_percentage,
            "discipline_allocation": allocation,
            "training_phases": self._generate_training_phases(allocation),
            "recommendations": self._generate_recommendations(available_plansheets, requirements)
        }
        
        print(f"   Coverage: {coverage_percentage:.1f}% of recommended dataset")
        print(f"   Discipline allocation:")
        for discipline, count in allocation.items():
            print(f"     - {discipline}: {count} plansheets")
        
        return training_plan
    
    def _prioritize_disciplines(self, discipline_requirements: List[DisciplineRequirements]) -> List[str]:
        """Prioritize disciplines based on complexity and symbol diversity."""
        # Sort by total symbols (more symbols = higher priority)
        sorted_disciplines = sorted(discipline_requirements, 
                                  key=lambda x: x.total_symbols, reverse=True)
        return [d.discipline for d in sorted_disciplines]
    
    def _allocate_plansheets(self, available: int, priority_order: List[str]) -> Dict[str, int]:
        """Allocate available plansheets across disciplines."""
        allocation = {}
        remaining = available
        
        # Allocate based on priority and complexity
        for discipline in priority_order:
            if remaining <= 0:
                break
            
            # Allocate 20-30% of remaining plansheets to each discipline
            allocated = max(5, min(remaining // 4, 20))  # At least 5, at most 20 per discipline
            allocation[discipline] = allocated
            remaining -= allocated
        
        return allocation
    
    def _generate_training_phases(self, allocation: Dict[str, int]) -> List[Dict[str, Any]]:
        """Generate training phases based on allocation."""
        phases = []
        
        # Phase 1: Core disciplines (traffic, electrical, structural)
        core_disciplines = ["traffic", "electrical", "structural"]
        phase1_plansheets = sum(allocation.get(d, 0) for d in core_disciplines)
        
        phases.append({
            "phase": 1,
            "name": "Core Disciplines",
            "disciplines": core_disciplines,
            "plansheets": phase1_plansheets,
            "focus": "Basic symbol recognition for major disciplines"
        })
        
        # Phase 2: Supporting disciplines
        supporting_disciplines = ["drainage", "mechanical", "utilities"]
        phase2_plansheets = sum(allocation.get(d, 0) for d in supporting_disciplines)
        
        if phase2_plansheets > 0:
            phases.append({
                "phase": 2,
                "name": "Supporting Disciplines",
                "disciplines": supporting_disciplines,
                "plansheets": phase2_plansheets,
                "focus": "Expand to supporting disciplines"
            })
        
        # Phase 3: Specialized disciplines
        specialized_disciplines = ["landscape"]
        phase3_plansheets = sum(allocation.get(d, 0) for d in specialized_disciplines)
        
        if phase3_plansheets > 0:
            phases.append({
                "phase": 3,
                "name": "Specialized Disciplines",
                "disciplines": specialized_disciplines,
                "plansheets": phase3_plansheets,
                "focus": "Specialized symbol recognition"
            })
        
        return phases
    
    def _generate_recommendations(self, available: int, requirements: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on available plansheets."""
        recommendations = []
        
        if available < requirements["overall_min_plansheets"]:
            recommendations.append("Start with high-priority disciplines (traffic, electrical, structural)")
            recommendations.append("Use data augmentation to increase effective dataset size")
            recommendations.append("Focus on symbol quality over quantity")
            recommendations.append("Consider collecting more plansheets from completed projects")
        
        elif available < requirements["overall_recommended_plansheets"]:
            recommendations.append("Good starting point - focus on symbol diversity")
            recommendations.append("Ensure good coverage across all major disciplines")
            recommendations.append("Use transfer learning from pre-trained models")
            recommendations.append("Implement active learning to identify gaps")
        
        else:
            recommendations.append("Excellent dataset size - focus on quality and annotation consistency")
            recommendations.append("Consider fine-tuning for specific project types")
            recommendations.append("Implement continuous learning pipeline")
            recommendations.append("Explore advanced techniques like few-shot learning")
        
        recommendations.append("Use cross-validation to ensure robust model performance")
        recommendations.append("Implement quality control for symbol annotations")
        recommendations.append("Consider ensemble methods for improved accuracy")
        
        return recommendations

def main():
    """Main function to demonstrate training requirements analysis."""
    analyzer = TrainingRequirementsAnalyzer()
    
    # Calculate comprehensive requirements
    requirements = analyzer.calculate_training_requirements()
    
    # Generate training plans for different scenarios
    scenarios = [50, 100, 200, 500, 1000]
    
    print(f"\n🎯 TRAINING PLANS FOR DIFFERENT SCENARIOS:")
    print("=" * 70)
    
    for plansheets in scenarios:
        print(f"\n📊 SCENARIO: {plansheets} plansheets available")
        print("-" * 50)
        training_plan = analyzer.generate_training_plan(plansheets)
        
        print(f"   Training phases:")
        for phase in training_plan["training_phases"]:
            print(f"     Phase {phase['phase']}: {phase['name']} "
                  f"({phase['plansheets']} plansheets)")
        
        print(f"   Key recommendations:")
        for i, rec in enumerate(training_plan["recommendations"][:3], 1):
            print(f"     {i}. {rec}")

if __name__ == "__main__":
    main()
