import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.document_entity_matcher import (
    DataValidator, 
    EntityExtractor,
    EntityMatcher,
    OutputGenerator
)

def test_document_entity_matcher():
    """Test the document entity matcher with sample data."""
    print("=== Testing Engineering Document Entity Matcher ===")
    
    # Paths to test files
    tables_dir = Path(__file__).parent / "Tables"
    csv_path = tables_dir / "WIM_Equipment.csv"
    pdf_path = tables_dir / "M01-17-ITS-Tolling-2b.pdf"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"CSV Path: {csv_path}")
    print(f"PDF Path: {pdf_path}")
    
    # 1. Validate input files
    print("\n1. Validating input files...")
    data_validator = DataValidator()
    
    if not csv_path.exists() or not pdf_path.exists():
        print("Error: Test files not found!")
        return
    
    csv_valid = data_validator.validate_csv_structure(csv_path, ["Cable Type"])
    pdf_valid = data_validator.validate_pdf(pdf_path)
    
    print(f"CSV Valid: {csv_valid}")
    print(f"PDF Valid: {pdf_valid}")
    
    if not csv_valid or not pdf_valid:
        print("Error: Input validation failed!")
        return
    
    # 2. Extract entities from CSV
    print("\n2. Extracting entities from CSV...")
    entity_extractor = EntityExtractor()
    entities = entity_extractor.extract_cable_types_from_csv(csv_path)
    
    print(f"Found {len(entities)} entities in CSV")
    for i, entity in enumerate(entities[:5]):
        print(f"  - {entity}")
    if len(entities) > 5:
        print(f"  - ... and {len(entities) - 5} more")
    
    # 3. Generate variations for fuzzy matching
    print("\n3. Generating entity variations for matching...")
    entity_variations = {}
    for entity in entities:
        variations = entity_extractor.generate_entity_variations(entity)
        entity_variations[entity] = variations
    
    print(f"Generated variations for {len(entity_variations)} entities")
    sample_entity = next(iter(entity_variations))
    print(f"Sample variations for '{sample_entity}':")
    for var in entity_variations[sample_entity][:3]:
        print(f"  - {var}")
    if len(entity_variations[sample_entity]) > 3:
        print(f"  - ... and {len(entity_variations[sample_entity]) - 3} more")
    
    # 4. Match entities in PDF
    print("\n4. Matching entities in PDF...")
    entity_matcher = EntityMatcher(fuzzy_match_threshold=0.8)
    match_results = entity_matcher.find_entities_in_pdf(pdf_path, entity_variations)
    
    found_count = sum(1 for data in match_results.values() if data["count"] > 0)
    total_matches = sum(data["count"] for data in match_results.values())
    print(f"Found {found_count}/{len(entities)} entities with {total_matches} total matches")
    
    # 5. Generate outputs
    print("\n5. Generating highlighted PDF and summary report...")
    output_generator = OutputGenerator()
    
    output_pdf = output_dir / "test_highlighted.pdf"
    output_report = output_dir / "test_match_report.json"
    
    output_generator.highlight_entities_in_pdf(
        pdf_path,
        entity_variations,
        match_results,
        output_pdf
    )
    
    output_generator.generate_summary_report(
        match_results,
        entity_matcher.match_metrics,
        output_report
    )
    
    print("\n=== Test Complete ===")
    print(f"Highlighted PDF saved to: {output_pdf}")
    print(f"Match report saved to: {output_report}")

if __name__ == "__main__":
    test_document_entity_matcher()
