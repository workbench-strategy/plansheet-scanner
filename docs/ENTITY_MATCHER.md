# Engineering Document Entity Matcher

This tool automates the process of finding and highlighting specific entities (such as cable types, equipment IDs, etc.) across engineering documents. It uses fuzzy matching to account for inconsistencies in how entities are named or referred to in documentation.

## Features

- **Multi-format Data Parsing**: Parse and validate data from CSV, XML, and PDF sources
- **Entity Matching**: Match entities such as cable types, equipment IDs, and stationing values across documents
- **Advanced Matching Algorithms**: Implements fuzzy matching, regex extraction, and context-aware entity recognition
- **Output Generation**: Creates annotated PDFs with highlighted entities and comprehensive summary reports
- **Error Handling**: Robust error handling for edge cases like missing data or duplicate entity names
- **Metrics & Reporting**: Detailed logging and reporting for match accuracy, coverage, and performance

## Installation

```bash
# Clone the repository
git clone https://github.com/workbench-strategy/plansheet-scanner.git
cd plansheet-scanner

# Install requirements
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Basic usage
python src/cli/entity_matcher_cli.py --csv path/to/entities.csv --pdf path/to/document.pdf

# With additional options
python src/cli/entity_matcher_cli.py --csv Tables/WIM_Equipment.csv --pdf Tables/M01-17-ITS-Tolling-2b.pdf --output-dir ./output --fuzzy-threshold 0.85 --verbose
```

### Options

- `--csv`: Path to CSV file containing entities (e.g., cable types)
- `--pdf`: Path to PDF document to analyze
- `--output-dir`: Directory to save output files (default: ./output)
- `--fuzzy-threshold`: Threshold for fuzzy matching (0-1, default: 0.8)
- `--verbose`: Enable verbose logging

## Example Workflow

1. **Prepare CSV Input**: Create a CSV file with your entities (e.g., cable types, equipment IDs)
2. **Run the Tool**: Execute the CLI tool, pointing to your CSV and PDF files
3. **Review Outputs**:
   - `*_highlighted.pdf`: A copy of your original PDF with entities highlighted in different colors
   - `*_match_report.json`: A detailed report of all matches found, including statistics and locations

## Advanced Usage

### Integration with Other Tools

The document entity matcher is designed to be integrated with other tools in your engineering workflow:

```python
from src.core.document_entity_matcher import EntityExtractor, EntityMatcher, OutputGenerator

# Extract entities from your data source
extractor = EntityExtractor()
entities = extractor.extract_cable_types_from_csv("your_data.csv")

# Match entities in your document
matcher = EntityMatcher()
results = matcher.find_entities_in_pdf("your_document.pdf", entities)

# Generate outputs
generator = OutputGenerator()
generator.highlight_entities_in_pdf("your_document.pdf", entities, results, "highlighted.pdf")
generator.generate_summary_report(results, matcher.match_metrics, "report.json")
```

## Compliance and Standards

This tool is designed to comply with internal engineering standards and support reproducibility across infrastructure projects. It maintains detailed logs of all operations and provides comprehensive reports that can be used for auditing and quality assurance.

## Requirements

- Python 3.7+
- PyMuPDF (fitz)
- pandas
- Other dependencies listed in requirements.txt

## License

See [LICENSE](LICENSE) for details.
