# Cable Entity Matching Pipeline Documentation

## Overview

The Cable Entity Matching Pipeline is designed to identify and highlight cable references in engineering documents. It processes PDF plan sheets and CSV data to generate color-coded visualizations and detailed reports.

## Features

### Core Features

- **CSV Cable Data Extraction**: Identifies cable types and associated metadata from CSV files
- **Intelligent Pattern Matching**: Finds exact matches, variations, and "new" cable references
- **Station Callout Integration**: Associates cables with their station locations
- **Visual Highlighting**: Customizable border width and semi-transparent fill colors
- **Comprehensive Reporting**: Generates detailed JSON reports with match statistics

### Advanced Capabilities

- **"New" Cable Identification**: Detects references to new cable installations (after the word "NEW")
- **Station Callout Detection**: Extracts and links station values to specific cables
- **Variation Handling**: Accounts for hyphenation, spacing differences, and case sensitivity
- **Pipeline Architecture**: Modular design with stages for extraction, matching, highlighting and reporting

## Technical Architecture

The system is built using a modular pipeline architecture:

1. **Data Validation Stage**: Validates input files and formats
2. **CSV Extraction Stage**: Extracts cable types and metadata from CSV
3. **Variation Generation Stage**: Creates variations of entity names for better matching
4. **PDF Search Stage**: Searches for entities and variations in PDF documents
5. **PDF Highlighting Stage**: Highlights matches with customizable styling
6. **Report Generation Stage**: Creates JSON report with detailed analysis

## Usage

### Basic Usage

```bash
python run_cable_matcher.py --csv <csv_file> --pdf <pdf_file>
```

### Advanced Options

```bash
python run_cable_matcher.py --csv <csv_file> --pdf <pdf_file> \
  --border-width 0.8 \
  --fill-opacity 0.2 \
  --output-dir ./custom_output \
  --verbose
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--csv` | string | (required) | Path to CSV file with cable information |
| `--pdf` | string | (required) | Path to PDF document to analyze |
| `--output-dir` | string | "./output" | Directory to save outputs |
| `--border-width` | float | 1.0 | Width of highlight borders |
| `--fill-opacity` | float | 0.15 | Opacity of highlight fills (0-1) |
| `--verbose` | flag | false | Enable detailed logging |

## Output Files

The pipeline generates the following output files:

1. **Highlighted PDF**: A copy of the input PDF with color-coded highlights for each cable entity
2. **JSON Report**: A detailed report with match statistics and location information
3. **Log File**: A detailed log of the pipeline execution (when using --verbose)

Each output file includes a timestamp to prevent overwriting previous results.

## CSV Format

The system expects CSV files with cable information. The following columns are recognized:

- **Cable Type** (required): The name of the cable (e.g., "FTC Distribution")
- **Station/STA**: Station information associated with the cable (e.g., "STA 123+45.67")
- Other columns are preserved in the report but not used for matching

## Implementation Details

### "New" Cable Detection

The system identifies references to new cable installations by looking for patterns like:

- "new <cable_name>"
- "NEW <cable_name>"
- "New <cable_name>"

These are highlighted distinctively and tracked separately in the report.

### Station Callout Extraction

Station values are extracted using regular expressions that identify patterns like:

- "STA 123+45.67"
- "Station 123+45"
- "123+45.67"

### Highlight Styling

- **Border**: Customizable width (default: 1.0)
- **Fill**: Semi-transparent color (default opacity: 0.15)
- **"New" cable references**: Slightly thicker border and higher opacity

## Future Enhancements

- **Batch Processing**: Support for processing multiple PDFs at once
- **Interactive Mode**: GUI for real-time exploration of matches
- **Machine Learning**: Improve matching accuracy using ML techniques
- **Database Integration**: Store and query match results in a database
- **Web Interface**: Develop a web-based interface for easier access
