"""
Cable Entity Matching Pipeline

This module implements a pipeline architecture for matching cable entities in engineering documents.
It supports:
1. CSV data extraction with full metadata
2. Enhanced entity matching with new cable identification
3. Station callout association
4. Improved visualization with semi-transparent highlights
"""

import os
import csv
import json
import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import random
import fitz  # PyMuPDF
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cable_matcher_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CableMatcherPipeline")

# Pipeline base classes
class PipelineStage:
    """Base class for pipeline stages"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data and pass to next stage
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dict[str, Any]: Processed data dictionary
        """
        raise NotImplementedError("Subclasses must implement process()")
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make the stage callable"""
        return self.process(data)

class Pipeline:
    """A pipeline that processes data through multiple stages"""
    
    def __init__(self, stages: List[PipelineStage]):
        """
        Initialize the pipeline with a list of stages
        
        Args:
            stages: List of pipeline stages to execute in sequence
        """
        self.stages = stages
        
    def run(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the pipeline
        
        Args:
            initial_data: Initial data dictionary
            
        Returns:
            Dict[str, Any]: Final processed data
        """
        data = initial_data
        
        for i, stage in enumerate(self.stages):
            logger.info(f"Running pipeline stage {i+1}/{len(self.stages)}: {stage.__class__.__name__}")
            try:
                data = stage(data)
            except Exception as e:
                logger.error(f"Error in pipeline stage {stage.__class__.__name__}: {str(e)}")
                raise
                
        return data

# Data validation stage
class DataValidationStage(PipelineStage):
    """Validates input data"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input paths and data
        
        Args:
            data: Input data with file paths
            
        Returns:
            Dict[str, Any]: Validated data
        """
        csv_path = data.get("csv_path")
        pdf_path = data.get("pdf_path")
        
        # Validate CSV
        if csv_path:
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # Check CSV content
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    raise ValueError(f"Empty CSV file: {csv_path}")
                
            # Check for Cable Type column
            has_cable_type = False
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if "Cable Type" in line:
                        has_cable_type = True
                        break
                        
            if not has_cable_type:
                raise ValueError("CSV file does not contain a 'Cable Type' column")
        else:
            raise ValueError("No CSV path provided")
            
        # Validate PDF
        if pdf_path:
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            # Check if PDF is readable
            try:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                if page_count == 0:
                    raise ValueError(f"PDF has no pages: {pdf_path}")
                    
                data["pdf_page_count"] = page_count
                doc.close()
            except Exception as e:
                raise ValueError(f"Error opening PDF: {str(e)}")
        else:
            raise ValueError("No PDF path provided")
            
        logger.info(f"Data validation completed: CSV and PDF files are valid")
        return data

# CSV extraction stage
class CSVExtractionStage(PipelineStage):
    """Extracts cable data from CSV"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract cable types and metadata from CSV
        
        Args:
            data: Input data with CSV path
            
        Returns:
            Dict[str, Any]: Data with extracted cable info
        """
        csv_path = data.get("csv_path")
        
        if not csv_path:
            raise ValueError("No CSV path provided")
            
        try:
            # Find the header row
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            header_row = -1
            for i, line in enumerate(lines):
                if "Cable Type" in line:
                    header_row = i
                    break
                    
            if header_row == -1:
                raise ValueError("Could not find 'Cable Type' column in CSV")
                
            # Read CSV with pandas starting from the header row
            df = pd.read_csv(csv_path, skiprows=range(header_row))
            
            # Process each cable type with its metadata
            cable_info = []
            
            for _, row in df.iterrows():
                if "Cable Type" in df.columns and pd.notna(row["Cable Type"]):
                    cable_data = {"name": row["Cable Type"].strip()}
                    
                    # Add other columns
                    for col in df.columns:
                        if col != "Cable Type" and pd.notna(row[col]):
                            # Convert column name to snake_case
                            key = col.lower().replace(" ", "_")
                            cable_data[key] = row[col]
                            
                            # Extract station values if present
                            if "station" in key.lower() or "sta" in key.lower():
                                stations = self._extract_stations(str(row[col]))
                                if stations:
                                    cable_data["station_values"] = stations
                                
                                # Extract station ranges
                                station_ranges = self._extract_station_ranges(str(row[col]))
                                if station_ranges:
                                    cable_data["station_ranges"] = station_ranges
                                    
                    cable_info.append(cable_data)
            
            logger.info(f"Extracted {len(cable_info)} cable records from CSV")
            data["cable_info"] = cable_info
            
            # Also create a simple list of cable names for backward compatibility
            data["cable_names"] = [cable["name"] for cable in cable_info]
            
        except Exception as e:
            logger.error(f"Error extracting data from CSV: {str(e)}")
            raise
            
        return data
        
    def _extract_stations(self, text: str) -> List[str]:
        """Extract station values from text"""
        station_patterns = [
            r'STA\s+(\d+\+\d+(?:\.\d+)?)',
            r'Station\s+(\d+\+\d+(?:\.\d+)?)',
            r'(\d+\+\d+(?:\.\d+)?)'
        ]
        
        results = []
        for pattern in station_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                station = match.group(1) if len(match.groups()) > 0 else match.group(0)
                results.append(station)
                
        return results
        
    def _extract_station_ranges(self, text: str) -> List[Tuple[float, float]]:
        """Extract station ranges from text
        
        Returns a list of tuples (start_station, end_station) as float values
        """
        # Extract individual stations first
        stations = self._extract_stations(text)
        
        # Handle range patterns
        range_patterns = [
            r'STA\s+(\d+\+\d+(?:\.\d+)?)\s+to\s+(\d+\+\d+(?:\.\d+)?)',
            r'Station\s+(\d+\+\d+(?:\.\d+)?)\s+to\s+(\d+\+\d+(?:\.\d+)?)',
            r'From\s+(\d+\+\d+(?:\.\d+)?)\s+to\s+(\d+\+\d+(?:\.\d+)?)',
            r'(\d+\+\d+(?:\.\d+)?)\s+to\s+(\d+\+\d+(?:\.\d+)?)'
        ]
        
        ranges = []
        for pattern in range_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = self._station_to_float(match.group(1))
                end = self._station_to_float(match.group(2))
                ranges.append((start, end))
        
        # If we have exactly two stations but no explicit range, treat as a range
        if len(stations) == 2 and not ranges:
            start = self._station_to_float(stations[0])
            end = self._station_to_float(stations[1])
            ranges.append((start, end))
        
        return ranges
        
    def _station_to_float(self, station: str) -> float:
        """Convert station string (e.g., '123+45.67') to float value"""
        parts = station.split('+')
        if len(parts) != 2:
            return 0.0
        
        try:
            return float(parts[0]) * 100 + float(parts[1])
        except ValueError:
            return 0.0

# Entity variation generation stage
class VariationGenerationStage(PipelineStage):
    """Generates variations of entity names"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate variations for each cable name
        
        Args:
            data: Input data with cable info
            
        Returns:
            Dict[str, Any]: Data with variations added
        """
        cable_info = data.get("cable_info", [])
        
        if not cable_info:
            logger.warning("No cable info found to generate variations")
            return data
            
        # Generate variations for each cable
        variations_map = {}
        all_variations = {}
        
        for cable in cable_info:
            cable_name = cable["name"]
            variations = self._generate_variations(cable_name)
            variations_map[cable_name] = variations
            
            # Map each variation back to the original
            for var in variations:
                all_variations[var] = cable_name
                
        logger.info(f"Generated variations for {len(cable_info)} cables")
        data["variations_map"] = variations_map
        data["all_variations"] = all_variations
        
        return data
        
    def _generate_variations(self, entity: str) -> List[str]:
        """Generate variations of entity names"""
        variations = [entity]  # Original name
        
        # Handle FTC variations
        if "FTC" in entity:
            parts = entity.split()
            if len(parts) >= 2:
                if parts[0] == "FTC":
                    # Add "Distribution FTC" variation
                    variations.append(f"{' '.join(parts[1:])} FTC")
                elif parts[-1] == "FTC":
                    # Already in "Distribution FTC" format, add the reverse
                    variations.append(f"FTC {' '.join(parts[:-1])}")
        
        # Handle other variations like SR 167, I-5, etc.
        if "SR 167" in entity:
            variations.append(entity.replace("SR 167", "SR-167"))
            variations.append(entity.replace("SR 167", "SR167"))
        
        if "I-5" in entity:
            variations.append(entity.replace("I-5", "I5"))
            variations.append(entity.replace("I-5", "Interstate 5"))
        
        # Add minor variations for distribution, mainline, etc.
        if "Distribution" in entity:
            variations.append(entity.replace("Distribution", "Dist"))
            variations.append(entity.replace("Distribution", "Dist."))
        
        if "Mainline" in entity:
            variations.append(entity.replace("Mainline", "Main"))
            variations.append(entity.replace("Mainline", "Main Line"))
        
        # Add uppercase and lowercase variations
        variations.append(entity.upper())
        variations.append(entity.lower())
        
        return variations

# PDF search stage
class PDFSearchStage(PipelineStage):
    """Searches for entities in PDF documents"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search PDF for entities and their variations
        
        Args:
            data: Input data with PDF path and variations
            
        Returns:
            Dict[str, Any]: Data with search results
        """
        pdf_path = data.get("pdf_path")
        variations_map = data.get("variations_map", {})
        all_variations = data.get("all_variations", {})
        
        if not pdf_path or not variations_map:
            logger.warning("Missing PDF path or variations for search")
            return data
            
        # Open the PDF
        doc = fitz.open(pdf_path)
        logger.info(f"Searching for {len(variations_map)} entities in PDF: {pdf_path}")
        
        # Initialize results structure
        search_results = {}
        for original in variations_map.keys():
            search_results[original] = {
                "count": 0,
                "pages": set(),
                "variations": {},
                "locations": [],
                "new_instances": [],
                "station_references": []
            }
            
        # Keep track of highlighted positions to avoid duplicates
        highlighted_positions = set()
        
        # Keep track of metrics
        metrics = {
            "exact_matches": 0,
            "variation_matches": 0,
            "new_cable_matches": 0,
            "station_callouts": 0,
            "duplicate_avoidance": 0
        }
        
        # Search through PDF pages
        for page_idx, page in enumerate(doc):
            page_text = page.get_text()
            logger.debug(f"Scanning page {page_idx+1}")
            
            # First, look for "new" cable references
            self._find_new_cable_references(
                page_text, 
                page_idx, 
                variations_map, 
                search_results, 
                metrics
            )
            
            # Search for each entity and its variations
            for original, variations in variations_map.items():
                for variation in variations:
                    text_instances = page.search_for(variation)
                    
                    if not text_instances:
                        continue
                        
                    # Process matches
                    for inst in text_instances:
                        # Create a unique identifier for this position to avoid duplicates
                        position_key = (
                            round(inst[0], 2), round(inst[1], 2),
                            round(inst[2], 2), round(inst[3], 2),
                            page_idx
                        )
                        
                        if position_key in highlighted_positions:
                            metrics["duplicate_avoidance"] += 1
                            continue
                            
                        highlighted_positions.add(position_key)
                        
                        # Update statistics
                        search_results[original]["count"] += 1
                        search_results[original]["pages"].add(page_idx + 1)
                        
                        if variation not in search_results[original]["variations"]:
                            search_results[original]["variations"][variation] = 0
                        search_results[original]["variations"][variation] += 1
                        
                        # Store location
                        surrounding_text = self._get_surrounding_text(page_text, variation)
                        location_info = {
                            "page": page_idx + 1,
                            "rect": [inst[0], inst[1], inst[2], inst[3]],
                            "variation": variation,
                            "surrounding_text": surrounding_text
                        }
                        
                        # Look for station values near this match
                        station_values = self._extract_stations_from_text(surrounding_text)
                        if station_values:
                            location_info["station_values"] = station_values
                            search_results[original]["station_references"].extend(station_values)
                            metrics["station_callouts"] += len(station_values)
                            
                            # Also extract station ranges if possible
                            station_ranges = self._extract_station_ranges(surrounding_text)
                            if station_ranges:
                                location_info["station_ranges"] = [
                                    {
                                        "start": start,
                                        "end": end,
                                        "start_station": f"{int(start // 100)}+{start % 100:.2f}".rstrip('0').rstrip('.'),
                                        "end_station": f"{int(end // 100)}+{end % 100:.2f}".rstrip('0').rstrip('.')
                                    }
                                    for start, end in station_ranges
                                ]
                            
                        search_results[original]["locations"].append(location_info)
                        
                        # Update metrics
                        if variation == original:
                            metrics["exact_matches"] += 1
                        else:
                            metrics["variation_matches"] += 1
                            
        # Close the document
        doc.close()
        
        # Convert page sets to sorted lists for better output
        for entity in search_results:
            search_results[entity]["pages"] = sorted(list(search_results[entity]["pages"]))
            # Remove duplicates from station references
            search_results[entity]["station_references"] = list(set(search_results[entity]["station_references"]))
            
        logger.info(f"Search completed with {sum(r['count'] for r in search_results.values())} total matches")
        data["search_results"] = search_results
        data["search_metrics"] = metrics
        
        return data
        
    def _find_new_cable_references(self, text: str, page_idx: int, variations_map: Dict[str, List[str]], 
                                search_results: Dict[str, Dict], metrics: Dict[str, int]) -> None:
        """Find references to new cables"""
        # For each original cable name and its variations
        for original, variations in variations_map.items():
            # Join all variations with OR for the regex
            variation_pattern = '|'.join(map(re.escape, variations))
            
            # Create patterns for "new" cable references
            new_patterns = [
                r'new\s+(?:\w+\s+){0,3}(' + variation_pattern + r')',  # "new <up to 3 words> <cable_type>"
                r'NEW\s+(?:\w+\s+){0,3}(' + variation_pattern + r')',  # Same but uppercase
                r'New\s+(?:\w+\s+){0,3}(' + variation_pattern + r')'   # Same with capitalized first letter
            ]
            
            # Try each pattern
            for pattern in new_patterns:
                
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Extract the matched text and surrounding context
                    full_match = match.group(0)
                    match_start, match_end = match.span(0)
                    
                    # Get surrounding text for context
                    start_pos = max(0, match_start - 50)
                    end_pos = min(len(text), match_end + 50)
                    context = text[start_pos:end_pos]
                    
                    # Record this as a new cable reference
                    new_ref = {
                        "text": full_match,
                        "context": context,
                        "page": page_idx + 1
                    }
                    
                    search_results[original]["new_instances"].append(new_ref)
                    metrics["new_cable_matches"] += 1
                    
    def _get_surrounding_text(self, text: str, search_term: str) -> str:
        """Get text surrounding a search term"""
        # Find the position of the search term
        pos = text.find(search_term)
        if pos == -1:
            return ""
            
        # Get 50 characters before and after
        start = max(0, pos - 50)
        end = min(len(text), pos + len(search_term) + 50)
        
        return text[start:end]
        
    def _extract_stations_from_text(self, text: str) -> List[str]:
        """Extract station values from text"""
        station_patterns = [
            r'STA\s+(\d+\+\d+(?:\.\d+)?)',
            r'Station\s+(\d+\+\d+(?:\.\d+)?)',
            r'(\d+\+\d+(?:\.\d+)?)'
        ]
        
        results = []
        for pattern in station_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                station = match.group(1) if len(match.groups()) > 0 else match.group(0)
                results.append(station)
                
        return results
        
    def _extract_station_ranges(self, text: str) -> List[Tuple[float, float]]:
        """Extract station ranges from text
        
        Returns a list of tuples (start_station, end_station) as float values
        """
        # Extract individual stations first
        stations = self._extract_stations_from_text(text)
        
        # Handle range patterns
        range_patterns = [
            r'STA\s+(\d+\+\d+(?:\.\d+)?)\s+to\s+(\d+\+\d+(?:\.\d+)?)',
            r'Station\s+(\d+\+\d+(?:\.\d+)?)\s+to\s+(\d+\+\d+(?:\.\d+)?)',
            r'From\s+(\d+\+\d+(?:\.\d+)?)\s+to\s+(\d+\+\d+(?:\.\d+)?)',
            r'(\d+\+\d+(?:\.\d+)?)\s+to\s+(\d+\+\d+(?:\.\d+)?)'
        ]
        
        ranges = []
        for pattern in range_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = self._station_to_float(match.group(1))
                end = self._station_to_float(match.group(2))
                ranges.append((start, end))
        
        # If we have exactly two stations but no explicit range, treat as a range
        if len(stations) == 2 and not ranges:
            start = self._station_to_float(stations[0])
            end = self._station_to_float(stations[1])
            ranges.append((start, end))
        
        return ranges
        
    def _station_to_float(self, station: str) -> float:
        """Convert station string (e.g., '123+45.67') to float value"""
        parts = station.split('+')
        if len(parts) != 2:
            return 0.0
        
        try:
            return float(parts[0]) * 100 + float(parts[1])
        except ValueError:
            return 0.0

# PDF highlighting stage
class PDFHighlightingStage(PipelineStage):
    """Highlights entities in PDF and generates output"""
    
    def __init__(self, border_width: float = 1.0, fill_opacity: float = 0.15):
        """
        Initialize with styling options
        
        Args:
            border_width: Width of highlight borders
            fill_opacity: Opacity of highlight fills (0-1)
        """
        self.border_width = border_width
        self.fill_opacity = fill_opacity
        
    def _station_to_float(self, station: str) -> float:
        """Convert station string (e.g., '123+45.67') to float value"""
        parts = station.split('+')
        if len(parts) != 2:
            return 0.0
        
        try:
            return float(parts[0]) * 100 + float(parts[1])
        except ValueError:
            return 0.0
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Highlight entities in PDF and save output
        
        Args:
            data: Input data with PDF path and search results
            
        Returns:
            Dict[str, Any]: Data with output paths added
        """
        pdf_path = data.get("pdf_path")
        search_results = data.get("search_results", {})
        output_dir = data.get("output_dir", Path("."))
        timestamp = data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        cable_info = data.get("cable_info", [])
        
        if not pdf_path or not search_results:
            logger.warning("Missing PDF path or search results for highlighting")
            return data
            
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create output filenames
        base_name = Path(pdf_path).stem
        output_pdf = output_dir / f"{base_name}_{timestamp}_highlighted.pdf"
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        logger.info(f"PDF has {len(doc)} pages")
        
        # Extract station ranges from cable info
        station_range_map = {}
        station_range_list = []
        
        for cable in cable_info:
            cable_name = cable["name"]
            if "station_ranges" in cable:
                ranges = cable["station_ranges"]
                for i, range_val in enumerate(ranges):
                    range_id = f"{cable_name}_range_{i}"
                    station_range_list.append((range_id, range_val))
                    
                    # Map range_id back to entity
                    if cable_name not in station_range_map:
                        station_range_map[cable_name] = []
                    station_range_map[cable_name].append((range_id, range_val))
        
        # Assign colors based on station ranges
        range_color_map = {}
        if station_range_list:
            range_color_map = self._assign_colors_by_station(station_range_list)
            
            # Create a legend for the range colors
            data["station_range_colors"] = {
                range_id: {
                    "entity": range_id.split("_range_")[0],
                    "range": (start, end),
                    "color": range_color_map[range_id]
                }
                for range_id, (start, end) in station_range_list
            }
        
        # Generate default colors for entities without station ranges
        entity_colors = self._generate_fiber_cable_colors(list(search_results.keys()))
        
        # Track highlighted positions to avoid duplicates
        highlighted_positions = set()
        
        # Store station-specific highlights for legend
        station_highlights = {}
        
        # Highlight entities in the document (skip legend page if it exists)
        for page_idx, page in enumerate(doc):
            # Skip the first page if it's a legend page
            if page_idx == 0:
                page_text = page.get_text().upper()
                if "LEGEND" in page_text or "ENTITY MATCH" in page_text:
                    logger.info(f"Skipping legend page (page {page_idx + 1})")
                    continue
            
            logger.debug(f"Processing page {page_idx + 1}")
            
            # Get the page text to look for station references
            page_text = page.get_text()
            page_stations = []
            for pattern in [r'STA\s+(\d+\+\d+(?:\.\d+)?)', r'Station\s+(\d+\+\d+(?:\.\d+)?)', r'(\d+\+\d+(?:\.\d+)?)']:
                for match in re.finditer(pattern, page_text, re.IGNORECASE):
                    station = match.group(1) if len(match.groups()) > 0 else match.group(0)
                    page_stations.append(self._station_to_float(station))
            
            for entity, result in search_results.items():
                if not result["locations"]:
                    continue
                
                # Find the appropriate color for this entity on this page
                color = None
                range_id = None
                
                # If we have station ranges for this entity, find which one applies to this page
                if entity in station_range_map and page_stations:
                    # Find which range the page stations belong to
                    for r_id, (start, end) in station_range_map[entity]:
                        for station in page_stations:
                            if start <= station <= end:
                                color = range_color_map[r_id]
                                range_id = r_id
                                
                                # Track this for the legend
                                if r_id not in station_highlights:
                                    station_highlights[r_id] = {
                                        "entity": entity,
                                        "range": (start, end),
                                        "color": color,
                                        "pages": set()
                                    }
                                station_highlights[r_id]["pages"].add(page_idx + 1)
                                break
                        
                        if color:  # Found a matching range
                            break
                
                # If no station range matched, use default color
                if not color:
                    color = entity_colors[entity]
                
                darker_color = self._darken_color(color)
                
                # Find matches on this page
                page_matches = [loc for loc in result["locations"] if loc["page"] == page_idx + 1]
                
                for match in page_matches:
                    rect = fitz.Rect(match["rect"])
                    
                    # Create position key to avoid duplicates
                    position_key = tuple(match["rect"]) + (page_idx,)
                    
                    if position_key in highlighted_positions:
                        continue
                    
                    # Track this match with its range_id for later reporting
                    if "range_id" not in match and range_id:
                        match["range_id"] = range_id
                    
                    # Add highlight with reduced border and semi-transparent fill
                    page.draw_rect(
                        rect, 
                        color=darker_color,  # Darker border color
                        width=self.border_width,  # Reduced border width
                        fill=color + [self.fill_opacity]  # Semi-transparent fill
                    )
                    
                    # Mark as highlighted
                    highlighted_positions.add(position_key)
        
        # Save the highlighted PDF (without legend page)
        doc.save(output_pdf)
        doc.close()
        
        logger.info(f"Highlighted PDF saved to: {output_pdf}")
        data["output_pdf"] = output_pdf
        
        # Create separate legend file
        legend_path = output_dir / f"{base_name}_{timestamp}_legend.pdf"
        self._create_separate_legend(
            legend_path, entity_colors, search_results, Path(pdf_path).name, 
            data.get("search_metrics", {}), station_highlights
        )
        data["legend_pdf"] = legend_path
        
        # Store station range data for report
        if station_highlights:
            data["station_range_highlights"] = station_highlights
        
        return data
        
    def _generate_fiber_cable_colors(self, entities: List[str]) -> Dict[str, List[float]]:
        """
        Generate colors using fiber cable color order.
        
        Args:
            entities: List of entity names
            
        Returns:
            Dictionary mapping entities to RGB colors (0-1 range)
        """
        # Fiber cable color order (RGB values in 0-1 range)
        fiber_colors = [
            [0.0, 0.0, 1.0],    # Blue
            [1.0, 0.5, 0.0],    # Orange
            [0.0, 1.0, 0.0],    # Green
            [0.6, 0.4, 0.2],    # Brown
            [0.5, 0.5, 0.5],    # Slate/Gray
            [1.0, 1.0, 1.0],    # White
            [1.0, 0.0, 0.0],    # Red
            [0.0, 0.0, 0.0],    # Black
            [1.0, 1.0, 0.0],    # Yellow
            [0.5, 0.0, 0.5],    # Violet
            [1.0, 0.8, 0.8],    # Rose/Pink
            [0.0, 1.0, 1.0],    # Aqua/Light Blue
        ]
        
        entity_colors = {}
        for i, entity in enumerate(entities):
            if i < len(fiber_colors):
                entity_colors[entity] = fiber_colors[i]
            else:
                # Generate random color for additional entities
                entity_colors[entity] = [random.random() for _ in range(3)]
        
        return entity_colors
        
    def _generate_color(self) -> List[float]:
        """Generate a random color in the range 0-1 for PyMuPDF"""
        return [random.random() for _ in range(3)]
        
    def _darken_color(self, color: List[float], factor: float = 0.7) -> List[float]:
        """Make a color darker by reducing its values"""
        return [max(0, c * factor) for c in color]
        
    def _generate_distinct_colors(self, num_colors: int) -> List[List[float]]:
        """
        Generate a list of visually distinct colors
        
        Args:
            num_colors: Number of colors to generate
            
        Returns:
            List of RGB colors in 0-1 range
        """
        colors = []
        hue_step = 1.0 / num_colors
        
        for i in range(num_colors):
            hue = i * hue_step
            # HSV with high saturation and value for distinctiveness
            r, g, b = self._hsv_to_rgb(hue, 0.8, 0.9)
            colors.append([r, g, b])
        
        return colors
        
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """
        Convert HSV color to RGB
        
        Args:
            h: Hue (0-1)
            s: Saturation (0-1)
            v: Value (0-1)
            
        Returns:
            Tuple of (r, g, b) values in 0-1 range
        """
        if s == 0:
            return v, v, v
            
        h *= 6
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q
            
    def _assign_colors_by_station(self, station_ranges: List[Tuple[str, Tuple[float, float]]]) -> Dict[str, List[float]]:
        """
        Assign colors to station ranges ensuring adjacent ranges get different colors
        
        Args:
            station_ranges: List of (range_id, (start_station, end_station)) tuples
            
        Returns:
            Dict mapping range_id to RGB color (0-1 range)
        """
        # Sort ranges by start station
        sorted_ranges = sorted(station_ranges, key=lambda x: x[1][0])
        
        # Generate a color palette with enough distinct colors
        # We'll use at least 4 colors to ensure we can always avoid adjacent duplicates
        num_colors = max(4, len(sorted_ranges) // 3 + 1)
        colors = self._generate_distinct_colors(num_colors)
        
        # Assign colors ensuring adjacent ranges get different colors
        color_assignments = {}
        used_colors = []
        
        for i, (range_id, _) in enumerate(sorted_ranges):
            # For the first range, pick first color
            if i == 0:
                color_assignments[range_id] = colors[0]
                used_colors.append(colors[0])
                continue
            
            # For subsequent ranges, avoid colors used by adjacent ranges
            adjacent_colors = []
            if i > 0:
                prev_range_id = sorted_ranges[i-1][0]
                adjacent_colors.append(color_assignments[prev_range_id])
            
            # Choose a color not used by adjacent ranges
            available_colors = [c for c in colors if all(not self._colors_similar(c, ac) for ac in adjacent_colors)]
            if available_colors:
                color_assignments[range_id] = available_colors[0]
                used_colors.append(available_colors[0])
            else:
                # If somehow we run out of colors, just pick one that's least used
                color_counts = {}
                for c in colors:
                    color_counts[tuple(c)] = sum(1 for used_c in used_colors if self._colors_similar(c, used_c))
                
                best_color = min(colors, key=lambda c: color_counts[tuple(c)])
                color_assignments[range_id] = best_color
                used_colors.append(best_color)
        
        return color_assignments
        
    def _colors_similar(self, color1: List[float], color2: List[float], threshold: float = 0.1) -> bool:
        """Check if two colors are similar"""
        return sum((a - b) ** 2 for a, b in zip(color1, color2)) < threshold
        
    def _create_separate_legend(self, legend_path: Path, entity_colors: Dict[str, List[float]], 
                              search_results: Dict[str, Dict], pdf_name: str,
                              metrics: Dict[str, int], station_highlights: Dict[str, Dict] = None) -> None:
        """
        Create a separate legend PDF file.
        
        Args:
            legend_path: Path for the legend PDF
            entity_colors: Dictionary mapping entities to their colors
            search_results: Search results from PDFSearchStage
            pdf_name: Name of the original PDF file
            metrics: Search metrics
            station_highlights: Optional dictionary of station range highlights
        """
        logger.info(f"Creating separate legend file: {legend_path}")
        
        # Create a new PDF document for the legend
        legend_doc = fitz.open()
        
        # Create a new page with standard dimensions
        legend_page = legend_doc.new_page(width=612, height=792)  # Standard letter size
        
        # Add title and metadata
        legend_y = 50  # Starting Y position
        legend_x = 50  # X position
        legend_spacing = 20  # Spacing between entries
        
        # Add title
        legend_page.insert_text(
            (legend_x, legend_y), 
            "CABLE ENTITY MATCH LEGEND",
            fontsize=18, 
            color=(0, 0, 0)
        )
        legend_y += 30
        
        # Add metadata
        total_matches = sum(len(result["locations"]) for result in search_results.values())
        legend_page.insert_text(
            (legend_x, legend_y),
            f"PDF: {pdf_name} | Total matches: {total_matches}",
            fontsize=12, 
            color=(0, 0, 0)
        )
        legend_y += 40
        
        # Add table header
        legend_page.insert_text((legend_x, legend_y), "Entity", fontsize=14, color=(0, 0, 0))
        legend_page.insert_text((legend_x + 300, legend_y), "Count", fontsize=14, color=(0, 0, 0))
        legend_page.insert_text((legend_x + 350, legend_y), "Color", fontsize=14, color=(0, 0, 0))
        legend_y += 25
        
        # Add separator line
        legend_page.draw_line(
            (legend_x, legend_y), 
            (legend_x + 450, legend_y),
            color=(0, 0, 0), 
            width=1.0
        )
        legend_y += 15
        
        # Sort entities by match count (descending)
        sorted_entities = sorted(
            search_results.items(), 
            key=lambda x: len(x[1]["locations"]), 
            reverse=True
        )
        
        # Add entity rows
        for entity, result in sorted_entities:
            count = len(result["locations"])
            if count == 0:
                continue  # Skip entities with no matches
                
            # Add entity name and count
            legend_page.insert_text(
                (legend_x, legend_y),
                f"{entity}",
                fontsize=12, 
                color=(0, 0, 0)
            )
            legend_page.insert_text(
                (legend_x + 300, legend_y),
                f"{count}",
                fontsize=12, 
                color=(0, 0, 0)
            )
            
            # Add color sample
            legend_page.draw_rect(
                fitz.Rect(legend_x + 350, legend_y - 12, legend_x + 380, legend_y + 2),
                color=entity_colors[entity],
                fill=entity_colors[entity],
                width=0
            )
            legend_y += legend_spacing
            
            # Add page references
            if result["locations"]:
                pages = list(set(loc["page"] for loc in result["locations"]))
                pages.sort()
                pages_str = ", ".join(str(p) for p in pages[:10])
                if len(pages) > 10:
                    pages_str += f", ... ({len(pages) - 10} more)"
                    
                legend_page.insert_text(
                    (legend_x + 20, legend_y),
                    f"Pages: {pages_str}",
                    fontsize=9, 
                    color=(0.5, 0.5, 0.5)
                )
                legend_y += legend_spacing - 5
            
            legend_y += 5  # Extra space between entities
        
        # Save the legend PDF
        legend_doc.save(legend_path)
        legend_doc.close()
        
        logger.info(f"Legend PDF saved to: {legend_path}")

# Report generation stage
class ReportGenerationStage(PipelineStage):
    """Generates JSON report with detailed analysis"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive report
        
        Args:
            data: Input data with search results
            
        Returns:
            Dict[str, Any]: Data with report path added
        """
        search_results = data.get("search_results", {})
        output_dir = data.get("output_dir", Path("."))
        timestamp = data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        pdf_path = data.get("pdf_path")
        
        if not search_results:
            logger.warning("No search results for report generation")
            return data
            
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create output filename
        base_name = Path(pdf_path).stem
        output_report = output_dir / f"{base_name}_{timestamp}_report.json"
        
        # Prepare report data
        report = {
            "generated_at": datetime.now().isoformat(),
            "input_files": {
                "pdf": str(pdf_path),
                "csv": str(data.get("csv_path"))
            },
            "summary": {
                "total_entities": len(search_results),
                "entities_found": sum(1 for result in search_results.values() if result["count"] > 0),
                "entities_not_found": sum(1 for result in search_results.values() if result["count"] == 0),
                "total_matches": sum(result["count"] for result in search_results.values()),
                "metrics": data.get("search_metrics", {})
            },
            "entity_details": {}
        }
        
        # Add station range information if available
        station_range_highlights = data.get("station_range_highlights")
        if station_range_highlights:
            # Format station ranges for the report
            station_ranges = {}
            for range_id, range_info in station_range_highlights.items():
                entity_name = range_info["entity"]
                start, end = range_info["range"]
                
                # Convert start and end back to station format
                start_station = f"{int(start // 100)}+{start % 100:.2f}".rstrip('0').rstrip('.')
                end_station = f"{int(end // 100)}+{end % 100:.2f}".rstrip('0').rstrip('.')
                
                if entity_name not in station_ranges:
                    station_ranges[entity_name] = []
                    
                # Format the range info
                range_data = {
                    "start_station": start_station,
                    "end_station": end_station,
                    "pages": sorted(list(range_info["pages"]))
                }
                station_ranges[entity_name].append(range_data)
            
            # Add to report
            report["station_ranges"] = station_ranges
        
        # Add detailed entity information
        for entity, result in search_results.items():
            # Convert sets to lists for JSON serialization
            if isinstance(result.get("pages"), set):
                result["pages"] = sorted(list(result["pages"]))
                
            report["entity_details"][entity] = {
                "count": result["count"],
                "pages": result["pages"],
                "variations": result["variations"],
                "station_references": result.get("station_references", []),
                "new_instances_count": len(result.get("new_instances", [])),
            }
            
            # Add location summaries without full details to keep report size reasonable
            location_summary = []
            for loc in result.get("locations", [])[:20]:  # Limit to 20 locations
                summary = {
                    "page": loc["page"],
                    "variation": loc["variation"]
                }
                if "station_values" in loc:
                    summary["station_values"] = loc["station_values"]
                    
                location_summary.append(summary)
                
            if len(result.get("locations", [])) > 20:
                report["entity_details"][entity]["locations_note"] = (
                    f"Only showing 20 of {len(result['locations'])} total locations"
                )
                
            report["entity_details"][entity]["location_samples"] = location_summary
        
        # Save report as JSON
        with open(output_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Summary report saved to: {output_report}")
        data["output_report"] = output_report
        
        return data

# Main pipeline function
def create_entity_matching_pipeline(
    border_width: float = 1.0,
    fill_opacity: float = 0.15
) -> Pipeline:
    """
    Create the complete entity matching pipeline
    
    Args:
        border_width: Width of highlight borders
        fill_opacity: Opacity of highlight fills (0-1)
        
    Returns:
        Pipeline: Configured pipeline
    """
    stages = [
        DataValidationStage(),
        CSVExtractionStage(),
        VariationGenerationStage(),
        PDFSearchStage(),
        PDFHighlightingStage(border_width=border_width, fill_opacity=fill_opacity),
        ReportGenerationStage()
    ]
    
    return Pipeline(stages)

def run_pipeline(
    csv_path: str,
    pdf_path: str,
    output_dir: str,
    border_width: float = 1.0,
    fill_opacity: float = 0.15,
    one_line_mode: bool = False
) -> Dict[str, Any]:
    """
    Run the entity matching pipeline
    
    Args:
        csv_path: Path to CSV file with cable info
        pdf_path: Path to PDF document
        output_dir: Directory to save outputs
        border_width: Width of highlight borders
        fill_opacity: Opacity of highlight fills (0-1)
        one_line_mode: Enable one-line diagram mode (optimized for one-line diagrams)
        
    Returns:
        Dict[str, Any]: Results of the pipeline
    """
    # Initialize pipeline
    pipeline = create_entity_matching_pipeline(
        border_width=border_width,
        fill_opacity=fill_opacity
    )
    
    # Create initial data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    initial_data = {
        "csv_path": csv_path,
        "pdf_path": pdf_path,
        "output_dir": output_dir,
        "timestamp": timestamp
    }
    
    # Run the pipeline
    logger.info(f"Starting entity matching pipeline with timestamp {timestamp}")
    results = pipeline.run(initial_data)
    logger.info("Pipeline completed successfully")
    
    return results

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cable Entity Matching Pipeline")
    
    parser.add_argument(
        "--csv", 
        required=True,
        help="Path to CSV file with cable information"
    )
    
    parser.add_argument(
        "--pdf", 
        required=True,
        help="Path to PDF document to analyze"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="./output",
        help="Directory to save outputs (default: ./output)"
    )
    
    parser.add_argument(
        "--border-width", 
        type=float, 
        default=1.0,
        help="Width of highlight borders (default: 1.0)"
    )
    
    parser.add_argument(
        "--fill-opacity", 
        type=float, 
        default=0.15,
        help="Opacity of highlight fills, 0-1 (default: 0.15)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    # Run the pipeline
    try:
        results = run_pipeline(
            csv_path=args.csv,
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            border_width=args.border_width,
            fill_opacity=args.fill_opacity
        )
        
        # Print summary
        print(f"\nEntity matching completed successfully:")
        print(f"- Highlighted PDF: {results.get('output_pdf')}")
        print(f"- Report: {results.get('output_report')}")
        print(f"- Entities found: {results['summary']['entities_found']}/{results['summary']['total_entities']}")
        print(f"- Total matches: {results['summary']['total_matches']}")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        raise
