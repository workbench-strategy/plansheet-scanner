import csv
import fitz  # PyMuPDF
import random
import re
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import pandas as pd
from difflib import get_close_matches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cable_matcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CableMatcher")

class DataValidator:
    """Validates and standardizes input data from various sources."""
    
    @staticmethod
    def validate_csv_structure(file_path: Path, required_columns: List[str]) -> bool:
        """
        Validate if CSV has required columns.
        
        Args:
            file_path: Path to the CSV file
            required_columns: List of column names that must be present
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    logger.error(f"Empty CSV file: {file_path}")
                    return False
            
            # Special handling for the specific CSV structure with blank first row
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if any(required_col in line for required_col in required_columns):
                        logger.info(f"Found header with required columns at line {i+1}")
                        return True
            
            # Try with pandas as fallback
            try:
                df = pd.read_csv(file_path, skiprows=[0])  # Skip the first blank row
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if not missing_columns:
                    logger.info("Found all required columns after skipping the first row")
                    return True
                    
                # Try with no skipping as another fallback
                df = pd.read_csv(file_path)
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.error(f"Missing required columns in CSV: {missing_columns}")
                    return False
            except Exception as e:
                logger.warning(f"Pandas read failed: {str(e)}")
                # Continue with manual validation
            
            logger.info("CSV structure validated successfully")
            return True
        except Exception as e:
            logger.error(f"Error validating CSV structure: {str(e)}")
            return False
    
    @staticmethod
    def validate_pdf(file_path: Path) -> bool:
        """
        Validate if a PDF file exists and is readable.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            if not file_path.exists():
                logger.error(f"PDF file not found: {file_path}")
                return False
                
            doc = fitz.open(file_path)
            page_count = len(doc)
            if page_count == 0:
                logger.error(f"PDF has no pages: {file_path}")
                return False
                
            doc.close()
            logger.info(f"PDF validation successful: {file_path} ({page_count} pages)")
            return True
        except Exception as e:
            logger.error(f"Error validating PDF: {str(e)}")
            return False
    
    @staticmethod
    def clean_and_standardize_data(data: List[str]) -> List[str]:
        """
        Clean and standardize text data by removing extra whitespace, 
        normalizing case, etc.
        
        Args:
            data: List of strings to clean
            
        Returns:
            List[str]: Cleaned data
        """
        cleaned_data = []
        for item in data:
            if not item:
                continue
                
            # Remove extra whitespace and standardize
            clean_item = re.sub(r'\s+', ' ', item).strip()
            
            # Other cleaning steps can be added here
            
            if clean_item:
                cleaned_data.append(clean_item)
                
        return cleaned_data

class EntityExtractor:
    """Extracts and processes entities from various data sources."""
    
    @staticmethod
    def extract_new_cable_references(text: str, cable_types: List[str]) -> Dict[str, List[str]]:
        """
        Extract references to new cables by finding cable names that appear after the word "new".
        
        Args:
            text: The text to search in
            cable_types: List of cable types to look for
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping cable types to their "new" references
        """
        results = {}
        
        # Pattern for finding "new <something>" with optional words in between
        new_patterns = [
            r'new\s+(\w+\s+){0,3}({})',  # "new <up to 3 words> <cable_type>"
            r'NEW\s+(\w+\s+){0,3}({})',  # Same but uppercase
            r'New\s+(\w+\s+){0,3}({})',  # Same with capitalized first letter
        ]
        
        for cable_type in cable_types:
            results[cable_type] = []
            
            # Build regex patterns with the cable type
            for pattern_template in new_patterns:
                pattern = pattern_template.format(re.escape(cable_type))
                
                # Find all matches
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Get the full match
                    full_match = match.group(0)
                    results[cable_type].append(full_match)
        
        return results
    
    @staticmethod
    def extract_cable_types_from_csv(csv_path: Path) -> List[str]:
        """
        Extract cable types from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List[str]: List of cable types
        """
        cable_types = []
        
        try:
            # First try with pandas for better error handling
            df = pd.read_csv(csv_path, skip_blank_lines=True)
            
            # Check if "Cable Type" column exists
            if "Cable Type" in df.columns:
                cable_types = df["Cable Type"].dropna().tolist()
                logger.info(f"Extracted {len(cable_types)} cable types using pandas")
                return [ct for ct in cable_types if ct.strip()]
            
            # If column not found, try manual parsing
            with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
                lines = csvfile.readlines()
                
                # Find the header line (should contain "Cable Type")
                header_index = -1
                for i, line in enumerate(lines):
                    if "Cable Type" in line:
                        header_index = i
                        break
                
                if header_index == -1:
                    logger.error("Could not find 'Cable Type' in any line")
                    return []
                    
                logger.info(f"Found header at line {header_index + 1}")
                
                # Parse the CSV starting from the header line
                reader = csv.reader(lines[header_index:])
                header = next(reader)
                
                # Find the index of the Cable Type column
                cable_col_index = None
                for i, col_name in enumerate(header):
                    if "Cable Type" in col_name:
                        cable_col_index = i
                        break
                
                if cable_col_index is None:
                    logger.error("Could not find 'Cable Type' column in header")
                    return []
                    
                logger.info(f"Found 'Cable Type' column at index {cable_col_index}")
                
                # Now read the cable types
                for row in reader:
                    if row and len(row) > cable_col_index and row[cable_col_index].strip():
                        cable_types.append(row[cable_col_index].strip())
                        logger.debug(f"Added cable type: {row[cable_col_index].strip()}")
            
            logger.info(f"Extracted {len(cable_types)} cable types from CSV")
            
        except Exception as e:
            logger.error(f"Error extracting cable types: {str(e)}")
            
        return cable_types
    
         @staticmethod
     def generate_entity_variations(entity: str) -> List[str]:
         """
         Generate variations of entity names to handle inconsistencies.
         
         Args:
             entity: Original entity name
             
         Returns:
             List[str]: List of variations
         """
         variations = [entity]  # Original name
         
         # Handle FTC variations
         if "FTC" in entity:
             # Split the name and check if it's in the format "FTC Distribution"
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
         
         # Add "NEW" prefix variations for cable types
         if "FTC" in entity or "SR 167" in entity or "I-5" in entity:
             variations.append(f"NEW {entity}")
             variations.append(f"new {entity}")
             variations.append(f"New {entity}")
         
                   # Add context variations for common phrases
          if "FTC" in entity and "Toll" in entity:
              variations.append("FTC TOLL")
              variations.append("ftc toll")
              variations.append("FTC Toll")
              # Also add the reverse combination
              variations.append("TOLL FTC")
              variations.append("toll ftc")
              variations.append("Toll FTC")
          
          if "Distribution" in entity and "Fiber" in entity:
              variations.append("Distribution Fiber")
              variations.append("distribution fiber")
              variations.append("DISTRIBUTION FIBER")
         
         return variations

    @staticmethod
    def extract_stationing_values(text: str) -> List[str]:
        """
        Extract stationing values from text using regex patterns.
        
        Args:
            text: Text to search for stationing values
            
        Returns:
            List[str]: List of stationing values found
        """
        # Common stationing patterns:
        # - STA 78+35
        # - 3630+00
        # - Station 120+45.67
        station_patterns = [
            r'STA\s+(\d+\+\d+(?:\.\d+)?)',
            r'Station\s+(\d+\+\d+(?:\.\d+)?)',
            r'(\d+\+\d+(?:\.\d+)?)'
        ]
        
        results = []
        for pattern in station_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                station_value = match.group(1) if len(match.groups()) > 0 else match.group(0)
                results.append(station_value)
                
        return results
        
    @staticmethod
    def extract_station_callouts_for_cables(text: str, cable_types: List[str], context_window: int = 100) -> Dict[str, List[str]]:
        """
        Extract station callouts associated with cable types.
        
        Args:
            text: The text to search in
            cable_types: List of cable types to look for
            context_window: Number of characters around cable mention to search for stations
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping cable types to their station callouts
        """
        results = {}
        
        # Station patterns as above
        station_patterns = [
            r'STA\s+(\d+\+\d+(?:\.\d+)?)',
            r'Station\s+(\d+\+\d+(?:\.\d+)?)',
            r'(\d+\+\d+(?:\.\d+)?)'
        ]
        
        # Compile the station regex patterns
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in station_patterns]
        
        for cable_type in cable_types:
            results[cable_type] = []
            
            # Find all mentions of this cable type
            cable_mentions = []
            for match in re.finditer(re.escape(cable_type), text):
                start_pos = max(0, match.start() - context_window)
                end_pos = min(len(text), match.end() + context_window)
                context = text[start_pos:end_pos]
                cable_mentions.append(context)
            
            # For each mention, find station callouts in context
            for context in cable_mentions:
                for pattern in compiled_patterns:
                    for station_match in pattern.finditer(context):
                        station = station_match.group()
                        if station not in results[cable_type]:
                            results[cable_type].append(station)
        
        return results

class EntityMatcher:
    """Implements matching algorithms to find entities in documents."""
    
    def __init__(self, fuzzy_match_threshold: float = 0.8):
        """
        Initialize the matcher.
        
        Args:
            fuzzy_match_threshold: Threshold for fuzzy matching (0-1)
        """
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.match_metrics = {
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "new_cable_matches": 0,
            "missed_matches": 0,
            "duplicate_matches": 0
        }
        self.station_callouts = {}
        self.new_cable_references = {}
    
    def fuzzy_match(self, query: str, choices: List[str], threshold: float = None) -> List[str]:
        """
        Perform fuzzy matching to find similar strings.
        
        Args:
            query: String to match
            choices: List of possible matches
            threshold: Optional custom threshold
            
        Returns:
            List[str]: List of matches
        """
        if threshold is None:
            threshold = self.fuzzy_match_threshold
            
        matches = get_close_matches(query, choices, n=3, cutoff=threshold)
        return matches
    
    def find_entities_in_pdf(self, pdf_path: Path, entities: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Search for entities and their variations in a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            entities: Dictionary mapping original entity names to their variations
            
        Returns:
            Dict[str, Dict]: Dictionary with match statistics for each entity
        """
        doc = fitz.open(pdf_path)
        logger.info(f"Searching for {len(entities)} entities in PDF: {pdf_path}")
        
        # Reset metrics
        self.match_metrics = {
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "new_cable_matches": 0,
            "missed_matches": 0,
            "duplicate_matches": 0
        }
        
        # Create dictionary to store results
        results = {}
        highlighted_positions = set()
        
        # Reset storage for station callouts and new cable references
        self.station_callouts = {}
        self.new_cable_references = {}
        
        # Map variations back to original entities
        variation_to_original = {}
        original_entities = list(entities.keys())
        
        for original, variations in entities.items():
            results[original] = {
                "count": 0,
                "pages": set(),
                "variations": {},
                "locations": [],
                "stations": [],
                "new_references": []
            }
            self.station_callouts[original] = []
            self.new_cable_references[original] = []
            
            for variation in variations:
                variation_to_original[variation] = original
        
        # Extract full text for finding new cable references and station callouts
        full_text = ""
        for page in doc:
            full_text += page.get_text() + " "
            
        # Extract station callouts associated with cables
        extractor = EntityExtractor()
        all_stations = extractor.extract_station_callouts_for_cables(full_text, original_entities)
        new_cables = extractor.extract_new_cable_references(full_text, original_entities)
        
        # Store these for later use
        for cable, stations in all_stations.items():
            self.station_callouts[cable] = stations
            results[cable]["stations"] = stations
        
        for cable, refs in new_cables.items():
            self.new_cable_references[cable] = refs
            results[cable]["new_references"] = refs
            
        # Search through PDF pages
        for page_idx, page in enumerate(doc):
            page_text = page.get_text()
            logger.debug(f"Scanning page {page_idx+1}")
            
            # First, look for "new" cable references
            for original in original_entities:
                # Look for "new <cable>" in this page's text
                new_cable_patterns = [
                    rf'new\s+(\w+\s+){{0,3}}{re.escape(original)}',
                    rf'NEW\s+(\w+\s+){{0,3}}{re.escape(original)}',
                    rf'New\s+(\w+\s+){{0,3}}{re.escape(original)}'
                ]
                
                for pattern in new_cable_patterns:
                    for match in re.finditer(pattern, page_text):
                        # Get the match position
                        match_text = match.group(0)
                        match_pos = match.span()
                        
                        # Try to find the exact position in the page
                        text_instances = page.search_for(match_text)
                        
                        for inst in text_instances:
                            # Create unique position identifier
                            position_key = (
                                round(inst[0], 2), round(inst[1], 2),
                                round(inst[2], 2), round(inst[3], 2),
                                page_idx
                            )
                            
                            if position_key in highlighted_positions:
                                self.match_metrics["duplicate_matches"] += 1
                                continue
                                
                            highlighted_positions.add(position_key)
                            
                            # Update statistics
                            results[original]["count"] += 1
                            results[original]["pages"].add(page_idx + 1)
                            
                            # Track as new cable reference
                            new_ref_key = f"new {original}"
                            if new_ref_key not in results[original]["variations"]:
                                results[original]["variations"][new_ref_key] = 0
                            results[original]["variations"][new_ref_key] += 1
                            
                            # Store location (page numbers will be adjusted for legend page later)
                            results[original]["locations"].append({
                                "page": page_idx + 1,
                                "rect": [inst[0], inst[1], inst[2], inst[3]],
                                "variation": new_ref_key,
                                "is_new_cable": True
                            })
                            
                            self.match_metrics["new_cable_matches"] += 1
            
            # Then do regular entity searching
            for original, variations in entities.items():
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
                            self.match_metrics["duplicate_matches"] += 1
                            continue
                            
                        highlighted_positions.add(position_key)
                        
                        # Update statistics
                        results[original]["count"] += 1
                        results[original]["pages"].add(page_idx + 1)
                        
                        if variation not in results[original]["variations"]:
                            results[original]["variations"][variation] = 0
                        results[original]["variations"][variation] += 1
                        
                        # Store location (can be used for visualization)
                        results[original]["locations"].append({
                            "page": page_idx + 1,
                            "rect": [inst[0], inst[1], inst[2], inst[3]],
                            "variation": variation,
                            "is_new_cable": False
                        })
                        
                        # Update metrics
                        if variation == original:
                            self.match_metrics["exact_matches"] += 1
                        else:
                            self.match_metrics["fuzzy_matches"] += 1
        
        # Convert page sets to sorted lists for better output
        for entity in results:
            results[entity]["pages"] = sorted(list(results[entity]["pages"]))
            
        # Calculate missed entities
        missed_entities = [entity for entity, data in results.items() if data["count"] == 0]
        self.match_metrics["missed_matches"] = len(missed_entities)
        
        doc.close()
        return results

class OutputGenerator:
    """Generates various output formats from processed data."""
    
    @staticmethod
    def generate_fiber_cable_color(index):
        """
        Generate colors following the multistrand fiber cable color order.
        
        Args:
            index: 0-based index for the color sequence
            
        Returns:
            List[float]: RGB color values in range 0-1
        """
        # Standard multistrand fiber cable color order
        fiber_colors = [
            [0.0, 0.0, 1.0],    # 1. Blue
            [1.0, 0.5, 0.0],    # 2. Orange
            [0.0, 0.5, 0.0],    # 3. Green
            [0.6, 0.4, 0.2],    # 4. Brown
            [0.5, 0.5, 0.5],    # 5. Slate (Gray)
            [1.0, 1.0, 1.0],    # 6. White
            [1.0, 0.0, 0.0],    # 7. Red
            [0.0, 0.0, 0.0],    # 8. Black
            [1.0, 1.0, 0.0],    # 9. Yellow
            [0.5, 0.0, 0.5],    # 10. Violet
            [1.0, 0.75, 0.8],   # 11. Rose (Pink)
            [0.0, 1.0, 1.0],    # 12. Aqua (Light Blue)
        ]
        
        # If we have more entities than colors, cycle through the colors
        color_index = index % len(fiber_colors)
        return fiber_colors[color_index]
    
    @staticmethod
    def generate_random_color():
        """
        Generate a random RGB color in the range 0-1 as required by PyMuPDF.
        
        Returns:
            List[float]: RGB color values
        """
        # Generate colors with higher saturation for better visibility
        hue = random.random()  # 0-1
        saturation = random.uniform(0.6, 0.9)  # Higher saturation for vibrance
        value = random.uniform(0.7, 1.0)  # Fairly bright colors
        
        # Convert HSV to RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs((h % 2) - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return [r + m, g + m, b + m]
    
    @staticmethod
    def darker_color(color, factor=0.7):
        """
        Create a darker version of a color.
        
        Args:
            color: Original RGB color (list of 3 floats, 0-1)
            factor: Darkening factor (0-1, lower makes color darker)
            
        Returns:
            List[float]: Darker RGB color
        """
        return [max(0, c * factor) for c in color]
    
    @staticmethod
    def highlight_entities_in_pdf(
        pdf_path: Path,
        entities: Dict[str, List[str]],
        match_results: Dict[str, Dict],
        output_path: Path
    ):
        """
        Highlight entities in PDF and generate a separate legend file.
        
        Args:
            pdf_path: Path to input PDF
            entities: Dictionary of entities and their variations
            match_results: Match results from EntityMatcher
            output_path: Path for output highlighted PDF
        """
        logger.info(f"Highlighting entities in PDF: {pdf_path}")
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        logger.info(f"PDF has {len(doc)} pages")
        
        # Generate colors for each entity using fiber cable color order
        entity_colors = OutputGenerator._generate_fiber_cable_colors(list(entities.keys()))
        
        # Track highlighted positions to avoid duplicates
        highlighted_positions = set()
        
        # Highlight entities in the document (skip legend page if it exists)
        for page_idx, page in enumerate(doc):
            # Skip the first page if it's a legend page (check by looking for "LEGEND" text)
            if page_idx == 0:
                page_text = page.get_text().upper()
                if "LEGEND" in page_text or "ENTITY MATCH" in page_text:
                    logger.info(f"Skipping legend page (page {page_idx + 1})")
                    continue
            
            logger.debug(f"Processing page {page_idx + 1}")
            
            for entity, variations in entities.items():
                if entity not in match_results or not match_results[entity]["locations"]:
                    continue
                    
                color = entity_colors[entity]
                
                # Find matches on this page
                page_matches = [loc for loc in match_results[entity]["locations"] 
                              if loc["page"] == page_idx]
                
                for match in page_matches:
                    # Get the original coordinates from search_for
                    original_rect = fitz.Rect(match["rect"])
                    
                    # Create position key to avoid duplicates
                    position_key = tuple(match["rect"]) + (page_idx,)
                    
                    if position_key in highlighted_positions:
                        continue
                    
                    # FIX: Apply coordinate system corrections
                    corrected_rect = OutputGenerator._correct_highlight_coordinates(page, original_rect)
                    
                                         # Highlight the match with corrected coordinates
                     try:
                         # Method 1: Try highlight annotation with transparency
                         highlight = page.add_highlight_annot(corrected_rect)
                         # Set transparency for see-through effect
                         highlight.set_opacity(0.3)  # 30% opacity for transparency
                         logger.debug(f"Highlight annotation created for {entity} on page {page_idx + 1}")
                     except Exception as e:
                         logger.warning(f"Failed to create highlight annotation: {e}")
                         # Fallback to drawing rectangle with transparency
                         page.draw_rect(corrected_rect, color=color, width=0.25, fill=color, opacity=0.3)
                     
                     # Draw a subtle border for definition
                     page.draw_rect(corrected_rect, color=color, width=0.25)
                    
                    # Mark as highlighted
                    highlighted_positions.add(position_key)
        
        # Save the highlighted PDF (without legend page)
        doc.save(output_path)
        doc.close()
        
        logger.info(f"Highlighted PDF saved to: {output_path}")
        
        # Create separate legend file
        legend_path = output_path.parent / f"{output_path.stem}_legend.pdf"
        OutputGenerator._create_separate_legend(
            legend_path, entity_colors, match_results, pdf_path.name
        )
        
        return legend_path
    
    @staticmethod
    def _generate_fiber_cable_colors(entities: List[str]) -> Dict[str, List[float]]:
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
            [1.0, 0.0, 1.0],    # Magenta (replaced White)
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
    
    @staticmethod
    def _correct_highlight_coordinates(page, rect):
        """
        Correct coordinates for highlighting based on page properties.
        
        Args:
            page: PyMuPDF page object
            rect: Original rectangle from search_for
            
        Returns:
            fitz.Rect: Corrected rectangle for highlighting
        """
        # Get page properties
        page_rect = page.rect
        rotation = page.rotation
        
        # Start with the original rectangle
        corrected_rect = fitz.Rect(rect)
        
        # Handle page rotation
        if rotation != 0:
            logger.debug(f"Page has rotation: {rotation} degrees")
            
            # For rotated pages, we need to transform coordinates
            if rotation == 90:
                # 90 degree rotation: (x, y) -> (y, width-x)
                corrected_rect = fitz.Rect(
                    corrected_rect.y0, 
                    page_rect.width - corrected_rect.x1,
                    corrected_rect.y1, 
                    page_rect.width - corrected_rect.x0
                )
            elif rotation == 180:
                # 180 degree rotation: (x, y) -> (width-x, height-y)
                corrected_rect = fitz.Rect(
                    page_rect.width - corrected_rect.x1,
                    page_rect.height - corrected_rect.y1,
                    page_rect.width - corrected_rect.x0,
                    page_rect.height - corrected_rect.y0
                )
            elif rotation == 270:
                # 270 degree rotation: (x, y) -> (height-y, x)
                corrected_rect = fitz.Rect(
                    page_rect.height - corrected_rect.y1,
                    corrected_rect.x0,
                    page_rect.height - corrected_rect.y0,
                    corrected_rect.x1
                )
        
        # Ensure coordinates are within page bounds
        corrected_rect.x0 = max(0, min(corrected_rect.x0, page_rect.width))
        corrected_rect.y0 = max(0, min(corrected_rect.y0, page_rect.height))
        corrected_rect.x1 = max(0, min(corrected_rect.x1, page_rect.width))
        corrected_rect.y1 = max(0, min(corrected_rect.y1, page_rect.height))
        
        # Ensure rectangle has positive dimensions
        if corrected_rect.width <= 0:
            corrected_rect.x1 = corrected_rect.x0 + 1
        if corrected_rect.height <= 0:
            corrected_rect.y1 = corrected_rect.y0 + 1
        
        logger.debug(f"Original rect: {rect}")
        logger.debug(f"Corrected rect: {corrected_rect}")
        
        return corrected_rect
    
    @staticmethod
    def _create_separate_legend(
        legend_path: Path,
        entity_colors: Dict[str, List[float]], 
        match_results: Dict[str, Dict],
        pdf_name: str
    ):
        """
        Create a separate legend PDF file.
        
        Args:
            legend_path: Path for the legend PDF
            entity_colors: Dictionary mapping entities to their colors
            match_results: Match results from EntityMatcher
            pdf_name: Name of the original PDF file
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
            "ENTITY MATCH LEGEND",
            fontsize=18, 
            color=(0, 0, 0)
        )
        legend_y += 30
        
        # Add metadata
        total_matches = sum(data["count"] for data in match_results.values())
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
            match_results.items(), 
            key=lambda x: x[1]["count"], 
            reverse=True
        )
        
        # Add entity rows
        for entity, data in sorted_entities:
            if data["count"] == 0:
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
                f"{data['count']}",
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
            
            # Add variation details
            if data["variations"]:
                has_variations = False
                for var, count in data["variations"].items():
                    if var != entity:  # Skip the original form
                        if not has_variations:
                            legend_page.insert_text(
                                (legend_x + 20, legend_y),
                                "Found as:",
                                fontsize=10, 
                                color=(0.3, 0.3, 0.3)
                            )
                            legend_y += legend_spacing - 5
                            has_variations = True
                            
                        legend_page.insert_text(
                            (legend_x + 40, legend_y),
                            f"'{var}' ({count} occurrences)",
                            fontsize=10, 
                            color=(0.3, 0.3, 0.3)
                        )
                        legend_y += legend_spacing - 5
                
                if has_variations:
                    legend_y += 5  # Extra space after variations
            
            # Add page references
            if data["pages"]:
                pages_str = ", ".join(str(p) for p in data["pages"][:10])
                if len(data["pages"]) > 10:
                    pages_str += f", ... ({len(data['pages']) - 10} more)"
                    
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
    
    @staticmethod
    def generate_summary_report(
        match_results: Dict[str, Dict],
        metrics: Dict[str, int],
        output_path: Path
    ):
        """
        Generate a detailed summary report in JSON format.
        
        Args:
            match_results: Match results from EntityMatcher
            metrics: Matching metrics
            output_path: Path to save the report
        """
        # Prepare report data
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_entities": len(match_results),
                "entities_found": sum(1 for data in match_results.values() if data["count"] > 0),
                "entities_not_found": sum(1 for data in match_results.values() if data["count"] == 0),
                "total_matches": sum(data["count"] for data in match_results.values()),
                "metrics": metrics
            },
            "entity_details": match_results
        }
        
        # Convert sets to lists for JSON serialization
        for entity, data in report["entity_details"].items():
            if isinstance(data["pages"], set):
                data["pages"] = sorted(list(data["pages"]))
        
        # Save report as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Summary report saved to: {output_path}")

def configure_parser():
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Engineering Document Entity Matcher"
    )
    
    parser.add_argument(
        "--csv", 
        type=str, 
        required=True,
        help="Path to CSV file containing entities (e.g., cable types)"
    )
    
    parser.add_argument(
        "--pdf", 
        type=str, 
        required=True,
        help="Path to PDF document to analyze"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output",
        help="Directory to save output files (default: ./output)"
    )
    
    parser.add_argument(
        "--fuzzy-threshold", 
        type=float, 
        default=0.8,
        help="Threshold for fuzzy matching (0-1, default: 0.8)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser

def main():
    """Main function to execute the workflow."""
    # Parse command-line arguments
    parser = configure_parser()
    args = parser.parse_args()
    
    # Configure logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert paths to Path objects
    csv_path = Path(args.csv)
    pdf_path = Path(args.pdf)
    
    # Get current timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Validate input files
    logger.info("Step 1: Validating input files")
    data_validator = DataValidator()
    
    if not data_validator.validate_csv_structure(csv_path, ["Cable Type"]):
        logger.error("CSV validation failed")
        return 1
        
    if not data_validator.validate_pdf(pdf_path):
        logger.error("PDF validation failed")
        return 1
    
    # 2. Extract entities from CSV
    logger.info("Step 2: Extracting entities from CSV")
    entity_extractor = EntityExtractor()
    entities = entity_extractor.extract_cable_types_from_csv(csv_path)
    
    if not entities:
        logger.error("No entities extracted from CSV")
        return 1
        
    # Clean and standardize entities
    entities = data_validator.clean_and_standardize_data(entities)
    logger.info(f"Found {len(entities)} valid entities after cleaning")
    
    # 3. Generate variations for fuzzy matching
    logger.info("Step 3: Generating entity variations for matching")
    entity_variations = {}
    for entity in entities:
        variations = entity_extractor.generate_entity_variations(entity)
        entity_variations[entity] = variations
        logger.debug(f"Generated {len(variations)} variations for '{entity}'")
    
    # 4. Match entities in PDF
    logger.info("Step 4: Matching entities in PDF")
    entity_matcher = EntityMatcher(fuzzy_match_threshold=args.fuzzy_threshold)
    match_results = entity_matcher.find_entities_in_pdf(pdf_path, entity_variations)
    
    # 5. Generate highlighted PDF and legend
    logger.info("Step 5: Generating highlighted PDF and legend")
    output_generator = OutputGenerator()
    
    # Create output filenames with timestamp for uniqueness
    base_name = pdf_path.stem
    output_pdf = output_dir / f"{base_name}_{timestamp}_highlighted.pdf"
    output_report = output_dir / f"{base_name}_{timestamp}_report.json"
    
    legend_path = output_generator.highlight_entities_in_pdf(
        pdf_path,
        entity_variations,
        match_results,
        output_pdf
    )
    
    # 6. Generate summary report
    logger.info("Step 6: Generating summary report")
    output_generator.generate_summary_report(
        match_results,
        entity_matcher.match_metrics,
        output_report
    )
    
    # 7. Print summary to console
    found_count = sum(1 for data in match_results.values() if data["count"] > 0)
    total_matches = sum(data["count"] for data in match_results.values())
    
    logger.info(f"Summary: Found {found_count}/{len(entities)} entities with {total_matches} total matches")
    logger.info(f"Metrics: {entity_matcher.match_metrics}")
    logger.info(f"Highlighted PDF saved to: {output_pdf}")
    logger.info(f"Legend PDF saved to: {legend_path}")
    logger.info(f"Match report saved to: {output_report}")
    
    # Print a clear message about the timestamp for better user awareness
    logger.info(f"Timestamp for this run: {timestamp}")
    logger.info(f"All outputs are saved with timestamp to prevent overwriting previous results")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
