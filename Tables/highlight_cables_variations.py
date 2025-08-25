import csv
import fitz  # PyMuPDF
import random
from pathlib import Path
import re

def correct_highlight_coordinates(page, rect):
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
        print(f"Page has rotation: {rotation} degrees")
        
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
    
    print(f"Original rect: {rect}")
    print(f"Corrected rect: {corrected_rect}")
    
    return corrected_rect

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
        [1.0, 0.0, 1.0],    # 6. Magenta (replaced White)
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

def generate_random_color():
    """Generate a random RGB color in the range 0-1 as required by PyMuPDF."""
    return [random.random() for _ in range(3)]

def extract_cable_types(csv_path):
    """Extract cable types from the CSV file."""
    cable_types = []
    
    # Read the file line by line to properly handle the specific format
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        lines = csvfile.readlines()
        
        # Find the header line (should contain "Cable Type")
        header_index = -1
        for i, line in enumerate(lines):
            if "Cable Type" in line:
                header_index = i
                break
        
        if header_index == -1:
            print("Error: Could not find 'Cable Type' in any line")
            return []
            
        print(f"Found header at line {header_index + 1}")
        
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
            print("Error: Could not find 'Cable Type' column in header")
            return []
            
        print(f"Found 'Cable Type' column at index {cable_col_index}")
        
        # Now read the cable types
        for row in reader:
            if row and len(row) > cable_col_index and row[cable_col_index].strip():
                cable_types.append(row[cable_col_index].strip())
                print(f"Added cable type: {row[cable_col_index].strip()}")
    
    return cable_types

def generate_cable_variations(cable_type):
    """Generate variations of cable names to handle inconsistencies."""
    variations = [cable_type]  # Original name
    
    # Handle FTC variations
    if "FTC" in cable_type:
        # Split the name and check if it's in the format "FTC Distribution"
        parts = cable_type.split()
        if len(parts) >= 2:
            if parts[0] == "FTC":
                # Add "Distribution FTC" variation
                variations.append(f"{' '.join(parts[1:])} FTC")
            elif parts[-1] == "FTC":
                # Already in "Distribution FTC" format, add the reverse
                variations.append(f"FTC {' '.join(parts[:-1])}")
    
    # Handle other variations like SR 167, I-5, etc.
    if "SR 167" in cable_type:
        variations.append(cable_type.replace("SR 167", "SR-167"))
        variations.append(cable_type.replace("SR 167", "SR167"))
    
    if "I-5" in cable_type:
        variations.append(cable_type.replace("I-5", "I5"))
        variations.append(cable_type.replace("I-5", "Interstate 5"))
    
    # Add uppercase and lowercase variations
    variations.append(cable_type.upper())
    variations.append(cable_type.lower())
    
    # Add context variations for common phrases
    if "FTC" in cable_type and "Toll" in cable_type:
        variations.append("FTC TOLL")
        variations.append("ftc toll")
        variations.append("FTC Toll")
        # Also add the reverse combination
        variations.append("TOLL FTC")
        variations.append("toll ftc")
        variations.append("Toll FTC")
    
    return variations

def highlight_cables_in_pdf(pdf_path, cable_types, output_path):
    """Highlight cable types in the PDF and generate a separate legend."""
    
    print(f"Highlighting cables in: {pdf_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    print(f"PDF has {len(doc)} pages")
    
    # Generate colors using fiber cable color order
    cable_colors = generate_fiber_cable_colors(cable_types)
    
    # Track highlighted positions to avoid duplicates
    highlighted_positions = set()
    
    # Statistics tracking
    total_matches = 0
    matches_by_cable = {cable: 0 for cable in cable_types}
    matches_by_variation = {}
    
    # Process each page (skip legend page if it exists)
    for page_num, page in enumerate(doc):
        # Skip the first page if it's a legend page
        if page_num == 0:
            page_text = page.get_text().upper()
            if "LEGEND" in page_text or "ENTITY MATCH" in page_text:
                print(f"Skipping legend page (page {page_num + 1})")
                continue
        
        print(f"Processing page {page_num + 1}")
        
        # Process each cable type
        for cable in cable_types:
            color = cable_colors[cable]
            
            # Get all variations for this cable
            all_variations = generate_cable_variations(cable)
            
            # Search for each variation
            for var in all_variations:
                # Search for this variation on the current page
                instances = page.search_for(var)
                
                for inst in instances:
                    # Create position key to avoid duplicates
                    rect_tuple = tuple(inst) + (page_num,)
                    
                    # Only highlight if we haven't highlighted this position before
                    if rect_tuple not in highlighted_positions:
                        # FIX: Apply coordinate system corrections
                        corrected_rect = correct_highlight_coordinates(page, inst)
                        
                        try:
                            # Method 1: Try highlight annotation with transparency
                            highlight = page.add_highlight_annot(corrected_rect)
                            # Set transparency for see-through effect
                            highlight.set_opacity(0.3)  # 30% opacity for transparency
                        except Exception as e:
                            print(f"Warning: Failed to create highlight annotation: {e}")
                            # Fallback to drawing rectangle with transparency
                            page.draw_rect(corrected_rect, color=color, width=0.25, fill=color, opacity=0.3)
                        
                        # Draw a subtle border for definition
                        page.draw_rect(corrected_rect, color=color, width=0.25)
                        total_matches += 1
                        matches_by_cable[cable] += 1
                        
                        # Track which variations were found
                        if var not in matches_by_variation:
                            matches_by_variation[var] = 0
                        matches_by_variation[var] += 1
                        
                        # Mark this position as highlighted
                        highlighted_positions.add(rect_tuple)
    
    print(f"Total matches found: {total_matches}")
    for cable, count in matches_by_cable.items():
        print(f"  - '{cable}': {count} matches")
    
    print("\nMatches by variation:")
    for var, count in matches_by_variation.items():
        original = generate_cable_variations(var) # Re-generate variations to get original
        print(f"  - '{var}' (of '{original}'): {count} matches")

    # Save the highlighted PDF (without legend page)
    doc.save(output_path)
    doc.close()
    
    print(f"Highlighted PDF saved to: {output_path}")
    
    # Create separate legend file
    legend_path = output_path.parent / f"{output_path.stem}_legend.pdf"
    create_separate_legend(legend_path, cable_colors, matches_by_cable, matches_by_variation, pdf_path)
    
    return legend_path

def generate_fiber_cable_colors(cable_types):
    """Generate colors using fiber cable color order."""
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
    
    cable_colors = {}
    for i, cable in enumerate(cable_types):
        if i < len(fiber_colors):
            cable_colors[cable] = fiber_colors[i]
        else:
            # Generate random color for additional cables
            cable_colors[cable] = [random.random() for _ in range(3)]
    
    return cable_colors

def create_separate_legend(legend_path, cable_colors, matches_by_cable, matches_by_variation, pdf_path):
    """Create a separate legend PDF file."""
    print(f"Creating separate legend file: {legend_path}")
    
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
        "CABLE MATCH LEGEND",
        fontsize=18, 
        color=(0, 0, 0)
    )
    legend_y += 30
    
    # Add metadata
    total_matches = sum(matches_by_cable.values())
    legend_page.insert_text(
        (legend_x, legend_y),
        f"PDF: {Path(pdf_path).name} | Total matches: {total_matches}",
        fontsize=12, 
        color=(0, 0, 0)
    )
    legend_y += 40
    
    # Add table header
    legend_page.insert_text((legend_x, legend_y), "Cable Type", fontsize=14, color=(0, 0, 0))
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
    
    # Sort cables by match count (descending)
    sorted_cables = sorted(
        matches_by_cable.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Add cable rows
    for cable, count in sorted_cables:
        if count == 0:
            continue  # Skip cables with no matches
            
        # Add cable name and count
        legend_page.insert_text(
            (legend_x, legend_y),
            f"{cable}",
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
            color=cable_colors[cable],
            fill=cable_colors[cable],
            width=0
        )
        legend_y += legend_spacing
        
        # Add variation details
        variations_found = [var for var, var_count in matches_by_variation.items() 
                          if var_count > 0 and any(cable in var for cable in generate_cable_variations(cable))]
        
        if variations_found:
            legend_page.insert_text(
                (legend_x + 20, legend_y),
                "Found as:",
                fontsize=10, 
                color=(0.3, 0.3, 0.3)
            )
            legend_y += legend_spacing - 5
            
            for var in variations_found[:5]:  # Limit to first 5 variations
                var_count = matches_by_variation.get(var, 0)
                legend_page.insert_text(
                    (legend_x + 40, legend_y),
                    f"'{var}' ({var_count} occurrences)",
                    fontsize=10, 
                    color=(0.3, 0.3, 0.3)
                )
                legend_y += legend_spacing - 5
            
            if len(variations_found) > 5:
                legend_page.insert_text(
                    (legend_x + 40, legend_y),
                    f"... and {len(variations_found) - 5} more variations",
                    fontsize=10, 
                    color=(0.3, 0.3, 0.3)
                )
                legend_y += legend_spacing - 5
        
        legend_y += 5  # Extra space between cables
    
    # Save the legend PDF
    legend_doc.save(legend_path)
    legend_doc.close()
    
    print(f"Legend PDF saved to: {legend_path}")

def main():
    # Define file paths
    tables_folder = Path(__file__).parent
    csv_path = tables_folder / 'WIM_Equipment.csv'
    pdf_path = tables_folder / 'M01-17-ITS-Tolling-2b.pdf'
    output_pdf_path = tables_folder / 'Highlighted_Plansheets_with_variations.pdf'

    # Extract cable types from the CSV
    cable_types = extract_cable_types(csv_path)
    print(f"Extracted {len(cable_types)} cable types from CSV")
    
    # Filter out empty strings
    cable_types = [cable for cable in cable_types if cable.strip()]
    print(f"After filtering: {len(cable_types)} valid cable types")
    
    if not cable_types:
        print("No valid cable types found in CSV. Please check the file format.")
        return
        
    # Check if the PDF exists
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return
    
    # Highlight cables in the PDF
    legend_path = highlight_cables_in_pdf(pdf_path, cable_types, output_pdf_path)

    print(f"Highlighted PDF saved to: {output_pdf_path}")
    print(f"Legend PDF saved to: {legend_path}")
    print("Please open the PDF to check the highlighting results.")

if __name__ == "__main__":
    main()
