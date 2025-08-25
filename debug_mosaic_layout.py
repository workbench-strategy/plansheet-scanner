#!/usr/bin/env python3
"""
Debug Mosaic Layout - Implements user's strategy:
1. North-first rotation (all sheets north-up)
2. Sheet grouping by names/matchlines
3. Detailed debugging of layout logic
4. Connectivity mapping
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.core.edge_matcher_fixed import EdgeMatcher, EdgeMatch

def setup_logging():
    """Set up detailed logging for debugging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug_mosaic_layout.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_sheets(pdf_path: str, max_sheets: int = 5) -> Tuple[List[Dict], fitz.Document]:
    """Load sheets from PDF file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading up to {max_sheets} sheets from {pdf_path}")
    
    doc = fitz.open(pdf_path)
    sheets = []
    
    for i in range(min(len(doc), max_sheets)):
        page = doc[i]
        rect = page.rect
        width_in = rect.width / 72.0
        height_in = rect.height / 72.0
        
        sheet_info = {
            'page': page,
            'width_in': width_in,
            'height_in': height_in,
            'page_index': i,
            'doc_index': 0,
            'sheet_name': None,
            'page_number': None,
            'matchlines': {},  # Will store matchline info for each edge
            'north_rotation': 0.0  # Will store detected north rotation
        }
        
        sheets.append(sheet_info)
        logger.info(f"Loaded sheet {i+1}: {width_in:.1f}\" x {height_in:.1f}\"")
    
    return sheets, doc

def detect_north_rotation(page: fitz.Page) -> float:
    """
    Detect north rotation for a page.
    For now, return 0.0 (assume north-up) - we can enhance this later.
    """
    # TODO: Implement actual north arrow detection
    # For debugging, assume all sheets are north-up
    return 0.0

def extract_sheet_info(sheets: List[Dict], matcher: EdgeMatcher) -> None:
    """
    Extract sheet names, page numbers, and matchlines for all sheets.
    This creates the connectivity map.
    """
    logger = logging.getLogger(__name__)
    logger.info("=== EXTRACTING SHEET INFORMATION ===")
    
    for i, sheet in enumerate(sheets):
        logger.info(f"\n--- Processing Sheet {i} ---")
        
        # Extract sheet name and page number
        sheet['sheet_name'] = matcher._extract_sheet_name(sheet['page'])
        sheet['page_number'] = matcher._extract_page_number(sheet['page'])
        
        logger.info(f"Sheet {i}: name='{sheet['sheet_name']}', page={sheet['page_number']}")
        
        # Detect north rotation
        sheet['north_rotation'] = detect_north_rotation(sheet['page'])
        logger.info(f"Sheet {i}: north rotation = {sheet['north_rotation']}°")
        
        # Extract matchlines for each edge
        for edge in ['left', 'right', 'top', 'bottom']:
            matchline_info = matcher._extract_matchline_info(sheet['page'], edge)
            sheet['matchlines'][edge] = matchline_info
            
            if matchline_info.get('has_matchline'):
                target_sheets = matchline_info.get('target_sheets', [])
                target_pages = matchline_info.get('target_pages', [])
                logger.info(f"Sheet {i} {edge} edge: targets sheets {target_sheets}, pages {target_pages}")
                
                # Show raw text for debugging
                raw_text = matchline_info.get('raw_text', '')[:100]
                if raw_text:
                    logger.debug(f"Sheet {i} {edge} edge text: '{raw_text}...'")

def create_connectivity_map(sheets: List[Dict]) -> Dict:
    """
    Create a connectivity map showing which sheets should connect to which.
    Returns a dictionary mapping sheet names to their target connections.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n=== CREATING CONNECTIVITY MAP ===")
    
    connectivity = {}
    
    for i, sheet in enumerate(sheets):
        sheet_name = sheet['sheet_name']
        if not sheet_name:
            logger.warning(f"Sheet {i} has no sheet name, skipping connectivity")
            continue
            
        connectivity[sheet_name] = {
            'sheet_index': i,
            'targets': {},  # edge -> [target_sheets]
            'targeted_by': set()  # sheets that target this one
        }
        
        logger.info(f"\n--- Connectivity for {sheet_name} (Sheet {i}) ---")
        
        # Find all targets for this sheet
        for edge, matchline_info in sheet['matchlines'].items():
            if matchline_info.get('has_matchline'):
                target_sheets = matchline_info.get('target_sheets', [])
                target_pages = matchline_info.get('target_pages', [])
                
                if target_sheets or target_pages:
                    connectivity[sheet_name]['targets'][edge] = {
                        'sheets': target_sheets,
                        'pages': target_pages
                    }
                    logger.info(f"  {edge} edge targets: sheets={target_sheets}, pages={target_pages}")
        
        # Find sheets that target this one
        for j, other_sheet in enumerate(sheets):
            if i == j:
                continue
                
            other_name = other_sheet['sheet_name']
            if not other_name:
                continue
                
            for edge, matchline_info in other_sheet['matchlines'].items():
                if matchline_info.get('has_matchline'):
                    target_sheets = matchline_info.get('target_sheets', [])
                    if sheet_name in target_sheets:
                        connectivity[sheet_name]['targeted_by'].add(other_name)
                        logger.info(f"  Targeted by {other_name} on {edge} edge")
    
    return connectivity

def analyze_connectivity(connectivity: Dict) -> None:
    """
    Analyze the connectivity map and suggest placement strategy.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n=== CONNECTIVITY ANALYSIS ===")
    
    # Find sheets with most connections (potential hubs)
    connection_counts = {}
    for sheet_name, info in connectivity.items():
        outgoing = len(info['targets'])
        incoming = len(info['targeted_by'])
        total = outgoing + incoming
        connection_counts[sheet_name] = total
        
        logger.info(f"{sheet_name}: {outgoing} outgoing, {incoming} incoming, {total} total connections")
    
    # Find the best starting sheet (most connections)
    if connection_counts:
        best_sheet = max(connection_counts.items(), key=lambda x: x[1])
        logger.info(f"\nBest starting sheet: {best_sheet[0]} with {best_sheet[1]} connections")
    
    # Find isolated sheets
    isolated = [name for name, count in connection_counts.items() if count == 0]
    if isolated:
        logger.warning(f"Isolated sheets (no connections): {isolated}")
    
    # Find sheets with multiple matchlines (hubs)
    hubs = []
    for sheet_name, info in connectivity.items():
        if len(info['targets']) > 1:
            hubs.append(sheet_name)
            logger.info(f"Hub sheet {sheet_name} has {len(info['targets'])} outgoing connections")
    
    if hubs:
        logger.info(f"Hub sheets: {hubs}")

def debug_layout_attempt(sheets: List[Dict], connectivity: Dict, matches: List[EdgeMatch]) -> None:
    """
    Debug the layout attempt and show why sheets aren't being placed.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n=== DEBUGGING LAYOUT ATTEMPT ===")
    
    # Show all matches found
    logger.info(f"Found {len(matches)} matches:")
    for match in matches:
        sheet_a_name = sheets[match.sheet_a].get('sheet_name', f'Sheet{match.sheet_a}')
        sheet_b_name = sheets[match.sheet_b].get('sheet_name', f'Sheet{match.sheet_b}')
        logger.info(f"  {sheet_a_name} {match.side_a} -> {sheet_b_name} {match.side_b}: score={match.score:.3f}")
    
    # Show placement logic
    logger.info("\n--- Placement Logic ---")
    
    # Start with first sheet
    placed = {0}
    positions = {0: (0, 0)}
    
    logger.info(f"Starting with sheet 0 at position (0, 0)")
    
    # Try to place other sheets
    for match in matches:
        if match.sheet_a in placed and match.sheet_b not in placed:
            base_pos = positions[match.sheet_a]
            new_pos = (base_pos[0] + match.offset_x, base_pos[1] + match.offset_y)
            
            sheet_a_name = sheets[match.sheet_a].get('sheet_name', f'Sheet{match.sheet_a}')
            sheet_b_name = sheets[match.sheet_b].get('sheet_name', f'Sheet{match.sheet_b}')
            
            logger.info(f"Placing {sheet_b_name} relative to {sheet_a_name}:")
            logger.info(f"  Base position: {base_pos}")
            logger.info(f"  Offset: ({match.offset_x:.1f}, {match.offset_y:.1f})")
            logger.info(f"  New position: {new_pos}")
            
            positions[match.sheet_b] = new_pos
            placed.add(match.sheet_b)
            
        elif match.sheet_b in placed and match.sheet_a not in placed:
            # Reverse direction
            base_pos = positions[match.sheet_b]
            new_pos = (base_pos[0] - match.offset_x, base_pos[1] - match.offset_y)
            
            sheet_a_name = sheets[match.sheet_a].get('sheet_name', f'Sheet{match.sheet_a}')
            sheet_b_name = sheets[match.sheet_b].get('sheet_name', f'Sheet{match.sheet_b}')
            
            logger.info(f"Placing {sheet_a_name} relative to {sheet_b_name}:")
            logger.info(f"  Base position: {base_pos}")
            logger.info(f"  Offset: (-{match.offset_x:.1f}, -{match.offset_y:.1f})")
            logger.info(f"  New position: {new_pos}")
            
            positions[match.sheet_a] = new_pos
            placed.add(match.sheet_a)
    
    # Show final placement
    logger.info(f"\n--- Final Placement ---")
    logger.info(f"Placed sheets: {placed}")
    for i, pos in positions.items():
        sheet_name = sheets[i].get('sheet_name', f'Sheet{i}')
        logger.info(f"  {sheet_name}: position {pos}")
    
    # Show unplaced sheets
    unplaced = set(range(len(sheets))) - placed
    if unplaced:
        logger.warning(f"Unplaced sheets: {unplaced}")
        for i in unplaced:
            sheet_name = sheets[i].get('sheet_name', f'Sheet{i}')
            logger.warning(f"  {sheet_name} (index {i}) was not placed")

def create_debug_visualization(sheets: List[Dict], connectivity: Dict, matches: List[EdgeMatch], 
                             output_path: str = "debug_mosaic_layout.png") -> None:
    """
    Create a detailed debug visualization showing connectivity and placement.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating debug visualization")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Plot 1: Sheet connectivity network
    ax1.set_title("Sheet Connectivity Network", fontsize=14, fontweight='bold')
    G = nx.DiGraph()
    
    # Add nodes
    for sheet_name, info in connectivity.items():
        G.add_node(sheet_name, index=info['sheet_index'])
    
    # Add edges
    for sheet_name, info in connectivity.items():
        for edge, targets in info['targets'].items():
            for target_sheet in targets['sheets']:
                if target_sheet in connectivity:
                    G.add_edge(sheet_name, target_sheet, edge=edge)
    
    # Draw network
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=10, font_weight='bold',
            arrows=True, edge_color='gray', arrowsize=20)
    
    # Plot 2: Sheet information table
    ax2.set_title("Sheet Information", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    table_data = []
    for i, sheet in enumerate(sheets):
        sheet_name = sheet.get('sheet_name', f'Sheet{i}')
        page_num = sheet.get('page_number', 'None')
        north_rot = sheet.get('north_rotation', 0.0)
        table_data.append([f'Sheet {i}', sheet_name, page_num, f'{north_rot}°'])
    
    table = ax2.table(cellText=table_data, 
                     colLabels=['Index', 'Name', 'Page', 'North Rot'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Plot 3: Matchline analysis
    ax3.set_title("Matchline Analysis", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    matchline_data = []
    for i, sheet in enumerate(sheets):
        sheet_name = sheet.get('sheet_name', f'Sheet{i}')
        for edge, matchline_info in sheet['matchlines'].items():
            if matchline_info.get('has_matchline'):
                targets = matchline_info.get('target_sheets', [])
                pages = matchline_info.get('target_pages', [])
                matchline_data.append([sheet_name, edge, str(targets), str(pages)])
    
    if matchline_data:
        matchline_table = ax3.table(cellText=matchline_data,
                                   colLabels=['Sheet', 'Edge', 'Target Sheets', 'Target Pages'],
                                   cellLoc='left', loc='center')
        matchline_table.auto_set_font_size(False)
        matchline_table.set_fontsize(8)
        matchline_table.scale(1, 1.5)
    else:
        ax3.text(0.5, 0.5, 'No matchlines found', ha='center', va='center', fontsize=12)
    
    # Plot 4: Placement attempt
    ax4.set_title("Placement Attempt", fontsize=14, fontweight='bold')
    
    # Calculate canvas size
    max_x = max(s['width_in'] for s in sheets)
    max_y = max(s['height_in'] for s in sheets)
    canvas_width = max_x * len(sheets) * 2
    canvas_height = max_y * len(sheets) * 2
    
    ax4.set_xlim(-canvas_width/2, canvas_width/2)
    ax4.set_ylim(-canvas_height/2, canvas_height/2)
    ax4.grid(True, alpha=0.3)
    
    # Place sheets based on matches
    positions = {}
    placed = set()
    
    # Start with first sheet
    if sheets:
        positions[0] = (0, 0)
        placed.add(0)
        
        sheet = sheets[0]
        rect = patches.Rectangle((0, 0), sheet['width_in'], sheet['height_in'], 
                               linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7)
        ax4.add_patch(rect)
        ax4.text(sheet['width_in']/2, sheet['height_in']/2, f"Sheet 0\n{sheet.get('sheet_name', 'Unknown')}", 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Place other sheets
    for match in matches:
        if match.sheet_a in placed and match.sheet_b not in placed:
            base_pos = positions[match.sheet_a]
            new_pos = (base_pos[0] + match.offset_x, base_pos[1] + match.offset_y)
            positions[match.sheet_b] = new_pos
            placed.add(match.sheet_b)
            
            sheet = sheets[match.sheet_b]
            rect = patches.Rectangle(new_pos, sheet['width_in'], sheet['height_in'], 
                                   linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.7)
            ax4.add_patch(rect)
            ax4.text(new_pos[0] + sheet['width_in']/2, new_pos[1] + sheet['height_in']/2, 
                    f"Sheet {match.sheet_b}\n{sheet.get('sheet_name', 'Unknown')}", 
                    ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Draw connection line
            ax4.plot([base_pos[0] + sheets[match.sheet_a]['width_in']/2, 
                     new_pos[0] + sheet['width_in']/2], 
                    [base_pos[1] + sheets[match.sheet_a]['height_in']/2, 
                     new_pos[1] + sheet['height_in']/2], 
                    'r-', linewidth=2, alpha=0.8)
    
    # Show unplaced sheets
    unplaced = set(range(len(sheets))) - placed
    if unplaced:
        y_offset = max_y + 2
        for i in unplaced:
            sheet = sheets[i]
            rect = patches.Rectangle((-max_x, y_offset), sheet['width_in'], sheet['height_in'], 
                                   linewidth=2, edgecolor='orange', facecolor='lightyellow', alpha=0.7)
            ax4.add_patch(rect)
            ax4.text(-max_x + sheet['width_in']/2, y_offset + sheet['height_in']/2, 
                    f"Sheet {i}\n{sheet.get('sheet_name', 'Unknown')}\n(UNPLACED)", 
                    ha='center', va='center', fontsize=10, fontweight='bold')
            y_offset += sheet['height_in'] + 1
    
    ax4.set_xlabel("X (inches)")
    ax4.set_ylabel("Y (inches)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Debug visualization saved to: {output_path}")
    plt.show()

def main():
    """Main debug function."""
    parser = argparse.ArgumentParser(description="Debug mosaic layout with detailed analysis")
    parser.add_argument("--pdf", default="scripts/plans.pdf", help="Path to PDF file")
    parser.add_argument("--sheets", type=int, default=5, help="Number of sheets to test")
    parser.add_argument("--output", default="debug_mosaic_layout.png", help="Output image path")
    parser.add_argument("--dpi", type=int, default=120, help="DPI for correlation")
    parser.add_argument("--threshold", type=float, default=0.35, help="Correlation threshold")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting debug mosaic layout analysis")
    
    # Check if PDF exists
    if not os.path.exists(args.pdf):
        logger.error(f"PDF file not found: {args.pdf}")
        return 1
    
    doc = None
    try:
        # Load sheets
        sheets, doc = load_sheets(args.pdf, args.sheets)
        if not sheets:
            logger.error("No sheets loaded")
            return 1
        
        # Create edge matcher
        matcher = EdgeMatcher(
            dpi=args.dpi,
            threshold=args.threshold,
            text_hints=True
        )
        
        # Extract sheet information
        extract_sheet_info(sheets, matcher)
        
        # Create connectivity map
        connectivity = create_connectivity_map(sheets)
        
        # Analyze connectivity
        analyze_connectivity(connectivity)
        
        # Find matches
        logger.info("\n=== FINDING EDGE MATCHES ===")
        matches = matcher.find_matches(sheets)
        
        # Debug layout attempt
        debug_layout_attempt(sheets, connectivity, matches)
        
        # Create debug visualization
        create_debug_visualization(sheets, connectivity, matches, args.output)
        
        logger.info("Debug analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Debug analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up document
        if doc:
            doc.close()

if __name__ == "__main__":
    exit(main())

