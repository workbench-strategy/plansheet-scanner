#!/usr/bin/env python3
"""
Smart Plan Seamer - Main Integration Module
"""

import logging
import csv
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import fitz  # PyMuPDF

from .north_detector import NorthDetector
from .content_cropper import ContentCropper
from .edge_matcher import EdgeMatcher, EdgeMatch
from .canvas_scaler import CanvasScaler, CanvasLimits

@dataclass
class SheetInfo:
    """Information about a processed sheet."""
    sheet_id: str
    page_num: int
    rotate_deg: float
    crop_bbox_in: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    placed: Optional[Tuple[float, float]]  # (x_in, y_in) or None if unplaced
    w_in: float
    h_in: float
    neighbor_id: Optional[str] = None
    side_a: Optional[str] = None
    side_b: Optional[str] = None
    dx_in: Optional[float] = None
    dy_in: Optional[float] = None
    match_score: Optional[float] = None
    notes: str = ""

@dataclass
class SeamingResult:
    """Result of the smart plan seaming process."""
    sheets: List[SheetInfo]
    matches: List[EdgeMatch]
    canvas_limits: CanvasLimits
    output_pdf_path: str
    qa_report_path: str

class SmartPlanSeamer:
    """Main smart plan seamer that integrates all components."""
    
    def __init__(self, 
                 north_template_path: Optional[str] = None,
                 assume_north_up: bool = False,
                 dpi_correlation: int = 120,
                 crop_content: bool = True,
                 max_canvas_inches: Tuple[float, float] = (100.0, 100.0),
                 correlation_threshold: float = 0.35,
                 stripe_width_inches: float = 0.8,
                 text_hints: bool = True):
        """
        Initialize the smart plan seamer.
        
        Args:
            north_template_path: Path to north arrow template
            assume_north_up: Skip north detection if True
            dpi_correlation: DPI for correlation operations
            crop_content: Whether to crop content boundaries
            max_canvas_inches: Maximum canvas size (width, height)
            correlation_threshold: Minimum correlation score for matches
            stripe_width_inches: Width of edge stripes for correlation
            text_hints: Whether to use text hints for matchline detection
        """
        self.north_template_path = north_template_path
        self.assume_north_up = assume_north_up
        self.dpi_correlation = dpi_correlation
        self.crop_content = crop_content
        self.max_canvas_inches = max_canvas_inches
        self.correlation_threshold = correlation_threshold
        self.stripe_width_inches = stripe_width_inches
        self.text_hints = text_hints
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.north_detector = NorthDetector(
            template_path=north_template_path,
            dpi=dpi_correlation
        )
        
        self.content_cropper = ContentCropper(
            dpi=dpi_correlation,
            padding_inches=0.2,
            white_threshold=245
        )
        
        self.edge_matcher = EdgeMatcher(
            dpi=dpi_correlation,
            stripe_width_inches=stripe_width_inches,
            threshold=correlation_threshold,
            text_hints=text_hints
        )
        
        self.canvas_scaler = CanvasScaler(
            max_width_inches=max_canvas_inches[0],
            max_height_inches=max_canvas_inches[1]
        )
    
    def process_sheets(self, pdf_path: str, output_dir: str, 
                      dry_run: bool = False) -> SeamingResult:
        """
        Process plan sheets through the complete pipeline.
        
        Args:
            pdf_path: Path to input PDF
            output_dir: Output directory for results
            dry_run: If True, don't create output PDF
            
        Returns:
            SeamingResult with all processing information
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing PDF: {pdf_path}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Dry run: {dry_run}")
        
        # Load PDF
        doc = fitz.open(pdf_path)
        pages = [doc.load_page(i) for i in range(doc.page_count)]
        
        self.logger.info(f"Loaded {len(pages)} pages")
        
        # Step 1: North detection and rotation
        rotations = self._detect_north_rotations(pages)
        
        # Step 2: Content cropping
        crop_results = self._crop_content(pages, rotations)
        
        # Step 3: Edge matching
        matches = self._find_edge_matches(pages, rotations, crop_results)
        
        # Step 4: Layout calculation
        sheet_infos = self._calculate_layout(pages, rotations, crop_results, matches)
        
        # Step 5: Canvas scaling
        canvas_limits = self._calculate_canvas_scaling(sheet_infos)
        
        # Step 6: Generate output
        output_pdf_path = str(output_path / "mosaic.pdf")
        qa_report_path = str(output_path / "qa_report.csv")
        
        if not dry_run:
            self._create_output_pdf(pages, rotations, crop_results, canvas_limits, output_pdf_path, sheet_infos)
        
        # Step 7: Generate QA report
        self._generate_qa_report(sheet_infos, matches, canvas_limits, qa_report_path)
        
        doc.close()
        
        return SeamingResult(
            sheets=sheet_infos,
            matches=matches,
            canvas_limits=canvas_limits,
            output_pdf_path=output_pdf_path,
            qa_report_path=qa_report_path
        )
    
    def _detect_north_rotations(self, pages: List[fitz.Page]) -> List[float]:
        """Detect north rotations for all pages."""
        self.logger.info("Detecting north rotations...")
        
        rotations = []
        for i, page in enumerate(pages):
            if self.assume_north_up:
                rotation = 0.0
                self.logger.info(f"Page {i+1}: Assuming north up (0°)")
            else:
                rotation = self.north_detector.detect_north_rotation(page)
                self.logger.info(f"Page {i+1}: Detected rotation {rotation}°")
            
            rotations.append(rotation)
        
        return rotations
    
    def _crop_content(self, pages: List[fitz.Page], 
                     rotations: List[float]) -> List[Tuple[float, float]]:
        """Crop content boundaries for all pages."""
        self.logger.info("Cropping content boundaries...")
        
        if not self.crop_content:
            # Return full page dimensions
            results = []
            for page in pages:
                width_pt = page.rect.width
                height_pt = page.rect.height
                width_in = width_pt / 72.0  # Convert points to inches
                height_in = height_pt / 72.0
                results.append((width_in, height_in))
            
            self.logger.info("Content cropping disabled - using full page dimensions")
            return results
        
        # Apply rotations and crop
        rotated_pages = []
        for page, rotation in zip(pages, rotations):
            if rotation != 0:
                # Create rotation matrix
                matrix = fitz.Matrix(1, 1)
                if rotation == 90:
                    matrix = fitz.Matrix(0, 1, -1, 0, 0, 0)
                elif rotation == 180:
                    matrix = fitz.Matrix(-1, 0, 0, -1, 0, 0)
                elif rotation == 270:
                    matrix = fitz.Matrix(0, -1, 1, 0, 0, 0)
                # For now, just use original page (rotation would need more complex handling)
                rotated_pages.append(page)
            else:
                rotated_pages.append(page)
        
        crop_results = self.content_cropper.crop_page_batch(rotated_pages, False)
        
        self.logger.info(f"Cropped {len(crop_results)} pages")
        return crop_results
    
    def _find_edge_matches(self, pages: List[fitz.Page], 
                          rotations: List[float],
                          crop_results: List[Tuple[float, float]]) -> List[EdgeMatch]:
        """Find edge matches between pages."""
        self.logger.info("Finding edge matches...")
        
        matches = self.edge_matcher.find_matches(pages, rotations, crop_results)
        
        self.logger.info(f"Found {len(matches)} edge matches")
        return matches
    
    def _calculate_layout(self, pages: List[fitz.Page],
                         rotations: List[float],
                         crop_results: List[Tuple[float, float]],
                         matches: List[EdgeMatch]) -> List[SheetInfo]:
        """Calculate layout positions for all sheets."""
        self.logger.info("Calculating layout...")
        
        sheet_infos = []
        
        # Create adjacency graph from matches
        adjacency = {}
        for match in matches:
            if match.sheet_a not in adjacency:
                adjacency[match.sheet_a] = []
            if match.sheet_b not in adjacency:
                adjacency[match.sheet_b] = []
            
            adjacency[match.sheet_a].append((match.sheet_b, match))
            adjacency[match.sheet_b].append((match.sheet_a, match))
        
        # Simple layout algorithm: place sheets in a grid
        placed_sheets = set()
        positions = {}
        
        # Start with first sheet at origin
        if pages:
            positions[0] = (0.0, 0.0)
            placed_sheets.add(0)
        
        # Place connected sheets
        for match in matches:
            if match.sheet_a in placed_sheets and match.sheet_b not in placed_sheets:
                # Place sheet_b relative to sheet_a
                pos_a = positions[match.sheet_a]
                if match.side_a == "right" and match.side_b == "left":
                    pos_b = (pos_a[0] + crop_results[match.sheet_a][0], pos_a[1])
                elif match.side_a == "left" and match.side_b == "right":
                    pos_b = (pos_a[0] - crop_results[match.sheet_b][0], pos_a[1])
                elif match.side_a == "bottom" and match.side_b == "top":
                    pos_b = (pos_a[0], pos_a[1] + crop_results[match.sheet_a][1])
                elif match.side_a == "top" and match.side_b == "bottom":
                    pos_b = (pos_a[0], pos_a[1] - crop_results[match.sheet_b][1])
                else:
                    # Fallback: place to the right
                    pos_b = (pos_a[0] + crop_results[match.sheet_a][0], pos_a[1])
                
                positions[match.sheet_b] = pos_b
                placed_sheets.add(match.sheet_b)
        
        # Create sheet infos
        for i, (page, rotation, (width, height)) in enumerate(zip(pages, rotations, crop_results)):
            # Find match info for this sheet
            match_info = None
            for match in matches:
                if match.sheet_a == i or match.sheet_b == i:
                    match_info = match
                    break
            
            # Determine placement
            if i in positions:
                placed = positions[i]
                notes = ""
            else:
                placed = None
                notes = "unplaced_fallback"
            
            # Create crop bbox (simplified - would need actual crop rect)
            crop_bbox = (0.0, 0.0, width, height)
            
            sheet_info = SheetInfo(
                sheet_id=f"sheet_{i+1}",
                page_num=i+1,
                rotate_deg=rotation,
                crop_bbox_in=crop_bbox,
                placed=placed,
                w_in=width,
                h_in=height,
                neighbor_id=match_info.sheet_b if match_info and match_info.sheet_a == i else 
                          match_info.sheet_a if match_info and match_info.sheet_b == i else None,
                side_a=match_info.side_a if match_info and match_info.sheet_a == i else 
                       match_info.side_b if match_info and match_info.sheet_b == i else None,
                side_b=match_info.side_b if match_info and match_info.sheet_a == i else 
                       match_info.side_a if match_info and match_info.sheet_b == i else None,
                dx_in=match_info.offset_x if match_info else None,
                dy_in=match_info.offset_y if match_info else None,
                match_score=match_info.score if match_info else None,
                notes=notes
            )
            
            sheet_infos.append(sheet_info)
        
        self.logger.info(f"Calculated layout for {len(sheet_infos)} sheets")
        return sheet_infos
    
    def _calculate_canvas_scaling(self, sheet_infos: List[SheetInfo]) -> CanvasLimits:
        """Calculate canvas scaling based on sheet layout."""
        self.logger.info("Calculating canvas scaling...")
        
        # Find bounding box of all placed sheets
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for sheet in sheet_infos:
            if sheet.placed:
                x, y = sheet.placed
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + sheet.w_in)
                max_y = max(max_y, y + sheet.h_in)
        
        if min_x == float('inf'):
            # No sheets placed, use default
            width_inches = 24.0
            height_inches = 36.0
        else:
            width_inches = max_x - min_x
            height_inches = max_y - min_y
        
        canvas_limits = self.canvas_scaler.calculate_scaling(width_inches, height_inches)
        
        return canvas_limits
    
    def _create_output_pdf(self, pages: List[fitz.Page],
                          rotations: List[float],
                          crop_results: List[Tuple[float, float]],
                          canvas_limits: CanvasLimits,
                          output_path: str,
                          sheet_infos: List[SheetInfo]):
        """Create the output PDF mosaic."""
        self.logger.info("Creating output PDF mosaic...")
        
        # Create new document
        doc = fitz.open()
        
        # Calculate canvas size in points (72 points per inch)
        canvas_width_pt = canvas_limits.scaled_width_inches * 72.0
        canvas_height_pt = canvas_limits.scaled_height_inches * 72.0
        
        # Create a single page for the mosaic
        mosaic_page = doc.new_page(width=canvas_width_pt, height=canvas_height_pt)
        
        # Find the minimum coordinates to normalize to positive space
        min_x = min_y = float('inf')
        for sheet_info in sheet_infos:
            if sheet_info.placed:
                x, y = sheet_info.placed
                min_x = min(min_x, x)
                min_y = min(min_y, y)
        
        if min_x == float('inf'):
            min_x = min_y = 0.0
        
        # Place each sheet at its calculated position
        for i, sheet_info in enumerate(sheet_infos):
            if sheet_info.placed:
                x, y = sheet_info.placed
                
                # Normalize coordinates to positive space
                x_norm = x - min_x
                y_norm = y - min_y
                
                # Convert inches to points
                x_pt = x_norm * 72.0
                y_pt = y_norm * 72.0
                
                # Get the original page
                page = pages[i]
                
                # Apply rotation if needed
                if rotations[i] != 0:
                    # Create rotation matrix
                    matrix = fitz.Matrix(1, 1)
                    if rotations[i] == 90:
                        matrix = fitz.Matrix(0, 1, -1, 0, 0, 0)
                    elif rotations[i] == 180:
                        matrix = fitz.Matrix(-1, 0, 0, -1, 0, 0)
                    elif rotations[i] == 270:
                        matrix = fitz.Matrix(0, -1, 1, 0, 0, 0)
                else:
                    matrix = fitz.Matrix(1, 1)
                
                # Apply scaling
                scale_matrix = fitz.Matrix(canvas_limits.scale_factor, canvas_limits.scale_factor)
                final_matrix = scale_matrix * matrix
                
                # Create rectangle for placement
                page_width_pt = sheet_info.w_in * 72.0 * canvas_limits.scale_factor
                page_height_pt = sheet_info.h_in * 72.0 * canvas_limits.scale_factor
                rect = fitz.Rect(x_pt, y_pt, x_pt + page_width_pt, y_pt + page_height_pt)
                
                # Place the page content
                mosaic_page.show_pdf_page(rect, page.parent, page.number, matrix=final_matrix)
                
                self.logger.debug(f"Placed sheet {i+1} at ({x_norm:.2f}, {y_norm:.2f})")
        
        # Save the mosaic PDF
        doc.save(output_path)
        doc.close()
        
        self.logger.info(f"Output PDF mosaic created: {output_path}")
    
    def _generate_qa_report(self, sheet_infos: List[SheetInfo],
                           matches: List[EdgeMatch],
                           canvas_limits: CanvasLimits,
                           qa_report_path: str):
        """Generate QA CSV report."""
        self.logger.info("Generating QA report...")
        
        with open(qa_report_path, 'w', newline='') as csvfile:
            fieldnames = [
                'sheet_id', 'page_num', 'rotate_deg', 'crop_bbox_in(x0,y0,x1,y1)',
                'placed(x_in,y_in)', 'w_in', 'h_in',
                'neighbor_id', 'side_a', 'side_b', 'dx_in', 'dy_in', 'match_score',
                'notes'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for sheet in sheet_infos:
                row = {
                    'sheet_id': sheet.sheet_id,
                    'page_num': sheet.page_num,
                    'rotate_deg': sheet.rotate_deg,
                    'crop_bbox_in(x0,y0,x1,y1)': f"({sheet.crop_bbox_in[0]:.2f},{sheet.crop_bbox_in[1]:.2f},{sheet.crop_bbox_in[2]:.2f},{sheet.crop_bbox_in[3]:.2f})",
                    'placed(x_in,y_in)': f"({sheet.placed[0]:.2f},{sheet.placed[1]:.2f})" if sheet.placed else "None",
                    'w_in': sheet.w_in,
                    'h_in': sheet.h_in,
                    'neighbor_id': sheet.neighbor_id,
                    'side_a': sheet.side_a,
                    'side_b': sheet.side_b,
                    'dx_in': f"{sheet.dx_in:.2f}" if sheet.dx_in is not None else "None",
                    'dy_in': f"{sheet.dy_in:.2f}" if sheet.dy_in is not None else "None",
                    'match_score': f"{sheet.match_score:.3f}" if sheet.match_score is not None else "None",
                    'notes': sheet.notes
                }
                writer.writerow(row)
        
        self.logger.info(f"QA report generated: {qa_report_path}")
