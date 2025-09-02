"""
Advanced Geospatial Processing System

This module provides enhanced geospatial processing capabilities including:
- Multi-format support (GeoJSON, Shapefile, KML, KMZ, etc.)
- Advanced coordinate transformations and projections
- Spatial analysis (distance calculations, intersections, etc.)
- 3D visualization capabilities
- GIS integration with external systems

Author: Plansheet Scanner Team
Date: 2024
"""

import json
import logging
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Geospatial libraries
try:
    import geopandas as gpd
    import pyproj
    from pyproj import CRS, Transformer
    from shapely.affinity import rotate, scale, translate
    from shapely.geometry import LineString, MultiPolygon, Point, Polygon
    from shapely.ops import transform

    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    logger.warning(
        "Geospatial libraries not available. Install geopandas, shapely, and pyproj for full functionality."
    )

# 3D visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install plotly for 3D visualization.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GeospatialPoint:
    """Represents a geospatial point with metadata."""

    x: float
    y: float
    z: Optional[float] = None
    crs: str = "EPSG:4326"  # Default to WGS84
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeospatialLine:
    """Represents a geospatial line with metadata."""

    points: List[GeospatialPoint]
    length: Optional[float] = None
    crs: str = "EPSG:4326"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeospatialPolygon:
    """Represents a geospatial polygon with metadata."""

    exterior: List[GeospatialPoint]
    holes: List[List[GeospatialPoint]] = field(default_factory=list)
    area: Optional[float] = None
    crs: str = "EPSG:4326"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialAnalysisResult:
    """Result of spatial analysis operations."""

    analysis_type: str
    result: Union[float, bool, List[Any]]
    units: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinateTransformation:
    """Represents a coordinate transformation."""

    source_crs: str
    target_crs: str
    transformation_matrix: Optional[np.ndarray] = None
    accuracy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedGeospatialProcessor:
    """Advanced geospatial processing system."""

    def __init__(self):
        if not GEOSPATIAL_AVAILABLE:
            raise ImportError(
                "Geospatial libraries required. Install geopandas, shapely, and pyproj."
            )

        self.supported_formats = {
            "geojson": self._read_geojson,
            "shapefile": self._read_shapefile,
            "kml": self._read_kml,
            "kmz": self._read_kmz,
            "gpx": self._read_gpx,
            "csv": self._read_csv,
        }

        self.common_crs = {
            "WGS84": "EPSG:4326",
            "UTM_10N": "EPSG:32610",
            "UTM_11N": "EPSG:32611",
            "NAD83": "EPSG:4269",
            "NAD27": "EPSG:4267",
            "Web_Mercator": "EPSG:3857",
        }

        logger.info("AdvancedGeospatialProcessor initialized")

    def read_file(self, file_path: str) -> gpd.GeoDataFrame:
        """
        Read a geospatial file in any supported format.

        Args:
            file_path: Path to the geospatial file

        Returns:
            GeoDataFrame containing the geospatial data
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()

        if file_extension == ".geojson":
            return self._read_geojson(file_path)
        elif file_extension == ".shp":
            return self._read_shapefile(file_path)
        elif file_extension == ".kml":
            return self._read_kml(file_path)
        elif file_extension == ".kmz":
            return self._read_kmz(file_path)
        elif file_extension == ".gpx":
            return self._read_gpx(file_path)
        elif file_extension == ".csv":
            return self._read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _read_geojson(self, file_path: Path) -> gpd.GeoDataFrame:
        """Read GeoJSON file."""
        try:
            gdf = gpd.read_file(file_path)
            logger.info(f"Loaded GeoJSON file: {file_path}")
            return gdf
        except Exception as e:
            logger.error(f"Error reading GeoJSON file {file_path}: {e}")
            raise

    def _read_shapefile(self, file_path: Path) -> gpd.GeoDataFrame:
        """Read Shapefile."""
        try:
            gdf = gpd.read_file(file_path)
            logger.info(f"Loaded Shapefile: {file_path}")
            return gdf
        except Exception as e:
            logger.error(f"Error reading Shapefile {file_path}: {e}")
            raise

    def _read_kml(self, file_path: Path) -> gpd.GeoDataFrame:
        """Read KML file."""
        try:
            gdf = gpd.read_file(file_path, driver="KML")
            logger.info(f"Loaded KML file: {file_path}")
            return gdf
        except Exception as e:
            logger.error(f"Error reading KML file {file_path}: {e}")
            raise

    def _read_kmz(self, file_path: Path) -> gpd.GeoDataFrame:
        """Read KMZ file (compressed KML)."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Find the KML file
                kml_files = list(Path(temp_dir).glob("*.kml"))
                if not kml_files:
                    raise ValueError("No KML file found in KMZ archive")

                gdf = self._read_kml(kml_files[0])
                logger.info(f"Loaded KMZ file: {file_path}")
                return gdf
        except Exception as e:
            logger.error(f"Error reading KMZ file {file_path}: {e}")
            raise

    def _read_gpx(self, file_path: Path) -> gpd.GeoDataFrame:
        """Read GPX file."""
        try:
            gdf = gpd.read_file(file_path, driver="GPX")
            logger.info(f"Loaded GPX file: {file_path}")
            return gdf
        except Exception as e:
            logger.error(f"Error reading GPX file {file_path}: {e}")
            raise

    def _read_csv(self, file_path: Path) -> gpd.GeoDataFrame:
        """Read CSV file with coordinate columns."""
        try:
            # Try to read as regular CSV first
            df = gpd.read_file(file_path)

            # Check if it has geometry column
            if "geometry" in df.columns:
                gdf = gpd.GeoDataFrame(df, geometry="geometry")
            else:
                # Try to create geometry from lat/lon columns
                if "latitude" in df.columns and "longitude" in df.columns:
                    geometry = [
                        Point(xy) for xy in zip(df["longitude"], df["latitude"])
                    ]
                    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                else:
                    raise ValueError(
                        "CSV must have 'geometry' column or 'latitude'/'longitude' columns"
                    )

            logger.info(f"Loaded CSV file: {file_path}")
            return gdf
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            raise

    def transform_coordinates(
        self, gdf: gpd.GeoDataFrame, target_crs: str
    ) -> gpd.GeoDataFrame:
        """
        Transform coordinates to a different CRS.

        Args:
            gdf: Input GeoDataFrame
            target_crs: Target coordinate reference system

        Returns:
            Transformed GeoDataFrame
        """
        try:
            if gdf.crs is None:
                logger.warning("Input GeoDataFrame has no CRS, assuming WGS84")
                gdf = gdf.set_crs("EPSG:4326")

            transformed_gdf = gdf.to_crs(target_crs)
            logger.info(f"Transformed coordinates from {gdf.crs} to {target_crs}")
            return transformed_gdf
        except Exception as e:
            logger.error(f"Error transforming coordinates: {e}")
            raise

    def calculate_distance(
        self, point1: GeospatialPoint, point2: GeospatialPoint
    ) -> SpatialAnalysisResult:
        """
        Calculate distance between two points.

        Args:
            point1: First point
            point2: Second point

        Returns:
            Distance calculation result
        """
        try:
            # Create Shapely points
            p1 = Point(point1.x, point1.y)
            p2 = Point(point2.x, point2.y)

            # Calculate distance
            distance = p1.distance(p2)

            # Convert to meters if in degrees
            if point1.crs == "EPSG:4326" and point2.crs == "EPSG:4326":
                # Approximate conversion to meters (1 degree ≈ 111,000 meters)
                distance_meters = distance * 111000
                units = "meters"
            else:
                distance_meters = distance
                units = "units"

            return SpatialAnalysisResult(
                analysis_type="distance",
                result=distance_meters,
                units=units,
                metadata={
                    "point1": {"x": point1.x, "y": point1.y, "crs": point1.crs},
                    "point2": {"x": point2.x, "y": point2.y, "crs": point2.crs},
                },
            )
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            raise

    def calculate_area(self, polygon: GeospatialPolygon) -> SpatialAnalysisResult:
        """
        Calculate area of a polygon.

        Args:
            polygon: Input polygon

        Returns:
            Area calculation result
        """
        try:
            # Create Shapely polygon
            exterior_coords = [(p.x, p.y) for p in polygon.exterior]
            holes_coords = [[(p.x, p.y) for p in hole] for hole in polygon.holes]

            shapely_polygon = Polygon(exterior_coords, holes_coords)
            area = shapely_polygon.area

            # Convert to square meters if in degrees
            if polygon.crs == "EPSG:4326":
                # Approximate conversion (1 degree² ≈ 12,321,000 m²)
                area_sq_meters = area * 12321000
                units = "square_meters"
            else:
                area_sq_meters = area
                units = "square_units"

            return SpatialAnalysisResult(
                analysis_type="area",
                result=area_sq_meters,
                units=units,
                metadata={
                    "polygon_crs": polygon.crs,
                    "num_vertices": len(polygon.exterior),
                },
            )
        except Exception as e:
            logger.error(f"Error calculating area: {e}")
            raise

    def check_intersection(
        self,
        geometry1: Union[GeospatialPolygon, GeospatialLine],
        geometry2: Union[GeospatialPolygon, GeospatialLine],
    ) -> SpatialAnalysisResult:
        """
        Check if two geometries intersect.

        Args:
            geometry1: First geometry
            geometry2: Second geometry

        Returns:
            Intersection analysis result
        """
        try:
            # Convert to Shapely geometries
            shapely_geom1 = self._to_shapely_geometry(geometry1)
            shapely_geom2 = self._to_shapely_geometry(geometry2)

            # Check intersection
            intersects = shapely_geom1.intersects(shapely_geom2)

            # Calculate intersection area/length if they intersect
            intersection_result = None
            if intersects:
                intersection = shapely_geom1.intersection(shapely_geom2)
                if intersection.geom_type == "Polygon":
                    intersection_result = intersection.area
                elif intersection.geom_type == "LineString":
                    intersection_result = intersection.length
                elif intersection.geom_type == "Point":
                    intersection_result = 0.0

            return SpatialAnalysisResult(
                analysis_type="intersection",
                result=intersects,
                metadata={
                    "intersection_area": intersection_result,
                    "geometry1_type": geometry1.__class__.__name__,
                    "geometry2_type": geometry2.__class__.__name__,
                },
            )
        except Exception as e:
            logger.error(f"Error checking intersection: {e}")
            raise

    def _to_shapely_geometry(self, geometry: Union[GeospatialPolygon, GeospatialLine]):
        """Convert custom geometry to Shapely geometry."""
        if isinstance(geometry, GeospatialPolygon):
            exterior_coords = [(p.x, p.y) for p in geometry.exterior]
            holes_coords = [[(p.x, p.y) for p in hole] for hole in geometry.holes]
            return Polygon(exterior_coords, holes_coords)
        elif isinstance(geometry, GeospatialLine):
            coords = [(p.x, p.y) for p in geometry.points]
            return LineString(coords)
        else:
            raise ValueError(f"Unsupported geometry type: {type(geometry)}")

    def buffer_geometry(
        self,
        geometry: Union[GeospatialPolygon, GeospatialLine, GeospatialPoint],
        distance: float,
    ) -> GeospatialPolygon:
        """
        Create a buffer around a geometry.

        Args:
            geometry: Input geometry
            distance: Buffer distance

        Returns:
            Buffered polygon
        """
        try:
            # Convert to Shapely geometry
            shapely_geom = self._to_shapely_geometry(geometry)

            # Create buffer
            buffered = shapely_geom.buffer(distance)

            # Convert back to custom format
            if buffered.geom_type == "Polygon":
                exterior = [
                    GeospatialPoint(x, y, crs=geometry.crs)
                    for x, y in buffered.exterior.coords
                ]
                holes = []
                for interior in buffered.interiors:
                    hole = [
                        GeospatialPoint(x, y, crs=geometry.crs)
                        for x, y in interior.coords
                    ]
                    holes.append(hole)

                return GeospatialPolygon(exterior, holes, crs=geometry.crs)
            else:
                raise ValueError("Buffer operation did not produce a polygon")
        except Exception as e:
            logger.error(f"Error creating buffer: {e}")
            raise

    def simplify_geometry(
        self, geometry: Union[GeospatialPolygon, GeospatialLine], tolerance: float
    ) -> Union[GeospatialPolygon, GeospatialLine]:
        """
        Simplify a geometry by reducing the number of vertices.

        Args:
            geometry: Input geometry
            tolerance: Simplification tolerance

        Returns:
            Simplified geometry
        """
        try:
            # Convert to Shapely geometry
            shapely_geom = self._to_shapely_geometry(geometry)

            # Simplify
            simplified = shapely_geom.simplify(tolerance)

            # Convert back to custom format
            if isinstance(geometry, GeospatialPolygon):
                exterior = [
                    GeospatialPoint(x, y, crs=geometry.crs)
                    for x, y in simplified.exterior.coords
                ]
                holes = []
                for interior in simplified.interiors:
                    hole = [
                        GeospatialPoint(x, y, crs=geometry.crs)
                        for x, y in interior.coords
                    ]
                    holes.append(hole)

                return GeospatialPolygon(exterior, holes, crs=geometry.crs)
            elif isinstance(geometry, GeospatialLine):
                points = [
                    GeospatialPoint(x, y, crs=geometry.crs)
                    for x, y in simplified.coords
                ]
                return GeospatialLine(points, crs=geometry.crs)
        except Exception as e:
            logger.error(f"Error simplifying geometry: {e}")
            raise

    def export_to_format(
        self, gdf: gpd.GeoDataFrame, output_path: str, format_type: str = "geojson"
    ):
        """
        Export GeoDataFrame to various formats.

        Args:
            gdf: Input GeoDataFrame
            output_path: Output file path
            format_type: Output format (geojson, shapefile, kml, csv)
        """
        try:
            output_path = Path(output_path)

            if format_type == "geojson":
                gdf.to_file(output_path, driver="GeoJSON")
            elif format_type == "shapefile":
                gdf.to_file(output_path, driver="ESRI Shapefile")
            elif format_type == "kml":
                gdf.to_file(output_path, driver="KML")
            elif format_type == "csv":
                # Export as CSV with WKT geometry
                df = gdf.copy()
                df["geometry"] = df["geometry"].astype(str)
                df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")

            logger.info(f"Exported to {format_type}: {output_path}")
        except Exception as e:
            logger.error(f"Error exporting to {format_type}: {e}")
            raise


class GeospatialVisualizer:
    """3D visualization capabilities for geospatial data."""

    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for 3D visualization. Install plotly.")

        logger.info("GeospatialVisualizer initialized")

    def create_3d_map(
        self,
        gdf: gpd.GeoDataFrame,
        elevation_column: Optional[str] = None,
        color_column: Optional[str] = None,
        title: str = "3D Map",
    ) -> go.Figure:
        """
        Create a 3D map visualization.

        Args:
            gdf: Input GeoDataFrame
            elevation_column: Column to use for elevation (Z-axis)
            color_column: Column to use for coloring
            title: Plot title

        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()

            # Extract coordinates
            if gdf.geometry.geom_type.iloc[0] == "Point":
                # Points
                x_coords = gdf.geometry.x
                y_coords = gdf.geometry.y
                z_coords = gdf[elevation_column] if elevation_column else [0] * len(gdf)

                # Color mapping
                colors = gdf[color_column] if color_column else None

                fig.add_trace(
                    go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode="markers",
                        marker=dict(
                            size=5, color=colors, colorscale="Viridis", opacity=0.8
                        ),
                        text=gdf.index,
                        name="Points",
                    )
                )

            elif gdf.geometry.geom_type.iloc[0] == "LineString":
                # Lines
                for idx, row in gdf.iterrows():
                    coords = list(row.geometry.coords)
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    z_coords = (
                        [row.get(elevation_column, 0)] * len(coords)
                        if elevation_column
                        else [0] * len(coords)
                    )

                    fig.add_trace(
                        go.Scatter3d(
                            x=x_coords,
                            y=y_coords,
                            z=z_coords,
                            mode="lines",
                            line=dict(width=3),
                            name=f"Line {idx}",
                        )
                    )

            elif gdf.geometry.geom_type.iloc[0] == "Polygon":
                # Polygons
                for idx, row in gdf.iterrows():
                    coords = list(row.geometry.exterior.coords)
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    z_coords = (
                        [row.get(elevation_column, 0)] * len(coords)
                        if elevation_column
                        else [0] * len(coords)
                    )

                    fig.add_trace(
                        go.Scatter3d(
                            x=x_coords,
                            y=y_coords,
                            z=z_coords,
                            mode="lines",
                            line=dict(width=2),
                            fill="toself",
                            fillcolor="rgba(0,100,80,0.2)",
                            name=f"Polygon {idx}",
                        )
                    )

            # Update layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    zaxis_title="Elevation" if elevation_column else "Z",
                ),
                showlegend=True,
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating 3D map: {e}")
            raise

    def create_heatmap(
        self, gdf: gpd.GeoDataFrame, value_column: str, title: str = "Heatmap"
    ) -> go.Figure:
        """
        Create a heatmap visualization.

        Args:
            gdf: Input GeoDataFrame
            value_column: Column to use for heatmap values
            title: Plot title

        Returns:
            Plotly figure object
        """
        try:
            # Extract point coordinates and values
            if gdf.geometry.geom_type.iloc[0] != "Point":
                raise ValueError("Heatmap requires point geometries")

            x_coords = gdf.geometry.x
            y_coords = gdf.geometry.y
            values = gdf[value_column]

            fig = go.Figure(
                data=go.Densitymapbox(
                    lat=y_coords,
                    lon=x_coords,
                    z=values,
                    radius=20,
                    colorscale="Viridis",
                    colorbar=dict(title=value_column),
                )
            )

            fig.update_layout(
                title=title,
                mapbox=dict(
                    center=dict(lat=y_coords.mean(), lon=x_coords.mean()),
                    zoom=10,
                    style="open-street-map",
                ),
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            raise

    def save_plot(self, fig: go.Figure, output_path: str, format_type: str = "html"):
        """
        Save plot to file.

        Args:
            fig: Plotly figure
            output_path: Output file path
            format_type: Output format (html, png, jpg, pdf)
        """
        try:
            if format_type == "html":
                fig.write_html(output_path)
            elif format_type in ["png", "jpg", "jpeg"]:
                fig.write_image(output_path)
            elif format_type == "pdf":
                fig.write_image(output_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            logger.info(f"Saved plot to {output_path}")
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            raise


class GISIntegrator:
    """Integration with external GIS systems."""

    def __init__(self):
        self.supported_systems = ["qgis", "arcgis", "postgis", "geoserver"]
        logger.info("GISIntegrator initialized")

    def export_for_qgis(self, gdf: gpd.GeoDataFrame, output_path: str):
        """Export data in QGIS-compatible format."""
        try:
            # Export as GeoJSON (QGIS native support)
            gdf.to_file(output_path, driver="GeoJSON")

            # Create QGIS project file
            self._create_qgis_project(output_path, gdf)

            logger.info(f"Exported QGIS-compatible data to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting for QGIS: {e}")
            raise

    def export_for_arcgis(self, gdf: gpd.GeoDataFrame, output_path: str):
        """Export data in ArcGIS-compatible format."""
        try:
            # Export as Shapefile (ArcGIS native support)
            gdf.to_file(output_path, driver="ESRI Shapefile")

            # Create ArcGIS metadata
            self._create_arcgis_metadata(output_path, gdf)

            logger.info(f"Exported ArcGIS-compatible data to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting for ArcGIS: {e}")
            raise

    def _create_qgis_project(self, output_path: str, gdf: gpd.GeoDataFrame):
        """Create a QGIS project file."""
        # This is a simplified version - in practice, you'd use QGIS Python API
        project_content = f"""
        <!DOCTYPE qgis>
        <qgis version="3.0.0">
            <title>Plansheet Scanner Export</title>
            <layer-tree-group>
                <layer-tree-layer source="{output_path}" name="Plansheet Data"/>
            </layer-tree-group>
        </qgis>
        """

        project_path = output_path.replace(".geojson", ".qgs")
        with open(project_path, "w") as f:
            f.write(project_content)

    def _create_arcgis_metadata(self, output_path: str, gdf: gpd.GeoDataFrame):
        """Create ArcGIS metadata file."""
        metadata = {
            "name": "Plansheet Scanner Export",
            "description": "Exported from Plansheet Scanner System",
            "crs": str(gdf.crs),
            "feature_count": len(gdf),
            "export_date": datetime.now().isoformat(),
        }

        metadata_path = output_path.replace(".shp", ".xml")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Geospatial Processing")
    parser.add_argument("--input", required=True, help="Input geospatial file")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument(
        "--format",
        choices=["geojson", "shapefile", "kml", "csv"],
        default="geojson",
        help="Output format",
    )
    parser.add_argument("--crs", help="Target coordinate reference system")
    parser.add_argument(
        "--visualize", action="store_true", help="Create 3D visualization"
    )
    parser.add_argument(
        "--elevation-column", help="Column to use for elevation in 3D visualization"
    )

    args = parser.parse_args()

    # Initialize processor
    processor = AdvancedGeospatialProcessor()

    # Read input file
    gdf = processor.read_file(args.input)
    print(f"Loaded {len(gdf)} features from {args.input}")

    # Transform coordinates if requested
    if args.crs:
        gdf = processor.transform_coordinates(gdf, args.crs)
        print(f"Transformed to CRS: {args.crs}")

    # Export if output specified
    if args.output:
        processor.export_to_format(gdf, args.output, args.format)
        print(f"Exported to {args.output}")

    # Create visualization if requested
    if args.visualize and PLOTLY_AVAILABLE:
        visualizer = GeospatialVisualizer()
        fig = visualizer.create_3d_map(gdf, elevation_column=args.elevation_column)

        viz_output = (
            args.output.replace(f".{args.format}", "_3d.html")
            if args.output
            else "visualization.html"
        )
        visualizer.save_plot(fig, viz_output)
        print(f"3D visualization saved to {viz_output}")


if __name__ == "__main__":
    main()
