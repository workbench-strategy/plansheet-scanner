"""
Advanced Geospatial Intelligence System for Plansheet Scanner

This module provides advanced geospatial intelligence capabilities including intelligent spatial analysis,
3D modeling, advanced GIS integration, and spatial pattern recognition.

Features:
- Intelligent spatial analysis and pattern recognition
- 3D modeling and visualization
- Advanced GIS integration and data fusion
- Spatial data quality assessment
- Automated spatial insights generation
- Multi-scale spatial analysis
"""

import asyncio
import json
import math
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Geospatial imports
try:
    import geopandas as gpd
    import pyproj
    from pyproj import Transformer
    from shapely.affinity import rotate, scale, translate
    from shapely.geometry import LineString, MultiPolygon, Point, Polygon
    from shapely.ops import transform, unary_union

    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False

# Visualization imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import joblib

# ML imports
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Database imports
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


@dataclass
class SpatialFeature:
    """Spatial feature data structure"""

    feature_id: str
    geometry: Any  # Shapely geometry object
    properties: Dict[str, Any]
    feature_type: str
    confidence: float
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class SpatialAnalysis:
    """Spatial analysis result"""

    analysis_id: str
    analysis_type: str
    input_features: List[str]
    output_features: List[str]
    metrics: Dict[str, float]
    insights: List[str]
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class SpatialPattern:
    """Spatial pattern data structure"""

    pattern_id: str
    pattern_type: str  # 'cluster', 'corridor', 'hotspot', 'void'
    geometry: Any
    confidence: float
    significance: float
    features: List[str]
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class SpatialQuality:
    """Spatial data quality assessment"""

    quality_id: str
    feature_id: str
    accuracy_score: float
    precision_score: float
    completeness_score: float
    consistency_score: float
    overall_score: float
    issues: List[str]
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class SpatialModel:
    """3D spatial model data structure"""

    model_id: str
    name: str
    model_type: str  # 'terrain', 'building', 'infrastructure'
    vertices: np.ndarray
    faces: np.ndarray
    textures: Optional[Dict[str, Any]]
    properties: Dict[str, Any]
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class GeospatialDataset:
    """Geospatial dataset data structure"""

    dataset_id: str
    name: str
    description: str
    crs: str
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    feature_count: int
    file_path: str
    created_at: datetime
    metadata: Dict[str, Any]


class AdvancedGeospatialIntelligence:
    """Advanced Geospatial Intelligence System"""

    def __init__(
        self,
        data_dir: str = "geospatial_data",
        db_config: Optional[Dict[str, Any]] = None,
    ):
        if not GEOSPATIAL_AVAILABLE:
            raise ImportError(
                "Geospatial libraries (geopandas, shapely, pyproj) are required"
            )

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Storage
        self.spatial_features: Dict[str, SpatialFeature] = {}
        self.spatial_analyses: Dict[str, SpatialAnalysis] = {}
        self.spatial_patterns: Dict[str, SpatialPattern] = {}
        self.spatial_quality: Dict[str, SpatialQuality] = {}
        self.spatial_models: Dict[str, SpatialModel] = {}
        self.geospatial_datasets: Dict[str, GeospatialDataset] = {}

        # Analysis tools
        self.clustering_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Coordinate systems
        self.default_crs = "EPSG:4326"  # WGS84
        self.projection_crs = "EPSG:3857"  # Web Mercator

        # Database connection
        self.db_engine = None
        if db_config:
            self._setup_database(db_config)

        # Initialize spatial analysis tools
        self._setup_spatial_tools()

    def _setup_database(self, db_config: Dict[str, Any]):
        """Setup database connection for geospatial data storage"""
        try:
            connection_string = (
                f"postgresql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            self.db_engine = create_engine(connection_string)

            # Create geospatial tables
            self._create_geospatial_tables()

        except Exception as e:
            print(f"Warning: Database setup failed: {e}")

    def _create_geospatial_tables(self):
        """Create geospatial tables in database"""
        if not self.db_engine:
            return

        try:
            with self.db_engine.connect() as conn:
                # Spatial features table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS spatial_features (
                        feature_id VARCHAR(255) PRIMARY KEY,
                        geometry GEOMETRY,
                        properties JSONB,
                        feature_type VARCHAR(100) NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Spatial analyses table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS spatial_analyses (
                        analysis_id VARCHAR(255) PRIMARY KEY,
                        analysis_type VARCHAR(100) NOT NULL,
                        input_features TEXT[],
                        output_features TEXT[],
                        metrics JSONB,
                        insights TEXT[],
                        created_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Spatial patterns table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS spatial_patterns (
                        pattern_id VARCHAR(255) PRIMARY KEY,
                        pattern_type VARCHAR(50) NOT NULL,
                        geometry GEOMETRY,
                        confidence DOUBLE PRECISION NOT NULL,
                        significance DOUBLE PRECISION NOT NULL,
                        features TEXT[],
                        created_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Spatial quality table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS spatial_quality (
                        quality_id VARCHAR(255) PRIMARY KEY,
                        feature_id VARCHAR(255) NOT NULL,
                        accuracy_score DOUBLE PRECISION NOT NULL,
                        precision_score DOUBLE PRECISION NOT NULL,
                        completeness_score DOUBLE PRECISION NOT NULL,
                        consistency_score DOUBLE PRECISION NOT NULL,
                        overall_score DOUBLE PRECISION NOT NULL,
                        issues TEXT[],
                        created_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                # Geospatial datasets table
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS geospatial_datasets (
                        dataset_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        crs VARCHAR(50) NOT NULL,
                        bounds DOUBLE PRECISION[],
                        feature_count INTEGER NOT NULL,
                        file_path VARCHAR(500) NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                    )
                )

                conn.commit()

        except Exception as e:
            print(f"Warning: Failed to create geospatial tables: {e}")

    def _setup_spatial_tools(self):
        """Setup spatial analysis tools"""
        # Initialize clustering models
        self.clustering_models["dbscan"] = DBSCAN(eps=0.1, min_samples=5)
        self.clustering_models["kmeans"] = KMeans(n_clusters=5, random_state=42)

        # Initialize scalers
        self.scalers["standard"] = StandardScaler()

    def add_spatial_feature(
        self,
        geometry: Any,
        properties: Dict[str, Any],
        feature_type: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpatialFeature:
        """Add a spatial feature"""
        feature = SpatialFeature(
            feature_id=str(uuid.uuid4()),
            geometry=geometry,
            properties=properties,
            feature_type=feature_type,
            confidence=confidence,
            created_at=datetime.now(),
            metadata=metadata or {},
        )

        self.spatial_features[feature.feature_id] = feature

        # Store in database
        if self.db_engine:
            try:
                with self.db_engine.connect() as conn:
                    # Convert geometry to WKT for database storage
                    geometry_wkt = (
                        geometry.wkt if hasattr(geometry, "wkt") else str(geometry)
                    )

                    conn.execute(
                        text(
                            """
                        INSERT INTO spatial_features
                        (feature_id, geometry, properties, feature_type, confidence, created_at, metadata)
                        VALUES (:feature_id, ST_GeomFromText(:geometry), :properties, :feature_type, :confidence, :created_at, :metadata)
                    """
                        ),
                        {
                            "feature_id": feature.feature_id,
                            "geometry": geometry_wkt,
                            "properties": json.dumps(properties),
                            "feature_type": feature_type,
                            "confidence": confidence,
                            "created_at": feature.created_at,
                            "metadata": json.dumps(metadata or {}),
                        },
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error storing spatial feature: {e}")

        return feature

    def load_geospatial_dataset(
        self,
        file_path: str,
        name: str,
        description: str = "",
        crs: Optional[str] = None,
    ) -> GeospatialDataset:
        """Load a geospatial dataset from file"""
        try:
            # Load dataset
            gdf = gpd.read_file(file_path)

            # Set CRS if provided
            if crs:
                gdf = gdf.set_crs(crs)
            elif gdf.crs is None:
                gdf = gdf.set_crs(self.default_crs)

            # Get bounds
            bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)

            # Create dataset record
            dataset = GeospatialDataset(
                dataset_id=str(uuid.uuid4()),
                name=name,
                description=description,
                crs=str(gdf.crs),
                bounds=tuple(bounds),
                feature_count=len(gdf),
                file_path=file_path,
                created_at=datetime.now(),
                metadata={
                    "columns": list(gdf.columns),
                    "geometry_type": str(gdf.geometry.geom_type.iloc[0])
                    if len(gdf) > 0
                    else "Unknown",
                },
            )

            self.geospatial_datasets[dataset.dataset_id] = dataset

            # Convert to spatial features
            for idx, row in gdf.iterrows():
                properties = {col: row[col] for col in gdf.columns if col != "geometry"}
                self.add_spatial_feature(
                    geometry=row.geometry,
                    properties=properties,
                    feature_type=dataset.metadata.get("geometry_type", "Unknown"),
                    metadata={"dataset_id": dataset.dataset_id, "row_index": idx},
                )

            # Store in database
            if self.db_engine:
                try:
                    with self.db_engine.connect() as conn:
                        conn.execute(
                            text(
                                """
                            INSERT INTO geospatial_datasets
                            (dataset_id, name, description, crs, bounds, feature_count, file_path, created_at, metadata)
                            VALUES (:dataset_id, :name, :description, :crs, :bounds, :feature_count, :file_path, :created_at, :metadata)
                        """
                            ),
                            asdict(dataset),
                        )
                        conn.commit()
                except Exception as e:
                    print(f"Error storing geospatial dataset: {e}")

            return dataset

        except Exception as e:
            print(f"Error loading geospatial dataset: {e}")
            raise

    def analyze_spatial_patterns(
        self, feature_ids: List[str], analysis_type: str = "clustering"
    ) -> SpatialAnalysis:
        """Analyze spatial patterns in features"""
        try:
            # Get features
            features = [
                self.spatial_features[fid]
                for fid in feature_ids
                if fid in self.spatial_features
            ]

            if len(features) < 2:
                raise ValueError("Need at least 2 features for spatial analysis")

            # Extract coordinates
            coords = []
            for feature in features:
                if hasattr(feature.geometry, "centroid"):
                    centroid = feature.geometry.centroid
                    coords.append([centroid.x, centroid.y])
                elif hasattr(feature.geometry, "coords"):
                    coords.append(list(feature.geometry.coords[0]))

            coords = np.array(coords)

            # Perform analysis
            if analysis_type == "clustering":
                patterns, metrics = self._perform_clustering_analysis(coords, features)
            elif analysis_type == "density":
                patterns, metrics = self._perform_density_analysis(coords, features)
            elif analysis_type == "correlation":
                patterns, metrics = self._perform_correlation_analysis(features)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")

            # Generate insights
            insights = self._generate_spatial_insights(patterns, metrics, analysis_type)

            # Create analysis result
            analysis = SpatialAnalysis(
                analysis_id=str(uuid.uuid4()),
                analysis_type=analysis_type,
                input_features=feature_ids,
                output_features=[p.pattern_id for p in patterns],
                metrics=metrics,
                insights=insights,
                created_at=datetime.now(),
                metadata={},
            )

            self.spatial_analyses[analysis.analysis_id] = analysis

            # Store patterns
            for pattern in patterns:
                self.spatial_patterns[pattern.pattern_id] = pattern

            return analysis

        except Exception as e:
            print(f"Error in spatial pattern analysis: {e}")
            raise

    def _perform_clustering_analysis(
        self, coords: np.ndarray, features: List[SpatialFeature]
    ) -> Tuple[List[SpatialPattern], Dict[str, float]]:
        """Perform clustering analysis"""
        # Scale coordinates
        coords_scaled = self.scalers["standard"].fit_transform(coords)

        # Perform DBSCAN clustering
        dbscan = self.clustering_models["dbscan"]
        labels = dbscan.fit_predict(coords_scaled)

        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        silhouette_avg = (
            silhouette_score(coords_scaled, labels) if n_clusters > 1 else 0
        )

        # Create patterns
        patterns = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise points
                continue

            cluster_points = coords[labels == cluster_id]
            cluster_features = [
                f for i, f in enumerate(features) if labels[i] == cluster_id
            ]

            # Create convex hull for cluster
            if len(cluster_points) >= 3:
                from scipy.spatial import ConvexHull

                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                geometry = Polygon(hull_points)
            else:
                # For small clusters, create a buffer around points
                geometry = Point(cluster_points[0]).buffer(0.01)

            pattern = SpatialPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type="cluster",
                geometry=geometry,
                confidence=len(cluster_points) / len(coords),
                significance=silhouette_avg,
                features=[f.feature_id for f in cluster_features],
                created_at=datetime.now(),
                metadata={
                    "cluster_id": int(cluster_id),
                    "point_count": len(cluster_points),
                    "cluster_center": cluster_points.mean(axis=0).tolist(),
                },
            )
            patterns.append(pattern)

        metrics = {
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_avg,
            "noise_points": np.sum(labels == -1),
            "total_points": len(coords),
        }

        return patterns, metrics

    def _perform_density_analysis(
        self, coords: np.ndarray, features: List[SpatialFeature]
    ) -> Tuple[List[SpatialPattern], Dict[str, float]]:
        """Perform density analysis"""
        # Calculate point density using kernel density estimation
        from scipy.stats import gaussian_kde

        # Estimate density
        kde = gaussian_kde(coords.T)

        # Create grid for density calculation
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        x_grid = np.linspace(x_min, x_max, 50)
        y_grid = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])

        # Calculate density
        density = kde(positions).reshape(X.shape)

        # Find high-density areas (hotspots)
        threshold = np.percentile(density, 90)  # Top 10% density
        hotspot_mask = density > threshold

        # Create hotspot patterns
        patterns = []
        from scipy import ndimage

        labeled_hotspots, num_hotspots = ndimage.label(hotspot_mask)

        for hotspot_id in range(1, num_hotspots + 1):
            hotspot_coords = positions[:, labeled_hotspots.ravel() == hotspot_id]

            if len(hotspot_coords.T) >= 3:
                # Create convex hull for hotspot
                from scipy.spatial import ConvexHull

                hull = ConvexHull(hotspot_coords.T)
                hull_points = hotspot_coords.T[hull.vertices]
                geometry = Polygon(hull_points)
            else:
                # For small hotspots, create a buffer
                geometry = Point(hotspot_coords.T[0]).buffer(0.01)

            # Find features in hotspot
            hotspot_features = []
            for feature in features:
                if hasattr(feature.geometry, "centroid"):
                    centroid = feature.geometry.centroid
                    if geometry.contains(centroid):
                        hotspot_features.append(feature)

            pattern = SpatialPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type="hotspot",
                geometry=geometry,
                confidence=len(hotspot_features) / len(features),
                significance=np.mean(density[labeled_hotspots == hotspot_id]),
                features=[f.feature_id for f in hotspot_features],
                created_at=datetime.now(),
                metadata={
                    "hotspot_id": hotspot_id,
                    "density_threshold": threshold,
                    "max_density": np.max(density[labeled_hotspots == hotspot_id]),
                },
            )
            patterns.append(pattern)

        metrics = {
            "num_hotspots": num_hotspots,
            "density_threshold": threshold,
            "max_density": np.max(density),
            "mean_density": np.mean(density),
        }

        return patterns, metrics

    def _perform_correlation_analysis(
        self, features: List[SpatialFeature]
    ) -> Tuple[List[SpatialPattern], Dict[str, float]]:
        """Perform spatial correlation analysis"""
        # Extract numeric properties for correlation analysis
        numeric_props = {}
        for feature in features:
            for key, value in feature.properties.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_props:
                        numeric_props[key] = []
                    numeric_props[key].append(value)

        if len(numeric_props) < 2:
            # Not enough numeric properties for correlation
            return [], {
                "error": "Insufficient numeric properties for correlation analysis"
            }

        # Calculate correlations
        correlations = {}
        for prop1 in numeric_props:
            for prop2 in numeric_props:
                if prop1 != prop2:
                    corr = np.corrcoef(numeric_props[prop1], numeric_props[prop2])[0, 1]
                    if not np.isnan(corr):
                        correlations[f"{prop1}_vs_{prop2}"] = corr

        # Find significant correlations
        significant_correlations = {
            k: v for k, v in correlations.items() if abs(v) > 0.7
        }

        # Create correlation patterns
        patterns = []
        for corr_name, corr_value in significant_correlations.items():
            # Create a pattern representing the correlation
            pattern = SpatialPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type="correlation",
                geometry=Point(0, 0),  # Placeholder geometry
                confidence=abs(corr_value),
                significance=abs(corr_value),
                features=[f.feature_id for f in features],
                created_at=datetime.now(),
                metadata={
                    "correlation_name": corr_name,
                    "correlation_value": corr_value,
                    "properties_analyzed": list(numeric_props.keys()),
                },
            )
            patterns.append(pattern)

        metrics = {
            "total_correlations": len(correlations),
            "significant_correlations": len(significant_correlations),
            "max_correlation": max(correlations.values()) if correlations else 0,
            "min_correlation": min(correlations.values()) if correlations else 0,
        }

        return patterns, metrics

    def _generate_spatial_insights(
        self,
        patterns: List[SpatialPattern],
        metrics: Dict[str, float],
        analysis_type: str,
    ) -> List[str]:
        """Generate insights from spatial analysis"""
        insights = []

        if analysis_type == "clustering":
            n_clusters = metrics.get("n_clusters", 0)
            silhouette_score = metrics.get("silhouette_score", 0)

            if n_clusters > 0:
                insights.append(f"Identified {n_clusters} distinct spatial clusters")

            if silhouette_score > 0.7:
                insights.append("Strong spatial clustering pattern detected")
            elif silhouette_score > 0.5:
                insights.append("Moderate spatial clustering pattern detected")
            else:
                insights.append("Weak spatial clustering pattern detected")

        elif analysis_type == "density":
            num_hotspots = metrics.get("num_hotspots", 0)
            if num_hotspots > 0:
                insights.append(f"Identified {num_hotspots} high-density hotspots")

            max_density = metrics.get("max_density", 0)
            if max_density > 0:
                insights.append(f"Maximum density concentration: {max_density:.3f}")

        elif analysis_type == "correlation":
            significant_correlations = metrics.get("significant_correlations", 0)
            if significant_correlations > 0:
                insights.append(
                    f"Found {significant_correlations} significant spatial correlations"
                )

            max_correlation = metrics.get("max_correlation", 0)
            if abs(max_correlation) > 0.8:
                insights.append("Strong spatial correlation patterns detected")

        return insights

    def assess_spatial_quality(self, feature_id: str) -> SpatialQuality:
        """Assess spatial data quality"""
        if feature_id not in self.spatial_features:
            raise ValueError(f"Feature {feature_id} not found")

        feature = self.spatial_features[feature_id]
        issues = []

        # Accuracy assessment
        accuracy_score = 1.0
        if hasattr(feature.geometry, "is_valid"):
            if not feature.geometry.is_valid:
                accuracy_score *= 0.5
                issues.append("Invalid geometry")

        # Precision assessment
        precision_score = 1.0
        if hasattr(feature.geometry, "length") or hasattr(feature.geometry, "area"):
            # Check for very small geometries (potential precision issues)
            if hasattr(feature.geometry, "length") and feature.geometry.length < 0.001:
                precision_score *= 0.8
                issues.append("Very small geometry (potential precision issue)")
            elif hasattr(feature.geometry, "area") and feature.geometry.area < 0.0001:
                precision_score *= 0.8
                issues.append("Very small area (potential precision issue)")

        # Completeness assessment
        completeness_score = 1.0
        if not feature.properties:
            completeness_score *= 0.7
            issues.append("No properties defined")
        else:
            # Check for missing critical properties
            critical_props = ["id", "name", "type"]
            missing_props = [
                prop for prop in critical_props if prop not in feature.properties
            ]
            if missing_props:
                completeness_score *= 0.9
                issues.append(f"Missing properties: {', '.join(missing_props)}")

        # Consistency assessment
        consistency_score = 1.0
        if feature.confidence < 0.8:
            consistency_score *= 0.8
            issues.append("Low confidence score")

        # Calculate overall score
        overall_score = (
            accuracy_score + precision_score + completeness_score + consistency_score
        ) / 4

        quality = SpatialQuality(
            quality_id=str(uuid.uuid4()),
            feature_id=feature_id,
            accuracy_score=accuracy_score,
            precision_score=precision_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            overall_score=overall_score,
            issues=issues,
            created_at=datetime.now(),
            metadata={},
        )

        self.spatial_quality[quality.quality_id] = quality

        return quality

    def create_3d_model(
        self,
        name: str,
        model_type: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        textures: Optional[Dict[str, Any]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> SpatialModel:
        """Create a 3D spatial model"""
        model = SpatialModel(
            model_id=str(uuid.uuid4()),
            name=name,
            model_type=model_type,
            vertices=vertices,
            faces=faces,
            textures=textures,
            properties=properties or {},
            created_at=datetime.now(),
            metadata={},
        )

        self.spatial_models[model.model_id] = model
        return model

    def generate_3d_visualization(
        self, model_ids: List[str], output_path: Optional[str] = None
    ) -> str:
        """Generate 3D visualization of spatial models"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for 3D visualization")

        try:
            # Create 3D figure
            fig = go.Figure()

            for model_id in model_ids:
                if model_id not in self.spatial_models:
                    continue

                model = self.spatial_models[model_id]

                # Create 3D mesh
                fig.add_trace(
                    go.Mesh3d(
                        x=model.vertices[:, 0],
                        y=model.vertices[:, 1],
                        z=model.vertices[:, 2],
                        i=model.faces[:, 0],
                        j=model.faces[:, 1],
                        k=model.faces[:, 2],
                        name=model.name,
                        color="lightblue",
                        opacity=0.8,
                    )
                )

            # Update layout
            fig.update_layout(
                title="3D Spatial Models Visualization",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                width=800,
                height=600,
            )

            # Save or return
            if output_path:
                fig.write_html(output_path)
                return output_path
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"3d_visualization_{timestamp}.html"
                fig.write_html(output_path)
                return output_path

        except Exception as e:
            print(f"Error generating 3D visualization: {e}")
            raise

    def transform_coordinates(self, geometry: Any, from_crs: str, to_crs: str) -> Any:
        """Transform geometry between coordinate reference systems"""
        try:
            transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
            return transform(transformer.transform, geometry)
        except Exception as e:
            print(f"Error transforming coordinates: {e}")
            raise

    def calculate_spatial_metrics(self, feature_ids: List[str]) -> Dict[str, float]:
        """Calculate spatial metrics for a set of features"""
        features = [
            self.spatial_features[fid]
            for fid in feature_ids
            if fid in self.spatial_features
        ]

        if len(features) < 2:
            return {"error": "Need at least 2 features for spatial metrics"}

        metrics = {}

        # Calculate centroids
        centroids = []
        for feature in features:
            if hasattr(feature.geometry, "centroid"):
                centroid = feature.geometry.centroid
                centroids.append([centroid.x, centroid.y])

        if centroids:
            centroids = np.array(centroids)

            # Calculate spatial extent
            min_coords = centroids.min(axis=0)
            max_coords = centroids.max(axis=0)
            extent = max_coords - min_coords

            metrics["spatial_extent_x"] = extent[0]
            metrics["spatial_extent_y"] = extent[1]
            metrics["spatial_extent_area"] = extent[0] * extent[1]

            # Calculate centroid
            mean_centroid = centroids.mean(axis=0)
            metrics["mean_centroid_x"] = mean_centroid[0]
            metrics["mean_centroid_y"] = mean_centroid[1]

            # Calculate nearest neighbor distances
            from scipy.spatial.distance import pdist, squareform

            distances = pdist(centroids)
            if len(distances) > 0:
                metrics["min_distance"] = distances.min()
                metrics["max_distance"] = distances.max()
                metrics["mean_distance"] = distances.mean()
                metrics["std_distance"] = distances.std()

        # Calculate area/length metrics
        total_area = 0
        total_length = 0
        for feature in features:
            if hasattr(feature.geometry, "area"):
                total_area += feature.geometry.area
            if hasattr(feature.geometry, "length"):
                total_length += feature.geometry.length

        metrics["total_area"] = total_area
        metrics["total_length"] = total_length
        metrics["feature_count"] = len(features)

        return metrics

    def export_spatial_data(
        self, format: str = "geojson", feature_ids: Optional[List[str]] = None
    ) -> str:
        """Export spatial data to various formats"""
        try:
            if feature_ids is None:
                features = list(self.spatial_features.values())
            else:
                features = [
                    self.spatial_features[fid]
                    for fid in feature_ids
                    if fid in self.spatial_features
                ]

            if not features:
                raise ValueError("No features to export")

            # Create GeoDataFrame
            geometries = [f.geometry for f in features]
            properties = [f.properties for f in features]

            gdf = gpd.GeoDataFrame(
                properties, geometry=geometries, crs=self.default_crs
            )

            # Export based on format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format.lower() == "geojson":
                filename = f"spatial_export_{timestamp}.geojson"
                gdf.to_file(filename, driver="GeoJSON")
            elif format.lower() == "shapefile":
                filename = f"spatial_export_{timestamp}.shp"
                gdf.to_file(filename, driver="ESRI Shapefile")
            elif format.lower() == "csv":
                filename = f"spatial_export_{timestamp}.csv"
                # Export coordinates and properties
                data = []
                for feature in features:
                    if hasattr(feature.geometry, "centroid"):
                        centroid = feature.geometry.centroid
                        row = {"x": centroid.x, "y": centroid.y, **feature.properties}
                        data.append(row)

                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            return filename

        except Exception as e:
            print(f"Error exporting spatial data: {e}")
            raise


def main():
    """Main function to run the advanced geospatial intelligence system"""
    # Initialize system
    gis = AdvancedGeospatialIntelligence()

    # Create sample spatial features
    point1 = Point(0, 0)
    point2 = Point(1, 1)
    point3 = Point(2, 2)
    point4 = Point(0.1, 0.1)
    point5 = Point(1.1, 1.1)

    # Add features
    feature1 = gis.add_spatial_feature(
        point1, {"id": 1, "name": "Point 1", "value": 10}, "point"
    )
    feature2 = gis.add_spatial_feature(
        point2, {"id": 2, "name": "Point 2", "value": 20}, "point"
    )
    feature3 = gis.add_spatial_feature(
        point3, {"id": 3, "name": "Point 3", "value": 30}, "point"
    )
    feature4 = gis.add_spatial_feature(
        point4, {"id": 4, "name": "Point 4", "value": 15}, "point"
    )
    feature5 = gis.add_spatial_feature(
        point5, {"id": 5, "name": "Point 5", "value": 25}, "point"
    )

    # Analyze spatial patterns
    feature_ids = [
        feature1.feature_id,
        feature2.feature_id,
        feature3.feature_id,
        feature4.feature_id,
        feature5.feature_id,
    ]

    analysis = gis.analyze_spatial_patterns(feature_ids, "clustering")
    print(f"Spatial analysis completed: {len(analysis.insights)} insights generated")

    # Assess spatial quality
    quality = gis.assess_spatial_quality(feature1.feature_id)
    print(f"Spatial quality score: {quality.overall_score:.3f}")

    # Calculate spatial metrics
    metrics = gis.calculate_spatial_metrics(feature_ids)
    print(f"Spatial metrics: {metrics}")

    # Export data
    export_file = gis.export_spatial_data("geojson", feature_ids)
    print(f"Data exported to: {export_file}")


if __name__ == "__main__":
    main()
