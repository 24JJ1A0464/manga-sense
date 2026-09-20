"""
Manganese-Sense: Terrain & Accessibility Filtering Module
Phase 4: DEM processing, slope calculation, feasibility masking
"""

import numpy as np
from typing import Tuple, Optional
import logging
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform as transform_coordinates, transform_bounds
from rasterio.enums import Resampling
import planetary_computer as pc
from pystac_client import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_dem_elevation_resampled(bbox: Tuple[float, float, float, float],
                                   target_shape: Tuple[int, int]) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Fetch Copernicus DEM GLO-30 elevation data and resample to target shape.
    
    This ensures the DEM matches the Sentinel-2 10m grid resolution exactly,
    preventing shape mismatch errors during terrain filtering.
    
    Args:
        bbox: Target bounding box (min_lon, min_lat, max_lon, max_lat)
        target_shape: Target (height, width) tuple from Sentinel-2 array
        
    Returns:
        Tuple of (elevation_array, transform) at target resolution
    """
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox
    )
    
    items = list(search.item_collection())
    if not items:
        raise ValueError("No DEM data found for the given bbox")
    
    item = items[0]
    asset = item.assets["data"]
    href = asset.href
    
    target_height, target_width = target_shape
    
    with rasterio.open(href) as src:
        src_crs = src.crs
        bbox_proj = transform_bounds("EPSG:4326", src_crs, *bbox)
        window = from_bounds(*bbox_proj, transform=src.transform)
        window = window.round_offsets().round_shape()
        
        elevation = src.read(
            1,
            window=window,
            out_shape=(target_height, target_width),
            resampling=Resampling.bilinear
        )
        transform = src.window_transform(window)
    
    logger.info(f"Fetched DEM data resampled to: {elevation.shape} (target: {target_shape})")
    return elevation, transform


def calculate_slope(elevation: np.ndarray, 
                    transform: object, 
                    pixel_size_m: float = 30.0) -> np.ndarray:
    """
    Calculate topographic slope in degrees from elevation data.
    
    Uses numpy.gradient to compute the rate of change in elevation
    in both x and y directions, then computes the slope magnitude.
    
    Formula:
    - slope_x = dZ/dx (change in elevation per meter in x direction)
    - slope_y = dZ/dy (change in elevation per meter in y direction)
    - slope = arctan(sqrt(slope_x^2 + slope_y^2)) * (180/pi) to convert to degrees
    
    Args:
        elevation: 2D elevation array (meters)
        transform: Rasterio affine transform (used to get pixel resolution)
        pixel_size_m: Pixel size in meters (default 30m for Copernicus DEM GLO-30)
        
    Returns:
        2D slope array in degrees
    """
    dzdx, dzdy = np.gradient(elevation.astype(np.float32), pixel_size_m)
    
    slope_radians = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_degrees = np.degrees(slope_radians)
    
    logger.info(f"Slope calculated: min={np.nanmin(slope_degrees):.2f}°, "
                f"max={np.nanmax(slope_degrees):.2f}°, "
                f"mean={np.nanmean(slope_degrees):.2f}°")
    
    return slope_degrees


def apply_feasibility_mask(predictions: np.ndarray, 
                           probabilities: np.ndarray, 
                           slope: np.ndarray, 
                           slope_threshold: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply terrain feasibility filter: mask out steep terrain (>30 degrees).
    
    Manganese deposits on slopes > 30 degrees are considered unfeasible
    for extraction due to safety and economic constraints.
    
    Args:
        predictions: Binary prediction grid (1=detected, 0=background)
        probabilities: Probability grid (0-1)
        slope: Slope grid in degrees
        slope_threshold: Maximum feasible slope in degrees
        
    Returns:
        Tuple of (filtered_predictions, filtered_probabilities)
    """
    steep_mask = slope > slope_threshold
    
    filtered_predictions = predictions.copy()
    filtered_probabilities = probabilities.copy()
    
    filtered_predictions[steep_mask] = 0
    filtered_probabilities[steep_mask] = 0.0
    
    n_removed = np.sum((predictions == 1) & steep_mask)
    logger.info(f"Feasibility filter: removed {n_removed} detections on slopes > {slope_threshold}°")
    
    return filtered_predictions, filtered_probabilities


def extract_top_coordinates(predictions: np.ndarray, 
                            probabilities: np.ndarray, 
                            transform: object, 
                            top_n: int = 50,
                            source_crs: object = "EPSG:4326") -> list:
    """
    Extract GPS coordinates of top N high-probability manganese detections.
    
    Uses rasterio.transform.xy to convert pixel coordinates to geographic coordinates.
    
    Args:
        predictions: Binary prediction grid
        probabilities: Probability grid
        transform: Rasterio affine transform
        top_n: Number of top detections to extract
        source_crs: CRS of the raster transform, typically the Sentinel-2 UTM CRS
        
    Returns:
        List of dictionaries with keys: lat, lon, probability, row, col
    """
    detection_mask = predictions == 1
    
    if not np.any(detection_mask):
        logger.warning("No manganese detections found")
        return []
    
    prob_flat = probabilities[detection_mask]
    rows, cols = np.where(detection_mask)
    
    top_indices = np.argsort(prob_flat)[-top_n:][::-1]
    coordinates = []
    for idx in top_indices:
        row, col = rows[idx], cols[idx]
        prob = prob_flat[idx]
        
        x, y = transform * (col, row)
        lon, lat = transform_coordinates(
            source_crs,
            "EPSG:4326",
            [x],
            [y]
        )
        lon, lat = lon[0], lat[0]
        
        coordinates.append({
            'latitude': lat,
            'longitude': lon,
            'probability': float(prob),
            'row': int(row),
            'col': int(col)
        })
    
    logger.info(f"Extracted top {len(coordinates)} detection coordinates")
    return coordinates


def create_geojson_features(coordinates: list) -> dict:
    """
    Create GeoJSON FeatureCollection from detection coordinates.
    
    Args:
        coordinates: List of coordinate dictionaries
        
    Returns:
        GeoJSON FeatureCollection
    """
    features = []
    for i, coord in enumerate(coordinates):
        feature = {
            "type": "Feature",
            "properties": {
                "id": i + 1,
                "probability": coord['probability'],
                "latitude": coord['latitude'],
                "longitude": coord['longitude']
            },
            "geometry": {
                "type": "Point",
                "coordinates": [coord['longitude'], coord['latitude']]
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }