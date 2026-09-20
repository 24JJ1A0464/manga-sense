"""
Manganese-Sense: Data Ingestion Module
Phase 1: Connect to Planetary Computer STAC API and fetch Sentinel-2 bands
using memory-efficient windowed reads.
"""

import planetary_computer as pc
from pystac_client import Client
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_sentinel2_items(bbox: Tuple[float, float, float, float],
                           date_range: str,
                           cloud_cover_threshold: float = 5.0) -> list:
    """
    Query the Planetary Computer STAC API for Sentinel-2 L2A items.
    
    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        date_range: Date range string in format "YYYY-MM-DD/YYYY-MM-DD"
        cloud_cover_threshold: Maximum cloud cover percentage
        
    Returns:
        List of STAC items matching the query
    """
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_threshold}}
    )
    
    items = list(search.item_collection())
    logger.info(f"Found {len(items)} Sentinel-2 items with < {cloud_cover_threshold}% cloud cover")
    return items


def select_best_item(items: list) -> Optional[object]:
    """
    Select the best item based on lowest cloud cover.
    
    Args:
        items: List of STAC items
        
    Returns:
        Best matching STAC item or None if no items found
    """
    if not items:
        logger.warning("No items found matching criteria")
        return None
    
    best_item = min(items, key=lambda item: item.properties.get("eo:cloud_cover", 100))
    logger.info(f"Selected item: {best_item.id} with cloud cover: {best_item.properties.get('eo:cloud_cover')}%")
    return best_item


def read_band_window(item: object, band_name: str, bbox: Tuple[float, float, float, float]) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Read a specific band using memory-efficient windowed read.
    
    Args:
        item: STAC item
        band_name: Band identifier (e.g., 'B04', 'B08', 'B11', 'B12', 'SCL')
        bbox: Target bounding box (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        Tuple of (band_data_array, transform)
    """
    asset = item.assets[band_name]
    href = asset.href
    
    with rasterio.open(href) as src:
        src_crs = src.crs
        
        bbox_wgs84 = bbox
        bbox_proj = transform_bounds("EPSG:4326", src_crs, *bbox_wgs84)
        
        window = from_bounds(*bbox_proj, transform=src.transform)
        window = window.round_offsets().round_shape()
        
        band_data = src.read(1, window=window)
        out_transform = src.window_transform(window)
        
    return band_data, out_transform


def fetch_sentinel2_bands(item: object, bbox: Tuple[float, float, float, float],
                          bands: list = None) -> Dict[str, Tuple[np.ndarray, rasterio.Affine]]:
    """
    Fetch multiple Sentinel-2 bands for the given bbox using windowed reads.
    All bands are resampled to the 10m grid (B04/B08 resolution) using bilinear interpolation.
    
    Args:
        item: STAC item
        bbox: Target bounding box (min_lon, min_lat, max_lon, max_lat)
        bands: List of band names to fetch. Defaults to required bands.
        
    Returns:
        Dictionary mapping band_name -> (array, transform)
        All arrays have the same shape (target_height, target_width) at 10m resolution.
    """
    if bands is None:
        bands = ['B04', 'B08', 'B11', 'B12', 'SCL']
    
    reference_band = 'B04'
    reference_asset = item.assets[reference_band]
    
    with rasterio.open(reference_asset.href) as src:
        src_crs = src.crs
        bbox_proj = transform_bounds("EPSG:4326", src_crs, *bbox)
        window = from_bounds(*bbox_proj, transform=src.transform)
        window = window.round_offsets().round_shape()
        
        target_height = int(window.height)
        target_width = int(window.width)
        reference_transform = src.window_transform(window)
    
    logger.info(f"Target grid (10m): {target_height} x {target_width}")
    
    band_data = {}
    
    for band in bands:
        logger.info(f"Reading band {band} at 10m resolution...")
        asset = item.assets[band]
        href = asset.href
        
        with rasterio.open(href) as src:
            src_crs = src.crs
            bbox_proj = transform_bounds("EPSG:4326", src_crs, *bbox)
            window = from_bounds(*bbox_proj, transform=src.transform)
            window = window.round_offsets().round_shape()
            
            band_data_resampled = src.read(
                1,
                window=window,
                out_shape=(target_height, target_width),
                resampling=Resampling.bilinear
            )
            out_transform = src.window_transform(window)
        
        band_data[band] = (band_data_resampled, out_transform)
    
    return band_data


def fetch_dem_elevation(bbox: Tuple[float, float, float, float]) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Fetch Copernicus DEM GLO-30 elevation data for the given bbox.
    
    Args:
        bbox: Target bounding box (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        Tuple of (elevation_array, transform)
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
    
    with rasterio.open(href) as src:
        src_crs = src.crs
        bbox_proj = transform_bounds("EPSG:4326", src_crs, *bbox)
        window = from_bounds(*bbox_proj, transform=src.transform)
        window = window.round_offsets().round_shape()
        
        elevation = src.read(1, window=window)
        transform = src.window_transform(window)
    
    logger.info(f"Fetched DEM data: shape={elevation.shape}")
    return elevation, transform