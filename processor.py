"""
Manganese-Sense: Pre-processing & Feature Engineering Module
Phase 2: Cloud masking, vegetation masking, anomaly filtering
Phase 3: Spectral index calculation and feature matrix preparation
"""

import numpy as np
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_cloud_mask(scl_band: np.ndarray) -> np.ndarray:
    """
    Create a cloud mask from the Scene Classification Layer (SCL).
    
    SCL Classes to mask (set to True for masking):
    - 3: Cloud shadows
    - 8: Cloud medium probability
    - 9: Cloud high probability
    - 10: Thin cirrus
    
    Args:
        scl_band: SCL band array
        
    Returns:
        Boolean mask where True = pixel should be masked (invalid)
    """
    cloud_classes = [3, 8, 9, 10]
    cloud_mask = np.isin(scl_band, cloud_classes)
    logger.info(f"Cloud mask: {np.sum(cloud_mask)} pixels masked out of {scl_band.size}")
    return cloud_mask


def calculate_ndvi(b08: np.ndarray, b04: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    
    Formula: NDVI = (NIR - Red) / (NIR + Red)
    Where: NIR = B08, Red = B04
    
    NDVI ranges from -1 to 1:
    - Values > 0.3 typically indicate dense vegetation
    - Values < 0 indicate water, snow, or bare soil
    
    Args:
        b08: NIR band (Band 8)
        b04: Red band (Band 4)
        
    Returns:
        NDVI array
    """
    numerator = b08.astype(np.float32) - b04.astype(np.float32)
    denominator = b08.astype(np.float32) + b04.astype(np.float32)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(denominator != 0, numerator / denominator, np.nan)
    
    return ndvi


def apply_vegetation_mask(ndvi: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Create vegetation mask based on NDVI threshold.
    
    Args:
        ndvi: NDVI array
        threshold: NDVI threshold above which pixels are considered dense vegetation
        
    Returns:
        Boolean mask where True = pixel should be masked (dense vegetation)
    """
    veg_mask = ndvi > threshold
    logger.info(f"Vegetation mask: {np.sum(veg_mask)} pixels masked out of {ndvi.size}")
    return veg_mask


def mask_bands(bands: Dict[str, np.ndarray], 
               cloud_mask: np.ndarray, 
               veg_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply cloud and vegetation masks to all bands, replacing masked pixels with NaN.
    
    Args:
        bands: Dictionary of band_name -> array
        cloud_mask: Boolean cloud mask
        veg_mask: Boolean vegetation mask
        
    Returns:
        Dictionary of masked bands
    """
    combined_mask = cloud_mask | veg_mask
    
    masked_bands = {}
    for name, data in bands.items():
        masked_data = data.astype(np.float32)
        masked_data[combined_mask] = np.nan
        masked_bands[name] = masked_data
    
    logger.info(f"Applied combined mask: {np.sum(combined_mask)} total pixels masked")
    return masked_bands


def calculate_swir_ratio(b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """
    Calculate SWIR Ratio: B11 / B12
    
    This ratio helps identify mineral alterations associated with manganese deposits.
    SWIR1 (B11) and SWIR2 (B12) are sensitive to different mineral absorption features.
    
    Args:
        b11: SWIR1 band (Band 11)
        b12: SWIR2 band (Band 12)
        
    Returns:
        SWIR Ratio array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(b12 != 0, b11.astype(np.float32) / b12.astype(np.float32), np.nan)
    return ratio


def calculate_ferrous_ratio(b11: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """
    Calculate Ferrous Composite Ratio: B11 / B08
    
    This ratio enhances detection of ferrous minerals including manganese oxides.
    SWIR1 (B11) is sensitive to iron/manganese absorption features, while NIR (B08)
    provides a reference for surface brightness.
    
    Args:
        b11: SWIR1 band (Band 11)
        b08: NIR band (Band 8)
        
    Returns:
        Ferrous Composite Ratio array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(b08 != 0, b11.astype(np.float32) / b08.astype(np.float32), np.nan)
    return ratio


def prepare_feature_matrix(bands: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Prepare the feature matrix for ML inference.
    
    Flattens 2D spatial bands into 2D feature matrix of shape (N_pixels, M_features).
    
    Features (7 total):
    1. B04 (Red)
    2. B08 (NIR)
    3. B11 (SWIR1)
    4. B12 (SWIR2)
    5. NDVI
    6. SWIR_Ratio (B11/B12)
    7. Ferrous_Ratio (B11/B08)
    
    Args:
        bands: Dictionary containing all required bands
        
    Returns:
        Tuple of (feature_matrix, valid_pixel_mask, original_shape)
        - feature_matrix: Shape (N_valid_pixels, 7)
        - valid_pixel_mask: Boolean array of shape (H, W) indicating valid pixels
        - original_shape: Tuple (H, W) of original spatial dimensions
    """
    h, w = bands['B04'].shape
    original_shape = (h, w)
    
    ndvi = calculate_ndvi(bands['B08'], bands['B04'])
    swir_ratio = calculate_swir_ratio(bands['B11'], bands['B12'])
    ferrous_ratio = calculate_ferrous_ratio(bands['B11'], bands['B08'])
    
    feature_stack = np.stack([
        bands['B04'],
        bands['B08'],
        bands['B11'],
        bands['B12'],
        ndvi,
        swir_ratio,
        ferrous_ratio
    ], axis=-1)
    
    valid_mask = ~np.any(np.isnan(feature_stack), axis=-1)
    
    feature_matrix = feature_stack[valid_mask]
    
    logger.info(f"Feature matrix shape: {feature_matrix.shape} (valid pixels: {np.sum(valid_mask)}/{h*w})")
    return feature_matrix, valid_mask, original_shape


def reshape_predictions(predictions: np.ndarray, 
                        probabilities: np.ndarray, 
                        valid_mask: np.ndarray, 
                        original_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape 1D predictions back to 2D spatial grid.
    
    Args:
        predictions: 1D array of class predictions
        probabilities: 1D array of class probabilities (positive class)
        valid_mask: Boolean mask of valid pixels
        original_shape: Tuple (H, W) of original spatial dimensions
        
    Returns:
        Tuple of (prediction_grid, probability_grid) both of shape (H, W)
    """
    h, w = original_shape
    
    pred_grid = np.zeros((h, w), dtype=np.uint8)
    prob_grid = np.zeros((h, w), dtype=np.float32)
    
    pred_grid[valid_mask] = predictions
    prob_grid[valid_mask] = probabilities
    
    pred_grid[~valid_mask] = 0
    prob_grid[~valid_mask] = 0.0
    
    return pred_grid, prob_grid