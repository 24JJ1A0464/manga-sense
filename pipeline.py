"""
Manganese-Sense: Main Pipeline Orchestrator
Command-line interface for running the full detection pipeline.
"""

import argparse
import json
import logging
import rasterio
import sys
from typing import Tuple
import numpy as np

from data_ingestion import (search_sentinel2_items, select_best_item, 
                            fetch_sentinel2_bands)
from processor import (apply_cloud_mask, calculate_ndvi, apply_vegetation_mask,
                       mask_bands, prepare_feature_matrix, reshape_predictions)
from model import train_synthetic_model, predict_manganese, get_feature_importance
from terrain import (calculate_slope, apply_feasibility_mask, 
                      extract_top_coordinates, create_geojson_features,
                      fetch_dem_elevation_resampled)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse bbox string 'min_lon,min_lat,max_lon,max_lat' into tuple."""
    coords = [float(x.strip()) for x in bbox_str.split(',')]
    if len(coords) != 4:
        raise ValueError("Invalid bbox format. Use: min_lon,min_lat,max_lon,max_lat")
    return tuple(coords)


def run_pipeline(bbox: Tuple[float, float, float, float], 
                 date_range: str,
                 top_n: int = 50,
                 output_json: str = None) -> dict:
    """
    Run the complete manganese detection pipeline.
    """
    logger.info(f"Starting pipeline for bbox={bbox}, date_range={date_range}")
    
    model = train_synthetic_model()
    
    logger.info("Fetching Sentinel-2 imagery...")
    items = search_sentinel2_items(bbox, date_range)
    if not items:
        raise ValueError("No Sentinel-2 imagery found")
    
    item = select_best_item(items)
    logger.info(f"Selected item: {item.id}")
    
    logger.info("Reading spectral bands (windowed)...")
    bands_data = fetch_sentinel2_bands(item, bbox)
    bands = {name: data for name, (data, _) in bands_data.items()}
    reference_transform = list(bands_data.values())[0][1]
    
    logger.info("Applying cloud and vegetation masks...")
    cloud_mask = apply_cloud_mask(bands['SCL'])
    ndvi = calculate_ndvi(bands['B08'], bands['B04'])
    veg_mask = apply_vegetation_mask(ndvi)
    masked_bands = mask_bands(bands, cloud_mask, veg_mask)
    
    logger.info("Preparing feature matrix...")
    feature_matrix, valid_mask, original_shape = prepare_feature_matrix(masked_bands)
    
    logger.info("Running ML inference...")
    predictions, probabilities = predict_manganese(model, feature_matrix)
    pred_grid, prob_grid = reshape_predictions(predictions, probabilities, valid_mask, original_shape)
    
    logger.info("Fetching DEM and calculating slope...")
    target_shape = pred_grid.shape
    elevation, dem_transform = fetch_dem_elevation_resampled(bbox, target_shape)
    slope = calculate_slope(elevation, dem_transform)
    
    logger.info("Applying terrain feasibility filter...")
    filtered_pred, filtered_prob = apply_feasibility_mask(pred_grid, prob_grid, slope)
    total_detected_count = int(np.sum(filtered_pred))
    
    logger.info("Extracting top detection coordinates...")
    with rasterio.open(item.assets['B04'].href) as source:
        source_crs = source.crs
    display_top_n = min(top_n, 50)
    coordinates = extract_top_coordinates(
        filtered_pred,
        filtered_prob,
        reference_transform,
        display_top_n,
        source_crs
    )
    
    results = {
        'coordinates': coordinates,
        'geojson': create_geojson_features(coordinates),
        'bbox': bbox,
        'item_id': item.id,
        'cloud_cover': item.properties.get('eo:cloud_cover', 'N/A'),
        'feature_importance': get_feature_importance(model),
        'n_detections_raw': int(np.sum(pred_grid)),
        'n_detections_filtered': total_detected_count,
        'total_detected_count': total_detected_count,
    }
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_json}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Manganese-Sense: Manganese Deposit Detection Pipeline")
    parser.add_argument('--bbox', type=str, required=True, 
                        help='Bounding box: min_lon,min_lat,max_lon,max_lat')
    parser.add_argument('--date-range', type=str, default='2024-01-01/2024-12-31',
                        help='Date range YYYY-MM-DD/YYYY-MM-DD')
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of top detections to return')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    try:
        bbox = parse_bbox(args.bbox)
        results = run_pipeline(bbox, args.date_range, args.top_n, args.output)
        
        print(f"\n{'='*50}")
        print(f"MANGANESE-SENSE DETECTION RESULTS")
        print(f"{'='*50}")
        print(f"Image: {results['item_id']}")
        print(f"Cloud Cover: {results['cloud_cover']}%")
        print(f"Raw Detections: {results['n_detections_raw']}")
        print(f"Filtered Detections: {results['n_detections_filtered']}")
        print(f"\nTop {len(results['coordinates'])} Coordinates:")
        for i, coord in enumerate(results['coordinates'], 1):
            print(f"  {i:2d}. Lat: {coord['latitude']:.6f}, Lon: {coord['longitude']:.6f}, "
                  f"Prob: {coord['probability']*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()