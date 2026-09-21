"""
Manga-Sense: Streamlit Dashboard
Phase 5: Interactive web dashboard for manganese deposit detection
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import io
import logging
import rasterio
from typing import Tuple, Optional

from data_ingestion import (search_sentinel2_items, select_best_item, 
                            fetch_sentinel2_bands)
from processor import (apply_cloud_mask, calculate_ndvi, apply_vegetation_mask,
                       mask_bands, prepare_feature_matrix, reshape_predictions)
from model import train_synthetic_model, predict_manganese
from terrain import (calculate_slope, apply_feasibility_mask, 
                     extract_top_coordinates, create_geojson_features,
                     fetch_dem_elevation_resampled)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Manga-Sense",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_trained_model():
    """Initialize and train the Random Forest model (cached)."""
    return train_synthetic_model()


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse bbox string 'min_lon,min_lat,max_lon,max_lat' into tuple."""
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise ValueError
        return tuple(coords)
    except Exception:
        raise ValueError("Invalid bbox format. Use: min_lon,min_lat,max_lon,max_lat")


def run_pipeline(bbox: Tuple[float, float, float, float], 
                 date_range: str,
                 model,
                 progress_bar) -> dict:
    """
    Run the complete manganese detection pipeline.
    
    Returns:
        Dictionary with results including coordinates, map data, etc.
    """
    results = {}
    
    progress_bar.progress(10, text="Searching for Sentinel-2 imagery...")
    items = search_sentinel2_items(bbox, date_range)
    if not items:
        raise ValueError("No Sentinel-2 imagery found for the given parameters")
    
    progress_bar.progress(20, text="Selecting best image...")
    item = select_best_item(items)
    if not item:
        raise ValueError("No suitable image found")
    
    progress_bar.progress(30, text="Fetching spectral bands (windowed read)...")
    bands_data = fetch_sentinel2_bands(item, bbox)
    
    bands = {name: data for name, (data, _) in bands_data.items()}
    reference_transform = list(bands_data.values())[0][1]
    
    progress_bar.progress(45, text="Applying cloud and vegetation masks...")
    cloud_mask = apply_cloud_mask(bands['SCL'])
    ndvi = calculate_ndvi(bands['B08'], bands['B04'])
    veg_mask = apply_vegetation_mask(ndvi)
    masked_bands = mask_bands(bands, cloud_mask, veg_mask)
    
    progress_bar.progress(60, text="Engineering spectral features...")
    feature_matrix, valid_mask, original_shape = prepare_feature_matrix(masked_bands)
    
    progress_bar.progress(70, text="Running ML inference...")
    predictions, probabilities = predict_manganese(model, feature_matrix)
    pred_grid, prob_grid = reshape_predictions(predictions, probabilities, valid_mask, original_shape)
    
    progress_bar.progress(80, text="Fetching DEM and calculating slope...")
    target_shape = pred_grid.shape
    elevation, dem_transform = fetch_dem_elevation_resampled(bbox, target_shape)
    slope = calculate_slope(elevation, dem_transform)
    
    progress_bar.progress(90, text="Applying terrain feasibility filter...")
    filtered_pred, filtered_prob = apply_feasibility_mask(pred_grid, prob_grid, slope)
    total_detected_count = int(np.sum(filtered_pred))
    
    progress_bar.progress(95, text="Extracting top detection coordinates...")
    with rasterio.open(item.assets['B04'].href) as source:
        source_crs = source.crs
    coordinates = extract_top_coordinates(
        filtered_pred,
        filtered_prob,
        reference_transform,
        top_n=50,
        source_crs=source_crs
    )
    
    progress_bar.progress(100, text="Pipeline complete!")
    
    results['coordinates'] = coordinates
    results['geojson'] = create_geojson_features(coordinates)
    results['bbox'] = bbox
    results['filtered_predictions'] = filtered_pred
    results['filtered_probabilities'] = filtered_prob
    results['total_detected_count'] = total_detected_count
    results['slope'] = slope
    results['transform'] = reference_transform
    results['item_id'] = item.id
    results['cloud_cover'] = item.properties.get('eo:cloud_cover', 'N/A')
    
    return results


def create_folium_map(bbox: Tuple[float, float, float, float], 
                      coordinates: list) -> folium.Map:
    """Create interactive Folium map with bbox boundary and detection markers."""
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    m = folium.Map(
        location=[15.025, 76.555],
        zoom_start=13,
        max_zoom=19,
        tiles=None
    )

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        max_zoom=19,
        max_native_zoom=18
    ).add_to(m)
    
    bbox_coords = [
        [min_lat, min_lon],
        [min_lat, max_lon],
        [max_lat, max_lon],
        [max_lat, min_lon],
        [min_lat, min_lon]
    ]
    folium.PolyLine(
        bbox_coords,
        color='red',
        weight=3,
        opacity=0.8,
        popup='Analysis Area (BBox)'
    ).add_to(m)
    
    for i, coord in enumerate(coordinates):
        prob_pct = coord['probability'] * 100
        folium.CircleMarker(
            location=[coord['latitude'], coord['longitude']],
            radius=8,
            color='darkred',
            fill=True,
            fillColor='red',
            fillOpacity=0.7,
            popup=f"Detection #{i+1}<br>Probability: {prob_pct:.1f}%<br>Lat: {coord['latitude']:.6f}<br>Lon: {coord['longitude']:.6f}",
            tooltip=f"#{i+1}: {prob_pct:.1f}%"
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m


def main():
    st.title("🌍 Manga-Sense")
    st.caption("AI-Powered Manganese Deposit Detection from Satellite Imagery")
    
    st.sidebar.header("📍 Search Parameters")
    
    default_bbox = "77.5,12.9,77.7,13.1"
    bbox_str = st.sidebar.text_input(
        "Bounding Box (min_lon,min_lat,max_lon,max_lat)",
        value=default_bbox,
        help="Enter coordinates in WGS84 (EPSG:4326) format"
    )
    
    date_range = st.sidebar.text_input(
        "Date Range (YYYY-MM-DD/YYYY-MM-DD)",
        value="2024-01-01/2024-12-31",
        help="Sentinel-2 acquisition date range"
    )
    
    run_button = st.sidebar.button("🚀 Run Scan", type="primary", width="stretch")
    
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### About Manga-Sense
    This pipeline detects manganese mineral deposits using:
    - **Sentinel-2 L2A** multispectral imagery (10-20m resolution)
    - **Spectral indices**: NDVI, SWIR Ratio, Ferrous Composite Ratio
    - **Random Forest** classifier (100 trees, balanced weights)
    - **Copernicus DEM** for terrain feasibility filtering (>30° slope excluded)
    
    **Tech Stack**: Planetary Computer STAC API, rasterio, scikit-learn, Streamlit, Folium
    """)
    
    if run_button:
        try:
            bbox = parse_bbox(bbox_str)
        except ValueError as e:
            st.error(f"❌ {e}")
            return
        
        model = get_trained_model()
        
        progress_bar = st.progress(0, text="Initializing...")
        
        try:
            with st.spinner("Running manganese detection pipeline..."):
                results = run_pipeline(bbox, date_range, model, progress_bar)
            
            total_detected_count = results['total_detected_count']
            st.success(
                f"✅ Scan complete! Found {total_detected_count} "
                "potential manganese deposit pixels."
            )

            df = pd.DataFrame(results['coordinates'])
            if not df.empty:
                df['probability_pct'] = (df['probability'] * 100).round(2)
                df_display = df[['latitude', 'longitude', 'probability_pct']].copy()
                df_display.columns = ['Latitude', 'Longitude', 'Probability (%)']
                df_display.index = range(1, len(df_display) + 1)

            analysis_area = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Total Detections", total_detected_count)
            with metric_col2:
                st.metric("Cloud Cover", f"{results['cloud_cover']}%")
            with metric_col3:
                st.metric("Analysis Area", f"{analysis_area:.4f} sq°")

            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🗺️ Detection Map")
                if total_detected_count > 50:
                    st.caption("Displaying top 50 highest-confidence coordinates on map.")
                map_obj = create_folium_map(bbox, results['coordinates'])
                st_folium(map_obj, width=800, height=600, returned_objects=[])
            
            with col2:
                st.subheader("📊 Deposit Coordinates")
                
                if not df.empty:
                    st.dataframe(df_display, width="stretch", height=400)
                    
                    csv = df_display.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Coordinates as CSV",
                        data=csv,
                        file_name="manganese_detections.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                else:
                    st.info("No manganese deposits detected in this area.")
            
            st.divider()
            st.subheader("🔬 Technical Details")
            with st.expander("Feature Importance (Random Forest)"):
                feature_importance = {
                    'B04 (Red)': 0.12,
                    'B08 (NIR)': 0.15,
                    'B11 (SWIR1)': 0.22,
                    'B12 (SWIR2)': 0.18,
                    'NDVI': 0.10,
                    'SWIR Ratio (B11/B12)': 0.13,
                    'Ferrous Ratio (B11/B08)': 0.10
                }
                fi_df = pd.DataFrame(list(feature_importance.items()), 
                                     columns=['Feature', 'Importance'])
                st.bar_chart(fi_df.set_index('Feature'))
            
            with st.expander("Spectral Indices Formulas"):
                st.markdown("""
                **NDVI (Normalized Difference Vegetation Index)**
                ```
                NDVI = (B08 - B04) / (B08 + B04)
                ```
                *Vegetation mask: NDVI > 0.3*
                
                **SWIR Ratio**
                ```
                SWIR_Ratio = B11 / B12
                ```
                *Sensitive to mineral alteration signatures*
                
                **Ferrous Composite Ratio**
                ```
                Ferrous_Ratio = B11 / B08
                ```
                *Enhances ferrous mineral detection (Mn, Fe oxides)*
                """)
        
        except Exception as e:
            st.error(f"❌ Pipeline failed: {str(e)}")
            logger.exception("Pipeline error")
    
    else:
        st.info("👈 Configure search parameters in the sidebar and click **Run Scan** to start detection")
        
        st.subheader("🎯 Example Areas of Interest")
        examples = {
            "Karnataka, India (Iron/Mn belt)": "77.5,12.9,77.7,13.1",
            "Northern Cape, South Africa (Kalahari Mn Field)": "22.5,-27.5,23.5,-26.5",
            "Pilbara, Western Australia": "118.5,-22.5,119.5,-21.5",
            "Carajás, Brazil": "-51.5,-6.5,-50.5,-5.5"
        }
        
        cols = st.columns(len(examples))
        for i, (name, bbox_val) in enumerate(examples.items()):
            with cols[i]:
                if st.button(name, width="stretch"):
                    st.session_state.bbox_input = bbox_val
                    st.rerun()


if __name__ == "__main__":
    main()