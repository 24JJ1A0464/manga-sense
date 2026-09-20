# Manga-Sense 🌍

AI-powered manganese mineral deposit detection pipeline using multi-spectral Sentinel-2 satellite imagery and machine learning.

## Overview

Manganese-Sense is an end-to-end Python pipeline that:
1. **Ingests** Sentinel-2 L2A surface reflectance data from Microsoft Planetary Computer STAC API
2. **Processes** spectral bands using memory-efficient windowed reads (no full GeoTIFF downloads)
3. **Filters** clouds (SCL band) and vegetation (NDVI > 0.3)
4. **Engineers** spectral features: NDVI, SWIR Ratio (B11/B12), Ferrous Composite Ratio (B11/B08)
5. **Classifies** manganese deposits using Random Forest (100 trees, balanced weights)
6. **Filters** unfeasible terrain using Copernicus DEM GLO-30 (slope > 30° excluded)
7. **Serves** results via interactive Streamlit + Folium dashboard

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Streamlit Dashboard (Recommended)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Command Line Pipeline

```bash
python pipeline.py --bbox "77.5,12.9,77.7,13.1" --date-range "2024-01-01/2024-12-31" --top-n 50 --output results.json
```

## Architecture

```
project/
├── app.py              # Streamlit dashboard
├── pipeline.py         # CLI pipeline orchestrator
├── data_ingestion.py   # Phase 1: STAC API + windowed reads
├── processor.py        # Phase 2-3: Masking + Feature Engineering
├── model.py            # Phase 3: Random Forest ML
├── terrain.py          # Phase 4: DEM + Slope + Feasibility
├── requirements.txt    # Dependencies
└── spec.txt            # Original specification
```

## Spectral Indices (Mathematical Foundations)

| Index | Formula | Purpose |
|-------|---------|---------|
| **NDVI** | `(B08 - B04) / (B08 + B04)` | Vegetation masking (threshold: > 0.3) |
| **SWIR Ratio** | `B11 / B12` | Mineral alteration detection |
| **Ferrous Ratio** | `B11 / B08` | Ferrous mineral enhancement (Mn, Fe oxides) |

## Cloud Masking (SCL Classes)

Pixels masked when SCL equals:
- **3**: Cloud shadows
- **8**: Cloud medium probability
- **9**: Cloud high probability
- **10**: Thin cirrus

## ML Features (7 dimensions)

1. B04 (Red, 10m)
2. B08 (NIR, 10m)
3. B11 (SWIR1, 20m)
4. B12 (SWIR2, 20m)
5. NDVI
6. SWIR Ratio
7. Ferrous Ratio

## Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)
```

## Terrain Filtering

- **Source**: Copernicus DEM GLO-30 (30m resolution)
- **Method**: `numpy.gradient` for slope calculation
- **Threshold**: Slope > 30° → reclassified as invalid (0)

## Output

- **Interactive Map**: Folium map with BBox boundary + clickable detection markers
- **CSV Export**: Latitude, Longitude, Probability for top N detections
- **GeoJSON**: FeatureCollection for GIS integration

## Example Regions

| Region | Bounding Box |
|--------|--------------|
| Karnataka, India | `77.5,12.9,77.7,13.1` |
| Kalahari Mn Field, South Africa | `22.5,-27.5,23.5,-26.5` |
| Pilbara, Western Australia | `118.5,-22.5,119.5,-21.5` |
| Carajás, Brazil | `-51.5,-6.5,-50.5,-5.5` |

## Requirements

- Python 3.10+
- Internet connection (Planetary Computer API)
- 4GB+ RAM recommended

## License

MIT License - Built for Prospecta Team
