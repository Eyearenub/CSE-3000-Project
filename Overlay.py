import pandas as pd
import geopandas as gpd
import json
import os
import requests
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

# Create output directories
os.makedirs('race_maps_output', exist_ok=True)
os.makedirs('data/outputs', exist_ok=True)

# -------------------------------------------------------------------
# 1. DOWNLOAD NYPD COMPLAINTS DATA (from testcrime.py)
# -------------------------------------------------------------------
print("Fetching NYPD complaints...")
endpoint = "https://data.cityofnewyork.us/resource/5uac-w243.json"
params = {
    "$limit": 1_400_000,
    "$where": "(cmplnt_fr_dt between '2023-01-01T00:00:00' and '2025-12-31T23:59:59')"
}
df = pd.DataFrame(requests.get(endpoint, params=params).json())
print(f"Rows downloaded: {len(df):,}")

# -------------------------------------------------------------------
# 2. CLEAN / FEATURE ENGINEER (from testcrime.py)
# -------------------------------------------------------------------
df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')
df['cmplnt_fr_tm'] = pd.to_datetime(df['cmplnt_fr_tm'], format='%H:%M:%S', errors='coerce')
df = df.dropna(subset=['cmplnt_fr_dt', 'law_cat_cd'])

# date/time / categorical engineering
df['day_of_week'] = df['cmplnt_fr_dt'].dt.dayofweek
df['month'] = df['cmplnt_fr_dt'].dt.month
df['hour_of_day'] = df['cmplnt_fr_tm'].dt.hour

df['sus_race_code'] = df['susp_race'].astype('category').cat.codes
df['vic_race_code'] = df['vic_race'].astype('category').cat.codes
df['borough_code'] = df['boro_nm'].astype('category').cat.codes
df['is_completed'] = (df['crm_atpt_cptd_cd'] == 'COMPLETED').astype(int)

df['year'] = df['cmplnt_fr_dt'].dt.year
features = ['day_of_week', 'month', 'hour_of_day', 'sus_race_code', 'vic_race_code', 'borough_code', 'is_completed']

# Split data by year
train_df = df[df['year'] == 2024].copy()
score_df = df[df['year'] == 2025].copy()

# Fix coordinates
for d in (train_df, score_df):
    d['latitude'] = pd.to_numeric(d['latitude'], errors='coerce')
    d['longitude'] = pd.to_numeric(d['longitude'], errors='coerce')
    d.dropna(subset=['latitude', 'longitude'], inplace=True)

print(f"2024 train rows: {len(train_df):,}")
print(f"2025 score rows: {len(score_df):,}")

# -------------------------------------------------------------------
# 3. TRAIN RANDOM FOREST MODEL (from testcrime.py)
# -------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100,
                           class_weight='balanced',
                           random_state=42)
rf.fit(train_df[features], train_df['law_cat_cd'])
print("Model fitted.")

# -------------------------------------------------------------------
# 4. PROCESS ACTUAL AND PREDICTED CRIMES
# -------------------------------------------------------------------
# First, save actual 2025 crime data by category
actual_heat_lists = {}
for cat in ['FELONY', 'MISDEMEANOR', 'VIOLATION']:
    actual_slice_df = score_df[score_df['law_cat_cd'] == cat]
    actual_slice_df.to_csv(f'data/outputs/actual_{cat.lower()}s_2025.csv', index=False)
    print(f"Saved → actual_{cat.lower()}s_2025.csv ({len(actual_slice_df):,} rows)")
    actual_heat_lists[cat] = [[r['latitude'], r['longitude']] for _, r in actual_slice_df.iterrows()]

# Now generate and save predictions
score_df['predicted_label'] = rf.predict(score_df[features])
score_df.to_csv('data/outputs/predicted_crimes_2025.csv', index=False)
print("Saved → data/outputs/predicted_crimes_2025.csv")

# Class-specific CSVs & heat lists for predictions
predicted_heat_lists = {}
for cat in ['FELONY', 'MISDEMEANOR', 'VIOLATION']:
    slice_df = score_df[score_df['predicted_label'] == cat]
    slice_df.to_csv(f'data/outputs/predicted_{cat.lower()}s_2025.csv', index=False)
    print(f"Saved → predicted_{cat.lower()}s_2025.csv ({len(slice_df):,} rows)")
    predicted_heat_lists[cat] = [[r['latitude'], r['longitude']] for _, r in slice_df.iterrows()]

# -------------------------------------------------------------------
# 5. LOAD DEMOGRAPHICS DATA (from overlay.py)
# -------------------------------------------------------------------
print("Loading demographics data...")
try:
    gdf = gpd.read_file("nyc_blocks_with_race_data.geojson")
    # Simplify GeoDataFrame for performance
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0025, preserve_topology=True)

    # Select demographic fields
    gdf = gdf[[
        'geometry',
        'Percent_Black', 'Percent_White', 'Percent_Asian',
        'Percent_American_and_Alaska_Native',
        'Percent_Native_Hawaiian_AAPI',
        'Percent_Some_Other_Race',
        'Percent_Two_or_more_races'
    ]]
    gdf_json = gdf.to_json()
    print("Demographics data loaded successfully.")
except Exception as e:
    print(f"Error loading demographics data: {e}")
    print("Creating empty GeoJSON as fallback.")
    gdf_json = '{}'

# -------------------------------------------------------------------
# 6. GET BOROUGH BOUNDARIES FOR LABELS
# -------------------------------------------------------------------
print("Loading borough boundaries...")
try:
    # For this example, we'll create a simplified dictionary of borough coordinates
    # In a real implementation, you might load this from a GeoJSON file
    borough_centers = {
        "MANHATTAN": [40.7831, -73.9712],
        "BROOKLYN": [40.6782, -73.9442],
        "QUEENS": [40.7282, -73.7949],
        "BRONX": [40.8448, -73.8648],
        "STATEN ISLAND": [40.5795, -74.1502]
    }
    print("Borough data loaded.")
except Exception as e:
    print(f"Error loading borough data: {e}")
    borough_centers = {}

# -------------------------------------------------------------------
# 7. BUILD COMBINED OVERLAY MAP
# -------------------------------------------------------------------
print("Building combined overlay map...")

# First, create the HTML template with placeholders that don't involve f-string backslash issues
html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>NYC Crime Prediction vs Actual & Demographics (2025)</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css">
    <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <style>
        html, body, #map {
            margin: 0;
            height: 100vh;
            width: 100%;
        }
        .info {
            padding: 6px 8px;
            font: 14px/16px Arial, Helvetica, sans-serif;
            background: white;
            background: rgba(255,255,255,0.8);
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            border-radius: 5px;
        }
        .legend {
            line-height: 18px;
            color: #555;
        }
        .legend i {
            width: 18px;
            height: 18px;
            float: left;
            margin-right: 8px;
            opacity: 0.7;
        }
        .borough-label {
            background: rgba(255,255,255,0.5);
            border: none;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.2);
            padding: 3px 5px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        // Initialize map
        var map = L.map('map').setView([40.7128, -74.0060], 11);
        
        // Add base tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);
        
        // Info panel
        var info = L.control({position: 'bottomleft'});
        info.onAdd = function() {
            this._div = L.DomUtil.create('div', 'info');
            this.update();
            return this._div;
        };
        
        info.update = function(props) {
            this._div.innerHTML = '<h4>NYC Crime & Demographics (2025)</h4>' +
                '<p>Toggle layers in top-right.<br>' +
                'PREDICTED: Model predictions<br>' +
                'ACTUAL: Real crime data<br>' +
                'Click census blocks for demographic details.</p>';
        };
        info.addTo(map);
        
        // ----- Heat layers for PREDICTED crime classes -----
        var predictedFelonyHeat = L.heatLayer(PREDICTED_FELONY_HEAT_PLACEHOLDER, {
            radius: 10, 
            blur: 15, 
            maxZoom: 17,
            gradient: {0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        });
        
        var predictedMisdemeanorHeat = L.heatLayer(PREDICTED_MISDEMEANOR_HEAT_PLACEHOLDER, {
            radius: 10, 
            blur: 15, 
            maxZoom: 17,
            gradient: {0.2: 'purple', 0.4: 'magenta', 0.6: 'orange', 0.8: 'yellow', 1.0: 'white'}
        });
        
        var predictedViolationHeat = L.heatLayer(PREDICTED_VIOLATION_HEAT_PLACEHOLDER, {
            radius: 10, 
            blur: 15, 
            maxZoom: 17,
            gradient: {0.2: 'green', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
        });
        
        // ----- Heat layers for ACTUAL crime classes -----
        var actualFelonyHeat = L.heatLayer(ACTUAL_FELONY_HEAT_PLACEHOLDER, {
            radius: 10, 
            blur: 15, 
            maxZoom: 17,
            gradient: {0.2: 'darkblue', 0.4: 'blue', 0.6: 'royalblue', 0.8: 'lightskyblue', 1.0: 'deepskyblue'}
        });
        
        var actualMisdemeanorHeat = L.heatLayer(ACTUAL_MISDEMEANOR_HEAT_PLACEHOLDER, {
            radius: 10, 
            blur: 15, 
            maxZoom: 17,
            gradient: {0.2: 'darkmagenta', 0.4: 'darkviolet', 0.6: 'mediumorchid', 0.8: 'plum', 1.0: 'violet'}
        });
        
        var actualViolationHeat = L.heatLayer(ACTUAL_VIOLATION_HEAT_PLACEHOLDER, {
            radius: 10, 
            blur: 15, 
            maxZoom: 17,
            gradient: {0.2: 'darkgreen', 0.4: 'green', 0.6: 'forestgreen', 0.8: 'limegreen', 1.0: 'lime'}
        });
        
        // ----- Census demographics layer -----
        function onEachFeature(feature, layer) {
            if (feature.properties) {
                let props = feature.properties;
                let content = `
                    <b>Demographics:</b><br>
                    % Black: ${props.Percent_Black.toFixed(1)}%<br>
                    % White: ${props.Percent_White.toFixed(1)}%<br>
                    % Asian: ${props.Percent_Asian.toFixed(1)}%<br>
                    % Am. Indian/Alaska Native: ${props.Percent_American_and_Alaska_Native.toFixed(1)}%<br>
                    % Native Hawaiian/AAPI: ${props.Percent_Native_Hawaiian_AAPI.toFixed(1)}%<br>
                    % Some Other Race: ${props.Percent_Some_Other_Race.toFixed(1)}%<br>
                    % Two or More Races: ${props.Percent_Two_or_more_races.toFixed(1)}%<br>
                `;
                layer.bindPopup(content);
            }
        }
        
        var demographicsLayer = L.geoJSON(DEMOGRAPHICS_GEOJSON_PLACEHOLDER, {
            style: function(feature) {
                return {
                    color: "black",
                    weight: 0.3,
                    fillOpacity: 0
                };
            },
            onEachFeature: onEachFeature
        });
        
        // ----- Borough labels -----
        var boroughLabels = L.layerGroup();
        
        // Add borough label markers
        BOROUGH_LABELS_PLACEHOLDER
        
        // ----- Set up layer controls -----
        var baseLayers = {
            "OpenStreetMap": L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors'
            })
        };
        
        var overlayLayers = {
            // Predicted crime layers
            "PREDICTED - Felonies": predictedFelonyHeat,
            "PREDICTED - Misdemeanors": predictedMisdemeanorHeat,
            "PREDICTED - Violations": predictedViolationHeat,
            
            // Actual crime layers
            "ACTUAL - Felonies": actualFelonyHeat,
            "ACTUAL - Misdemeanors": actualMisdemeanorHeat,
            "ACTUAL - Violations": actualViolationHeat,
            
            // Base layers
            "Census Demographics": demographicsLayer,
            "Borough Labels": boroughLabels
        };
        
        L.control.layers(baseLayers, overlayLayers, {collapsed: false}).addTo(map);
        
        // Add default layers to map
        demographicsLayer.addTo(map);
        boroughLabels.addTo(map);
        predictedFelonyHeat.addTo(map);
        actualFelonyHeat.addTo(map);
        
        // Add legend for heat map gradients
        var legend = L.control({position: 'bottomright'});
        legend.onAdd = function(map) {
            var div = L.DomUtil.create('div', 'info legend');
            div.innerHTML = '<h4>Legend</h4>' +
                            '<b>Predicted Crime:</b><br>' +
                            '<i style="background: red"></i> High<br>' +
                            '<i style="background: yellow"></i> Medium<br>' +
                            '<i style="background: blue"></i> Low<br>' +
                            '<br>' +
                            '<b>Actual Crime:</b><br>' +
                            '<i style="background: deepskyblue"></i> Felony<br>' +
                            '<i style="background: violet"></i> Misdemeanor<br>' +
                            '<i style="background: lime"></i> Violation<br>';
            return div;
        };
        legend.addTo(map);
    </script>
</body>
</html>
"""

# Create the borough labels
borough_labels_code = []
for borough, coords in borough_centers.items():
    label_code = f"""L.marker([{coords[0]}, {coords[1]}], {{
            icon: L.divIcon({{
                className: 'borough-label',
                html: '{borough}',
                iconSize: [100, 20],
                iconAnchor: [50, 10]
            }})
        }}).addTo(boroughLabels);"""
    borough_labels_code.append(label_code)

# Replace all the placeholders
html = html_template.replace('PREDICTED_FELONY_HEAT_PLACEHOLDER', json.dumps(predicted_heat_lists.get('FELONY', [])))
html = html.replace('PREDICTED_MISDEMEANOR_HEAT_PLACEHOLDER', json.dumps(predicted_heat_lists.get('MISDEMEANOR', [])))
html = html.replace('PREDICTED_VIOLATION_HEAT_PLACEHOLDER', json.dumps(predicted_heat_lists.get('VIOLATION', [])))
html = html.replace('ACTUAL_FELONY_HEAT_PLACEHOLDER', json.dumps(actual_heat_lists.get('FELONY', [])))
html = html.replace('ACTUAL_MISDEMEANOR_HEAT_PLACEHOLDER', json.dumps(actual_heat_lists.get('MISDEMEANOR', [])))
html = html.replace('ACTUAL_VIOLATION_HEAT_PLACEHOLDER', json.dumps(actual_heat_lists.get('VIOLATION', [])))

# Replace demographics geojson
if gdf_json != '{}':
    html = html.replace('DEMOGRAPHICS_GEOJSON_PLACEHOLDER', json.dumps(json.loads(gdf_json)))
else:
    html = html.replace('DEMOGRAPHICS_GEOJSON_PLACEHOLDER', '{}')

# Replace borough labels
html = html.replace('BOROUGH_LABELS_PLACEHOLDER', '\n        '.join(borough_labels_code))

output_path = 'race_maps_output/nyc_crime_demographics_overlay.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Combined overlay map saved → {output_path}")
print("Done ✓")