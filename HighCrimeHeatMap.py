import requests
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
import json
warnings.filterwarnings("ignore")

"""

This code predicts the crimes for a year and generates a prediction heat map. 
We train the data on 2024 crimes and then test on 2025.

"""
# Create output directories
os.makedirs('race_maps_output', exist_ok=True)
os.makedirs('data/outputs', exist_ok=True)

print("Starting data collection and processing...")

# ========== STEP 1: GET DATA ==========
endpoint = "https://data.cityofnewyork.us/resource/5uac-w243.json"
params = {
    "$limit": 1400000,
    "$where": "(cmplnt_fr_dt between '2023-01-01T00:00:00' and '2024-12-31T23:59:59')"
}
response = requests.get(endpoint, params=params)
data = response.json()
df = pd.DataFrame(data)
print(f"Downloaded {len(df)} records from NYC Open Data API")

# ========== STEP 2: CLEAN + FEATURE ENGINEER ==========

# Dates
df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')
df['cmplnt_fr_tm'] = pd.to_datetime(df['cmplnt_fr_tm'], format='%H:%M:%S', errors='coerce')
df = df.dropna(subset=['cmplnt_fr_dt'])

# Feature engineering
df['day_of_week'] = df['cmplnt_fr_dt'].dt.dayofweek
df['month'] = df['cmplnt_fr_dt'].dt.month
df['hour_of_day'] = df['cmplnt_fr_tm'].dt.hour
df['borough_code'] = df['boro_nm'].astype('category').cat.codes
df['is_completed'] = (df['crm_atpt_cptd_cd'] == 'COMPLETED').astype(int)
df = df.dropna(subset=['law_cat_cd'])

# ========== STEP 3: SPLIT DATA ==========
features = ['day_of_week', 'month', 'hour_of_day', 'borough_code', 'is_completed']
target = 'law_cat_cd'
df['year'] = df['cmplnt_fr_dt'].dt.year
df_2024 = df[df['year'] == 2024]
df_2025 = df[df['year'] == 2025]

# Ensure we have data for both years
print(f"2024 records: {len(df_2024)}")
print(f"2025 records: {len(df_2025)}")

if len(df_2024) == 0:
    print("WARNING: No 2024 data found. Using all available data for training.")
    df_2024 = df  # Use all data for training if no 2024 data

# Make sure coordinates are properly formatted
for data_df in [df_2024, df_2025]:
    data_df['latitude'] = pd.to_numeric(data_df['latitude'], errors='coerce')
    data_df['longitude'] = pd.to_numeric(data_df['longitude'], errors='coerce')
    
# Drop rows without coordinates
df_2024 = df_2024.dropna(subset=['latitude', 'longitude'])
df_2025 = df_2025.dropna(subset=['latitude', 'longitude'])

X_train = df_2024[features]
y_train = df_2024[target]
X_future = df_2025[features]

# ========== STEP 4: TRAIN MODEL ==========

print("Training RandomForest model...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# ========== STEP 5: PREDICT FOR 2025 ==========
if len(df_2025) > 0:
    df_2025['predicted_label'] = rf_model.predict(X_future)
    high_crime = df_2025[df_2025['predicted_label'] == 'FELONY']
    print(f"Predicted {len(high_crime)} felony incidents for 2025")
else:
    print("WARNING: No 2025 data available for prediction. Using 2024 data with predictions.")
    X_future = df_2024[features]
    df_2024['predicted_label'] = rf_model.predict(X_train)
    high_crime = df_2024[df_2024['predicted_label'] == 'FELONY']

# ========== STEP 6: SAVE RESULTS WITHOUT USING FOLIUM SAVE ==========

# Clean coordinates one more time
high_crime = high_crime[
    high_crime['latitude'].notna() & 
    high_crime['longitude'].notna() &
    (high_crime['latitude'] != 0.0) & 
    (high_crime['longitude'] != 0.0)
]

# Save the CSV first (this should work regardless of map issues)
high_crime.to_csv('data/outputs/predicted_felonies_2025.csv', index=False)
print("Saved CSV of predicted high-crime zones.")

# ========== STEP 7: CREATE HTML MAP DIRECTLY ==========
# This avoids the Folium .save() method that's causing the error

# Prepare data points for the map
heat_data = []
for _, row in high_crime.iterrows():
    try:
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        if np.isfinite(lat) and np.isfinite(lon):
            heat_data.append([lat, lon])
    except (ValueError, TypeError):
        continue

print(f"Preparing map with {len(heat_data)} data points...")

# Convert to JSON string for embedding in HTML
heat_data_json = json.dumps(heat_data)

# Create custom HTML with Leaflet.js and heatmap
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>NYC Predicted Felony Heatmap (2025)</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Load Leaflet CSS and JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
    
    <!-- Load Leaflet Heat plugin -->
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    
    <style>
        body {{
            margin: 0;
            padding: 0;
        }}
        #map {{
            width: 100%;
            height: 100vh;
        }}
        .info {{
            padding: 6px 8px;
            font: 14px/16px Arial, Helvetica, sans-serif;
            background: white;
            background: rgba(255,255,255,0.8);
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <script>
        // Initialize map centered on NYC
        var map = L.map('map').setView([40.7128, -74.0060], 11);
        
        // Add OpenStreetMap tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }}).addTo(map);
        
        // Add information panel
        var info = L.control();
        info.onAdd = function(map) {{
            this._div = L.DomUtil.create('div', 'info');
            this._div.innerHTML = '<h4>NYC Predicted Felony Heatmap (2025)</h4>' +
                                  '<p>Showing predicted locations of felony crimes in NYC for 2025</p>' +
                                  '<p>Total predictions: {len(high_crime)} incidents</p>';
            return this._div;
        }};
        info.addTo(map);
        
        // Add heat map layer with data
        var heat_data = {heat_data_json};
        var heat = L.heatLayer(heat_data, {{
            radius: 10,
            blur: 15,
            maxZoom: 17,
            gradient: {{0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}}
        }}).addTo(map);
    </script>
</body>
</html>
"""

# Save the HTML file
html_file_path = 'race_maps_output/nyc_predicted_high_crime_2025.html'
with open(html_file_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Successfully created heatmap HTML: {html_file_path}")
print("Analysis complete!")


