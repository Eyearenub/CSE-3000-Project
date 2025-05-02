# ------------------------------------------------------------------------------
# NYC Crime Prediction (2025) — layered heat‑map for FELONY / MISDEMEANOR / VIOLATION
# ------------------------------------------------------------------------------
# 1) Downloads 2023‑25 NYPD complaint data
# 2) Trains a RandomForest on 2024 only
# 3) Predicts law_cat_cd for 2025
# 4) Writes one CSV with **all** predictions + per‑class CSVs
# 5) Builds an HTML heat‑map where each class is a toggleable layer
# ------------------------------------------------------------------------------
# Output files:
#   data/outputs/predicted_crimes_2025.csv
#   data/outputs/predicted_felonies_2025.csv
#   data/outputs/predicted_misdemeanors_2025.csv
#   data/outputs/predicted_violations_2025.csv
#   race_maps_output/nyc_predicted_crimes_2025_layers.html  (interactive map)
# ------------------------------------------------------------------------------

import requests, os, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 0.  FOLDERS
# -------------------------------------------------------------------
os.makedirs('race_maps_output', exist_ok=True)
os.makedirs('data/outputs',      exist_ok=True)

# -------------------------------------------------------------------
# 1.  DOWNLOAD 2023‑25 DATA
# -------------------------------------------------------------------
print("Fetching NYPD complaints …")
endpoint = "https://data.cityofnewyork.us/resource/5uac-w243.json"
params = {
    "$limit": 1_400_000,
    "$where": "(cmplnt_fr_dt between '2023-01-01T00:00:00' and '2025-12-31T23:59:59')"
}
df = pd.DataFrame(requests.get(endpoint, params=params).json())
print(f"Rows downloaded: {len(df):,}")

# -------------------------------------------------------------------
# 2.  CLEAN / FEATURE ENGINEER
# -------------------------------------------------------------------
df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')
df['cmplnt_fr_tm'] = pd.to_datetime(df['cmplnt_fr_tm'], format='%H:%M:%S', errors='coerce')
df = df.dropna(subset=['cmplnt_fr_dt', 'law_cat_cd'])

# date/time / categorical engineering
df['day_of_week']  = df['cmplnt_fr_dt'].dt.dayofweek
df['month']        = df['cmplnt_fr_dt'].dt.month
df['hour_of_day']  = df['cmplnt_fr_tm'].dt.hour


df['sus_race_code'] = df['susp_race'].astype('category').cat.codes
df['vic_race_code'] = df['vic_race'].astype('category').cat.codes
df['borough_code'] = df['boro_nm'].astype('category').cat.codes
df['is_completed'] = (df['crm_atpt_cptd_cd'] == 'COMPLETED').astype(int)


df['year']         = df['cmplnt_fr_dt'].dt.year
features = ['day_of_week','month','hour_of_day','sus_race_code','vic_race_code','borough_code','is_completed']

# split
train_df  = df[df['year'] == 2024].copy()
score_df  = df[df['year'] == 2025].copy()

# coordinates -> numeric; keep only rows w/ finite lat/lon
for d in (train_df, score_df):
    d['latitude']  = pd.to_numeric(d['latitude'],  errors='coerce')
    d['longitude'] = pd.to_numeric(d['longitude'], errors='coerce')
    d.dropna(subset=['latitude','longitude'], inplace=True)

print(f"2024 train rows: {len(train_df):,}")
print(f"2025 score rows: {len(score_df):,}")

# -------------------------------------------------------------------
# 3.  TRAIN RANDOM‑FOREST
# -------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100,
                            class_weight='balanced',
                            random_state=42)
rf.fit(train_df[features], train_df['law_cat_cd'])
print("Model fitted.")

# -------------------------------------------------------------------
# 4.  PREDICT 2025 (ALL CLASSES)
# -------------------------------------------------------------------
score_df['predicted_label'] = rf.predict(score_df[features])
score_df.to_csv('data/outputs/predicted_crimes_2025.csv', index=False)
print("Saved → data/outputs/predicted_crimes_2025.csv")

# class‑specific CSVs & heat lists
heat_lists = {}
for cat in ['FELONY','MISDEMEANOR','VIOLATION']:
    slice_df = score_df[score_df['predicted_label'] == cat]
    slice_df.to_csv(f'data/outputs/predicted_{cat.lower()}s_2025.csv', index=False)
    print(f"Saved → predicted_{cat.lower()}s_2025.csv  ({len(slice_df):,} rows)")
    heat_lists[cat] = [[r['latitude'], r['longitude']] for _, r in slice_df.iterrows()]

print("Building layered heat‑map …")

# -------------------------------------------------------------------
# 5.  BUILD LAYERED HEAT‑MAP (Leaflet + HeatLayer + LayerControl)
# -------------------------------------------------------------------
html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>NYC Predicted Crime Heat‑Map (2025)</title>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.3/dist/leaflet.css'>
<script src='https://unpkg.com/leaflet@1.9.3/dist/leaflet.js'></script>
<script src='https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js'></script>
<style>html,body,#map{{margin:0;height:100vh}}</style></head>
<body><div id='map'></div>
<script>
var map = L.map('map').setView([40.7128,-74.0060], 11);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

// base info panel
var info = L.control();
info.onAdd = function() {{
  this._div = L.DomUtil.create('div','info');
  this._div.innerHTML = '<b>NYC Predicted Crime (2025)</b><br>' +
                        'Toggle layers in top‑right corner.';
  return this._div; }};
info.addTo(map);

// ----- heat layers per class -----
var felonyHeat      = L.heatLayer({json.dumps(heat_lists['FELONY'])},      {{radius:10, blur:15, maxZoom:17}});
var misdemeanorHeat = L.heatLayer({json.dumps(heat_lists['MISDEMEANOR'])}, {{radius:10, blur:15, maxZoom:17}});
var violationHeat   = L.heatLayer({json.dumps(heat_lists['VIOLATION'])},   {{radius:10, blur:15, maxZoom:17}});

var overlays = {{
  'Felony': felonyHeat,
  'Misdemeanor': misdemeanorHeat,
  'Violation': violationHeat
}};
L.control.layers(null, overlays, {{collapsed:false}}).addTo(map);

// default ‑> show all three layers
felonyHeat.addTo(map);
misdemeanorHeat.addTo(map);
violationHeat.addTo(map);
</script></body></html>
"""

out_path = 'race_maps_output/nyc_predicted_crimes_2025_layers.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Layered heat‑map saved → {out_path}")
print("Done ✔︎")
