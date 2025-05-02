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
    "$where": "(cmplnt_fr_dt between '2023-01-01T00:00:00' "
              "and '2025-12-31T23:59:59')"
}
df = pd.DataFrame(requests.get(endpoint, params=params).json())
print(f"Rows downloaded: {len(df):,}")

# -------------------------------------------------------------------
# 2.  CLEAN / FEATURE ENGINEER
# -------------------------------------------------------------------
df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')
df['cmplnt_fr_tm'] = pd.to_datetime(df['cmplnt_fr_tm'],
                                    format='%H:%M:%S', errors='coerce')
df = df.dropna(subset=['cmplnt_fr_dt', 'law_cat_cd'])

df['day_of_week']  = df['cmplnt_fr_dt'].dt.dayofweek
df['month']        = df['cmplnt_fr_dt'].dt.month
df['hour_of_day']  = df['cmplnt_fr_tm'].dt.hour
df['borough_code'] = df['boro_nm'].astype('category').cat.codes
df['is_completed'] = (df['crm_atpt_cptd_cd'] == 'COMPLETED').astype(int)
df['year']         = df['cmplnt_fr_dt'].dt.year

features = ['day_of_week','month','hour_of_day',
            'borough_code','is_completed']
target   = 'law_cat_cd'

train_df  = df[df['year'] == 2024]
score_df  = df[df['year'] == 2025]

for d in (train_df, score_df):
    d['latitude']  = pd.to_numeric(d['latitude'],  errors='coerce')
    d['longitude'] = pd.to_numeric(d['longitude'], errors='coerce')
    d.dropna(subset=['latitude','longitude'], inplace=True)

print(f"2024 train rows: {len(train_df):,}")
print(f"2025 score rows: {len(score_df):,}")

# -------------------------------------------------------------------
# 3.  TRAIN RANDOM‑FOREST
# -------------------------------------------------------------------
rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
     )
rf.fit(train_df[features], train_df[target])
print("Model fitted.")

# -------------------------------------------------------------------
# 4.  PREDICT 2025 (ALL CLASSES)
# -------------------------------------------------------------------
score_df['predicted_label'] = rf.predict(score_df[features])
score_df.to_csv('data/outputs/predicted_crimes_2025.csv', index=False)
print("Saved → data/outputs/predicted_crimes_2025.csv")

# Optional: per‑class slices (handy for downstream analyses / maps)
for cat in ['FELONY','MISDEMEANOR','VIOLATION']:
    out = score_df[score_df['predicted_label'] == cat]
    out.to_csv(f'data/outputs/predicted_{cat.lower()}s_2025.csv',
               index=False)

# -------------------------------------------------------------------
# 5.  BUILD ONE HEAT‑MAP WITH *ALL* PREDICTIONS
#     (size / colour encodes nothing fancy: just presence)
# -------------------------------------------------------------------
heat_points = [[r['latitude'], r['longitude']]
               for _, r in score_df.iterrows()
               if np.isfinite(r['latitude']) and np.isfinite(r['longitude'])]
print(f"Heat‑map points: {len(heat_points):,}")

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>NYC Predicted Crime Heat‑Map (2025)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>html,body,#map{{margin:0;height:100vh}}</style></head>
<body><div id="map"></div>
<script>
var map=L.map('map').setView([40.7128,-74.0060],11);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution:'&copy; OpenStreetMap contributors'
}}).addTo(map);

var info=L.control();
info.onAdd=function(m){{this._div=L.DomUtil.create('div','info');
this._div.innerHTML='<b>Predicted Crime Heat‑Map (2025)</b><br>'
                   +'Total predictions: {len(score_df):,}';return this._div}};
info.addTo(map);

L.heatLayer({json.dumps(heat_points)}, {{
  radius:10, blur:15, maxZoom:17,
  gradient:{{0.2:'blue',0.4:'lime',0.6:'yellow',0.8:'orange',1.0:'red'}}
}}).addTo(map);
</script></body></html>
"""
with open('race_maps_output/nyc_predicted_crimes_2025.html','w',
          encoding='utf-8') as f:
    f.write(html)
print("Heat‑map written → race_maps_output/nyc_predicted_crimes_2025.html")
print("All done ✔︎")
