import pandas as pd
import geopandas as gpd
import json
import os

# Load data
gdf = gpd.read_file("nyc_blocks_with_race_data.geojson")
df = pd.read_csv("data/outputs/predicted_felonies_2025.csv")

# Clean coordinates
df = df[df['latitude'].notna() & df['longitude'].notna()]
heat_data = df[['latitude', 'longitude']].values.tolist()

# Simplify GeoDataFrame for performance
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0025, preserve_topology=True)

# Select and prepare demographic fields
gdf = gdf[[
    'geometry',
    'Percent_Black', 'Percent_White', 'Percent_Asian',
    'Percent_American_and_Alaska_Native',
    'Percent_Native_Hawaiian_AAPI',
    'Percent_Some_Other_Race',
    'Percent_Two_or_more_races'
]]
gdf_json = gdf.to_json()
heat_data_json = json.dumps(heat_data)


# This is to generate the map
# Escape curly braces in JS (double them for Python f-string)
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>NYC Heatmap + Demographics</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    
    <style>
        #map {{
            height: 100vh;
        }}
        .info {{
            padding: 8px;
            background: white;
            background: rgba(255,255,255,0.9);
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            border-radius: 5px;
            font: 14px Arial, sans-serif;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([40.7128, -74.0060], 11);

        L.tileLayer('https://{{{{s}}}}.tile.openstreetmap.org/{{{{z}}}}/{{{{x}}}}/{{{{y}}}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        var heat = L.heatLayer({heat_data_json}, {{
            radius: 10,
            blur: 15,
            maxZoom: 17,
            gradient: {{0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}}
        }}).addTo(map);

        function onEachFeature(feature, layer) {{
            let props = feature.properties;
            let content = `
                <b>Demographics:</b><br>
                % Black: ${{props.Percent_Black.toFixed(1)}}%<br>
                % White: ${{props.Percent_White.toFixed(1)}}%<br>
                % Asian: ${{props.Percent_Asian.toFixed(1)}}%<br>
                % Am. Indian/Alaska Native: ${{props.Percent_American_and_Alaska_Native.toFixed(1)}}%<br>
                % Native Hawaiian/AAPI: ${{props.Percent_Native_Hawaiian_AAPI.toFixed(1)}}%<br>
                % Some Other Race: ${{props.Percent_Some_Other_Race.toFixed(1)}}%<br>
                % Two or More Races: ${{props.Percent_Two_or_more_races.toFixed(1)}}%<br>
            `;
            layer.bindPopup(content);
        }}

        var geojson = L.geoJson({json.dumps(json.loads(gdf_json))}, {{
            style: function (feature) {{
                return {{
                    color: "black",
                    weight: 0.3,
                    fillOpacity: 0
                }};
            }},
            onEachFeature: onEachFeature
        }}).addTo(map);
    </script>
</body>
</html>
"""


# Save the file
output_path = "race_maps_output/leaflet_heatmap_demographics.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Map saved to {output_path}")
