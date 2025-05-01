import pandas as pd
import geopandas as gpd
import numpy as np
import os
import folium

# Loading the race data
df = pd.read_csv('/data/outputs/NYC_Borough_Race_Populations_Blocks.csv')

# Loading the shapes of the census blocks and county borders
gdf = gpd.read_file('NYC Shapefile/nyc_census_blocks.shp')
boroughs_gdf = gpd.read_file('/NYC Shapefile/nyc_counties.shp')


# Simplifying shapes so its easier to load
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0005, preserve_topology=True)
boroughs_gdf['geometry'] = boroughs_gdf['geometry'].simplify(0.0005, preserve_topology=True)


# census shapefile + census race data
merged = gdf.merge(df, left_on='GEOIDFQ20', right_on='GEO_ID')
merged = merged[(merged['Total Pop One Race'] + merged['Two or More Races Pop']) > 0] # Only keeping populated areas

print(merged.columns)

merged.to_file("nyc_blocks_with_race_data.geojson", driver='GeoJSON')





# Creating borders and labels
borough_borders = boroughs_gdf[['geometry', 'NAME']]

borough_borders['centroid'] = borough_borders['geometry'].centroid


# Adding a borough column
merged['borough'] = np.where(merged['GEOIDFQ20'].str.contains('US36005'), 'Bronx',
                     np.where(merged['GEOIDFQ20'].str.contains('US36047'), 'Brooklyn',
                     np.where(merged['GEOIDFQ20'].str.contains('US36061'), 'Manhattan',
                     np.where(merged['GEOIDFQ20'].str.contains('US36081'), 'Queens',
                     np.where(merged['GEOIDFQ20'].str.contains('US36085'), 'Staten Island', 'Unknown')))))

# Saving the maps we'll generate
os.makedirs('race_maps_output', exist_ok=True)

# Calculate % of each race population
merged['Percent_Black'] = (merged['Black or African American Alone Pop'] / (merged['Total Pop One Race'] + merged['Two or More Races Pop'])) * 100
merged['Percent_White'] = (merged['White Alone'] / (merged['Total Pop One Race'] + merged['Two or More Races Pop'])) * 100
merged['Percent_American_and_Alaska_Native'] = (merged['American Indian and Alaska Native Alone Pop'] / (merged['Total Pop One Race'] + merged['Two or More Races Pop'])) * 100
merged['Percent_Asian'] = (merged['Asian Alone Pop'] / (merged['Total Pop One Race'] + merged['Two or More Races Pop'])) * 100
merged['Percent_Native_Hawaiian_AAPI'] = (merged['Native Hawaiian and Other Pacific Islander Alone Pop'] / (merged['Total Pop One Race'] + merged['Two or More Races Pop'])) * 100
merged['Percent_Some_Other_Race'] = (merged['Some Other Race Alone Pop'] /(merged['Total Pop One Race'] + merged['Two or More Races Pop'])) * 100
merged['Percent_Two_or_more_races'] = (merged['Two or More Races Pop'] / (merged['Total Pop One Race'] + merged['Two or More Races Pop'])) * 100



# Define racial percentage columns
race_columns = {
    'Percent_Black': ('% Black Population', 'YlOrRd'),
    'Percent_White': ('% White Population', 'Blues'),
    'Percent_Asian': ('% Asian Population', 'PuBu'),
    'Percent_American_and_Alaska_Native':('% American Indian Alaska Native Population', 'Purples'),
    'Percent_Native_Hawaiian_AAPI': ('% Native Hawaiian/AAPI Population', 'BuPu'),
    'Percent_Some_Other_Race':('% Some Other Race Population', 'Greens'),
    'Percent_Two_or_more_races':('% Two or More Races Population', 'Oranges')
}


# Loop over each race to create a clean standalone map
for race_col, (legend_label, color_scale) in race_columns.items():
    print(f"🗺️ Generating map for {legend_label}...")

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

    # Add choropleth layer
    folium.Choropleth(
        geo_data=merged,
        data=merged,
        columns=['GEO_ID', race_col],
        key_on='feature.properties.GEO_ID',
        fill_color=color_scale,
        fill_opacity=0.7,
        line_opacity=0.1,
        legend_name=legend_label,
        nan_fill_color='white'
    ).add_to(m)


    # Add borough labels using centroids
    for _, row in borough_borders.iterrows():
        lat = row['centroid'].y
        lon = row['centroid'].x
        borough_name = row['NAME']
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f"""<div style="font-size: 12pt; font-weight: bold; color: black;">{borough_name}</div>"""
            )
        ).add_to(m)

        borders_for_map = borough_borders.drop(columns='centroid')

        folium.GeoJson(
            borders_for_map,
            name='Borough Borders',
            style_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'black',
                'weight': 2.5
            }
        ).add_to(m)

    # Save each map
    filename = f"race_maps_output/nyc_{race_col.lower().replace(' ', '_')}.html"
    m.save(filename)
    print(f"Saved map: {filename}")
