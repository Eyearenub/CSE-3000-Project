import geopandas as gpd


# gdf = gpd.read_file("/Users/joelduah/Downloads/tl_2024_36_tabblock20/tl_2024_36_tabblock20.shp")

# print(gdf.columns)
# print(gdf.head())

# nyc_fips = ['005', '047', '061', '081', '085']
# nyc_blocks = gdf[gdf['COUNTYFP20'].isin(nyc_fips)]

# # Save the reduced shapefile
# nyc_blocks.to_file("nyc_census_blocks.shp")

gdf2 = gpd.read_file("/Users/joelduah/Downloads/tl_2024_36_cousub/tl_2024_36_cousub.shp")

print(gdf2.columns)
print(gdf2.head())

nyc_fips = ['005', '047', '061', '081', '085']
nyc_blocks = gdf2[gdf2['COUNTYFP'].isin(nyc_fips)]

nyc_blocks.to_file("nyc_counties.shp")

