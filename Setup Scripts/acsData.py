import pandas as pd

"""
Script to clean ACS data 

We are using the 2020 American Community Survey Data to find how many people of each race lives
in a census block so that we can make a detailed choropleth layer showing where each race is most densely populated.


Need to grab the New York City Borough Population data for only New York City Boroughs

We only want New York City Boroughs
Bronx - Bronx County - US36005
Brooklyn - Kings County - US36047
Manhattan - New York County - US36061
Queens - Queens County - US36081
Staten Island - Richmond County - US36085
"""


# Counties to keep (NYC Boroughs)
allowed_counties = ['Bronx County', 'Kings County', 'New York County', 'Queens County','Richmond County']

# Loading the full race demographic file
# NOT actual file path file is too large to be part of the github repo
df = pd.read_csv('/ACS_RaceInBlock/DECENNIALPL2020.P1-Data.csv')

# Columns to keep
columns_keeping = ['GEO_ID','NAME','P1_002N','P1_003N','P1_004N','P1_005N','P1_006N','P1_007N','P1_008N','P1_009N']

# Rename columns for clarity
df_clean = df[columns_keeping].rename(columns={
    'GEO_ID': 'GEO_ID',
    'NAME': 'County Name',
    'P1_002N': 'Total Pop One Race',
    'P1_003N': 'White Alone',
    'P1_004N': 'Black or African American Alone Pop',
    'P1_005N': 'American Indian and Alaska Native Alone Pop',
    'P1_006N': 'Asian Alone Pop',
    'P1_007N': 'Native Hawaiian and Other Pacific Islander Alone Pop',
    'P1_008N': 'Some Other Race Alone Pop',
    'P1_009N': 'Two or More Races Pop'
})

# Filter rows that contain any of the allowed NYC counties
df_filtered = df_clean[df_clean['County Name'].str.contains('|'.join(allowed_counties), regex=True)]

# Save the cleaned and filtered file
df_filtered.to_csv("/data/outputs/NYC_Borough_Race_Populations_Blocks.csv", index=False)