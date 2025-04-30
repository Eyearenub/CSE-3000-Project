import pandas as pd
"""
Need to grab the New York City Borough Population data for only New York City Boroughs

We only want New York City Boroughs
Bronx - Bronx County - US36005
Brooklyn - Kings County - US36047
Manhattan - New York County - US36061
Queens - Queens County - US36081
Staten Island - Richmond County - US36085


"""
# Load the CSV into a DataFrame
df = pd.read_csv('data/ACSDP1Y2023/ACSDP1Y2023.DP05-Data.csv')

allowed_counties = ['Bronx County', 'Kings County', 'New York County', 'Queens County','Richmond County']


columns_keeping = ['GEO_ID', 'NAME', 'DP05_0001E', 'DP05_0021E', 'DP05_0026E', 
                   'DP05_0027E','DP05_0075E', 'DP05_0076E','DP05_0081E','DP05_0082E',
                   'DP05_0083E','DP05_0084E','DP05_0085E','DP05_0086E','DP05_0087E', 'DP05_0088E']

# Selecting and renaming columns
df_clean = df[columns_keeping].rename(columns={
    'GEO_ID':'GEO_ID',
    'NAME': 'County Name',
    'DP05_0001E': 'Total Population',
    'DP05_0021E': '18 Up Pop',
    'DP05_0026E':'18 Up Male Pop',
    'DP05_0027E':'18 Up Female Pop',
    'DP05_0075E':'HL pop',
    'DP05_0076E':'HL any race Pop',
    'DP05_0081E':'Not HL Pop',
    'DP05_0082E':'Not HL White',
    'DP05_0083E': 'Not HL Black',
    'DP05_0084E':'Not HL Ame Ind',
    'DP05_0085E':'Not HL Asian',
    'DP05_0086E':'Not HL Haw PI',
    'DP05_0087E':'Not HL Other Race',
    'DP05_0088E':'Not HL Two or more race'
})


# Keep rows where NAME starts with allowed counties
df_filtered = df_clean[df_clean['County Name'].str.startswith(tuple(allowed_counties))]

df_filtered.to_csv("NYC_Borough_Race_Populations.csv", index=False)
