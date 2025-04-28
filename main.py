import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

"""
Build a Random Forest model using NYPD crime data to predict crime occurrence or type.
Evaluate accuracy with precision, recall, F1-score, and confusion matrix.
Analyze for bias (racial, socioeconomic bias).
Reflect ethically on potential risks of predictive policing.
"""

# Just grabbing the data from the NYCPD website
endpoint = "https://data.cityofnewyork.us/resource/5uac-w243.json"

response = requests.get(endpoint)
data = response.json()

df = pd.DataFrame(data)




# Cleaning Data / Feature engineering 

"""
Useful columns possibly
BORO_NM
CMPLNT_FR_DT
CMPLNT_FR_TM
CMPLNT_TO_DT
CMPLNT_TO_TM
CRM_ATPT_CPTD_CD
HADEVELOPT
jurisdiction_code
JURIS_DESC
LAW_CAT_CD
LOC_OF_OCCUR_DESC
OFNS_DESC
pd_desc
perm_typ_desc
rpt_dt
susp_age_group
susp_race
susp_sex
vic_age
vic_race
vic_sex
Latitude
Longitude
geocoded_column
"""

# Create useful features:
# - Day of week from date
# - Hour of day
# - Borough
# - Crime type
# Convert dates

df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')


# Drop rows with invalid dates
df = df.dropna(subset=['cmplnt_fr_dt'])


# Convert dates
df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['cmplnt_fr_dt'])

# Creating new columns that better represent patterns in the data

df['day_of_week'] = df['cmplnt_fr_dt'].dt.dayofweek  # Converting days of week to 0 (Monday) - 6 (Sunday)
df['month'] = df['cmplnt_fr_dt'].dt.month # April = 4
df['hour_of_day'] = pd.to_datetime(df['cmplnt_fr_tm'], format='%H:%M:%S', errors='coerce').dt.hour

# "BROOKLYN" -> 0, "MANHATTAN" -> 1, etc.
df['borough_code'] = df['boro_nm'].astype('category').cat.codes

# 'COMPLETED' becomes 1, 'ATTEMPTED' becomes 0
df['is_completed'] = (df['crm_atpt_cptd_cd'] == 'COMPLETED').astype(int)

# 3. Select features and target
features = ['day_of_week', 'month', 'hour_of_day', 'borough_code', 'is_completed']
target = 'law_cat_cd'  # Crime severity: FELONY, MISDEMEANOR, VIOLATION

# If a column doesn't have any of the target values
df = df.dropna(subset=[target])