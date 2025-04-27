import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


endpoint = "https://data.cityofnewyork.us/resource/5uac-w243.json"

response = requests.get(endpoint)
data = response.json()

df = pd.DataFrame(data)


"""
Useful columns
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


# Convert dates
df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['cmplnt_fr_dt'])


df['day_of_week'] = df['cmplnt_fr_dt'].dt.dayofweek  # Monday=0
df['month'] = df['cmplnt_fr_dt'].dt.month
df['hour_of_day'] = pd.to_datetime(df['cmplnt_fr_tm'], format='%H:%M:%S', errors='coerce').dt.hour

print(df['month'])