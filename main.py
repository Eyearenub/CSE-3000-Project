import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for clean output

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


# RANDOM FOREST PORTION

X = df[features] 
y = df[target]

# Split data into train/test sets
# We have a fixed random seed for reproducability
# We're using 80% of the data to train our features and targets
# We're using the other 20% for testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Train Random Forest
# class_weight="balanced" because 
#   - One class (like "Felony") happens way more often than others (like "Violation"). = imbalance
#   - It makes the model pay more attention to underrepresented classes. Gives higher weight to rare classes and lower weight to frequent classes

# For the future: See what the other variations of trees produce options are: balanced_subsample (Balances ind trees), {dict} we manually set, none treats all classes equally

# n_estimator - How many decision trees to build inside random forest. More = better accuracy, beyond 1000 not worth it
#   - Can slightly change F-1 Scores
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
)


rf_model.fit(X_train, y_train)

# Cross-Validation (5-fold)
# - Splits the training data in 5 diff parts 
# Trains and tests the model 5 diff times
# Calculates the F1 Macro for each split
# Use 4 models for training and the last for testing
 
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1_macro')

print("Cross-Validation F1 Macro Scores:", cv_scores)
print("Average CV F1 Macro Score:", np.mean(cv_scores))

# Evaluate on Test Set
y_pred = rf_model.predict(X_test)


# F1 Score tells 
#   - How many felonies were caught correctly
#   - How precisely it labels felonies

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))


# Confusion matrix
#   - How many times you predicted Felony correctly
#   - How many times you predicted a misdemeanor when it was really a felony

print("\nConfusion Matrix on Test Set:")
print(confusion_matrix(y_test, y_pred))


importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# Export Test Results to CSV
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

results_df.to_csv('crime_predictions_results.csv', index=False)

print("Predictions exported to crime_predictions_results.csv")