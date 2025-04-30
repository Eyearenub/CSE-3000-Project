import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# Just grabbing the data from the NYCPD website
endpoint = "https://data.cityofnewyork.us/resource/5uac-w243.json"


params = {
    "$limit": 1400000,
    "$where": "cmplnt_fr_dt between '2024-01-01T00:00:00' and '2024-12-31T23:59:59'"
}


response = requests.get(endpoint, params=params)
data = response.json()


df = pd.DataFrame(data)

# 1. Make sure your data is sorted by date
df = df.sort_values(by='cmplnt_fr_dt')
df = df.dropna(subset=['law_cat_cd'])  # make sure target column is valid

# 2. Extract year-month for slicing
df['year_month'] = df['cmplnt_fr_dt'].dt.to_period('M')

# 3. Define features and target
features = ['day_of_week', 'month', 'hour_of_day', 'borough_code', 'is_completed']
target = 'law_cat_cd'

# 4. Prepare storage
f1_scores = []
report_snapshots = []

# 5. Loop through each month from 2nd month onward
months = sorted(df['year_month'].unique())

for i in range(1, len(months)):
    train_months = months[:i]
    test_month = months[i]

    # Split train/test based on month
    train_df = df[df['year_month'].isin(train_months)]
    test_df = df[df['year_month'] == test_month]

    # Skip if test month has too few samples
    if len(test_df) < 500:
        continue

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Train and test Random Forest
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Collect macro F1
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report['macro avg']['f1-score']
    f1_scores.append({'Test_Month': str(test_month), 'F1_Macro': f1})

    # Optional: save the whole report too
    snapshot = pd.DataFrame(report).transpose().reset_index()
    snapshot['Test_Month'] = str(test_month)
    report_snapshots.append(snapshot)

    print(f"{test_month}: F1_macro = {f1:.4f}")

# 6. Export F1 scores
f1_df = pd.DataFrame(f1_scores)
f1_df.to_csv('backtesting_f1_scores.csv', index=False)

# 7. Export all classification reports per month
full_report_df = pd.concat(report_snapshots, ignore_index=True)
full_report_df.to_csv('backtesting_full_classification_reports.csv', index=False)

print("✅ Backtesting complete. Reports saved.")
