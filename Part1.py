import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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


params = {
    "$limit": 1400000,
    "$where": "cmplnt_fr_dt between '2024-01-01T00:00:00' and '2024-12-31T23:59:59'"
}


response = requests.get(endpoint, params=params)
data = response.json()


df = pd.DataFrame(data)

initial1 = len(df)



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
initial2 = len(df)

# Creating new columns that better represent patterns in the data
df['day_of_week'] = df['cmplnt_fr_dt'].dt.dayofweek  # Converting days of week to 0 (Monday) - 6 (Sunday)
df['month'] = df['cmplnt_fr_dt'].dt.month # April = 4
df['hour_of_day'] = pd.to_datetime(df['cmplnt_fr_tm'], format='%H:%M:%S', errors='coerce').dt.hour

# "BROOKLYN" -> 0, "MANHATTAN" -> 1, etc.
df['borough_code'] = df['boro_nm'].astype('category').cat.codes

# 'COMPLETED' becomes 1, 'ATTEMPTED' becomes 0
df['is_completed'] = (df['crm_atpt_cptd_cd'] == 'COMPLETED').astype(int)

# Select features and target
features = ['day_of_week', 'month', 'hour_of_day', 'borough_code', 'is_completed']
target = 'law_cat_cd'  # Crime severity: FELONY, MISDEMEANOR, VIOLATION

# If a column doesn't have any of the target values
df = df.dropna(subset=[target])
initial3 = len(df)







# TRAINING ALGORITHMS

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

# Models to compare
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'SVM': SVC(class_weight='balanced'),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42)
}


"""
Classification report:

Precision: When the model predicts this class, how often is it correct?
The higher the better

Recall: Out of all the actual crimes of this class, how many did the model correctly find?
If there were 12 true felonies, and the model predicted 9 of them correctly 8/12 = 0.67
The greater it is the fewer the false negatives

F1 Score: The balance betwen precision and recall. High only when both preciison and recall are high

Support: How many samples were there of this class in the test set?

Accuracy - How many total predictions were correct across all classes?

Macro Avg - Averae of precision, recall, and F1 for each class treating all classes equally. 

Weighted Average - Classes with more support (more samples) get more influence
    - Gives more realistic performance measure when your data is imbalanced
"""

# Store all reports and macro F1s
all_reports = []
macro_f1_scores = {}

# Loop through models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save macro F1 score
    macro_f1 = report['macro avg']['f1-score']
    macro_f1_scores[name] = macro_f1

    # Format classification report
    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df = report_df.rename(columns={'index': 'Class'})
    report_df.insert(0, 'Model', name)
    all_reports.append(report_df)

# Combine all reports into one DataFrame
final_report_df = pd.concat(all_reports, ignore_index=True)

# Export all classification reports to CSV
final_report_df.to_csv('data/outputs/multi_model_classification_reports.csv', index=False)




# BACK TESTING - PREDICTING THE NEXT MONTH

# Sorting and parsing the data
df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')
df = df.sort_values('cmplnt_fr_dt')
df['year_month'] = df['cmplnt_fr_dt'].dt.to_period('M')

# Filter for only 2024
df = df[df['cmplnt_fr_dt'].dt.year == 2024]

# Get unique months
months = sorted(df['year_month'].unique())

# Prepare backtest tracking
backtest_results = []

for i in range(1, len(months)):  # start at month 1 (second month)
    train_months = months[:i]
    test_month = months[i]

    train_df = df[df['year_month'].isin(train_months)]
    test_df = df[df['year_month'] == test_month]

    # Number of features * 10, determining if we have enough test data
    if len(test_df) < 30:
        continue

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    print(f"\nBacktesting on {test_month}...")

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            macro_f1 = report['macro avg']['f1-score']

            backtest_results.append({
                'Month': str(test_month),
                'Model': model_name,
                'F1_Macro': macro_f1
            })
            print(f"{model_name}: F1_macro = {macro_f1:.3f}")

        except Exception as e:
            print(f"{model_name} failed on {test_month}: {e}")

# Create DataFrame and export
backtest_df = pd.DataFrame(backtest_results)
backtest_df.to_csv('data/outputs/backtesting_model_performance.csv', index=False)
print("\n Backtesting results saved to backtesting_model_performance.csv")


# Pivot for plotting
pivot = backtest_df.pivot(index='Month', columns='Model', values='F1_Macro')
pivot = pivot.sort_index()

plt.figure(figsize=(10, 6))
for model in pivot.columns:
    plt.plot(pivot.index.astype(str), pivot[model], marker='o', label=model)

plt.title('Backtesting: Monthly F1 Macro Score per Model (2024)')
plt.xlabel('Month')
plt.ylabel('F1 Macro Score')
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Graphs/backtesting_f1_scores_over_time.png')
plt.show()









# CREATING GRAPH
class_labels = ['FELONY', 'MISDEMEANOR', 'VIOLATION']
grouped = final_report_df[final_report_df['Class'].isin(class_labels)]

# Pivot the data to get F1 scores per class per model
pivot = grouped.pivot(index='Class', columns='Model', values='f1-score')

# Plot grouped bar chart
labels = pivot.index.tolist()
models_list = pivot.columns.tolist()
x = np.arange(len(labels))  # group positions
width = 0.15  # width of each bar

plt.figure(figsize=(10, 6))
for i, model in enumerate(models_list):
    plt.bar(x + i * width, pivot[model], width, label=model)

plt.ylabel('F1 Score')
plt.title('Per-Class F1 Scores by Model')
plt.xticks(x + width * (len(models_list)-1)/2, labels)
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Graphs/per_class_f1_by_model.png')
plt.show()


