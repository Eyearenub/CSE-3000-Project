"""
ETHICAL NOTICE:
This script is for academic research purposes only. Predictive policing algorithms can reinforce existing biases
in criminal justice data and potentially lead to discriminatory outcomes. Any operational deployment must undergo
rigorous fairness auditing, bias testing, and regular review. Results should always be treated as supplementary
information rather than definitive predictions, and human judgment must remain central to decision-making processes.
The existence of a correlation does not imply causation, and predictions may reflect historical policing patterns
rather than actual crime distribution.

Requirements:
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- matplotlib==3.7.2
- seaborn==0.12.2
- requests==2.31.0
- joblib==1.3.1
"""

import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import joblib
import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings

# Configuration parameters
CONFIG = {
    'api_limit': 1400000,
    'start_date': '2024-01-01T00:00:00',
    'end_date': '2024-12-31T23:59:59',
    'test_size': 0.2,
    'random_state': 42,
    'output_dir': 'data/outputs',
    'plots_dir': 'plots',
    'binary_plots_dir': 'plots/binary',
    'multiclass_plots_dir': 'plots/multiclass',
    'models_dir': 'models'
}

# Ensure directories exist
for directory in [CONFIG['output_dir'], CONFIG['plots_dir'], CONFIG['binary_plots_dir'], 
                 CONFIG['multiclass_plots_dir'], CONFIG['models_dir']]:
    os.makedirs(directory, exist_ok=True)

warnings.filterwarnings("ignore")  # Suppress warnings for clean output


def fetch_crime_data():
    """Fetch crime data from the NYPD Open Data API"""
    print("Fetching NYPD crime data...")
    
    endpoint = "https://data.cityofnewyork.us/resource/5uac-w243.json"
    params = {
        "$limit": CONFIG['api_limit'],
        "$where": f"cmplnt_fr_dt between '{CONFIG['start_date']}' and '{CONFIG['end_date']}'"
    }
    
    response = requests.get(endpoint, params=params)
    data = response.json()
    
    df = pd.DataFrame(data)
    print(f"Retrieved {len(df)} records")
    return df


def clean_and_preprocess(df):
    """Clean and preprocess the crime data"""
    print("Cleaning and preprocessing data...")
    
    initial_length = len(df)
    print(f"Initial dataset size: {initial_length} records")
    
    # Convert dates and times
    df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['cmplnt_fr_dt'])
    print(f"After removing invalid dates: {len(df)} records")
    
    # Feature engineering
    df['day_of_week'] = df['cmplnt_fr_dt'].dt.dayofweek  # 0 (Monday) - 6 (Sunday)
    df['month'] = df['cmplnt_fr_dt'].dt.month
    df['hour_of_day'] = pd.to_datetime(df['cmplnt_fr_tm'], format='%H:%M:%S', errors='coerce').dt.hour
    
    # Location features
    df['borough_code'] = df['boro_nm'].astype('category').cat.codes
    
    # Crime features
    df['is_completed'] = (df['crm_atpt_cptd_cd'] == 'COMPLETED').astype(int)
    df['sus_race_code'] = df['susp_race'].astype('category').cat.codes
    df['vic_race_code'] = df['vic_race'].astype('category').cat.codes
    
    # Ensure latitude and longitude are numeric
    for col in ['latitude', 'longitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create mappings for categorical variables for later use
    category_mappings = {
        'borough_code': dict(enumerate(df['boro_nm'].astype('category').cat.categories)),
        'sus_race_code': dict(enumerate(df['susp_race'].astype('category').cat.categories)),
        'vic_race_code': dict(enumerate(df['vic_race'].astype('category').cat.categories))
    }
    
    # Drop rows with missing target
    df = df.dropna(subset=['law_cat_cd'])
    print(f"Final dataset size: {len(df)} records")
    
    return df, category_mappings


def generate_negative_samples(df, num_samples=None):
    """
    Generate negative samples (times/places with no recorded crime)
    
    Strategy:
    1. Create a grid across NYC boroughs and time periods
    2. For each grid point, check if a crime occurred
    3. If no crime occurred, mark as a negative sample (crime_occurred = 0)
    """
    print("Generating negative samples...")
    
    if num_samples is None:
        # Generate as many negative samples as there are positive samples (balanced)
        num_samples = len(df)
    
    # Get the range of dates, hours, and boroughs from the original data
    min_date = df['cmplnt_fr_dt'].min()
    max_date = df['cmplnt_fr_dt'].max()
    
    # Get all dates in string format for easier handling
    all_dates = []
    curr_date = min_date
    while curr_date <= max_date:
        all_dates.append(curr_date.strftime('%Y-%m-%d'))
        curr_date += pd.Timedelta(days=1)
    
    # Create a set of all existing date-hour-borough combinations using string representation
    existing_combinations = set()
    for _, row in df.iterrows():
        if pd.notna(row['hour_of_day']) and pd.notna(row['borough_code']):
            date_str = row['cmplnt_fr_dt'].strftime('%Y-%m-%d')
            existing_combinations.add((
                date_str,
                int(row['hour_of_day']),
                int(row['borough_code'])
            ))
    
    # Generate negative samples
    negative_samples = []
    attempts = 0
    max_attempts = num_samples * 10  # Limit attempts to avoid infinite loop
    
    while len(negative_samples) < num_samples and attempts < max_attempts:
        # Randomly select date, hour, and borough
        random_date_str = np.random.choice(all_dates)
        random_date = pd.to_datetime(random_date_str)
        random_hour = np.random.randint(0, 24)
        random_borough = np.random.randint(0, df['borough_code'].max() + 1)
        
        # Check if this combination exists in the original data
        if (random_date_str, random_hour, random_borough) not in existing_combinations:
            # This is a time/place with no recorded crime
            day_of_week = random_date.dayofweek
            month = random_date.month
            
            # For demographic features, use the distribution from the original data
            random_sus_race = np.random.choice(df['sus_race_code'].dropna())
            random_vic_race = np.random.choice(df['vic_race_code'].dropna())
            
            negative_samples.append({
                'cmplnt_fr_dt': pd.Timestamp(random_date),
                'day_of_week': day_of_week,
                'month': month,
                'hour_of_day': random_hour,
                'borough_code': random_borough,
                'sus_race_code': random_sus_race,
                'vic_race_code': random_vic_race,
                'is_completed': 0,  # Not relevant for negative samples
                'law_cat_cd': 'NONE',  # No crime type
                'crime_occurred': 0  # Binary target is 0 for negative samples
            })
            
            # Add to existing combinations to avoid duplicates
            existing_combinations.add((random_date, random_hour, random_borough))
        
        attempts += 1
    
    # Create a DataFrame from the negative samples
    negative_df = pd.DataFrame(negative_samples)
    print(f"Generated {len(negative_df)} negative samples")
    
    return negative_df


def prepare_features_and_targets(df, negative_df=None):
    """Prepare features and targets for both classification tasks"""
    print("Preparing features and targets...")
    
    # Add binary target to original data
    df['crime_occurred'] = 1
    
    # Combine with negative samples if provided
    if negative_df is not None:
        df = pd.concat([df, negative_df], ignore_index=True)
    
    # Define features and targets
    features = ['day_of_week', 'month', 'hour_of_day', 'sus_race_code', 
                'vic_race_code', 'borough_code', 'is_completed']
    
    multiclass_target = 'law_cat_cd'  # Crime severity: FELONY, MISDEMEANOR, VIOLATION
    binary_target = 'crime_occurred'  # Whether a crime occurred: 1 (yes) or 0 (no)
    
    # Select only the rows with valid feature values for all features
    df = df.dropna(subset=features)
    
    # Split for multiclass classification (using only positive samples)
    multiclass_df = df[df[binary_target] == 1].copy()
    
    X_multiclass = multiclass_df[features]
    y_multiclass = multiclass_df[multiclass_target]
    
    # Split for binary classification (using all samples)
    X_binary = df[features]
    y_binary = df[binary_target]
    
    # Train-test split for both tasks
    X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
        X_multiclass, y_multiclass, test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'], stratify=y_multiclass
    )
    
    X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(
        X_binary, y_binary, test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'], stratify=y_binary
    )
    
    print(f"Multiclass task: {len(X_multi_train)} train samples, {len(X_multi_test)} test samples")
    print(f"Binary task: {len(X_bin_train)} train samples, {len(X_bin_test)} test samples")
    
    return {
        'multiclass': (X_multi_train, X_multi_test, y_multi_train, y_multi_test),
        'binary': (X_bin_train, X_bin_test, y_bin_train, y_bin_test),
        'features': features
    }


def create_and_train_models():
    """Define models for both classification tasks"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=CONFIG['random_state']),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=CONFIG['random_state']),
        'SVM': SVC(class_weight='balanced', probability=True, random_state=CONFIG['random_state']),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=CONFIG['random_state'])
    }
    return models


def evaluate_and_save_models(models, data_splits):
    """Train, evaluate, and save all models for both tasks"""
    
    # Extract data
    X_multi_train, X_multi_test, y_multi_train, y_multi_test = data_splits['multiclass']
    X_bin_train, X_bin_test, y_bin_train, y_bin_test = data_splits['binary']
    feature_names = data_splits['features']
    
    # Results storage
    multiclass_reports = []
    binary_reports = []
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Multiclass training and evaluation
        print(f"  Training {name} for multiclass task...")
        model.fit(X_multi_train, y_multi_train)
        y_multi_pred = model.predict(X_multi_test)
        
        # Save model
        joblib.dump(model, f"{CONFIG['models_dir']}/{name.lower().replace(' ', '_')}_multiclass.pkl")
        
        # Generate classification report
        multi_report = classification_report(y_multi_test, y_multi_pred, output_dict=True)
        
        # Format and save report
        multi_report_df = pd.DataFrame(multi_report).transpose().reset_index()
        multi_report_df = multi_report_df.rename(columns={'index': 'Class'})
        multi_report_df.insert(0, 'Model', name)
        multiclass_reports.append(multi_report_df)
        
        # Save individual model report
        multi_report_df.to_csv(f"{CONFIG['output_dir']}/metrics_{name.lower().replace(' ', '_')}_multiclass.csv", index=False)
        
        # Confusion matrix
        cm = confusion_matrix(y_multi_test, y_multi_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sorted(y_multi_test.unique()), 
                   yticklabels=sorted(y_multi_test.unique()))
        plt.title(f'{name} Confusion Matrix - Multiclass')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{CONFIG['multiclass_plots_dir']}/confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plt.close()
        
        # Binary training and evaluation
        print(f"  Training {name} for binary task...")
        
        # Create a fresh instance of the model for binary classification
        binary_model = type(model)(**model.get_params())
        binary_model.fit(X_bin_train, y_bin_train)
        y_bin_pred = binary_model.predict(X_bin_test)
        
        # Save model
        joblib.dump(binary_model, f"{CONFIG['models_dir']}/{name.lower().replace(' ', '_')}_binary.pkl")
        
        # Generate classification report
        bin_report = classification_report(y_bin_test, y_bin_pred, output_dict=True)
        
        # Format and save report
        bin_report_df = pd.DataFrame(bin_report).transpose().reset_index()
        bin_report_df = bin_report_df.rename(columns={'index': 'Class'})
        bin_report_df.insert(0, 'Model', name)
        binary_reports.append(bin_report_df)
        
        # Save individual model report
        bin_report_df.to_csv(f"{CONFIG['output_dir']}/metrics_{name.lower().replace(' ', '_')}_binary.csv", index=False)
        
        # Binary confusion matrix
        cm = confusion_matrix(y_bin_test, y_bin_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Crime', 'Crime'], 
                   yticklabels=['No Crime', 'Crime'])
        plt.title(f'{name} Confusion Matrix - Binary')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{CONFIG['binary_plots_dir']}/confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plt.close()
        
        # ROC curve (only for binary classification)
        if hasattr(binary_model, "predict_proba"):
            y_score = binary_model.predict_proba(X_bin_test)[:, 1]
        else:  # For SVM without probability=True
            y_score = binary_model.decision_function(X_bin_test)
        
        fpr, tpr, _ = roc_curve(y_bin_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve - Binary Classification')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{CONFIG['binary_plots_dir']}/roc_curve_{name.lower().replace(' ', '_')}.png")
        plt.close()
        
        # Feature importance (for Random Forest only)
        if name == 'Random Forest':
            # Multiclass
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importances - Multiclass Classification')
            plt.bar(range(len(feature_names)), importances[indices], align='center')
            plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f"{CONFIG['multiclass_plots_dir']}/feature_importance_random_forest.png")
            plt.close()
            
            # Binary
            importances = binary_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importances - Binary Classification')
            plt.bar(range(len(feature_names)), importances[indices], align='center')
            plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f"{CONFIG['binary_plots_dir']}/feature_importance_random_forest.png")
            plt.close()
    
    # Combine all reports
    final_multi_report = pd.concat(multiclass_reports, ignore_index=True)
    final_multi_report.to_csv(f"{CONFIG['output_dir']}/multi_model_classification_reports.csv", index=False)
    
    final_bin_report = pd.concat(binary_reports, ignore_index=True)
    final_bin_report.to_csv(f"{CONFIG['output_dir']}/binary_model_classification_reports.csv", index=False)
    
    print("\nAll model evaluations complete and saved.")


def interactive_inference(category_mappings):
    """
    Interactive inference utility using the trained Random Forest binary model
    """
    print("\n" + "="*50)
    print("Interactive Crime Prediction Tool")
    print("="*50)
    
    # Load the Random Forest binary model
    try:
        model = joblib.load(f"{CONFIG['models_dir']}/random_forest_binary.pkl")
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        return
    
    # Get user input
    print("\nPlease provide the following information:")
    
    # Date input
    while True:
        date_str = input("Date (YYYY-MM-DD): ")
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
    
    # Time input
    while True:
        time_str = input("Time (HH:MM): ")
        try:
            time_obj = datetime.datetime.strptime(time_str, "%H:%M").time()
            break
        except ValueError:
            print("Invalid time format. Please use HH:MM.")
    
    # Borough input
    print("\nAvailable boroughs:")
    for code, borough in category_mappings['borough_code'].items():
        print(f"{code}: {borough}")
    
    while True:
        try:
            borough_code = int(input("Borough (enter code): "))
            if borough_code in category_mappings['borough_code']:
                break
            else:
                print("Invalid borough code. Please choose from the list above.")
        except ValueError:
            print("Please enter a number.")
    
    # Demographic group input
    print("\nAvailable demographic groups (suspect):")
    for code, race in category_mappings['sus_race_code'].items():
        print(f"{code}: {race}")
    
    while True:
        try:
            sus_race_code = int(input("Suspect demographic group (enter code): "))
            if sus_race_code in category_mappings['sus_race_code']:
                break
            else:
                print("Invalid demographic code. Please choose from the list above.")
        except ValueError:
            print("Please enter a number.")
    
    print("\nAvailable demographic groups (victim):")
    for code, race in category_mappings['vic_race_code'].items():
        print(f"{code}: {race}")
    
    while True:
        try:
            vic_race_code = int(input("Victim demographic group (enter code): "))
            if vic_race_code in category_mappings['vic_race_code']:
                break
            else:
                print("Invalid demographic code. Please choose from the list above.")
        except ValueError:
            print("Please enter a number.")
    
    # Preprocess the input
    day_of_week = date_obj.weekday()
    month = date_obj.month
    hour_of_day = time_obj.hour
    is_completed = 1  # Assume completed for prediction purposes
    
    # Create a feature vector
    features = ['day_of_week', 'month', 'hour_of_day', 'sus_race_code', 
                'vic_race_code', 'borough_code', 'is_completed']
    
    feature_vector = pd.DataFrame({
        'day_of_week': [day_of_week],
        'month': [month],
        'hour_of_day': [hour_of_day],
        'sus_race_code': [sus_race_code],
        'vic_race_code': [vic_race_code],
        'borough_code': [borough_code],
        'is_completed': [is_completed]
    })
    
    # Make prediction
    probability = model.predict_proba(feature_vector)[0, 1]
    classification = "Yes" if probability >= 0.5 else "No"
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importance = list(zip(features, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    top_features = feature_importance[:5]
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted probability a crime occurred: {probability:.2f}")
    print(f"Classification (threshold 0.5): {classification}")
    
    print("\nTop-5 feature importances driving this prediction:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
    
    print("\nInput summary:")
    print(f"  Date: {date_obj}")
    print(f"  Time: {time_obj}")
    print(f"  Borough: {category_mappings['borough_code'][borough_code]}")
    print(f"  Suspect demographic: {category_mappings['sus_race_code'][sus_race_code]}")
    print(f"  Victim demographic: {category_mappings['vic_race_code'][vic_race_code]}")
    
    print("\nNOTE: This prediction is for research purposes only and should not be used in operational settings.")


def main():
    """Main function to orchestrate the entire process"""
    # Fetch crime data
    crime_df = fetch_crime_data()
    
    # Clean and preprocess data
    cleaned_df, category_mappings = clean_and_preprocess(crime_df)
    
    # Generate negative samples
    negative_df = generate_negative_samples(cleaned_df)
    
    # Prepare features and targets
    data_splits = prepare_features_and_targets(cleaned_df, negative_df)
    
    # Create models
    models = create_and_train_models()
    
    # Train, evaluate, and save models
    evaluate_and_save_models(models, data_splits)
    
    # Run interactive inference
    interactive_inference(category_mappings)


if __name__ == "__main__":
    main()