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

print(df.columns)
print(df.head())