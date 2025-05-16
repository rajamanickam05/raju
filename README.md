# raju
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Load dataset (example: US DOT's accident data or similar)
df = pd.read_csv('accident_data.csv')

# Preview
print(df.head())

# Drop irrelevant columns and handle missing data
df = df.dropna()
df = df.drop(columns=['ID', 'Source', 'Description'])  # Adjust as needed

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Define features and label (e.g., accident severity: 0 = low, 1 = high)
X = df.drop('Severity', axis=1)  # Replace 'Severity' with actual column name
y = df['Severity']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model: XGBoost or RandomForest
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
feat_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature Importances")
sns.barplot(x=importances[indices][:10], y=feat_names[indices][:10])
plt.tight_layout()
plt.show()
