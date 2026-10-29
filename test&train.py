import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load your dataset
df = pd.read_csv("final_filtered_with_cravings.csv")

# Ensure monthly columns are 15-bit strings
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for month in months:
    df[month] = df[month].apply(lambda x: str(int(x)).zfill(15))
    for i in range(15):
        df[f"{month}_dish{i+1}"] = df[month].str[i].astype(int)

# Process Dec_craving column to get target label
df["Dec_craving"] = df["Dec_craving"].apply(lambda x: str(int(x)).zfill(15))
df["label_dish1"] = df["Dec_craving"].str[0].astype(int)  # Predicting Dish 1

# Features = Jan to Nov dish history
feature_cols = [f"{month}_dish{i+1}" for month in months[:-1] for i in range(15)]
X = df[feature_cols]
y = df["label_dish1"]

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (Path 1: Logistic Regression)
model_lr = LogisticRegression(max_iter=200, class_weight="balanced")
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)

# Evaluate
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)


import joblib
joblib.dump(model_lr, "craving_model_dish1.pkl")