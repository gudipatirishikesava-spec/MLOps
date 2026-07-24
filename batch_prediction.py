import pandas as pd
import joblib

# Load Dataset
df = pd.read_csv("processed_student_data.csv")

print("Dataset Loaded Successfully")

# Separate Features
X = df.drop(columns=["G3"])

# Convert Categorical Columns
X = pd.get_dummies(X)

# Load Model
model = joblib.load("deployment/deployment_model.pkl")

print("Deployment Model Loaded Successfully")

# Match Features
expected_columns = model.feature_names_in_

for column in expected_columns:
    if column not in X.columns:
        X[column] = 0

X = X[expected_columns]

# Predict
predictions = model.predict(X)

# Store Predictions
df["Predicted_G3"] = predictions.round(2)

# Save Output
df.to_csv("batch_predictions.csv", index=False)

print("\nBatch Prediction Completed Successfully")

print(df[["G3", "Predicted_G3"]].head())

print("\nTotal Students :", len(df))

print("\nPrediction File Saved")
print("File Name : batch_predictions.csv")