import joblib
import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\MLOps\processed_student_data.csv")

X = df.drop(columns=["G3"])

# Apply same encoding
X = pd.get_dummies(X)

# Load model
model = joblib.load("deployment/deployment_model.pkl")

prediction = model.predict(X.iloc[[0]])

print("Prediction :", round(prediction[0],2))

print("Actual :", df.iloc[0]["G3"])