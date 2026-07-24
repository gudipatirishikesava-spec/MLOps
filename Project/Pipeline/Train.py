import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load processed dataset
df = pd.read_csv(r"D:\MLOps\processed_student_data.csv")

X = df.drop(columns=["G3"])
y = df["G3"]

# Encode categorical variables
X = pd.get_dummies(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "deployment/deployment_model.pkl")

print("Model Training Completed")