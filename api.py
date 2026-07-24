from flask import Flask, request, jsonify
import joblib
import pandas as pd

# ---------------------------
# Create Flask App
# ---------------------------

app = Flask(__name__)

# ---------------------------
# Load Model
# ---------------------------

model = joblib.load("deployment/deployment_model.pkl")

# ---------------------------
# Health Check API
# ---------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status":"API is Running Successfully"
    })

# ---------------------------
# Prediction API
# ---------------------------

@app.route("/predict", methods=["GET"])
def predict():

    df = pd.read_csv("processed_student_data.csv")

    X = df.drop(columns=["G3"])

    prediction = model.predict(X.iloc[[0]])

    return jsonify({

        "Predicted_G3": round(float(prediction[0]),2),

        "Actual_G3": int(df.iloc[0]["G3"])

    })


# ---------------------------
# Run Flask
# ---------------------------

if __name__ == "__main__":

    app.run(debug=True)