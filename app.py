from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import logging
from flask_jwt_extended import JWTManager, jwt_required, create_access_token


# Initialize Flask app
app = Flask(__name__)


# Secret key for encoding JWTs
app.config["JWT_SECRET_KEY"] = "your_secret_key"  # Replace with a strong secret key
jwt = JWTManager(app)


# Configure logging
logging.basicConfig(level=logging.INFO)


# Load trained model and scaler
MODEL_PATH = "models/random_forest_model.pkl"
SCALER_PATH = "models/scaler.pkl"


try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_features = scaler.feature_names_in_
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    expected_features = []


# Dummy User data - This can be replaced with a database
USERS = {"admin": "password123"}  # Example credentials




def encode_input_data(user_input):
    """Format user input to match training data."""
    df = pd.DataFrame([user_input])
   
    # Ensure required columns are present
    for col in expected_features:
        if col not in df.columns:
            df[col] = np.nan  # Fill missing columns with NaN
   
    df = df[expected_features]  # Ensure column order matches training data
    df_scaled = scaler.transform(df)  # Scale input data
    return df_scaled




def calculate_dynamic_threshold(user_input):
    """Dynamically calculate the threshold based on input data."""
    numeric_values = [float(v) for v in user_input.values() if isinstance(v, (int, float)) and not np.isnan(v)]
    return round(sum(numeric_values) / len(numeric_values), 2) if numeric_values else 0.5  # Default threshold for balance




@app.route('/login', methods=['POST'])
def login():
    username = request.json.get("username")
    password = request.json.get("password")


    if USERS.get(username) == password:
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    else:
        return jsonify({"msg": "Invalid username or password"}), 401




@app.route('/')
def home():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    try:
        user_data = request.json
        if not user_data:
            return jsonify({"error": "No input data provided"}), 400


        # Ensure required features are present
        missing_keys = [key for key in expected_features if key not in user_data]
        if missing_keys:
            return jsonify({"error": f"Missing input fields: {', '.join(missing_keys)}"}), 400


        # Fixed threshold for Normal Activity
        threshold = 0.7  # Set a fixed threshold for Normal Activity classification


        # Prepare input data for the model
        input_data = encode_input_data(user_data)
       
        # Predict the probabilities
        probabilities = model.predict_proba(input_data)[0]
        probability_normal = probabilities[0]
        probability_anomalous = probabilities[1]
       
        # Apply the threshold to classify activity
        if probability_normal >= threshold:
            result = "Normal Activity"
        else:
            result = "Anomalous Activity Detected"


        # Return the response with the classification and probability details
        return jsonify({
            "prediction": result,
            "probability_normal": round(probability_normal, 4),
            "probability_anomalous": round(probability_anomalous, 4),
            "threshold": threshold,
            "confidence_message": f"The model is {round(probability_normal * 100, 2)}% confident that the activity is normal."
        })
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500
   
if __name__ == '__main__':
    app.run(debug=True)
