import json
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "car_action_model.pkl")
METRICS_PATH = os.path.join("model", "metrics.json")

bundle = None
model_error = None

if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception:
        model_error = "Machine learning model could not be loaded."
else:
    model_error = "Machine learning model not found. Please run train_model.py first."

NUMERIC_RANGES = {
    "speed": (0, 150),
    "acceleration": (-10, 10),
    "steering_angle": (-45, 45),
    "distance_ahead": (0, 200),
}
ALLOWED = {
    "lane_position": {"LEFT", "CENTER", "RIGHT"},
    "traffic_density": {"LOW", "MEDIUM", "HIGH"},
    "traffic_light": {"RED", "YELLOW", "GREEN", "NONE"},
    "obstacle_detected": {"YES", "NO"},
    "road_condition": {"DRY", "WET", "SLIPPERY"},
    "weather": {"CLEAR", "RAIN", "FOG"},
    "road_curvature": {"STRAIGHT", "LEFT", "RIGHT"},
}

MESSAGES = {
    "ACCELERATE": "The vehicle should increase speed gradually while maintaining a clear path.",
    "BRAKE": "The vehicle should reduce speed and prepare to stop if the condition persists.",
    "MAINTAIN_SPEED": "Conditions support maintaining the current speed with continued monitoring.",
    "TURN_LEFT": "The vehicle should steer toward the left curve or direction.",
    "TURN_RIGHT": "The vehicle should steer toward the right curve or direction.",
    "CHANGE_LANE_LEFT": "The vehicle should prepare for a controlled lane change to the left.",
    "CHANGE_LANE_RIGHT": "The vehicle should prepare for a controlled lane change to the right.",
    "STOP": "The vehicle should come to a controlled stop before proceeding.",
}

def validate_payload(data):
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [k for k in list(NUMERIC_RANGES) + list(ALLOWED) if k not in data]
    if missing:
        raise ValueError("Missing input fields: " + ", ".join(missing))

    clean = {}
    for key, (low, high) in NUMERIC_RANGES.items():
        try:
            value = float(data[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number.")
        if not np.isfinite(value) or not (low <= value <= high):
            raise ValueError(f"{key} must be between {low} and {high}.")
        clean[key] = value

    for key, options in ALLOWED.items():
        value = str(data[key]).upper()
        if value not in options:
            raise ValueError(f"Invalid value for {key}.")
        clean[key] = value

    return clean

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "running"})

@app.route("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Training metrics not found. Please run train_model.py first."}), 404
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({"error": "Training metrics could not be read."}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if bundle is None:
        return jsonify({"error": model_error or "Machine learning model is unavailable."}), 503

    try:
        data = validate_payload(request.get_json(silent=True))
        frame = pd.DataFrame([data], columns=bundle["feature_names"])
        pipeline = bundle["pipeline"]
        action = str(pipeline.predict(frame)[0])

        confidence = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(frame)[0]
            confidence = float(np.max(probabilities) * 100)

        response = {
            "action": action,
            "confidence": round(confidence, 2) if confidence is not None else None,
            "message": MESSAGES.get(action, "The model predicted a driving action.")
        }
        return jsonify(response)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Prediction failed. Please verify the inputs and trained model."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
