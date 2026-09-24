import os
import numpy as np
import pandas as pd

SEED = 42
N = 6000
rng = np.random.default_rng(SEED)

LANES = ["LEFT", "CENTER", "RIGHT"]
TRAFFIC = ["LOW", "MEDIUM", "HIGH"]
LIGHTS = ["RED", "YELLOW", "GREEN", "NONE"]
OBSTACLE = ["YES", "NO"]
ROAD = ["DRY", "WET", "SLIPPERY"]
WEATHER = ["CLEAR", "RAIN", "FOG"]
CURVE = ["STRAIGHT", "LEFT", "RIGHT"]

def choose_action(r):
    # Deterministic priority rules with small controlled randomness to avoid
    # a perfectly separable synthetic dataset.
    if r["traffic_light"] == "RED":
        if r["distance_ahead"] <= 12 or r["speed"] >= 55:
            return "STOP"
        return "BRAKE"
    if r["traffic_light"] == "YELLOW":
        if r["distance_ahead"] <= 18 or r["speed"] >= 55:
            return "BRAKE"
    if r["obstacle_detected"] == "YES":
        if r["distance_ahead"] <= 10:
            return "STOP"
        if r["distance_ahead"] <= 30:
            return "BRAKE"
    if r["road_condition"] == "SLIPPERY" and r["speed"] > 50:
        return "BRAKE"
    if r["weather"] == "FOG" and r["speed"] > 65 and r["distance_ahead"] < 45:
        return "BRAKE"
    if r["road_curvature"] == "LEFT" and abs(r["steering_angle"]) >= 8:
        return "TURN_LEFT"
    if r["road_curvature"] == "RIGHT" and abs(r["steering_angle"]) >= 8:
        return "TURN_RIGHT"

    # Lane-change situations: dense traffic and enough space ahead.
    if r["traffic_density"] == "HIGH" and r["distance_ahead"] >= 35:
        if r["lane_position"] == "CENTER":
            return "CHANGE_LANE_LEFT" if r["steering_angle"] < 0 else "CHANGE_LANE_RIGHT"
        if r["lane_position"] == "LEFT" and r["steering_angle"] > 10:
            return "CHANGE_LANE_RIGHT"
        if r["lane_position"] == "RIGHT" and r["steering_angle"] < -10:
            return "CHANGE_LANE_LEFT"

    if r["speed"] < 18 and r["distance_ahead"] > 30 and r["traffic_light"] in ["GREEN", "NONE"]:
        return "ACCELERATE"
    if r["speed"] > 95 and r["distance_ahead"] < 35:
        return "BRAKE"
    if r["acceleration"] < -3 and r["speed"] < 25:
        return "ACCELERATE"

    return "MAINTAIN_SPEED"

def make_row():
    speed = rng.uniform(0, 150)
    acceleration = np.clip(rng.normal(0, 2.5), -10, 10)
    steering = np.clip(rng.normal(0, 13), -45, 45)
    distance = rng.uniform(2, 200)

    row = {
        "speed": round(speed, 2),
        "acceleration": round(acceleration, 2),
        "steering_angle": round(steering, 2),
        "distance_ahead": round(distance, 2),
        "lane_position": rng.choice(LANES, p=[0.2, 0.6, 0.2]),
        "traffic_density": rng.choice(TRAFFIC, p=[0.35, 0.4, 0.25]),
        "traffic_light": rng.choice(LIGHTS, p=[0.15, 0.12, 0.55, 0.18]),
        "obstacle_detected": rng.choice(OBSTACLE, p=[0.18, 0.82]),
        "road_condition": rng.choice(ROAD, p=[0.7, 0.2, 0.1]),
        "weather": rng.choice(WEATHER, p=[0.72, 0.2, 0.08]),
        "road_curvature": rng.choice(CURVE, p=[0.62, 0.19, 0.19]),
    }
    action = choose_action(row)

    # Add limited label noise for a more realistic classification exercise.
    if rng.random() < 0.025:
        alternatives = [a for a in [
            "ACCELERATE", "BRAKE", "MAINTAIN_SPEED", "TURN_LEFT",
            "TURN_RIGHT", "CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT", "STOP"
        ] if a != action]
        action = rng.choice(alternatives)
    row["action"] = action
    return row

df = pd.DataFrame([make_row() for _ in range(N)])
out = os.path.join("dataset", "car_actions.csv")
os.makedirs("dataset", exist_ok=True)
df.to_csv(out, index=False)

print(f"Generated {len(df)} records: {out}")
print("\nClass distribution:")
print(df["action"].value_counts())
