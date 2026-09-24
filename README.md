# Car Action Prediction for Autonomous Driving Using Machine Learning

## Project description

A college-level Flask web application that predicts the next simulated driving action from vehicle and environmental parameters. A Scikit-learn preprocessing-and-classification pipeline performs the actual prediction. The synthetic dataset is generated with domain-inspired rules only for training-data creation.

**Academic disclaimer:** This project is an academic machine-learning simulation. It is not intended for controlling real autonomous vehicles.

## Objectives

- Generate a realistic synthetic driving-state dataset.
- Compare four classification algorithms.
- Select the best test-set model using weighted F1 score.
- Persist preprocessing and the trained classifier together with Joblib.
- Provide a working browser interface connected to Flask.
- Display model metrics, a confusion matrix, and a visual simulation.

## Features

- 8-class driving action prediction:
  - ACCELERATE
  - BRAKE
  - MAINTAIN_SPEED
  - TURN_LEFT
  - TURN_RIGHT
  - CHANGE_LANE_LEFT
  - CHANGE_LANE_RIGHT
  - STOP
- 11 input features.
- Numeric validation and categorical selection.
- Random Forest / other model comparison.
- Actual test metrics loaded dynamically from `model/metrics.json`.
- Actual confusion matrix generated during training.
- Three ready-to-load demonstration scenarios.
- Responsive dashboard.
- `/health`, `/metrics`, and `/predict` Flask endpoints.

## Technologies

- HTML5, CSS3, JavaScript
- Python, Flask
- Pandas, NumPy
- Scikit-learn
- Joblib
- Matplotlib, Seaborn
- Chart.js (browser CDN)

## Dataset

`generate_dataset.py` creates 6,000 synthetic records at `dataset/car_actions.csv`.

Features:

1. speed
2. acceleration
3. steering_angle
4. distance_ahead
5. lane_position
6. traffic_density
7. traffic_light
8. obstacle_detected
9. road_condition
10. weather
11. road_curvature

The target is `action`.

The generator uses rule-based relationships such as red lights and short distances producing braking/stopping behavior, low speed on a clear road producing acceleration, curves producing turns, and dense traffic with sufficient distance producing lane changes. A small amount of controlled label noise is included for a less perfectly separable academic classification problem.

## Machine learning algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

All models use the same train/test split and a Scikit-learn `ColumnTransformer`:

- numeric columns: median imputation + standard scaling
- categorical columns: most-frequent imputation + one-hot encoding

The selected pipeline is saved as `model/car_action_model.pkl`.

## Project structure

```text
car-action-prediction/
├── app.py
├── train_model.py
├── generate_dataset.py
├── requirements.txt
├── README.md
├── dataset/
│   └── car_actions.csv
├── model/
│   └── car_action_model.pkl
└── templates/
    └── index.html
    └── static/
        ├── style.css
        ├── script.js
        └── confusion_matrix.png
```

## Installation

Open the project folder in VS Code terminal.

### 1. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\activate.bat
```

### 2. Install packages

```powershell
pip install -r requirements.txt
```

## Generate the dataset

```powershell
python generate_dataset.py
```

Expected output includes the number of generated records and action distribution.

## Train and evaluate models

```powershell
python train_model.py
```

This creates:

- `model/car_action_model.pkl`
- `model/metrics.json`
- `static/confusion_matrix.png`

The terminal prints accuracy, precision, recall, F1 score, and a classification report for every model.

## Run the Flask application

```powershell
python app.py
```

Open:

http://127.0.0.1:5000

## How to use the website

1. Enter/select vehicle and environmental parameters.
2. Optionally click Red Light, Clear Highway, or Left Curve.
3. Click **PREDICT ACTION**.
4. The browser sends JSON to `POST /predict`.
5. Flask validates the data and sends it through the saved ML pipeline.
6. The model returns an action and probability-based confidence.
7. The dashboard updates without a page reload.
8. The car/road visualization changes to match the predicted action.

## Example API request

```json
{
  "speed": 50,
  "acceleration": 0,
  "steering_angle": 0,
  "distance_ahead": 8,
  "lane_position": "CENTER",
  "traffic_density": "HIGH",
  "traffic_light": "RED",
  "obstacle_detected": "NO",
  "road_condition": "DRY",
  "weather": "CLEAR",
  "road_curvature": "STRAIGHT"
}
```

The response has this form:

```json
{
  "action": "STOP",
  "confidence": 91.23,
  "message": "The vehicle should come to a controlled stop before proceeding."
}
```

The exact action and confidence are produced by the trained model and can vary after retraining.

## API endpoints

- `GET /` — dashboard
- `POST /predict` — model prediction
- `GET /metrics` — training/evaluation metrics
- `GET /health` — application health check

## Future scope

- Larger real-world or high-fidelity simulated datasets.
- Temporal models using sequences of sensor states.
- More detailed vehicle dynamics and map context.
- Model calibration and uncertainty estimation.
- Explainability with SHAP or permutation importance.
- More advanced optimization for route and maneuver planning.
- Hardware/simulator integration in a controlled research environment.

## Limitations

- The dataset is synthetic and not a substitute for real driving data.
- A single tabular state does not model the full temporal nature of driving.
- The visual simulation is illustrative only.
- High confidence is not a safety guarantee.
- This system must not be connected to a real vehicle controller.

## Complete run order

```powershell
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
python app.py
```

Then visit `http://127.0.0.1:5000`.
