import json
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join("dataset", "car_actions.csv")
MODEL_DIR = "model"
STATIC_DIR = "static"
MODEL_PATH = os.path.join(MODEL_DIR, "car_action_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
CM_PATH = os.path.join(STATIC_DIR, "confusion_matrix.png")

TARGET = "action"
NUMERIC = ["speed", "acceleration", "steering_angle", "distance_ahead"]
CATEGORICAL = [
    "lane_position", "traffic_density", "traffic_light",
    "obstacle_detected", "road_condition", "weather", "road_curvature"
]

def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL)
    ])

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset not found. Run: python generate_dataset.py")

df = pd.read_csv(DATA_PATH)
X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=18, min_samples_leaf=2,
        random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "SVM": SVC(kernel="rbf", C=2.0, probability=True, random_state=42)
}

results = {}
fitted = {}
best_name = None
best_f1 = -1

print("=" * 70)
print("CAR ACTION PREDICTION - MODEL TRAINING")
print("=" * 70)

for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, average="weighted", zero_division=0))
    }
    results[name] = metrics
    fitted[name] = pipe

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    if metrics["f1_score"] > best_f1:
        best_f1 = metrics["f1_score"]
        best_name = name

best_model = fitted[best_name]
best_pred = best_model.predict(X_test)
labels = list(best_model.named_steps["classifier"].classes_)

cm = confusion_matrix(y_test, best_pred, labels=labels)
plt.figure(figsize=(11, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(CM_PATH, dpi=160)
plt.close()

bundle = {
    "pipeline": best_model,
    "algorithm": best_name,
    "feature_names": NUMERIC + CATEGORICAL,
    "classes": labels
}
joblib.dump(bundle, MODEL_PATH)

metrics_payload = {
    "algorithm": best_name,
    "task": "Multi-Class Classification",
    "number_of_classes": len(labels),
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "features": NUMERIC + CATEGORICAL,
    "results": results,
    "best_model_metrics": results[best_name],
    "classes": labels
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2)

print("\n" + "=" * 70)
print(f"Selected model: {best_name}")
print(f"Saved model   : {MODEL_PATH}")
print(f"Saved metrics : {METRICS_PATH}")
print(f"Saved matrix  : {CM_PATH}")
print("=" * 70)
print("\nClassification report for selected model:")
print(classification_report(y_test, best_pred, zero_division=0))
