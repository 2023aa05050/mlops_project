import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from joblib import dump
from src.preprocess import load_data, split_data
import numpy as np
import os
import json

# Function to train the model and log metrics
def train_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Metrics calculation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Handle multi-class ROC AUC score
    if len(np.unique(y_train)) > 2:  # Multi-class case
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="weighted")
    else:  # Binary case
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Log parameters and metrics to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    })

    # Log confusion matrix as an artifact
    cm_file = "confusion_matrix.json"
    with open(cm_file, "w") as f:
        json.dump(cm.tolist(), f)
    mlflow.log_artifact(cm_file)
    os.remove(cm_file)

    # Log the model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # Save the model locally
    dump(model, "models/model.pkl")
    print("Model saved locally as 'models/model.pkl'")

    # Print metrics and classification report
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    # Load and split data
    X, y = load_data()  # Custom preprocessing function
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Start an MLflow experiment run
    with mlflow.start_run():
        train_model(X_train, X_test, y_train, y_test)
