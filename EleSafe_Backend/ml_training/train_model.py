import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAINING_DATA_PATH, RF_MODEL_PATH, SCALER_PATH,
    METRICS_PATH, MODELS_DIR, RANDOM_STATE,
    TEST_SIZE, VALIDATION_SIZE
)


def load_training_data():
    """Load training data"""
    print("Loading training data...")

    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"✗ Training data not found at: {TRAINING_DATA_PATH}")
        print("Run: python ml_training/preprocessing/create_training_data.py")
        return None

    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"✓ Loaded {len(df)} samples")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Positive: {df['conflict'].sum()}")
    print(f"  Negative: {len(df) - df['conflict'].sum()}")

    return df


def prepare_data(df):
    """Split data into train/validation/test sets"""
    print("\nPreparing data...")

    # Separate features and labels
    X = df.drop('conflict', axis=1)
    y = df['conflict']

    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Second split: train and validation
    val_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"✓ Train set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Scale features using StandardScaler"""
    print("\nScaling features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("✓ Features scaled")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest model...")

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    # Train
    model.fit(X_train, y_train)

    print("✓ Model trained")

    return model


def evaluate_model(model, X, y, dataset_name=""):
    """Evaluate model performance"""
    print(f"\nEvaluating on {dataset_name}...")

    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Print results
    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Specificity: {specificity:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn}  FP: {fp}")
    print(f"  FN: {fn}  TP: {tp}")

    # Return metrics
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'specificity': float(specificity),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }


def save_model(model, scaler, metrics):
    """Save model, scaler, and metrics"""
    print("\nSaving model...")

    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model
    joblib.dump(model, RF_MODEL_PATH)
    print(f"✓ Model saved to: {RF_MODEL_PATH}")

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Scaler saved to: {SCALER_PATH}")

    # Save metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {METRICS_PATH}")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Random Forest Model Training")
    print("=" * 60 + "\n")

    # Load data
    df = load_training_data()
    if df is None:
        return

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )

    # Train model
    model = train_random_forest(X_train_scaled, y_train)

    # Evaluate on all sets
    train_metrics = evaluate_model(model, X_train_scaled, y_train, "Train Set")
    val_metrics = evaluate_model(model, X_val_scaled, y_val, "Validation Set")
    test_metrics = evaluate_model(model, X_test_scaled, y_test, "Test Set")

    # Combine metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'model_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': RANDOM_STATE
        }
    }

    # Save everything
    save_model(model, scaler, all_metrics)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")


if __name__ == '__main__':
    main()