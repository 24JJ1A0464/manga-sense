"""
Manganese-Sense: Machine Learning Module
Phase 3: Random Forest classification for manganese detection
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_random_forest_model(n_estimators: int = 100, 
                                max_depth: int = 12, 
                                class_weight: str = 'balanced',
                                random_state: int = 42) -> RandomForestClassifier:
    """
    Initialize a Random Forest classifier optimized for mineral detection.
    
    Parameters:
    - n_estimators=100: Number of trees in the forest
    - max_depth=12: Maximum depth of each tree (prevents overfitting)
    - class_weight='balanced': Automatically adjust weights for imbalanced classes
    - random_state=42: For reproducible results
    
    Returns:
        Configured RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    logger.info(f"Created RandomForest: n_estimators={n_estimators}, max_depth={max_depth}")
    return model


def generate_mock_training_data(n_samples: int = 5000,
                                 n_features: int = 7,
                                 random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mock geological training data for manganese detection.
    
    In production, this would be replaced with actual labeled geological survey data.
    This function simulates the spectral signatures of manganese-bearing minerals
    (e.g., pyrolusite, rhodochrosite, psilomelane) vs background geology.
    
    Args:
        n_samples: Number of training samples to generate
        n_features: Number of features (should be 7 for our pipeline)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train)
    """
    if n_features != 7:
        raise ValueError("Manganese detection requires exactly 7 features")

    rng = np.random.default_rng(random_state)

    # Feature order: B04, B08, B11, B12, NDVI, SWIR_Ratio, Ferrous_Ratio.
    X = np.column_stack([
        rng.uniform(0.05, 0.45, n_samples),
        rng.uniform(0.10, 0.80, n_samples),
        rng.uniform(0.10, 0.75, n_samples),
        rng.uniform(0.10, 0.65, n_samples),
        rng.uniform(-0.80, 0.80, n_samples),
        rng.uniform(0.70, 1.60, n_samples),
        rng.uniform(0.20, 2.00, n_samples),
    ])

    # Keep the synthetic observations imperfect like real sensor measurements.
    X += rng.normal(0.0, 0.025, X.shape)

    manganese_signature = (X[:, 5] > 1.15) & (X[:, 4] < 0.2)
    y = manganese_signature.astype(np.uint8)
    
    n_positive = np.sum(y)
    logger.info(f"Generated mock training data: {n_samples} samples, {n_positive} positive ({n_positive/n_samples*100:.1f}%)")
    
    return X, y


def train_model(model: RandomForestClassifier, 
                X_train: np.ndarray, 
                y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train the Random Forest model on geological training data.
    
    Args:
        model: Unfitted RandomForestClassifier
        X_train: Training feature matrix (N_samples, N_features)
        y_train: Training labels (N_samples,)
        
    Returns:
        Fitted RandomForestClassifier
    """
    logger.info(f"Training model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    return model


def train_synthetic_model(random_state: int = 42) -> RandomForestClassifier:
    """Generate synthetic geological data and return a fitted classifier."""
    model = create_random_forest_model(
        n_estimators=100,
        class_weight='balanced',
        random_state=random_state,
    )
    X_train, y_train = generate_mock_training_data(
        n_samples=5000,
        random_state=random_state,
    )
    return train_model(model, X_train, y_train)


def evaluate_model(model: RandomForestClassifier, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray) -> dict:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Fitted RandomForestClassifier
        X_test: Test feature matrix
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Model Evaluation:\n{classification_report(y_test, y_pred)}")
    
    return {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'feature_importances': model.feature_importances_.tolist()
    }


def predict_manganese(model: RandomForestClassifier, 
                      feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the feature matrix to detect manganese deposits.
    
    Args:
        model: Fitted RandomForestClassifier
        feature_matrix: Feature matrix of shape (N_pixels, 7)
        
    Returns:
        Tuple of (predictions, probabilities)
        - predictions: Binary array (1=manganese detected, 0=background)
        - probabilities: Probability of positive class (manganese)
    """
    logger.info(f"Running inference on {feature_matrix.shape[0]} pixels...")
    
    predictions = model.predict(feature_matrix)
    probabilities = model.predict_proba(feature_matrix)[:, 1]
    
    n_detections = np.sum(predictions)
    logger.info(f"Detections: {n_detections} positive pixels out of {len(predictions)}")
    
    return predictions, probabilities


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> RandomForestClassifier:
    """Load trained model from disk."""
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def get_feature_importance(model: RandomForestClassifier) -> dict:
    """
    Get feature importance scores for interpretability.
    
    Feature names correspond to:
    0: B04 (Red)
    1: B08 (NIR)
    2: B11 (SWIR1)
    3: B12 (SWIR2)
    4: NDVI
    5: SWIR_Ratio (B11/B12)
    6: Ferrous_Ratio (B11/B08)
    """
    feature_names = ['B04_Red', 'B08_NIR', 'B11_SWIR1', 'B12_SWIR2', 
                     'NDVI', 'SWIR_Ratio', 'Ferrous_Ratio']
    
    importances = model.feature_importances_
    return dict(zip(feature_names, importances))