import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_data():
    # Go to project root from src/preprocessing.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "student-mat.csv")

    # Load dataset
    df = pd.read_csv(file_path, sep=';')

    # Create target column: pass = 1, fail = 0
    df["pass"] = (df["G3"] >= 10).astype(int)

    # Separate target and features
    target = df["pass"]
    features = df.drop(["G1", "G2", "G3", "pass"], axis=1)

    # Convert categorical columns into numeric
    features = pd.get_dummies(features, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test