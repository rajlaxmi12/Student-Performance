import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_and_analyze(X_train, X_test, y_train, y_test):
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    importances = model.feature_importances_
    feature_names = X_train.columns

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    # Save feature importance plot
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_df["Feature"], feature_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Important Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = os.path.join(reports_dir, "feature_importance.png")
    plt.savefig(plot_path)
    plt.close()

    return model, accuracy, report, cm, feature_df, plot_path