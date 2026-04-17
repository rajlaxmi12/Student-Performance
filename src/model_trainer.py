import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def train_and_analyze(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Extract Feature Importance
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette='magma')
    plt.title('Key Drivers of Student Success')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png') # Save for your dashboard
    
    return model