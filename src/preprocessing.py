import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    # Load with semicolon separator
    df = pd.read_csv(filepath, sep=';')
    
    # Domain Logic: Binary Classification (Pass/Fail)
    # G3 < 10 is a fail in the Portuguese system
    df['pass'] = (df['G3'] >= 10).astype(int)
    
    # Drop original grades to prevent "data leakage" 
    # (The model shouldn't see G1/G2 to predict G3)
    target = df['pass']
    features = df.drop(['G1', 'G2', 'G3', 'pass'], axis=1)
    
    # One-Hot Encoding for categorical variables (school, sex, address, etc.)
    features = pd.get_dummies(features, drop_first=True)
    
    return train_test_split(features, target, test_size=0.2, random_state=42)