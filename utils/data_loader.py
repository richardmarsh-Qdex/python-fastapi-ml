import pandas as pd
import numpy as np

def load_sample_data():
    """Load sample training data"""
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'label': np.random.randint(0, 2, 100)
    }
    return pd.DataFrame(data)

def prepare_training_data(df):
    """Prepare data for training"""
    X = df[['feature1', 'feature2', 'feature3']].values
    y = df['label'].values
    return X, y
